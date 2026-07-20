"""OHLCV history retrieval.

Wraps yfinance behind a narrow interface (`fetch_history`, `fetch_quote`) so a
paid provider can be substituted by reimplementing this module alone.

Missing-data policy, stated explicitly because it materially affects model
validity:

* Rows with a null Close are **dropped**, never forward-filled. Forward-filling
  a close manufactures a zero-return day that the model would learn as a real
  low-volatility observation.
* Null Volume is filled with 0 (a real halted-trading reading) but flagged in
  `DataQualityReport` so downstream volume features can be suppressed.
* Duplicate index entries keep the last occurrence -- providers occasionally
  emit a provisional bar followed by the settled one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from app.core.config import get_settings
from app.core.errors import (
    DataUnavailableError,
    ProviderError,
    RateLimitedError,
    UnknownTickerError,
)
from app.core.logging import get_logger
from app.data.cache.store import cache_key, get_cache

logger = get_logger(__name__)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5


@dataclass
class DataQualityReport:
    """What had to be repaired to make the series usable.

    Surfaced through the API so the dashboard can warn when a prediction rests
    on a patchy series, rather than presenting it with false confidence.
    """

    rows_received: int = 0
    rows_usable: int = 0
    dropped_null_close: int = 0
    filled_null_volume: int = 0
    duplicate_dates_removed: int = 0
    gap_days_detected: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.warnings

    def to_dict(self) -> dict[str, object]:
        return {
            "rows_received": self.rows_received,
            "rows_usable": self.rows_usable,
            "dropped_null_close": self.dropped_null_close,
            "filled_null_volume": self.filled_null_volume,
            "duplicate_dates_removed": self.duplicate_dates_removed,
            "gap_days_detected": self.gap_days_detected,
            "is_clean": self.is_clean,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class PriceHistory:
    """Cleaned OHLCV series plus its provenance."""

    symbol: str
    frame: pd.DataFrame
    fetched_at: float
    served_from_cache: bool
    quality: DataQualityReport

    @property
    def last_session_date(self) -> date | None:
        if self.frame.empty:
            return None
        return self.frame.index[-1].date()


def _normalise_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """Clean a provider frame and record every repair applied."""
    report = DataQualityReport(rows_received=len(raw))

    if raw.empty:
        return raw, report

    frame = raw.copy()

    # yfinance returns a MultiIndex column layout when given a list of tickers;
    # flatten defensively so single-ticker and list call sites behave alike.
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.rename(columns={"Adj Close": "AdjClose"})

    if "AdjClose" not in frame.columns and "Close" in frame.columns:
        # auto_adjust=True already folded splits/dividends into Close.
        frame["AdjClose"] = frame["Close"]

    missing = [column for column in ("Open", "High", "Low", "Close") if column not in frame.columns]
    if missing:
        raise ProviderError(f"Provider response missing required columns: {missing}")

    if "Volume" not in frame.columns:
        frame["Volume"] = 0
        report.warnings.append("Provider returned no Volume column; volume features unavailable.")

    frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is not None:
        # Normalise to naive dates: bars are daily, and a tz-aware index makes
        # joins against the benchmark series fail when feeds disagree on tz.
        frame.index = frame.index.tz_localize(None)
    frame.index.name = "Date"

    duplicates = int(frame.index.duplicated(keep="last").sum())
    if duplicates:
        frame = frame[~frame.index.duplicated(keep="last")]
        report.duplicate_dates_removed = duplicates

    frame = frame.sort_index()

    null_close = int(frame["Close"].isna().sum())
    if null_close:
        frame = frame[frame["Close"].notna()]
        report.dropped_null_close = null_close
        report.warnings.append(
            f"Dropped {null_close} row(s) with no closing price (not forward-filled)."
        )

    null_volume = int(frame["Volume"].isna().sum())
    if null_volume:
        frame["Volume"] = frame["Volume"].fillna(0)
        report.filled_null_volume = null_volume
        if null_volume > len(frame) * 0.05:
            report.warnings.append(
                f"{null_volume} row(s) had no volume; volume-based signals are unreliable."
            )

    # Flag unusually long gaps (>5 calendar days beyond a normal weekend),
    # which usually mean a suspension or a provider outage.
    if len(frame) > 1:
        gaps = frame.index.to_series().diff().dt.days.dropna()
        long_gaps = int((gaps > 5).sum())
        if long_gaps:
            report.gap_days_detected = long_gaps
            report.warnings.append(
                f"{long_gaps} gap(s) longer than 5 days — possible suspension or missing data."
            )

    frame = frame[[column for column in OHLCV_COLUMNS if column in frame.columns]]
    report.rows_usable = len(frame)
    return frame, report


def fetch_history(
    symbol: str,
    *,
    period: str = "10y",
    interval: str = "1d",
    use_cache: bool = True,
) -> PriceHistory:
    """Fetch cleaned daily OHLCV history for one symbol.

    Raises:
        UnknownTickerError: Provider returned nothing for this symbol.
        RateLimitedError / ProviderError: Upstream failure after retries.
    """
    settings = get_settings()
    cache = get_cache()
    key = cache_key(symbol.upper(), period, interval)

    if use_cache:
        hit = cache.get_frame("ohlcv", key)
        if hit is not None:
            frame, fetched_at = hit
            logger.debug("OHLCV cache hit for %s (%s)", symbol, period)
            quality = DataQualityReport(rows_received=len(frame), rows_usable=len(frame))
            return PriceHistory(symbol, frame, fetched_at, True, quality)

    raw = _download_with_retry(symbol, period=period, interval=interval)

    if raw is None or raw.empty:
        raise UnknownTickerError(
            f"No market data returned for '{symbol}'.",
            detail=(
                "The symbol may be delisted, mistyped, or missing the market suffix "
                "(Indian NSE tickers require a '.NS' suffix, e.g. 'RELIANCE.NS')."
            ),
        )

    frame, quality = _normalise_frame(raw)

    if frame.empty:
        raise DataUnavailableError(
            f"'{symbol}' returned {quality.rows_received} row(s), none usable.",
            detail="Every row lacked a closing price.",
        )

    fetched_at = time.time()
    if use_cache:
        cache.set_frame(
            "ohlcv", key, frame,
            ttl_seconds=settings.cache_ttl_daily_bars, fetched_at=fetched_at,
        )

    logger.info(
        "Fetched %s: %d usable bars (%s to %s)",
        symbol, len(frame), frame.index[0].date(), frame.index[-1].date(),
    )
    return PriceHistory(symbol, frame, fetched_at, False, quality)


def _download_with_retry(symbol: str, *, period: str, interval: str) -> pd.DataFrame | None:
    """Download with exponential backoff, translating provider faults."""
    import yfinance as yf

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            frame = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            return frame
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()

            if "rate limit" in message or "too many requests" in message or "429" in message:
                # Backing off further is pointless within one request; surface
                # it so the UI can tell the user to retry shortly.
                raise RateLimitedError(
                    "Market data provider rate-limited this request.",
                    retry_after=60,
                ) from exc

            if attempt < MAX_RETRIES:
                delay = RETRY_BACKOFF_SECONDS ** attempt
                logger.warning(
                    "Fetch attempt %d/%d for %s failed (%s); retrying in %.1fs",
                    attempt, MAX_RETRIES, symbol, exc, delay,
                )
                time.sleep(delay)

    raise ProviderError(
        f"Market data provider failed for '{symbol}' after {MAX_RETRIES} attempts.",
        detail=str(last_error),
    )
