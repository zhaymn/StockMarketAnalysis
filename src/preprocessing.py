"""Data acquisition and feature engineering for historical stock data.

Pipeline: download raw OHLCV data from Yahoo Finance -> handle missing
values -> engineer technical-indicator features -> hand off a clean,
leakage-free DataFrame to ``src.dataset`` for scaling and windowing.

All indicators below are computed strictly from past-and-current rows via
pandas rolling/exponential windows, so no future information leaks into a
given row's feature values.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils import (
    DATA_DIR,
    DataDownloadError,
    InsufficientDataError,
    get_date_range,
    get_logger,
    validate_ticker,
)

logger = get_logger(__name__)

# Below this many rows, indicators (esp. SMA_50) and LSTM sequences are not
# meaningful, so we fail fast rather than silently training on noise.
MIN_ROWS_REQUIRED = 100

RAW_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse yfinance's MultiIndex columns (field, ticker) to plain field names."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def download_stock_data(
    ticker: str,
    years: int = 5,
    max_retries: int = 3,
    cache: bool = True,
) -> pd.DataFrame:
    """Download daily OHLCV history for a ticker from Yahoo Finance.

    Args:
        ticker: Any Yahoo Finance-recognized symbol (e.g. ``"AAPL"``,
            ``"RELIANCE.NS"``, ``"TCS.NS"``).
        years: Number of years of history to request (1, 3, 5, or 10 typical).
        max_retries: Number of download attempts before giving up, with
            exponential-ish backoff between attempts (network hiccups happen).
        cache: If True, persist the raw download to ``data/{ticker}_raw.csv``.

    Returns:
        DataFrame indexed by ``Date`` with columns
        ``Open, High, Low, Close, Adj Close, Volume``.

    Raises:
        DataDownloadError: If the download fails after all retries, or
            returns no rows.
        InsufficientDataError: If fewer than ``MIN_ROWS_REQUIRED`` rows come back.
    """
    ticker = validate_ticker(ticker)
    start_date, end_date = get_date_range(years)

    last_error: Exception | None = None
    df: pd.DataFrame | None = None

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
                actions=False,
                threads=False,
            )
            if df is None or df.empty:
                raise DataDownloadError(f"No data returned for ticker '{ticker}'.")
            break
        except Exception as exc:  # yfinance can raise several distinct error types
            last_error = exc
            logger.warning(
                "Attempt %d/%d failed to download '%s': %s", attempt, max_retries, ticker, exc
            )
            if attempt < max_retries:
                time.sleep(1.5 * attempt)

    if df is None or df.empty:
        raise DataDownloadError(
            f"Failed to download data for '{ticker}' after {max_retries} attempts."
        ) from last_error

    df = _flatten_columns(df)
    df.index.name = "Date"
    df = df.sort_index()

    missing_cols = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataDownloadError(
            f"Downloaded data for '{ticker}' is missing expected columns: {missing_cols}"
        )
    df = df[RAW_COLUMNS]

    if len(df) < MIN_ROWS_REQUIRED:
        raise InsufficientDataError(
            f"Only {len(df)} rows downloaded for '{ticker}'; at least "
            f"{MIN_ROWS_REQUIRED} are required for reliable modeling."
        )

    logger.info(
        "Downloaded %d rows for %s (%s to %s)", len(df), ticker, start_date, end_date
    )

    if cache:
        _cache_dataframe(df, ticker, suffix="raw")

    return df


def _cache_dataframe(df: pd.DataFrame, ticker: str, suffix: str) -> None:
    """Write a DataFrame to ``data/{ticker}_{suffix}.csv``, best-effort."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = DATA_DIR / f"{ticker}_{suffix}.csv"
        df.to_csv(path)
        logger.info("Cached %s data to %s", suffix, path)
    except OSError as exc:
        logger.warning("Could not write cache file for %s: %s", ticker, exc)


# --------------------------------------------------------------------------- #
# Missing value handling
# --------------------------------------------------------------------------- #


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps in OHLCV data (e.g. holidays, feed hiccups) without leaking future data.

    Strategy: forward-fill (carry the last known price forward) then
    back-fill any remaining leading gaps, and finally drop any row that is
    still incomplete.

    Args:
        df: Raw OHLCV DataFrame, possibly containing NaNs.

    Returns:
        A DataFrame with no missing values.
    """
    df = df.copy()
    missing_before = int(df.isna().sum().sum())

    if missing_before:
        df = df.ffill().bfill()

    remaining_na_rows = df.isna().any(axis=1).sum()
    if remaining_na_rows:
        df = df.dropna()

    if missing_before:
        logger.info(
            "Missing-value handling: %d NaN cells filled, %d incomplete rows dropped.",
            missing_before,
            remaining_na_rows,
        )
    return df


# --------------------------------------------------------------------------- #
# Technical indicator feature engineering
# --------------------------------------------------------------------------- #


def add_moving_averages(df: pd.DataFrame, windows: Sequence[int] = (10, 20, 50)) -> pd.DataFrame:
    """Add Simple and Exponential Moving Average columns.

    SMA_w[t] = mean(Close[t-w+1 .. t])
    EMA_w[t] = alpha * Close[t] + (1 - alpha) * EMA_w[t-1], alpha = 2 / (w + 1)
    """
    df = df.copy()
    for w in windows:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w, min_periods=w).mean()
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False, min_periods=w).mean()
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Add the Relative Strength Index (Wilder's smoothing method).

    RSI = 100 - 100 / (1 + RS), where RS = avg_gain / avg_loss over the
    lookback window, with gains/losses smoothed via Wilder's EMA
    (equivalent to ``ewm(alpha=1/window, adjust=False)``).
    """
    df = df.copy()
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # When avg_loss is 0 (pure uptrend over the window), RSI is defined as 100.
    df[f"RSI_{window}"] = rsi.fillna(100)
    return df


def add_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Add MACD, its signal line, and the MACD histogram.

    MACD = EMA_fast(Close) - EMA_slow(Close)
    Signal = EMA_signal(MACD)
    Histogram = MACD - Signal
    """
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False, min_periods=slow).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()

    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Hist"] = macd_line - signal_line
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Bands: a rolling mean +/- ``num_std`` rolling standard deviations.

    Upper = SMA_w + num_std * rolling_std_w
    Lower = SMA_w - num_std * rolling_std_w
    """
    df = df.copy()
    rolling_mean = df["Close"].rolling(window=window, min_periods=window).mean()
    rolling_std = df["Close"].rolling(window=window, min_periods=window).std()

    df["BB_Middle"] = rolling_mean
    df["BB_Upper"] = rolling_mean + num_std * rolling_std
    df["BB_Lower"] = rolling_mean - num_std * rolling_std
    return df


def add_volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add volume-based features: a volume moving average and On-Balance Volume.

    OBV accumulates volume signed by the day's price direction:
    OBV[t] = OBV[t-1] + Volume[t] if Close[t] > Close[t-1]
    OBV[t] = OBV[t-1] - Volume[t] if Close[t] < Close[t-1]
    OBV[t] = OBV[t-1]             otherwise
    """
    df = df.copy()
    df[f"Volume_MA_{window}"] = df["Volume"].rolling(window=window, min_periods=window).mean()

    price_direction = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"] = (price_direction * df["Volume"]).cumsum()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full technical-indicator feature engineering pipeline.

    Adds moving averages, RSI, MACD, Bollinger Bands, and volume features,
    then drops the warm-up rows that inevitably contain NaNs (e.g. the
    first 49 rows for a 50-day SMA).

    Args:
        df: Clean OHLCV DataFrame (post ``handle_missing_values``).

    Returns:
        Feature-engineered DataFrame with no NaNs.
    """
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volume_features(df)

    rows_before = len(df)
    df = df.dropna()
    rows_dropped = rows_before - len(df)
    logger.info(
        "Feature engineering added %d indicator columns; dropped %d warm-up rows.",
        df.shape[1] - len(RAW_COLUMNS),
        rows_dropped,
    )
    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def load_and_prepare_data(ticker: str, years: int = 5, cache: bool = True) -> pd.DataFrame:
    """End-to-end pipeline: download -> clean -> engineer features.

    Args:
        ticker: Ticker symbol, e.g. ``"AAPL"`` or ``"TCS.NS"``.
        years: Years of history to pull (1, 3, 5, or 10).
        cache: Whether to persist raw and processed CSVs under ``data/``.

    Returns:
        A feature-engineered, leakage-free DataFrame ready for
        ``src.dataset`` to scale and window.

    Raises:
        DataDownloadError: If data cannot be retrieved.
        InsufficientDataError: If too little data is available after
            cleaning and indicator warm-up to build a usable dataset.
    """
    ticker = validate_ticker(ticker)
    raw_df = download_stock_data(ticker, years=years, cache=cache)
    clean_df = handle_missing_values(raw_df)
    featured_df = engineer_features(clean_df)

    if len(featured_df) < MIN_ROWS_REQUIRED:
        raise InsufficientDataError(
            f"Only {len(featured_df)} rows remain for '{ticker}' after feature "
            f"engineering; at least {MIN_ROWS_REQUIRED} are required. Try a "
            "longer history (e.g. 5 or 10 years)."
        )

    if cache:
        _cache_dataframe(featured_df, ticker, suffix="processed")

    return featured_df
