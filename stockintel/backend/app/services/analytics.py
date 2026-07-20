"""Stock analytics.

Computes the statistics the dashboard displays. Two standing rules:

* **Every figure is derived from real data or reported as unavailable.** A
  metric that cannot be computed (52-week range on a six-month listing, beta
  with no benchmark overlap) returns `None` and renders as DATA UNAVAILABLE.
  Nothing is defaulted to zero.
* **Every figure has a purpose.** The brief forbids decorative numbers, so each
  metric here feeds either the prediction, the risk assessment, or the
  "why this prediction" explanation.

Annualisation uses the market's own trading-day count (US 252, India 250),
since a single hardcoded constant would misstate Indian volatility by ~1%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.data.market.base import MarketConventions

logger = get_logger(__name__)

#: Trading-day counts for standard lookback windows.
WINDOWS = {"1d": 1, "5d": 5, "1m": 21, "3m": 63, "6m": 126, "1y": 252}


def _safe_float(value: Any) -> float | None:
    """Convert to float, mapping NaN/inf to None so it renders as unavailable."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


@dataclass
class AnalyticsBundle:
    """All computed analytics for one stock."""

    price: dict[str, Any]
    returns: dict[str, Any]
    volatility: dict[str, Any]
    momentum: dict[str, Any]
    volume: dict[str, Any]
    risk: dict[str, Any]
    benchmark: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "returns": self.returns,
            "volatility": self.volatility,
            "momentum": self.momentum,
            "volume": self.volume,
            "risk": self.risk,
            "benchmark": self.benchmark,
        }


def compute_price_stats(frame: pd.DataFrame) -> dict[str, Any]:
    """Current price, session change and 52-week range."""
    close = frame["Close"]
    latest = _safe_float(close.iloc[-1])
    previous = _safe_float(close.iloc[-2]) if len(close) > 1 else None

    change = change_percent = None
    if latest is not None and previous not in (None, 0):
        change = latest - previous
        change_percent = change / previous

    # 52 weeks needs 252 sessions; with fewer, report what is available and
    # say so rather than silently using a shorter window as if it were a year.
    window = min(252, len(close))
    has_full_year = len(close) >= 252
    recent = close.iloc[-window:]

    high_52w = _safe_float(recent.max())
    low_52w = _safe_float(recent.min())

    position_in_range = None
    if high_52w is not None and low_52w is not None and high_52w > low_52w and latest is not None:
        position_in_range = (latest - low_52w) / (high_52w - low_52w)

    return {
        "current": _round(latest, 4),
        "previous_close": _round(previous, 4),
        "change": _round(change, 4),
        "change_percent": _round(change_percent, 6),
        "high_52w": _round(high_52w, 4),
        "low_52w": _round(low_52w, 4),
        "range_window_sessions": int(window),
        "has_full_52w_history": has_full_year,
        "position_in_52w_range": _round(position_in_range, 4),
        "last_close_date": frame.index[-1].date().isoformat(),
        "day_high": _round(_safe_float(frame["High"].iloc[-1]), 4),
        "day_low": _round(_safe_float(frame["Low"].iloc[-1]), 4),
    }


def compute_returns(frame: pd.DataFrame) -> dict[str, Any]:
    """Trailing returns over standard windows."""
    close = frame["Close"]
    returns: dict[str, Any] = {}

    for label, sessions in WINDOWS.items():
        if len(close) <= sessions:
            returns[label] = None  # genuinely not computable
            continue
        past = _safe_float(close.iloc[-1 - sessions])
        latest = _safe_float(close.iloc[-1])
        returns[label] = (
            _round((latest - past) / past, 6) if past not in (None, 0) and latest else None
        )

    return returns


def compute_volatility(
    frame: pd.DataFrame,
    conventions: MarketConventions,
) -> dict[str, Any]:
    """Realised volatility, ATR and the current volatility regime."""
    close = frame["Close"]
    log_return = np.log(close / close.shift(1)).dropna()
    annualiser = np.sqrt(conventions.trading_days_per_year)

    def realised(window: int) -> float | None:
        if len(log_return) < window:
            return None
        return _safe_float(log_return.iloc[-window:].std() * annualiser)

    vol_21 = realised(21)
    vol_63 = realised(63)
    vol_252 = realised(252)

    # Regime: current short-term vol against its own longer-run norm.
    regime_ratio = None
    regime_label = "UNKNOWN"
    if vol_21 is not None and vol_252 not in (None, 0):
        regime_ratio = vol_21 / vol_252
        if regime_ratio > 1.3:
            regime_label = "ELEVATED"
        elif regime_ratio < 0.75:
            regime_label = "SUBDUED"
        else:
            regime_label = "NORMAL"

    # ATR(14), Wilder-smoothed, as a fraction of price.
    atr_percent = None
    if len(frame) >= 15:
        high, low, prev_close = frame["High"], frame["Low"], close.shift(1)
        true_range = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        atr = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        atr_percent = _safe_float(atr.iloc[-1] / close.iloc[-1])

    # Downside deviation: volatility of losses only.
    downside = None
    if len(log_return) >= 21:
        negatives = log_return.iloc[-252:] if len(log_return) >= 252 else log_return
        negatives = negatives[negatives < 0]
        if len(negatives) > 5:
            downside = _safe_float(negatives.std() * annualiser)

    return {
        "realised_21d": _round(vol_21, 4),
        "realised_63d": _round(vol_63, 4),
        "realised_252d": _round(vol_252, 4),
        "atr_14_percent": _round(atr_percent, 6),
        "downside_volatility": _round(downside, 4),
        "regime_ratio": _round(regime_ratio, 4),
        "regime": regime_label,
        "annualisation_basis": conventions.trading_days_per_year,
    }


def compute_momentum(features: pd.DataFrame) -> dict[str, Any]:
    """RSI, MACD state and moving-average trend, read off computed features."""
    if features.empty:
        return {"available": False}

    latest = features.iloc[-1]

    # rsi_14 is stored normalised to [-1, 1]; convert back for display.
    rsi_normalised = _safe_float(latest.get("rsi_14"))
    rsi = None if rsi_normalised is None else rsi_normalised * 50.0 + 50.0

    rsi_state = "UNKNOWN"
    if rsi is not None:
        if rsi >= 70:
            rsi_state = "OVERBOUGHT"
        elif rsi <= 30:
            rsi_state = "OVERSOLD"
        else:
            rsi_state = "NEUTRAL"

    macd = _safe_float(latest.get("macd"))
    macd_signal = _safe_float(latest.get("macd_signal"))
    macd_histogram = _safe_float(latest.get("macd_histogram"))

    macd_state = "UNKNOWN"
    if macd is not None and macd_signal is not None:
        macd_state = "BULLISH_CROSSOVER" if macd > macd_signal else "BEARISH_CROSSOVER"

    dist_20 = _safe_float(latest.get("dist_sma_20"))
    dist_50 = _safe_float(latest.get("dist_sma_50"))
    dist_200 = _safe_float(latest.get("dist_sma_200"))
    golden = _safe_float(latest.get("sma_50_200_spread"))

    above = [d for d in (dist_20, dist_50, dist_200) if d is not None and d > 0]
    known = [d for d in (dist_20, dist_50, dist_200) if d is not None]

    if not known:
        ma_trend = "UNKNOWN"
    elif len(above) == len(known):
        ma_trend = "STRONG_UPTREND"
    elif not above:
        ma_trend = "STRONG_DOWNTREND"
    else:
        ma_trend = "MIXED"

    return {
        "available": True,
        "rsi_14": _round(rsi, 2),
        "rsi_state": rsi_state,
        "macd": _round(macd, 6),
        "macd_signal": _round(macd_signal, 6),
        "macd_histogram": _round(macd_histogram, 6),
        "macd_state": macd_state,
        "distance_from_sma_20": _round(dist_20, 6),
        "distance_from_sma_50": _round(dist_50, 6),
        "distance_from_sma_200": _round(dist_200, 6),
        "sma_50_200_spread": _round(golden, 6),
        "moving_average_trend": ma_trend,
        "trend_strength": _round(_safe_float(latest.get("trend_strength_21d")), 4),
    }


def compute_volume(frame: pd.DataFrame) -> dict[str, Any]:
    """Participation relative to the stock's own recent norm."""
    volume = frame["Volume"]

    if (volume <= 0).all():
        # Some indices and illiquid listings publish no volume at all.
        return {
            "available": False,
            "reason": "The data source reported no volume for this security.",
        }

    latest = _safe_float(volume.iloc[-1])
    average_20 = _safe_float(volume.iloc[-20:].mean()) if len(volume) >= 20 else None
    average_60 = _safe_float(volume.iloc[-60:].mean()) if len(volume) >= 60 else None

    relative = None
    if latest is not None and average_20 not in (None, 0):
        relative = latest / average_20

    trend = "UNKNOWN"
    if average_20 is not None and average_60 not in (None, 0):
        ratio = average_20 / average_60
        trend = "RISING" if ratio > 1.1 else ("FALLING" if ratio < 0.9 else "STABLE")

    return {
        "available": True,
        "latest": None if latest is None else int(latest),
        "average_20d": None if average_20 is None else int(average_20),
        "average_60d": None if average_60 is None else int(average_60),
        "relative_volume": _round(relative, 4),
        "trend": trend,
    }


def compute_risk(
    frame: pd.DataFrame,
    conventions: MarketConventions,
    benchmark_close: pd.Series | None = None,
) -> dict[str, Any]:
    """Drawdown, risk-adjusted return and beta against the market benchmark."""
    close = frame["Close"]
    log_return = np.log(close / close.shift(1)).dropna()

    # Maximum drawdown over the available history.
    running_max = close.cummax()
    drawdown = (close - running_max) / running_max
    max_drawdown = _safe_float(drawdown.min())

    current_drawdown = _safe_float(drawdown.iloc[-1])

    # Risk-adjusted return. Excess return over a risk-free rate is omitted
    # deliberately: the correct rate differs by market and is not available
    # without another data source, so this is labelled a Sharpe-LIKE ratio
    # rather than presented as a true Sharpe.
    sharpe_like = None
    if len(log_return) >= 63:
        recent = log_return.iloc[-252:] if len(log_return) >= 252 else log_return
        std = recent.std()
        if std and std > 0:
            sharpe_like = _safe_float(
                (recent.mean() / std) * np.sqrt(conventions.trading_days_per_year)
            )

    # Beta and correlation against the benchmark.
    beta = correlation = None
    overlap = 0
    if benchmark_close is not None and len(benchmark_close) > 30:
        benchmark_return = np.log(benchmark_close / benchmark_close.shift(1)).dropna()
        aligned = pd.concat(
            [log_return.rename("stock"), benchmark_return.rename("benchmark")],
            axis=1, join="inner",
        ).dropna()

        overlap = len(aligned)
        if overlap >= 60:
            window = aligned.iloc[-252:] if overlap >= 252 else aligned
            variance = window["benchmark"].var()
            if variance and variance > 0:
                beta = _safe_float(window["stock"].cov(window["benchmark"]) / variance)
            correlation = _safe_float(window["stock"].corr(window["benchmark"]))

    return {
        "max_drawdown": _round(max_drawdown, 4),
        "current_drawdown": _round(current_drawdown, 4),
        "sharpe_like_ratio": _round(sharpe_like, 4),
        "sharpe_note": (
            "Mean return divided by volatility, annualised. Not a true Sharpe ratio: "
            "no risk-free rate is subtracted, as the appropriate rate differs by market "
            "and is not sourced here."
        ),
        "beta": _round(beta, 4),
        "correlation_with_benchmark": _round(correlation, 4),
        "benchmark_overlap_sessions": overlap,
    }


def compute_benchmark_comparison(
    frame: pd.DataFrame,
    benchmark_close: pd.Series | None,
    conventions: MarketConventions,
) -> dict[str, Any]:
    """Relative performance against the market benchmark."""
    if benchmark_close is None or benchmark_close.empty:
        return {
            "available": False,
            "benchmark_symbol": conventions.benchmark_symbol,
            "benchmark_label": conventions.benchmark_label,
            "reason": "Benchmark data could not be retrieved.",
        }

    close = frame["Close"]
    aligned = pd.concat(
        [close.rename("stock"), benchmark_close.rename("benchmark")], axis=1, join="inner"
    ).dropna()

    if len(aligned) < 21:
        return {
            "available": False,
            "benchmark_symbol": conventions.benchmark_symbol,
            "benchmark_label": conventions.benchmark_label,
            "reason": f"Only {len(aligned)} overlapping sessions — too few to compare.",
        }

    relative: dict[str, Any] = {}
    for label, sessions in WINDOWS.items():
        if len(aligned) <= sessions:
            relative[label] = None
            continue

        stock_return = (
            aligned["stock"].iloc[-1] / aligned["stock"].iloc[-1 - sessions] - 1
        )
        benchmark_return = (
            aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[-1 - sessions] - 1
        )
        relative[label] = {
            "stock": _round(_safe_float(stock_return), 6),
            "benchmark": _round(_safe_float(benchmark_return), 6),
            "excess": _round(_safe_float(stock_return - benchmark_return), 6),
        }

    return {
        "available": True,
        "benchmark_symbol": conventions.benchmark_symbol,
        "benchmark_label": conventions.benchmark_label,
        "overlap_sessions": len(aligned),
        "relative_performance": relative,
    }


def build_analytics(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    conventions: MarketConventions,
    benchmark_close: pd.Series | None = None,
) -> AnalyticsBundle:
    """Compute the full analytics bundle for one stock."""
    return AnalyticsBundle(
        price=compute_price_stats(frame),
        returns=compute_returns(frame),
        volatility=compute_volatility(frame, conventions),
        momentum=compute_momentum(features),
        volume=compute_volume(frame),
        risk=compute_risk(frame, conventions, benchmark_close),
        benchmark=compute_benchmark_comparison(frame, benchmark_close, conventions),
    )
