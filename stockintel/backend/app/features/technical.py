"""Technical feature engineering.

Two rules govern this module.

**1. No future data.** Every value at index `t` is computed only from bars at
`t` or earlier. All rolling windows are trailing; no `center=True`, no
`shift(-n)`, no `bfill`. `assert_no_lookahead` in `app.features.leakage`
verifies this empirically rather than on trust.

**2. Curated, not exhaustive.** The brief lists ~15 candidate indicator
families; naively emitting all of them produces heavily collinear columns
(SMA-20, EMA-20 and Bollinger mid-band are near-identical), which destabilises
linear meta-learners and makes tree feature-importance meaningless by splitting
one signal's credit across three columns. So indicators are emitted mostly as
**normalised, stationary derivatives** -- distance-from-MA rather than the MA
level, %B rather than raw band positions -- which are comparable across
symbols and price regimes. A raw SMA-200 of 1800 tells a model about Reliance's
price scale, not its trend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)

#: Longest lookback any feature needs. Rows before this are dropped because
#: their features are computed from a partial window and are not comparable.
MAX_LOOKBACK = 200


@dataclass(frozen=True)
class FeatureGroup:
    """A named block of related features, for UI attribution and ablation."""

    name: str
    columns: tuple[str, ...]
    description: str


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (an EMA with alpha = 1/period).

    Wilder's original RSI/ATR definitions use this, not a simple mean. The
    difference is small but persistent, and using the wrong one makes published
    indicator values disagree with any charting platform the user cross-checks.
    """
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index, Wilder's formulation. Range 0-100."""
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    avg_gain = _wilder_smooth(gains, period)
    avg_loss = _wilder_smooth(losses, period)

    # A zero average loss means an unbroken up-run: RSI is 100 by definition.
    # Guarding here avoids a divide-by-zero inf that would poison the scaler.
    relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    return rsi.where(avg_loss != 0.0, 100.0).where(avg_gain != 0.0, other=rsi)


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line and histogram.

    Returned normalised by price so values are comparable across symbols: a raw
    MACD of 3.0 means something very different on a $20 stock and a $900 one.
    """
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()

    macd_line = (ema_fast - ema_slow) / close
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range, normalised by close (i.e. as a fraction of price)."""
    high, low, close = frame["High"], frame["Low"], frame["Close"]
    prev_close = close.shift(1)

    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return _wilder_smooth(true_range, period) / close


def compute_bollinger(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    """Bollinger %B and bandwidth.

    %B locates price within the bands (0 = lower, 1 = upper); bandwidth measures
    band width relative to the mid-band and is a clean volatility-regime proxy.
    Both are scale-free, unlike the raw band levels.
    """
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()

    upper = mid + num_std * std
    lower = mid - num_std * std

    band_range = (upper - lower).replace(0.0, np.nan)
    percent_b = (close - lower) / band_range
    bandwidth = (upper - lower) / mid
    return percent_b, bandwidth


def build_features(
    frame: pd.DataFrame,
    *,
    trading_days_per_year: int = 252,
    drop_warmup: bool = True,
) -> tuple[pd.DataFrame, list[FeatureGroup]]:
    """Build the full technical feature matrix from cleaned OHLCV.

    Args:
        frame: Cleaned OHLCV, DatetimeIndex ascending (see `data.market.prices`).
        trading_days_per_year: Market-specific annualisation factor.
        drop_warmup: Drop the first `MAX_LOOKBACK` rows, whose features come
            from partial windows. Leave True for training; set False only when
            you need feature values aligned to the original index.

    Returns:
        `(features, groups)` -- the matrix, and its group metadata for the
        "input signals" panel and for ablation studies.
    """
    if frame.empty:
        raise ValueError("Cannot build features from an empty frame.")

    close = frame["Close"]
    volume = frame["Volume"]
    features = pd.DataFrame(index=frame.index)

    # --- Returns -----------------------------------------------------------
    # Log returns are the modelling primitive: additive across time and far
    # closer to symmetric than simple returns.
    log_return = np.log(close / close.shift(1))
    features["log_return_1d"] = log_return
    features["return_1d"] = close.pct_change()

    # Lagged returns give a tabular model the short-horizon autocorrelation
    # structure an LSTM reads directly from its sequence input.
    for lag in (1, 2, 3, 5, 10):
        features[f"log_return_lag_{lag}"] = log_return.shift(lag)

    # Cumulative returns over trailing windows: momentum at several scales.
    for window in (5, 10, 21, 63):
        features[f"return_{window}d"] = close.pct_change(window)

    # --- Moving-average relationships --------------------------------------
    # Distance from the MA, not the MA level -- scale-free and stationary.
    for window in (20, 50, 200):
        sma = close.rolling(window, min_periods=window).mean()
        features[f"dist_sma_{window}"] = (close - sma) / sma

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    features["dist_ema_12"] = (close - ema_12) / ema_12
    features["ema_12_26_spread"] = (ema_12 - ema_26) / ema_26

    # Golden/death-cross state as a single signed feature.
    sma_50 = close.rolling(50, min_periods=50).mean()
    sma_200 = close.rolling(200, min_periods=200).mean()
    features["sma_50_200_spread"] = (sma_50 - sma_200) / sma_200

    # --- Momentum oscillators ---------------------------------------------
    # Centred on 0 and scaled to roughly [-1, 1]: keeps the LSTM's inputs in a
    # comparable range to the return features without a separate scaler.
    features["rsi_14"] = (compute_rsi(close, 14) - 50.0) / 50.0

    macd_line, macd_signal, macd_hist = compute_macd(close)
    features["macd"] = macd_line
    features["macd_signal"] = macd_signal
    features["macd_histogram"] = macd_hist

    # --- Volatility --------------------------------------------------------
    for window in (10, 21):
        features[f"volatility_{window}d"] = (
            log_return.rolling(window, min_periods=window).std()
            * np.sqrt(trading_days_per_year)
        )

    # Volatility regime: current vol against its own 6-month norm. >1 means the
    # stock is currently more volatile than it usually is -- a key input to the
    # risk level shown beside the prediction.
    vol_21 = features["volatility_21d"]
    features["volatility_regime"] = vol_21 / vol_21.rolling(126, min_periods=126).mean()

    features["atr_14"] = compute_atr(frame, 14)

    percent_b, bandwidth = compute_bollinger(close)
    features["bollinger_percent_b"] = percent_b
    features["bollinger_bandwidth"] = bandwidth

    # Downside deviation: only negative returns. Distinguishes "volatile
    # because it is falling" from "volatile because it is rising".
    downside = log_return.where(log_return < 0.0, 0.0)
    features["downside_volatility_21d"] = (
        downside.rolling(21, min_periods=21).std() * np.sqrt(trading_days_per_year)
    )

    # --- Volume ------------------------------------------------------------
    # Guarded against the all-zero volume case (some indices report no volume),
    # which would otherwise produce inf and silently break scaling.
    safe_volume = volume.replace(0, np.nan)
    avg_volume_20 = safe_volume.rolling(20, min_periods=20).mean()
    features["relative_volume"] = safe_volume / avg_volume_20
    features["log_volume_change"] = np.log(safe_volume / safe_volume.shift(1))
    features["volume_trend_5d"] = (
        safe_volume.rolling(5, min_periods=5).mean() / avg_volume_20
    )

    # --- Range position ----------------------------------------------------
    # Where price sits inside its trailing range: 1 = at the high, 0 = at the low.
    for window in (20, 252):
        rolling_high = close.rolling(window, min_periods=window).max()
        rolling_low = close.rolling(window, min_periods=window).min()
        span = (rolling_high - rolling_low).replace(0.0, np.nan)
        features[f"range_position_{window}d"] = (close - rolling_low) / span

    # --- Trend strength ----------------------------------------------------
    # R^2 of a least-squares line through the last 21 log-closes: how *cleanly*
    # trending the move is, independent of direction. Signed by slope so a
    # clean downtrend reads negative.
    features["trend_strength_21d"] = _rolling_trend_strength(np.log(close), window=21)

    features = features.replace([np.inf, -np.inf], np.nan)

    if drop_warmup:
        features = features.iloc[MAX_LOOKBACK:]

    groups = _feature_groups(features.columns)
    logger.debug("Built %d technical features over %d rows", features.shape[1], len(features))
    return features, groups


def _rolling_trend_strength(log_close: pd.Series, window: int = 21) -> pd.Series:
    """Signed R^2 of a trailing linear fit to log price.

    Vectorised via rolling moments rather than a per-window `polyfit`, which
    would be ~100x slower over a decade of daily bars.
    """
    x = pd.Series(np.arange(len(log_close), dtype=float), index=log_close.index)

    mean_x = x.rolling(window, min_periods=window).mean()
    mean_y = log_close.rolling(window, min_periods=window).mean()

    cov = (x * log_close).rolling(window, min_periods=window).mean() - mean_x * mean_y
    var_x = (x * x).rolling(window, min_periods=window).mean() - mean_x**2
    var_y = (log_close * log_close).rolling(window, min_periods=window).mean() - mean_y**2

    denominator = (var_x * var_y).pow(0.5).replace(0.0, np.nan)
    correlation = cov / denominator
    # correlation^2 is R^2; the sign of the correlation carries the direction.
    return correlation.abs() * correlation


def _feature_groups(columns: pd.Index) -> list[FeatureGroup]:
    """Partition emitted columns into named groups for UI attribution."""
    definitions: list[tuple[str, tuple[str, ...], str]] = [
        ("Returns & momentum",
         ("log_return", "return_", "dist_sma", "dist_ema", "ema_12_26", "sma_50_200"),
         "Trailing returns and price position relative to moving averages."),
        ("Oscillators",
         ("rsi_", "macd"),
         "RSI and MACD, normalised to be comparable across price scales."),
        ("Volatility & risk",
         ("volatility_", "atr_", "bollinger_", "downside_"),
         "Realised volatility, ATR, Bollinger geometry and downside deviation."),
        ("Volume",
         ("relative_volume", "log_volume", "volume_trend"),
         "Participation relative to the stock's own recent norm."),
        ("Trend & range",
         ("range_position", "trend_strength"),
         "Position within the trailing range and trend cleanliness."),
    ]

    groups: list[FeatureGroup] = []
    for name, prefixes, description in definitions:
        matched = tuple(
            column for column in columns if any(column.startswith(p) for p in prefixes)
        )
        if matched:
            groups.append(FeatureGroup(name=name, columns=matched, description=description))
    return groups
