"""Prediction-target construction.

The brief is explicit that "tomorrow's price = $X" is a misleading output, and
it is right: a next-day price regressor trained on a near-random-walk series
learns to echo today's close, scores a flattering R^2, and predicts nothing.
So the targets here are directional.

**1-day** — binary. `UP` if the next session's log return is positive.

**5-day** — three-class, with a neutral band scaled by realised volatility:

    BULLISH   r_5d > +k·sigma_5d
    BEARISH   r_5d < -k·sigma_5d
    NEUTRAL   otherwise

The sigma scaling is the important part. A fixed band (say ±2%) means something
completely different for a utility and for a small-cap, and across calm and
panicked regimes; classes built that way are incoherent and unlearnable. A
volatility-scaled band makes NEUTRAL "moved less than this stock normally moves
over five days" -- a real, earnable state rather than a measure-zero boundary.

`sigma` at time `t` uses only returns up to `t`, so the label is knowable at
prediction time apart from its forward return, which is exactly the intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd


class Direction(IntEnum):
    """1-day binary target."""

    DOWN = 0
    UP = 1


class Outlook(IntEnum):
    """5-day three-class target. Ordinal: BEARISH < NEUTRAL < BULLISH."""

    BEARISH = 0
    NEUTRAL = 1
    BULLISH = 2


#: Neutral half-width in units of forward-horizon sigma. 0.5 puts roughly a
#: third of historical observations in NEUTRAL for a typical large-cap, giving
#: three usable classes rather than a degenerate one.
DEFAULT_NEUTRAL_BAND_SIGMA = 0.5

#: Trailing window for the volatility estimate that scales the neutral band.
VOLATILITY_WINDOW = 63


@dataclass(frozen=True)
class TargetSpec:
    """Definition of one prediction target, carried through to the UI."""

    name: str
    horizon_days: int
    n_classes: int
    class_labels: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "horizon_days": self.horizon_days,
            "n_classes": self.n_classes,
            "class_labels": list(self.class_labels),
            "description": self.description,
        }


DIRECTION_1D = TargetSpec(
    name="direction_1d",
    horizon_days=1,
    n_classes=2,
    class_labels=("DOWN", "UP"),
    description="Direction of the next trading session's close-to-close return.",
)

OUTLOOK_5D = TargetSpec(
    name="outlook_5d",
    horizon_days=5,
    n_classes=3,
    class_labels=("BEARISH", "NEUTRAL", "BULLISH"),
    description=(
        "Five-session outlook. BULLISH/BEARISH require a move larger than "
        "0.5x the stock's own recent 5-day volatility; otherwise NEUTRAL."
    ),
)

AVAILABLE_TARGETS: dict[str, TargetSpec] = {
    DIRECTION_1D.name: DIRECTION_1D,
    OUTLOOK_5D.name: OUTLOOK_5D,
}


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """Log return from `t` to `t + horizon`.

    The final `horizon` rows are NaN by construction -- their outcome has not
    happened yet. Callers must drop them before training; keeping them would
    mean training on labels that do not exist.
    """
    return np.log(close.shift(-horizon) / close)


def build_direction_target(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Binary UP/DOWN target over `horizon` sessions."""
    forward = forward_log_return(close, horizon)
    target = (forward > 0).astype("float")
    return target.where(forward.notna()).rename(f"direction_{horizon}d")


def build_outlook_target(
    close: pd.Series,
    horizon: int = 5,
    *,
    band_sigma: float = DEFAULT_NEUTRAL_BAND_SIGMA,
    volatility_window: int = VOLATILITY_WINDOW,
) -> tuple[pd.Series, pd.Series]:
    """Three-class outlook with a volatility-scaled neutral band.

    Returns:
        `(target, band_halfwidth)` -- the class labels, and the band used at
        each timestamp, which the UI displays so the thresholds are inspectable
        rather than magic.
    """
    log_return = np.log(close / close.shift(1))

    # Trailing daily sigma, scaled to the horizon by sqrt-of-time. Uses only
    # data up to t, so the threshold is knowable at prediction time.
    daily_sigma = log_return.rolling(volatility_window, min_periods=volatility_window).std()
    horizon_sigma = daily_sigma * np.sqrt(horizon)
    band = band_sigma * horizon_sigma

    forward = forward_log_return(close, horizon)

    target = pd.Series(np.nan, index=close.index, dtype="float")
    valid = forward.notna() & band.notna()

    target[valid & (forward > band)] = float(Outlook.BULLISH)
    target[valid & (forward < -band)] = float(Outlook.BEARISH)
    target[valid & (forward.abs() <= band)] = float(Outlook.NEUTRAL)

    return target.rename(f"outlook_{horizon}d"), band.rename("neutral_band")


def build_target(close: pd.Series, spec: TargetSpec) -> tuple[pd.Series, pd.Series | None]:
    """Build the target described by `spec`.

    Returns `(target, band)`; `band` is None for binary targets.
    """
    if spec.n_classes == 2:
        return build_direction_target(close, spec.horizon_days), None
    target, band = build_outlook_target(close, spec.horizon_days)
    return target, band


def class_balance(target: pd.Series, spec: TargetSpec) -> dict[str, float]:
    """Observed class proportions.

    The majority-class share is the floor any classifier must clear to be worth
    anything -- reported alongside every accuracy figure in this platform, so
    "54% accurate" can be read against the "53% always-guess-up" baseline it
    needs to beat.
    """
    clean = target.dropna()
    if clean.empty:
        return {label: 0.0 for label in spec.class_labels}

    counts = clean.value_counts(normalize=True)
    return {
        label: float(counts.get(float(index), 0.0))
        for index, label in enumerate(spec.class_labels)
    }


def align_features_and_target(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Inner-join features and target, dropping rows either side cannot supply.

    Drops the warm-up head (features NaN) and the unresolved tail (target NaN)
    in one step, keeping X and y index-identical -- misalignment here is the
    single easiest way to silently train against shifted labels.
    """
    combined = features.join(target.rename("__target__"), how="inner")
    combined = combined.dropna(subset=["__target__"])
    combined = combined.dropna()

    aligned_target = combined["__target__"]
    aligned_features = combined.drop(columns="__target__")
    return aligned_features, aligned_target
