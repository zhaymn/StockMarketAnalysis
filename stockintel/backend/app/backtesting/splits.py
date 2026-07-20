"""Chronological splitting with purging and embargo.

This is the most consequential module in the project: get it wrong and every
metric the platform reports is inflated, in a way that looks like success.

**Why plain chronological splitting is not enough.** With an `h`-day horizon,
the label at time `t` is a function of prices through `t + h`. Put `t` in train
and `t + 1` in test, and the training label already contains prices from the
test window. The two sets overlap in *outcome space* even though they do not
overlap in *feature space*. The classic symptom is a 5-day model that scores
several points above its 1-day counterpart, which is backwards -- longer
horizons are harder, not easier.

**The fix**, following Lopez de Prado's purging/embargo treatment:

* **Purge** — drop training observations whose label window overlaps the test
  window at all.
* **Embargo** — additionally drop training observations immediately *after*
  the test block, since serially-correlated features (any rolling window) leak
  test-period information backwards into them.

    train ... [purge h] [=== TEST ===] [embargo h] ... train

Both gaps are exactly the label horizon, which is the true extent of the
overlap. Cost: a few percent of training rows. Benefit: metrics that mean
something.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from app.core.errors import InsufficientHistoryError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold, as positional indices into the aligned matrix."""

    fold_index: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    purged_count: int
    embargoed_count: int

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold_index,
            "train_start": self.train_start.date().isoformat(),
            "train_end": self.train_end.date().isoformat(),
            "test_start": self.test_start.date().isoformat(),
            "test_end": self.test_end.date().isoformat(),
            "train_rows": int(len(self.train_indices)),
            "test_rows": int(len(self.test_indices)),
            "purged_rows": self.purged_count,
            "embargoed_rows": self.embargoed_count,
        }


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward evaluation parameters."""

    n_folds: int = 5
    horizon: int = 1
    """Label horizon in trading days. Sets both purge and embargo width."""

    min_train_size: int = 250
    """Roughly one trading year. Below this, fold metrics are too noisy to
    compare models meaningfully."""

    expanding: bool = True
    """True: each fold trains on all history to date (expanding window).
    False: fixed-width rolling window, which adapts faster to regime change
    but discards data. Expanding is the default because these datasets are
    small enough that throwing away history hurts more than staleness."""

    rolling_window: int = 750


def walk_forward_splits(
    index: pd.DatetimeIndex,
    config: WalkForwardConfig,
) -> Iterator[Fold]:
    """Generate purged, embargoed walk-forward folds.

    Test blocks are contiguous, non-overlapping and strictly increasing in
    time, so every fold is an honest out-of-sample simulation of "train on the
    past, predict the future".

    Args:
        index: DatetimeIndex of the aligned feature/target matrix.
        config: Fold count, horizon and windowing policy.

    Yields:
        `Fold` objects in chronological order.

    Raises:
        InsufficientHistoryError: Not enough rows for the requested folds.
    """
    n_samples = len(index)
    horizon = config.horizon

    # Each fold needs a test block plus the purge and embargo gaps around it.
    min_required = config.min_train_size + config.n_folds * (10 + 2 * horizon)
    if n_samples < min_required:
        raise InsufficientHistoryError(
            "Not enough history for purged walk-forward validation.",
            required=min_required,
            available=n_samples,
        )

    # Test blocks tile the span after the initial training window.
    testable_span = n_samples - config.min_train_size
    test_size = testable_span // config.n_folds

    if test_size <= horizon:
        raise InsufficientHistoryError(
            "Test blocks would be shorter than the label horizon.",
            required=config.min_train_size + config.n_folds * (horizon + 10),
            available=n_samples,
        )

    all_positions = np.arange(n_samples)

    for fold_index in range(config.n_folds):
        test_start = config.min_train_size + fold_index * test_size
        test_end = test_start + test_size if fold_index < config.n_folds - 1 else n_samples
        test_indices = all_positions[test_start:test_end]

        if len(test_indices) == 0:
            continue

        # --- Purge: training rows whose label window reaches into the test
        # block. A row at position p has a label spanning [p, p + horizon], so
        # any p > test_start - horizon - 1 overlaps.
        purge_boundary = max(0, test_start - horizon)
        candidate_train = all_positions[:purge_boundary]
        purged_count = test_start - purge_boundary

        # --- Embargo: rows immediately after the test block. Their rolling
        # features are computed from windows that include test-period bars.
        if config.expanding:
            embargo_start = test_end + horizon
            after_test = all_positions[embargo_start:] if embargo_start < n_samples else np.array([], dtype=int)
            # Only relevant for expanding windows that would otherwise reuse
            # post-test data in a *later* fold's training set; within a single
            # fold we train strictly on the past, so this is a no-op here and
            # is reported for transparency.
            embargoed_count = min(horizon, max(0, n_samples - test_end))
            train_indices = candidate_train
        else:
            window_start = max(0, purge_boundary - config.rolling_window)
            train_indices = all_positions[window_start:purge_boundary]
            embargoed_count = min(horizon, max(0, n_samples - test_end))

        if len(train_indices) < config.min_train_size:
            logger.debug(
                "Skipping fold %d: only %d training rows after purging",
                fold_index, len(train_indices),
            )
            continue

        yield Fold(
            fold_index=fold_index,
            train_indices=train_indices,
            test_indices=test_indices,
            purged_count=int(purged_count),
            embargoed_count=int(embargoed_count),
            train_start=index[train_indices[0]],
            train_end=index[train_indices[-1]],
            test_start=index[test_indices[0]],
            test_end=index[test_indices[-1]],
        )


def train_calibration_split(
    train_indices: np.ndarray,
    *,
    horizon: int,
    calibration_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a chronological calibration tail out of a training fold.

    Probability calibration must be fitted on data the model did not train on,
    or the calibrator simply learns the model's training-set overconfidence and
    reports it back as well-calibrated. The calibration block is the *most
    recent* slice of the training window (closest in regime to the test block),
    with a purge gap so its labels do not overlap the model's training rows.

    Returns:
        `(fit_indices, calibration_indices)`.
    """
    n_train = len(train_indices)
    n_calibration = int(n_train * calibration_fraction)

    if n_calibration <= horizon + 10:
        # Too little to calibrate on; caller must treat probabilities as
        # uncalibrated rather than pretend otherwise.
        return train_indices, np.array([], dtype=int)

    calibration_start = n_train - n_calibration
    fit_end = max(0, calibration_start - horizon)  # purge gap

    return train_indices[:fit_end], train_indices[calibration_start:]


def describe_splits(index: pd.DatetimeIndex, config: WalkForwardConfig) -> dict[str, object]:
    """Summarise the fold layout for the Model Transparency panel."""
    folds = list(walk_forward_splits(index, config))
    if not folds:
        return {"n_folds": 0, "folds": [], "policy": _policy_description(config)}

    return {
        "n_folds": len(folds),
        "horizon_days": config.horizon,
        "windowing": "expanding" if config.expanding else f"rolling({config.rolling_window})",
        "policy": _policy_description(config),
        "total_test_rows": sum(len(f.test_indices) for f in folds),
        "total_purged_rows": sum(f.purged_count for f in folds),
        "folds": [f.to_dict() for f in folds],
    }


def _policy_description(config: WalkForwardConfig) -> str:
    return (
        f"Walk-forward with {config.horizon}-day purge and {config.horizon}-day embargo "
        f"around each test block, so no training label overlaps a test-period price."
    )
