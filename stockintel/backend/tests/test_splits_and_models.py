"""Tests for split integrity, metrics honesty and model behaviour.

The split tests matter most: they verify the property that makes every
downstream metric meaningful. They use a synthetic index because the property
under test (no train/test overlap in label space) is about index arithmetic,
not about market data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtesting.metrics import evaluate_calibration, evaluate_classification
from app.backtesting.splits import (
    WalkForwardConfig,
    train_calibration_split,
    walk_forward_splits,
)
from app.core.errors import InsufficientHistoryError


@pytest.fixture
def business_index():
    return pd.DatetimeIndex(pd.bdate_range("2018-01-01", periods=1500))


# --- Split integrity -------------------------------------------------------

@pytest.mark.parametrize("horizon", [1, 5, 10])
def test_train_never_overlaps_test_label_window(business_index, horizon):
    """The core guarantee: no training label may span into the test block."""
    config = WalkForwardConfig(n_folds=5, horizon=horizon)

    for fold in walk_forward_splits(business_index, config):
        train_max = fold.train_indices.max()
        test_min = fold.test_indices.min()

        # A training row at position p has a label covering [p, p + horizon].
        # That window must end strictly before the test block starts.
        assert train_max + horizon <= test_min, (
            f"Fold {fold.fold_index}: training row {train_max} has a label "
            f"reaching position {train_max + horizon}, but the test block "
            f"starts at {test_min}. This is leakage."
        )


def test_train_is_strictly_before_test(business_index):
    """No fold may train on data that comes after its test block."""
    config = WalkForwardConfig(n_folds=5, horizon=5)
    for fold in walk_forward_splits(business_index, config):
        assert fold.train_indices.max() < fold.test_indices.min()
        assert fold.train_end < fold.test_start


def test_test_blocks_are_disjoint_and_ordered(business_index):
    config = WalkForwardConfig(n_folds=5, horizon=1)
    folds = list(walk_forward_splits(business_index, config))

    assert len(folds) >= 3
    seen: set[int] = set()
    previous_end = -1

    for fold in folds:
        positions = set(fold.test_indices.tolist())
        assert not (positions & seen), "Test blocks must not overlap"
        seen |= positions
        assert fold.test_indices.min() > previous_end
        previous_end = int(fold.test_indices.max())


def test_purge_widens_with_horizon(business_index):
    """A longer horizon must purge more rows -- that is the whole mechanism."""
    purged_by_horizon = {}
    for horizon in (1, 10):
        config = WalkForwardConfig(n_folds=5, horizon=horizon)
        folds = list(walk_forward_splits(business_index, config))
        purged_by_horizon[horizon] = sum(f.purged_count for f in folds)

    assert purged_by_horizon[10] > purged_by_horizon[1]


def test_short_history_raises_rather_than_producing_junk_folds():
    short_index = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=60))
    with pytest.raises(InsufficientHistoryError):
        list(walk_forward_splits(short_index, WalkForwardConfig(n_folds=5, horizon=5)))


def test_calibration_split_is_purged_and_chronological():
    train_indices = np.arange(1000)
    fit, calibration = train_calibration_split(train_indices, horizon=5)

    assert len(fit) > 0 and len(calibration) > 0
    # Calibration block must come after the fit block, with a purge gap.
    assert fit.max() + 5 <= calibration.min()
    assert calibration.max() == 999


# --- Metrics honesty -------------------------------------------------------

def test_skill_score_is_zero_for_majority_guessing():
    """A model that always guesses the majority class must score zero skill."""
    y_true = np.array([1] * 70 + [0] * 30)
    y_pred = np.ones(100, dtype=int)

    report = evaluate_classification(y_true, y_pred, ("DOWN", "UP"))
    assert report.accuracy == pytest.approx(0.70)
    assert report.baseline_accuracy == pytest.approx(0.70)
    assert report.skill_score == pytest.approx(0.0, abs=1e-9)
    assert not report.to_dict()["beats_baseline"]


def test_skill_score_is_negative_when_worse_than_baseline():
    y_true = np.array([1] * 70 + [0] * 30)
    y_pred = np.zeros(100, dtype=int)

    report = evaluate_classification(y_true, y_pred, ("DOWN", "UP"))
    assert report.skill_score < 0


def test_uncalibrated_probabilities_are_rejected():
    """An overconfident model must fail the calibration gate."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    # Confident predictions that are pure noise.
    y_proba = rng.choice([0.05, 0.95], size=500)

    report = evaluate_calibration(y_true, y_proba)
    assert not report.is_calibrated
    assert report.brier_skill_score < 0
    assert "withheld" in report.interpretation()


def test_well_calibrated_probabilities_pass_the_gate():
    rng = np.random.default_rng(1)
    y_proba = rng.uniform(0.05, 0.95, size=4000)
    # Outcomes generated to actually match the stated probabilities.
    y_true = (rng.uniform(size=4000) < y_proba).astype(int)

    report = evaluate_calibration(y_true, y_proba)
    assert report.is_calibrated
    assert report.brier_skill_score > 0
    assert report.expected_calibration_error < 0.10
