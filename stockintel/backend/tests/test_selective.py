"""Tests for selective prediction.

The first test is the important one. Abstention preferentially answers where
one class dominates, which inflates the majority-class rate on the answered
subset (measured: +0.017 mean across 36 real configurations, higher in 25/36).
Scoring against the *full-period* baseline therefore flatters a selective model
for free. On real data that difference flipped the verdict from "beats baseline
in 17/36" to the honest "11/36" -- so this invariant is what keeps the platform
from reporting an edge it does not have.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.models.selective import (
    confidence_margin,
    evaluate_at_threshold,
    fit_threshold,
    risk_coverage_curve,
)

LABELS_3 = ("BEARISH", "NEUTRAL", "BULLISH")
LABELS_2 = ("DOWN", "UP")


def test_baseline_is_computed_on_the_answered_subset_not_the_full_period():
    """The anti-flattery guard.

    Construct a case where the model answers only on a class-skewed subset. If
    the baseline were computed over the full period it would look skilful; on
    the answered subset it is exactly break-even.
    """
    # 100 samples: the first 40 are almost all class 1, the rest are balanced.
    y_true = np.array([1] * 36 + [0] * 4 + [0, 1] * 30)

    # High confidence only on the first 40 (the skewed block).
    proba = np.zeros((100, 2))
    proba[:40] = [0.05, 0.95]      # margin 0.90 -> answered
    proba[40:] = [0.48, 0.52]      # margin 0.04 -> abstained

    report = evaluate_at_threshold(y_true, proba, 0.5, LABELS_2)

    assert report.n_answered == 40
    # It predicts class 1 on all 40 and is right 36 times.
    assert report.selective_accuracy == pytest.approx(0.90)

    # The honest bar is the subset's majority rate (36/40), NOT the full-period
    # rate (66/100 = 0.66). Against the wrong bar this would look like a large
    # edge; against the right one it is exactly zero skill.
    assert report.baseline_accuracy == pytest.approx(0.90)
    assert report.skill_score == pytest.approx(0.0, abs=1e-9)
    assert not report.has_edge


def test_confidence_margin_uses_top_two_gap():
    # Same top probability, different runner-up: different confidence.
    decisive = np.array([[0.40, 0.20, 0.40]])   # tie at top -> margin 0
    clear = np.array([[0.40, 0.30, 0.30]])      # margin 0.10

    assert confidence_margin(decisive)[0] == pytest.approx(0.0)
    assert confidence_margin(clear)[0] == pytest.approx(0.10)


def test_threshold_targets_requested_coverage():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(3), size=1000)
    y = rng.integers(0, 3, size=1000)

    threshold, report = fit_threshold(y, proba, LABELS_3, target_coverage=0.20)
    # Quantile targeting should land close to the requested coverage.
    assert 0.15 <= report.coverage <= 0.25
    assert np.isfinite(threshold)


def test_abstains_when_validation_is_too_small_to_fit():
    rng = np.random.default_rng(1)
    proba = rng.dirichlet(np.ones(3), size=10)
    y = rng.integers(0, 3, size=10)

    threshold, report = fit_threshold(y, proba, LABELS_3)
    assert threshold == float("inf")
    assert report.coverage == 0.0
    assert not report.has_edge
    assert "withheld" in report.interpretation()


def test_zero_coverage_reports_withheld_rather_than_zero_accuracy():
    y = np.array([0, 1, 2] * 40)
    proba = np.full((120, 3), 1 / 3)  # no confidence anywhere

    report = evaluate_at_threshold(y, proba, threshold=0.5, class_labels=LABELS_3)
    assert report.n_answered == 0
    assert not report.has_edge
    # Must not be presented as "0% accurate" -- it made no claims at all.
    assert "withheld" in report.interpretation()


def test_risk_coverage_curve_detects_an_informative_confidence_signal():
    """A genuine signal slopes downward; the test asserts we can detect that."""
    rng = np.random.default_rng(2)
    n = 2000
    y = rng.integers(0, 2, size=n)

    # Construct probabilities that are genuinely more reliable when confident.
    proba = np.zeros((n, 2))
    for i in range(n):
        confident = i < n // 4
        correct = rng.uniform() < (0.85 if confident else 0.50)
        p = (0.95 if confident else 0.55) if correct else (0.05 if confident else 0.45)
        proba[i] = [1 - p, p] if y[i] == 1 else [p, 1 - p]

    curve = risk_coverage_curve(y, proba, LABELS_2)
    assert len(curve) > 5

    coverages = np.array([p["coverage"] for p in curve])
    accuracies = np.array([p["accuracy"] for p in curve])
    slope = np.polyfit(coverages, accuracies, 1)[0]
    assert slope < -0.02, "Should detect that accuracy rises as coverage falls"


def test_uninformative_confidence_produces_a_flat_curve():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=2000)
    proba = rng.dirichlet(np.ones(2), size=2000)  # confidence unrelated to truth

    curve = risk_coverage_curve(y, proba, LABELS_2)
    coverages = np.array([p["coverage"] for p in curve])
    accuracies = np.array([p["accuracy"] for p in curve])
    slope = np.polyfit(coverages, accuracies, 1)[0]
    assert abs(slope) < 0.10, "Random confidence must not look informative"
