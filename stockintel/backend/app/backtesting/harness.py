"""Walk-forward evaluation harness.

Runs every candidate model over identical purged/embargoed folds and reports
what actually happened. It also produces the out-of-fold prediction matrix the
ensemble meta-learner trains on -- which is the only leak-free way to stack:
training a meta-learner on in-sample base-model predictions teaches it to trust
models that have memorised the training set.

Regime breakdown is computed here too, because an aggregate accuracy can hide
a model that works only in calm markets and fails exactly when it matters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.backtesting.metrics import (
    ClassificationReport,
    aggregate_fold_reports,
    evaluate_calibration,
    evaluate_classification,
)
from app.backtesting.splits import Fold, WalkForwardConfig, walk_forward_splits
from app.core.logging import get_logger
from app.models.base import Predictor

logger = get_logger(__name__)


@dataclass
class ModelResult:
    """Walk-forward result for one model."""

    model_name: str
    family: str
    description: str

    fold_reports: list[ClassificationReport] = field(default_factory=list)
    aggregate: dict[str, object] = field(default_factory=dict)
    calibration: dict[str, object] | None = None
    regime_breakdown: dict[str, dict[str, object]] = field(default_factory=dict)
    feature_importance: dict[str, float] | None = None
    fit_seconds: float = 0.0

    #: Out-of-fold probabilities, aligned to `oof_index`. Feeds the ensemble.
    oof_proba: np.ndarray | None = None
    oof_index: pd.DatetimeIndex | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model_name,
            "family": self.family,
            "description": self.description,
            "aggregate": self.aggregate,
            "calibration": self.calibration,
            "regime_breakdown": self.regime_breakdown,
            "fit_seconds": round(self.fit_seconds, 2),
            "folds": [r.to_dict() for r in self.fold_reports],
        }
        if self.feature_importance:
            top = sorted(self.feature_importance.items(), key=lambda kv: -kv[1])[:15]
            payload["top_features"] = [
                {"feature": name, "importance": round(value, 4)} for name, value in top
            ]
        return payload


def run_walk_forward(
    model_factory,
    x: pd.DataFrame,
    y: np.ndarray,
    class_labels: tuple[str, ...],
    config: WalkForwardConfig,
    *,
    regime_series: pd.Series | None = None,
    random_seed: int = 42,
) -> ModelResult:
    """Evaluate one model across purged walk-forward folds.

    Args:
        model_factory: Zero-arg callable returning a fresh `Predictor`. A
            factory, not an instance -- every fold must train from scratch, or
            fold 5 carries fold 1's fitted state and the evaluation is void.
        x: Aligned feature matrix.
        y: Aligned integer class labels.
        class_labels: Class names by index.
        config: Walk-forward configuration.
        regime_series: Optional per-row regime label for the breakdown.
    """
    y = np.asarray(y).astype(int)
    folds = list(walk_forward_splits(x.index, config))

    if not folds:
        raise ValueError("Walk-forward produced no usable folds.")

    probe_model = model_factory()
    result = ModelResult(
        model_name=probe_model.name,
        family=probe_model.family,
        description=probe_model.description,
    )

    oof_proba = np.full((len(x), len(class_labels)), np.nan)
    started = time.perf_counter()

    for fold in folds:
        model = model_factory()

        x_train = x.iloc[fold.train_indices]
        y_train = y[fold.train_indices]
        x_test = x.iloc[fold.test_indices]
        y_test = y[fold.test_indices]

        if len(np.unique(y_train)) < 2:
            logger.warning(
                "Fold %d has a single class in training; skipping.", fold.fold_index
            )
            continue

        try:
            model.fit(x_train, y_train)
            proba = model.predict_proba(x_test)
            predictions = proba.argmax(axis=1)
        except Exception as exc:
            # One model failing must not abort the whole comparison; the
            # dashboard reports it as unavailable rather than silently omitting.
            logger.error("Model %s failed on fold %d: %s", model.name, fold.fold_index, exc)
            continue

        oof_proba[fold.test_indices] = proba
        result.fold_reports.append(
            evaluate_classification(y_test, predictions, class_labels, proba)
        )

    result.fit_seconds = time.perf_counter() - started

    if not result.fold_reports:
        raise ValueError(f"Model {result.model_name} produced no usable folds.")

    result.aggregate = aggregate_fold_reports(result.fold_reports)

    evaluated = ~np.isnan(oof_proba).any(axis=1)
    result.oof_proba = oof_proba
    result.oof_index = x.index

    # --- Calibration, over all out-of-fold predictions pooled --------------
    if evaluated.sum() > 50:
        y_evaluated = y[evaluated]
        proba_evaluated = oof_proba[evaluated]

        if len(class_labels) == 2:
            positive_proba = proba_evaluated[:, 1]
            binary_truth = y_evaluated
        else:
            # For the 3-class outlook, calibrate the directional call that the
            # UI actually surfaces: P(BULLISH) against "was it bullish".
            bullish_index = len(class_labels) - 1
            positive_proba = proba_evaluated[:, bullish_index]
            binary_truth = (y_evaluated == bullish_index).astype(int)

        result.calibration = evaluate_calibration(binary_truth, positive_proba).to_dict()

    # --- Regime breakdown --------------------------------------------------
    if regime_series is not None:
        result.regime_breakdown = _regime_breakdown(
            y, oof_proba, evaluated, regime_series, x.index, class_labels
        )

    # --- Feature importance, refit on all data for a stable ranking --------
    try:
        final_model = model_factory()
        final_model.fit(x, y)
        result.feature_importance = final_model.feature_importance()
    except Exception as exc:
        logger.debug("Could not compute feature importance for %s: %s", result.model_name, exc)

    logger.info(
        "%s: accuracy %.4f (baseline %.4f) over %d folds",
        result.model_name,
        result.aggregate.get("accuracy_mean", float("nan")),
        result.aggregate.get("baseline_accuracy_mean", float("nan")),
        len(result.fold_reports),
    )
    return result


def run_selective_walk_forward(
    model_factory,
    x: pd.DataFrame,
    y: np.ndarray,
    class_labels: tuple[str, ...],
    config: WalkForwardConfig,
    *,
    random_seed: int = 42,
) -> dict[str, object]:
    """Walk-forward evaluation of a model that may abstain.

    Per fold the training window is split chronologically into a fit block and
    a purged validation tail. The model trains on the fit block; the abstention
    threshold is fitted on the validation tail; both are then applied unchanged
    to the test block. The threshold never sees test data, so the reported
    selective accuracy is a genuine out-of-sample number.

    Test-block predictions are pooled across folds before scoring, since any
    single fold may contain too few answered samples to score stably.
    """
    from app.backtesting.splits import train_calibration_split
    from app.models.selective import (
        SelectiveReport,
        evaluate_at_threshold,
        fit_threshold,
        risk_coverage_curve,
    )

    y = np.asarray(y).astype(int)
    folds = list(walk_forward_splits(x.index, config))
    if not folds:
        raise ValueError("Walk-forward produced no usable folds.")

    pooled_truth: list[np.ndarray] = []
    pooled_proba: list[np.ndarray] = []
    pooled_answered: list[np.ndarray] = []
    fold_thresholds: list[float] = []

    for fold in folds:
        fit_indices, validation_indices = train_calibration_split(
            fold.train_indices, horizon=config.horizon, calibration_fraction=0.25
        )

        if len(validation_indices) == 0 or len(fit_indices) < config.min_train_size // 2:
            logger.debug("Fold %d: no usable validation tail; skipping.", fold.fold_index)
            continue

        y_fit = y[fit_indices]
        if len(np.unique(y_fit)) < 2:
            continue

        model = model_factory()
        try:
            model.fit(x.iloc[fit_indices], y_fit)
            proba_validation = model.predict_proba(x.iloc[validation_indices])
            proba_test = model.predict_proba(x.iloc[fold.test_indices])
        except Exception as exc:
            logger.error("Selective run failed on fold %d: %s", fold.fold_index, exc)
            continue

        threshold, _ = fit_threshold(y[validation_indices], proba_validation, class_labels)
        fold_thresholds.append(threshold)

        from app.models.selective import confidence_margin

        pooled_truth.append(y[fold.test_indices])
        pooled_proba.append(proba_test)
        pooled_answered.append(confidence_margin(proba_test) >= threshold)

    if not pooled_truth:
        return {"available": False, "reason": "No fold produced a usable validation split."}

    truth = np.concatenate(pooled_truth)
    proba = np.concatenate(pooled_proba)
    answered = np.concatenate(pooled_answered)

    n_answered = int(answered.sum())
    if n_answered == 0:
        return {
            "available": True,
            "abstains_always": True,
            "coverage": 0.0,
            "interpretation": (
                "The model's confidence never cleared its fitted threshold out-of-sample. "
                "No directional calls are issued."
            ),
        }

    # Score the answered subset directly, using the per-fold thresholds already
    # applied -- not a single global threshold refitted here, which would leak.
    y_answered = truth[answered]
    predictions = proba[answered].argmax(axis=1)
    accuracy = float((predictions == y_answered).mean())

    counts = np.bincount(y_answered, minlength=len(class_labels))
    baseline = float(counts.max() / n_answered)
    denominator = 1.0 - baseline
    skill = (accuracy - baseline) / denominator if denominator > 1e-9 else 0.0

    report = SelectiveReport(
        coverage=n_answered / len(truth),
        n_answered=n_answered,
        n_total=len(truth),
        selective_accuracy=accuracy,
        baseline_accuracy=baseline,
        skill_score=float(skill),
        threshold=float(np.median(fold_thresholds)) if fold_thresholds else 0.0,
        class_distribution_answered={
            label: int(counts[index]) for index, label in enumerate(class_labels)
        },
    )

    payload = report.to_dict()
    payload["available"] = True
    payload["abstains_always"] = False
    payload["median_fold_threshold"] = round(float(np.median(fold_thresholds)), 4)
    payload["risk_coverage_curve"] = risk_coverage_curve(truth, proba, class_labels)
    return payload


def _regime_breakdown(
    y: np.ndarray,
    oof_proba: np.ndarray,
    evaluated: np.ndarray,
    regime_series: pd.Series,
    index: pd.DatetimeIndex,
    class_labels: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    """Performance sliced by market regime.

    A model that is accurate overall but fails in high-volatility regimes is
    dangerous precisely when a user most wants a prediction, so this is shown
    in the UI rather than buried.
    """
    regimes = regime_series.reindex(index)
    breakdown: dict[str, dict[str, object]] = {}

    for regime_name in regimes.dropna().unique():
        mask = evaluated & (regimes == regime_name).to_numpy()
        if mask.sum() < 30:
            # Too few observations for a stable estimate; omitting is more
            # honest than publishing an accuracy from 12 samples.
            continue

        report = evaluate_classification(
            y[mask], oof_proba[mask].argmax(axis=1), class_labels, oof_proba[mask]
        )
        breakdown[str(regime_name)] = {
            "n_samples": report.n_samples,
            "accuracy": round(report.accuracy, 4),
            "baseline_accuracy": round(report.baseline_accuracy, 4),
            "skill_score": round(report.skill_score, 4),
            "beats_baseline": report.skill_score > 0,
        }

    return breakdown


def build_regime_labels(features: pd.DataFrame) -> pd.Series:
    """Label each row with its market regime, from features already computed.

    Volatility terciles crossed with trend direction. Thresholds come from the
    stock's own history, so "high volatility" means high *for this stock*.
    """
    if "volatility_21d" not in features.columns:
        return pd.Series("unknown", index=features.index)

    volatility = features["volatility_21d"]
    low, high = volatility.quantile([0.33, 0.67])

    trend = features.get("dist_sma_50")
    labels = pd.Series(index=features.index, dtype="object")

    for timestamp in features.index:
        vol = volatility.get(timestamp, np.nan)
        if pd.isna(vol):
            labels[timestamp] = "unknown"
            continue

        vol_label = "high-vol" if vol > high else ("low-vol" if vol < low else "mid-vol")

        if trend is not None and not pd.isna(trend.get(timestamp, np.nan)):
            trend_label = "uptrend" if trend[timestamp] > 0 else "downtrend"
            labels[timestamp] = f"{vol_label} / {trend_label}"
        else:
            labels[timestamp] = vol_label

    return labels
