"""Classification and calibration metrics.

Two principles shape this module.

**1. Every accuracy is reported against its baseline.** "54% directional
accuracy" is meaningless alone -- if the stock rose on 53% of days, that
model has produced one percentage point of signal. So `skill_score` is
computed everywhere and surfaced beside the raw figure.

**2. Probabilities must earn the right to be displayed.** `CalibrationReport`
implements the gate: a probability is shown only if it beats the base rate on
the Brier score out-of-sample. Otherwise the UI shows direction with
PROBABILITY NOT CALIBRATED. A confident-looking "68%" from an uncalibrated
model is exactly the fabrication the brief forbids -- it is a number, but it
is not a probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.core.logging import get_logger

logger = get_logger(__name__)

#: A model must beat the majority-class baseline's Brier score by at least this
#: fraction before its probabilities are labelled calibrated. Small but
#: strictly positive: it must actually add information, not merely tie.
MIN_BRIER_SKILL = 0.01


@dataclass
class ClassificationReport:
    """Out-of-sample classification performance."""

    n_samples: int
    n_classes: int
    class_labels: tuple[str, ...]

    accuracy: float
    baseline_accuracy: float
    """Majority-class rate on the test set -- the bar to clear."""

    skill_score: float
    """(accuracy - baseline) / (1 - baseline). Zero means no skill;
    negative means worse than always guessing the majority class."""

    precision_macro: float
    recall_macro: float
    f1_macro: float
    matthews_corrcoef: float
    """Robust to class imbalance in a way accuracy and F1 are not."""

    roc_auc: float | None
    log_loss: float | None
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "class_labels": list(self.class_labels),
            "accuracy": round(self.accuracy, 4),
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "skill_score": round(self.skill_score, 4),
            "beats_baseline": self.skill_score > 0,
            "precision_macro": round(self.precision_macro, 4),
            "recall_macro": round(self.recall_macro, 4),
            "f1_macro": round(self.f1_macro, 4),
            "matthews_corrcoef": round(self.matthews_corrcoef, 4),
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "log_loss": round(self.log_loss, 4) if self.log_loss is not None else None,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion,
        }


@dataclass
class CalibrationReport:
    """Whether predicted probabilities can be trusted as probabilities."""

    brier_score: float
    baseline_brier: float
    brier_skill_score: float
    """1 - (brier / baseline_brier). Positive means better than always
    predicting the base rate."""

    is_calibrated: bool
    """The display gate. False means the UI must suppress the probability."""

    reliability_bins: list[dict[str, float]]
    """Predicted vs observed frequency per bin -- the reliability diagram."""

    max_calibration_error: float
    expected_calibration_error: float

    def to_dict(self) -> dict[str, object]:
        return {
            "brier_score": round(self.brier_score, 4),
            "baseline_brier": round(self.baseline_brier, 4),
            "brier_skill_score": round(self.brier_skill_score, 4),
            "is_calibrated": self.is_calibrated,
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "max_calibration_error": round(self.max_calibration_error, 4),
            "reliability_bins": self.reliability_bins,
            "interpretation": self.interpretation(),
        }

    def interpretation(self) -> str:
        if not self.is_calibrated:
            return (
                "Predicted probabilities did not beat the base rate out-of-sample. "
                "Probability values are withheld; only direction is reported."
            )
        return (
            f"Probabilities carry {self.brier_skill_score:.1%} Brier skill over the "
            f"base rate, with {self.expected_calibration_error:.1%} average "
            f"calibration error."
        )


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: tuple[str, ...],
    y_proba: np.ndarray | None = None,
) -> ClassificationReport:
    """Score predictions against outcomes.

    Args:
        y_true: Observed class indices.
        y_pred: Predicted class indices.
        class_labels: Names, ordered by class index.
        y_proba: Optional `(n_samples, n_classes)` probabilities.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n_classes = len(class_labels)

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty prediction set.")

    accuracy = float(accuracy_score(y_true, y_pred))

    # The bar: always predict whichever class is most common in this test set.
    class_counts = np.bincount(y_true, minlength=n_classes)
    baseline_accuracy = float(class_counts.max() / len(y_true))

    denominator = 1.0 - baseline_accuracy
    skill = (accuracy - baseline_accuracy) / denominator if denominator > 1e-9 else 0.0

    labels = list(range(n_classes))
    per_class: dict[str, dict[str, float]] = {}
    precisions = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    for index, label in enumerate(class_labels):
        per_class[label] = {
            "precision": round(float(precisions[index]), 4),
            "recall": round(float(recalls[index]), 4),
            "f1": round(float(f1s[index]), 4),
            "support": int(class_counts[index]),
        }

    roc_auc: float | None = None
    loss: float | None = None
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            if n_classes == 2:
                roc_auc = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                roc_auc = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
            loss = float(log_loss(y_true, y_proba, labels=labels))
        except ValueError as exc:
            # e.g. a class absent from this test block. Report None rather than
            # a misleading number.
            logger.debug("Could not compute AUC/log-loss: %s", exc)

    return ClassificationReport(
        n_samples=len(y_true),
        n_classes=n_classes,
        class_labels=class_labels,
        accuracy=accuracy,
        baseline_accuracy=baseline_accuracy,
        skill_score=float(skill),
        precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        matthews_corrcoef=float(matthews_corrcoef(y_true, y_pred)),
        roc_auc=roc_auc,
        log_loss=loss,
        per_class=per_class,
        confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    )


def evaluate_calibration(
    y_true_binary: np.ndarray,
    y_proba_positive: np.ndarray,
    *,
    n_bins: int = 10,
) -> CalibrationReport:
    """Assess whether probabilities behave like probabilities.

    Of all the samples a well-calibrated model assigns 70% to, about 70% should
    actually occur. This measures that directly and sets the display gate.
    """
    y_true_binary = np.asarray(y_true_binary).astype(int)
    y_proba_positive = np.clip(np.asarray(y_proba_positive, dtype=float), 0.0, 1.0)

    brier = float(brier_score_loss(y_true_binary, y_proba_positive))

    # Baseline: predict the observed base rate for every sample.
    base_rate = float(y_true_binary.mean())
    baseline_brier = float(base_rate * (1.0 - base_rate))

    skill = 1.0 - (brier / baseline_brier) if baseline_brier > 1e-9 else 0.0

    # Reliability diagram.
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, float]] = []
    calibration_errors: list[tuple[float, float]] = []

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_proba_positive >= lower) & (
            y_proba_positive < upper if upper < 1.0 else y_proba_positive <= upper
        )
        count = int(mask.sum())
        if count == 0:
            continue

        mean_predicted = float(y_proba_positive[mask].mean())
        observed_frequency = float(y_true_binary[mask].mean())
        bins.append({
            "bin_lower": round(float(lower), 3),
            "bin_upper": round(float(upper), 3),
            "mean_predicted": round(mean_predicted, 4),
            "observed_frequency": round(observed_frequency, 4),
            "count": count,
        })
        calibration_errors.append((abs(mean_predicted - observed_frequency), count))

    if calibration_errors:
        total = sum(count for _, count in calibration_errors)
        expected_error = sum(error * count for error, count in calibration_errors) / total
        max_error = max(error for error, _ in calibration_errors)
    else:
        expected_error = max_error = 0.0

    return CalibrationReport(
        brier_score=brier,
        baseline_brier=baseline_brier,
        brier_skill_score=float(skill),
        is_calibrated=skill > MIN_BRIER_SKILL,
        reliability_bins=bins,
        max_calibration_error=float(max_error),
        expected_calibration_error=float(expected_error),
    )


def aggregate_fold_reports(reports: list[ClassificationReport]) -> dict[str, object]:
    """Combine per-fold reports into the headline out-of-sample figure.

    Reports the standard deviation across folds alongside the mean, because a
    model averaging 55% at ±12% across folds is not meaningfully better than
    its baseline -- and showing only the mean would hide that.
    """
    if not reports:
        return {"n_folds": 0}

    accuracies = np.array([r.accuracy for r in reports])
    baselines = np.array([r.baseline_accuracy for r in reports])
    skills = np.array([r.skill_score for r in reports])
    f1s = np.array([r.f1_macro for r in reports])
    mccs = np.array([r.matthews_corrcoef for r in reports])
    aucs = [r.roc_auc for r in reports if r.roc_auc is not None]

    folds_beating_baseline = int((accuracies > baselines).sum())

    return {
        "n_folds": len(reports),
        "total_test_samples": int(sum(r.n_samples for r in reports)),
        "accuracy_mean": round(float(accuracies.mean()), 4),
        "accuracy_std": round(float(accuracies.std()), 4),
        "baseline_accuracy_mean": round(float(baselines.mean()), 4),
        "skill_score_mean": round(float(skills.mean()), 4),
        "f1_macro_mean": round(float(f1s.mean()), 4),
        "matthews_corrcoef_mean": round(float(mccs.mean()), 4),
        "roc_auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
        "folds_beating_baseline": folds_beating_baseline,
        "consistency": f"{folds_beating_baseline}/{len(reports)} folds beat baseline",
    }
