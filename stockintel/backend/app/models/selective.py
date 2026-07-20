"""Selective prediction — models allowed to say "no edge today".

The walk-forward comparison established the problem this solves. Across four
stocks and two horizons, every ML model produced ROC-AUC of 0.52-0.55 (never
0.50) and positive MCC, yet none beat majority-class accuracy. That pattern is
diagnostic: real ranking signal, destroyed by forcing a hard decision at the
argmax threshold on a 53/47 class split.

Selective classification (Chow's reject option; El-Yaniv & Wiener's
risk-coverage framework) is the standard answer. The model answers only when
its probability departs from the base rate by more than a fitted margin, and
abstains otherwise. Accuracy is then measured *on the answered subset*, with
coverage reported alongside -- "called 18% of sessions at 61% accuracy" is a
claim that can be true, where "53% every day" is not.

**The honesty detail that makes this real.** The comparison baseline must be
the majority-class rate computed *on the same answered subset*, not on the full
period. Otherwise a model games the metric trivially by answering only when the
majority class is obvious, scoring 70% against a 53% full-period baseline while
adding nothing. `SelectiveReport.baseline_accuracy` is always the subset rate.

**Thresholds are fitted on validation data**, never on test. Choosing the
threshold that maximises test accuracy and then reporting that accuracy is
just a slower way of overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

#: Never report a threshold fitted on fewer answered validation samples than
#: this -- the resulting accuracy estimate is too noisy to act on.
MIN_ANSWERED_FOR_FIT = 40

#: Coverage floor. A model answering under 5% of sessions is not a product,
#: however good those few calls look.
MIN_COVERAGE = 0.05

#: Coverage ceiling for threshold search. Above this the model is effectively
#: always answering, which is the regime already shown not to work.
MAX_COVERAGE = 0.80


@dataclass
class SelectiveReport:
    """Accuracy on answered predictions, with coverage."""

    coverage: float
    """Fraction of samples the model chose to answer."""

    n_answered: int
    n_total: int

    selective_accuracy: float
    baseline_accuracy: float
    """Majority-class rate ON THE ANSWERED SUBSET. See module docstring."""

    skill_score: float
    threshold: float
    class_distribution_answered: dict[str, int] = field(default_factory=dict)

    @property
    def has_edge(self) -> bool:
        """Whether answering beat guessing the majority on the same subset."""
        return self.skill_score > 0 and self.n_answered >= MIN_ANSWERED_FOR_FIT

    def to_dict(self) -> dict[str, object]:
        return {
            "coverage": round(self.coverage, 4),
            "n_answered": self.n_answered,
            "n_total": self.n_total,
            "selective_accuracy": round(self.selective_accuracy, 4),
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "skill_score": round(self.skill_score, 4),
            "threshold": round(self.threshold, 4),
            "has_edge": self.has_edge,
            "class_distribution_answered": self.class_distribution_answered,
            "interpretation": self.interpretation(),
        }

    def interpretation(self) -> str:
        if self.n_answered < MIN_ANSWERED_FOR_FIT:
            return (
                f"Answered only {self.n_answered} of {self.n_total} samples — too few "
                f"to establish an edge. Predictions are withheld."
            )
        if not self.has_edge:
            return (
                f"On the {self.coverage:.0%} of sessions where a call was made, accuracy "
                f"was {self.selective_accuracy:.1%} against a {self.baseline_accuracy:.1%} "
                f"majority-class rate on those same sessions — no demonstrated edge."
            )
        return (
            f"A directional call was made on {self.coverage:.0%} of sessions. On those, "
            f"accuracy was {self.selective_accuracy:.1%} against a {self.baseline_accuracy:.1%} "
            f"majority-class rate on the same sessions."
        )


def confidence_margin(proba: np.ndarray) -> np.ndarray:
    """Per-sample confidence: top probability minus runner-up.

    Preferred over raw max-probability because it measures how *decisively* the
    model separated its top choice. With three classes, a 0.40 top probability
    means something very different against a 0.39 runner-up than a 0.20 one.
    """
    if proba.shape[1] < 2:
        return np.zeros(len(proba))
    partitioned = np.partition(proba, -2, axis=1)
    return partitioned[:, -1] - partitioned[:, -2]


def evaluate_at_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    class_labels: tuple[str, ...],
) -> SelectiveReport:
    """Score the answered subset at one abstention threshold."""
    y_true = np.asarray(y_true).astype(int)
    margins = confidence_margin(proba)
    answered = margins >= threshold

    n_total = len(y_true)
    n_answered = int(answered.sum())

    if n_answered == 0:
        return SelectiveReport(
            coverage=0.0, n_answered=0, n_total=n_total,
            selective_accuracy=0.0, baseline_accuracy=0.0,
            skill_score=0.0, threshold=threshold,
        )

    y_answered = y_true[answered]
    predictions = proba[answered].argmax(axis=1)

    accuracy = float((predictions == y_answered).mean())

    # The fair bar: majority class *on this subset*.
    counts = np.bincount(y_answered, minlength=len(class_labels))
    baseline = float(counts.max() / n_answered)

    denominator = 1.0 - baseline
    skill = (accuracy - baseline) / denominator if denominator > 1e-9 else 0.0

    return SelectiveReport(
        coverage=n_answered / n_total,
        n_answered=n_answered,
        n_total=n_total,
        selective_accuracy=accuracy,
        baseline_accuracy=baseline,
        skill_score=float(skill),
        threshold=threshold,
        class_distribution_answered={
            label: int(counts[index]) for index, label in enumerate(class_labels)
        },
    )


#: Fraction of sessions the model aims to answer. Empirically derived, not
#: guessed: risk-coverage curves across 36 stock/horizon/model configurations
#: showed accuracy rising as coverage falls (mean slope -0.055, negative in
#: 25/36), with the usable edge concentrated below ~25% coverage. Above that,
#: the answered set is diluted by low-confidence calls and skill vanishes.
DEFAULT_TARGET_COVERAGE = 0.20


def fit_threshold(
    y_validation: np.ndarray,
    proba_validation: np.ndarray,
    class_labels: tuple[str, ...],
    *,
    target_coverage: float = DEFAULT_TARGET_COVERAGE,
) -> tuple[float, SelectiveReport]:
    """Choose an abstention threshold on validation data.

    **Why target a coverage quantile rather than maximise validation skill.**
    The first implementation searched for the threshold with the best measured
    skill on validation. That failed out-of-sample -- it reliably selected
    ~50% coverage and produced negative test skill, because at these sample
    sizes the validation skill curve is dominated by noise, and its argmax is
    therefore mostly noise too. Estimating a *quantile* of the margin
    distribution is far more stable than estimating the *argmax of a noisy
    accuracy curve*: it is one less quantity inferred from small data.

    So the threshold is simply the margin value that answers the most confident
    `target_coverage` of validation samples. The model's job is to rank; the
    coverage target decides how deep into that ranking to act.

    Returns:
        `(threshold, validation_report)`. An infinite threshold means "never
        answer", which is a legitimate outcome, not a failure.
    """
    margins = confidence_margin(proba_validation)

    if len(margins) == 0 or margins.max() <= 0:
        return float("inf"), _abstain_report(len(y_validation))

    target_coverage = float(np.clip(target_coverage, MIN_COVERAGE, MAX_COVERAGE))

    # Answer the top `target_coverage` fraction by confidence margin.
    threshold = float(np.quantile(margins, 1.0 - target_coverage))

    report = evaluate_at_threshold(y_validation, proba_validation, threshold, class_labels)

    if report.n_answered < MIN_ANSWERED_FOR_FIT:
        # Widen just enough to reach a scoreable sample, rather than reporting
        # an accuracy computed from a handful of observations.
        needed = MIN_ANSWERED_FOR_FIT / max(1, len(margins))
        if needed <= MAX_COVERAGE:
            threshold = float(np.quantile(margins, 1.0 - needed))
            report = evaluate_at_threshold(y_validation, proba_validation, threshold, class_labels)
        else:
            logger.info("Validation set too small to fit an abstention threshold; abstaining.")
            return float("inf"), _abstain_report(len(y_validation))

    return threshold, report


def _abstain_report(n_total: int) -> SelectiveReport:
    return SelectiveReport(
        coverage=0.0, n_answered=0, n_total=n_total,
        selective_accuracy=0.0, baseline_accuracy=0.0,
        skill_score=0.0, threshold=float("inf"),
    )


def _is_better(candidate: SelectiveReport, incumbent: SelectiveReport) -> bool:
    """Higher skill wins; near-ties break toward higher coverage."""
    if candidate.skill_score > incumbent.skill_score + 0.01:
        return True
    if abs(candidate.skill_score - incumbent.skill_score) <= 0.01:
        return candidate.coverage > incumbent.coverage
    return False


def risk_coverage_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    class_labels: tuple[str, ...],
    *,
    n_points: int = 20,
) -> list[dict[str, float]]:
    """Accuracy as a function of coverage, for the UI.

    A genuine confidence signal produces a downward-sloping curve: accuracy
    rises as coverage falls. A flat curve means the model's confidence is
    uninformative -- which is itself worth showing the user, since it says the
    probability ordering carries nothing.
    """
    margins = confidence_margin(proba)
    if len(margins) == 0:
        return []

    curve: list[dict[str, float]] = []
    for quantile in np.linspace(0.0, 0.9, n_points):
        threshold = float(np.quantile(margins, quantile))
        report = evaluate_at_threshold(y_true, proba, threshold, class_labels)
        if report.n_answered < 20:
            continue
        curve.append({
            "threshold": round(threshold, 4),
            "coverage": round(report.coverage, 4),
            "accuracy": round(report.selective_accuracy, 4),
            "baseline_accuracy": round(report.baseline_accuracy, 4),
            "skill_score": round(report.skill_score, 4),
            "n_answered": report.n_answered,
        })
    return curve
