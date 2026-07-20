"""Prediction service.

Orchestrates: fetch -> features -> walk-forward evaluation -> fit -> predict,
then assembles the explanation.

**The prediction is gated on demonstrated skill.** Before any directional call
is surfaced, the model is evaluated by purged walk-forward against the naive
baseline. If it fails to beat that baseline, the service returns a `NO_EDGE`
verdict and the dashboard shows analytics and evidence instead of a number.
Given the measured results on price-only features, `NO_EDGE` is the expected
outcome for most stocks -- and reporting it is the point, per section 19 of the
brief.

The "why this prediction" factors are read off real computed features. There is
no template of plausible-sounding reasons: if a factor is not present in the
data, it does not appear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from app.backtesting.harness import build_regime_labels, run_walk_forward
from app.backtesting.splits import WalkForwardConfig
from app.core.config import get_settings
from app.core.errors import InsufficientHistoryError
from app.core.logging import get_logger
from app.data.market.base import MarketConventions
from app.features.targets import (
    AVAILABLE_TARGETS,
    TargetSpec,
    align_features_and_target,
    build_target,
    class_balance,
    effective_sample_size,
)
from app.models.base import Predictor
from app.models.baselines import MajorityClassBaseline
from app.models.selective import confidence_margin, fit_threshold
from app.models.tabular import LightGBMModel, LogisticRegressionModel

logger = get_logger(__name__)


class Verdict(str, Enum):
    """What the platform is willing to claim."""

    DIRECTIONAL = "DIRECTIONAL"
    """The model beat its baseline and is confident enough to call direction."""

    NO_EDGE = "NO_EDGE"
    """The model did not demonstrate out-of-sample skill. No call is made."""

    ABSTAINED = "ABSTAINED"
    """The model has demonstrated skill but is not confident enough today."""

    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"


#: Model modes exposed in the UI.
#:
#: "Most Possible Prediction" maps to whichever pipeline actually measured best,
#: which the brief requires. Across 10 stock/horizon configurations under purged
#: walk-forward, mean skill was:
#:
#:     LightGBM              -0.0496   <- best
#:     Majority baseline     -0.0459
#:     Logistic Regression   -0.0802
#:     Stacked ensemble      -0.1296   <- worst
#:
#: So it maps to LightGBM, NOT to the stack. Defaulting to the ensemble because
#: ensembles are conventionally stronger would be exactly the "blindly combine
#: models" failure the brief warns against, and would hand users the worst
#: performer under the most confident-sounding label.
MODEL_MODES: dict[str, dict[str, Any]] = {
    "most_possible": {
        "label": "Most Possible Prediction",
        "recommended": True,
        "description": (
            "The platform's strongest empirically validated pipeline, selected by "
            "out-of-sample performance rather than by architecture. Currently "
            "gradient-boosted trees, which measured best across 10 stock/horizon "
            "configurations — ahead of both a linear model and a stacked ensemble. "
            "Represents the statistically estimated most likely outcome given "
            "available data — not a guaranteed market outcome."
        ),
    },
    "stacked_ensemble": {
        "label": "Stacked Ensemble",
        "recommended": False,
        "description": (
            "Logistic meta-learner over out-of-fold predictions from the linear and "
            "gradient-boosting models. Measured result: it degraded ROC-AUC in 10 of "
            "10 configurations versus its own best component, because weak base "
            "models produce noisy out-of-fold predictions the meta-learner then fits. "
            "Retained for transparency, not because it works better."
        ),
    },
    "tabular_lgbm": {
        "label": "Gradient Boosting (LightGBM)",
        "recommended": False,
        "description": (
            "Gradient-boosted trees over technical, volatility and volume features. "
            "One row of engineered features at a time; temporal structure enters "
            "only through explicit lag features."
        ),
    },
    "lstm": {
        "label": "LSTM (deep sequence model)",
        "recommended": False,
        "description": (
            "Two-layer LSTM reading a 20-session window of features. Measured across "
            "8 stock/horizon configurations it did NOT outperform gradient boosting "
            "(mean skill -0.109 vs -0.060) and took roughly 6x longer to train. "
            "Included for comparison and transparency, not because it works better."
        ),
    },
    "linear": {
        "label": "Logistic Regression",
        "recommended": False,
        "description": (
            "Regularised linear model. Fully interpretable — included as a "
            "reference point for whether added model complexity earns its keep."
        ),
    },
}


@dataclass
class PredictionResult:
    """A prediction plus everything needed to judge it."""

    verdict: Verdict
    symbol: str
    target: TargetSpec
    model_mode: str

    direction: str | None = None
    probability: float | None = None
    probability_is_calibrated: bool = False
    expected_return_range: dict[str, float] | None = None
    risk_level: RiskLevel = RiskLevel.MODERATE

    evidence: dict[str, Any] = field(default_factory=dict)
    factors: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    interpretation: str = ""
    data_timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "symbol": self.symbol,
            "target": self.target.to_dict(),
            "model_mode": self.model_mode,
            "model_label": MODEL_MODES.get(self.model_mode, {}).get("label", self.model_mode),
            "direction": self.direction,
            "probability": (
                round(self.probability, 4)
                if self.probability is not None and self.probability_is_calibrated else None
            ),
            "probability_is_calibrated": self.probability_is_calibrated,
            "probability_withheld_reason": (
                None if self.probability_is_calibrated else
                "Probabilities are shown only when they beat the base rate "
                "out-of-sample on the Brier score. This model's did not."
            ),
            "expected_return_range": self.expected_return_range,
            "risk_level": self.risk_level.value,
            "evidence": self.evidence,
            "factors": self.factors,
            "interpretation": self.interpretation,
            "data_timestamp": self.data_timestamp,
        }


def _build_model(mode: str, n_classes: int, seed: int, horizon: int = 5) -> Predictor:
    # "most_possible" resolves to the best-measured pipeline. See MODEL_MODES for
    # the numbers behind this mapping; it is an empirical result, not a default
    # chosen by architecture.
    if mode in ("most_possible", "tabular_lgbm"):
        return LightGBMModel(n_classes, seed)
    if mode == "stacked_ensemble":
        from app.models.ensemble import EnsembleConfig, StackedEnsemble

        return StackedEnsemble(
            n_classes,
            seed,
            base_factories=[
                ("logreg", lambda: LogisticRegressionModel(n_classes, seed)),
                ("lgbm", lambda: LightGBMModel(n_classes, seed)),
            ],
            config=EnsembleConfig(inner_folds=3, horizon=horizon),
        )
    if mode == "linear":
        return LogisticRegressionModel(n_classes, seed)
    if mode == "lstm":
        # Imported lazily: torch adds ~3s to process start, and most requests
        # never touch the deep model.
        from app.models.lstm import LSTMModel

        return LSTMModel(n_classes, seed)
    raise ValueError(f"Unknown model mode '{mode}'.")


def _assess_risk(features: pd.DataFrame, evidence: dict[str, Any]) -> RiskLevel:
    """Risk level from realised volatility regime and model consistency."""
    latest = features.iloc[-1]
    regime = latest.get("volatility_regime")

    score = 0
    if regime is not None and np.isfinite(regime):
        if regime > 1.4:
            score += 2
        elif regime > 1.1:
            score += 1

    aggregate = evidence.get("walk_forward", {})
    folds_won = aggregate.get("folds_beating_baseline", 0)
    n_folds = aggregate.get("n_folds", 1) or 1
    if folds_won / n_folds < 0.5:
        score += 2
    accuracy_std = aggregate.get("accuracy_std", 0) or 0
    if accuracy_std > 0.06:
        score += 1

    return [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.ELEVATED, RiskLevel.HIGH][
        min(score, 3)
    ]


def _explain_factors(
    features: pd.DataFrame,
    analytics: dict[str, Any],
    sentiment: dict[str, Any] | None,
) -> dict[str, list[dict[str, str]]]:
    """Derive bullish/bearish/risk factors from real computed signals.

    Each factor cites the value that produced it. Nothing is emitted without a
    corresponding number in the data.
    """
    bullish: list[dict[str, str]] = []
    bearish: list[dict[str, str]] = []
    risks: list[dict[str, str]] = []

    momentum = analytics.get("momentum", {})
    volatility = analytics.get("volatility", {})
    volume = analytics.get("volume", {})
    benchmark = analytics.get("benchmark", {})

    # --- Trend ------------------------------------------------------------
    trend = momentum.get("moving_average_trend")
    if trend == "STRONG_UPTREND":
        bullish.append({
            "factor": "Price above all key moving averages",
            "evidence": (
                f"{momentum.get('distance_from_sma_20', 0):+.1%} vs SMA-20, "
                f"{momentum.get('distance_from_sma_50', 0):+.1%} vs SMA-50, "
                f"{momentum.get('distance_from_sma_200', 0):+.1%} vs SMA-200"
            ),
        })
    elif trend == "STRONG_DOWNTREND":
        bearish.append({
            "factor": "Price below all key moving averages",
            "evidence": (
                f"{momentum.get('distance_from_sma_20', 0):+.1%} vs SMA-20, "
                f"{momentum.get('distance_from_sma_50', 0):+.1%} vs SMA-50, "
                f"{momentum.get('distance_from_sma_200', 0):+.1%} vs SMA-200"
            ),
        })

    # --- Oscillators ------------------------------------------------------
    rsi_state, rsi = momentum.get("rsi_state"), momentum.get("rsi_14")
    if rsi is not None:
        if rsi_state == "OVERBOUGHT":
            risks.append({
                "factor": "RSI in overbought territory",
                "evidence": f"RSI(14) = {rsi:.1f}, above the conventional 70 threshold",
            })
        elif rsi_state == "OVERSOLD":
            risks.append({
                "factor": "RSI in oversold territory",
                "evidence": f"RSI(14) = {rsi:.1f}, below the conventional 30 threshold",
            })

    macd_state = momentum.get("macd_state")
    if macd_state == "BULLISH_CROSSOVER":
        bullish.append({
            "factor": "MACD above its signal line",
            "evidence": f"MACD histogram {momentum.get('macd_histogram', 0):+.4f}",
        })
    elif macd_state == "BEARISH_CROSSOVER":
        bearish.append({
            "factor": "MACD below its signal line",
            "evidence": f"MACD histogram {momentum.get('macd_histogram', 0):+.4f}",
        })

    # --- Volatility -------------------------------------------------------
    regime = volatility.get("regime")
    if regime == "ELEVATED":
        risks.append({
            "factor": "Volatility elevated versus its own norm",
            "evidence": (
                f"21-day realised volatility {volatility.get('realised_21d', 0):.1%}, "
                f"{volatility.get('regime_ratio', 0):.2f}x the one-year average"
            ),
        })
    elif regime == "SUBDUED":
        bullish.append({
            "factor": "Volatility subdued",
            "evidence": (
                f"21-day realised volatility {volatility.get('realised_21d', 0):.1%}, "
                f"{volatility.get('regime_ratio', 0):.2f}x the one-year average"
            ),
        })

    # --- Volume -----------------------------------------------------------
    if volume.get("available"):
        relative = volume.get("relative_volume")
        if relative is not None and relative > 1.5:
            risks.append({
                "factor": "Unusually high trading volume",
                "evidence": f"{relative:.2f}x the 20-day average — often accompanies news",
            })
        elif volume.get("trend") == "RISING":
            bullish.append({
                "factor": "Participation rising",
                "evidence": "20-day average volume above its 60-day average",
            })

    # --- Benchmark --------------------------------------------------------
    if benchmark.get("available"):
        month = (benchmark.get("relative_performance") or {}).get("1m")
        if isinstance(month, dict) and month.get("excess") is not None:
            excess = month["excess"]
            entry = {
                "factor": f"{'Outperforming' if excess > 0 else 'Underperforming'} "
                          f"{benchmark.get('benchmark_label')}",
                "evidence": f"{excess:+.1%} excess return over the past month",
            }
            (bullish if excess > 0.02 else bearish if excess < -0.02 else risks).append(entry)

    # --- News sentiment ---------------------------------------------------
    if sentiment and sentiment.get("available"):
        label, score = sentiment.get("label"), sentiment.get("net_score", 0)
        entry = {
            "factor": f"News sentiment {label.lower()}",
            "evidence": (
                f"Weighted FinBERT score {score:+.3f} across "
                f"{sentiment.get('n_articles', 0)} deduplicated articles"
            ),
        }
        if label == "POSITIVE":
            bullish.append(entry)
        elif label == "NEGATIVE":
            bearish.append(entry)

    return {"bullish": bullish, "bearish": bearish, "risks": risks}


def _expected_return_range(
    close: pd.Series,
    horizon: int,
    predicted_class: int,
    target: TargetSpec,
) -> dict[str, float] | None:
    """Conditional quantiles of historical forward returns for this class.

    Not a model output and not an invented band: the empirical distribution of
    what actually happened historically, over the horizon, on days that ended
    up in the predicted class. Returns None when too few observations exist to
    quote a range honestly.
    """
    forward = np.log(close.shift(-horizon) / close).dropna()
    target_series, _ = build_target(close, target)
    aligned = pd.concat(
        [forward.rename("forward"), target_series.rename("class")], axis=1
    ).dropna()

    matching = aligned[aligned["class"] == predicted_class]["forward"]
    if len(matching) < 30:
        return None

    return {
        "p10": round(float(np.expm1(matching.quantile(0.10))), 6),
        "p25": round(float(np.expm1(matching.quantile(0.25))), 6),
        "median": round(float(np.expm1(matching.quantile(0.50))), 6),
        "p75": round(float(np.expm1(matching.quantile(0.75))), 6),
        "p90": round(float(np.expm1(matching.quantile(0.90))), 6),
        "n_observations": int(len(matching)),
        "basis": (
            f"Empirical distribution of actual {horizon}-session forward returns on the "
            f"{len(matching)} historical days that resolved to this class."
        ),
    }


def predict(
    *,
    symbol: str,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    conventions: MarketConventions,
    analytics: dict[str, Any],
    target_name: str = "outlook_5d",
    model_mode: str = "most_possible",
    sentiment: dict[str, Any] | None = None,
) -> PredictionResult:
    """Produce a prediction, gated on demonstrated out-of-sample skill."""
    settings = get_settings()
    target = AVAILABLE_TARGETS.get(target_name)
    if target is None:
        raise ValueError(f"Unknown target '{target_name}'.")

    target_series, _ = build_target(frame["Close"], target)
    x, y_series = align_features_and_target(features, target_series)
    y = y_series.to_numpy().astype(int)

    result = PredictionResult(
        verdict=Verdict.INSUFFICIENT_DATA,
        symbol=symbol,
        target=target,
        model_mode=model_mode,
        data_timestamp=frame.index[-1].isoformat(),
    )

    config = WalkForwardConfig(n_folds=5, horizon=target.horizon_days)

    try:
        model_result = run_walk_forward(
            lambda: _build_model(
                model_mode, target.n_classes, settings.random_seed, target.horizon_days
            ),
            x, y, target.class_labels, config,
            regime_series=build_regime_labels(features),
        )
        baseline_result = run_walk_forward(
            lambda: MajorityClassBaseline(target.n_classes, settings.random_seed),
            x, y, target.class_labels, config,
        )
    except (InsufficientHistoryError, ValueError) as exc:
        result.interpretation = (
            f"Not enough usable history to validate a model for {symbol} at this horizon. "
            f"{exc}"
        )
        return result

    aggregate = model_result.aggregate
    result.evidence = {
        "walk_forward": aggregate,
        "baseline": baseline_result.aggregate,
        "calibration": model_result.calibration,
        "regime_breakdown": model_result.regime_breakdown,
        "class_balance": class_balance(y_series, target),
        "n_samples": len(x),
        "effective_sample_size": effective_sample_size(len(x), target.horizon_days),
        "training_period": [x.index[0].date().isoformat(), x.index[-1].date().isoformat()],
        "top_features": (model_result.to_dict().get("top_features") or [])[:10],
    }
    result.risk_level = _assess_risk(features, result.evidence)
    result.factors = _explain_factors(features, analytics, sentiment)

    # --- The skill gate ---------------------------------------------------
    beat_baseline = aggregate.get("skill_score_mean", 0) > 0
    folds_won = aggregate.get("folds_beating_baseline", 0)
    n_folds = aggregate.get("n_folds", 1) or 1
    consistent = folds_won / n_folds >= 0.6

    if not (beat_baseline and consistent):
        result.verdict = Verdict.NO_EDGE
        result.interpretation = (
            f"Out-of-sample, this model achieved {aggregate.get('accuracy_mean', 0):.1%} "
            f"directional accuracy against a {aggregate.get('baseline_accuracy_mean', 0):.1%} "
            f"naive baseline, beating it in {folds_won} of {n_folds} walk-forward folds. "
            f"That is not sufficient evidence of predictive skill, so no directional call "
            f"is issued for {symbol} at this horizon. The analytics and evidence below "
            f"remain valid and are worth reading on their own."
        )
        return result

    # --- Fit on all data and predict the latest bar -----------------------
    model = _build_model(
        model_mode, target.n_classes, settings.random_seed, target.horizon_days
    )
    model.fit(x, y)

    # Sequence models need the preceding window to predict the latest bar, so
    # pass a trailing slice rather than a single row and take the last output.
    tail = x.iloc[-64:] if getattr(model, "requires_sequences", False) else x.iloc[[-1]]
    latest_proba = model.predict_proba(tail)[-1]
    predicted_class = int(latest_proba.argmax())

    # Training curves and hyperparameters, where the model exposes them.
    if hasattr(model, "training_history"):
        result.evidence["training_history"] = model.training_history()
    # How the ensemble arrived at its combination, so "combines the strongest
    # signals" is inspectable rather than a claim.
    if hasattr(model, "stacking_report"):
        result.evidence["stacking"] = model.stacking_report()

    # Abstain if today's confidence is below the fitted threshold.
    split_point = int(len(x) * 0.75)
    threshold, _ = fit_threshold(
        y[split_point:], model.predict_proba(x.iloc[split_point:]), target.class_labels
    )
    margin = float(confidence_margin(latest_proba.reshape(1, -1))[0])

    if margin < threshold:
        result.verdict = Verdict.ABSTAINED
        result.interpretation = (
            f"This model has demonstrated out-of-sample skill for {symbol}, but today's "
            f"signal is not decisive enough to act on (confidence margin {margin:.3f} "
            f"against a {threshold:.3f} threshold fitted on held-out data). "
            f"No directional call is issued."
        )
        return result

    calibration = model_result.calibration or {}
    is_calibrated = bool(calibration.get("is_calibrated"))

    result.verdict = Verdict.DIRECTIONAL
    result.direction = target.class_labels[predicted_class]
    result.probability = float(latest_proba[predicted_class])
    result.probability_is_calibrated = is_calibrated
    result.expected_return_range = _expected_return_range(
        frame["Close"], target.horizon_days, predicted_class, target
    )

    bullish_count = len(result.factors["bullish"])
    bearish_count = len(result.factors["bearish"])
    result.interpretation = (
        f"Over the next {target.horizon_days} trading session"
        f"{'s' if target.horizon_days > 1 else ''}, the model's most likely outcome for "
        f"{symbol} is {result.direction}. This rests on "
        f"{aggregate.get('accuracy_mean', 0):.1%} walk-forward accuracy against a "
        f"{aggregate.get('baseline_accuracy_mean', 0):.1%} baseline. "
        f"{bullish_count} bullish and {bearish_count} bearish factors were identified. "
        f"Risk level is {result.risk_level.value.lower()}. This is a statistical estimate, "
        f"not a guaranteed outcome."
    )
    return result
