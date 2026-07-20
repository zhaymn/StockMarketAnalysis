"""Regression evaluation metrics for stock closing-price forecasts.

All functions here operate on raw (already inverse-transformed) price
values, not scaled ``[0, 1]`` model outputs — callers are expected to run
predictions through ``src.dataset.inverse_transform_target`` first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import get_logger

logger = get_logger(__name__)

#: Plain-English explanation of each metric, meant for display in the
#: Streamlit dashboard (e.g. as a caption or tooltip next to each number).
METRIC_EXPLANATIONS: dict[str, str] = {
    "mae": (
        "Mean Absolute Error - the average absolute gap between predicted and actual "
        "closing prices, in the stock's own currency. Lower is better, and it's directly "
        "interpretable: 'the model is off by about $X on a typical day.'"
    ),
    "mse": (
        "Mean Squared Error - the average of the squared prediction errors. Squaring makes "
        "large misses count disproportionately more than small ones, so MSE is sensitive to "
        "outlier errors. Its units are squared currency, which makes it hard to read directly "
        "(RMSE below is the more interpretable version)."
    ),
    "rmse": (
        "Root Mean Squared Error - the square root of MSE, back in raw currency units. "
        "Like MSE, it penalizes large errors more than MAE does, so RMSE noticeably larger "
        "than MAE indicates a few bad misses rather than uniformly small errors."
    ),
    "mape": (
        "Mean Absolute Percentage Error - the average error as a percentage of the actual "
        "price. Being scale-independent, it's the metric to use when comparing forecast "
        "quality across tickers at very different price levels (e.g. a $10 stock vs. a "
        "$1,000 stock)."
    ),
    "r2": (
        "R-squared (coefficient of determination) - the fraction of variance in actual prices "
        "that the model's predictions explain. 1.0 is a perfect fit, 0.0 is no better than "
        "always predicting the mean price, and negative values are worse than that trivial "
        "baseline."
    ),
    "directional_accuracy": (
        "Directional Accuracy - the percentage of days the model correctly predicted whether "
        "the price would rise or fall relative to the previous actual close. For trading "
        "decisions this often matters more than raw price error, since a strategy only needs "
        "the direction right to be profitable."
    ),
}


@dataclass
class EvaluationMetrics:
    """A bundle of regression metrics computed on one set of predictions."""

    mae: float
    mse: float
    rmse: float
    mape: float
    r2: float
    directional_accuracy: float
    n_samples: int

    def to_dict(self) -> dict[str, float]:
        """Return the metrics as a plain dict (e.g. for JSON export or a Streamlit table)."""
        return asdict(self)

    def summary(self) -> str:
        """Render a human-readable, aligned report with each metric explained."""
        lines = ["Evaluation Report", "=" * 60]
        for key, value in self.to_dict().items():
            if key == "n_samples":
                continue
            unit = "%" if key in ("mape", "directional_accuracy") else ""
            lines.append(f"{key.upper():22s}: {value:>10.4f}{unit}")
            if key in METRIC_EXPLANATIONS:
                lines.append(f"  -> {METRIC_EXPLANATIONS[key]}")
        lines.append(f"\nComputed over {self.n_samples} predictions.")
        return "\n".join(lines)


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate metrics on empty arrays.")
    return y_true, y_pred


# --------------------------------------------------------------------------- #
# Individual metrics
# --------------------------------------------------------------------------- #


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE = mean(|y_true - y_pred| / |y_true|) * 100.

    A tiny epsilon guards against division by zero; stock closing prices are
    never actually zero, but this keeps the function safe on arbitrary input.
    """
    epsilon = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of days the predicted direction of movement matched the actual direction.

    Both directions are measured relative to the *previous actual* close
    (not the previous prediction), since each prediction in the test set is
    made from a real historical window rather than a chain of prior
    predictions:

        actual_direction[t]    = sign(y_true[t] - y_true[t-1])
        predicted_direction[t] = sign(y_pred[t] - y_true[t-1])
        directional_accuracy   = mean(actual_direction == predicted_direction) * 100

    Requires at least 2 samples; returns ``NaN`` otherwise (undefined).
    """
    if len(y_true) < 2:
        logger.warning("directional_accuracy requires >= 2 samples; got %d.", len(y_true))
        return float("nan")

    actual_direction = np.sign(y_true[1:] - y_true[:-1])
    predicted_direction = np.sign(y_pred[1:] - y_true[:-1])
    return float(np.mean(actual_direction == predicted_direction) * 100)


# --------------------------------------------------------------------------- #
# Combined evaluation
# --------------------------------------------------------------------------- #


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationMetrics:
    """Compute MAE, MSE, RMSE, MAPE, R^2, and directional accuracy in one call.

    Args:
        y_true: Actual closing prices, raw scale, shape ``(n,)``.
        y_pred: Predicted closing prices, raw scale, shape ``(n,)``.

    Returns:
        An ``EvaluationMetrics`` bundle.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else float("nan")
    dir_acc = directional_accuracy(y_true, y_pred)

    metrics = EvaluationMetrics(
        mae=float(mae), mse=float(mse), rmse=rmse, mape=mape,
        r2=float(r2), directional_accuracy=dir_acc, n_samples=len(y_true),
    )
    logger.info(
        "Evaluated %d predictions: MAE=%.4f RMSE=%.4f MAPE=%.2f%% R2=%.4f DirAcc=%.2f%%",
        metrics.n_samples, metrics.mae, metrics.rmse, metrics.mape, metrics.r2,
        metrics.directional_accuracy,
    )
    return metrics


def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Residuals = actual - predicted. Feeds the residual plot and error histogram.

    Positive residuals mean the model under-predicted; negative means it over-predicted.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return y_true - y_pred


# --------------------------------------------------------------------------- #
# Naive persistence baseline, for context
# --------------------------------------------------------------------------- #


def naive_baseline_predictions(y_true: np.ndarray) -> np.ndarray:
    """A persistence ("no change") baseline: predict tomorrow's close as today's actual close.

    Aligned against ``y_true[1:]`` (there is no naive prediction for the
    very first sample, since it has no preceding actual value).
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if len(y_true) < 2:
        raise ValueError("naive_baseline_predictions requires at least 2 samples.")
    return y_true[:-1]


def evaluate_against_naive_baseline(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[EvaluationMetrics, EvaluationMetrics]:
    """Evaluate the model and a persistence baseline on the same aligned window.

    Stock prices are close to a random walk, so a trivial "predict no change"
    baseline is a surprisingly strong competitor. Comparing against it is
    the honest way to show a model is adding real value rather than just
    tracking yesterday's price.

    Args:
        y_true: Actual closing prices, raw scale, shape ``(n,)``, n >= 2.
        y_pred: Model-predicted closing prices, raw scale, shape ``(n,)``.

    Returns:
        ``(model_metrics, naive_baseline_metrics)``, both computed over the
        same ``y_true[1:]`` targets for a fair comparison.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    if len(y_true) < 2:
        raise ValueError("evaluate_against_naive_baseline requires at least 2 samples.")

    model_metrics = evaluate_predictions(y_true[1:], y_pred[1:])
    naive_preds = naive_baseline_predictions(y_true)
    naive_metrics = evaluate_predictions(y_true[1:], naive_preds)

    logger.info(
        "Model vs. naive baseline -- RMSE: %.4f vs %.4f | DirAcc: %.2f%% vs %.2f%%",
        model_metrics.rmse, naive_metrics.rmse,
        model_metrics.directional_accuracy, naive_metrics.directional_accuracy,
    )
    return model_metrics, naive_metrics
