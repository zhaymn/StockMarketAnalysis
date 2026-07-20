"""Inference: load a trained model and forecast future closing prices.

Two forecasting modes are provided:

- **Next-day prediction** — a single step, using a real historical window.
  This is the model's most reliable output, since every input feature is
  genuine market data.
- **Recursive multi-day forecasting** — the model's own prediction for day
  t+1 is fed back in as (approximated) input for predicting day t+2, and so
  on out to a user-defined horizon. Because only the closing price is
  actually predicted, the recursive step approximates the day's Open/High/
  Low/Adj Close as equal to the predicted Close (a "flat bar") and Volume as
  the trailing 20-day average — see ``forecast_future`` for details. This is
  a standard simplification in multi-step stock forecasting, but it means
  uncertainty compounds with horizon length: treat day 1 of a forecast as
  far more reliable than day 30.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import keras
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.dataset import get_last_window, inverse_transform_target
from src.preprocessing import RAW_COLUMNS, engineer_features
from src.utils import (
    BASE_DIR,
    MODELS_DIR,
    ModelNotFoundError,
    get_logger,
    load_artifact,
    validate_ticker,
)

logger = get_logger(__name__)

#: Forecasts beyond this many days rely so heavily on the flat-bar/avg-volume
#: approximation (compounded that many times) that we log a warning.
LONG_HORIZON_WARNING_THRESHOLD = 60


@dataclass
class LoadedModelBundle:
    """A trained model plus everything needed to feed it and interpret its output."""

    model: keras.Model
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    feature_columns: list[str]
    target_column: str
    window_size: int
    ticker: str
    model_name: str


# --------------------------------------------------------------------------- #
# Loading a trained model from disk
# --------------------------------------------------------------------------- #


def load_model_bundle(ticker: str | None = None, model_name: str = "lstm") -> LoadedModelBundle:
    """Load a trained model, its scalers, and its metadata from ``models/`` (or the project root).

    Args:
        ticker: Which ticker's model to load, e.g. ``"AAPL"``. If ``None``,
            loads the project-root default (``model.keras``, written by
            ``src.train.train_and_save`` with ``set_as_default=True``).
        model_name: Architecture name used to namespace the saved run
            (matches ``src.model.MODEL_REGISTRY`` keys, e.g. ``"lstm"``).
            Ignored when ``ticker`` is ``None``.

    Returns:
        A ``LoadedModelBundle`` ready to pass to ``predict_next_day`` or ``forecast_future``.

    Raises:
        ModelNotFoundError: If no trained model exists at the resolved paths.
    """
    if ticker is None:
        model_path = BASE_DIR / "model.keras"
        scaler_path = BASE_DIR / "model_scalers.joblib"
        metadata_path = BASE_DIR / "model_metadata.json"
    else:
        ticker = validate_ticker(ticker)
        run_id = f"{ticker}_{model_name}"
        model_path = MODELS_DIR / f"{run_id}.keras"
        scaler_path = MODELS_DIR / f"{run_id}_scalers.joblib"
        metadata_path = MODELS_DIR / f"{run_id}_metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        raise ModelNotFoundError(
            f"No trained model found at {model_path}. Train one first via "
            "src.train.train_and_save."
        )

    model = keras.models.load_model(model_path)
    scalers = load_artifact(scaler_path)
    metadata = json.loads(metadata_path.read_text())

    logger.info("Loaded model bundle for %s (%s), window_size=%d", metadata["ticker"], metadata["model_name"], metadata["window_size"])

    return LoadedModelBundle(
        model=model,
        feature_scaler=scalers["feature_scaler"],
        target_scaler=scalers["target_scaler"],
        feature_columns=metadata["feature_columns"],
        target_column=metadata["target_column"],
        window_size=metadata["window_size"],
        ticker=metadata["ticker"],
        model_name=metadata["model_name"],
    )


# --------------------------------------------------------------------------- #
# Next-day prediction
# --------------------------------------------------------------------------- #


def predict_next_day(
    df_featured: pd.DataFrame,
    model: keras.Model,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    feature_columns: list[str],
    window_size: int,
) -> tuple[pd.Timestamp, float]:
    """Predict the very next trading day's closing price from real historical data.

    Args:
        df_featured: Feature-engineered DataFrame (from
            ``src.preprocessing.load_and_prepare_data``), ending on the most
            recent available trading day.
        model: A trained Keras model.
        feature_scaler: The scaler fit on this model's training data.
        target_scaler: The target scaler, for inverse-transforming the output.
        feature_columns: Columns to feed the model, in training order.
        window_size: Number of trailing days the model expects as input.

    Returns:
        ``(predicted_date, predicted_close)``.
    """
    window = get_last_window(df_featured, feature_scaler, feature_columns, window_size)
    scaled_pred = model.predict(window, verbose=0)
    predicted_close = float(inverse_transform_target(target_scaler, scaled_pred.reshape(-1))[0])
    predicted_date = _next_business_day(df_featured.index[-1])

    logger.info("Predicted next close for %s: %.2f", predicted_date.date(), predicted_close)
    return predicted_date, predicted_close


# --------------------------------------------------------------------------- #
# Recursive multi-day forecasting
# --------------------------------------------------------------------------- #


def _next_business_day(from_date: pd.Timestamp) -> pd.Timestamp:
    """The next weekday after ``from_date`` (skips Sat/Sun; does not account for market holidays)."""
    return pd.Timestamp(from_date) + pd.offsets.BDay(1)


def forecast_future(
    df_featured: pd.DataFrame,
    model: keras.Model,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    feature_columns: list[str],
    window_size: int,
    horizon: int,
) -> pd.DataFrame:
    """Recursively forecast closing prices for ``horizon`` future trading days.

    At each step: predict the next close from the current window, then
    approximate that day's full OHLCV row (Open = High = Low = Adj Close =
    predicted Close; Volume = trailing 20-day average volume) and append it
    to the history before re-deriving technical indicators and predicting
    the next step. See the module docstring for why this approximation is
    necessary and what it implies about forecast reliability at longer
    horizons.

    Args:
        df_featured: Feature-engineered DataFrame ending on the most recent
            real trading day.
        model: A trained Keras model.
        feature_scaler: The scaler fit on this model's training data.
        target_scaler: The target scaler, for inverse-transforming outputs.
        feature_columns: Columns to feed the model, in training order.
        window_size: Number of trailing days the model expects as input.
        horizon: Number of future trading days to forecast (>= 1).

    Returns:
        A DataFrame indexed by predicted date with a single ``Predicted_Close`` column.

    Raises:
        ValueError: If ``horizon`` is not a positive integer.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if horizon > LONG_HORIZON_WARNING_THRESHOLD:
        logger.warning(
            "Forecasting %d days ahead: predictions this far out compound the "
            "flat-bar/avg-volume approximation %d times and should be treated "
            "as indicative trend, not precise price targets.", horizon, horizon,
        )

    raw_history = df_featured[RAW_COLUMNS].copy()
    predicted_dates: list[pd.Timestamp] = []
    predicted_closes: list[float] = []

    for _ in range(horizon):
        featured = engineer_features(raw_history)
        window = get_last_window(featured, feature_scaler, feature_columns, window_size)
        scaled_pred = model.predict(window, verbose=0)
        predicted_close = float(inverse_transform_target(target_scaler, scaled_pred.reshape(-1))[0])

        next_date = _next_business_day(raw_history.index[-1])
        avg_volume = raw_history["Volume"].tail(20).mean()

        new_row = pd.DataFrame(
            {
                "Open": [predicted_close],
                "High": [predicted_close],
                "Low": [predicted_close],
                "Close": [predicted_close],
                "Adj Close": [predicted_close],
                "Volume": [avg_volume],
            },
            index=[next_date],
        )
        raw_history = pd.concat([raw_history, new_row])

        predicted_dates.append(next_date)
        predicted_closes.append(predicted_close)

    result = pd.DataFrame({"Predicted_Close": predicted_closes}, index=pd.DatetimeIndex(predicted_dates, name="Date"))
    logger.info(
        "Forecasted %d trading days ahead: %s -> %.2f ... %s -> %.2f",
        horizon, result.index[0].date(), result["Predicted_Close"].iloc[0],
        result.index[-1].date(), result["Predicted_Close"].iloc[-1],
    )
    return result


# --------------------------------------------------------------------------- #
# Bundle-based convenience wrappers (what app.py will typically call)
# --------------------------------------------------------------------------- #


def predict_next_close(df_featured: pd.DataFrame, bundle: LoadedModelBundle) -> tuple[pd.Timestamp, float]:
    """``predict_next_day`` using a ``LoadedModelBundle`` instead of separate arguments."""
    return predict_next_day(
        df_featured, bundle.model, bundle.feature_scaler, bundle.target_scaler,
        bundle.feature_columns, bundle.window_size,
    )


def forecast(df_featured: pd.DataFrame, bundle: LoadedModelBundle, horizon: int) -> pd.DataFrame:
    """``forecast_future`` using a ``LoadedModelBundle`` instead of separate arguments."""
    return forecast_future(
        df_featured, bundle.model, bundle.feature_scaler, bundle.target_scaler,
        bundle.feature_columns, bundle.window_size, horizon,
    )
