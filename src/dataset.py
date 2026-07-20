"""Scaling, sliding-window sequence generation, and time-series-aware splitting.

Turns the feature-engineered DataFrame from ``src.preprocessing`` into
model-ready NumPy tensors, while carefully avoiding the two most common
sources of look-ahead leakage in time-series ML:

1. **Scaler leakage** — ``MinMaxScaler`` is fit only on the training split,
   never on the full dataset, so test-set statistics never influence scaling.
2. **Split leakage** — the train/test split is chronological (no shuffling),
   so no test-period row is ever used to predict a train-period target.

To avoid throwing away the first ``window_size`` test samples (which would
otherwise lack enough preceding context), the tail of the *training* period
is used as context for the first few test windows. This is not leakage:
that context is genuinely-past, already-observed data relative to every
test target it is paired with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils import InsufficientDataError, get_logger

logger = get_logger(__name__)

DEFAULT_WINDOW_SIZE = 60
DEFAULT_TEST_SIZE = 0.2
DEFAULT_TARGET_COLUMN = "Close"


@dataclass
class PreparedDataset:
    """Bundle of everything a training/evaluation/prediction pipeline needs."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    feature_columns: list[str]
    target_column: str
    target_index: int
    window_size: int

    @property
    def n_features(self) -> int:
        """Number of input features per timestep."""
        return len(self.feature_columns)


# --------------------------------------------------------------------------- #
# Chronological split
# --------------------------------------------------------------------------- #


def time_series_train_test_split(
    df: pd.DataFrame, test_size: float = DEFAULT_TEST_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame chronologically (no shuffling) into train/test.

    Args:
        df: Feature-engineered DataFrame, sorted ascending by date.
        test_size: Fraction of rows (from the end) reserved for testing.

    Returns:
        ``(train_df, test_df)`` where every ``train_df`` row precedes every
        ``test_df`` row in time.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info(
        "Chronological split: %d train rows (%s to %s), %d test rows (%s to %s)",
        len(train_df), train_df.index.min(), train_df.index.max(),
        len(test_df), test_df.index.min(), test_df.index.max(),
    )
    return train_df, test_df


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #


def fit_scalers(
    train_df: pd.DataFrame, feature_columns: list[str], target_column: str
) -> tuple[MinMaxScaler, MinMaxScaler]:
    """Fit a feature scaler and a target scaler, both on the training split only.

    A separate single-column scaler is kept for the target so predictions
    can be inverse-transformed back to raw price without needing the full
    feature vector.

    Args:
        train_df: The training-period DataFrame (must not include test rows).
        feature_columns: Columns to scale as model inputs.
        target_column: Column to scale as the prediction target.

    Returns:
        ``(feature_scaler, target_scaler)``, both already fit.
    """
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    feature_scaler.fit(train_df[feature_columns])
    target_scaler.fit(train_df[[target_column]])

    return feature_scaler, target_scaler


# --------------------------------------------------------------------------- #
# Sliding-window sequence generation
# --------------------------------------------------------------------------- #


def create_sequences(
    features: np.ndarray, target: np.ndarray, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences from scaled feature/target arrays.

    For each valid position ``i``, the input window is
    ``features[i : i + window_size]`` and the label is
    ``target[i + window_size]`` — i.e. "given the last ``window_size`` days,
    predict the very next day."

    Args:
        features: Scaled feature array of shape ``(n_samples, n_features)``.
        target: Scaled target array of shape ``(n_samples,)`` or ``(n_samples, 1)``.
        window_size: Number of past timesteps per input sequence.

    Returns:
        ``(X, y)`` where ``X`` has shape ``(n_sequences, window_size, n_features)``
        and ``y`` has shape ``(n_sequences,)``.
    """
    target = target.reshape(-1)
    n_sequences = len(features) - window_size

    if n_sequences <= 0:
        raise InsufficientDataError(
            f"Cannot build sequences: {len(features)} rows is not enough for "
            f"window_size={window_size}. Need at least {window_size + 1} rows."
        )

    X = np.empty((n_sequences, window_size, features.shape[1]), dtype=np.float32)
    y = np.empty((n_sequences,), dtype=np.float32)

    for i in range(n_sequences):
        X[i] = features[i : i + window_size]
        y[i] = target[i + window_size]

    return X, y


def get_last_window(
    df: pd.DataFrame,
    feature_scaler: MinMaxScaler,
    feature_columns: list[str],
    window_size: int,
) -> np.ndarray:
    """Build the single most recent scaled window, ready for ``model.predict``.

    Shared by ``src.predict`` so the exact same scaling/windowing logic used
    at training time is reused at inference time (no duplicated logic that
    could silently drift out of sync).

    Args:
        df: Full feature-engineered DataFrame (chronologically sorted).
        feature_scaler: A scaler already fit on training data.
        feature_columns: Columns to include, in the same order used at fit time.
        window_size: Number of trailing rows to include in the window.

    Returns:
        Array of shape ``(1, window_size, n_features)``.
    """
    if len(df) < window_size:
        raise InsufficientDataError(
            f"Need at least {window_size} rows to build a prediction window, got {len(df)}."
        )

    recent = df[feature_columns].iloc[-window_size:]
    scaled = feature_scaler.transform(recent)
    return scaled.reshape(1, window_size, len(feature_columns)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def prepare_dataset(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    window_size: int = DEFAULT_WINDOW_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    feature_columns: list[str] | None = None,
) -> PreparedDataset:
    """End-to-end dataset preparation: split -> scale -> window.

    Args:
        df: Feature-engineered, leakage-free DataFrame from
            ``src.preprocessing.load_and_prepare_data``.
        target_column: Column to predict (default ``"Close"``).
        window_size: Number of past days used to predict the next day.
        test_size: Fraction of the data (chronologically last) held out for testing.
        feature_columns: Columns to use as model inputs. Defaults to every
            column in ``df`` (the target column is normally included as an
            input too, since past close prices are informative of future ones).

    Returns:
        A fully-populated ``PreparedDataset``.

    Raises:
        InsufficientDataError: If there is not enough data to build at least
            one training and one test sequence for the requested window size.
    """
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in DataFrame columns.")

    feature_columns = feature_columns or list(df.columns)
    if target_column not in feature_columns:
        feature_columns = [*feature_columns, target_column]

    min_rows_needed = window_size * 2 + 1  # enough for a meaningful train and test split
    if len(df) < min_rows_needed:
        raise InsufficientDataError(
            f"Need at least {min_rows_needed} rows for window_size={window_size} "
            f"(got {len(df)}). Try a longer history or a smaller window_size."
        )

    train_df, test_df = time_series_train_test_split(df, test_size=test_size)

    if len(train_df) <= window_size:
        raise InsufficientDataError(
            f"Training split has only {len(train_df)} rows, which is not more "
            f"than window_size={window_size}. Reduce window_size, reduce "
            "test_size, or download more history."
        )

    feature_scaler, target_scaler = fit_scalers(train_df, feature_columns, target_column)
    target_index = feature_columns.index(target_column)

    train_features_scaled = feature_scaler.transform(train_df[feature_columns])
    test_features_scaled = feature_scaler.transform(test_df[feature_columns])
    train_target_scaled = target_scaler.transform(train_df[[target_column]])
    test_target_scaled = target_scaler.transform(test_df[[target_column]])

    X_train, y_train = create_sequences(train_features_scaled, train_target_scaled, window_size)
    train_dates = train_df.index[window_size:]

    # Prepend the tail of the training period so every test-set day still
    # yields a prediction, without ever using data from *after* that day.
    combined_features = np.concatenate(
        [train_features_scaled[-window_size:], test_features_scaled], axis=0
    )
    combined_target = np.concatenate(
        [train_target_scaled[-window_size:], test_target_scaled], axis=0
    )
    X_test, y_test = create_sequences(combined_features, combined_target, window_size)
    test_dates = test_df.index

    logger.info(
        "Prepared dataset: X_train=%s, X_test=%s, %d features, window_size=%d, target=%r",
        X_train.shape, X_test.shape, len(feature_columns), window_size, target_column,
    )

    return PreparedDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_dates=train_dates,
        test_dates=test_dates,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_columns=feature_columns,
        target_column=target_column,
        target_index=target_index,
        window_size=window_size,
    )


def inverse_transform_target(scaler: MinMaxScaler, scaled_values: np.ndarray) -> np.ndarray:
    """Inverse-transform a 1-D array of scaled target values back to raw price scale.

    Args:
        scaler: The (already-fit) target scaler from ``PreparedDataset.target_scaler``.
        scaled_values: 1-D array of values in the scaler's ``feature_range``.

    Returns:
        1-D array of values in the original price scale.
    """
    return scaler.inverse_transform(scaled_values.reshape(-1, 1)).reshape(-1)
