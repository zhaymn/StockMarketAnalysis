"""Training loop: callbacks, checkpointing, TensorBoard, and artifact persistence.

``train_model`` runs a single ``model.fit`` call with a standard callback
stack (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard).
``train_and_save`` wraps that with everything a separate inference process
(``src.predict``, ``app.py``) needs afterward: the best checkpoint reloaded
from disk, the fitted scalers, and a metadata JSON describing exactly how
the model's inputs were constructed (feature columns, window size, etc.).
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import keras

from src.dataset import PreparedDataset
from src.utils import (
    BASE_DIR,
    LOGS_DIR,
    MODELS_DIR,
    DEFAULT_SEED,
    get_logger,
    save_artifact,
    set_global_seed,
    timer,
    validate_ticker,
)

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters and settings for the ``model.fit`` call."""

    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.1
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    use_tensorboard: bool = True
    use_csv_logger: bool = True
    verbose: int = 1
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if not 0 < self.validation_split < 1:
            raise ValueError(f"validation_split must be in (0, 1), got {self.validation_split}")


@dataclass
class TrainingResult:
    """Everything downstream code (evaluation, prediction, the dashboard) needs."""

    model: keras.Model
    history: dict[str, list[float]]
    model_path: Path
    scaler_path: Path
    metadata_path: Path
    history_path: Path
    best_val_loss: float
    trained_epochs: int
    training_seconds: float


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #


def build_callbacks(
    checkpoint_path: Path,
    config: TrainingConfig,
    tensorboard_log_dir: Path | None = None,
    csv_log_path: Path | None = None,
    extra_callbacks: list[keras.callbacks.Callback] | None = None,
) -> list[keras.callbacks.Callback]:
    """Assemble the standard training callback stack.

    Args:
        checkpoint_path: Where ``ModelCheckpoint`` saves the best model (by ``val_loss``).
        config: Training hyperparameters controlling patience/factors below.
        tensorboard_log_dir: If given (and ``config.use_tensorboard``), enables TensorBoard logging.
        csv_log_path: If given (and ``config.use_csv_logger``), appends per-epoch metrics to this CSV.
        extra_callbacks: Additional callbacks to append, e.g. a Streamlit progress-bar
            callback injected by ``app.py`` — keeps this module UI-agnostic.

    Returns:
        The list of callbacks to pass to ``model.fit``.
    """
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_lr,
            verbose=1,
        ),
    ]

    if config.use_tensorboard and tensorboard_log_dir is not None:
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(keras.callbacks.TensorBoard(log_dir=str(tensorboard_log_dir), histogram_freq=1))

    if config.use_csv_logger and csv_log_path is not None:
        csv_log_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(keras.callbacks.CSVLogger(str(csv_log_path), append=False))

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    return callbacks


# --------------------------------------------------------------------------- #
# Core training
# --------------------------------------------------------------------------- #


@timer
def train_model(
    model: keras.Model,
    dataset: PreparedDataset,
    config: TrainingConfig | None = None,
    checkpoint_path: Path | None = None,
    tensorboard_log_dir: Path | None = None,
    csv_log_path: Path | None = None,
    extra_callbacks: list[keras.callbacks.Callback] | None = None,
) -> keras.callbacks.History:
    """Fit a compiled model on ``dataset.X_train``/``y_train``.

    ``validation_split`` is applied by Keras to the *trailing* slice of the
    training arrays (taken before any shuffling), which for chronologically
    ordered sequences is itself a chronological hold-out — so validation
    still never sees data from after its own targets, and never touches
    ``dataset.X_test`` at all.

    Args:
        model: A compiled ``keras.Model`` from ``src.model``.
        dataset: The ``PreparedDataset`` from ``src.dataset.prepare_dataset``.
        config: Training hyperparameters; defaults to ``TrainingConfig()``.
        checkpoint_path: Where to save the best-val-loss checkpoint.
            Defaults to ``models/checkpoint.keras``.
        tensorboard_log_dir: TensorBoard log directory (optional).
        csv_log_path: Per-epoch metrics CSV path (optional).
        extra_callbacks: Additional Keras callbacks to run alongside the standard stack.

    Returns:
        The Keras ``History`` object (``.history`` is a dict of per-epoch metric lists).
    """
    config = config or TrainingConfig()
    set_global_seed(config.seed)

    checkpoint_path = checkpoint_path or (MODELS_DIR / "checkpoint.keras")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = build_callbacks(
        checkpoint_path, config, tensorboard_log_dir, csv_log_path, extra_callbacks
    )

    logger.info(
        "Starting training: up to %d epochs, batch_size=%d, validation_split=%.2f, "
        "train_sequences=%d",
        config.epochs, config.batch_size, config.validation_split, len(dataset.X_train),
    )

    try:
        history = model.fit(
            dataset.X_train,
            dataset.y_train,
            validation_split=config.validation_split,
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=config.verbose,
        )
    except Exception:
        logger.exception("Training failed for model %r", model.name)
        raise

    final_epoch = len(history.history.get("loss", []))
    logger.info("Training finished after %d epochs (of %d max).", final_epoch, config.epochs)
    return history


# --------------------------------------------------------------------------- #
# Orchestration: train + persist everything needed for later inference
# --------------------------------------------------------------------------- #


def train_and_save(
    ticker: str,
    model: keras.Model,
    dataset: PreparedDataset,
    model_name: str = "lstm",
    config: TrainingConfig | None = None,
    extra_callbacks: list[keras.callbacks.Callback] | None = None,
    set_as_default: bool = True,
) -> TrainingResult:
    """Train a model and persist the model, scalers, metadata, and history to ``models/``.

    Args:
        ticker: Ticker symbol the model was trained on, e.g. ``"AAPL"``.
        model: A compiled ``keras.Model`` (from ``src.model.build_model_by_name`` or similar).
        dataset: The ``PreparedDataset`` used to train ``model``.
        model_name: Architecture name (matches ``src.model.MODEL_REGISTRY`` keys), used
            purely to namespace saved artifacts, e.g. ``"AAPL_lstm.keras"``.
        config: Training hyperparameters.
        extra_callbacks: Additional Keras callbacks.
        set_as_default: If True, also copies the trained model/scalers/metadata to
            project-root ``model.keras`` (and matching files) as the app's default model.

    Returns:
        A ``TrainingResult`` bundling the reloaded best model and all artifact paths.
    """
    ticker = validate_ticker(ticker)
    config = config or TrainingConfig()

    run_id = f"{ticker}_{model_name}"
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_path = MODELS_DIR / f"{run_id}.keras"
    tensorboard_log_dir = LOGS_DIR / "tensorboard" / f"{run_id}_{timestamp}"
    csv_log_path = LOGS_DIR / f"{run_id}_epochs.csv"

    history = train_model(
        model, dataset, config, checkpoint_path, tensorboard_log_dir, csv_log_path, extra_callbacks
    )

    # Reload from disk so the returned model is provably identical to what was persisted,
    # rather than trusting in-memory state after EarlyStopping's restore_best_weights.
    best_model = keras.models.load_model(checkpoint_path)

    scaler_path = MODELS_DIR / f"{run_id}_scalers.joblib"
    save_artifact(
        {"feature_scaler": dataset.feature_scaler, "target_scaler": dataset.target_scaler},
        scaler_path,
    )

    val_losses = history.history.get("val_loss", history.history["loss"])
    metadata = {
        "ticker": ticker,
        "model_name": model_name,
        "feature_columns": dataset.feature_columns,
        "target_column": dataset.target_column,
        "target_index": dataset.target_index,
        "window_size": dataset.window_size,
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "epochs_run": len(history.history["loss"]),
        "best_val_loss": float(min(val_losses)),
    }
    metadata_path = MODELS_DIR / f"{run_id}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    history_path = MODELS_DIR / f"{run_id}_history.json"
    history_path.write_text(json.dumps(history.history, indent=2))

    logger.info(
        "Saved model=%s scalers=%s metadata=%s history=%s",
        checkpoint_path, scaler_path, metadata_path, history_path,
    )

    if set_as_default:
        _copy_as_default(checkpoint_path, scaler_path, metadata_path)

    return TrainingResult(
        model=best_model,
        history=history.history,
        model_path=checkpoint_path,
        scaler_path=scaler_path,
        metadata_path=metadata_path,
        history_path=history_path,
        best_val_loss=metadata["best_val_loss"],
        trained_epochs=metadata["epochs_run"],
        training_seconds=0.0,  # populated by the @timer log line; not tracked per-call here
    )


def _copy_as_default(model_path: Path, scaler_path: Path, metadata_path: Path) -> None:
    """Copy a trained run's artifacts to project root as the app's default model.

    Produces ``model.keras``, ``model_scalers.joblib``, and ``model_metadata.json``
    at the project root, matching the top-level ``model.keras`` called out in the
    project layout — this is what ``app.py`` loads when no specific run is selected.
    """
    try:
        shutil.copy2(model_path, BASE_DIR / "model.keras")
        shutil.copy2(scaler_path, BASE_DIR / "model_scalers.joblib")
        shutil.copy2(metadata_path, BASE_DIR / "model_metadata.json")
        logger.info("Set %s as the default model.keras", model_path.name)
    except OSError as exc:
        logger.warning("Could not set default model artifacts: %s", exc)
