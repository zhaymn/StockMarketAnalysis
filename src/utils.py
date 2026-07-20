"""Shared utilities: paths, logging, reproducibility, and small helpers.

Every other module in ``src`` imports from here instead of re-implementing
path resolution, logger configuration, or seeding logic.
"""

from __future__ import annotations

import functools
import logging
import os
import random
import time
from datetime import date, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, TypeVar

import joblib
import numpy as np

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #

# src/utils.py -> src/ -> project root
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
LOGS_DIR: Path = BASE_DIR / "logs"
NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"


def ensure_project_dirs() -> None:
    """Create the standard project directories if they do not already exist."""
    for directory in (DATA_DIR, MODELS_DIR, LOGS_DIR, NOTEBOOKS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Custom exceptions
# --------------------------------------------------------------------------- #


class StockPredictorError(Exception):
    """Base exception for all application-specific errors."""


class DataDownloadError(StockPredictorError):
    """Raised when historical market data cannot be retrieved."""


class InsufficientDataError(StockPredictorError):
    """Raised when there is not enough data to build a valid sequence dataset."""


class ModelNotFoundError(StockPredictorError):
    """Raised when a saved model or scaler artifact cannot be located on disk."""


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

_CONFIGURED_LOGGERS: set[str] = set()


def get_logger(name: str, log_filename: str = "app.log", level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger that writes to both console and a rotating file.

    Args:
        name: Usually ``__name__`` of the calling module.
        log_filename: File under ``LOGS_DIR`` that this logger writes to.
        level: Minimum severity level to emit.

    Returns:
        A configured ``logging.Logger`` instance. Safe to call repeatedly;
        handlers are only attached once per logger name.
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOGS_DIR / log_filename, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # File logging is best-effort; console logging alone is still usable.
        logger.warning("Could not attach file handler at %s", LOGS_DIR / log_filename)

    _CONFIGURED_LOGGERS.add(name)
    return logger


logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

DEFAULT_SEED: int = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy, and TensorFlow RNGs for reproducible runs.

    TensorFlow is imported lazily so that modules which only need the
    lightweight helpers in this file are not forced to pay TensorFlow's
    import cost.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        logger.warning("TensorFlow not available; skipped TF seeding.")


# --------------------------------------------------------------------------- #
# Date / ticker helpers
# --------------------------------------------------------------------------- #

PERIOD_TO_YEARS: dict[str, int] = {
    "1 Year": 1,
    "3 Years": 3,
    "5 Years": 5,
    "10 Years": 10,
}


def get_date_range(years: int, end: date | None = None) -> tuple[str, str]:
    """Compute an ISO ``(start_date, end_date)`` string pair spanning N years.

    yfinance's ``period`` parameter only supports a fixed set of buckets
    (``1y``, ``2y``, ``5y``, ``10y``, ...) which does not include ``3y``.
    Using explicit start/end dates instead supports any horizon uniformly.

    Args:
        years: Number of years of history to span.
        end: End of the range; defaults to today.

    Returns:
        ``(start_date, end_date)`` formatted as ``YYYY-MM-DD``.
    """
    if years <= 0:
        raise ValueError(f"years must be positive, got {years}")

    end_date = end or date.today()
    start_date = end_date - timedelta(days=365 * years + years // 4)  # account for leap years
    return start_date.isoformat(), end_date.isoformat()


def validate_ticker(ticker: str) -> str:
    """Normalize a user-supplied ticker symbol (strip whitespace, upper-case).

    Args:
        ticker: Raw ticker input, e.g. ``" aapl "`` or ``"infy.ns"``.

    Returns:
        The normalized ticker, e.g. ``"AAPL"`` or ``"INFY.NS"``.

    Raises:
        ValueError: If the input is empty after stripping.
    """
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("Ticker symbol must not be empty.")
    return cleaned


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #


def save_artifact(obj: Any, path: Path) -> None:
    """Persist an arbitrary Python object (e.g. a fitted scaler) via joblib.

    Args:
        obj: The object to serialize.
        path: Destination file path; parent directories are created as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Saved artifact to %s", path)


def load_artifact(path: Path) -> Any:
    """Load a joblib-serialized object from disk.

    Args:
        path: Path to the serialized artifact.

    Returns:
        The deserialized object.

    Raises:
        ModelNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(f"No artifact found at {path}")
    return joblib.load(path)


# --------------------------------------------------------------------------- #
# Misc decorators
# --------------------------------------------------------------------------- #

F = TypeVar("F", bound=Callable[..., Any])


def timer(func: F) -> F:
    """Decorator that logs a function's wall-clock execution time."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.2fs", func.__qualname__, elapsed)
        return result

    return wrapper  # type: ignore[return-value]
