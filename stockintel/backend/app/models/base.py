"""Common model interface.

Every predictor -- naive baseline, logistic regression, LightGBM, LSTM,
ensemble -- implements `Predictor`, so the backtesting harness can evaluate
them under identical conditions. Comparing models trained through different
code paths is how flattering-but-false results get published; a single
interface makes the comparison structurally fair.

Scaling is deliberately the *model's* responsibility, not the harness's. A
scaler fitted on the full dataset before splitting leaks test-set distribution
into training. Each model fits its own scaler inside `fit`, on training rows
only, per fold.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ModelMetadata:
    """Describes a trained model for the Model Transparency panel."""

    name: str
    family: str
    """"baseline" | "linear" | "tree" | "deep" | "ensemble"."""

    description: str
    hyperparameters: dict[str, object] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    n_training_rows: int = 0
    training_period: tuple[str, str] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "hyperparameters": self.hyperparameters,
            "n_features": len(self.feature_names),
            "n_training_rows": self.n_training_rows,
            "training_period": list(self.training_period) if self.training_period else None,
        }


class Predictor(ABC):
    """A model that predicts a class label and a probability distribution."""

    name: str
    family: str
    description: str

    #: Set True by models that need a 2D sequence window rather than a flat
    #: feature row, so the harness knows to hand them sequence data.
    requires_sequences: bool = False

    def __init__(self, n_classes: int, random_seed: int = 42) -> None:
        self.n_classes = n_classes
        self.random_seed = random_seed
        self._is_fitted = False
        self._feature_names: list[str] = []

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        """Fit on training rows only. Must fit any scaler internally."""

    @abstractmethod
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Return `(n_samples, n_classes)` probabilities."""

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predicted class indices (argmax of `predict_proba`)."""
        return np.asarray(self.predict_proba(x)).argmax(axis=1)

    def feature_importance(self) -> dict[str, float] | None:
        """Per-feature contribution, if the model exposes one.

        Drives the "Why this prediction?" panel. Returning None is correct for
        models with no meaningful notion of importance -- inventing one would
        fabricate an explanation.
        """
        return None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name=self.name,
            family=self.family,
            description=self.description,
            feature_names=list(self._feature_names),
        )

    def _record_fit(self, x: pd.DataFrame) -> None:
        self._feature_names = list(x.columns)
        self._is_fitted = True
