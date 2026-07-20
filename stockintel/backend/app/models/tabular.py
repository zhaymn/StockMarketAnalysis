"""Tabular models: logistic regression, random forest, LightGBM.

Each fits its own scaler inside `fit`, on training rows only. Hyperparameters
are deliberately conservative -- these datasets are ~1000 rows with a signal-
to-noise ratio near zero, a regime where an unconstrained gradient-boosted
model memorises noise perfectly and generalises not at all. The regularisation
here is not timidity; it is the difference between a model that beats the
baseline and one that does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.core.logging import get_logger
from app.models.base import Predictor

logger = get_logger(__name__)


class LogisticRegressionModel(Predictor):
    """L2-regularised logistic regression on standardised features.

    The interpretable reference point. If a gradient-boosted forest cannot beat
    a linear model on this data, the extra capacity is fitting noise -- a
    comparison worth making explicitly rather than assuming.
    """

    name = "Logistic Regression"
    family = "linear"
    description = (
        "L2-regularised linear classifier on standardised technical features. "
        "Interpretable reference model."
    )

    def __init__(self, n_classes: int, random_seed: int = 42, C: float = 0.1) -> None:
        super().__init__(n_classes, random_seed)
        # C=0.1 is strong regularisation: with ~34 correlated features over
        # ~1000 noisy rows, the unregularised fit is badly overconfident.
        self.C = C
        self._scaler = StandardScaler()
        self._model: LogisticRegression | None = None

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        x_scaled = self._scaler.fit_transform(x.to_numpy(dtype=float))
        self._model = LogisticRegression(
            C=self.C,
            max_iter=2000,
            random_state=self.random_seed,
            class_weight="balanced",
        )
        self._model.fit(x_scaled, np.asarray(y).astype(int))
        self._record_fit(x)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        assert self._model is not None, "Model not fitted"
        x_scaled = self._scaler.transform(x[self._feature_names].to_numpy(dtype=float))
        return self._align_proba(self._model.predict_proba(x_scaled), self._model.classes_)

    def _align_proba(self, proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """Expand to full class width.

        A training fold can legitimately lack a class (a quiet stretch with no
        BEARISH weeks). sklearn then emits a narrower matrix; indexing it by
        class id downstream would silently read the wrong column.
        """
        if proba.shape[1] == self.n_classes:
            return proba
        full = np.zeros((proba.shape[0], self.n_classes))
        for column, class_id in enumerate(classes.astype(int)):
            full[:, class_id] = proba[:, column]
        return full

    def feature_importance(self) -> dict[str, float] | None:
        if self._model is None:
            return None
        # Mean absolute coefficient across classes; features are standardised,
        # so magnitudes are comparable.
        coefficients = np.abs(self._model.coef_).mean(axis=0)
        total = coefficients.sum()
        if total <= 0:
            return None
        return {
            name: float(value / total)
            for name, value in zip(self._feature_names, coefficients)
        }


class RandomForestModel(Predictor):
    """Random forest with depth capped to resist noise-fitting."""

    name = "Random Forest"
    family = "tree"
    description = "Bagged decision trees capturing non-linear feature interactions."

    def __init__(self, n_classes: int, random_seed: int = 42) -> None:
        super().__init__(n_classes, random_seed)
        self._model: RandomForestClassifier | None = None

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        self._model = RandomForestClassifier(
            n_estimators=300,
            # Depth 6 and a 20-row leaf minimum: unconstrained trees drive
            # training accuracy to ~100% on this data and test accuracy to
            # chance.
            max_depth=6,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=self.random_seed,
            n_jobs=-1,
        )
        self._model.fit(x.to_numpy(dtype=float), np.asarray(y).astype(int))
        self._record_fit(x)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        assert self._model is not None, "Model not fitted"
        proba = self._model.predict_proba(x[self._feature_names].to_numpy(dtype=float))
        if proba.shape[1] == self.n_classes:
            return proba
        full = np.zeros((proba.shape[0], self.n_classes))
        for column, class_id in enumerate(self._model.classes_.astype(int)):
            full[:, class_id] = proba[:, column]
        return full

    def feature_importance(self) -> dict[str, float] | None:
        if self._model is None:
            return None
        importances = self._model.feature_importances_
        total = importances.sum()
        if total <= 0:
            return None
        return {
            name: float(value / total)
            for name, value in zip(self._feature_names, importances)
        }


class LightGBMModel(Predictor):
    """Gradient-boosted trees, heavily regularised for small noisy datasets."""

    name = "LightGBM"
    family = "tree"
    description = (
        "Gradient-boosted decision trees over technical, market-context and "
        "sentiment features."
    )

    def __init__(self, n_classes: int, random_seed: int = 42) -> None:
        super().__init__(n_classes, random_seed)
        self._model = None

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        import lightgbm as lgb

        y = np.asarray(y).astype(int)
        objective = "binary" if self.n_classes == 2 else "multiclass"

        params: dict[str, object] = {
            "objective": objective,
            "learning_rate": 0.03,
            "n_estimators": 300,
            # num_leaves=15 with depth 4: far below LightGBM's defaults, which
            # are tuned for datasets orders of magnitude larger than this.
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 30,
            # Row and column subsampling decorrelate the ensemble, which
            # matters more than usual when features are as collinear as
            # technical indicators are.
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": self.random_seed,
            "verbosity": -1,
            "n_jobs": -1,
        }
        if objective == "multiclass":
            params["num_class"] = self.n_classes

        self._model = lgb.LGBMClassifier(**params)
        self._model.fit(x.to_numpy(dtype=float), y)
        self._record_fit(x)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        assert self._model is not None, "Model not fitted"
        proba = self._model.predict_proba(x[self._feature_names].to_numpy(dtype=float))
        proba = np.asarray(proba)

        if proba.ndim == 1:  # binary LightGBM can return a 1D positive-class vector
            proba = np.column_stack([1.0 - proba, proba])
        if proba.shape[1] == self.n_classes:
            return proba

        full = np.zeros((proba.shape[0], self.n_classes))
        for column, class_id in enumerate(np.asarray(self._model.classes_).astype(int)):
            full[:, class_id] = proba[:, column]
        return full

    def feature_importance(self) -> dict[str, float] | None:
        if self._model is None:
            return None
        # "gain" — total loss reduction attributed to each feature. Far more
        # meaningful than the default "split" count, which over-credits
        # high-cardinality features that are split on often but usefully never.
        importances = self._model.booster_.feature_importance(importance_type="gain")
        total = importances.sum()
        if total <= 0:
            return None
        return {
            name: float(value / total)
            for name, value in zip(self._feature_names, importances)
        }


TABULAR_MODELS: tuple[type[Predictor], ...] = (
    LogisticRegressionModel,
    RandomForestModel,
    LightGBMModel,
)
