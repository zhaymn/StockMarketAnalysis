"""Stacked ensemble — the engine behind "Most Possible Prediction".

**Why stacking rather than fixed weights.** The brief explicitly rules out
hand-invented blends like "40% LSTM, 30% sentiment, 30% technical". Those
numbers encode a belief about relative model quality that nobody has measured.
Here the combination is *learned*: a logistic meta-learner is fitted on base
model outputs and discovers the weighting from data, including the entirely
legitimate outcome of learning to ignore a base model completely.

**The leakage trap this module exists to avoid.** The naive implementation
trains base models on the training window, predicts that same window, and fits
the meta-learner on those predictions. Those predictions are in-sample, so the
most overfit base model looks the most accurate, and the meta-learner learns to
trust it — producing a stack that scores brilliantly in training and collapses
out-of-sample. The only correct input is *out-of-fold* predictions.

So `fit` runs a nested, purged walk-forward inside the training window:

    outer training window T
      ├─ inner fold 1: train on T[:a] → predict T[a:b]   ┐
      ├─ inner fold 2: train on T[:b] → predict T[b:c]   ├─ OOF predictions
      └─ inner fold 3: train on T[:c] → predict T[c:]    ┘
                                    ↓
                    meta-learner fitted on OOF preds → y
                                    ↓
                 base models refitted on all of T, then applied to the
                 outer test block and mapped through the meta-learner

Every inner split carries the same purge and embargo as the outer one, so no
label window ever straddles a boundary.

Cost is real: `inner_folds × n_base_models + n_base_models` fits per outer
fold. That is the price of a stack whose reported score means something.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.backtesting.splits import WalkForwardConfig, walk_forward_splits
from app.core.logging import get_logger
from app.models.base import Predictor

logger = get_logger(__name__)

BaseFactory = Callable[[], Predictor]


@dataclass
class EnsembleConfig:
    """Stacking parameters."""

    inner_folds: int = 3
    """Inner walk-forward folds used to generate out-of-fold predictions.
    Three balances OOF coverage against fit cost; more folds give the
    meta-learner more rows but multiply training time."""

    horizon: int = 5
    """Label horizon, sizing the inner purge and embargo."""

    meta_regularisation: float = 1.0
    """Inverse regularisation strength (sklearn `C`) for the meta-learner.
    Kept fairly strong: the meta-learner sees only a few hundred OOF rows and
    a handful of correlated inputs, which overfits easily."""

    min_inner_train: int = 200


class StackedEnsemble(Predictor):
    """Learned combination of base predictors."""

    name = "Most Possible Prediction"
    family = "ensemble"
    description = (
        "Combines the platform's strongest validated signals and models through a "
        "meta-learner fitted on out-of-fold predictions. Represents the statistically "
        "estimated most likely outcome given available data — not a guaranteed "
        "market outcome."
    )

    def __init__(
        self,
        n_classes: int,
        random_seed: int = 42,
        base_factories: list[tuple[str, BaseFactory]] | None = None,
        config: EnsembleConfig | None = None,
    ) -> None:
        super().__init__(n_classes, random_seed)
        self.config = config or EnsembleConfig()

        if base_factories is None:
            from app.models.tabular import LightGBMModel, LogisticRegressionModel

            base_factories = [
                ("logreg", lambda: LogisticRegressionModel(n_classes, random_seed)),
                ("lgbm", lambda: LightGBMModel(n_classes, random_seed)),
            ]

        self.base_factories = base_factories
        self._base_models: list[tuple[str, Predictor]] = []
        self._meta: LogisticRegression | None = None
        self._meta_classes: np.ndarray | None = None
        self._oof_coverage = 0
        self._base_weights: dict[str, float] = {}

    # -- out-of-fold generation --------------------------------------------
    def _generate_oof(
        self, x: pd.DataFrame, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Produce out-of-fold base predictions over the training window.

        Returns `(meta_features, mask)` where `mask` marks rows that received a
        prediction. Early rows fall inside the first inner training window and
        never get one; they are excluded from meta-training rather than filled.
        """
        n_rows = len(x)
        n_bases = len(self.base_factories)
        meta_features = np.full((n_rows, n_bases * self.n_classes), np.nan)

        inner_config = WalkForwardConfig(
            n_folds=self.config.inner_folds,
            horizon=self.config.horizon,
            min_train_size=min(self.config.min_inner_train, max(50, n_rows // 3)),
        )

        try:
            folds = list(walk_forward_splits(x.index, inner_config))
        except Exception as exc:
            logger.warning("Inner walk-forward unavailable (%s); ensemble degrades.", exc)
            return meta_features, np.zeros(n_rows, dtype=bool)

        for fold in folds:
            x_train = x.iloc[fold.train_indices]
            y_train = y[fold.train_indices]
            x_test = x.iloc[fold.test_indices]

            if len(np.unique(y_train)) < 2:
                continue

            for base_index, (base_name, factory) in enumerate(self.base_factories):
                try:
                    model = factory()
                    model.fit(x_train, y_train)
                    proba = model.predict_proba(x_test)
                except Exception as exc:
                    logger.warning(
                        "Base model %s failed on inner fold %d: %s",
                        base_name, fold.fold_index, exc,
                    )
                    continue

                start = base_index * self.n_classes
                meta_features[fold.test_indices, start : start + self.n_classes] = proba

        mask = ~np.isnan(meta_features).any(axis=1)
        return meta_features, mask

    # -- training -----------------------------------------------------------
    def fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        y = np.asarray(y).astype(int)

        meta_features, mask = self._generate_oof(x, y)
        self._oof_coverage = int(mask.sum())

        # Refit base models on the full training window. These are the models
        # that will actually serve predictions; the inner-fold copies existed
        # only to produce honest OOF inputs.
        self._base_models = []
        for base_name, factory in self.base_factories:
            model = factory()
            try:
                model.fit(x, y)
                self._base_models.append((base_name, model))
            except Exception as exc:
                logger.error("Base model %s failed to fit: %s", base_name, exc)

        if not self._base_models:
            raise ValueError("No base model could be fitted; ensemble unavailable.")

        # --- Meta-learner --------------------------------------------------
        if self._oof_coverage < 60 or len(np.unique(y[mask])) < 2:
            # Not enough honest OOF rows to fit a meta-learner. Falling back to
            # an equal-weight average is a documented, inspectable degradation
            # -- not a silent one; `used_meta_learner` reports it.
            logger.info(
                "Only %d OOF rows; falling back to equal-weight averaging.",
                self._oof_coverage,
            )
            self._meta = None
        else:
            self._meta = LogisticRegression(
                C=self.config.meta_regularisation,
                max_iter=2000,
                random_state=self.random_seed,
                class_weight="balanced",
            )
            self._meta.fit(meta_features[mask], y[mask])
            self._meta_classes = self._meta.classes_
            self._compute_base_weights()

        self._record_fit(x)

    def _compute_base_weights(self) -> None:
        """Relative influence of each base model, from meta coefficients.

        Surfaced in the UI so "combines the strongest signals" is inspectable
        rather than a marketing claim. A base model the meta-learner learned to
        ignore shows a weight near zero, which is a real and useful finding.
        """
        if self._meta is None:
            return

        coefficients = np.abs(self._meta.coef_).mean(axis=0)
        totals: dict[str, float] = {}

        for base_index, (base_name, _) in enumerate(self.base_factories):
            start = base_index * self.n_classes
            totals[base_name] = float(coefficients[start : start + self.n_classes].sum())

        total = sum(totals.values())
        self._base_weights = (
            {name: value / total for name, value in totals.items()}
            if total > 0
            else {name: 0.0 for name in totals}
        )

    # -- inference ----------------------------------------------------------
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if not self._base_models:
            raise ValueError("Ensemble not fitted.")

        base_probas: list[np.ndarray] = []
        for _, model in self._base_models:
            base_probas.append(model.predict_proba(x[self._feature_names]))

        if self._meta is None:
            # Equal-weight fallback.
            return np.mean(base_probas, axis=0)

        meta_features = np.concatenate(base_probas, axis=1)
        proba = self._meta.predict_proba(meta_features)

        if proba.shape[1] == self.n_classes:
            return proba

        # A class absent from the meta-training rows yields a narrower matrix.
        full = np.zeros((proba.shape[0], self.n_classes))
        for column, class_id in enumerate(np.asarray(self._meta_classes).astype(int)):
            full[:, class_id] = proba[:, column]
        return full

    def feature_importance(self) -> dict[str, float] | None:
        """Base-model influence, not raw feature importance.

        The ensemble's inputs are model outputs, so this reports which models
        the meta-learner leans on. Reporting the underlying features here would
        conflate two different questions.
        """
        if not self._base_weights:
            return None
        return {f"model: {name}": weight for name, weight in self._base_weights.items()}

    def stacking_report(self) -> dict[str, object]:
        """How the combination was arrived at, for the transparency panel."""
        return {
            "base_models": [name for name, _ in self.base_factories],
            "used_meta_learner": self._meta is not None,
            "combination_method": (
                "Logistic meta-learner fitted on out-of-fold base predictions"
                if self._meta is not None
                else "Equal-weight average (too few out-of-fold rows to fit a meta-learner)"
            ),
            "oof_rows_available": self._oof_coverage,
            "inner_folds": self.config.inner_folds,
            "base_model_weights": {
                name: round(weight, 4) for name, weight in self._base_weights.items()
            },
        }
