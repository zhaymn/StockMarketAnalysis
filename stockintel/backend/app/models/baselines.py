"""Naive baselines.

These exist to be beaten, and they are surprisingly hard to beat. Daily equity
returns are close to a random walk with a small upward drift, so "always
predict UP" scores 51-54% on most large-caps over most periods. Any model that
cannot clear that number reliably has learned nothing, however sophisticated
its architecture.

The platform reports these alongside every model result. If the ensemble ties
the majority-class baseline, the dashboard will say so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.models.base import Predictor


class MajorityClassBaseline(Predictor):
    """Always predicts the most frequent class in the training set.

    The primary bar. For 1-day direction this is "always UP" on most equities;
    for the 5-day outlook it is usually "always NEUTRAL".
    """

    name = "Majority class"
    family = "baseline"
    description = (
        "Always predicts whichever class was most common during training. "
        "The minimum bar any real model must clear."
    )

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        counts = np.bincount(np.asarray(y).astype(int), minlength=self.n_classes)
        self._majority_class = int(counts.argmax())
        # Class frequencies double as this model's probability estimate, which
        # is in fact perfectly calibrated -- it just carries no information.
        self._class_frequencies = counts / max(1, counts.sum())
        self._record_fit(x)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.tile(self._class_frequencies, (len(x), 1))


class MomentumBaseline(Predictor):
    """Predicts that the most recent return persists.

    A real if crude trading heuristic, and a sharper test than the majority
    class: it tests whether a model beats simple trend-following, not merely
    whether it beats a constant.
    """

    name = "Momentum persistence"
    family = "baseline"
    description = (
        "Predicts the last observed return direction will continue. "
        "Tests whether a model beats naive trend-following."
    )

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        if "log_return_1d" not in x.columns:
            raise ValueError("MomentumBaseline requires the 'log_return_1d' feature.")

        # Empirical accuracy of the persistence rule on the training set,
        # used as this model's confidence. Derived from data, not assigned.
        y = np.asarray(y).astype(int)
        signal = self._signal_to_class(x["log_return_1d"].to_numpy())
        self._confidence = float((signal == y).mean()) if len(y) else 0.5
        self._confidence = min(max(self._confidence, 0.5), 0.95)
        self._record_fit(x)

    def _signal_to_class(self, last_return: np.ndarray) -> np.ndarray:
        if self.n_classes == 2:
            return (last_return > 0).astype(int)
        # 3-class: map a positive last return to BULLISH(2), negative to
        # BEARISH(0). Persistence has no notion of NEUTRAL.
        return np.where(last_return > 0, 2, 0)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        predicted = self._signal_to_class(x["log_return_1d"].to_numpy())

        proba = np.full((len(x), self.n_classes), (1.0 - self._confidence))
        proba /= max(1, self.n_classes - 1)
        proba[np.arange(len(x)), predicted] = self._confidence
        return proba


class RandomBaseline(Predictor):
    """Samples from the training class distribution. The true floor.

    Distinct from the majority baseline: it shows what pure chance looks like
    at this class balance, which is the reference for the ROC-AUC of 0.5 line.
    """

    name = "Random (stratified)"
    family = "baseline"
    description = "Draws predictions from the training class distribution."

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        counts = np.bincount(np.asarray(y).astype(int), minlength=self.n_classes)
        self._class_frequencies = counts / max(1, counts.sum())
        self._rng = np.random.default_rng(self.random_seed)
        self._record_fit(x)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.tile(self._class_frequencies, (len(x), 1))

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self._rng.choice(self.n_classes, size=len(x), p=self._class_frequencies)


BASELINE_MODELS: tuple[type[Predictor], ...] = (
    MajorityClassBaseline,
    MomentumBaseline,
    RandomBaseline,
)
