"""Stacked ensemble tests.

The first test is the one that matters: the meta-learner must be fitted on
out-of-fold base predictions. Fitting on in-sample predictions is the classic
stacking bug — it produces a model that looks excellent in validation and fails
in production, because the meta-learner learned to trust whichever base model
memorised the training set hardest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.models.baselines import MajorityClassBaseline
from app.models.base import Predictor
from app.models.ensemble import EnsembleConfig, StackedEnsemble
from app.models.tabular import LightGBMModel, LogisticRegressionModel


def make_data(n: int = 900, seed: int = 0, n_classes: int = 3):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "log_return_1d": rng.normal(scale=0.02, size=n),
            "signal": rng.normal(size=n),
            "noise_a": rng.normal(size=n),
            "noise_b": rng.normal(size=n),
        },
        index=pd.bdate_range("2019-01-01", periods=n),
    )
    # A learnable relationship so the stack has something to combine.
    logits = frame["signal"].to_numpy()
    y = np.digitize(logits + rng.normal(scale=0.5, size=n), [-0.5, 0.5])
    return frame, y.astype(int)[:n] % n_classes


@pytest.fixture
def config():
    return EnsembleConfig(inner_folds=3, horizon=5, min_inner_train=150)


def _factories(n_classes: int, seed: int = 0):
    return [
        ("logreg", lambda: LogisticRegressionModel(n_classes, seed)),
        ("lgbm", lambda: LightGBMModel(n_classes, seed)),
    ]


# --- The core guarantee ----------------------------------------------------

class LeakDetectingModel(Predictor):
    """Base model that records which row indices it was asked to predict."""

    name = "leak-detector"
    family = "baseline"
    description = "Test double."

    predicted_index_sets: list[set] = []
    trained_index_sets: list[set] = []

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        LeakDetectingModel.trained_index_sets.append(set(x.index))
        counts = np.bincount(np.asarray(y).astype(int), minlength=self.n_classes)
        self._freq = counts / max(1, counts.sum())
        self._record_fit(x)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        LeakDetectingModel.predicted_index_sets.append(set(x.index))
        return np.tile(self._freq, (len(x), 1))


def test_meta_learner_trains_on_out_of_fold_predictions_only(config):
    """No inner fold may predict rows it was trained on."""
    LeakDetectingModel.trained_index_sets = []
    LeakDetectingModel.predicted_index_sets = []

    frame, y = make_data(900)
    ensemble = StackedEnsemble(
        3, 0,
        base_factories=[("leaky", lambda: LeakDetectingModel(3, 0))],
        config=config,
    )
    ensemble.fit(frame, y)

    # Pair each inner training set with the prediction that followed it. The
    # final fit/predict pair is the full-data refit, which legitimately covers
    # everything, so it is excluded.
    inner_pairs = list(
        zip(
            LeakDetectingModel.trained_index_sets[:-1],
            LeakDetectingModel.predicted_index_sets,
        )
    )
    assert len(inner_pairs) >= 2, "Expected several inner folds"

    for trained_on, predicted in inner_pairs:
        overlap = trained_on & predicted
        assert not overlap, (
            f"Meta-learner fed in-sample predictions: {len(overlap)} rows were "
            f"both trained on and predicted within one inner fold."
        )


def test_inner_folds_respect_the_purge_gap(config):
    """Inner splits must purge by the horizon, exactly like outer splits."""
    LeakDetectingModel.trained_index_sets = []
    LeakDetectingModel.predicted_index_sets = []

    frame, y = make_data(900)
    ensemble = StackedEnsemble(
        3, 0,
        base_factories=[("leaky", lambda: LeakDetectingModel(3, 0))],
        config=config,
    )
    ensemble.fit(frame, y)

    positions = {timestamp: i for i, timestamp in enumerate(frame.index)}

    for trained_on, predicted in zip(
        LeakDetectingModel.trained_index_sets[:-1],
        LeakDetectingModel.predicted_index_sets,
    ):
        if not trained_on or not predicted:
            continue
        last_train = max(positions[t] for t in trained_on)
        first_test = min(positions[t] for t in predicted)
        # A training label spanning `horizon` days must end before the test block.
        assert last_train + config.horizon <= first_test


# --- Behaviour -------------------------------------------------------------

def test_fit_and_predict_shapes(config):
    frame, y = make_data(800)
    ensemble = StackedEnsemble(3, 0, base_factories=_factories(3), config=config)
    ensemble.fit(frame, y)

    proba = ensemble.predict_proba(frame)
    assert proba.shape == (800, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_reports_learned_base_weights(config):
    frame, y = make_data(800)
    ensemble = StackedEnsemble(3, 0, base_factories=_factories(3), config=config)
    ensemble.fit(frame, y)

    report = ensemble.stacking_report()
    assert report["used_meta_learner"] is True
    assert report["oof_rows_available"] > 100

    weights = report["base_model_weights"]
    assert set(weights) == {"logreg", "lgbm"}
    # Weights are learned, so they must not be a hardcoded even split.
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights != {"logreg": 0.5, "lgbm": 0.5}


def test_falls_back_transparently_when_oof_is_too_small():
    """Degradation must be reported, not silent."""
    frame, y = make_data(260)
    ensemble = StackedEnsemble(
        3, 0,
        base_factories=_factories(3),
        config=EnsembleConfig(inner_folds=3, horizon=5, min_inner_train=100),
    )
    ensemble.fit(frame, y)

    report = ensemble.stacking_report()
    if not report["used_meta_learner"]:
        assert "Equal-weight" in report["combination_method"]
        # Must still produce valid probabilities.
        proba = ensemble.predict_proba(frame)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_survives_a_failing_base_model(config):
    """One broken base model must not sink the ensemble."""

    class BrokenModel(Predictor):
        name = "broken"
        family = "baseline"
        description = "Always fails."

        def fit(self, x, y):
            raise RuntimeError("intentional failure")

        def predict_proba(self, x):
            raise RuntimeError("intentional failure")

    frame, y = make_data(800)
    ensemble = StackedEnsemble(
        3, 0,
        base_factories=[
            ("broken", lambda: BrokenModel(3, 0)),
            ("logreg", lambda: LogisticRegressionModel(3, 0)),
        ],
        config=config,
    )
    ensemble.fit(frame, y)

    proba = ensemble.predict_proba(frame)
    assert proba.shape == (800, 3)
    assert np.isfinite(proba).all()


def test_raises_when_every_base_model_fails(config):
    class BrokenModel(Predictor):
        name = "broken"
        family = "baseline"
        description = "Always fails."

        def fit(self, x, y):
            raise RuntimeError("intentional failure")

        def predict_proba(self, x):
            raise RuntimeError("intentional failure")

    frame, y = make_data(800)
    ensemble = StackedEnsemble(
        3, 0, base_factories=[("broken", lambda: BrokenModel(3, 0))], config=config
    )
    with pytest.raises(ValueError, match="No base model"):
        ensemble.fit(frame, y)


def test_feature_importance_reports_models_not_features(config):
    frame, y = make_data(800)
    ensemble = StackedEnsemble(3, 0, base_factories=_factories(3), config=config)
    ensemble.fit(frame, y)

    importance = ensemble.feature_importance()
    assert importance is not None
    # Inputs are model outputs, so the keys must name models.
    assert all(key.startswith("model: ") for key in importance)
