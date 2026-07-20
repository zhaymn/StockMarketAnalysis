"""LSTM correctness and leakage tests.

The window-direction test is the important one. If sequences were built
forward-looking, the model would score spectacularly on the synthetic task
below and the bug would be invisible in aggregate metrics — it would just look
like the deep model finally working.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.models.lstm import LSTMConfig, LSTMModel, _build_sequences

pytest.importorskip("torch")


@pytest.fixture
def small_config():
    return LSTMConfig(sequence_length=5, hidden_size=8, num_layers=1, max_epochs=6, patience=3)


# --- Sequence construction -------------------------------------------------

def test_window_contains_only_past_and_present():
    """Window for row i must be rows [i-L+1 … i] — never row i+1."""
    values = np.arange(20, dtype=float).reshape(20, 1)
    sequences = _build_sequences(values, sequence_length=4)

    assert sequences.shape == (20, 4, 1)

    for i in range(20):
        window = sequences[i, :, 0]
        # The window must end on row i.
        assert window[-1] == float(i), f"Row {i} window ends at {window[-1]}, expected {i}"
        # And must contain nothing from the future.
        assert window.max() <= float(i), f"Row {i} window saw the future: {window}"


def test_early_rows_are_left_padded_not_dropped():
    values = np.arange(10, dtype=float).reshape(10, 1)
    sequences = _build_sequences(values, sequence_length=4)

    # Output length equals input length — no rows silently discarded.
    assert len(sequences) == 10
    # Row 0 has no history, so its window is row 0 repeated.
    assert list(sequences[0, :, 0]) == [0.0, 0.0, 0.0, 0.0]
    # Row 2 has partial history.
    assert list(sequences[2, :, 0]) == [0.0, 0.0, 1.0, 2.0]
    # Row 5 has a full window.
    assert list(sequences[5, :, 0]) == [2.0, 3.0, 4.0, 5.0]


def test_empty_input_returns_empty_sequences():
    sequences = _build_sequences(np.empty((0, 3)), sequence_length=5)
    assert sequences.shape == (0, 5, 3)


# --- Training behaviour ----------------------------------------------------

def _synthetic_data(n: int = 400, seed: int = 0):
    """Features whose PAST carries the label, so a correct LSTM can learn it."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    frame = pd.DataFrame(
        {"signal": signal, "noise": rng.normal(size=n)},
        index=pd.bdate_range("2020-01-01", periods=n),
    )
    # Label at t depends on the signal at t-1 — learnable only from history.
    y = (pd.Series(signal).shift(1).fillna(0) > 0).astype(int).to_numpy()
    return frame, y


def test_fit_and_predict_shapes(small_config):
    frame, y = _synthetic_data(300)
    model = LSTMModel(n_classes=2, random_seed=0, config=small_config)
    model.fit(frame, y)

    proba = model.predict_proba(frame)
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_scaler_is_fitted_on_training_rows_only(small_config):
    """A scaler fitted on the full series leaks test distribution into training."""
    frame, y = _synthetic_data(300)
    model = LSTMModel(n_classes=2, random_seed=0, config=small_config)
    model.fit(frame, y)

    split = int(len(frame) * (1.0 - small_config.validation_fraction))
    expected_mean = frame.to_numpy()[:split].mean(axis=0)

    # The fitted scaler's mean must match the training portion, not the whole set.
    assert np.allclose(model._scaler.mean_, expected_mean, atol=1e-9)
    assert not np.allclose(model._scaler.mean_, frame.to_numpy().mean(axis=0), atol=1e-9)


def test_training_is_reproducible(small_config):
    frame, y = _synthetic_data(300)

    first = LSTMModel(2, random_seed=7, config=small_config)
    first.fit(frame, y)
    second = LSTMModel(2, random_seed=7, config=small_config)
    second.fit(frame, y)

    assert np.allclose(first.predict_proba(frame), second.predict_proba(frame), atol=1e-6)


def test_different_seeds_produce_different_models(small_config):
    frame, y = _synthetic_data(300)

    first = LSTMModel(2, random_seed=1, config=small_config)
    first.fit(frame, y)
    second = LSTMModel(2, random_seed=99, config=small_config)
    second.fit(frame, y)

    assert not np.allclose(first.predict_proba(frame), second.predict_proba(frame), atol=1e-6)


def test_early_stopping_restores_best_checkpoint():
    frame, y = _synthetic_data(400)
    config = LSTMConfig(sequence_length=5, hidden_size=8, num_layers=1, max_epochs=100, patience=4)

    model = LSTMModel(2, random_seed=0, config=config)
    model.fit(frame, y)
    history = model.training_history()

    assert history["epochs_run"] <= 100
    # The restored weights must come from the best epoch, not the last one.
    assert history["best_epoch"] <= history["epochs_run"] - 1
    if history["early_stopped"]:
        val = history["val_loss"]
        assert val[history["best_epoch"]] == min(val)


def test_can_learn_a_genuinely_learnable_temporal_signal():
    """Sanity check that the plumbing works at all.

    If the model cannot beat chance on data where the label IS the previous
    row's feature sign, then a poor result on market data would be
    uninterpretable — it could be a bug rather than an absence of signal.
    """
    frame, y = _synthetic_data(600, seed=3)
    config = LSTMConfig(sequence_length=5, hidden_size=24, num_layers=1, max_epochs=60, patience=10)

    model = LSTMModel(2, random_seed=0, config=config)
    split = 400
    model.fit(frame.iloc[:split], y[:split])

    predictions = model.predict_proba(frame.iloc[split:]).argmax(axis=1)
    accuracy = (predictions == y[split:]).mean()

    assert accuracy > 0.75, f"Only {accuracy:.3f} on a learnable signal — check the pipeline"


def test_predicts_more_than_one_class_on_imbalanced_data():
    """Class weighting must prevent collapse to majority-only prediction."""
    rng = np.random.default_rng(5)
    n = 400
    frame = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n)},
        index=pd.bdate_range("2020-01-01", periods=n),
    )
    # 80/20 imbalance.
    y = (rng.uniform(size=n) < 0.2).astype(int)

    config = LSTMConfig(sequence_length=5, hidden_size=16, num_layers=1, max_epochs=40, patience=8)
    model = LSTMModel(2, random_seed=0, config=config)
    model.fit(frame, y)

    predictions = model.predict_proba(frame).argmax(axis=1)
    assert len(np.unique(predictions)) > 1, "Collapsed to a single class despite weighting"


def test_feature_importance_is_none_rather_than_fabricated():
    frame, y = _synthetic_data(200)
    config = LSTMConfig(sequence_length=5, hidden_size=8, num_layers=1, max_epochs=3, patience=2)
    model = LSTMModel(2, random_seed=0, config=config)
    model.fit(frame, y)

    assert model.feature_importance() is None
