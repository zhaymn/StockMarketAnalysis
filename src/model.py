"""Deep learning model architectures for stock price prediction.

Provides a stacked LSTM as the primary architecture, plus GRU,
Bidirectional LSTM, CNN-LSTM hybrid, and Attention-LSTM variants built from
a single configurable factory (``build_model``) so the five named builders
below share one code path instead of duplicating layer-stacking logic.

Training-time concerns (EarlyStopping, ModelCheckpoint, LR scheduling,
TensorBoard) live in ``src.train`` — this module only constructs and
compiles the network (architecture + optimizer + loss are "model" concerns;
callbacks are "fit-call" concerns).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import keras
import tensorflow as tf
from keras import layers

from src.utils import get_logger

logger = get_logger(__name__)

CellType = Literal["lstm", "gru"]


@dataclass
class ModelConfig:
    """Hyperparameters for a recurrent stock-prediction network.

    These are exactly the knobs a hyperparameter search would tune, kept in
    one place so experiments only need to construct a different config
    rather than edit architecture code.
    """

    cell_type: CellType = "lstm"
    recurrent_units: tuple[int, ...] = (128, 64, 32)
    dropout_rate: float = 0.2
    dense_units: tuple[int, ...] = (25,)
    dense_dropout_rate: float = 0.1
    bidirectional: bool = False
    use_conv_block: bool = False
    conv_filters: int = 64
    conv_kernel_size: int = 3
    use_attention: bool = False
    learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        if not self.recurrent_units:
            raise ValueError("recurrent_units must contain at least one layer size.")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")


@keras.saving.register_keras_serializable(package="stock_predictor")
class AttentionLayer(layers.Layer):
    """Bahdanau-style additive attention pooling over an LSTM/GRU sequence output.

    Learns a scalar importance score for each timestep, then returns the
    softmax-weighted sum of timesteps as a single context vector — letting
    the model attend more to the days that matter most rather than treating
    the final timestep's hidden state as a bottleneck summary of the window.

    score_t = tanh(W . h_t + b)
    alpha_t = softmax_t(u^T . score_t)
    context = sum_t(alpha_t * h_t)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape: tuple[int, ...]) -> None:
        feature_dim = int(input_shape[-1])
        self.W = self.add_weight(
            name="attention_W", shape=(feature_dim, feature_dim),
            initializer="glorot_uniform", trainable=True,
        )
        self.b = self.add_weight(
            name="attention_b", shape=(feature_dim,), initializer="zeros", trainable=True,
        )
        self.u = self.add_weight(
            name="attention_u", shape=(feature_dim, 1),
            initializer="glorot_uniform", trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs: (batch, timesteps, features)
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)  # (batch, timesteps, features)
        logits = tf.tensordot(score, self.u, axes=1)  # (batch, timesteps, 1)
        attention_weights = tf.nn.softmax(logits, axis=1)  # (batch, timesteps, 1)
        context = tf.reduce_sum(inputs * attention_weights, axis=1)  # (batch, features)
        return context

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], input_shape[-1])


def _recurrent_layer(cell_type: CellType, units: int, return_sequences: bool) -> layers.Layer:
    """Instantiate a single LSTM or GRU layer with shared defaults."""
    if cell_type == "lstm":
        return layers.LSTM(units, return_sequences=return_sequences)
    if cell_type == "gru":
        return layers.GRU(units, return_sequences=return_sequences)
    raise ValueError(f"Unknown cell_type: {cell_type!r}")


def build_model(input_shape: tuple[int, int], config: ModelConfig | None = None) -> keras.Model:
    """Construct and compile a recurrent stock-prediction network.

    Architecture (top to bottom):
      [optional Conv1D + MaxPooling1D block]
      -> stacked LSTM/GRU layers (optionally Bidirectional), each followed by Dropout
      -> [optional Attention pooling, otherwise the last recurrent layer already
         collapses the time dimension]
      -> Dense hidden layers with Dropout
      -> Dense(1) linear output (regression head)

    Args:
        input_shape: ``(window_size, n_features)`` of a single input sequence.
        config: Hyperparameters; defaults to a 3-layer stacked LSTM.

    Returns:
        A compiled ``keras.Model`` using the Adam optimizer and MSE loss,
        tracking MAE as an additional metric.
    """
    config = config or ModelConfig()

    inputs = keras.Input(shape=input_shape, name="price_sequence")
    x = inputs

    if config.use_conv_block:
        x = layers.Conv1D(
            filters=config.conv_filters,
            kernel_size=config.conv_kernel_size,
            activation="relu",
            padding="causal",
            name="conv1d_feature_extractor",
        )(x)
        x = layers.MaxPooling1D(pool_size=2, padding="same")(x)

    n_recurrent_layers = len(config.recurrent_units)
    for i, units in enumerate(config.recurrent_units):
        is_last_recurrent_layer = i == n_recurrent_layers - 1
        # Keep the time dimension if this isn't the last recurrent layer, or
        # if attention pooling needs the full sequence to attend over.
        return_sequences = (not is_last_recurrent_layer) or config.use_attention

        recurrent = _recurrent_layer(config.cell_type, units, return_sequences)
        if config.bidirectional:
            recurrent = layers.Bidirectional(recurrent, name=f"bidirectional_{config.cell_type}_{i}")
        x = recurrent(x)
        x = layers.Dropout(config.dropout_rate)(x)

    if config.use_attention:
        x = AttentionLayer(name="attention_pooling")(x)

    for i, units in enumerate(config.dense_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        x = layers.Dropout(config.dense_dropout_rate)(x)

    outputs = layers.Dense(1, activation="linear", name="predicted_close")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="stock_price_predictor")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    logger.info(
        "Built %s model: %d recurrent layers %s, attention=%s, conv_block=%s, %d params",
        config.cell_type.upper(),
        n_recurrent_layers,
        "(bidirectional)" if config.bidirectional else "",
        config.use_attention,
        config.use_conv_block,
        model.count_params(),
    )
    return model


# --------------------------------------------------------------------------- #
# Named convenience builders (what the project brief asks for by name)
# --------------------------------------------------------------------------- #


def build_lstm_model(input_shape: tuple[int, int], **overrides) -> keras.Model:
    """Stacked LSTM (the primary architecture): 3 layers, dropout, dense head."""
    return build_model(input_shape, ModelConfig(cell_type="lstm", **overrides))


def build_gru_model(input_shape: tuple[int, int], **overrides) -> keras.Model:
    """Stacked GRU: fewer parameters than LSTM, often comparable accuracy."""
    return build_model(input_shape, ModelConfig(cell_type="gru", **overrides))


def build_bidirectional_lstm_model(input_shape: tuple[int, int], **overrides) -> keras.Model:
    """Bidirectional stacked LSTM: each layer sees the window both forwards and backwards."""
    return build_model(input_shape, ModelConfig(cell_type="lstm", bidirectional=True, **overrides))


def build_cnn_lstm_model(input_shape: tuple[int, int], **overrides) -> keras.Model:
    """CNN-LSTM hybrid: a Conv1D block extracts local patterns before the LSTM stack."""
    return build_model(input_shape, ModelConfig(cell_type="lstm", use_conv_block=True, **overrides))


def build_attention_lstm_model(input_shape: tuple[int, int], **overrides) -> keras.Model:
    """LSTM stack with additive-attention pooling instead of using only the final timestep."""
    return build_model(input_shape, ModelConfig(cell_type="lstm", use_attention=True, **overrides))


MODEL_REGISTRY: dict[str, Callable[..., keras.Model]] = {
    "lstm": build_lstm_model,
    "gru": build_gru_model,
    "bidirectional_lstm": build_bidirectional_lstm_model,
    "cnn_lstm": build_cnn_lstm_model,
    "attention_lstm": build_attention_lstm_model,
}


def build_model_by_name(name: str, input_shape: tuple[int, int], **overrides) -> keras.Model:
    """Look up and build a model by its registry name (used for model-comparison sweeps).

    Args:
        name: One of ``MODEL_REGISTRY`` keys, e.g. ``"lstm"``, ``"cnn_lstm"``.
        input_shape: ``(window_size, n_features)``.
        **overrides: Extra ``ModelConfig`` fields to override the preset defaults.

    Returns:
        A compiled ``keras.Model``.

    Raises:
        ValueError: If ``name`` is not a registered architecture.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name '{name}'. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](input_shape, **overrides)
