"""LSTM sequence classifier.

Built on PyTorch rather than TensorFlow so a single deep-learning framework
serves both this model and FinBERT.

**What this model is for.** The tabular models see one row of engineered
features at a time; any temporal structure has to be hand-encoded as lags. The
LSTM reads an ordered window of raw feature rows and can in principle learn
temporal patterns that no single row expresses. That is the hypothesis it
exists to test — and, given the measured results elsewhere in this project, it
may well fail to beat a linear model. Demonstrating that is a legitimate result.

Leakage audit, since sequence models make several kinds easy to introduce:

* **Scaling** — the scaler is fitted inside `fit`, on training rows only, and
  reused unchanged at predict time. Fitting on the full series before splitting
  is the single most common way a sequence model gets an inflated score.
* **Window direction** — window for row `i` is rows `[i-L+1 … i]`. Never `i+1`.
* **Block boundaries** — rows near the start of a block have no room for a full
  window. They are left-padded by repeating the earliest available row, so
  every value used to predict row `i` comes from a row at or before `i`. See
  `_build_sequences`.
* **Shuffling** — batches are shuffled *within* the training window only.
  Shuffling batch composition does not move information across time; what would
  leak is shuffling before the train/test split, which never happens here.
* **Validation split** — carved chronologically from the end of the training
  window, never at random.
* **Early stopping** — monitored on that internal validation split, never on
  the harness's test block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.core.logging import get_logger
from app.models.base import Predictor

logger = get_logger(__name__)


@dataclass
class LSTMConfig:
    """Hyperparameters.

    Small by deliberate choice. With ~1,500 training rows and a signal-to-noise
    ratio near zero, a larger network memorises the training window perfectly
    and generalises not at all — the failure mode this whole project is built
    to detect rather than accidentally ship.
    """

    sequence_length: int = 20
    """Trading days per window. ~1 month of sessions."""

    hidden_size: int = 48
    num_layers: int = 2
    dropout: float = 0.25

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 120

    patience: int = 15
    """Early-stopping patience, in epochs without validation improvement."""

    validation_fraction: float = 0.15
    """Chronological tail of the training window held out for early stopping."""

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
        }


def _build_sequences(
    values: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    """Turn `(n_rows, n_features)` into `(n_rows, sequence_length, n_features)`.

    Output row `i` holds rows `[i-L+1 … i]` of the input. Rows near the start
    have no room for a full window and are left-padded by repeating row 0.

    That padding is causally safe: every value in window `i` comes from a row
    at index `≤ i`. Repeating the earliest available row as a stand-in for
    rows *before* the block introduces no information from the future. The
    alternative — dropping the first `L-1` rows — would silently shorten test
    blocks and make fold metrics incomparable between models.
    """
    n_rows = len(values)
    if n_rows == 0:
        return np.empty((0, sequence_length, values.shape[1]), dtype=np.float32)

    # Left-pad by replicating the first row, then stride out the windows.
    padded = np.concatenate(
        [np.repeat(values[:1], sequence_length - 1, axis=0), values], axis=0
    )
    return np.stack(
        [padded[i : i + sequence_length] for i in range(n_rows)]
    ).astype(np.float32)


class LSTMModel(Predictor):
    """Two-layer LSTM over a window of technical features."""

    name = "LSTM"
    family = "deep"
    description = (
        "Two-layer LSTM reading a 20-session window of technical features. "
        "Tests whether temporal structure exists that per-row models cannot see."
    )
    requires_sequences = True

    def __init__(
        self,
        n_classes: int,
        random_seed: int = 42,
        config: LSTMConfig | None = None,
    ) -> None:
        super().__init__(n_classes, random_seed)
        self.config = config or LSTMConfig()
        self._scaler = StandardScaler()
        self._network = None
        self._device = None
        self._history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self._best_epoch = 0

    # -- construction -------------------------------------------------------
    def _build_network(self, n_features: int):
        import torch
        from torch import nn

        class SequenceClassifier(nn.Module):
            def __init__(self, n_features: int, config: LSTMConfig, n_classes: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=config.hidden_size,
                    num_layers=config.num_layers,
                    batch_first=True,
                    # PyTorch applies dropout between layers only, so it is a
                    # no-op with a single layer.
                    dropout=config.dropout if config.num_layers > 1 else 0.0,
                )
                self.dropout = nn.Dropout(config.dropout)
                self.head = nn.Linear(config.hidden_size, n_classes)

            def forward(self, x):
                output, _ = self.lstm(x)
                # Last timestep's hidden state summarises the window.
                return self.head(self.dropout(output[:, -1, :]))

        return SequenceClassifier(n_features, self.config, self.n_classes)

    def _seed_everything(self) -> None:
        import torch

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

    # -- training -----------------------------------------------------------
    def fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        self._seed_everything()
        self._device = torch.device("cpu")  # these models are tiny

        y = np.asarray(y).astype(int)
        raw = x.to_numpy(dtype=float)

        # --- Chronological validation split, BEFORE scaling -----------------
        split = int(len(raw) * (1.0 - self.config.validation_fraction))
        split = max(self.config.sequence_length + 1, min(split, len(raw) - 1))

        # --- Scaler fitted on the training portion only ---------------------
        self._scaler.fit(raw[:split])
        scaled = self._scaler.transform(raw)

        sequences = _build_sequences(scaled, self.config.sequence_length)

        x_train = torch.from_numpy(sequences[:split])
        y_train = torch.from_numpy(y[:split]).long()
        x_val = torch.from_numpy(sequences[split:])
        y_val = torch.from_numpy(y[split:]).long()

        self._network = self._build_network(raw.shape[1]).to(self._device)

        # Class weights so a 40/35/25 split does not collapse to majority-only
        # predictions, which is the degenerate optimum for plain cross-entropy.
        counts = np.bincount(y[:split], minlength=self.n_classes).astype(float)
        weights = np.where(counts > 0, len(y[:split]) / (self.n_classes * np.maximum(counts, 1)), 0.0)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32).to(self._device)
        )
        optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=self.config.batch_size,
            # Shuffles batch composition within the training window only. No
            # information crosses the train/validation or train/test boundary.
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_seed),
        )

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        self._history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.max_epochs):
            self._network.train()
            epoch_loss = 0.0

            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(self._network(batch_x), batch_y)
                loss.backward()
                # Gradient clipping: LSTMs on noisy financial series produce
                # occasional very large gradients that otherwise wreck the run.
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            train_loss = epoch_loss / max(1, len(x_train))

            self._network.eval()
            with torch.no_grad():
                val_loss = (
                    criterion(self._network(x_val), y_val).item()
                    if len(x_val) > 0 else train_loss
                )

            self._history["train_loss"].append(train_loss)
            self._history["val_loss"].append(val_loss)

            # --- Early stopping + best-weight checkpointing -----------------
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._network.state_dict().items()}
                self._best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    logger.debug("Early stopping at epoch %d (best %d)", epoch, self._best_epoch)
                    break

        # Restore the best checkpoint, not the last epoch's weights.
        if best_state is not None:
            self._network.load_state_dict(best_state)

        self._record_fit(x)

    # -- inference ----------------------------------------------------------
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        import torch

        assert self._network is not None, "Model not fitted"

        scaled = self._scaler.transform(x[self._feature_names].to_numpy(dtype=float))
        sequences = _build_sequences(scaled, self.config.sequence_length)

        self._network.eval()
        with torch.no_grad():
            logits = self._network(torch.from_numpy(sequences).to(self._device))
            return torch.softmax(logits, dim=-1).cpu().numpy()

    def feature_importance(self) -> dict[str, float] | None:
        """Not exposed.

        Gradient-based attributions over a recurrent model are not comparable
        to the gain-based importances the tree models report, and presenting
        them side by side in one ranking would be misleading. Returning None
        makes the UI omit the panel rather than invent an explanation.
        """
        return None

    def training_history(self) -> dict[str, object]:
        """Loss curves and the selected epoch, for the transparency panel."""
        return {
            "train_loss": [round(v, 5) for v in self._history["train_loss"]],
            "val_loss": [round(v, 5) for v in self._history["val_loss"]],
            "epochs_run": len(self._history["train_loss"]),
            "best_epoch": self._best_epoch,
            "early_stopped": len(self._history["train_loss"]) < self.config.max_epochs,
            "hyperparameters": self.config.to_dict(),
        }
