"""Multi-label classifier: k sigmoid outputs = P(number j appears in the next draw).

Training uses a temporal train/val split (no shuffling across the time boundary),
early stopping on validation loss, and a fully seeded, reproducible pipeline.
Checkpoints carry their architecture + training metadata so loaders never guess.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import duckdb
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from processing.config import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

logger = get_logger(__name__)

DEFAULT_HIDDEN_DIMS = [256, 128]
DEFAULT_DROPOUT = 0.2
#: L2 regularization — calibrated on MultiMulti so val loss stays flat instead of rising
#: (sweep result: best val 0.562633 vs theoretical floor 0.562335, rise +0.0002 over 40 epochs).
DEFAULT_WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.15


def set_seed(seed: int) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)


class MLP(nn.Module):
    """Configurable feed-forward net; outputs k raw logits (no sigmoid — use BCEWithLogitsLoss)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev, dim), nn.ReLU(), nn.Dropout(dropout)])
            prev = dim
        layers.append(nn.Linear(prev, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def load_data(db_file: str | Path) -> pd.DataFrame:
    """Load leak-free feature rows ordered by draw sequence."""
    query = "SELECT draw_seq, x, y FROM features ORDER BY draw_seq ASC"
    with duckdb.connect(db_file) as conn:
        try:
            results = conn.execute(query).df()
        except duckdb.CatalogException as exc:
            msg = f"No 'features' table in {db_file} — run the 'update' command first to build features"
            raise ValueError(msg) from exc

    for col in ["x", "y"]:
        results[col] = results[col].apply(json.loads)
    return results


def preprocess_data(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert feature rows to ``(x, y)`` float tensors with basic shape validation."""
    if df.empty:
        msg = "No feature rows loaded — run the 'update' command first"
        raise ValueError(msg)

    x = torch.tensor(df["x"].tolist(), dtype=torch.float32)
    y = torch.tensor(df["y"].tolist(), dtype=torch.float32)

    if x.shape[0] != y.shape[0]:
        msg = f"Feature/target row mismatch: {x.shape[0]} != {y.shape[0]}"
        raise ValueError(msg)
    return x, y


def top_n_from_logits(logits: torch.Tensor, n: int) -> list[tuple[int, float]]:
    """Return the ``n`` most probable numbers as ``(number, probability)``, highest first."""
    probs = torch.sigmoid(logits.flatten())
    top_probs, top_indices = torch.topk(probs, min(n, probs.shape[0]))
    return [(int(idx) + 1, float(prob)) for idx, prob in zip(top_indices.tolist(), top_probs.tolist())]


def _data_hash(x: torch.Tensor, y: torch.Tensor) -> str:
    """Short content hash of the training data for checkpoint provenance."""
    digest = hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes())
    return digest.hexdigest()[:16]


def _epoch_val_loss(model: nn.Module, criterion: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float(criterion(model(x), y).item())


def train_results(  # noqa: PLR0913, PLR0915
    db_file: str | Path,
    model_file: str | Path,
    plot_file: str | Path,
    hidden_dims: list[int] | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    seed: int = 42,
    patience: int = 10,
    dropout: float = DEFAULT_DROPOUT,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> None:
    """Train with a temporal split (first 85% train, last 15% val) and early stopping.

    Early stopping guards against overfitting: once val loss stops improving for
    ``patience`` epochs, further training only memorizes training noise, so we stop and
    keep the best-validation weights. ``patience <= 0`` disables it — training runs all
    ``epochs`` and the FINAL weights are saved instead of the best-validation ones.

    Saves a checkpoint dict carrying the architecture and training metadata, plus a
    train/val loss plot.
    """
    hidden_dims = hidden_dims or list(DEFAULT_HIDDEN_DIMS)
    set_seed(seed)

    x, y = preprocess_data(load_data(db_file))

    split = int(x.shape[0] * (1 - VAL_FRACTION))
    if split < 1 or split >= x.shape[0]:
        msg = f"Not enough rows ({x.shape[0]}) for a train/val split"
        raise ValueError(msg)
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    input_dim, output_dim = x.shape[1], y.shape[1]
    model = MLP(input_dim, hidden_dims, output_dim, dropout=dropout)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    early_stopping = patience > 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_state = {key: value.clone() for key, value in model.state_dict().items()}
    epochs_without_improvement = 0
    early_stopped = False

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_losses.append(sum(batch_losses) / len(batch_losses))
        val_losses.append(_epoch_val_loss(model, criterion, x_val, y_val))

        logger.info(
            "Epoch %s/%s, train loss: %.6f, val loss: %.6f",
            epoch + 1,
            epochs,
            train_losses[-1],
            val_losses[-1],
        )

        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if early_stopping and epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %s (best val loss: %.6f)", epoch + 1, best_val)
                early_stopped = True
                break

    # With early stopping we keep the best-validation weights; without it ("train to the
    # end") we keep the final-epoch weights, as explicitly requested via patience <= 0.
    saved_state = best_state if early_stopping else model.state_dict()
    saved_val = best_val if early_stopping else val_losses[-1]

    checkpoint = {
        "state_dict": saved_state,
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "output_dim": output_dim,
        "dropout": dropout,
        "seed": seed,
        "data_hash": _data_hash(x, y),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_loss": train_losses[-1],
        "val_loss": saved_val,
        "epochs_run": len(train_losses),
        "early_stopped": early_stopped,
    }
    torch.save(checkpoint, model_file)
    logger.info(
        "Model trained and saved as '%s' (%s epochs, saved val loss: %.6f)",
        model_file,
        len(train_losses),
        saved_val,
    )

    # Theoretical floor: BCE of always predicting the base rate. For a fair lottery no
    # model can do better in expectation — the val curve should sit just above this line.
    base_rate = float(y.mean())
    floor = -(base_rate * math.log(base_rate) + (1 - base_rate) * math.log(1 - base_rate))

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.axhline(floor, color="gray", linestyle="--", label=f"Theoretical floor ({floor:.4f})")
    plt.xlabel("Epochs")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.title("Train vs Validation Loss")
    plt.savefig(plot_file)
    plt.close()


def load_model(model_file: str | Path) -> nn.Module:
    """Rebuild a model from a metadata-carrying checkpoint."""
    checkpoint = torch.load(model_file, weights_only=True)
    model = MLP(
        checkpoint["input_dim"],
        checkpoint["hidden_dims"],
        checkpoint["output_dim"],
        dropout=checkpoint.get("dropout", DEFAULT_DROPOUT),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict_results(
    model_file: str | Path,
    features: list[float],
    n: int,
    approaches: int = 100,
    seed: int = 42,
) -> tuple[list[tuple[int, float]], dict[int, list[int]]]:
    """Predict the next draw: deterministic top ``n`` with probabilities + MC-dropout groups.

    The grouped view runs ``approaches`` stochastic forward passes with dropout enabled
    (Monte Carlo dropout — a standard uncertainty estimate): numbers that make the top ``n``
    in every pass land in the ``x{approaches}`` group, unstable ones in lower groups.
    Returns ``(all_k_numbers, {count: [numbers]})``: the first element ranks ALL k numbers
    by probability (descending); groups are ordered by count descending.
    """
    model = load_model(model_file)
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        deterministic_top = top_n_from_logits(logits, logits.numel())  # full k-number ranking

    set_seed(seed)
    model.train()  # enable dropout for stochastic passes
    counts: dict[int, int] = {}
    with torch.no_grad():
        for _ in range(approaches):
            for num, _prob in top_n_from_logits(model(x), n):
                counts[num] = counts.get(num, 0) + 1
    model.eval()

    grouped: dict[int, list[int]] = {}
    for num in sorted(counts, key=lambda num: (-counts[num], num)):
        grouped.setdefault(counts[num], []).append(num)

    return deterministic_top, grouped
