"""Backtest prediction quality against historical draws and compare to a random baseline."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from processing.config import GameType, game_file, get_game_config, get_logger
from processing.train import MLP, load_data, predict_next_draw, preprocess_data, process_predictions

logger = get_logger(__name__)


class EvalReport(TypedDict):
    """Result of a backtest: averaged hits vs the uniform-random expectation."""

    last_n: int
    avg_hits: float
    baseline: float
    lift: float
    hits: list[int]


def expected_random_hits(n: int, k: int) -> float:
    """Expected overlap between ``n`` uniformly-drawn numbers and ``n`` actual numbers in a pool of ``k``."""
    return n * n / k


def _actual_numbers(target: torch.Tensor) -> set[int]:
    """Round a draw-numbers tensor back to a set of integers."""
    return {int(round(v)) for v in target.tolist()}


def _train_subset(  # noqa: PLR0913
    x: torch.Tensor,
    y: torch.Tensor,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> nn.Module:
    """Train a fresh MLP in-memory on ``(x, y)`` for walk-forward backtesting."""
    model = MLP(x.shape[1], hidden_dim, y.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    return model


def evaluate_game(  # noqa: PLR0913
    game_type: GameType,
    last_n: int = 20,
    hidden_dim: int = 128,
    *,
    retrain: bool = False,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> EvalReport:
    """Backtest the model over the last ``last_n`` draws.

    Default (``retrain=False``) loads the saved model and scores its reconstruction of each draw.
    ``retrain=True`` runs a true walk-forward: for draw *i*, a fresh model is trained on draws ``[0, i)``.
    """
    config = get_game_config(game_type)
    n, k = config["n"], config["k"]

    results = load_data(game_file(game_type, "duckdb"))
    x, y = preprocess_data(results)
    total = x.shape[0]
    last_n = min(last_n, total - 1)
    if last_n < 1:
        msg = "Not enough draws to backtest"
        raise ValueError(msg)

    static_model: nn.Module | None = None
    if not retrain:
        static_model = MLP(x.shape[1], hidden_dim, y.shape[1])
        static_model.load_state_dict(torch.load(game_file(game_type, "pth"), weights_only=True))

    hits: list[int] = []
    for i in range(total - last_n, total):
        if retrain:
            model = _train_subset(x[:i], y[:i], hidden_dim, epochs, batch_size, learning_rate)
            logger.info("Walk-forward trained on %s draws for draw index %s", i, i)
        else:
            model = static_model  # type: ignore[assignment]

        predicted = set(process_predictions(predict_next_draw(model, x[i]), n, k))
        actual = _actual_numbers(y[i])
        hits.append(len(predicted & actual))

    avg_hits = sum(hits) / len(hits)
    baseline = expected_random_hits(n, k)
    lift = avg_hits / baseline if baseline else 0.0

    mode = "walk-forward" if retrain else "static"
    logger.info("Backtest over last %s draws of %s (%s):", last_n, config["prefix"], mode)
    logger.info("  avg hits:        %.3f / %s", avg_hits, n)
    logger.info("  random baseline: %.3f", baseline)
    logger.info("  lift over random: %.2fx", lift)

    return {"last_n": last_n, "avg_hits": avg_hits, "baseline": baseline, "lift": lift, "hits": hits}
