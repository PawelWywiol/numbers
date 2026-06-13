"""Backtest prediction quality against historical draws with honest baselines.

Walk-forward mode trains a fresh model per evaluated draw on strictly earlier rows;
features are leak-free by construction (x_i derives only from draws < i), so row
slicing is sufficient. Reports uniform-random and frequency baselines, an exact
binomial significance test, and probability calibration.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from pathlib import Path

from processing.config import GameType, game_file, get_game_config, get_logger
from processing.train import (
    MLP,
    load_data,
    load_model,
    preprocess_data,
    set_seed,
    top_n_from_logits,
)

logger = get_logger(__name__)

CALIBRATION_BUCKETS = 10
HIT_THRESHOLD = 0.5
SIGNIFICANCE_LEVEL = 0.05


class CalibrationBucket(TypedDict):
    """One probability decile: how often numbers with predicted prob in this range actually hit."""

    low: float
    high: float
    count: int
    mean_predicted: float
    observed_rate: float


class EvalReport(TypedDict):
    """Backtest result: averaged hits vs uniform and frequency baselines, with significance."""

    last_n: int
    avg_hits: float
    baseline: float
    freq_baseline: float
    lift: float
    p_value: float
    hits: list[int]
    calibration: list[CalibrationBucket]


def expected_random_hits(n: int, k: int) -> float:
    """Expected overlap between ``n`` uniformly-drawn numbers and ``n`` actual numbers in a pool of ``k``."""
    return n * n / k


def binomial_p_value(successes: int, trials: int, p: float) -> float:
    """Exact two-sided binomial test (sum of outcomes no more likely than the observed one).

    Computed in log-space (``lgamma``) so it stays stable for large ``trials``. Slightly
    conservative for lottery hits, whose per-draw distribution is hypergeometric (smaller
    variance than binomial).
    """
    if trials == 0:
        return 1.0

    def log_pmf(j: int) -> float:
        return (
            math.lgamma(trials + 1)
            - math.lgamma(j + 1)
            - math.lgamma(trials - j + 1)
            + j * math.log(p)
            + (trials - j) * math.log(1 - p)
        )

    observed = log_pmf(successes)
    tolerance = 1e-9
    total = sum(math.exp(lp) for j in range(trials + 1) if (lp := log_pmf(j)) <= observed + tolerance)
    return min(total, 1.0)


def _actual_numbers(target: torch.Tensor) -> set[int]:
    """Decode a k-dim binary target vector back to the set of drawn numbers."""
    return {j + 1 for j, hit in enumerate(target.tolist()) if hit >= HIT_THRESHOLD}


def _frequency_top_n(y_history: torch.Tensor, n: int) -> set[int]:
    """Top ``n`` most frequent numbers in the historical targets (the naive-player baseline)."""
    counts = y_history.sum(dim=0)
    return {int(idx) + 1 for idx in torch.topk(counts, n).indices.tolist()}


def hypergeometric_distribution(pick: int, drawn: int, k: int) -> list[float]:
    """Random baseline: P(exactly ``h`` of your ``pick`` numbers are among ``drawn`` drawn from ``k``).

    Returns probabilities for ``h = 0..pick``. This is what pure chance produces — a model's
    hit histogram is only meaningful compared against it.
    """
    denom = math.comb(k, pick)
    return [math.comb(drawn, h) * math.comb(k - drawn, pick - h) / denom for h in range(pick + 1)]


def hit_distribution(model_file: str | Path, db_file: str | Path, pick: int) -> list[int]:
    """Histogram of correct picks: for every draw, how many of the top-``pick`` predictions were drawn.

    Uses the saved (static) model in a single batched forward pass over all draws — fast, but
    in-sample (the model trained on these draws), so it is optimistic vs a walk-forward backtest.
    Returns a list where index ``h`` holds the number of draws with exactly ``h`` correct picks.
    """
    x, y = preprocess_data(load_data(db_file))
    model = load_model(model_file)
    with torch.no_grad():
        logits = model(x)
    top_idx = torch.topk(logits, pick, dim=1).indices
    hits = torch.gather(y, 1, top_idx).sum(dim=1).to(torch.int64)
    return torch.bincount(hits, minlength=pick + 1).tolist()


def _train_subset(  # noqa: PLR0913
    x: torch.Tensor,
    y: torch.Tensor,
    hidden_dims: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> nn.Module:
    """Train a fresh classifier in-memory on ``(x, y)`` for walk-forward backtesting."""
    model = MLP(x.shape[1], hidden_dims, y.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def _calibration(probs_outcomes: list[tuple[float, float]]) -> list[CalibrationBucket]:
    """Bucket (predicted probability, outcome) pairs into deciles."""
    buckets: list[CalibrationBucket] = []
    for b in range(CALIBRATION_BUCKETS):
        low, high = b / CALIBRATION_BUCKETS, (b + 1) / CALIBRATION_BUCKETS
        is_last = b == CALIBRATION_BUCKETS - 1
        in_bucket = [(p, o) for p, o in probs_outcomes if low <= p < high or (is_last and p == 1.0)]
        if not in_bucket:
            continue
        buckets.append(
            {
                "low": low,
                "high": high,
                "count": len(in_bucket),
                "mean_predicted": sum(p for p, _ in in_bucket) / len(in_bucket),
                "observed_rate": sum(o for _, o in in_bucket) / len(in_bucket),
            },
        )
    return buckets


def evaluate_game(  # noqa: PLR0913
    game_type: GameType,
    last_n: int = 20,
    hidden_dims: list[int] | None = None,
    *,
    retrain: bool = False,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    seed: int = 42,
) -> EvalReport:
    """Backtest the model over the last ``last_n`` draws.

    ``retrain=True`` runs a true walk-forward: for draw *i*, a fresh model is trained on
    rows ``[0, i)`` only. The default static mode scores the saved model (which has seen
    most of these draws during training) — fast sanity check, NOT an honest backtest.
    """
    config = get_game_config(game_type)
    n, k = config["n"], config["k"]
    hidden_dims = hidden_dims or [256, 128]
    set_seed(seed)

    x, y = preprocess_data(load_data(game_file(game_type, "duckdb")))
    total = x.shape[0]
    last_n = min(last_n, total - 1)
    if last_n < 1:
        msg = "Not enough draws to backtest"
        raise ValueError(msg)

    static_model: nn.Module | None = None
    if not retrain:
        logger.warning("Static mode: saved model has seen these draws in training — use --retrain for honest results")
        static_model = load_model(game_file(game_type, "pth"))

    hits: list[int] = []
    freq_hits: list[int] = []
    probs_outcomes: list[tuple[float, float]] = []

    for i in range(total - last_n, total):
        model = _train_subset(x[:i], y[:i], hidden_dims, epochs, batch_size, learning_rate) if retrain else static_model
        assert model is not None  # noqa: S101

        with torch.no_grad():
            logits = model(x[i].unsqueeze(0))
        predicted = {num for num, _ in top_n_from_logits(logits, n)}
        actual = _actual_numbers(y[i])
        hits.append(len(predicted & actual))
        freq_hits.append(len(_frequency_top_n(y[:i], n) & actual))

        probs = torch.sigmoid(logits.flatten()).tolist()
        probs_outcomes.extend(zip(probs, y[i].tolist()))

        if retrain:
            logger.info("Walk-forward draw %s/%s: %s hits", i - (total - last_n) + 1, last_n, hits[-1])

    avg_hits = sum(hits) / len(hits)
    freq_baseline = sum(freq_hits) / len(freq_hits)
    baseline = expected_random_hits(n, k)
    lift = avg_hits / baseline if baseline else 0.0
    p_value = binomial_p_value(sum(hits), last_n * n, n / k)
    calibration = _calibration(probs_outcomes)

    mode = "walk-forward" if retrain else "static (in-sample!)"
    logger.info("Backtest over last %s draws of %s (%s):", last_n, config["prefix"], mode)
    logger.info("  avg hits:           %.3f / %s", avg_hits, n)
    logger.info("  random baseline:    %.3f", baseline)
    logger.info("  frequency baseline: %.3f", freq_baseline)
    logger.info("  lift over random:   %.2fx", lift)
    significance = "(significant)" if p_value < SIGNIFICANCE_LEVEL else "(no signal)"
    logger.info("  binomial p-value:   %.4f %s", p_value, significance)
    for bucket in calibration:
        logger.info(
            "  calib [%.1f-%.1f): n=%5d, predicted %.3f vs observed %.3f",
            bucket["low"],
            bucket["high"],
            bucket["count"],
            bucket["mean_predicted"],
            bucket["observed_rate"],
        )

    return {
        "last_n": last_n,
        "avg_hits": avg_hits,
        "baseline": baseline,
        "freq_baseline": freq_baseline,
        "lift": lift,
        "p_value": p_value,
        "hits": hits,
        "calibration": calibration,
    }
