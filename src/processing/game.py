"""Game data pipeline: load raw JSON into DuckDB, engineer leak-free features, train, predict.

Feature contract (the core anti-leakage rule):
    Features ``x_i`` for draw ``i`` are computed ONLY from draws ``0..i-1``.
    Target ``y_i`` is the k-dim binary vector of draw ``i``'s numbers.
    The same snapshot logic applied after the final draw yields the inference
    features for the (unknown) next draw.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import deque
from typing import TYPE_CHECKING

import duckdb

from processing.config import (
    DEFAULT_APPROACHES,
    GameType,
    PrizePlan,
    game_file,
    get_game_config,
    get_logger,
)
from processing.evaluate import hit_distribution, hypergeometric_distribution
from processing.train import predict_results, train_results

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

#: Draws skipped at the start of history — their features rest on too little data.
WARMUP_DRAWS = 10
#: Rolling window length for the short-term frequency feature.
WINDOW_SIZE = 50


def resolve_results(game_type: GameType) -> None:
    """Rebuild the game's DuckDB ``results`` table from ``data/<prefix>.json``.

    Draws are ordered by ``drawSystemId`` ascending and assigned a dense ``draw_seq``
    (0..N-1) — all temporal logic uses ``draw_seq`` because draw ids may have gaps.

    Only results whose ``gameType`` matches the requested game are kept — API dumps bundle
    side games (e.g. LottoPlus, SuperSzansa) under the same draw ids.
    """
    game_config = get_game_config(game_type)
    expected_game = game_config["prefix"]

    json_file = game_file(game_type, "json")
    if not json_file.exists():
        msg = f"File {json_file} does not exist"
        raise FileNotFoundError(msg)

    with json_file.open(encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])
    if not items:
        msg = f"File {json_file} is empty"
        raise ValueError(msg)

    draws: dict[int, list[int]] = {}
    found_game_types: set[str] = set()
    for item in items:
        for result in item.get("results", []):
            result_game = result.get("gameType")
            if result_game:
                found_game_types.add(result_game)
            if result_game is not None and result_game != expected_game:
                continue
            draw_id: int = result.get("drawSystemId", 0)
            draw_numbers: list[int] = result.get("resultsJson", []) + result.get("specialResults", [])
            if not draw_numbers or not draw_id:
                continue
            draws[draw_id] = draw_numbers

    if not draws:
        msg = (
            f"File {json_file} contains no draws for gameType '{expected_game}'"
            f" (found: {sorted(found_game_types) or 'none'})"
        )
        raise ValueError(msg)

    rows = [(seq, draw_id, json.dumps(draws[draw_id])) for seq, draw_id in enumerate(sorted(draws))]

    db_file = game_file(game_type, "duckdb")
    with duckdb.connect(db_file) as db:
        db.sql("DROP TABLE IF EXISTS results")
        db.sql(
            """
            CREATE TABLE results (
                draw_seq INTEGER PRIMARY KEY,
                draw_id INTEGER UNIQUE,
                draw_numbers TEXT
            )
            """,
        )
        db.executemany("INSERT INTO results VALUES (?, ?, ?)", rows)

    logger.info("Inserted %s draws into %s", len(rows), db_file)


def _snapshot_features(  # noqa: PLR0913
    i: int,
    k: int,
    counts: Sequence[int],
    last_seen: Sequence[int],
    streak: Sequence[int],
    window: deque[set[int]],
    prev_draw: list[int] | None,
) -> list[float]:
    """Feature vector describing the state BEFORE draw ``i`` (i.e. after draws ``0..i-1``).

    Layout: ``[freq_norm(k), recency_norm(k), streak(k), window_freq(k), prev_sum_norm, prev_odd_ratio]``.
    """
    counts_min, counts_max = min(counts), max(counts)
    divider = (counts_max - counts_min) or 1
    freq_norm = [(c - counts_min) / divider for c in counts]

    recency_norm = [1.0 if last_seen[j] < 0 else (i - 1 - last_seen[j]) / max(i, 1) for j in range(k)]

    streak_f = [float(s) for s in streak]

    window_len = len(window)
    window_freq = [sum(1 for drawn in window if j + 1 in drawn) / window_len if window_len else 0.0 for j in range(k)]

    if prev_draw:
        prev_sum_norm = sum(prev_draw) / (len(prev_draw) * k)
        prev_odd_ratio = sum(1 for v in prev_draw if v % 2) / len(prev_draw)
    else:
        prev_sum_norm = 0.5
        prev_odd_ratio = 0.5

    return freq_norm + recency_norm + streak_f + window_freq + [prev_sum_norm, prev_odd_ratio]


def compute_feature_rows(
    draws: list[list[int]],
    k: int,
    warmup: int = WARMUP_DRAWS,
) -> tuple[list[tuple[int, list[float], list[int]]], list[float]]:
    """Compute leak-free ``(draw_seq, x, y)`` rows plus the inference features for the next draw.

    For each draw ``i`` the state counters are snapshotted FIRST (x_i depends only on
    draws < i) and updated with draw ``i``'s numbers AFTER. The trailing snapshot taken
    once all draws are consumed is the feature vector for the not-yet-drawn next draw.
    """
    counts = [0] * k
    last_seen = [-1] * k
    streak = [0] * k
    window: deque[set[int]] = deque(maxlen=WINDOW_SIZE)

    rows: list[tuple[int, list[float], list[int]]] = []
    prev_draw: list[int] | None = None

    for i, draw_numbers in enumerate(draws):
        x = _snapshot_features(i, k, counts, last_seen, streak, window, prev_draw)
        drawn = set(draw_numbers)
        if i >= warmup:
            y = [1 if j + 1 in drawn else 0 for j in range(k)]
            rows.append((i, x, y))

        for j in range(k):
            if j + 1 in drawn:
                counts[j] += 1
                last_seen[j] = i
                streak[j] += 1
            else:
                streak[j] = 0
        window.append(drawn)
        prev_draw = draw_numbers

    x_next = _snapshot_features(len(draws), k, counts, last_seen, streak, window, prev_draw)
    return rows, x_next


def _load_draws(game_type: GameType) -> list[list[int]]:
    """Load all draws ordered by ``draw_seq`` from the game's DuckDB."""
    db_file = game_file(game_type, "duckdb")
    with duckdb.connect(db_file) as db:
        raw = db.sql("SELECT draw_numbers FROM results ORDER BY draw_seq ASC").fetchall()
    return [json.loads(numbers) for (numbers,) in raw]


def preprocess_results(game_type: GameType) -> None:
    """Compute leak-free features for every draw and persist them in the ``features`` table."""
    game_config = get_game_config(game_type)
    n, k = game_config["n"], game_config["k"]
    db_file = game_file(game_type, "duckdb")

    draws = _load_draws(game_type)
    valid_draws = []
    for seq, draw_numbers in enumerate(draws):
        if len(draw_numbers) < n:
            logger.warning("Skipping draw_seq %s: expected >=%s numbers, got %s", seq, n, len(draw_numbers))
            continue
        valid_draws.append(draw_numbers)

    rows, _ = compute_feature_rows(valid_draws, k)
    if not rows:
        msg = f"Not enough draws to build features (need > {WARMUP_DRAWS})"
        raise ValueError(msg)

    db_rows = [(seq, json.dumps(x), json.dumps(y)) for seq, x, y in rows]
    with duckdb.connect(db_file) as db:
        db.sql("DROP TABLE IF EXISTS features")
        db.sql(
            """
            CREATE TABLE features (
                draw_seq INTEGER PRIMARY KEY,
                x TEXT,
                y TEXT
            )
            """,
        )
        db.executemany("INSERT INTO features VALUES (?, ?, ?)", db_rows)

    logger.info("Stored %s feature rows", len(db_rows))


def _features_exist(game_type: GameType) -> bool:
    """Return whether the game's DuckDB already has a populated ``features`` table."""
    db_file = game_file(game_type, "duckdb")
    if not db_file.exists():
        return False
    with duckdb.connect(db_file) as db:
        found = db.sql("SELECT 1 FROM information_schema.tables WHERE table_name = 'features'").fetchall()
    return bool(found)


def ensure_features(game_type: GameType) -> None:
    """Build the ``features`` table from JSON if it is missing, so train/predict just work."""
    if _features_exist(game_type):
        return
    logger.info("No features for %s yet — building from data/%s.json ...", game_type.value, game_type.value)
    resolve_results(game_type)
    preprocess_results(game_type)


def build_inference_features(game_type: GameType) -> list[float]:
    """Return the feature vector for the next (unknown) draw — state after the final draw."""
    game_config = get_game_config(game_type)
    n, k = game_config["n"], game_config["k"]
    draws = [d for d in _load_draws(game_type) if len(d) >= n]
    _, x_next = compute_feature_rows(draws, k)
    return x_next


def generate_bets(pool: list[int], count: int, size: int) -> list[list[int]]:
    """Generate up to ``count`` deterministic bets from a probability-ordered ``pool``.

    Phase 1 walks the pool top-down in consecutive chunks: bet 1 = the ``size`` best
    numbers, bet 2 = the next ``size``, ... until the pool is exhausted (a final partial
    chunk is completed with the best numbers not already in it). Phase 2 continues with
    the next-best rank combinations in lexicographic order (top size-1 numbers + each
    following one, ...), skipping bets already emitted.

    Numbers inside a bet keep pool (probability) order. No repeated numbers within a bet,
    no repeated combinations. If ``pool`` has fewer than ``size`` numbers, bet size
    shrinks to the pool size. If fewer than ``count`` unique combinations exist, all of
    them are returned.
    """
    size_eff = min(size, len(pool))
    if size_eff == 0 or count < 1:
        return []

    target = min(count, math.comb(len(pool), size_eff))
    bets: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    def add(indices: tuple[int, ...]) -> None:
        if indices not in seen and len(bets) < target:
            seen.add(indices)
            bets.append([pool[i] for i in indices])

    # Phase 1: consecutive top-down chunks until the whole pool is used.
    full_chunks = len(pool) // size_eff
    for chunk in range(full_chunks):
        add(tuple(range(chunk * size_eff, (chunk + 1) * size_eff)))
    if len(pool) % size_eff:
        leftover = set(range(full_chunks * size_eff, len(pool)))
        fillers = (i for i in range(len(pool)) if i not in leftover)
        while len(leftover) < size_eff:
            leftover.add(next(fillers))
        add(tuple(sorted(leftover)))

    # Phase 2: restart from the top with the next-best combinations (lexicographic ranks).
    if len(bets) < target:
        for combo in itertools.combinations(range(len(pool)), size_eff):
            if len(bets) >= target:
                break
            add(combo)

    return bets


def train_game_results(  # noqa: PLR0913
    game_type: GameType,
    hidden_dims: list[int] | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    seed: int = 42,
    patience: int = 10,
    weight_decay: float | None = None,
) -> None:
    """Train the classifier on the game's leak-free features and save model + loss plot."""
    get_game_config(game_type)  # validate
    ensure_features(game_type)

    kwargs = {} if weight_decay is None else {"weight_decay": weight_decay}
    train_results(
        game_file(game_type, "duckdb"),
        game_file(game_type, "pth"),
        game_file(game_type, "png"),
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        patience=patience,
        **kwargs,
    )


def _log_hit_distribution_for_pick(game_type: GameType, pick: int) -> list[int]:
    """Log (and return) the hit histogram for the top-``pick`` predictions vs every actual draw."""
    k = get_game_config(game_type)["k"]
    counts = hit_distribution(game_file(game_type, "pth"), game_file(game_type, "duckdb"), pick)
    random_pct = hypergeometric_distribution(pick, get_game_config(game_type)["n"], k)
    total = sum(counts)

    logger.info("Hit distribution over %s draws (static model top-%s vs actual | random baseline):", total, pick)
    for h, count in enumerate(counts):
        model_pct = 100 * count / total if total else 0.0
        logger.info("  x%-2d: %6d  (%6.3f%% | random %6.3f%%)", h, count, model_pct, 100 * random_pct[h])
    return counts


def log_hit_distribution(game_type: GameType) -> None:
    """Log model hit histograms and, when configured, a theoretical payout per prize plan.

    Replays every draw with the static model (fast, single batched pass). Without ``prizes``
    a single top-n histogram is shown; with ``prizes`` one histogram + payout is shown per plan
    (MultiMulti: bet-5 and bet-10). In-sample, so optimistic vs a walk-forward backtest.
    """
    game_config = get_game_config(game_type)
    prizes = game_config.get("prizes")

    if not prizes:
        _log_hit_distribution_for_pick(game_type, game_config["n"])
        return

    for plan in prizes:
        logger.info("--- Plan: bet %s numbers (stake %g) ---", plan["size"], plan["stake"])
        counts = _log_hit_distribution_for_pick(game_type, plan["size"])
        _log_payout(counts, plan)


def _money(value: float) -> str:
    """Format a money amount without scientific notation (integers shown without decimals)."""
    return f"{value:,.0f}" if value == int(value) else f"{value:,.2f}"


def _log_payout(counts: list[int], plan: PrizePlan) -> None:
    """Log theoretical profit/loss: one ticket (top-``size`` picks) played per draw at the plan's stake."""
    stake, payouts = plan["stake"], plan["payouts"]
    total = sum(counts)

    staked = stake * total
    won = sum(count * payouts[h] for h, count in enumerate(counts))
    net = won - staked

    logger.info("Theoretical payout (1 ticket of %s numbers per draw, stake %s):", plan["size"], _money(stake))
    for h, count in enumerate(counts):
        if payouts[h]:
            logger.info("  x%-2d: %6d draws x %s = +%s", h, count, _money(payouts[h]), _money(count * payouts[h]))
    logger.info(
        "  staked -%s, won +%s  ->  net %s%s over %s draws",
        _money(staked),
        _money(won),
        "+" if net >= 0 else "-",
        _money(abs(net)),
        total,
    )


def predict_game_results(  # noqa: PLR0913
    game_type: GameType,
    target: list[str] | None = None,
    bets_count: int | None = None,
    bets_size: int | None = None,
    approaches: int = DEFAULT_APPROACHES,
    seed: int = 42,
    *,
    histogram: bool = False,
) -> None:
    """Predict the next draw: top-n probabilities, MC-dropout groups, and bets.

    Bets walk ALL k predicted numbers (probability order) in consecutive chunks, then
    continue with the next-best rank combinations — see :func:`generate_bets`. With
    ``histogram=True`` a hit-count distribution is logged below the bets.
    """
    game_config = get_game_config(game_type)
    n, k = game_config["n"], game_config["k"]
    bets_config = game_config["bets"]
    count = bets_count if bets_count is not None else bets_config["count"]
    size = bets_size if bets_size is not None else bets_config["size"]
    ensure_features(game_type)

    x_next = build_inference_features(game_type)
    top, grouped = predict_results(game_file(game_type, "pth"), x_next, n, approaches=approaches, seed=seed)

    target_array = [int(i) for i in target] if target else []

    logger.info("Predicted top %s numbers (number: probability):", k)
    predicted_numbers = [num for num, _ in top[:n]]  # top-n drive target hits and bets
    for num, prob in top:
        marker = " 🎯" if num in target_array else ""
        logger.info("  %2d: %.4f%s", num, prob, marker)
    if target_array:
        hits = sum(1 for i in target_array if i in predicted_numbers)
        logger.info("Target hits in top %s: %s of %s", n, hits, len(target_array))

    logger.info("Predictions:")
    for group_count, numbers in grouped.items():
        label = f"numbers predicted x{group_count}"
        if target_array:
            hits = sum(1 for i in target_array if i in numbers)
            logger.info("%23s: %s 🙈 target hits %s of %s", label, numbers, hits, len(numbers))
        else:
            logger.info("%23s: %s", label, numbers)

    pool = [num for num, _ in top]  # all k numbers, in probability order
    bets = generate_bets(pool, count, size)
    logger.info("Bets (%s x %s numbers from the top %s predicted):", len(bets), min(size, len(pool)), len(pool))
    for i, bet in enumerate(bets, start=1):
        logger.info("  bet %2d: %s", i, bet)

    if histogram:
        log_hit_distribution(game_type)
