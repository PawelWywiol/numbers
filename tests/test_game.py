"""Tests for the leak-free feature builder and bet generation."""

import json
import random

import duckdb
import pytest

from processing import config, game
from processing.config import GameType
from processing.game import compute_feature_rows, generate_bets

K = 8  # small synthetic pool


def _draws(count: int, seed: int = 1) -> list[list[int]]:
    rng = random.Random(seed)
    return [sorted(rng.sample(range(1, K + 1), 3)) for _ in range(count)]


def test_feature_rows_shapes_and_targets() -> None:
    draws = _draws(15)
    rows, x_next = compute_feature_rows(draws, K, warmup=10)

    assert len(rows) == 5  # 15 draws - 10 warmup
    seq, x, y = rows[0]
    assert seq == 10
    assert len(x) == 4 * K + 2  # freq, recency, streak, window + 2 scalars
    assert len(y) == K
    assert sum(y) == 3  # 3 numbers per synthetic draw
    assert all(y[j - 1] == 1 for j in draws[10])
    assert len(x_next) == 4 * K + 2


def test_no_leakage_appending_draw_keeps_prior_features() -> None:
    draws = _draws(20)
    rows_before, _ = compute_feature_rows(draws, K, warmup=10)
    rows_after, _ = compute_feature_rows([*draws, [1, 2, 3]], K, warmup=10)

    for (seq_a, x_a, y_a), (seq_b, x_b, y_b) in zip(rows_before, rows_after):
        assert seq_a == seq_b
        assert x_a == x_b  # byte-identical: features never depend on later draws
        assert y_a == y_b


def test_no_leakage_features_invariant_to_own_draw() -> None:
    draws = _draws(15)
    rows_original, _ = compute_feature_rows(draws, K, warmup=10)

    mutated = [list(d) for d in draws]
    mutated[12] = [6, 7, 8]  # replace draw 12's numbers entirely
    rows_mutated, _ = compute_feature_rows(mutated, K, warmup=10)

    x_original = next(x for seq, x, _ in rows_original if seq == 12)
    x_mutated = next(x for seq, x, _ in rows_mutated if seq == 12)
    assert x_original == x_mutated  # x_12 must not encode draw 12's outcome


def test_inference_features_extend_history() -> None:
    draws = _draws(15)
    _, x_next = compute_feature_rows(draws, K, warmup=10)
    rows_extended, _ = compute_feature_rows([*draws, [1, 2, 3]], K, warmup=10)

    # x_next == features of the (now appended) draw 15 in the extended history
    x_15 = next(x for seq, x, _ in rows_extended if seq == 15)
    assert x_next == x_15


def test_resolve_results_filters_by_game_type(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    payload = {
        "items": [
            {
                "results": [
                    {"drawSystemId": 1, "gameType": "Szybkie600", "resultsJson": [1, 2, 3, 4, 5, 6]},
                    {"drawSystemId": 1, "gameType": "OtherGame", "resultsJson": [9, 9, 9, 9, 9, 9, 9]},
                ],
            },
            {
                "results": [
                    {"drawSystemId": 2, "gameType": "Szybkie600", "resultsJson": [7, 8, 9, 10, 11, 12]},
                ],
            },
        ],
    }
    (tmp_path / "Szybkie600.json").write_text(json.dumps(payload), encoding="utf-8")

    game.resolve_results(GameType.Szybkie600)

    with duckdb.connect(str(tmp_path / "Szybkie600.duckdb")) as db:
        rows = db.sql("SELECT draw_seq, draw_id, draw_numbers FROM results ORDER BY draw_seq").fetchall()

    assert len(rows) == 2  # OtherGame result for draw 1 was NOT mixed in
    assert json.loads(rows[0][2]) == [1, 2, 3, 4, 5, 6]
    assert json.loads(rows[1][2]) == [7, 8, 9, 10, 11, 12]


def test_resolve_results_rejects_wrong_game_dump(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    payload = {
        "items": [
            {"results": [{"drawSystemId": 1, "gameType": "Lotto", "resultsJson": [1, 2, 3, 4, 5, 6]}]},
        ],
    }
    (tmp_path / "Szybkie600.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="no draws for gameType 'Szybkie600'.*Lotto"):
        game.resolve_results(GameType.Szybkie600)


def test_preprocess_skips_malformed_draws(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    db_path = str(tmp_path / "Szybkie600.duckdb")

    rng = random.Random(7)
    with duckdb.connect(db_path) as db:
        db.sql("CREATE TABLE results (draw_seq INTEGER PRIMARY KEY, draw_id INTEGER UNIQUE, draw_numbers TEXT)")
        for seq in range(12):
            numbers = sorted(rng.sample(range(1, 33), 6))
            db.execute("INSERT INTO results VALUES (?, ?, ?)", (seq, seq + 1, json.dumps(numbers)))
        db.execute("INSERT INTO results VALUES (?, ?, ?)", (12, 13, json.dumps([1, 2, 3])))  # malformed
        for seq in range(13, 16):
            numbers = sorted(rng.sample(range(1, 33), 6))
            db.execute("INSERT INTO results VALUES (?, ?, ?)", (seq, seq + 1, json.dumps(numbers)))

    game.preprocess_results(GameType.Szybkie600)

    with duckdb.connect(db_path) as db:
        count = db.sql("SELECT COUNT(*) FROM features").fetchone()[0]

    # 16 draws - 1 malformed - 10 warmup = 5 feature rows
    assert count == 5


#: Probability-ordered pool from the user's example (top 20 MultiMulti predictions).
_POOL = [39, 49, 76, 36, 71, 30, 18, 4, 78, 59, 6, 34, 12, 23, 22, 9, 55, 47, 48, 69]


def test_generate_bets_phase1_consecutive_chunks() -> None:
    bets = generate_bets(_POOL, count=4, size=5)

    assert bets == [
        [39, 49, 76, 36, 71],  # ranks 1-5
        [30, 18, 4, 78, 59],  # ranks 6-10
        [6, 34, 12, 23, 22],  # ranks 11-15
        [9, 55, 47, 48, 69],  # ranks 16-20
    ]


def test_generate_bets_phase2_wraps_with_next_best_combinations() -> None:
    bets = generate_bets(_POOL, count=7, size=5)

    # after the 4 chunks, restart from the top: top-4 + each next rank, skipping used combos
    assert bets[4] == [39, 49, 76, 36, 30]  # ranks 1,2,3,4,6
    assert bets[5] == [39, 49, 76, 36, 18]  # ranks 1,2,3,4,7
    assert bets[6] == [39, 49, 76, 36, 4]  # ranks 1,2,3,4,8


def test_generate_bets_unique_and_within_pool() -> None:
    bets = generate_bets(_POOL, count=50, size=5)

    assert len(bets) == 50
    assert len({tuple(sorted(bet)) for bet in bets}) == 50  # no repeated combination
    for bet in bets:
        assert len(set(bet)) == 5  # no repeated numbers within a bet
        assert all(num in _POOL for num in bet)


def test_generate_bets_partial_chunk_filled_from_top() -> None:
    bets = generate_bets([10, 20, 30, 40, 50, 60, 70], count=2, size=5)

    assert bets[0] == [10, 20, 30, 40, 50]  # ranks 1-5
    assert bets[1] == [10, 20, 30, 60, 70]  # leftover ranks 6-7 + best fillers 1-3


def test_generate_bets_pool_smaller_than_size() -> None:
    # only one unique 3-number combination exists in a 3-number pool
    assert generate_bets([3, 7, 11], count=8, size=5) == [[3, 7, 11]]


def test_generate_bets_caps_at_possible_combinations() -> None:
    bets = generate_bets([1, 2, 3, 4], count=8, size=3)

    assert len(bets) == 4  # C(4,3) = 4 unique combinations
    assert len({tuple(bet) for bet in bets}) == 4


def test_generate_bets_empty_pool() -> None:
    assert generate_bets([], count=8, size=5) == []


@pytest.mark.parametrize("count", [0, 1, 3])
def test_generate_bets_respects_count(count: int) -> None:
    assert len(generate_bets(list(range(1, 11)), count=count, size=4)) == count
