"""Integration test for feature preprocessing, incl. malformed-draw bounds checking."""

import json

import duckdb

from processing import config, game
from processing.config import GameType

_NUMBERS_VALID = [1, 2, 3, 4, 5, 6]  # matches Szybkie600 n=6
_NUMBERS_SHORT = [1, 2, 3]  # malformed: fewer than n


def _seed_db(path: str) -> None:
    with duckdb.connect(path) as db:
        db.sql(
            """
            CREATE TABLE results (
                draw_id INTEGER PRIMARY KEY,
                draw_numbers TEXT,
                distribution TEXT,
                step TEXT,
                repeats TEXT
            )
            """,
        )
        db.execute("INSERT INTO results (draw_id, draw_numbers) VALUES (?, ?)", (1, json.dumps(_NUMBERS_VALID)))
        db.execute("INSERT INTO results (draw_id, draw_numbers) VALUES (?, ?)", (2, json.dumps(_NUMBERS_SHORT)))


def test_preprocess_skips_malformed_draws(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    db_path = str(tmp_path / "Szybkie600.duckdb")
    _seed_db(db_path)

    game.preprocess_results(GameType.Szybkie600)

    with duckdb.connect(db_path) as db:
        rows = dict(db.sql("SELECT draw_id, distribution FROM results ORDER BY draw_id").fetchall())

    assert rows[1] is not None  # valid draw got features
    assert rows[2] is None  # malformed draw was skipped
    assert len(json.loads(rows[1])) == 6
