"""Game data pipeline: load raw JSON into DuckDB, engineer features, train, and predict."""

from __future__ import annotations

import json

import duckdb

from processing.config import (
    DEFAULT_APPROACHES,
    LOG_INTERVAL,
    GameType,
    game_file,
    get_game_config,
    get_logger,
)
from processing.train import predict_results, train_results

logger = get_logger(__name__)


def resolve_results(game_type: GameType) -> None:
    """Load ``data/<prefix>.json`` draw results into the game's DuckDB ``results`` table."""
    get_game_config(game_type)  # validate

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

    db_file = game_file(game_type, "duckdb")

    with duckdb.connect(db_file) as db:
        db.sql(
            """
            CREATE TABLE IF NOT EXISTS results (
                draw_id INTEGER PRIMARY KEY,
                draw_numbers TEXT,
                distribution TEXT,
                step TEXT,
                repeats TEXT
            )
            """,
        )

        for item in items:
            results = item.get("results", [])
            if not results:
                continue

            for result in results:
                draw_id: int = result.get("drawSystemId", 0)
                draw_numbers: list[int] = result.get("resultsJson", []) + result.get("specialResults", [])

                if not draw_numbers or not draw_id:
                    continue

                db.execute(
                    """
                    INSERT OR REPLACE INTO results (draw_id, draw_numbers)
                    VALUES (?, ?)
                    """,
                    (draw_id, json.dumps(draw_numbers)),
                )

                if draw_id % LOG_INTERVAL == 0:
                    logger.info("Inserted %s draws", draw_id)


def preprocess_results(game_type: GameType) -> None:
    """Compute distribution/step/repeats features for every draw and persist them in DuckDB."""
    game_config = get_game_config(game_type)
    n = game_config["n"]
    k = game_config["k"]
    db_file = game_file(game_type, "duckdb")

    distribution = [0] * k
    last_draw_id = [0] * k
    step = [0] * k
    repeats = [0] * k

    with duckdb.connect(db_file) as db:
        results = db.sql(
            "SELECT draw_id, draw_numbers FROM results ORDER BY draw_id ASC",
        ).fetchall()

        for draw_id, draw_numbers_str in results:
            draw_numbers: list[int] = json.loads(draw_numbers_str)

            if len(draw_numbers) < n:
                logger.warning("Skipping draw %s: expected >=%s numbers, got %s", draw_id, n, len(draw_numbers))
                continue

            for number in draw_numbers:
                distribution[number - 1] += 1
                step[number - 1] = draw_id - last_draw_id[number - 1]
                last_draw_id[number - 1] = draw_id

            distribution_min = min(distribution)
            distribution_max = max(distribution)

            for i in range(k):
                if last_draw_id[i] == draw_id:
                    repeats[i] += 1
                else:
                    repeats[i] = 0
                    step[i] = draw_id - last_draw_id[i]

            _distribution = [0.0] * n
            _step = [0] * n
            _repeats = [0] * n

            for i in range(n):
                number = draw_numbers[i]
                divider = distribution_max - distribution_min or 1
                _distribution[i] = (distribution[number - 1] - distribution_min) / divider
                _step[i] = step[number - 1] - 1
                _repeats[i] = repeats[number - 1] - 1

            db.execute(
                """
                UPDATE results
                SET distribution = ?,
                    step = ?,
                    repeats = ?
                WHERE draw_id = ?
                """,
                (
                    json.dumps(_distribution),
                    json.dumps(_step),
                    json.dumps(_repeats),
                    draw_id,
                ),
            )

            if draw_id % LOG_INTERVAL == 0:
                logger.info("Processed %s draws", draw_id)


def train_game_results(
    game_type: GameType,
    hidden_dim: int = 128,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> None:
    """Train the MLP on the game's preprocessed features and save model + loss plot."""
    get_game_config(game_type)  # validate

    train_results(
        game_file(game_type, "duckdb"),
        game_file(game_type, "pth"),
        game_file(game_type, "png"),
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


def predict_game_results(
    game_type: GameType,
    target: list[str] | None = None,
    approaches: int = DEFAULT_APPROACHES,
) -> None:
    """Predict the next draw and log numbers grouped by frequency, with target hits if provided."""
    game_config = get_game_config(game_type)
    n = game_config["n"]
    k = game_config["k"]
    db_file = game_file(game_type, "duckdb")
    model_file = game_file(game_type, "pth")

    logger.info("Predictions:")

    target_array = [int(i) for i in target] if target else []

    predictions = predict_results(db_file, model_file, approaches, n, k)
    for count, prediction in predictions.items():
        label = f"numbers predicted x{count}"
        if target_array:
            hits = sum(1 for i in target_array if i in prediction)
            logger.info("%23s: %s 🙈 target hits %s of %s", label, prediction, hits, len(prediction))
        else:
            logger.info("%23s: %s", label, prediction)
