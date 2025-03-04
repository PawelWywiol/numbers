import json
import os
from enum import Enum
from pathlib import Path
from typing import TypedDict

import duckdb
from dotenv import load_dotenv

from processing.train import predict_results, train_results

load_dotenv()


class GameType(Enum):
    Lotto = "Lotto"
    MultiMulti = "MultiMulti"
    Szybkie600 = "Szybkie600"

    @classmethod
    def from_str(cls, value: str) -> "GameType":
        try:
            return cls(value)
        except ValueError:
            msg = f"Invalid Mode: {value}"
            raise ValueError(msg) from None


class GameConfig(TypedDict):
    prefix: str
    n: int
    k: int


GAMES_CONFIG: dict[GameType, GameConfig] = {
    GameType.Lotto: {
        "prefix": "Lotto",
        "n": 6,
        "k": 49,
    },
    GameType.MultiMulti: {
        "prefix": "MultiMulti",
        "n": 20,
        "k": 80,
    },
    GameType.Szybkie600: {
        "prefix": "Szybkie600",
        "n": 6,
        "k": 32,
    },
}

DATA_DIR = Path(os.getenv("DATA_DIR", "data/")).resolve()


def resolve_results(game_type: GameType) -> None:
    if not isinstance(game_type, GameType):
        msg = f"Expected GameMode, got {type(game_type).__name__}"
        raise TypeError(msg)

    game_config = GAMES_CONFIG.get(game_type)
    if not game_config:
        msg = f"Game {game_type.value} is not supported"
        raise ValueError(msg)

    game_prefix = game_config.get("prefix")

    json_file = DATA_DIR / f"{game_prefix}.json"
    if not json_file.exists():
        msg = f"File {json_file} does not exist"
        raise FileNotFoundError(msg)

    with Path.open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])
    if not items:
        msg = f"File {json_file} is empty"
        raise ValueError(msg)

    db_file = (DATA_DIR / f"{game_prefix}.duckdb").resolve()

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

                if draw_id % 1000 == 0:
                    print(f"Inserted {draw_id} draws")  # noqa: T201


def preprocess_results(game_type: GameType) -> None:
    if not isinstance(game_type, GameType):
        msg = f"Expected GameMode, got {type(game_type).__name__}"
        raise TypeError(msg)

    game_config = GAMES_CONFIG.get(game_type)
    if not game_config:
        msg = f"Game {game_type.value} is not supported"
        raise ValueError(msg)

    game_prefix = game_config.get("prefix")
    db_file = (DATA_DIR / f"{game_prefix}.duckdb").resolve()

    distribution = [0] * game_config.get("k")
    last_draw_id = [0] * game_config.get("k")
    step = [0] * game_config.get("k")
    repeats = [0] * game_config.get("k")

    with duckdb.connect(db_file) as db:
        results = db.sql(
            "SELECT draw_id, draw_numbers FROM results ORDER BY draw_id ASC",
        ).fetchall()

        for draw_id, draw_numbers_str in results:
            draw_numbers: list[int] = json.loads(draw_numbers_str)

            for number in draw_numbers:
                distribution[number - 1] += 1
                step[number - 1] = draw_id - last_draw_id[number - 1]
                last_draw_id[number - 1] = draw_id

            distribution_min = min(distribution)
            distribution_max = max(distribution)

            for i in range(game_config.get("k")):
                if last_draw_id[i] == draw_id:
                    repeats[i] += 1
                else:
                    repeats[i] = 0
                    step[i] = draw_id - last_draw_id[i]

            _distribution = [0] * game_config.get("n")
            _step = [0] * game_config.get("n")
            _repeats = [0] * game_config.get("n")

            for i in range(game_config.get("n")):
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

            if draw_id % 1000 == 0:
                print(f"Processed {draw_id} draws")  # noqa: T201

        db.sql("""
            SELECT * FROM results
            ORDER BY draw_id DESC
            """).show()


def train_game_results(game_type: GameType) -> None:
    if not isinstance(game_type, GameType):
        msg = f"Expected GameMode, got {type(game_type).__name__}"
        raise TypeError(msg)

    game_config = GAMES_CONFIG.get(game_type)
    if not game_config:
        msg = f"Game {game_type.value} is not supported"
        raise ValueError(msg)

    game_prefix = game_config.get("prefix")
    db_file = (DATA_DIR / f"{game_prefix}.duckdb").resolve()
    model_file = (DATA_DIR / f"{game_prefix}.pth").resolve()
    plot_file = (DATA_DIR / f"{game_prefix}.png").resolve()

    train_results(db_file, model_file, plot_file)


def predict_game_results(game_type: GameType, target: str) -> None:
    if not isinstance(game_type, GameType):
        msg = f"Expected GameMode, got {type(game_type).__name__}"
        raise TypeError(msg)

    game_config = GAMES_CONFIG.get(game_type)
    if not game_config:
        msg = f"Game {game_type.value} is not supported"
        raise ValueError(msg)

    game_prefix = game_config.get("prefix")
    n = game_config.get("n")
    k = game_config.get("k")
    db_file = (DATA_DIR / f"{game_prefix}.duckdb").resolve()
    model_file = (DATA_DIR / f"{game_prefix}.pth").resolve()

    print("Predictions:")  # noqa: T201

    target_array = [int(i) for i in target] if target else []
    hits = 0

    predictions = predict_results(db_file, model_file, 100, n, k)
    for count, prediction in predictions.items():
        if target_array:
            hits = sum(1 for i in target_array if i in prediction)
        print(f"{'x' + str(count):>4}: {prediction}, {hits}/{len(prediction)}")  # noqa: T201
