"""Central configuration: game definitions, filesystem paths, constants, logging.

Game parameters (``n`` draws from a pool of ``k``) live in ``games.json`` so they can
be tuned without touching code. The :class:`GameType` enum stays the static, type-checked
source of truth for *which* games exist; ``games.json`` keys must match its members.
"""

import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data/")).resolve()
GAMES_FILE = Path(__file__).parent / "games.json"

#: How often progress is logged while inserting/processing draws.
LOG_INTERVAL = 1000
#: Number of stochastic forward passes aggregated into a single prediction.
DEFAULT_APPROACHES = 100


class GameType(Enum):
    """Supported lottery games."""

    Lotto = "Lotto"
    MultiMulti = "MultiMulti"
    Szybkie600 = "Szybkie600"

    @classmethod
    def from_str(cls, value: str) -> "GameType":
        """Return the :class:`GameType` for ``value`` or raise ``ValueError`` with valid choices."""
        try:
            return cls(value)
        except ValueError:
            choices = ", ".join(member.value for member in cls)
            msg = f"Invalid game '{value}'. Valid choices: {choices}"
            raise ValueError(msg) from None


class GameConfig(TypedDict):
    """Parameters defining a game: file ``prefix``, ``n`` numbers drawn from a pool of ``k``."""

    prefix: str
    n: int
    k: int


def _load_games_config() -> dict[GameType, GameConfig]:
    """Load and validate ``games.json`` into a ``{GameType: GameConfig}`` mapping."""
    with GAMES_FILE.open(encoding="utf-8") as f:
        raw: dict[str, GameConfig] = json.load(f)

    valid = {member.value for member in GameType}
    config: dict[GameType, GameConfig] = {}
    for name, cfg in raw.items():
        if name not in valid:
            msg = f"games.json defines unknown game '{name}'; add it to GameType first"
            raise ValueError(msg)
        config[GameType(name)] = cfg
    return config


GAMES_CONFIG: dict[GameType, GameConfig] = _load_games_config()


def get_game_config(game_type: GameType) -> GameConfig:
    """Return the :class:`GameConfig` for ``game_type``.

    Raises ``TypeError`` if ``game_type`` is not a :class:`GameType`, ``ValueError`` if unsupported.
    """
    if not isinstance(game_type, GameType):
        msg = f"Expected GameType, got {type(game_type).__name__}"
        raise TypeError(msg)

    config = GAMES_CONFIG.get(game_type)
    if not config:
        msg = f"Game {game_type.value} is not supported"
        raise ValueError(msg)
    return config


def game_file(game_type: GameType, ext: str) -> Path:
    """Return the resolved ``data/<prefix>.<ext>`` path for ``game_type``."""
    prefix = get_game_config(game_type)["prefix"]
    return (DATA_DIR / f"{prefix}.{ext}").resolve()


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for CLI runs (no-op if already configured)."""
    logging.basicConfig(level=level, format="%(message)s")


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)
