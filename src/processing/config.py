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
#: Number of MC-dropout forward passes aggregated into the grouped prediction view.
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


class BetsConfig(TypedDict):
    """Bet generation: ``count`` bets of ``size`` numbers sampled from the top prediction group."""

    count: int
    size: int


class PrizePlan(TypedDict):
    """One ticket plan: bet ``size`` numbers at ``stake``; ``payouts[h]`` won at ``h`` hits (len size+1)."""

    size: int
    stake: float
    payouts: list[float]


class _GameConfigRequired(TypedDict):
    prefix: str
    n: int
    k: int
    bets: BetsConfig


class GameConfig(_GameConfigRequired, total=False):
    """Parameters defining a game: file ``prefix``, ``n`` numbers from a pool of ``k``, ``bets``, ``prizes``.

    ``prizes`` is a list of plans — one payout summary is produced per plan (MultiMulti has two:
    bet-5 and bet-10; Szybkie600/Lotto have one).
    """

    prizes: list[PrizePlan]


def _validate_bets(name: str, cfg: GameConfig) -> None:
    """Validate the ``bets`` section of a game config."""
    bets = cfg.get("bets")
    if not bets:
        msg = f"games.json game '{name}' is missing the 'bets' section ({{count, size[, pool]}})"
        raise ValueError(msg)

    count, size = bets.get("count", 0), bets.get("size", 0)
    if count < 1 or size < 1:
        msg = f"games.json game '{name}': bets count and size must be >= 1"
        raise ValueError(msg)
    if size > cfg["k"]:
        msg = f"games.json game '{name}': bets size must be <= k ({size} <= {cfg['k']})"
        raise ValueError(msg)


def _validate_prizes(name: str, cfg: GameConfig) -> None:
    """Validate the optional ``prizes`` list: each plan's payouts must cover 0..size hits."""
    prizes = cfg.get("prizes")
    if prizes is None:
        return

    for plan in prizes:
        size, stake, payouts = plan.get("size"), plan.get("stake"), plan.get("payouts")
        if not size or not (1 <= size <= cfg["k"]):
            msg = f"games.json game '{name}': prize plan size must be 1..k ({cfg['k']})"
            raise ValueError(msg)
        if stake is None or stake < 0:
            msg = f"games.json game '{name}': prize plan stake must be >= 0"
            raise ValueError(msg)
        if not payouts or len(payouts) != size + 1:
            msg = f"games.json game '{name}': prize plan payouts must have size+1={size + 1} entries (0..size hits)"
            raise ValueError(msg)


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
        _validate_bets(name, cfg)
        _validate_prizes(name, cfg)
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
