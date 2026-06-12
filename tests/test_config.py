"""Tests for game configuration loading and helpers."""

import pytest

from processing.config import GAMES_CONFIG, GameType, game_file, get_game_config


def test_games_config_loaded() -> None:
    assert set(GAMES_CONFIG) == set(GameType)
    assert GAMES_CONFIG[GameType.MultiMulti] == {"prefix": "MultiMulti", "n": 20, "k": 80}


def test_get_game_config_valid() -> None:
    config = get_game_config(GameType.Lotto)
    assert config["n"] == 6
    assert config["k"] == 49


def test_get_game_config_rejects_non_gametype() -> None:
    with pytest.raises(TypeError):
        get_game_config("MultiMulti")  # type: ignore[arg-type]


def test_from_str_valid() -> None:
    assert GameType.from_str("Lotto") is GameType.Lotto


def test_from_str_invalid_lists_choices() -> None:
    with pytest.raises(ValueError, match="MultiMulti"):
        GameType.from_str("Nope")


def test_game_file_path() -> None:
    path = game_file(GameType.Szybkie600, "duckdb")
    assert path.name == "Szybkie600.duckdb"
    assert path.is_absolute()
