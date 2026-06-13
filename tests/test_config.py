"""Tests for game configuration loading and helpers."""

import pytest

from processing.config import (
    GAMES_CONFIG,
    GameType,
    _validate_bets,
    _validate_prizes,
    game_file,
    get_game_config,
)


def test_games_config_loaded() -> None:
    assert set(GAMES_CONFIG) == set(GameType)
    multi = GAMES_CONFIG[GameType.MultiMulti]
    assert multi["n"] == 20
    assert multi["k"] == 80
    assert multi["bets"]["count"] >= 1
    assert 1 <= multi["bets"]["size"] <= multi["k"]


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


def test_validate_bets_missing() -> None:
    with pytest.raises(ValueError, match="missing the 'bets' section"):
        _validate_bets("X", {"prefix": "X", "n": 6, "k": 49})  # type: ignore[typeddict-item]


def test_validate_bets_size_exceeds_k() -> None:
    cfg = {"prefix": "X", "n": 6, "k": 49, "bets": {"count": 8, "size": 50}}
    with pytest.raises(ValueError, match="size must be <= k"):
        _validate_bets("X", cfg)  # type: ignore[arg-type]


def test_validate_bets_valid() -> None:
    cfg = {"prefix": "X", "n": 6, "k": 49, "bets": {"count": 8, "size": 6}}
    _validate_bets("X", cfg)  # type: ignore[arg-type]  # does not raise


def test_validate_prizes_absent_is_ok() -> None:
    _validate_prizes("X", {"prefix": "X", "n": 6, "k": 49, "bets": {"count": 8, "size": 6}})  # type: ignore[arg-type]


def test_validate_prizes_wrong_length() -> None:
    cfg = {
        "prefix": "X",
        "n": 6,
        "k": 49,
        "bets": {"count": 8, "size": 6},
        "prizes": [{"size": 6, "stake": 2, "payouts": [0, 1]}],
    }
    with pytest.raises(ValueError, match="payouts must have size\\+1=7"):
        _validate_prizes("X", cfg)  # type: ignore[arg-type]


def test_validate_prizes_bad_size() -> None:
    cfg = {
        "prefix": "X",
        "n": 6,
        "k": 49,
        "bets": {"count": 8, "size": 6},
        "prizes": [{"size": 99, "stake": 2, "payouts": [0]}],
    }
    with pytest.raises(ValueError, match="prize plan size must be 1..k"):
        _validate_prizes("X", cfg)  # type: ignore[arg-type]


def test_validate_prizes_multiple_plans_valid() -> None:
    cfg = {
        "prefix": "X",
        "n": 20,
        "k": 80,
        "bets": {"count": 8, "size": 10},
        "prizes": [
            {"size": 5, "stake": 2, "payouts": [0, 0, 0, 4, 8, 700]},
            {"size": 10, "stake": 2, "payouts": [0, 0, 0, 0, 2, 4, 12, 140, 520, 10000, 250000]},
        ],
    }
    _validate_prizes("X", cfg)  # type: ignore[arg-type]  # does not raise
