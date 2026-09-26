"""Tests for CLI dispatch: every path forwards its hyperparameters and predict options."""

import sys

import pytest

import main
from processing.config import GameType


@pytest.fixture
def calls(monkeypatch) -> list[tuple[str, tuple, dict]]:
    recorded: list[tuple[str, tuple, dict]] = []
    for name in ["resolve_results", "preprocess_results", "train_game_results", "predict_game_results"]:
        monkeypatch.setattr(
            main.game,
            name,
            lambda *args, _name=name, **kwargs: recorded.append((_name, args, kwargs)),
        )
    monkeypatch.setattr(main, "evaluate_game", lambda *args, **kwargs: recorded.append(("evaluate", args, kwargs)))
    return recorded


def _run(monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    main.main()


def test_update_loads_trains_and_predicts_with_options(monkeypatch, calls) -> None:
    _run(monkeypatch, ["update", "--game", "Lotto", "--epochs", "5", "--target", "1,2", "--bets-count", "3"])

    assert [name for name, _, _ in calls] == [
        "resolve_results",
        "preprocess_results",
        "train_game_results",
        "predict_game_results",
    ]
    _, train_args, train_kwargs = calls[2]
    assert train_args == (GameType.Lotto,)
    assert train_kwargs["epochs"] == 5
    _, predict_args, predict_kwargs = calls[3]
    assert predict_args == (GameType.Lotto, ["1", "2"])
    assert predict_kwargs["bets_count"] == 3


def test_train_forwards_predict_options(monkeypatch, calls) -> None:
    _run(monkeypatch, ["train", "--approaches", "7", "--bets-size", "4", "--histogram"])

    assert [name for name, _, _ in calls] == ["train_game_results", "predict_game_results"]
    _, _, predict_kwargs = calls[1]
    assert predict_kwargs["approaches"] == 7
    assert predict_kwargs["bets_size"] == 4
    assert predict_kwargs["histogram"] is True


def test_predict_only_predicts(monkeypatch, calls) -> None:
    _run(monkeypatch, ["predict", "--seed", "9"])

    assert [name for name, _, _ in calls] == ["predict_game_results"]
    assert calls[0][2]["seed"] == 9


def test_legacy_train_forwards_hyperparameters(monkeypatch, calls) -> None:
    _run(monkeypatch, ["--train", "--epochs", "3", "--patience", "0", "--target", "5"])

    assert [name for name, _, _ in calls] == ["train_game_results", "predict_game_results"]
    assert calls[0][2]["epochs"] == 3
    assert calls[0][2]["patience"] == 0
    assert calls[1][1] == (GameType.MultiMulti, ["5"])


def test_legacy_update_loads_and_trains(monkeypatch, calls) -> None:
    _run(monkeypatch, ["--update"])

    assert [name for name, _, _ in calls] == [
        "resolve_results",
        "preprocess_results",
        "train_game_results",
        "predict_game_results",
    ]


def test_evaluate_forwards_epochs(monkeypatch, calls) -> None:
    _run(monkeypatch, ["evaluate", "--retrain", "--epochs", "4", "--last-n", "2"])

    assert calls == [("evaluate", (GameType.MultiMulti,), {"last_n": 2, "retrain": True, "epochs": 4, "seed": 42})]
