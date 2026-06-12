"""Tests for the multi-label classifier: shapes, determinism, checkpoint round-trip."""

import json
import random

import duckdb
import pytest
import torch

from processing.train import (
    MLP,
    load_data,
    load_model,
    predict_results,
    preprocess_data,
    set_seed,
    top_n_from_logits,
    train_results,
)

K = 8
INPUT_DIM = 4 * K + 2


def test_top_n_from_logits_orders_by_probability() -> None:
    logits = torch.tensor([0.0, 3.0, -1.0, 2.0])
    top = top_n_from_logits(logits, 2)

    assert [num for num, _ in top] == [2, 4]  # 1-based indices of the two largest logits
    assert top[0][1] > top[1][1]
    assert all(0.0 <= prob <= 1.0 for _, prob in top)


def test_top_n_from_logits_caps_at_k() -> None:
    logits = torch.tensor([0.5, 1.5])
    assert len(top_n_from_logits(logits, 10)) == 2


def test_mlp_output_shape() -> None:
    model = MLP(INPUT_DIM, [16, 8], K)
    out = model(torch.zeros(3, INPUT_DIM))
    assert out.shape == (3, K)


def test_set_seed_makes_training_deterministic() -> None:
    set_seed(123)
    model_a = MLP(INPUT_DIM, [16], K)
    set_seed(123)
    model_b = MLP(INPUT_DIM, [16], K)

    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(param_a, param_b)


def _seed_features_db(path: str, rows: int = 40) -> None:
    rng = random.Random(5)
    with duckdb.connect(path) as db:
        db.sql("CREATE TABLE features (draw_seq INTEGER PRIMARY KEY, x TEXT, y TEXT)")
        for seq in range(rows):
            x = [rng.random() for _ in range(INPUT_DIM)]
            drawn = rng.sample(range(K), 3)
            y = [1 if j in drawn else 0 for j in range(K)]
            db.execute("INSERT INTO features VALUES (?, ?, ?)", (seq, json.dumps(x), json.dumps(y)))


def test_preprocess_data_validates_empty(tmp_path) -> None:
    db_path = str(tmp_path / "empty.duckdb")
    with duckdb.connect(db_path) as db:
        db.sql("CREATE TABLE features (draw_seq INTEGER PRIMARY KEY, x TEXT, y TEXT)")

    with pytest.raises(ValueError, match="No feature rows"):
        preprocess_data(load_data(db_path))


def test_train_checkpoint_roundtrip_and_deterministic_predict(tmp_path) -> None:
    db_path = str(tmp_path / "game.duckdb")
    model_path = str(tmp_path / "game.pth")
    plot_path = str(tmp_path / "game.png")
    _seed_features_db(db_path)

    train_results(db_path, model_path, plot_path, hidden_dims=[16], epochs=3, batch_size=8, seed=42)

    checkpoint = torch.load(model_path, weights_only=True)
    assert checkpoint["input_dim"] == INPUT_DIM
    assert checkpoint["hidden_dims"] == [16]
    assert checkpoint["output_dim"] == K
    assert checkpoint["seed"] == 42
    assert "data_hash" in checkpoint
    assert "trained_at" in checkpoint
    assert checkpoint["val_loss"] > 0

    model = load_model(model_path)
    assert isinstance(model, MLP)

    features = [0.5] * INPUT_DIM
    top_a, grouped_a = predict_results(model_path, features, 3, approaches=10, seed=1)
    top_b, grouped_b = predict_results(model_path, features, 3, approaches=10, seed=1)
    assert top_a == top_b  # deterministic ranking
    assert grouped_a == grouped_b  # MC-dropout groups reproducible with the same seed
    assert len(top_a) == K  # full k-number ranking, probability-descending
    assert sorted(num for num, _ in top_a) == list(range(1, K + 1))
    probs = [prob for _, prob in top_a]
    assert probs == sorted(probs, reverse=True)

    # groups: counts within [1, approaches], descending order, no number in two groups
    counts = list(grouped_a)
    assert counts == sorted(counts, reverse=True)
    assert all(1 <= c <= 10 for c in counts)
    all_numbers = [num for numbers in grouped_a.values() for num in numbers]
    assert len(all_numbers) == len(set(all_numbers))
    assert sum(1 for _ in all_numbers) >= 3  # at least the top-n appear across passes


def test_train_patience_zero_runs_all_epochs(tmp_path) -> None:
    db_path = str(tmp_path / "game.duckdb")
    _seed_features_db(db_path)

    train_results(
        db_path,
        str(tmp_path / "full.pth"),
        str(tmp_path / "full.png"),
        hidden_dims=[16],
        epochs=7,
        patience=0,  # early stopping disabled — train to the end
        seed=42,
    )

    checkpoint = torch.load(str(tmp_path / "full.pth"), weights_only=True)
    assert checkpoint["epochs_run"] == 7
    assert checkpoint["early_stopped"] is False


def test_train_early_stopping_can_stop_before_all_epochs(tmp_path) -> None:
    db_path = str(tmp_path / "game.duckdb")
    _seed_features_db(db_path)

    # lr=0 -> weights never change -> val loss is constant -> no improvement after
    # epoch 1, so patience=1 stops deterministically at epoch 2
    train_results(
        db_path,
        str(tmp_path / "es.pth"),
        str(tmp_path / "es.png"),
        hidden_dims=[16],
        epochs=50,
        learning_rate=0.0,
        patience=1,
        seed=42,
    )

    checkpoint = torch.load(str(tmp_path / "es.pth"), weights_only=True)
    assert checkpoint["epochs_run"] == 2
    assert checkpoint["early_stopped"] is True


def test_train_is_reproducible_with_same_seed(tmp_path) -> None:
    db_path = str(tmp_path / "game.duckdb")
    _seed_features_db(db_path)

    train_results(db_path, str(tmp_path / "a.pth"), str(tmp_path / "a.png"), hidden_dims=[16], epochs=2, seed=7)
    train_results(db_path, str(tmp_path / "b.pth"), str(tmp_path / "b.png"), hidden_dims=[16], epochs=2, seed=7)

    state_a = torch.load(str(tmp_path / "a.pth"), weights_only=True)["state_dict"]
    state_b = torch.load(str(tmp_path / "b.pth"), weights_only=True)["state_dict"]
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key])
