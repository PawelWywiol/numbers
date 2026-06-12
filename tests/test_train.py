"""Tests for prediction post-processing."""

import torch

from processing.train import process_predictions


def test_process_predictions_returns_n_unique_in_range() -> None:
    raw = torch.tensor([5.4, 5.6, 90.0, -3.0, 12.2, 12.4])
    result = process_predictions(raw, n=6, k=32)

    assert len(result) == 6
    assert len(set(result)) == 6
    assert all(1 <= num <= 32 for num in result)
    assert result == sorted(result)


def test_process_predictions_clamps_out_of_range() -> None:
    raw = torch.tensor([0.0, -100.0, 999.0])
    result = process_predictions(raw, n=3, k=49)

    assert all(1 <= num <= 49 for num in result)


def test_process_predictions_pads_when_too_few() -> None:
    raw = torch.tensor([7.0, 7.0, 7.0])  # collapses to a single unique number
    result = process_predictions(raw, n=6, k=49)

    assert len(result) == 6
    assert 7 in result


def test_process_predictions_trims_when_too_many() -> None:
    raw = torch.arange(1, 40, dtype=torch.float32)
    result = process_predictions(raw, n=6, k=49)

    assert len(result) == 6
