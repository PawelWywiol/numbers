"""Tests for backtest math."""

import torch

from processing.evaluate import _actual_numbers, expected_random_hits


def test_expected_random_hits() -> None:
    # 20 drawn from 80 against 20 actual -> 20*20/80 = 5.0
    assert expected_random_hits(20, 80) == 5.0
    assert expected_random_hits(6, 49) == 6 * 6 / 49


def test_actual_numbers_rounds_to_ints() -> None:
    tensor = torch.tensor([1.4, 2.6, 3.0])
    assert _actual_numbers(tensor) == {1, 3}
