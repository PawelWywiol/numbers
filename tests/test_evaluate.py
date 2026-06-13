"""Tests for backtest math: baselines, binomial significance, calibration."""

import math

import torch

from processing.evaluate import (
    _actual_numbers,
    _calibration,
    _frequency_top_n,
    binomial_p_value,
    expected_random_hits,
    hypergeometric_distribution,
)


def test_expected_random_hits() -> None:
    # 20 drawn from 80 against 20 actual -> 20*20/80 = 5.0
    assert expected_random_hits(20, 80) == 5.0
    assert expected_random_hits(6, 49) == 6 * 6 / 49


def test_actual_numbers_decodes_binary_vector() -> None:
    vector = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    assert _actual_numbers(vector) == {1, 3, 5}


def test_frequency_top_n() -> None:
    history = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
    )
    assert _frequency_top_n(history, 2) == {1, 2}


def test_binomial_p_value_fair_coin_exact() -> None:
    # P(X = 5 | n=10, p=0.5) is the most likely outcome -> two-sided p-value = 1.0
    assert math.isclose(binomial_p_value(5, 10, 0.5), 1.0, rel_tol=1e-12)


def test_binomial_p_value_extreme() -> None:
    # 10/10 heads on a fair coin: p = 2 * 0.5^10
    assert math.isclose(binomial_p_value(10, 10, 0.5), 2 * 0.5**10, rel_tol=1e-9)


def test_binomial_p_value_matches_expected_mean() -> None:
    # Hits at the expected rate should NOT be significant.
    assert binomial_p_value(250, 1000, 0.25) > 0.9


def test_binomial_p_value_detects_signal() -> None:
    # 350/1000 at p=0.25 is a strong deviation.
    assert binomial_p_value(350, 1000, 0.25) < 1e-10


def test_binomial_p_value_zero_trials() -> None:
    assert binomial_p_value(0, 0, 0.25) == 1.0


def test_calibration_buckets() -> None:
    pairs = [(0.05, 0.0), (0.05, 0.0), (0.95, 1.0), (0.95, 1.0), (0.55, 1.0), (0.55, 0.0)]
    buckets = _calibration(pairs)

    by_low = {bucket["low"]: bucket for bucket in buckets}
    assert by_low[0.0]["count"] == 2
    assert by_low[0.0]["observed_rate"] == 0.0
    assert by_low[0.5]["observed_rate"] == 0.5
    assert by_low[0.9]["observed_rate"] == 1.0
    assert math.isclose(by_low[0.9]["mean_predicted"], 0.95)


def test_hypergeometric_distribution_sums_to_one() -> None:
    dist = hypergeometric_distribution(6, 6, 49)  # pick 6, 6 drawn from 49
    assert len(dist) == 7  # h = 0..6
    assert math.isclose(sum(dist), 1.0, rel_tol=1e-12)
    assert all(p >= 0 for p in dist)


def test_hypergeometric_distribution_matches_known_value() -> None:
    # Pick 2 of 4, 2 drawn: P(0 hits) = C(2,0)C(2,2)/C(4,2) = 1/6
    dist = hypergeometric_distribution(2, 2, 4)
    assert math.isclose(dist[0], 1 / 6, rel_tol=1e-12)
    assert math.isclose(dist[2], 1 / 6, rel_tol=1e-12)
    assert math.isclose(dist[1], 4 / 6, rel_tol=1e-12)


def test_hypergeometric_distribution_pick_differs_from_drawn() -> None:
    # MultiMulti bet-5: pick 5, 20 drawn from 80 -> probabilities for 0..5 hits
    dist = hypergeometric_distribution(5, 20, 80)
    assert len(dist) == 6
    assert math.isclose(sum(dist), 1.0, rel_tol=1e-12)


def test_calibration_includes_probability_one() -> None:
    buckets = _calibration([(1.0, 1.0)])
    assert len(buckets) == 1
    assert buckets[0]["low"] == 0.9
