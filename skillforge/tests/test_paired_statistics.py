from itertools import product

import pytest

from src.evaluation.statistics import compare_paired_outcomes, exact_mcnemar_p_value


def test_armitage_berry_textbook_example():
    # Armitage & Berry, Statistical Methods in Medical Research (1994), p.127.
    # Fifty paired specimens: both=20, A-only=12, B-only=2, neither=16.
    # Official worked reproduction reports two-sided exact p=0.0129:
    # https://www.statsdirect.com/help/chi_square_tests/mcnemar.htm
    # A improvement is one-sided: (C(14,12)+C(14,13)+C(14,14))/2**14
    # = (91+14+1)/16384 = 0.0064697265625 (half the exact two-sided tail).
    baseline = {str(i): i < 20 or 32 <= i < 34 for i in range(50)}
    candidate = {str(i): i < 32 for i in range(50)}
    comparison = compare_paired_outcomes(baseline, candidate)
    assert comparison.candidate_only_passes == 12
    assert comparison.baseline_only_passes == 2
    assert comparison.p_value == 0.0064697265625
    assert round(2 * comparison.p_value, 4) == 0.0129
    assert comparison.baseline_pass_rate == 22 / 50
    assert comparison.candidate_pass_rate == 32 / 50
    assert comparison.should_promote


@pytest.mark.parametrize("n", range(1, 10))
def test_exact_tail_matches_exhaustive_fair_coin_outcomes(n):
    # Independent enumeration checks orientation, ties, and both extreme tails.
    outcomes = [sum(bits) for bits in product((0, 1), repeat=n)]
    for b in range(n + 1):
        assert exact_mcnemar_p_value(b, n - b) == sum(x >= b for x in outcomes) / 2**n


@pytest.mark.parametrize("b,c,p,promote", [
    (0, 0, 1, False), (0, 4, 1, False), (4, 0, 0.0625, True),
    (3, 0, 0.125, False), (5, 1, 0.109375, False),
    (6, 1, 0.0625, True), (3, 3, 0.65625, False),
    (1, 6, 0.9921875, False),
])
def test_promotion_direction_and_small_discordance_counts(b, c, p, promote):
    baseline = {"both": True, **{f"b{i}": False for i in range(b)}, **{f"c{i}": True for i in range(c)}}
    candidate = {"both": True, **{f"b{i}": True for i in range(b)}, **{f"c{i}": False for i in range(c)}}
    comparison = compare_paired_outcomes(baseline, dict(reversed(list(candidate.items()))))
    assert comparison.p_value == p
    assert comparison.should_promote is promote


def test_large_counts_do_not_overflow_and_strong_regression_never_promotes():
    assert 0 < exact_mcnemar_p_value(700, 500) < 0.10
    assert exact_mcnemar_p_value(500, 700) > 0.9
    assert exact_mcnemar_p_value(1200, 0) == 0.0  # float underflow, exact decision stays correct
    assert exact_mcnemar_p_value(0, 1200) == 1.0


@pytest.mark.parametrize("b,c", [(-1, 0), (0, -1), (True, 2), (2, False), (1.5, 0), (0, "1")])
def test_invalid_discordant_counts(b, c):
    with pytest.raises(ValueError, match="non-negative integers"):
        exact_mcnemar_p_value(b, c)


@pytest.mark.parametrize("baseline,candidate", [
    ({}, {}), ({"a": True}, {"b": True}), ({"a": True}, {}),
    ({"a": 1}, {"a": True}), ({"a": True}, {"a": None}),
    ({"": True}, {"": False}), ({1: True}, {1: False}),
])
def test_invalid_or_unpaired_outcomes(baseline, candidate):
    with pytest.raises(ValueError):
        compare_paired_outcomes(baseline, candidate)
