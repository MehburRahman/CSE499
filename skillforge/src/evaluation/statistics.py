"""Exact paired binary comparisons, shared by evolution and result analysis."""

from fractions import Fraction

from src.evolution.models import PairedComparison


def _mcnemar_tail(candidate_only_passes, baseline_only_passes):
    for count in (candidate_only_passes, baseline_only_passes):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("discordant counts must be non-negative integers")
    b, c = candidate_only_passes, baseline_only_passes
    n = b + c
    # H1 is candidate > baseline: P[Binomial(b+c, 1/2) >= b].
    # Symmetry lets us sum the shorter tail using exact integer arithmetic.
    # n=0 (or b=0) has tail probability 1.
    if b == 0:
        return Fraction(1)
    end = c if b > c else b - 1
    term = total = 1
    for k in range(1, end + 1):
        term = term * (n - k + 1) // k
        total += term
    denominator = 1 << n
    return Fraction(total if b > c else denominator - total, denominator)


def exact_mcnemar_p_value(candidate_only_passes: int, baseline_only_passes: int) -> float:
    """Exact one-sided p-value in the candidate-improvement direction.

    No chi-square approximation, continuity correction, doubled tail, or mid-p.
    Fagerland et al. (2013), exact conditional test:
    https://pmc.ncbi.nlm.nih.gov/articles/PMC3716987/
    """
    return float(_mcnemar_tail(candidate_only_passes, baseline_only_passes))


def compare_paired_outcomes(baseline, candidate) -> PairedComparison:
    """Compare {task_id: bool} maps; never pair by order or drop missing tasks."""
    if not baseline or baseline.keys() != candidate.keys():
        raise ValueError("paired outcomes must contain the same non-empty task IDs")
    for outcomes in (baseline, candidate):
        if any(not isinstance(key, str) or not key.strip() for key in outcomes):
            raise ValueError("task IDs must be non-empty strings")
        if any(type(value) is not bool for value in outcomes.values()):
            raise ValueError("each paired outcome must be a boolean")
    b = sum(candidate[key] and not baseline[key] for key in baseline)
    c = sum(baseline[key] and not candidate[key] for key in baseline)
    tail = _mcnemar_tail(b, c)
    # Compare fractions so rounding/underflow can never change the decision.
    promote = b > c and tail < Fraction(1, 10)
    n = len(baseline)
    return PairedComparison(
        task_count=n, baseline_pass_rate=sum(baseline.values()) / n,
        candidate_pass_rate=sum(candidate.values()) / n,
        candidate_only_passes=b, baseline_only_passes=c, p_value=float(tail),
        decision="promote" if promote else "reject",
        reason=(
            f"Exact one-sided McNemar: b={b}, c={c}, p={float(tail):.8g}; "
            + ("b > c and p < 0.10." if promote else "requires b > c and p < 0.10.")
            + " Recommendation only; registry unchanged."
        ),
    )
