"""Exact small-sample statistics for the alt-test, stdlib only.

Implements (paper-faithful):
- one-sided paired t-test of H0: mean(d) >= epsilon, with the Student-t
  CDF computed through the regularized incomplete beta function
  (continued-fraction evaluation; no scipy dependency, matching the
  repo's stdlib-only discipline for statistical tooling);
- Wilcoxon signed-rank test (normal approximation with tie correction
  and continuity correction) for the n < 30 path the paper prescribes;
- the Benjamini-Yekutieli (BY) FDR-controlling procedure for dependent
  hypotheses.

All functions are pure and deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction evaluation for the incomplete beta function."""
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = c * d
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """I_x(a, b) for 0 <= x <= 1, a > 0, b > 0."""
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"x must be in [0, 1], got {x}")
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"a and b must be positive, got {a}, {b}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    # Use symmetry to keep the continued fraction in its fast regime.
    if x < (a + 1.0) / (a + b + 2.0):
        front = math.exp(
            math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
            + a * math.log(x) + b * math.log(1.0 - x)
        )
        return front * _betacf(a, b, x) / a
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: int) -> float:
    """CDF of Student's t with df degrees of freedom at t."""
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    if t == 0.0:
        return 0.5
    x = df / (df + t * t)
    ib = regularized_incomplete_beta(x, df / 2.0, 0.5)
    if t > 0.0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def paired_t_test_one_sided(diffs: Sequence[float], epsilon: float) -> float:
    """One-sided paired t-test p-value for H0: mean(d) >= epsilon.

    Returns P(T <= t_obs) with t_obs = (mean(d) - epsilon) / (s / sqrt(n)).
    Degenerate zero-variance inputs resolve deterministically: when every
    difference is identical, the sign of (mean - epsilon) decides the
    outcome with certainty (p = 0.0 or p = 1.0).
    """
    n = len(diffs)
    if n < 2:
        raise ValueError(f"paired t-test needs n >= 2, got {n}")
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var == 0.0:
        # Deterministic: the mean is known exactly.
        return 0.0 if mean < epsilon else 1.0
    s = math.sqrt(var)
    t_obs = (mean - epsilon) / (s / math.sqrt(n))
    return student_t_cdf(t_obs, n - 1)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilcoxon_signed_rank_one_sided(diffs: Sequence[float], epsilon: float) -> float:
    """One-sided Wilcoxon signed-rank p-value for H0: median(d) >= epsilon.

    Applied to the epsilon-shifted differences e_i = d_i - epsilon, testing
    H0: median(e) >= 0 vs H1: median(e) < 0. Normal approximation with
    average-rank tie handling, tie correction to the variance, and a
    continuity correction. Zeros are dropped per the standard procedure.
    """
    shifted = [d - epsilon for d in diffs]
    nonzero = [e for e in shifted if e != 0.0]
    n = len(nonzero)
    if n == 0:
        # Every shifted difference is exactly zero: the median equals
        # epsilon, i.e. the data sit exactly on the H0 boundary.
        return 1.0
    order = sorted(range(n), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * n
    i = 0
    tie_correction = 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(nonzero[order[j + 1]]) == abs(nonzero[order[i]]):
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        tie_size = j - i + 1
        tie_correction += (tie_size**3 - tie_size) / 48.0
        i = j + 1
    w_plus = sum(r for e, r in zip(nonzero, ranks) if e > 0.0)
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction
    if var <= 0.0:
        return 1.0 if w_plus >= mu else 0.0
    # Lower-tail test: small W+ rejects. Continuity correction nudges up.
    z = (w_plus - mu + 0.5) / math.sqrt(var)
    return _normal_cdf(z)


def benjamini_yekutieli(p_values: Sequence[float], q: float) -> list[bool]:
    """Benjamini-Yekutieli FDR control for (possibly dependent) hypotheses.

    Sorts p-values ascending; with c(m) = sum_{i=1..m} 1/i, rejects
    H_(1)..H_(k) for the largest k with p_(k) <= k*q/(m*c(m)).
    Returns rejection flags in the original order.
    """
    m = len(p_values)
    if m == 0:
        return []
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    for p in p_values:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-values must be in [0, 1], got {p}")
    harmonic = sum(1.0 / i for i in range(1, m + 1))
    order = sorted(range(m), key=lambda i: p_values[i])
    sorted_p = [p_values[i] for i in order]
    cutoff = 0
    for k in range(1, m + 1):
        if sorted_p[k - 1] <= k * q / (m * harmonic):
            cutoff = k
    rejected = [False] * m
    for rank in range(cutoff):
        rejected[order[rank]] = True
    return rejected
