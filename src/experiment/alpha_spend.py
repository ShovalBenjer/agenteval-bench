"""Lan-DeMets alpha-spending with the O'Brien-Fleming spending function.

Implements (adaptation of published methods, stdlib only):
- Lan & DeMets (1983) alpha-spending framework: boundaries zc(k) are
  solved successively so the cumulative exit probability at look k equals
  the spending function at the information fraction
  (PMC4024106, eq. 2.1);
- the O'Brien-Fleming-like spending function
  alpha*(t) = 2 - 2*Phi(z_{alpha/2} / sqrt(t));
- Armitage, McPherson & Rowe (1969) recursive integration for the
  canonical joint distribution of sequential z-statistics
  (Cov(Z_j, Z_k) = sqrt(t_j / t_k)).

Convention note (established from the literature): the OF-like spending
function keeps its two-sided form even for one-sided tests; the
one-sided/two-sided choice enters only in the boundary solve (upper tail
vs two tails). Hence one-sided alpha=0.025 reproduces the tabulated K=2
boundaries ~(2.96, 1.97) (secondary-only oracle), while two-sided
alpha=0.05 gives ~(2.77, ~2.01). Both control their nominal level; the
difference is the tested hypothesis, not correctness.

All functions are pure and deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from experiment.types import ExperimentPlan, Sides


class PeekRefused(ValueError):
    """An interim analysis was attempted outside the pre-registered look schedule."""


def _phi(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _Phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def inv_normal_cdf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's approximation, ~1e-9 accuracy)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    def horner(coefs: tuple[float, ...], x: float) -> float:
        acc = 0.0
        for coef in coefs:
            acc = acc * x + coef
        return acc

    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return horner(c, q) / horner(d + (1.0,), q)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -horner(c, q) / horner(d + (1.0,), q)
    q = p - 0.5
    r = q * q
    return horner(a, r) * q / horner(b + (1.0,), r)


def obrien_fleming_spend(t: float, alpha: float) -> float:
    """Cumulative alpha spent by information fraction t (OF-like spending)."""
    if not 0.0 < t <= 1.0:
        raise ValueError(f"information fraction must be in (0, 1], got {t}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    z = inv_normal_cdf(1.0 - alpha / 2.0)
    return 2.0 * (1.0 - _Phi(z / math.sqrt(t)))


def _simpson_weights(n: int, h: float) -> list[float]:
    weights = [4.0 if j % 2 == 1 else 2.0 for j in range(n + 1)]
    weights[0] = weights[n] = 1.0
    return [w * h / 3.0 for w in weights]


def _solve(
    looks: tuple[float, ...], alpha: float, sides: Sides, *, zmax: float, grid_n: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Solve OF-spending boundaries; return (boundaries, cumulative exits)."""
    return _solve_cached(looks, alpha, sides, zmax, grid_n)


def _exit_prob(
    c: float,
    dens: list[float],
    grid: list[float],
    w: list[float],
    sides: Sides,
) -> float:
    """Exit probability at look boundary c given the continued-paths sub-density."""
    total = 0.0
    if sides == "two":
        for zj, dj, wj in zip(grid, dens, w):
            if abs(zj) >= c:
                total += wj * dj
    else:
        for zj, dj, wj in zip(grid, dens, w):
            if zj >= c:
                total += wj * dj
    return total


@lru_cache(maxsize=32)
def _solve_cached(
    looks: tuple[float, ...], alpha: float, sides: Sides, zmax: float, grid_n: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if grid_n % 2 == 1:
        raise ValueError(f"grid_n must be even, got {grid_n}")
    h = 2.0 * zmax / grid_n
    grid = [-zmax + j * h for j in range(grid_n + 1)]
    w = _simpson_weights(grid_n, h)
    # dens = sub-density of Z_k over paths that continued past all earlier looks.
    dens = [_phi(z) for z in grid]
    boundaries: list[float] = []
    cum_exit: list[float] = []
    spent_prev = 0.0
    for k, t in enumerate(looks):
        target = obrien_fleming_spend(t, alpha) - spent_prev

        lo, hi = 0.0, 12.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if _exit_prob(mid, dens, grid, w, sides) >= target:
                lo = mid
            else:
                hi = mid
        c = 0.5 * (lo + hi)
        boundaries.append(c)
        spent_prev += target
        cum_exit.append(spent_prev)
        if k < len(looks) - 1:
            t_next = looks[k + 1]
            r = math.sqrt(t / t_next)
            s = math.sqrt(1.0 - t / t_next)
            new = [0.0] * (grid_n + 1)
            for jj, z in enumerate(grid):
                total = 0.0
                for i, u in enumerate(grid):
                    if sides == "two":
                        if abs(u) >= c:
                            continue
                    elif u >= c:
                        continue
                    total += w[i] * dens[i] * _phi((z - r * u) / s) / s
                new[jj] = total
            dens = new
    return tuple(boundaries), tuple(cum_exit)


def group_sequential_boundaries(
    looks: tuple[float, ...],
    alpha: float,
    sides: Sides = "two",
    *,
    zmax: float = 10.0,
    grid_n: int = 600,
) -> tuple[float, ...]:
    """O'Brien-Fleming alpha-spending boundaries for the look schedule.

    looks: strictly increasing information fractions in (0, 1], last 1.0.
    Returns the per-look z critical values; reject at look k when
    |z_k| >= boundary_k (two-sided) or z_k >= boundary_k (one-sided).
    """
    prev = 0.0
    for t in looks:
        if not 0.0 < t <= 1.0 or t <= prev:
            raise ValueError(f"looks must be strictly increasing in (0, 1], got {looks}")
        prev = t
    if looks[-1] != 1.0:
        raise ValueError(f"final look must be at 1.0, got {looks[-1]}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    boundaries, _ = _solve(looks, alpha, sides, zmax=zmax, grid_n=grid_n)
    return boundaries


def spending_report(
    looks: tuple[float, ...], alpha: float, sides: Sides = "two"
) -> list[tuple[float, float, float]]:
    """Per-look (information fraction, boundary, incremental alpha spent)."""
    boundaries, cum = _solve(looks, alpha, sides, zmax=10.0, grid_n=600)
    rows = []
    prev = 0.0
    for t, b, c in zip(looks, boundaries, cum):
        rows.append((t, b, c - prev))
        prev = c
    return rows


@dataclass(frozen=True)
class LookDecision:
    """Outcome of one registered interim look."""

    fraction: float
    z: float
    boundary: float
    reject: bool
    cumulative_alpha_spent: float


class SequentialGate:
    """Enforces a pre-registered group-sequential stopping rule.

    Looks must be requested in registered order, each exactly once.
    Anything else — an unregistered fraction, a repeat, an out-of-order
    look — raises PeekRefused. This is the running no-peeking check:
    the only way to see interim results is through the spending schedule
    the experiment pre-registered.
    """

    def __init__(self, plan: ExperimentPlan) -> None:
        self._plan = plan
        self._boundaries, self._cum = _solve(
            plan.looks, plan.alpha, plan.sides, zmax=10.0, grid_n=600
        )
        self._taken: list[float] = []

    @property
    def boundaries(self) -> tuple[float, ...]:
        """Per-look z critical values, in look order."""
        return self._boundaries

    @property
    def pending(self) -> tuple[float, ...]:
        """Registered looks not yet taken, in order."""
        return tuple(t for t in self._plan.looks if t not in self._taken)

    def look(self, fraction: float, z: float) -> LookDecision:
        """Request the interim analysis at information fraction `fraction`.

        Raises PeekRefused for unregistered, duplicate, or out-of-order
        looks. Otherwise returns the accept/reject decision against the
        alpha-spending boundary.
        """
        if fraction not in self._plan.looks:
            raise PeekRefused(
                f"unregistered interim look at t={fraction}: plan {self._plan.name!r} "
                f"permits exactly {list(self._plan.looks)}"
            )
        if fraction in self._taken:
            raise PeekRefused(f"duplicate look at t={fraction}: already taken")
        expected = self.pending[0]
        if fraction != expected:
            raise PeekRefused(
                f"out-of-order look at t={fraction}: next registered look is t={expected}"
            )
        k = len(self._taken)
        boundary = self._boundaries[k]
        reject = (z >= boundary) if self._plan.sides == "one" else (abs(z) >= boundary)
        self._taken.append(fraction)
        return LookDecision(
            fraction=fraction,
            z=z,
            boundary=boundary,
            reject=reject,
            cumulative_alpha_spent=self._cum[k],
        )
