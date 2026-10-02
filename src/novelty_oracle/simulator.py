"""Deterministic revenue simulator for single-parameter mechanisms.

Quality = expected payoff under the battery's valuation mixture
(``battery.sample_mixture_values``). For each seed: draw
``draws_per_seed`` valuation vectors from a ``random.Random`` seeded by
``base_seed + seed_index``, run the mechanism, record mean revenue.
Statistics over the per-seed means give a 95% confidence interval.

Because the simulator samples the same mixture the brute-force battery
ranking is computed on, "simulated ranking matches battery ranking" is a
legitimate falsifiable check: both estimate the same expectation.

Determinism contract: the same (mechanism, config) always yields the same
SeedStats, because every random draw is a pure function of the seed.

Bidder model: direct revelation with truthful bidding (values are fed as
bids). Strategic bid-shading is out of scope — the oracle compares
mechanisms as functions, not equilibria.
"""

from __future__ import annotations

import math
import random

from .battery import sample_mixture_values
from .types import Mechanism, OracleConfig, SeedStats


def simulate(mechanism: Mechanism, config: OracleConfig) -> SeedStats:
    """Estimate expected revenue of a mechanism, deterministically."""
    seed_means: list[float] = []
    for s in range(config.n_seeds):
        rng = random.Random(config.base_seed + s)
        total = 0.0
        for _ in range(config.draws_per_seed):
            values = sample_mixture_values(rng)
            allocation = mechanism.allocate(values)
            payments = mechanism.pay(values, allocation)
            total += sum(payments)
        seed_means.append(total / config.draws_per_seed)

    n = len(seed_means)
    mean = sum(seed_means) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in seed_means) / (n - 1)
        std = math.sqrt(var)
        half_width = 1.96 * std / math.sqrt(n)
    else:
        std = 0.0
        half_width = 0.0
    return SeedStats(
        n_seeds=n,
        draws_per_seed=config.draws_per_seed,
        mean_payoff=mean,
        std_of_seed_means=std,
        ci_low=mean - half_width,
        ci_high=mean + half_width,
    )


def delta_ci(candidate: SeedStats, baseline: SeedStats) -> tuple[float, float, float]:
    """(delta, ci_low, ci_high) for candidate_mean - baseline_mean.

    Uses the normal approximation on the difference of the two
    independent seed-mean samples.
    """
    delta = candidate.mean_payoff - baseline.mean_payoff
    se = math.sqrt(
        candidate.std_of_seed_means**2 / max(candidate.n_seeds, 1)
        + baseline.std_of_seed_means**2 / max(baseline.n_seeds, 1)
    )
    half_width = 1.96 * se
    return delta, delta - half_width, delta + half_width
