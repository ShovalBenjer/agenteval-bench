"""Canonical valuation-profile battery (Law Article 1).

The battery is the fixed, versioned "test set" that behavioral novelty is
measured against. Construction is fully deterministic: the same
``BatterySpec`` always yields the same profiles.

The per-category samplers are also the simulator's quality distribution:
quality = expected payoff under the battery's valuation mixture. This is
what makes "simulated quality ranking matches brute-force battery
ranking" a legitimate falsifiable check — both estimate the same
expectation.

Why each category is in the battery:
  iid-uniform   — the core. Regular i.i.d. values are the setting where
                  revenue-optimality is known analytically (Myerson), so
                  quality has a ground truth here.
  asymmetric    — bidders drawn from different distributions. Optimal
                  mechanisms differ under asymmetry; a "novel" mechanism
                  that only works in the symmetric case is exposed here.
  tie           — exact ties. Tie-breaking rules are where implementations
                  that look identical behaviorally diverge.
  dominant      — one bidder an order of magnitude above the rest. Reserve
                  prices and participation decisions show up here.
  zero          — zero values present. Participation/edge behavior.
  skewed-beta   — non-uniform regular distribution. A mechanism tuned to
                  uniform (e.g. Myerson with the wrong reserve) degrades
                  here, so quality differences become visible.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from .types import BatterySpec, ValuationProfile

BATTERY_VERSION = 1
CONSTRUCTION_SEED = 77031

CategorySampler = Callable[[random.Random], tuple[float, ...]]


def _sample_iid_uniform(rng: random.Random) -> tuple[float, ...]:
    n = rng.choice((2, 3, 5))
    return tuple(rng.random() for _ in range(n))


def _sample_asymmetric(rng: random.Random) -> tuple[float, ...]:
    n = rng.choice((2, 3, 4))
    hi = [rng.uniform(0.3, 1.0) for _ in range(n)]
    return tuple(rng.uniform(0.0, h) for h in hi)


def _sample_tie(rng: random.Random) -> tuple[float, ...]:
    n = rng.choice((2, 3, 4))
    top = rng.random()
    values = [rng.random() * top for _ in range(n)]
    i, j = rng.sample(range(n), 2)
    values[i] = top
    values[j] = top
    return tuple(values)


def _sample_dominant(rng: random.Random) -> tuple[float, ...]:
    n = rng.choice((2, 3, 4))
    values = [rng.random() * 0.1 for _ in range(n)]
    values[rng.randrange(n)] = rng.uniform(0.7, 1.0)
    return tuple(values)


def _sample_zero(rng: random.Random) -> tuple[float, ...]:
    n = rng.choice((2, 3))
    values = [rng.random() for _ in range(n)]
    values[rng.randrange(n)] = 0.0
    return tuple(values)


def _sample_skewed_beta(rng: random.Random) -> tuple[float, ...]:
    n = rng.choice((2, 3, 5))
    return tuple(rng.betavariate(2.0, 5.0) for _ in range(n))


CATEGORY_SAMPLERS: dict[str, CategorySampler] = {
    "iid-uniform": _sample_iid_uniform,
    "asymmetric": _sample_asymmetric,
    "tie": _sample_tie,
    "dominant": _sample_dominant,
    "zero": _sample_zero,
    "skewed-beta": _sample_skewed_beta,
}

CATEGORY_COUNTS: dict[str, int] = {
    "iid-uniform": 120,
    "asymmetric": 30,
    "tie": 10,
    "dominant": 10,
    "zero": 5,
    "skewed-beta": 25,
}


def build_battery() -> tuple[BatterySpec, tuple[ValuationProfile, ...]]:
    """Build the canonical battery deterministically."""
    rng = random.Random(CONSTRUCTION_SEED)
    profiles: list[ValuationProfile] = []
    for category, count in CATEGORY_COUNTS.items():
        sampler = CATEGORY_SAMPLERS[category]
        for _ in range(count):
            profiles.append(
                ValuationProfile(
                    profile_id=f"{category}-{len(profiles):04d}",
                    values=sampler(rng),
                    category=category,
                )
            )
    spec = BatterySpec(
        version=BATTERY_VERSION,
        size=len(profiles),
        construction_seed=CONSTRUCTION_SEED,
    )
    return spec, tuple(profiles)


def sample_mixture_values(rng: random.Random) -> tuple[float, ...]:
    """Draw one valuation vector from the battery's mixture distribution."""
    total = sum(CATEGORY_COUNTS.values())
    pick = rng.randrange(total)
    running = 0
    for category, count in CATEGORY_COUNTS.items():
        running += count
        if pick < running:
            return CATEGORY_SAMPLERS[category](rng)
    raise AssertionError("unreachable: mixture weights sum to total")


def category_counts(profiles: tuple[ValuationProfile, ...]) -> dict[str, int]:
    """Count profiles per category (used by the demo and tests)."""
    counts: dict[str, int] = {}
    for p in profiles:
        counts[p.category] = counts.get(p.category, 0) + 1
    return counts
