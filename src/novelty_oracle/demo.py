"""Stranger-runnable demo: ``python -m novelty_oracle.demo``.

Judges three candidates against two baselines (first-price, second-price)
and prints the typed verdicts:
  1. an exact copy of second-price        -> NOT_INVENTION, novelty ~ 0
  2. a behaviorally-identical but syntactically-restructured second-price
     (the anti-gaming case)               -> NOT_INVENTION, similarity ~ 1
  3. third-price (genuinely different rules) -> NOT_INVENTION, high novelty
     but negative quality delta (novel, not better)

No network. Deterministic.
"""

from __future__ import annotations

import inspect

from .baselines import first_price, second_price, third_price
from .battery import category_counts
from .oracle import NoveltyOracle
from .types import Mechanism, OracleConfig, OracleVerdict


def _obfuscated_second_price_allocate(values: tuple[float, ...]) -> tuple[int, ...]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda pair: pair[1], reverse=True)
    champ = indexed[0][0]
    return tuple(1 if pos == champ else 0 for pos in range(len(values)))


def _obfuscated_second_price_pay(
    values: tuple[float, ...], allocation: tuple[int, ...]
) -> tuple[float, ...]:
    ranked = sorted(values, reverse=True)
    runner_up = ranked[1] if len(ranked) > 1 else 0.0
    out: list[float] = []
    for idx in range(len(values)):
        out.append(runner_up if allocation[idx] == 1 else 0.0)
    return tuple(out)


def obfuscated_second_price() -> Mechanism:
    """Second-price auction, restructured: different names, different flow."""
    src = "\n".join(
        inspect.getsource(fn)
        for fn in (_obfuscated_second_price_allocate, _obfuscated_second_price_pay)
    )
    return Mechanism(
        name="second-price-obfuscated",
        allocate=_obfuscated_second_price_allocate,
        pay=_obfuscated_second_price_pay,
        description="Second-price logic, rewritten with different syntax.",
        source=src,
    )


def _print_verdict(title: str, verdict: OracleVerdict) -> None:
    print(f"--- {title} ---")
    print(f"  {verdict.summary()}")
    q = verdict.evidence.quality
    print(
        f"  candidate payoff {q.candidate.mean_payoff:.4f} vs "
        f"best baseline '{q.best_baseline_name}'"
    )
    for name, stats in q.baselines.items():
        print(f"    baseline {name:14s} payoff {stats.mean_payoff:.4f}")
    n = verdict.evidence.novelty
    print(
        f"  novelty {n.novelty_score:.3f} "
        f"({n.n_disagreements_vs_nearest}/{n.n_profiles} profiles differ "
        f"vs '{n.nearest_baseline_name}', AST pre-filter hit: {n.ast_prefilter_hit})"
    )
    print()


def main() -> None:
    config = OracleConfig(n_seeds=16, draws_per_seed=1000)
    oracle = NoveltyOracle(config)
    baselines = (first_price(), second_price())
    spec_counts = category_counts(oracle.battery)
    print(
        f"battery v{oracle.battery_version}: "
        + ", ".join(f"{k}={v}" for k, v in sorted(spec_counts.items()))
    )
    print()

    _print_verdict(
        "candidate: exact copy of second-price",
        oracle.judge(second_price(), baselines),
    )
    _print_verdict(
        "candidate: obfuscated second-price (anti-gaming)",
        oracle.judge(obfuscated_second_price(), baselines),
    )
    _print_verdict(
        "candidate: third-price (genuinely novel rules)",
        oracle.judge(third_price(), baselines),
    )


if __name__ == "__main__":
    main()
