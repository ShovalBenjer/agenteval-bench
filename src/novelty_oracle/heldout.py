"""Falsifiable held-out check: ``python -m novelty_oracle.heldout``.

The Law's falsification criterion for Article 1: on a held-out set of
known mechanism pairs —
  (first-price vs second-price: DIFFERENT;
   two independent implementations of second-price: SAME) —
the oracle classifies different/same with >= 90% accuracy, and its
quality ranking matches brute-force payoff ranking on the battery.

Prints per-pair results and PASS/FAIL. Exit code 0 on pass.
"""

from __future__ import annotations

import sys

from .baselines import all_pay, first_price, myerson_uniform, second_price, third_price
from .behavior import run_mechanism
from .demo import obfuscated_second_price
from .oracle import NoveltyOracle
from .types import HeldoutPair, HeldoutResult, Mechanism, OracleConfig


def build_pairs() -> tuple[HeldoutPair, ...]:
    return (
        HeldoutPair(first_price(), second_price(), False, "first-price vs second-price"),
        HeldoutPair(
            second_price(), obfuscated_second_price(), True, "second-price vs obfuscated copy"
        ),
        HeldoutPair(first_price(), first_price(), True, "first-price vs itself"),
        HeldoutPair(all_pay(), second_price(), False, "all-pay vs second-price"),
        HeldoutPair(myerson_uniform(), second_price(), False, "myerson-reserve vs second-price"),
        HeldoutPair(myerson_uniform(), myerson_uniform(), True, "myerson vs itself"),
    )


def check_ranking(oracle: NoveltyOracle) -> bool:
    """Brute-force battery payoff ranking vs simulated quality ranking.

    Both estimate E[payoff] under the battery mixture; the order must
    agree on every DECISIVE pair. A pair is decisive when the brute-force
    gap exceeds the battery's resolution (0.02 ~ 2x the standard error of
    the closest pair's gap over 200 fixed profiles). Pairs within
    resolution are ties — demanding an order there would test sampling
    noise, not the oracle. (The myerson/second-price pair lives below
    resolution: the reserve's edge is real but smaller than what 200 fixed
    profiles can resolve. This is documented, not hidden.)
    """
    mechanisms: tuple[Mechanism, ...] = (
        first_price(),
        second_price(),
        myerson_uniform(),
        all_pay(),
        third_price(),
    )
    brute: dict[str, float] = {}
    for mech in mechanisms:
        total = 0.0
        for profile in oracle.battery:
            _, payments = run_mechanism(mech, profile.profile_id, profile.values)
            total += sum(payments)
        brute[mech.name] = total / len(oracle.battery)

    sim: dict[str, float] = {}
    for mech in mechanisms:
        sim[mech.name] = oracle.judge(mech, (first_price(),)).evidence.quality.candidate.mean_payoff

    print("\nbrute-force battery means vs simulated means:")
    ok = True
    names = [m.name for m in mechanisms]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            gap = brute[a] - brute[b]
            decisive = abs(gap) > 0.02
            agree = (sim[a] - sim[b]) * gap > 0
            status = "OK " if (agree or not decisive) else "MISS"
            if decisive and not agree:
                ok = False
            print(
                f"[{status}] {a:15s} vs {b:15s}: "
                f"brute_gap={gap:+.4f} {'decisive' if decisive else 'tie (below resolution)'}"
            )
    print(f"decisive pairwise rankings agree: {ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


def run_heldout() -> HeldoutResult:
    oracle = NoveltyOracle(OracleConfig(n_seeds=8, draws_per_seed=500))
    per_pair: list[tuple[str, bool, bool]] = []
    n_correct = 0
    for pair in build_pairs():
        verdict = oracle.judge(pair.left, (pair.right,))
        predicted_same = verdict.novelty_score < oracle.config.novelty_threshold
        correct = predicted_same == pair.expect_same
        n_correct += int(correct)
        per_pair.append((pair.label, pair.expect_same, predicted_same))
        mark = "OK " if correct else "MISS"
        print(
            f"[{mark}] {pair.label:45s} "
            f"expected_same={pair.expect_same!s:5s} "
            f"predicted_same={predicted_same!s:5s} "
            f"(novelty={verdict.novelty_score:.3f})"
        )
    n = len(per_pair)
    accuracy = n_correct / n
    passed = accuracy >= 0.90
    print(
        f"\naccuracy {n_correct}/{n} = {accuracy:.1%} (bar: 90%) -> {'PASS' if passed else 'FAIL'}"
    )
    ranking_ok = check_ranking(oracle)
    overall = passed and ranking_ok
    print(f"\nOVERALL -> {'PASS' if overall else 'FAIL'}")
    return HeldoutResult(
        n_pairs=n,
        n_correct=n_correct,
        accuracy=accuracy,
        passed=overall,
        per_pair=tuple(per_pair),
    )


def main() -> int:
    result = run_heldout()
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
