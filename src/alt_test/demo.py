"""Seeded synthetic demo of the alt-test benchmark.

Generates a discrete-label dataset with known ground truth: annotators
with graded noise rates and four synthetic judges — a majority-vote
oracle (Theorem 1: must reach rho = 1.0), a sharp judge, a noisy judge,
and a coin-flip judge. Runs the full procedure and prints the
leaderboard. Exit 0 on success; the demo asserts the expected ordering
(majority-oracle on top with rho 1.0, coin-flip not justified).
"""

from __future__ import annotations

import random

from alt_test.runner import render_text, run_alt_test
from alt_test.types import AltTestConfig, AltTestItem

CATEGORIES = ["entailment", "neutral", "contradiction"]


def make_dataset(
    n_items: int = 120,
    n_annotators: int = 5,
    seed: int = 7,
) -> list[AltTestItem]:
    """Build a synthetic discrete dataset with graded annotator noise."""
    rng = random.Random(seed)
    annotator_noise = [0.05, 0.10, 0.15, 0.20, 0.30][:n_annotators]
    items: list[AltTestItem] = []
    for i in range(n_items):
        truth = rng.choice(CATEGORIES)
        annotators = {}
        for a, noise in enumerate(annotator_noise):
            if rng.random() < noise:
                annotators[f"ann{a}"] = rng.choice([c for c in CATEGORIES if c != truth])
            else:
                annotators[f"ann{a}"] = truth
        majority = max(CATEGORIES, key=lambda c: sum(1 for v in annotators.values() if v == c))
        judges = {
            "majority-oracle": majority,
            "sharp-judge": truth if rng.random() > 0.08 else rng.choice(CATEGORIES),
            "noisy-judge": truth if rng.random() > 0.35 else rng.choice(CATEGORIES),
            "coinflip-judge": rng.choice(CATEGORIES),
        }
        items.append(
            AltTestItem(
                id=f"item-{i:03d}",
                task="discrete",
                annotator_labels=annotators,
                judge_labels=judges,
            )
        )
    return items


def main() -> int:
    items = make_dataset()
    config = AltTestConfig(epsilon=0.1, q=0.05, seed=7)
    report = run_alt_test(items, config)
    print(render_text(report))
    by_name = {v.judge: v for v in report.verdicts}
    assert by_name["majority-oracle"].rho == 1.0, "Theorem 1 pin: majority oracle must reach rho=1.0"
    assert report.verdicts[0].judge == "majority-oracle", "oracle must top the leaderboard"
    assert not by_name["coinflip-judge"].justified, "coin-flip judge must not be justified"
    assert by_name["sharp-judge"].rho > by_name["noisy-judge"].rho > by_name["coinflip-judge"].rho
    print("demo checks passed: oracle rho=1.0, leaderboard ordered, coin-flip not justified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
