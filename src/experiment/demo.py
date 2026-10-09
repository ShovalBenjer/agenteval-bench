"""Seeded end-to-end demo of experiment discipline (agenteval-bench#32).

Runs entirely on synthetic data with fixed seeds:

1. Registers a plan (n=800, alpha=0.05, looks at t=0.5 and t=1.0,
   pre-experiment covariate "pre_spend").
2. Shows the peeking problem: 2000 null A/B experiments, peeking at the
   nominal 5% level at each look, reject a FALSE positive far more often
   than 5% of the time.
3. Shows the fix: the same peeks through SequentialGate with
   O'Brien-Fleming alpha-spending keep the false-positive rate at ~5%.
4. Shows CUPED: with a pre-experiment covariate explaining most of the
   outcome variance, the adjusted estimator keeps the true effect while
   the variance collapses.
5. Violation drills: every refusal the issue demands (unregistered peek,
   duplicate look, unregistered plan, sample-size deviation, fabricated
   decision flag, post-treatment covariate, missing CUPED) is exercised
   and must fire.

Exit 0 only if every expectation holds; any failure raises, so CI can
gate on this demo directly.
"""

from __future__ import annotations

import math
import os
import random
import sys
import tempfile

from experiment.alpha_spend import PeekRefused, SequentialGate
from experiment.cuped import CovariateViolation, cuped_adjust
from experiment.registry import (
    ExperimentEvidence,
    ExperimentViolation,
    LookEvidence,
    check_experiment,
    register_plan,
    verify_experiment,
)
from experiment.types import CovariateSpec, ExperimentPlan

SEED = 20261009
N_EXP = 2000
N_PER_ARM = 400  # at t=1.0; t=0.5 uses the first half


def _zstat(t: list[float], c: list[float]) -> float:
    n1, n2 = len(t), len(c)
    m1 = sum(t) / n1
    m2 = sum(c) / n2
    v1 = sum((x - m1) ** 2 for x in t) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in c) / (n2 - 1)
    pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    return (m1 - m2) / math.sqrt(pooled * (1.0 / n1 + 1.0 / n2))


def _null_experiments(rng: random.Random) -> tuple[list[float], list[float]]:
    """One null experiment's (treated, control) outcomes at each look."""
    half = N_PER_ARM // 2
    treated = [rng.gauss(0.0, 1.0) for _ in range(N_PER_ARM)]
    control = [rng.gauss(0.0, 1.0) for _ in range(N_PER_ARM)]
    z_half = _zstat(treated[:half], control[:half])
    z_full = _zstat(treated, control)
    return [z_half, z_full], [treated, control]


def demo_peeking() -> tuple[float, float]:
    """Measured false-positive rates: naive peeking vs alpha-spending."""
    rng = random.Random(SEED)
    plan = ExperimentPlan(name="demo-null", n_planned=2 * N_PER_ARM, looks=(0.5, 1.0))
    naive_fp = 0
    of_fp = 0
    for _ in range(N_EXP):
        zs, _ = _null_experiments(rng)
        if any(abs(z) >= 1.96 for z in zs):
            naive_fp += 1
        gate = SequentialGate(plan)
        if any(gate.look(f, z).reject for f, z in zip((0.5, 1.0), zs)):
            of_fp += 1
    return naive_fp / N_EXP, of_fp / N_EXP


def demo_cuped() -> tuple[float, float]:
    """CUPED on seeded data: variance ratio and unbiasedness."""
    rng = random.Random(SEED + 1)
    n = 2000
    delta = 0.2
    xc = [rng.gauss(0.0, 1.0) for _ in range(n)]
    xt = [rng.gauss(0.0, 1.0) for _ in range(n)]
    yc = [2.0 * x + rng.gauss(0.0, 1.0) for x in xc]
    yt = [delta + 2.0 * x + rng.gauss(0.0, 1.0) for x in xt]
    res = cuped_adjust(yc, yt, xc, xt, CovariateSpec("pre_spend", "pre_experiment"))
    return res.variance_ratio, res.adjusted_diff - delta


def demo_drills(registry: str, digest: str, plan: ExperimentPlan) -> list[str]:
    """Every refusal the issue demands; each must fire. Returns drill names."""
    fired: list[str] = []

    gate = SequentialGate(plan)
    try:
        gate.look(0.37, 2.5)
    except PeekRefused:
        fired.append("unregistered-peek-refused")
    gate.look(0.5, 0.1)
    try:
        gate.look(0.5, 0.1)
    except PeekRefused:
        fired.append("duplicate-look-refused")
    try:
        gate.look(1.0, 0.1)  # in order after 0.5: fine, then...
        gate2 = SequentialGate(plan)
        gate2.look(1.0, 0.1)  # skip 0.5 -> out of order
    except PeekRefused:
        fired.append("out-of-order-look-refused")

    bad = verify_experiment("deadbeef" * 8, registry,
                            ExperimentEvidence(n_final=800, looks=()))
    if not bad.valid and bad.violations == ("UNREGISTERED_PLAN",):
        fired.append("unregistered-plan-named")

    ev = ExperimentEvidence(
        n_final=900,  # 12.5% over plan: beyond the 2% tolerance
        looks=(LookEvidence(0.5, 0.3, False), LookEvidence(1.0, 0.4, False)),
    )
    bad = verify_experiment(digest, registry, ev)
    if not bad.valid and "SAMPLE_DEVIATION" in bad.violations:
        fired.append("sample-deviation-named")

    gate3 = SequentialGate(plan)
    d1 = gate3.look(0.5, 0.2)
    d2 = gate3.look(1.0, 5.0)  # z=5 clears any boundary: really rejects
    assert d2.reject
    ev = ExperimentEvidence(
        n_final=800,
        looks=(
            LookEvidence(0.5, 0.2, d1.reject),
            LookEvidence(1.0, 5.0, False),  # fabricated: claims no reject
        ),
    )
    bad = verify_experiment(digest, registry, ev)
    if not bad.valid and "DECISION_MISMATCH" in bad.violations:
        fired.append("fabricated-decision-caught")

    try:
        cuped_adjust([1.0, 2.0], [1.5, 2.5], [0.1, 0.2], [0.1, 0.3],
                     CovariateSpec("post_spend", "post_treatment"))
    except CovariateViolation:
        fired.append("post-treatment-covariate-refused")

    ev = ExperimentEvidence(
        n_final=800,
        looks=(LookEvidence(0.5, 0.2, False), LookEvidence(1.0, 0.4, False)),
        cuped=None,  # plan declares a covariate: CUPED is required
    )
    bad = verify_experiment(digest, registry, ev)
    if not bad.valid and "CUPED_REQUIRED" in bad.violations:
        fired.append("missing-cuped-named")

    try:
        check_experiment(digest, registry, ev)
    except ExperimentViolation:
        fired.append("check-raises-for-ci")
    return fired


def main() -> int:
    plan = ExperimentPlan(
        name="demo-ab",
        n_planned=2 * N_PER_ARM,
        alpha=0.05,
        looks=(0.5, 1.0),
        covariate=CovariateSpec("pre_spend", "pre_experiment"),
    )
    with tempfile.TemporaryDirectory() as tmp:
        registry = os.path.join(tmp, "plans.jsonl")
        digest = register_plan(plan, registry)

        naive_fwer, of_fwer = demo_peeking()
        print(f"null experiments: {N_EXP}, looks at t=0.5 and t=1.0, nominal alpha=0.05")
        print(f"  naive peeking FWER:      {naive_fwer:.4f}  (must exceed 0.07: peeking inflates)")
        print(f"  OF alpha-spending FWER:  {of_fwer:.4f}  (must stay in [0.03, 0.07])")
        assert naive_fwer > 0.07, f"peeking should inflate FWER, got {naive_fwer}"
        assert 0.03 < of_fwer < 0.07, f"OF spending should control FWER, got {of_fwer}"

        ratio, bias = demo_cuped()
        print(f"CUPED: variance ratio {ratio:.3f} (must be < 0.35), "
              f"bias {bias:+.4f} (must be within 0.06 of 0)")
        assert ratio < 0.35, f"CUPED should cut variance, got ratio {ratio}"
        assert abs(bias) < 0.06, f"CUPED should stay unbiased, got bias {bias}"

        fired = demo_drills(registry, digest, plan)
        print(f"violation drills fired ({len(fired)}/9):")
        for name in fired:
            print(f"  - {name}")
        assert len(fired) == 9, f"expected 9 drills to fire, got {fired}"

        # The compliant path: correct evidence verifies clean.
        gate = SequentialGate(plan)
        d1 = gate.look(0.5, 0.2)
        d2 = gate.look(1.0, 0.4)
        rng = random.Random(SEED + 2)
        n = N_PER_ARM  # matches plan.n_planned = 2 * N_PER_ARM
        xc = [rng.gauss(0.0, 1.0) for _ in range(n)]
        xt = [rng.gauss(0.0, 1.0) for _ in range(n)]
        yc = [2.0 * x + rng.gauss(0.0, 1.0) for x in xc]
        yt = [0.2 + 2.0 * x + rng.gauss(0.0, 1.0) for x in xt]
        assert plan.covariate is not None
        cuped = cuped_adjust(yc, yt, xc, xt, plan.covariate)
        evidence = ExperimentEvidence(
            n_final=2 * n,
            looks=(
                LookEvidence(0.5, d1.z, d1.reject),
                LookEvidence(1.0, d2.z, d2.reject),
            ),
            cuped=cuped,
        )
        verdict = check_experiment(digest, registry, evidence)
        assert verdict.valid
        print("compliant experiment verifies clean: OK")
    print("demo: all expectations hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
