"""Seeded end-to-end demo of Simpson-safe aggregation (agenteval-bench#36).

Runs entirely on fixed integer counts — no randomness:

1. Replicates the Kohavi, Tang & Xu ch.18 worked example: Treatment
   beats Control on Friday (2.30% vs 2.02%) and on Saturday (1.2% vs
   1.00%), but looks worse pooled, because the treatment share rose from
   1% to 50% between periods. check_aggregation must name
   ALLOCATION_SHIFT and SIMPSON_REVERSAL, and win_claim must REJECT.
2. The stable-allocation path: same 50/50 share in both periods,
   consistent direction — verdict valid, disaggregation mandatory.
3. Refusal drills: pooled-only claim impossible, non-exhaustive
   decomposition named, allocation shift refused even without reversal,
   zero-delta stratum not a flip, raising seam carries the report.

Exit 0 only if every expectation holds; any failure raises, so CI can
gate on this demo directly.
"""

from __future__ import annotations

import sys

from experiment.simpson import (
    AggregationViolation,
    Stratum,
    assert_aggregation,
    check_aggregation,
    disaggregate,
    win_claim,
)


def kohavi_example() -> tuple[Stratum, Stratum]:
    """Friday/Saturday from Kohavi ch.18 as integer counts.

    Rates reproduce the book's figures to within 0.01pp:
    Friday 2.30% vs 2.02% at 1% treatment share;
    Saturday 1.2% vs 1.00% at 50% treatment share.
    """
    friday = Stratum(
        name="friday",
        n_treated=1000,
        n_control=99000,
        treated_successes=23,  # 2.30%
        control_successes=2000,  # 2.0202%
    )
    saturday = Stratum(
        name="saturday",
        n_treated=5000,
        n_control=5000,
        treated_successes=60,  # 1.20%
        control_successes=50,  # 1.00%
    )
    return friday, saturday


def demo_reversal() -> None:
    friday, saturday = kohavi_example()
    rep = disaggregate((friday, saturday))
    fri, sat = rep.strata
    assert abs(fri.treated_rate - 0.0230) < 0.0001, fri.treated_rate
    assert abs(fri.control_rate - 0.0202) < 0.0001, fri.control_rate
    assert abs(sat.treated_rate - 0.0120) < 0.0001, sat.treated_rate
    assert abs(sat.control_rate - 0.0100) < 0.0001, sat.control_rate
    assert abs(fri.treated_share - 0.01) < 1e-9
    assert abs(sat.treated_share - 0.50) < 1e-9
    print(f"friday:   treated {fri.treated_rate:.2%} vs control {fri.control_rate:.2%}")
    print(f"saturday: treated {sat.treated_rate:.2%} vs control {sat.control_rate:.2%}")
    print(
        f"pooled:   treated {rep.pooled_treated_rate:.2%} "
        f"vs control {rep.pooled_control_rate:.2%}"
    )
    # Treated wins each period, loses pooled: the reversal.
    assert fri.delta > 0 and sat.delta > 0
    assert rep.pooled_delta < 0, "expected the pooled reversal"

    verdict = check_aggregation((friday, saturday))
    assert not verdict.valid
    assert "ALLOCATION_SHIFT" in verdict.violations
    assert "SIMPSON_REVERSAL" in verdict.violations
    print(f"check_aggregation violations: {verdict.violations} (expected)")

    # The disaggregated numbers are returned even on the refusal path.
    assert len(verdict.report.strata) == 2

    claim = win_claim(
        (friday, saturday),
        pooled_n_treated=rep.pooled_n_treated,
        pooled_n_control=rep.pooled_n_control,
    )
    assert not claim.accepted
    assert "SIMPSON_REVERSAL" in claim.violations
    print("win_claim: REJECTED (Simpson reversal) — as the issue demands")


def demo_stable() -> None:
    a = Stratum("period-a", 5000, 5000, 115, 101)  # 2.30% vs 2.02%
    b = Stratum("period-b", 5000, 5000, 60, 50)  # 1.20% vs 1.00%
    verdict = check_aggregation((a, b))
    assert verdict.valid, verdict.violations
    rep = assert_aggregation((a, b))
    assert len(rep.strata) == 2, "disaggregation is mandatory output"
    claim = win_claim((a, b), rep.pooled_n_treated, rep.pooled_n_control)
    assert claim.accepted, claim.violations
    print("stable allocation + consistent direction: win claim ACCEPTED")


def demo_drills() -> list[str]:
    fired: list[str] = []

    # 1. There is no pooled-only path: empty strata is a contract error.
    try:
        disaggregate(())
    except ValueError:
        fired.append("pooled-only-path-impossible")

    # 2. Non-exhaustive decomposition: claimed pooled totals do not match
    #    the stratum sums — an inconsistent win cannot be claimed.
    #    (Consistent-but-fabricated totals are outside the seam's reach:
    #    it verifies reported aggregates, not raw data.)
    a = Stratum("a", 5000, 5000, 115, 101)
    b = Stratum("b", 5000, 5000, 60, 50)
    claim = win_claim((a, b), pooled_n_treated=9999, pooled_n_control=10000)
    if not claim.accepted and "NON_EXHAUSTIVE_DECOMPOSITION" in claim.violations:
        fired.append("non-exhaustive-decomposition-named")

    # 3. Allocation shift refuses pooled reporting even when no reversal
    #    occurs — the mechanism, not just the symptom, is gated.
    c = Stratum("c", 100, 9900, 5, 100)  # treated wins, 1% share
    d = Stratum("d", 5000, 5000, 150, 100)  # treated wins, 50% share
    verdict = check_aggregation((c, d))
    if (
        not verdict.valid
        and "ALLOCATION_SHIFT" in verdict.violations
        and "SIMPSON_REVERSAL" not in verdict.violations
    ):
        fired.append("allocation-shift-refused-without-reversal")

    # 4. Zero-delta stratum is neutral, not a flip.
    e = Stratum("e", 5000, 5000, 100, 100)  # delta exactly 0
    f = Stratum("f", 5000, 5000, 120, 100)  # treated wins
    verdict = check_aggregation((e, f))
    if verdict.valid:
        fired.append("zero-delta-stratum-not-a-flip")

    # 5. The raising seam carries the disaggregated report even when it
    #    refuses — refusal cannot be used to hide the numbers.
    friday, saturday = kohavi_example()
    try:
        assert_aggregation((friday, saturday))
    except AggregationViolation as exc:
        if len(exc.report.strata) == 2:  # type: ignore[attr-defined]
            fired.append("refusal-carries-disaggregation")

    # 6. Duplicate stratum names are a contract error (slices must be
    #    distinct to be exhaustive).
    try:
        disaggregate((a, a))
    except ValueError:
        fired.append("duplicate-stratum-refused")

    # 7. Boundary: spread exactly at tolerance passes, epsilon over fails.
    #    Shares are dyadic (0.5, 0.5625) so the spread is exact in binary.
    g = Stratum("g", 5000, 5000, 120, 100)
    h = Stratum("h", 5625, 4375, 135, 88)  # share 0.5625: spread 0.0625
    ok = check_aggregation((g, h), allocation_tolerance=0.0625)
    bad = check_aggregation((g, h), allocation_tolerance=0.0624)
    if ok.valid and "ALLOCATION_SHIFT" in bad.violations:
        fired.append("tolerance-boundary-exact")

    return fired


def main() -> int:
    demo_reversal()
    demo_stable()
    fired = demo_drills()
    print(f"refusal drills fired ({len(fired)}/7):")
    for name in fired:
        print(f"  - {name}")
    assert len(fired) == 7, f"expected 7 drills to fire, got {fired}"
    print("demo: all expectations hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
