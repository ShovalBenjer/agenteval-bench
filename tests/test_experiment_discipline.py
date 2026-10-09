"""Adversarial tests for experiment discipline (agenteval-bench#32).

Every mechanism the issue demands is pinned by a test that FAILS if the
mechanism is removed or weakened: peeking refusal, FWER inflation of
naive peeking vs control under alpha-spending, CUPED variance reduction
with unbiasedness, control-group-only theta, post-treatment covariate
refusal, and the registration seam's named violations.
"""

from __future__ import annotations

import math
import os
import random
import tempfile
from itertools import pairwise

import pytest

from experiment.alpha_spend import (
    PeekRefused,
    SequentialGate,
    group_sequential_boundaries,
    inv_normal_cdf,
    obrien_fleming_spend,
    spending_report,
)
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

PRE = CovariateSpec("pre_spend", "pre_experiment")
POST = CovariateSpec("post_spend", "post_treatment")


def _plan(**kw) -> ExperimentPlan:
    base = {"name": "test-plan", "n_planned": 800}
    base.update(kw)
    return ExperimentPlan(**base)  # type: ignore[arg-type]


@pytest.fixture()
def registry():
    with tempfile.TemporaryDirectory() as tmp:
        yield os.path.join(tmp, "plans.jsonl")


# ---------------------------------------------------------------------------
# Plan contracts
# ---------------------------------------------------------------------------


class TestPlanValidation:
    def test_rejects_non_increasing_looks(self):
        with pytest.raises(ValueError):
            _plan(looks=(0.5, 0.5, 1.0))

    def test_rejects_final_look_not_at_one(self):
        with pytest.raises(ValueError):
            _plan(looks=(0.5, 0.9))

    def test_rejects_bad_alpha(self):
        with pytest.raises(ValueError):
            _plan(alpha=1.5)

    def test_rejects_tiny_n(self):
        with pytest.raises(ValueError):
            _plan(n_planned=10)


# ---------------------------------------------------------------------------
# Inverse normal CDF
# ---------------------------------------------------------------------------


class TestInvNormalCdf:
    def test_known_quantiles(self):
        assert inv_normal_cdf(0.975) == pytest.approx(1.9599639845, abs=1e-6)
        assert inv_normal_cdf(0.5) == pytest.approx(0.0, abs=1e-9)
        assert inv_normal_cdf(0.025) == pytest.approx(-1.9599639845, abs=1e-6)

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            inv_normal_cdf(0.0)
        with pytest.raises(ValueError):
            inv_normal_cdf(1.0)


# ---------------------------------------------------------------------------
# Alpha-spending boundaries
# ---------------------------------------------------------------------------


class TestAlphaSpending:
    def test_spend_at_one_equals_alpha(self):
        assert obrien_fleming_spend(1.0, 0.05) == pytest.approx(0.05, abs=1e-9)

    def test_spend_increases_with_t(self):
        assert obrien_fleming_spend(0.25, 0.05) < obrien_fleming_spend(0.5, 0.05)

    def test_one_sided_matches_published_of(self):
        # Secondary-only oracle: the tabulated O'Brien-Fleming K=2
        # boundaries (2.96, 1.97) are the one-sided alpha=0.025 form of the
        # OF-like spending function. See module docstring for the convention.
        b = group_sequential_boundaries((0.5, 1.0), 0.025, sides="one")
        assert b[0] == pytest.approx(2.96, abs=0.05)
        assert b[1] == pytest.approx(1.97, abs=0.05)

    def test_boundaries_decrease_of_shape(self):
        b = group_sequential_boundaries((0.25, 0.5, 0.75, 1.0), 0.05)
        assert all(x > y for x, y in pairwise(b)), b
        assert b[0] > 3.0  # conservative early
        assert 1.9 < b[-1] < 2.1  # near-nominal at the final look

    def test_exit_increments_match_spending(self):
        # Pins the spending arithmetic behind spending_report (cumulative
        # exits are built from spending increments by construction). This
        # does NOT test the boundary solver — the Monte Carlo test below
        # is the independent pin of the solved boundaries.
        looks = (0.5, 1.0)
        rows = spending_report(looks, 0.05)
        prev_spent = 0.0
        for t, _b, inc in rows:
            expected = obrien_fleming_spend(t, 0.05) - prev_spent
            assert inc == pytest.approx(expected, abs=2e-3)
            prev_spent += expected

    def test_total_exit_equals_alpha(self):
        rows = spending_report((1 / 3, 2 / 3, 1.0), 0.05)
        assert sum(inc for _, _, inc in rows) == pytest.approx(0.05, abs=2e-3)

    def test_solver_exits_match_spending_monte_carlo(self):
        # Independent pin of the boundary SOLVER: simulate the canonical
        # joint distribution of sequential z-statistics directly (no
        # grid, no bisection — a different code path from the solver)
        # and check the realized per-look exit fractions match the
        # spending increments. A subtly wrong integration fails here.
        rng = random.Random(1234)
        looks = (0.5, 1.0)
        b = group_sequential_boundaries(looks, 0.05)  # two-sided
        n = 200000
        e1 = e2 = 0
        r = math.sqrt(0.5)
        s = math.sqrt(0.5)
        for _ in range(n):
            z1 = rng.gauss(0.0, 1.0)
            z2 = r * z1 + s * rng.gauss(0.0, 1.0)
            if abs(z1) >= b[0]:
                e1 += 1
            elif abs(z2) >= b[1]:
                e2 += 1
        # Spending increments: 0.00557 at t=0.5, 0.04443 at t=1.0.
        # Bands are ~6-12 sigma (seeded, deterministic) — wide enough
        # for grid quantization, tight enough to catch a broken solver.
        assert e1 / n == pytest.approx(0.00557, abs=0.002)
        assert e2 / n == pytest.approx(0.04443, abs=0.003)


# ---------------------------------------------------------------------------
# SequentialGate: the no-peeking running check
# ---------------------------------------------------------------------------


class TestSequentialGate:
    def test_unregistered_peek_refused(self):
        gate = SequentialGate(_plan())
        with pytest.raises(PeekRefused):
            gate.look(0.37, 2.5)

    def test_duplicate_look_refused(self):
        gate = SequentialGate(_plan())
        gate.look(0.5, 0.1)
        with pytest.raises(PeekRefused):
            gate.look(0.5, 0.1)

    def test_out_of_order_look_refused(self):
        gate = SequentialGate(_plan(looks=(0.25, 0.5, 1.0)))
        with pytest.raises(PeekRefused):
            gate.look(0.5, 0.1)

    def test_registered_looks_in_order_ok(self):
        gate = SequentialGate(_plan())
        d1 = gate.look(0.5, 0.1)
        d2 = gate.look(1.0, 0.1)
        assert not d1.reject and not d2.reject
        assert gate.pending == ()
        assert d1.boundary > d2.boundary  # OF: early boundary stricter

    def test_huge_z_rejects(self):
        gate = SequentialGate(_plan())
        assert gate.look(0.5, 10.0).reject

    def test_fraudulent_boundary_bypass_impossible(self):
        # A caller cannot smuggle a decision past the gate: the decision is
        # re-derived from z inside look(), never accepted as an argument.
        gate = SequentialGate(_plan())
        d = gate.look(0.5, 0.1)
        assert d.z == 0.1 and d.reject is False


# ---------------------------------------------------------------------------
# FWER: naive peeking inflates, alpha-spending controls (seeded)
# ---------------------------------------------------------------------------


def _null_zs(rng: random.Random, n_half: int) -> list[float]:
    def zstat(t: list[float], c: list[float]) -> float:
        m1, m2 = sum(t) / len(t), sum(c) / len(c)
        v1 = sum((x - m1) ** 2 for x in t) / (len(t) - 1)
        v2 = sum((x - m2) ** 2 for x in c) / (len(c) - 1)
        pooled = ((len(t) - 1) * v1 + (len(c) - 1) * v2) / (len(t) + len(c) - 2)
        return (m1 - m2) / math.sqrt(pooled * (1 / len(t) + 1 / len(c)))

    treated = [rng.gauss(0, 1) for _ in range(2 * n_half)]
    control = [rng.gauss(0, 1) for _ in range(2 * n_half)]
    return [
        zstat(treated[:n_half], control[:n_half]),
        zstat(treated, control),
    ]


class TestPeekingFwer:
    def test_naive_peeking_inflates_fwer(self):
        # The issue's premise, measured: one uncorrected interim peek at the
        # nominal 5% level pushes the false-positive rate well above 5%.
        rng = random.Random(7)
        fp = sum(
            any(abs(z) >= 1.96 for z in _null_zs(rng, 200)) for _ in range(500)
        )
        fwer = fp / 500
        assert fwer > 0.065, f"peeking should inflate FWER, got {fwer}"

    def test_alpha_spending_controls_fwer(self):
        rng = random.Random(7)
        plan = _plan()
        fp = 0
        for _ in range(500):
            gate = SequentialGate(plan)
            if any(gate.look(f, z).reject for f, z in zip((0.5, 1.0), _null_zs(rng, 200))):
                fp += 1
        fwer = fp / 500
        assert 0.025 < fwer < 0.075, f"OF spending should control FWER, got {fwer}"


# ---------------------------------------------------------------------------
# CUPED
# ---------------------------------------------------------------------------


def _cuped_data(seed: int, n: int = 1500, delta: float = 0.2, gamma: float = 2.0):
    rng = random.Random(seed)
    xc = [rng.gauss(0, 1) for _ in range(n)]
    xt = [rng.gauss(0, 1) for _ in range(n)]
    yc = [gamma * x + rng.gauss(0, 1) for x in xc]
    yt = [delta + gamma * x + rng.gauss(0, 1) for x in xt]
    return yc, yt, xc, xt


class TestCuped:
    def test_variance_reduction_measured(self):
        yc, yt, xc, xt = _cuped_data(11)
        res = cuped_adjust(yc, yt, xc, xt, PRE)
        assert res.variance_ratio < 0.4, f"expected variance cut, got {res.variance_ratio}"

    def test_unbiased_for_true_effect(self):
        yc, yt, xc, xt = _cuped_data(11, delta=0.2)
        res = cuped_adjust(yc, yt, xc, xt, PRE)
        assert res.adjusted_diff == pytest.approx(0.2, abs=0.06)

    def test_theta_from_control_group_only(self):
        # White-box pin against the leakage: theta must equal the
        # control-group slope, not a pooled slope. Corrupt the treatment
        # arm's X-Y relationship; theta must not move.
        yc, yt, xc, xt = _cuped_data(13)
        xt_bad = [x + 5.0 for x in xt]  # shift treatment covariates
        res = cuped_adjust(yc, yt, xc, xt_bad, PRE)
        mxc = sum(xc) / len(xc)
        myc = sum(yc) / len(yc)
        expected = sum((x - mxc) * (y - myc) for x, y in zip(xc, yc)) / sum(
            (x - mxc) ** 2 for x in xc
        )
        assert res.theta == pytest.approx(expected, rel=1e-9)

    def test_post_treatment_covariate_refused(self):
        yc, yt, xc, xt = _cuped_data(17)
        with pytest.raises(CovariateViolation):
            cuped_adjust(yc, yt, xc, xt, POST)

    def test_zero_variance_covariate_refused(self):
        with pytest.raises(ValueError):
            cuped_adjust([1.0, 2.0, 3.0], [1.0, 2.0, 3.0],
                         [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], PRE)

    def test_mismatched_lengths_refused(self):
        with pytest.raises(ValueError):
            cuped_adjust([1.0, 2.0], [1.0, 2.0], [1.0], [1.0, 2.0], PRE)


# ---------------------------------------------------------------------------
# Registration seam
# ---------------------------------------------------------------------------


def _evidence(plan: ExperimentPlan, digest: str, registry: str) -> ExperimentEvidence:
    gate = SequentialGate(plan)
    d1 = gate.look(0.5, 0.2)
    d2 = gate.look(1.0, 0.4)
    rng = random.Random(23)
    n = plan.n_planned // 2
    xc = [rng.gauss(0, 1) for _ in range(n)]
    xt = [rng.gauss(0, 1) for _ in range(n)]
    yc = [2.0 * x + rng.gauss(0, 1) for x in xc]
    yt = [0.2 + 2.0 * x + rng.gauss(0, 1) for x in xt]
    cuped = cuped_adjust(yc, yt, xc, xt, PRE) if plan.covariate else None
    return ExperimentEvidence(
        n_final=plan.n_planned,
        looks=(
            LookEvidence(0.5, d1.z, d1.reject),
            LookEvidence(1.0, d2.z, d2.reject),
        ),
        cuped=cuped,
    )


class TestRegistrationSeam:
    def test_compliant_experiment_verifies(self, registry):
        plan = _plan(covariate=PRE)
        digest = register_plan(plan, registry)
        verdict = verify_experiment(digest, registry, _evidence(plan, digest, registry))
        assert verdict.valid and verdict.violations == ()

    def test_unregistered_plan_named(self, registry):
        plan = _plan()
        digest = register_plan(plan, registry)
        bad = verify_experiment("f" * 64, registry, _evidence(plan, digest, registry))
        assert not bad.valid and bad.violations == ("UNREGISTERED_PLAN",)

    def test_sample_deviation_named(self, registry):
        plan = _plan()
        digest = register_plan(plan, registry)
        ev = _evidence(plan, digest, registry)
        ev = ExperimentEvidence(n_final=int(plan.n_planned * 1.1), looks=ev.looks)
        bad = verify_experiment(digest, registry, ev)
        assert not bad.valid and "SAMPLE_DEVIATION" in bad.violations

    def test_sample_within_tolerance_ok(self, registry):
        plan = _plan(n_tolerance=0.02)
        digest = register_plan(plan, registry)
        ev = _evidence(plan, digest, registry)
        ev = ExperimentEvidence(n_final=int(plan.n_planned * 1.01), looks=ev.looks)
        assert verify_experiment(digest, registry, ev).valid

    def test_decision_mismatch_caught(self, registry):
        plan = _plan()
        digest = register_plan(plan, registry)
        gate = SequentialGate(plan)
        gate.look(0.5, 0.2)
        d2 = gate.look(1.0, 8.0)  # really rejects
        assert d2.reject
        ev = ExperimentEvidence(
            n_final=plan.n_planned,
            looks=(LookEvidence(0.5, 0.2, False), LookEvidence(1.0, 8.0, False)),
        )
        bad = verify_experiment(digest, registry, ev)
        assert not bad.valid and "DECISION_MISMATCH" in bad.violations

    def test_look_schedule_mismatch_caught(self, registry):
        plan = _plan()
        digest = register_plan(plan, registry)
        ev = ExperimentEvidence(
            n_final=plan.n_planned,
            looks=(LookEvidence(0.5, 0.2, False),),  # missing the final look
        )
        bad = verify_experiment(digest, registry, ev)
        assert not bad.valid and "LOOK_SCHEDULE_MISMATCH" in bad.violations

    def test_missing_cuped_named(self, registry):
        plan = _plan(covariate=PRE)
        digest = register_plan(plan, registry)
        ev = _evidence(plan, digest, registry)
        ev = ExperimentEvidence(n_final=ev.n_final, looks=ev.looks, cuped=None)
        bad = verify_experiment(digest, registry, ev)
        assert not bad.valid and "CUPED_REQUIRED" in bad.violations

    def test_check_experiment_raises_for_ci(self, registry):
        plan = _plan(covariate=PRE)
        digest = register_plan(plan, registry)
        ev = _evidence(plan, digest, registry)
        ev = ExperimentEvidence(n_final=ev.n_final, looks=ev.looks, cuped=None)
        with pytest.raises(ExperimentViolation, match="CUPED_REQUIRED"):
            check_experiment(digest, registry, ev)

    def test_rewritten_registry_detected(self, registry):
        plan = _plan()
        digest = register_plan(plan, registry)
        with open(registry, "a", encoding="utf-8") as f:
            f.write('{"digest": "forged", "plan": {"name": "evil"}}\n')
        bad = verify_experiment(
            digest, registry, ExperimentEvidence(n_final=800, looks=())
        )
        assert not bad.valid and bad.violations == ("UNREGISTERED_PLAN",)

    def test_truncated_registry_named_not_raised(self, registry):
        plan = _plan()
        digest = register_plan(plan, registry)
        with open(registry, "w", encoding="utf-8") as f:
            f.write('{"digest": "abc", "plan": {"name":\n')  # truncated JSON
        bad = verify_experiment(
            digest, registry, ExperimentEvidence(n_final=800, looks=())
        )
        assert not bad.valid and bad.violations == ("REGISTRY_UNREADABLE",)

    def test_malformed_record_named_not_raised(self, registry):
        plan = _plan()
        digest = register_plan(plan, registry)
        with open(registry, "w", encoding="utf-8") as f:
            f.write('{"digest": "x", "plan": 42}\n')  # valid JSON, wrong shape
        bad = verify_experiment(
            digest, registry, ExperimentEvidence(n_final=800, looks=())
        )
        assert not bad.valid and bad.violations == ("REGISTRY_UNREADABLE",)

    def test_cuped_coverage_mismatch_named(self, registry):
        plan = _plan(covariate=PRE)
        digest = register_plan(plan, registry)
        ev = _evidence(plan, digest, registry)
        # CUPED computed on a different sample than the experiment's n.
        rng = random.Random(99)
        xc = [rng.gauss(0, 1) for _ in range(50)]
        xt = [rng.gauss(0, 1) for _ in range(50)]
        yc = [x + rng.gauss(0, 1) for x in xc]
        yt = [x + rng.gauss(0, 1) for x in xt]
        wrong = cuped_adjust(yc, yt, xc, xt, PRE)
        ev = ExperimentEvidence(n_final=ev.n_final, looks=ev.looks, cuped=wrong)
        bad = verify_experiment(digest, registry, ev)
        assert not bad.valid and "CUPED_COVERAGE_MISMATCH" in bad.violations


def _early_stop_evidence(plan: ExperimentPlan, z: float, claimed: bool):
    rng = random.Random(31)
    n_arm = plan.n_planned // 4  # n_final = n_planned * 0.5 -> half per arm
    xc = [rng.gauss(0, 1) for _ in range(n_arm)]
    xt = [rng.gauss(0, 1) for _ in range(n_arm)]
    yc = [2.0 * x + rng.gauss(0, 1) for x in xc]
    yt = [0.2 + 2.0 * x + rng.gauss(0, 1) for x in xt]
    cuped = cuped_adjust(yc, yt, xc, xt, PRE)
    return ExperimentEvidence(
        n_final=plan.n_planned // 2,
        looks=(LookEvidence(0.5, z, claimed),),
        cuped=cuped,
        stopped_at=0.5,
    )


class TestEarlyStopping:
    def test_early_stop_on_reject_verifies_clean(self, registry):
        plan = _plan(covariate=PRE)
        digest = register_plan(plan, registry)
        gate = SequentialGate(plan)
        assert gate.look(0.5, 8.0).reject  # really rejects
        ev = _early_stop_evidence(plan, z=8.0, claimed=True)
        assert verify_experiment(digest, registry, ev).valid

    def test_early_stop_without_reject_named(self, registry):
        plan = _plan(covariate=PRE)
        digest = register_plan(plan, registry)
        ev = _early_stop_evidence(plan, z=0.2, claimed=False)
        bad = verify_experiment(digest, registry, ev)
        assert not bad.valid and "EARLY_STOP_WITHOUT_REJECT" in bad.violations

    def test_early_stop_fabricated_reject_caught(self, registry):
        plan = _plan(covariate=PRE)
        digest = register_plan(plan, registry)
        ev = _early_stop_evidence(plan, z=0.2, claimed=True)  # claims reject, z says no
        bad = verify_experiment(digest, registry, ev)
        assert not bad.valid and "DECISION_MISMATCH" in bad.violations

    def test_early_stop_wrong_n_named(self, registry):
        plan = _plan(covariate=PRE)
        digest = register_plan(plan, registry)
        ev = _early_stop_evidence(plan, z=8.0, claimed=True)
        ev = ExperimentEvidence(
            n_final=plan.n_planned, looks=ev.looks, cuped=ev.cuped,
            stopped_at=0.5,
        )  # ran to full n but claims early stop
        bad = verify_experiment(digest, registry, ev)
        assert not bad.valid and "SAMPLE_DEVIATION" in bad.violations

    def test_early_stop_with_extra_look_named(self, registry):
        plan = _plan(covariate=PRE)
        digest = register_plan(plan, registry)
        ev = _early_stop_evidence(plan, z=8.0, claimed=True)
        ev = ExperimentEvidence(
            n_final=ev.n_final,
            looks=ev.looks + (LookEvidence(1.0, 0.1, False),),
            cuped=ev.cuped,
            stopped_at=0.5,
        )
        bad = verify_experiment(digest, registry, ev)
        assert not bad.valid and "LOOK_SCHEDULE_MISMATCH" in bad.violations

    def test_early_stop_unregistered_fraction(self, registry):
        plan = _plan(covariate=PRE)
        digest = register_plan(plan, registry)
        ev = _early_stop_evidence(plan, z=8.0, claimed=True)
        ev = ExperimentEvidence(
            n_final=ev.n_final, looks=ev.looks, cuped=ev.cuped,
            stopped_at=0.37,
        )
        bad = verify_experiment(digest, registry, ev)
        assert not bad.valid and "UNREGISTERED_PEEK" in bad.violations

    def test_stopped_at_one_is_full_schedule(self, registry):
        # stopped_at=1.0 after running every look is the full schedule,
        # not an early stop: a completed non-rejecting experiment with
        # stopped_at=1.0 verifies clean instead of EARLY_STOP_WITHOUT_REJECT.
        plan = _plan(covariate=PRE)
        digest = register_plan(plan, registry)
        ev = _evidence(plan, digest, registry)
        ev = ExperimentEvidence(
            n_final=ev.n_final, looks=ev.looks, cuped=ev.cuped, stopped_at=1.0
        )
        assert verify_experiment(digest, registry, ev).valid
