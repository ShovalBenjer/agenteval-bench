"""Experiment discipline: pre-registration, no-peeking enforcement, CUPED.

Implements agenteval-bench#32 (Kohavi, Tang & Xu, *Trustworthy Online
Controlled Experiments*): model-comparison experiments pre-register their
sample size and stopping rule; interim peeks are allowed only on the
registered look schedule and spend alpha through a Lan-DeMets
O'Brien-Fleming spending function; CUPED variance reduction applies where
the plan declares a pre-experiment covariate. Violations are refused
fail-closed so CI can gate on them.

Enforcement boundary (stated honestly): discipline is enforced at the
`registry.verify_experiment` / `registry.check_experiment` seam, which
replays evidence through a fresh `SequentialGate` and re-derives every
decision. The pure math helpers (`obrien_fleming_spend`,
`group_sequential_boundaries`, `cuped_adjust`) stay directly callable —
tests pin the seam, and the seam is what CI runs. The verifier is a
consistency checker over reported evidence (fractions vs plan, decisions
vs reported z's, arm n's vs n_final); it does not see raw data.

Aggregation discipline (agenteval-bench#36, Kohavi ch. 18) is enforced at
the `simpson.check_aggregation` / `simpson.assert_aggregation` /
`simpson.win_claim` seam: pooled win reporting is refused on allocation
shift across slices, and a win claim that flips direction in any stratum
of an exhaustive decomposition is rejected. There is no pooled-only
path to a win claim — `disaggregate` is mandatory output on every path.
"""

from __future__ import annotations

from experiment.alpha_spend import (
    LookDecision,
    PeekRefused,
    SequentialGate,
    group_sequential_boundaries,
    inv_normal_cdf,
    obrien_fleming_spend,
    spending_report,
)
from experiment.cuped import CovariateViolation, CupedResult, cuped_adjust
from experiment.registry import (
    ExperimentEvidence,
    ExperimentVerdict,
    ExperimentViolation,
    LookEvidence,
    check_experiment,
    load_plan,
    register_plan,
    verify_experiment,
)
from experiment.simpson import (
    DEFAULT_ALLOCATION_TOLERANCE,
    AggregationReport,
    AggregationVerdict,
    AggregationViolation,
    Stratum,
    StratumReport,
    WinVerdict,
    assert_aggregation,
    check_aggregation,
    disaggregate,
    pooled_rates,
    stratum_delta,
    win_claim,
)
from experiment.types import CovariateSpec, ExperimentPlan

__all__ = [
    "DEFAULT_ALLOCATION_TOLERANCE",
    "AggregationReport",
    "AggregationVerdict",
    "AggregationViolation",
    "CovariateSpec",
    "CovariateViolation",
    "CupedResult",
    "ExperimentEvidence",
    "ExperimentPlan",
    "ExperimentVerdict",
    "ExperimentViolation",
    "LookDecision",
    "LookEvidence",
    "PeekRefused",
    "SequentialGate",
    "Stratum",
    "StratumReport",
    "WinVerdict",
    "assert_aggregation",
    "check_aggregation",
    "check_experiment",
    "cuped_adjust",
    "disaggregate",
    "group_sequential_boundaries",
    "inv_normal_cdf",
    "load_plan",
    "obrien_fleming_spend",
    "pooled_rates",
    "register_plan",
    "spending_report",
    "stratum_delta",
    "verify_experiment",
    "win_claim",
]
