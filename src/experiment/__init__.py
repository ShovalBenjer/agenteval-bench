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
from experiment.types import CovariateSpec, ExperimentPlan

__all__ = [
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
    "check_experiment",
    "cuped_adjust",
    "group_sequential_boundaries",
    "inv_normal_cdf",
    "load_plan",
    "obrien_fleming_spend",
    "register_plan",
    "spending_report",
    "verify_experiment",
]
