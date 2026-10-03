"""Behavioral comparison of mechanisms (Law Article 1, Reading-2 amendment).

Novelty is measured as BEHAVIORAL distance: two mechanisms are compared
as functions — run both on every canonical battery profile and count the
fraction of profiles where (allocation, payments) differ beyond epsilon.

The AST pre-filter is only a cheap shortcut: if the normalized AST dumps
of two mechanisms' sources are identical, they are the same code and the
distance is 0 without running the battery. It is never the primary
measure — a renamed-variable copy must still score ~1.0 behavioral
similarity via the battery run, which is exactly the anti-gaming test.

Also: typed validation. Mechanisms must be deterministic and return
well-formed outcomes, else judging raises a typed MechanismError.
"""

from __future__ import annotations

import ast
import math

from .types import (
    InvalidOutcome,
    Mechanism,
    NonDeterministicMechanism,
    ValuationProfile,
)

_PROBE_PROFILES: tuple[tuple[float, ...], ...] = (
    (0.1, 0.9, 0.4),
    (0.5, 0.5, 0.5),
    (0.0, 0.0, 1.0),
    (0.99, 0.98, 0.97, 0.96),
)


def _check_outcome(
    mechanism: Mechanism,
    profile_id: str,
    values: tuple[float, ...],
    allocation: tuple[int, ...],
    payments: tuple[float, ...],
) -> None:
    n = len(values)
    if len(allocation) != n or len(payments) != n:
        raise InvalidOutcome(
            mechanism.name,
            profile_id,
            f"length mismatch: values={n} allocation={len(allocation)} payments={len(payments)}",
        )
    if any(a not in (0, 1) for a in allocation):
        raise InvalidOutcome(mechanism.name, profile_id, "allocation not in {0,1}")
    if sum(allocation) > 1:
        raise InvalidOutcome(mechanism.name, profile_id, "more than one winner")
    if any(p < 0.0 for p in payments):
        raise InvalidOutcome(mechanism.name, profile_id, "negative payment")
    if any(math.isnan(p) for p in payments):
        raise InvalidOutcome(mechanism.name, profile_id, "NaN payment")


def run_mechanism(
    mechanism: Mechanism, profile_id: str, values: tuple[float, ...]
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Run a mechanism once, with outcome validation."""
    allocation = mechanism.allocate(values)
    payments = mechanism.pay(values, allocation)
    _check_outcome(mechanism, profile_id, values, allocation, payments)
    return allocation, payments


def validate_mechanism(mechanism: Mechanism) -> None:
    """Raise InvalidOutcome / NonDeterministicMechanism on bad mechanisms."""
    for idx, values in enumerate(_PROBE_PROFILES):
        pid = f"probe-{idx}"
        first = run_mechanism(mechanism, pid, values)
        second = run_mechanism(mechanism, pid, values)
        if first != second:
            raise NonDeterministicMechanism(mechanism.name, pid)


def _normalized_ast(source: str) -> str:
    """Parse source and return a normalized AST dump (no positions)."""
    tree = ast.parse(source)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def ast_identical(left: Mechanism, right: Mechanism) -> bool:
    """Cheap pre-filter: True only when both sources parse to the same AST."""
    if left.source is None or right.source is None:
        return False
    try:
        return _normalized_ast(left.source) == _normalized_ast(right.source)
    except SyntaxError:
        return False


def outcomes_differ(
    left: tuple[tuple[int, ...], tuple[float, ...]],
    right: tuple[tuple[int, ...], tuple[float, ...]],
    epsilon: float,
) -> bool:
    """True when allocation or any payment differs beyond epsilon."""
    alloc_l, pay_l = left
    alloc_r, pay_r = right
    if alloc_l != alloc_r:
        return True
    return any(abs(a - b) > epsilon for a, b in zip(pay_l, pay_r))


def behavioral_distance(
    candidate: Mechanism,
    baseline: Mechanism,
    battery: tuple[ValuationProfile, ...],
    epsilon: float,
) -> tuple[float, tuple[str, ...]]:
    """Fraction of battery profiles where outcomes differ; plus disagreeing ids.

    The AST pre-filter short-circuits exact code duplicates to 0.0.
    """
    if ast_identical(candidate, baseline):
        return 0.0, ()
    disagreeing: list[str] = []
    for profile in battery:
        out_c = run_mechanism(candidate, profile.profile_id, profile.values)
        out_b = run_mechanism(baseline, profile.profile_id, profile.values)
        if outcomes_differ(out_c, out_b, epsilon):
            disagreeing.append(profile.profile_id)
    return len(disagreeing) / len(battery), tuple(disagreeing)
