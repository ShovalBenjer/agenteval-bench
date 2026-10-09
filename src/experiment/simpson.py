"""Simpson-safe aggregation: per-segment and per-period effect checks.

Implements agenteval-bench#36.

Source mechanism (established): Kohavi, Tang & Xu, *Trustworthy Online
Controlled Experiments* (2020), ch. 18 ("Simpson's Paradox"). Combining
results across periods with different allocation percentages can reverse
the apparent direction of an effect. The book's worked example: Treatment
beats Control on Friday (2.30% vs 2.02%) and on Saturday (1.2% vs 1.00%),
yet looks worse when the two days are pooled (1.20% vs 1.68%), because
the treatment share rose from 1% to 50% between periods. The same
reversal holds across segments: the OEC can increase for each of two
exhaustive segments yet decline overall. Pooling must never be the
default.

Adaptation (adaptation): model-comparison leaderboards pool scores across
eval runs, task categories, and time — exactly the conditions that
manufacture reversals. A "model X wins" claim computed by naive pooling
across runs with different task mixes or sampling rates is as
untrustworthy as the pooled Friday+Saturday estimate. Here a "slice" is a
period or a segment; "allocation" is the treated share within the slice,
which for leaderboards reads as the share of an agent's evaluated items
in each category or period.

Enforcement boundary (stated honestly): the seam is
``check_aggregation`` / ``assert_aggregation`` / ``win_claim`` below.
There is NO pooled-only path — a win claim can only be constructed from
strata, so pooled numbers can never be quoted without the disaggregated
numbers riding alongside them. The pure rate helpers
(``stratum_delta``, ``pooled_rates``) stay directly callable; the
seam functions are what tests pin and CI runs. This module verifies
reported aggregates; it does not see raw per-user data.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Default tolerance on the treatment-share spread across slices. A slice
#: pair whose treated shares differ by more than this refuses pooled
#: reporting. The book's worked example shifts 49pp; 5pp is the
#: conservative line: beyond it, pooled weights move enough to manufacture
#: a reversal from allocation alone.
DEFAULT_ALLOCATION_TOLERANCE = 0.05


class AggregationViolation(ValueError):
    """Pooled reporting is refused or a win claim is rejected (fail-closed)."""


@dataclass(frozen=True)
class Stratum:
    """One slice of an experiment: a period, a segment, a task category.

    Rates are derived from integer counts so the seam works on reported
    aggregates, not on prose. n_* > 0 and successes <= n are enforced at
    construction (contract-before-code on the named seam).
    """

    name: str
    n_treated: int
    n_control: int
    treated_successes: int
    control_successes: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stratum name must be non-empty")
        for field, value in (
            ("n_treated", self.n_treated),
            ("n_control", self.n_control),
            ("treated_successes", self.treated_successes),
            ("control_successes", self.control_successes),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field} must be a non-negative int, got {value!r}")
        if self.n_treated == 0 or self.n_control == 0:
            raise ValueError("each stratum needs both arms (n_treated>0, n_control>0)")
        if self.treated_successes > self.n_treated:
            raise ValueError("treated_successes cannot exceed n_treated")
        if self.control_successes > self.n_control:
            raise ValueError("control_successes cannot exceed n_control")


@dataclass(frozen=True)
class StratumReport:
    """Mandatory per-stratum disaggregation: never a win claim."""

    name: str
    n_treated: int
    n_control: int
    treated_share: float
    treated_rate: float
    control_rate: float
    delta: float  # treated_rate - control_rate, sign = direction of effect


@dataclass(frozen=True)
class AggregationReport:
    """Disaggregated numbers plus the pooled figures for transparency.

    The pooled figures are descriptive only — quoting a pooled *win*
    requires check_aggregation to pass and win_claim to accept.
    """

    strata: tuple[StratumReport, ...]
    pooled_n_treated: int
    pooled_n_control: int
    pooled_treated_rate: float
    pooled_control_rate: float
    pooled_delta: float
    pooled_treated_share: float


@dataclass(frozen=True)
class AggregationVerdict:
    """Outcome of check_aggregation: valid, or the named violations.

    report is attached on EVERY path — a refused pool still returns the
    disaggregated numbers, so refusal can never be used to hide them.
    """

    valid: bool
    violations: tuple[str, ...]
    notes: tuple[str, ...]
    report: AggregationReport


@dataclass(frozen=True)
class WinVerdict:
    """Outcome of win_claim: accepted, or rejected with named violations."""

    accepted: bool
    violations: tuple[str, ...]
    notes: tuple[str, ...]
    report: AggregationReport


def _rate(successes: int, n: int) -> float:
    return successes / n


def stratum_delta(s: Stratum) -> float:
    """Per-stratum effect direction: treated rate minus control rate."""
    return _rate(s.treated_successes, s.n_treated) - _rate(
        s.control_successes, s.n_control
    )


def _treated_share(s: Stratum) -> float:
    n = s.n_treated + s.n_control
    return s.n_treated / n


def disaggregate(strata: tuple[Stratum, ...]) -> AggregationReport:
    """Mandatory disaggregated output. Never a win claim."""
    if not strata:
        raise ValueError("need at least one stratum: there is no pooled-only path")
    seen: set[str] = set()
    for s in strata:
        if s.name in seen:
            raise ValueError(f"duplicate stratum name {s.name!r}")
        seen.add(s.name)
    reports = tuple(
        StratumReport(
            name=s.name,
            n_treated=s.n_treated,
            n_control=s.n_control,
            treated_share=_treated_share(s),
            treated_rate=_rate(s.treated_successes, s.n_treated),
            control_rate=_rate(s.control_successes, s.n_control),
            delta=stratum_delta(s),
        )
        for s in strata
    )
    n_t = sum(s.n_treated for s in strata)
    n_c = sum(s.n_control for s in strata)
    t_s = sum(s.treated_successes for s in strata)
    c_s = sum(s.control_successes for s in strata)
    pt = _rate(t_s, n_t)
    pc = _rate(c_s, n_c)
    return AggregationReport(
        strata=reports,
        pooled_n_treated=n_t,
        pooled_n_control=n_c,
        pooled_treated_rate=pt,
        pooled_control_rate=pc,
        pooled_delta=pt - pc,
        pooled_treated_share=n_t / (n_t + n_c),
    )


def pooled_rates(strata: tuple[Stratum, ...]) -> tuple[float, float]:
    """Descriptive pooled (treated_rate, control_rate). Not a win claim."""
    rep = disaggregate(strata)
    return rep.pooled_treated_rate, rep.pooled_control_rate


def check_aggregation(
    strata: tuple[Stratum, ...],
    *,
    allocation_tolerance: float = DEFAULT_ALLOCATION_TOLERANCE,
) -> AggregationVerdict:
    """Decide whether pooled reporting across slices is permitted.

    Refuses (violations) when:
    - ALLOCATION_SHIFT: treated share differs across slices by more than
      allocation_tolerance — the book's reversal mechanism;
    - SIMPSON_REVERSAL: the pooled delta's direction disagrees with a
      stratum delta's direction (a zero-delta stratum is neutral, not a
      flip).
    The report is returned on every path: refusal never hides the
    disaggregated numbers.
    """
    if not 0.0 <= allocation_tolerance <= 1.0:
        raise ValueError(
            f"allocation_tolerance must be in [0, 1], got {allocation_tolerance}"
        )
    report = disaggregate(strata)
    violations: list[str] = []
    notes: list[str] = []
    if len(strata) == 1:
        notes.append("single stratum: no pooling performed, nothing to refuse")
    else:
        shares = [r.treated_share for r in report.strata]
        spread = max(shares) - min(shares)
        if spread > allocation_tolerance:
            violations.append("ALLOCATION_SHIFT")
            notes.append(
                f"treated-share spread {spread:.4f} exceeds tolerance "
                f"{allocation_tolerance:.4f}: pooled reporting refused"
            )
        pooled_sign = _sign(report.pooled_delta)
        for r in report.strata:
            if _sign(r.delta) != 0 and _sign(r.delta) != pooled_sign:
                violations.append("SIMPSON_REVERSAL")
                notes.append(
                    f"pooled delta {report.pooled_delta:+.4f} disagrees with "
                    f"stratum {r.name!r} delta {r.delta:+.4f}: "
                    "pooled direction is a reversal, not a result"
                )
                break
    return AggregationVerdict(
        valid=not violations,
        violations=tuple(violations),
        notes=tuple(notes),
        report=report,
    )


def _sign(x: float) -> int:
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


def assert_aggregation(
    strata: tuple[Stratum, ...],
    *,
    allocation_tolerance: float = DEFAULT_ALLOCATION_TOLERANCE,
) -> AggregationReport:
    """check_aggregation, but violations raise AggregationViolation.

    The CI seam: call before quoting any pooled delta as a result.
    The raised exception carries the report — disaggregation stays
    mandatory even in the refusal path.
    """
    verdict = check_aggregation(strata, allocation_tolerance=allocation_tolerance)
    if not verdict.valid:
        exc = AggregationViolation(
            f"pooled reporting refused: {', '.join(verdict.violations)}"
        )
        exc.report = verdict.report  # type: ignore[attr-defined]
        raise exc
    return verdict.report


def win_claim(
    strata: tuple[Stratum, ...],
    pooled_n_treated: int,
    pooled_n_control: int,
    *,
    allocation_tolerance: float = DEFAULT_ALLOCATION_TOLERANCE,
) -> WinVerdict:
    """Rule on a 'treated beats control overall' claim from a decomposition.

    The decomposition must be exhaustive: the caller's pooled arm totals
    must reconcile with the stratum sums, else NON_EXHAUSTIVE_DECOMPOSITION
    — a win that survives only because a slice is missing is not a win.
    The claim is then accepted only if check_aggregation is clean AND the
    pooled delta favors treated. A claim that flips direction in any
    stratum is REJECTED, not averaged away.
    """
    verdict = check_aggregation(strata, allocation_tolerance=allocation_tolerance)
    violations = list(verdict.violations)
    notes = list(verdict.notes)
    rep = verdict.report
    if rep.pooled_n_treated != pooled_n_treated or rep.pooled_n_control != pooled_n_control:
        violations.append("NON_EXHAUSTIVE_DECOMPOSITION")
        notes.append(
            f"stratum arm sums ({rep.pooled_n_treated}, {rep.pooled_n_control}) "
            f"do not reconcile with claimed pooled totals "
            f"({pooled_n_treated}, {pooled_n_control})"
        )
    accepted = not violations and rep.pooled_delta > 0.0
    if not violations and rep.pooled_delta <= 0.0:
        notes.append(
            f"pooled delta {rep.pooled_delta:+.4f} does not favor treated: "
            "claim fails on the pooled numbers themselves"
        )
    return WinVerdict(
        accepted=accepted, violations=tuple(violations), notes=tuple(notes), report=rep
    )
