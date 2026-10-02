"""Typed contracts for the Novelty Oracle (Law Article 1).

A mechanism is an allocation rule plus a payment rule with a fixed signature.
The oracle measures two things, independently:
  quality  = simulator payoff of the candidate vs the best baseline (with a
             confidence interval over seeds);
  novelty  = BEHAVIORAL distance: allocation/payment compared as functions
             over the canonical valuation-profile battery (fraction of
             profiles where outcomes differ beyond epsilon). AST distance is
             only a cheap pre-filter, never the primary measure.

No ``Any`` at boundaries. Verdict states are a closed enum, not strings.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

# Fixed mechanism signature: values -> allocation, (values, allocation) -> payments.
AllocateFn = Callable[[tuple[float, ...]], tuple[int, ...]]
PayFn = Callable[[tuple[float, ...], tuple[int, ...]], tuple[float, ...]]


class VerdictKind(Enum):
    """Closed verdict states of the oracle."""

    INVENTION = "INVENTION"
    NOT_INVENTION = "NOT_INVENTION"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class Mechanism:
    """A single-parameter auction mechanism: allocation + payment rules."""

    name: str
    allocate: AllocateFn
    pay: PayFn
    description: str = ""
    # Optional source text for the AST pre-filter. When None, the oracle
    # skips the AST shortcut and goes straight to behavioral comparison.
    source: str | None = None


@dataclass(frozen=True)
class ValuationProfile:
    """One canonical valuation vector the battery measures behavior on."""

    profile_id: str
    values: tuple[float, ...]
    category: str


@dataclass(frozen=True)
class BatterySpec:
    """Versioned identity of the canonical battery."""

    version: int
    size: int
    construction_seed: int


@dataclass(frozen=True)
class OracleConfig:
    """Tunable-but-typed oracle parameters."""

    n_seeds: int = 32
    draws_per_seed: int = 2000
    epsilon: float = 1e-9
    novelty_threshold: float = 0.05
    quality_margin: float = 0.0
    min_seeds: int = 8
    base_seed: int = 20261003


@dataclass(frozen=True)
class SeedStats:
    """Quality measurement statistics over seeds."""

    n_seeds: int
    draws_per_seed: int
    mean_payoff: float
    std_of_seed_means: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class QualityReport:
    """Candidate payoff vs every baseline, with the winner identified."""

    candidate: SeedStats
    baselines: dict[str, SeedStats]
    best_baseline_name: str
    quality_delta: float  # candidate mean - best baseline mean
    delta_ci_low: float
    delta_ci_high: float


@dataclass(frozen=True)
class NoveltyReport:
    """Behavioral distance of the candidate to each baseline."""

    # distance[baseline_name] = fraction of battery profiles where the
    # candidate's (allocation, payments) differ beyond epsilon.
    distances: dict[str, float]
    nearest_baseline_name: str
    novelty_score: float  # min over baselines of distance
    battery_version: int
    n_profiles: int
    n_disagreements_vs_nearest: int
    sample_disagreeing_profiles: tuple[str, ...]
    ast_prefilter_hit: bool


@dataclass(frozen=True)
class VerdictEvidence:
    """Everything the verdict rests on, attached to the verdict."""

    quality: QualityReport
    novelty: NoveltyReport
    config: OracleConfig


@dataclass(frozen=True)
class OracleVerdict:
    """The oracle's typed verdict."""

    kind: VerdictKind
    quality_delta: float
    delta_ci: tuple[float, float]
    novelty_score: float
    nearest_baseline: str
    evidence: VerdictEvidence

    def summary(self) -> str:
        lo, hi = self.delta_ci
        return (
            f"verdict={self.kind.value} "
            f"quality_delta={self.quality_delta:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}]) "
            f"novelty={self.novelty_score:.3f} "
            f"nearest_baseline={self.nearest_baseline}"
        )


class MechanismError(Exception):
    """Base class for typed mechanism validation failures."""


@dataclass(frozen=True)
class InvalidOutcome(MechanismError):
    """The mechanism returned a malformed (allocation, payments) pair."""

    mechanism_name: str
    profile_id: str
    reason: str


@dataclass(frozen=True)
class NonDeterministicMechanism(MechanismError):
    """The mechanism gave different outputs for identical inputs."""

    mechanism_name: str
    profile_id: str


@dataclass(frozen=True)
class HeldoutPair:
    """One labeled pair for the falsifiable held-out classification test."""

    left: Mechanism
    right: Mechanism
    expect_same: bool
    label: str


@dataclass(frozen=True)
class HeldoutResult:
    """Outcome of the held-out classification run."""

    n_pairs: int
    n_correct: int
    accuracy: float
    passed: bool  # accuracy >= 0.90
    per_pair: tuple[tuple[str, bool, bool], ...]  # (label, expected_same, predicted_same)


def empty_evidence_placeholder() -> VerdictEvidence:
    """Placeholder evidence for verdicts constructed without a full run.

    Only used by the pure decision-logic unit tests, never by judge().
    """
    cfg = OracleConfig()
    stats = SeedStats(
        n_seeds=0,
        draws_per_seed=0,
        mean_payoff=0.0,
        std_of_seed_means=0.0,
        ci_low=0.0,
        ci_high=0.0,
    )
    quality = QualityReport(
        candidate=stats,
        baselines={},
        best_baseline_name="",
        quality_delta=0.0,
        delta_ci_low=0.0,
        delta_ci_high=0.0,
    )
    novelty = NoveltyReport(
        distances={},
        nearest_baseline_name="",
        novelty_score=0.0,
        battery_version=0,
        n_profiles=0,
        n_disagreements_vs_nearest=0,
        sample_disagreeing_profiles=(),
        ast_prefilter_hit=False,
    )
    return VerdictEvidence(quality=quality, novelty=novelty, config=cfg)


# Re-exported sequence type for callers that build batteries.
ProfileSequence = Sequence[ValuationProfile]
MechanismSequence = Sequence[Mechanism]


# Keep dataclass field defaults honest: this module intentionally defines no
# mutable module-level state.
__all__ = [
    "AllocateFn",
    "BatterySpec",
    "HeldoutPair",
    "HeldoutResult",
    "InvalidOutcome",
    "Mechanism",
    "MechanismError",
    "NonDeterministicMechanism",
    "NoveltyReport",
    "OracleConfig",
    "OracleVerdict",
    "PayFn",
    "QualityReport",
    "SeedStats",
    "ValuationProfile",
    "VerdictEvidence",
    "VerdictKind",
    "empty_evidence_placeholder",
]
