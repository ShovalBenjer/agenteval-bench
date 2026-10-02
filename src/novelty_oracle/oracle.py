"""The Novelty Oracle (Law Article 1).

``NoveltyOracle.judge`` runs the full pipeline:
  1. validate the candidate (deterministic, well-formed outcomes);
  2. measure quality: seeded revenue simulation vs every baseline;
  3. measure novelty: behavioral distance over the canonical battery;
  4. decide: INVENTION / NOT_INVENTION / ABSTAIN, with evidence attached.

The decision rule (pure function ``decide_verdict``, unit-tested):
  beats  := delta CI lies strictly above quality_margin (statistically
            better than the best baseline);
  novel  := novelty_score >= novelty_threshold (behaviorally distant
            from every known baseline);
  INVENTION      iff beats and novel;
  ABSTAIN        iff novel and the quality CI straddles the margin
                 (novel, but cannot tell whether better — never a guess);
  NOT_INVENTION  otherwise. In particular a behaviorally-identical copy
                 is NOT_INVENTION even when its quality CI straddles zero:
                 novelty is decided exactly, so there is nothing to
                 abstain about.
"""

from __future__ import annotations

from .battery import build_battery
from .behavior import ast_identical, behavioral_distance, validate_mechanism
from .simulator import delta_ci, simulate
from .types import (
    Mechanism,
    NoveltyReport,
    OracleConfig,
    OracleVerdict,
    QualityReport,
    SeedStats,
    ValuationProfile,
    VerdictEvidence,
    VerdictKind,
    empty_evidence_placeholder,
)


def decide_verdict(
    delta: float,
    ci_low: float,
    ci_high: float,
    novelty_score: float,
    config: OracleConfig,
) -> VerdictKind:
    """Pure decision logic over measured (delta, CI, novelty)."""
    beats = ci_low > config.quality_margin
    novel = novelty_score >= config.novelty_threshold
    if beats and novel:
        return VerdictKind.INVENTION
    if novel and ci_low <= config.quality_margin <= ci_high:
        return VerdictKind.ABSTAIN
    return VerdictKind.NOT_INVENTION


class NoveltyOracle:
    """Typed service: candidate mechanism in, typed verdict out."""

    def __init__(self, config: OracleConfig | None = None) -> None:
        self._config = config or OracleConfig()
        self._spec, self._battery = build_battery()

    @property
    def battery(self) -> tuple[ValuationProfile, ...]:
        return self._battery

    @property
    def battery_version(self) -> int:
        return self._spec.version

    @property
    def config(self) -> OracleConfig:
        return self._config

    def judge(
        self,
        candidate: Mechanism,
        baselines: tuple[Mechanism, ...],
    ) -> OracleVerdict:
        """Judge a candidate mechanism against named baselines."""
        if not baselines:
            raise ValueError("judge requires at least one baseline mechanism")
        config = self._config

        validate_mechanism(candidate)
        for baseline in baselines:
            validate_mechanism(baseline)

        # Quality: seeded simulation for candidate and every baseline.
        candidate_stats = simulate(candidate, config)
        baseline_stats: dict[str, SeedStats] = {}
        for baseline in baselines:
            baseline_stats[baseline.name] = simulate(baseline, config)
        best_name = max(baseline_stats, key=lambda n: baseline_stats[n].mean_payoff)
        delta, d_lo, d_hi = delta_ci(candidate_stats, baseline_stats[best_name])
        quality = QualityReport(
            candidate=candidate_stats,
            baselines=baseline_stats,
            best_baseline_name=best_name,
            quality_delta=delta,
            delta_ci_low=d_lo,
            delta_ci_high=d_hi,
        )

        # Novelty: behavioral distance to each baseline over the battery.
        distances: dict[str, float] = {}
        disagreeing_vs: dict[str, tuple[str, ...]] = {}
        for baseline in baselines:
            dist, disagreeing = behavioral_distance(
                candidate, baseline, self._battery, config.epsilon
            )
            distances[baseline.name] = dist
            disagreeing_vs[baseline.name] = disagreeing
        nearest = min(distances, key=lambda n: distances[n])
        novelty_score = distances[nearest]
        nearest_idx = [b.name for b in baselines].index(nearest)
        prefilter_hit = ast_identical(candidate, baselines[nearest_idx])
        novelty = NoveltyReport(
            distances=distances,
            nearest_baseline_name=nearest,
            novelty_score=novelty_score,
            battery_version=self._spec.version,
            n_profiles=self._spec.size,
            n_disagreements_vs_nearest=len(disagreeing_vs[nearest]),
            sample_disagreeing_profiles=disagreeing_vs[nearest][:5],
            ast_prefilter_hit=prefilter_hit,
        )

        kind = decide_verdict(delta, d_lo, d_hi, novelty_score, config)
        evidence = VerdictEvidence(quality=quality, novelty=novelty, config=config)
        return OracleVerdict(
            kind=kind,
            quality_delta=delta,
            delta_ci=(d_lo, d_hi),
            novelty_score=novelty_score,
            nearest_baseline=nearest,
            evidence=evidence,
        )


__all__ = ["NoveltyOracle", "decide_verdict", "empty_evidence_placeholder"]
