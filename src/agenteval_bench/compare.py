"""Run-vs-run comparison for regression detection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from agenteval_bench.models import RunResult

DEFAULT_REGRESSION_THRESHOLD = 0.05  # 5% degradation, per spec premortem #4


@dataclass
class ComparisonResult:
    """Outcome of comparing a candidate run against a baseline run."""

    baseline_suite: str
    candidate_suite: str
    baseline_pass_rate: float
    candidate_pass_rate: float
    delta: float  # candidate - baseline; negative means degradation
    threshold: float
    is_regression: bool
    regressed_cases: list[str] = field(default_factory=list)  # passed before, fail now
    improved_cases: list[str] = field(default_factory=list)  # failed before, pass now

    def summary(self) -> str:
        """Return a human-readable comparison summary."""
        lines = [
            f"Baseline:  {self.baseline_suite} (pass rate {self.baseline_pass_rate:.1%})",
            f"Candidate: {self.candidate_suite} (pass rate {self.candidate_pass_rate:.1%})",
            f"Delta: {self.delta:+.1%} (regression threshold: -{self.threshold:.1%})",
        ]
        if self.regressed_cases:
            lines.append(f"Regressed cases ({len(self.regressed_cases)}): "
                         + ", ".join(self.regressed_cases))
        if self.improved_cases:
            lines.append(f"Improved cases ({len(self.improved_cases)}): "
                         + ", ".join(self.improved_cases))
        lines.append("REGRESSION DETECTED" if self.is_regression else "No regression")
        return "\n".join(lines)


def load_run(path: str) -> RunResult:
    """Load a persisted run (JSON written via `--output`) from disk."""
    with open(path) as f:
        data = json.load(f)
    return RunResult.from_dict(data)


def compare_runs(
    baseline: RunResult,
    candidate: RunResult,
    threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> ComparisonResult:
    """Compare a candidate run against a baseline run.

    A regression is flagged when the candidate's pass rate drops more than
    `threshold` below the baseline's. Per-case transitions are reported for
    cases present in both runs; skipped cases are excluded.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")

    def scored_cases(run: RunResult) -> dict[str, bool]:
        return {
            r.case_id: r.passed
            for r in run.results
            if not r.details.get("skipped")
        }

    baseline_cases = scored_cases(baseline)
    candidate_cases = scored_cases(candidate)
    common = baseline_cases.keys() & candidate_cases.keys()

    regressed = sorted(c for c in common if baseline_cases[c] and not candidate_cases[c])
    improved = sorted(c for c in common if not baseline_cases[c] and candidate_cases[c])

    delta = candidate.pass_rate - baseline.pass_rate

    return ComparisonResult(
        baseline_suite=baseline.suite_name,
        candidate_suite=candidate.suite_name,
        baseline_pass_rate=baseline.pass_rate,
        candidate_pass_rate=candidate.pass_rate,
        delta=delta,
        threshold=threshold,
        is_regression=delta < -threshold,
        regressed_cases=regressed,
        improved_cases=improved,
    )
