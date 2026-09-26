"""Core eval runner — orchestrates suite execution."""

from __future__ import annotations

import random
from collections.abc import Callable

from agenteval_bench.models import EvalResult, EvalSuite, RunResult
from agenteval_bench.scoring import DeterministicScorer

AgentFn = Callable[[str], str]

DEFAULT_SEED = 42


class EvalRunner:
    """Runs eval suites against an agent function."""

    def __init__(self, scorer: DeterministicScorer | None = None) -> None:
        self.scorer = scorer or DeterministicScorer()

    def run(self, suite: EvalSuite, agent_fn: AgentFn, seed: int = DEFAULT_SEED) -> RunResult:
        """Execute all cases in the suite and return aggregate results.

        A ``random.Random(seed)`` stream is bound to the run and one draw per
        case is recorded in ``details["rng_draw"]``. The deterministic scorer
        needs no randomness today; the draw pins the stream so any future
        stochastic component (subset sampling, tie-breaking) replays
        bit-exactly under the same seed.
        """
        rng = random.Random(seed)
        results: list[EvalResult] = []

        for case in suite.cases:
            rng_draw = rng.random()
            if case.skip:
                results.append(
                    EvalResult(case_id=case.id, passed=False, score=0.0,
                               details={"skipped": True, "rng_draw": rng_draw})
                )
                continue

            agent_output = agent_fn(case.input)
            result = self.scorer.score(case, agent_output)
            result.details["rng_draw"] = rng_draw
            results.append(result)

        total = len(results)
        skipped = sum(1 for r in results if r.details.get("skipped"))
        scored = [r for r in results if not r.details.get("skipped")]
        passed = sum(1 for r in scored if r.passed)
        failed = len(scored) - passed
        pass_rate = passed / len(scored) if scored else 0.0

        return RunResult(
            suite_name=suite.name,
            results=results,
            total=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            pass_rate=pass_rate,
        )

    def run_ci(self, suite: EvalSuite, agent_fn: AgentFn, threshold: float = 1.0) -> RunResult:
        """Run in CI mode — returns result with pass_rate for threshold check."""
        result = self.run(suite, agent_fn)
        return result
