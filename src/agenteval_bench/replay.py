"""Deterministic seeded replay for eval runs.

A ReplayLog pins everything needed to reproduce a run bit-exactly:
the seed, the suite digest, the scorer, and every per-case output and RNG
draw. Re-running the same suite with the same seed must produce
byte-identical log bytes; :func:`verify_replay` / the ``replay --check``
CLI command enforce that.

The seeded RNG is bound to the run even though the current deterministic
scorer needs no randomness: any future stochastic component (subset
sampling, tie-breaking) draws from this stream, and the draws are recorded
so a replay can prove which stream was used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

REPLAY_VERSION = 1
SCORER_NAME = "DeterministicScorer"


def suite_digest(suite_bytes: bytes) -> str:
    """sha256 hex digest of the raw suite YAML — the suite's identity."""
    return "sha256:" + hashlib.sha256(suite_bytes).hexdigest()


@dataclass
class ReplayCase:
    case_id: str
    input: str
    output: str
    score: float
    passed: bool
    skipped: bool = False
    rng_draw: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "input": self.input,
            "output": self.output,
            "score": self.score,
            "passed": self.passed,
            "skipped": self.skipped,
            "rng_draw": self.rng_draw,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReplayCase:
        return cls(
            case_id=d["case_id"],
            input=d["input"],
            output=d["output"],
            score=d["score"],
            passed=d["passed"],
            skipped=d.get("skipped", False),
            rng_draw=d.get("rng_draw"),
        )


@dataclass
class ReplayLog:
    seed: int
    suite_name: str
    suite_file: str
    digest: str
    agent: str = "recorded"
    scorer: str = SCORER_NAME
    cases: list[ReplayCase] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        passed = sum(1 for c in self.cases if c.passed and not c.skipped)
        failed = sum(1 for c in self.cases if not c.passed and not c.skipped)
        skipped = sum(1 for c in self.cases if c.skipped)
        return {
            "tool": "agenteval-bench",
            "replay_version": REPLAY_VERSION,
            "seed": self.seed,
            "suite_name": self.suite_name,
            "suite_file": self.suite_file,
            "suite_digest": self.digest,
            "agent": self.agent,
            "scorer": self.scorer,
            "rng": "random.Random",
            "cases": [c.to_dict() for c in self.cases],
            "summary": {
                "total": len(self.cases),
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
            },
        }

    def to_json(self) -> str:
        """Canonical serialization — byte-identical for identical runs."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, ensure_ascii=False) + "\n"

    def write(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_json(cls, text: str) -> ReplayLog:
        d = json.loads(text)
        if d.get("replay_version") != REPLAY_VERSION:
            raise ValueError(f"unsupported replay_version: {d.get('replay_version')}")
        return cls(
            seed=d["seed"],
            suite_name=d["suite_name"],
            suite_file=d.get("suite_file", ""),
            digest=d["suite_digest"],
            agent=d.get("agent", "recorded"),
            scorer=d.get("scorer", SCORER_NAME),
            cases=[ReplayCase.from_dict(c) for c in d["cases"]],
        )

    @classmethod
    def load(cls, path: str) -> ReplayLog:
        with open(path, encoding="utf-8") as f:
            return cls.from_json(f.read())


def find_mismatches(expected: ReplayLog, actual: ReplayLog) -> list[str]:
    """Compare two logs case-by-case; empty list means bit-exact replay."""
    problems: list[str] = []
    if expected.digest != actual.digest:
        problems.append(f"suite_digest: {expected.digest} != {actual.digest}")
    if expected.seed != actual.seed:
        problems.append(f"seed: {expected.seed} != {actual.seed}")
    if len(expected.cases) != len(actual.cases):
        problems.append(f"case count: {len(expected.cases)} != {len(actual.cases)}")
        return problems
    for e, a in zip(expected.cases, actual.cases):
        if e.case_id != a.case_id:
            problems.append(f"case order/id: {e.case_id} != {a.case_id}")
            continue
        for attr in ("input", "output", "score", "passed", "skipped", "rng_draw"):
            ev, av = getattr(e, attr), getattr(a, attr)
            if ev != av:
                problems.append(f"{e.case_id}.{attr}: {ev!r} != {av!r}")
    return problems
