"""Typed contracts for the SWE-smith bug-injection benchmark pipeline (agenteval-bench#37).

A benchmark *instance* is a modified version of a target repo (stored as a
patch) that breaks at least one of the repo's own tests. Correctness is
defined by the system: a proposed fix either makes the tests pass or it
does not. No hand-written task lists anywhere in the coding-agent
dimension: every instance is produced by a seeded or explicitly-recorded
generator, and the curation step selects by config criteria only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BugsmithError(Exception):
    """Base error for the bugsmith pipeline. Never silently degraded."""


class BugStrategy(str, Enum):
    """Bug-generation strategies. They target different failure classes."""

    PROCEDURAL_AST = "procedural_ast"  # deterministic AST-level fault injection
    LLM_GENERATED = "llm_generated"  # model rewrites a function to inject a bug
    PR_MIRROR = "pr_mirror"  # revert a real pull request to recreate history


@dataclass(frozen=True)
class GenerationRecord:
    """Provenance that makes an instance reproducible from its inputs."""

    strategy: BugStrategy
    seed: int | None  # None only for PR_MIRROR (explicit patch, not seeded)
    generator_version: str
    target_file: str  # repo-relative path of the mutated file
    site_description: str  # human-readable location of the injected fault
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BugCandidate:
    """One generated (not yet validated) bug."""

    record: GenerationRecord
    patch: str  # unified diff, repo-relative paths, applies to a clean checkout

    @property
    def patch_sha(self) -> str:
        import hashlib

        return hashlib.sha256(self.patch.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of running the target repo's test suite with a patch applied."""

    candidate: BugCandidate
    baseline_passed: tuple[str, ...]  # test node IDs passing on the clean tree
    failed_after: tuple[str, ...]  # test node IDs failing/erroring with patch
    passed_after: tuple[str, ...]
    collection_error: bool = False

    @property
    def fail_to_pass(self) -> tuple[str, ...]:
        """Tests that passed clean and fail with the bug: the task's target set."""
        before = set(self.baseline_passed)
        return tuple(t for t in self.failed_after if t in before)

    @property
    def pass_to_pass(self) -> tuple[str, ...]:
        """Tests that pass both clean and buggy: the regression guard set."""
        after = set(self.passed_after)
        return tuple(t for t in self.baseline_passed if t in after)

    @property
    def is_valid_instance(self) -> bool:
        """Only instances that break at least one test survive."""
        return len(self.fail_to_pass) > 0


@dataclass(frozen=True)
class BenchmarkInstance:
    """A validated, curated benchmark instance ready for agent evaluation."""

    instance_id: str  # f"{strategy.value}__{patch_sha}"
    record: GenerationRecord
    patch: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    repo_digest: str  # sha256 of the target repo file set, pins the substrate

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "strategy": self.record.strategy.value,
            "seed": self.record.seed,
            "generator_version": self.record.generator_version,
            "target_file": self.record.target_file,
            "site_description": self.record.site_description,
            "extra": dict(self.record.extra),
            "patch": self.patch,
            "fail_to_pass": list(self.fail_to_pass),
            "pass_to_pass": list(self.pass_to_pass),
            "repo_digest": self.repo_digest,
        }

    @staticmethod
    def from_dict(doc: Mapping[str, Any]) -> BenchmarkInstance:
        record = GenerationRecord(
            strategy=BugStrategy(doc["strategy"]),
            seed=doc["seed"],
            generator_version=doc["generator_version"],
            target_file=doc["target_file"],
            site_description=doc["site_description"],
            extra=dict(doc.get("extra", {})),
        )
        return BenchmarkInstance(
            instance_id=doc["instance_id"],
            record=record,
            patch=doc["patch"],
            fail_to_pass=tuple(doc["fail_to_pass"]),
            pass_to_pass=tuple(doc["pass_to_pass"]),
            repo_digest=doc["repo_digest"],
        )


@dataclass(frozen=True)
class CurationConfig:
    """Config-driven subset selection. No hand-picking, ever."""

    seed: int
    fail_to_pass_min: int = 1
    fail_to_pass_max: int = 5
    max_instances: int = 10
    strategy_quota: Mapping[str, int] = field(default_factory=dict)
    # e.g. {"procedural_ast": 6, "pr_mirror": 2}; empty = no per-strategy cap

    def __post_init__(self) -> None:
        if self.fail_to_pass_min < 1:
            raise BugsmithError("fail_to_pass_min must be >= 1: only breaking instances survive")
        if self.fail_to_pass_max < self.fail_to_pass_min:
            raise BugsmithError("fail_to_pass_max < fail_to_pass_min")
        if self.max_instances < 1:
            raise BugsmithError("max_instances must be >= 1")
        for name in self.strategy_quota:
            try:
                BugStrategy(name)
            except ValueError:
                raise BugsmithError(f"unknown strategy in strategy_quota: {name!r}")
