"""Config-driven subset curation (agenteval-bench#37).

``harness.gather`` + curated subset in the issue's pipeline: from validated
instances, select a subset by config criteria only. The selection is a
pure function of ``(config, validated)`` — same inputs, same subset,
always. There is no hand-picking step anywhere in this module.
"""

from __future__ import annotations

import hashlib
import random

from bugsmith.types import (
    BenchmarkInstance,
    BugsmithError,
    CurationConfig,
    ValidationReport,
)


def _instance_id(report: ValidationReport, repo_digest: str) -> str:
    return (f"{report.candidate.record.strategy.value}"
            f"__{repo_digest[:8]}__{report.candidate.patch_sha}")


def to_instance(report: ValidationReport, repo_digest: str) -> BenchmarkInstance:
    if not report.is_valid_instance:
        raise BugsmithError("only validated instances can be curated")
    return BenchmarkInstance(
        instance_id=_instance_id(report, repo_digest),
        record=report.candidate.record,
        patch=report.candidate.patch,
        fail_to_pass=report.fail_to_pass,
        pass_to_pass=report.pass_to_pass,
        repo_digest=repo_digest,
    )


def select_subset(
    config: CurationConfig,
    validated: list[ValidationReport],
    repo_digest: str,
) -> list[BenchmarkInstance]:
    """Select the curated subset. Deterministic in (config, validated)."""
    rng = random.Random(config.seed)
    pool = [to_instance(r, repo_digest) for r in validated if r.is_valid_instance]

    # Criterion filter: FAIL_TO_PASS band from the config.
    pool = [
        inst for inst in pool
        if config.fail_to_pass_min <= len(inst.fail_to_pass) <= config.fail_to_pass_max
    ]
    # Deduplicate by patch content: the same bug twice is one instance.
    seen: set[str] = set()
    deduped: list[BenchmarkInstance] = []
    for inst in pool:
        h = hashlib.sha256(inst.patch.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        deduped.append(inst)

    # Strategy quotas, then a seeded shuffle for a stable but unbiased order.
    quota = {s: int(n) for s, n in config.strategy_quota.items()}
    counts: dict[str, int] = {}
    chosen: list[BenchmarkInstance] = []
    order = deduped[:]
    rng.shuffle(order)
    for inst in order:
        name = inst.record.strategy.value
        if name in quota and counts.get(name, 0) >= quota[name]:
            continue
        counts[name] = counts.get(name, 0) + 1
        chosen.append(inst)
        if len(chosen) >= config.max_instances:
            break
    return chosen
