"""SWE-smith bug-injection benchmark pipeline (agenteval-bench#37).

Turn a repo into an executable benchmark via bug injection: generate
modified versions of the codebase (stored as patches) that break the
repo's own tests, validate them inside Docker, and curate a config-driven
subset with FAIL_TO_PASS / PASS_TO_PASS lists.
"""

from bugsmith.buggen import (
    LLMGeneratedBugGenerator,
    PRMirrorGenerator,
    ProceduralBugGenerator,
)
from bugsmith.curate import select_subset, to_instance
from bugsmith.harness import DockerRunner, LocalRunner, baseline, validate
from bugsmith.images import TargetImage, build_image, dockerfile_text, repo_digest
from bugsmith.patch import PatchError, apply_patch
from bugsmith.types import (
    BenchmarkInstance,
    BugCandidate,
    BugsmithError,
    BugStrategy,
    CurationConfig,
    GenerationRecord,
    ValidationReport,
)

__all__ = [
    "BenchmarkInstance",
    "BugCandidate",
    "BugStrategy",
    "BugsmithError",
    "CurationConfig",
    "DockerRunner",
    "GenerationRecord",
    "LLMGeneratedBugGenerator",
    "LocalRunner",
    "PRMirrorGenerator",
    "PatchError",
    "ProceduralBugGenerator",
    "TargetImage",
    "ValidationReport",
    "apply_patch",
    "baseline",
    "build_image",
    "dockerfile_text",
    "repo_digest",
    "select_subset",
    "to_instance",
    "validate",
]
