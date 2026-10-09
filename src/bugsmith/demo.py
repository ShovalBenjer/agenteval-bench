"""Seeded end-to-end demo of the SWE-smith pipeline (agenteval-bench#37).

Runs the full issue pipeline inside Docker, in two phases:

Phase 1 — vendored ``mcalc`` fixture target (fast, fully deterministic):
1. ``create_images`` — build the sealed execution image.
2. ``bug_gen`` — 8 procedural-AST candidates (fixed seed) + 1 PR-mirror
   candidate (a reverted historical-style patch).
3. ``harness.valid`` — validate every candidate in a fresh container;
   only instances that break >= 1 test survive, with FAIL_TO_PASS /
   PASS_TO_PASS lists.
4. ``harness.gather`` + curation — config-driven subset selection.

Phase 2 — a genuine ecosystem repo (``tkem/cachetools``, pinned tag,
shallow clone): the same generate → validate → curate pipeline must
produce at least one valid instance. This is the issue's criterion 1:
the pipeline runs against a real small Python repo from the ecosystem,
inside Docker — not just the fixture.

Exit 0 only if every acceptance expectation of the issue holds; any
failure raises, so CI can gate on this demo directly. Neither target
tree is ever modified (digests pinned before/after).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from bugsmith.buggen import PRMirrorGenerator, ProceduralBugGenerator
from bugsmith.curate import select_subset
from bugsmith.harness import DockerRunner, baseline, validate
from bugsmith.images import TargetImage, build_image, repo_digest
from bugsmith.types import (
    BenchmarkInstance,
    BugCandidate,
    BugsmithError,
    CurationConfig,
    ValidationReport,
)

SEED = 20261009
N_PROCEDURAL = 8
HERE = Path(__file__).resolve().parent
FIXTURE_REPO = HERE / "fixtures" / "target"
PR_MIRROR_PATCH = HERE / "fixtures" / "pr_mirror_median.patch"

# Phase 2: a real, small, dependency-free ecosystem package, pinned for
# reproducibility.
ECOSYSTEM_REPO_URL = "https://github.com/tkem/cachetools.git"
ECOSYSTEM_REPO_TAG = "v7.2.1"
ECOSYSTEM_SEED = 20261009
ECOSYSTEM_N_PROCEDURAL = 4


def _pr_mirror_candidate() -> BugCandidate:
    patch = PR_MIRROR_PATCH.read_text(encoding="utf-8")
    gen = PRMirrorGenerator(pr_ref="mcalc#41-revert")
    (candidate,) = gen.generate(patch, ["src/mcalc/stats.py"])
    return candidate


def _check_reports(reports: list[ValidationReport], baseline_passed: frozenset[str]) -> None:
    baseline_set = set(baseline_passed)
    for r in reports:
        if not r.is_valid_instance:
            continue
        assert r.fail_to_pass, "valid instance has empty FAIL_TO_PASS"
        assert set(r.fail_to_pass) <= baseline_set, (
            "FAIL_TO_PASS names tests that did not pass on the clean tree"
        )
        assert set(r.pass_to_pass) <= baseline_set
        assert not (set(r.fail_to_pass) & set(r.pass_to_pass)), (
            "FAIL_TO_PASS and PASS_TO_PASS overlap"
        )


def run_pipeline(
    repo_root: Path,
    work_dir: Path,
    *,
    label: str,
    seed: int,
    n_procedural: int,
    extra_candidates: list[BugCandidate] | None = None,
    min_valid: int = 1,
) -> tuple[list[BenchmarkInstance], str]:
    """Generate → validate (Docker) → curate. Returns (subset, repo_digest)."""
    digest = repo_digest(repo_root)
    print(f"[{label}] target repo digest: {digest}")

    image: TargetImage = build_image(repo_root, work_dir / f"image-{label}")
    print(f"[{label}] built image {image.tag}")
    assert image.repo_digest == digest
    runner = DockerRunner(image)

    passed = baseline(repo_root, runner, work_dir / f"runs-{label}")
    print(f"[{label}] baseline: {len(passed)} tests passing on the clean tree")
    assert passed, "refusing an empty benchmark"

    procedural = ProceduralBugGenerator(seed).generate(repo_root, n_procedural)
    candidates = procedural + (extra_candidates or [])
    print(f"[{label}] generated {len(procedural)} procedural "
          f"+ {len(extra_candidates or [])} extra candidates")
    assert len(procedural) == n_procedural, (
        f"generator starved: {len(procedural)}/{n_procedural}"
    )

    # Seed reproducibility (criterion 3): same seed -> byte-identical patches.
    again = ProceduralBugGenerator(seed).generate(repo_root, n_procedural)
    assert [c.patch for c in again] == [c.patch for c in procedural], (
        "procedural generation is not reproducible from the seed"
    )
    print(f"[{label}] seed reproducibility: OK (byte-identical patches)")

    run_work = work_dir / f"runs-{label}"
    reports = [validate(c, repo_root, runner, run_work, passed) for c in candidates]
    valid = [r for r in reports if r.is_valid_instance]
    print(f"[{label}] validated: {len(valid)}/{len(reports)} instances break >= 1 test")
    assert len(valid) >= min_valid, (
        f"fewer than {min_valid} valid instances survived"
    )
    _check_reports(reports, passed)
    print(f"[{label}] FAIL_TO_PASS / PASS_TO_PASS accounting: OK")

    config = CurationConfig(
        seed=seed,
        fail_to_pass_min=1,
        fail_to_pass_max=4,
        max_instances=6,
        strategy_quota={"procedural_ast": 5, "pr_mirror": 1},
    )
    subset = select_subset(config, reports, digest)
    subset2 = select_subset(config, reports, digest)
    assert subset, "curation produced an empty subset"
    assert [i.instance_id for i in subset] == [i.instance_id for i in subset2], (
        "curation is not deterministic"
    )
    generated_patches = {c.patch for c in candidates}
    for inst in subset:
        assert inst.patch in generated_patches, (
            f"curated instance {inst.instance_id} did not come from a generator"
        )
        assert 1 <= len(inst.fail_to_pass) <= 4
    print(f"[{label}] curated subset: {len(subset)} instances "
          f"({', '.join(i.instance_id for i in subset)})")
    return subset, digest


def _clone_ecosystem_repo(dest: Path) -> Path:
    try:
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", ECOSYSTEM_REPO_TAG,
             ECOSYSTEM_REPO_URL, str(dest)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
            text=True,
        )
    except OSError as e:
        raise BugsmithError(f"git clone failed: {e}") from e
    if proc.returncode != 0:
        raise BugsmithError(
            f"could not clone the ecosystem target "
            f"{ECOSYSTEM_REPO_URL}@{ECOSYSTEM_REPO_TAG}:\n{proc.stdout[-2000:]}"
        )
    return dest


def main() -> int:
    digest_before = repo_digest(FIXTURE_REPO)

    with tempfile.TemporaryDirectory(prefix="bugsmith-demo-") as tmp:
        work_dir = Path(tmp)

        # Phase 1: fixture target.
        subset, digest = run_pipeline(
            FIXTURE_REPO,
            work_dir,
            label="fixture",
            seed=SEED,
            n_procedural=N_PROCEDURAL,
            extra_candidates=[_pr_mirror_candidate()],
            min_valid=2,
        )
        manifest = {
            "tool": "agenteval-bench",
            "command": "bugsmith-demo",
            "seed": SEED,
            "repo_digest": digest,
            "curated": [i.to_dict() for i in subset],
        }
        (work_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        print(f"manifest: {len(manifest['curated'])} curated instances")

        # Phase 2: genuine ecosystem repo, inside Docker.
        eco_root = _clone_ecosystem_repo(work_dir / "ecosystem")
        eco_subset, eco_digest = run_pipeline(
            eco_root,
            work_dir,
            label="ecosystem",
            seed=ECOSYSTEM_SEED,
            n_procedural=ECOSYSTEM_N_PROCEDURAL,
            min_valid=1,
        )
        print(f"[ecosystem] {ECOSYSTEM_REPO_URL}@{ECOSYSTEM_REPO_TAG}: "
              f"{len(eco_subset)} curated instances (digest {eco_digest})")

    digest_after = repo_digest(FIXTURE_REPO)
    assert digest_after == digest_before, "the fixture target was modified by the demo"
    print("fixture target untouched: OK")
    print("demo: all acceptance expectations hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
