"""Seeded end-to-end demo of the SWE-smith pipeline (agenteval-bench#37).

Runs the full issue pipeline against the vendored ``mcalc`` fixture
target, inside Docker:

1. ``create_images`` — build a Docker image holding a copy of the target.
2. ``bug_gen`` — 8 procedural-AST candidates (fixed seed) + 1 PR-mirror
   candidate (a reverted historical-style patch).
3. ``harness.valid`` — validate every candidate in a fresh container;
   only instances that break >= 1 test survive, with FAIL_TO_PASS /
   PASS_TO_PASS lists.
4. ``harness.gather`` + curation — config-driven subset selection.

Exit 0 only if every acceptance expectation of the issue holds; any
failure raises, so CI can gate on this demo directly. The fixture target
itself is never modified (digest pinned before/after).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from bugsmith.buggen import PRMirrorGenerator, ProceduralBugGenerator
from bugsmith.curate import select_subset
from bugsmith.harness import DockerRunner, baseline, validate
from bugsmith.images import build_image, repo_digest
from bugsmith.types import BugCandidate, CurationConfig

SEED = 20261009
N_PROCEDURAL = 8
HERE = Path(__file__).resolve().parent
FIXTURE_REPO = HERE / "fixtures" / "target"
PR_MIRROR_PATCH = HERE / "fixtures" / "pr_mirror_median.patch"


def _pr_mirror_candidate() -> BugCandidate:
    patch = PR_MIRROR_PATCH.read_text(encoding="utf-8")
    gen = PRMirrorGenerator(pr_ref="mcalc#41-revert")
    (candidate,) = gen.generate(patch, ["src/mcalc/stats.py"])
    return candidate


def main() -> int:
    digest_before = repo_digest(FIXTURE_REPO)
    print(f"target repo digest: {digest_before}")

    with tempfile.TemporaryDirectory(prefix="bugsmith-demo-") as tmp:
        work_dir = Path(tmp)

        # 1. create_images (inside Docker; hard fail without a daemon).
        image = build_image(FIXTURE_REPO, work_dir)
        print(f"built image {image.tag}")
        assert image.repo_digest == digest_before
        runner = DockerRunner(image)

        # Baseline: the clean tree must be green before bug injection.
        passed = baseline(FIXTURE_REPO, runner, work_dir)
        print(f"baseline: {len(passed)} tests passing on the clean tree")
        assert len(passed) >= 10, f"fixture too small: {len(passed)} tests"

        # 2. bug_gen.
        procedural = ProceduralBugGenerator(SEED).generate(FIXTURE_REPO, N_PROCEDURAL)
        mirror = _pr_mirror_candidate()
        candidates = procedural + [mirror]
        print(f"generated {len(procedural)} procedural + 1 pr-mirror candidates")
        assert len(procedural) == N_PROCEDURAL, (
            f"generator starved: {len(procedural)}/{N_PROCEDURAL}"
        )

        # Seed reproducibility (criterion 3): same seed -> byte-identical patches.
        again = ProceduralBugGenerator(SEED).generate(FIXTURE_REPO, N_PROCEDURAL)
        assert [c.patch for c in again] == [c.patch for c in procedural], (
            "procedural generation is not reproducible from the seed"
        )
        print("seed reproducibility: OK (byte-identical patches)")

        # 3. harness.valid.
        reports = [validate(c, FIXTURE_REPO, runner, work_dir, passed)
                   for c in candidates]
        valid = [r for r in reports if r.is_valid_instance]
        print(f"validated: {len(valid)}/{len(reports)} instances break >= 1 test")
        assert len(valid) >= 2, "fewer than 2 valid instances survived"
        baseline_set = set(passed)
        for r in valid:
            assert r.fail_to_pass, "valid instance has empty FAIL_TO_PASS"
            assert set(r.fail_to_pass) <= baseline_set, (
                "FAIL_TO_PASS names tests that did not pass on the clean tree"
            )
            assert set(r.pass_to_pass) <= baseline_set
            assert not (set(r.fail_to_pass) & set(r.pass_to_pass)), (
                "FAIL_TO_PASS and PASS_TO_PASS overlap"
            )
        print("FAIL_TO_PASS / PASS_TO_PASS accounting: OK")

        # 4. Curation: config-driven, deterministic, no hand-picking.
        config = CurationConfig(
            seed=SEED,
            fail_to_pass_min=1,
            fail_to_pass_max=4,
            max_instances=6,
            strategy_quota={"procedural_ast": 5, "pr_mirror": 1},
        )
        subset = select_subset(config, reports, digest_before)
        subset2 = select_subset(config, reports, digest_before)
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
        print(f"curated subset: {len(subset)} instances "
              f"({', '.join(i.instance_id for i in subset)})")

        manifest = {
            "tool": "agenteval-bench",
            "command": "bugsmith-demo",
            "seed": SEED,
            "repo_digest": digest_before,
            "baseline_tests": len(passed),
            "candidates": len(candidates),
            "valid_instances": len(valid),
            "curated": [i.to_dict() for i in subset],
        }
        manifest_path = work_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"manifest: {len(manifest['curated'])} curated instances")

    digest_after = repo_digest(FIXTURE_REPO)
    assert digest_after == digest_before, "the fixture target was modified by the demo"
    print("fixture target untouched: OK")
    print("demo: all acceptance expectations hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
