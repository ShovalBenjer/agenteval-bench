"""Validation harness for bug candidates (agenteval-bench#37).

``harness.valid`` in the issue's pipeline: run the target repo's test
suite on the clean tree (baseline), apply one candidate patch to a
throwaway copy, re-run the suite, and keep only candidates that break at
least one test. Reports carry FAIL_TO_PASS / PASS_TO_PASS lists computed
from test node IDs, never from prose.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bugsmith.images import TargetImage, repo_digest
from bugsmith.patch import PatchError, apply_patch
from bugsmith.types import BugCandidate, BugsmithError, ValidationReport

PYTEST_ARGS = ["-q", "--tb=no", "-p", "no:cacheprovider"]
RUN_TIMEOUT_S = 300

_FAILED_RE = re.compile(r"^(FAILED|ERROR) (\S+)")


@dataclass(frozen=True)
class TestOutcome:
    passed: frozenset[str]
    failed: frozenset[str]
    collection_error: bool


def parse_pytest_output(output: str, returncode: int) -> TestOutcome:
    """Extract test node IDs from ``pytest -q --tb=no`` output.

    A collection error poisons the whole run: every test counts as failed
    (the bug broke the suite structurally, not just behaviorally).
    """
    failed: set[str] = set()
    in_summary = False
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("====") and "short test summary info" in stripped:
            in_summary = True
            continue
        if in_summary:
            m = _FAILED_RE.match(stripped)
            if m:
                failed.add(m.group(2).split(" - ")[0])
                continue
            if stripped.startswith("===="):
                in_summary = False
    collection_error = (
        returncode == 2
        and ("error during collection" in output or "errors during collection" in output)
    )
    return TestOutcome(passed=frozenset(), failed=frozenset(failed),
                       collection_error=collection_error)


class Runner(Protocol):
    """Executes the target repo's test suite in a prepared work tree."""

    def run(self, work_tree: Path) -> TestOutcome: ...


def _pythonpath_for(tree: Path) -> str:
    src = tree / "src"
    return str(src if src.is_dir() else tree)


class LocalRunner:
    """Subprocess runner. Used by unit tests and as an explicit opt-in.

    Not a silent fallback: callers choose it deliberately (e.g. --local).
    """

    def run(self, work_tree: Path) -> TestOutcome:
        env = dict(os.environ)
        env["PYTHONPATH"] = _pythonpath_for(work_tree)
        proc = subprocess.run(
            ["python3", "-m", "pytest", *PYTEST_ARGS],
            cwd=str(work_tree),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=RUN_TIMEOUT_S,
            check=False,
            text=True,
        )
        outcome = parse_pytest_output(proc.stdout, proc.returncode)
        return outcome


class DockerRunner:
    """Runs the suite inside the built target image (the required path).

    The candidate's patched tree is bind-mounted read-write at /work; the
    image's own /target copy stays pristine.
    """

    def __init__(self, image: TargetImage) -> None:
        self._image = image

    def run(self, work_tree: Path) -> TestOutcome:
        inner = (
            "python3 -c \"import pathlib; "
            "p=pathlib.Path('/work/src'); "
            "print(p if p.is_dir() else pathlib.Path('/work'))\" "
        )
        proc = subprocess.run(
            ["docker", "run", "--rm",
             "-v", f"{work_tree.resolve()}:/work",
             self._image.tag, "sh", "-c",
             (
                 f"cd /work && PYTHONPATH=$({inner}) "
                 f"python3 -m pytest {' '.join(PYTEST_ARGS)}"
             )],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=RUN_TIMEOUT_S,
            check=False,
            text=True,
        )
        return parse_pytest_output(proc.stdout, proc.returncode)


def _collect_node_ids(work_tree: Path, runner: Runner) -> frozenset[str]:
    """All test node IDs, via --collect-only -q (independent of pass/fail)."""
    if isinstance(runner, DockerRunner):
        cmd = ["docker", "run", "--rm", "-v", f"{work_tree.resolve()}:/work",
               runner._image.tag, "sh", "-c",
               "cd /work && python3 -m pytest --collect-only -q -p no:cacheprovider"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              timeout=RUN_TIMEOUT_S, check=False, text=True)
    else:
        env = dict(os.environ)
        env["PYTHONPATH"] = _pythonpath_for(work_tree)
        proc = subprocess.run(
            ["python3", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
            cwd=str(work_tree), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=RUN_TIMEOUT_S, check=False, text=True,
        )
    ids = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" in line and not line.startswith(("=", "<", "ERROR")):
            ids.add(line.split(" ")[0])
    return frozenset(ids)


def _copy_tree(repo_root: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        repo_root, dest,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )


def validate(
    candidate: BugCandidate,
    repo_root: Path,
    runner: Runner,
    work_dir: Path,
    baseline_passed: frozenset[str],
) -> ValidationReport:
    """Validate one candidate: baseline vs patched test outcomes.

    ``baseline_passed`` is the clean-tree passing set (computed once per
    repo digest by :func:`validate_all`). The candidate's patched tree is
    a throwaway copy; the caller's tree is never mutated.
    """
    repo_root = repo_root.resolve()
    digest = repo_digest(repo_root)
    run_dir = work_dir / "runs" / digest / candidate.patch_sha
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    patched = run_dir / "patched"
    _copy_tree(repo_root, patched)
    try:
        apply_patch(patched, candidate.patch)
    except PatchError as e:
        raise BugsmithError(f"candidate patch does not apply: {e}") from e

    outcome = runner.run(patched)
    if outcome.collection_error:
        failed_after = frozenset(baseline_passed)  # structural break: everything fails
        passed_after = frozenset()
    else:
        # passed_after = collected - failed. Re-collect on the patched tree
        # so renamed/deleted tests cannot silently vanish from the math.
        collected = _collect_node_ids(patched, runner)
        failed_after = outcome.failed
        passed_after = collected - failed_after

    return ValidationReport(
        candidate=candidate,
        baseline_passed=tuple(sorted(baseline_passed)),
        failed_after=tuple(sorted(failed_after)),
        passed_after=tuple(sorted(passed_after)),
        collection_error=outcome.collection_error,
    )


def baseline(repo_root: Path, runner: Runner, work_dir: Path) -> frozenset[str]:
    """Test node IDs passing on the clean tree (cached per repo digest)."""
    repo_root = repo_root.resolve()
    digest = repo_digest(repo_root)
    cache = work_dir / "runs" / digest / "baseline_passed.txt"
    if cache.is_file():
        return frozenset(l.strip() for l in cache.read_text().splitlines() if l.strip())
    clean = work_dir / "runs" / digest / "clean"
    _copy_tree(repo_root, clean)
    outcome = runner.run(clean)
    if outcome.collection_error or outcome.failed:
        raise BugsmithError(
            "clean-tree baseline is not green: the target repo's own suite "
            f"must pass before bug injection (failed={sorted(outcome.failed)}, "
            f"collection_error={outcome.collection_error})"
        )
    passed = _collect_node_ids(clean, runner)
    if not passed:
        raise BugsmithError("baseline collected zero tests: refusing an empty benchmark")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(sorted(passed)) + "\n", encoding="utf-8")
    return passed
