"""Adversarial coverage test for .gitignore: every artifact class enumerated in
issue #18 must actually be ignored, and tracked source files must NOT be
silenced by an over-broad pattern. Enforcement is `git check-ignore`,
not prose.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Representative paths for every artifact class enumerated in issue #18.
MUST_BE_IGNORED = [
    # Python bytecode / compilation
    "__pycache__/engine.cpython-312.pyc",
    "src/agenteval_bench/engine.cpython-312.pyc",
    "legacy.pyo",
    "native.so",
    "Parser$py.class",
    # Packaging / build
    "build/lib/agenteval_bench/engine.py",
    "dist/agenteval-bench-0.1.0.tar.gz",
    "pkg.egg",
    "pkg.egg-info/PKG-INFO",
    "pip-wheel-metadata/top_level.txt",
    ".eggs/setuptools-68",
    # Virtual environments
    ".venv/bin/activate",
    "venv/bin/activate",
    "env/bin/activate",
    "ENV/bin/activate",
    # Generated dependency lock
    "uv.lock",
    # Test / coverage
    ".pytest_cache/CACHEDIR.TAG",
    ".coverage",
    ".coverage.localhost.1234.5678",
    "coverage.xml",
    "htmlcov/index.html",
    ".tox/py312/bin/pytest",
    ".nox/lint/bin/ruff",
    ".cache/pip/x",
    ".hypothesis/examples/x",
    # Lint / type-check caches
    ".ruff_cache/0.5.0/CACHEDIR.TAG",
    ".mypy_cache/3.12/engine.data.json",
    ".dmypy.json",
    ".pyre/x",
    ".pytype/py3.12/imports/x",
    # Notebooks
    ".ipynb_checkpoints/eda-checkpoint.ipynb",
    # Docs build output
    "docs/_build/html/index.html",
    "site/index.html",
    # Generated eval artifacts
    "reports/run1/junit.xml",
    "runs/2026-10-09/eval.json",
    "exp.report.json",
    # Secrets / local config / logs
    ".env",
    ".env.local",
    "model.pkl",
    "debug.log",
    # jev routing decision telemetry: append-only local audit log, never committed
    "jev/decisions.jsonl",
    # IDE / editor
    ".idea/workspace.xml",
    ".vscode/settings.json",
    "engine.py.swp",
    "engine.py.swo",
    "engine.py~",
    ".project",
    ".classpath",
    # OS artifacts
    ".DS_Store",
    ".AppleDouble",
    ".LSOverride",
    "Thumbs.db",
    "ehthumbs.db",
    "Desktop.ini",
    "$RECYCLE.BIN/x",
]

# Real source files that must never be swallowed by an over-broad pattern.
MUST_NOT_BE_IGNORED = [
    "src/agenteval_bench/engine.py",
    "src/agenteval_bench/__init__.py",
    "src/novelty_oracle/oracle.py",
    "jev/router.py",
    "jev/checks.py",
    "pyproject.toml",
    "README.md",
    "AGENTS.md",
    "tests/test_engine.py",
    ".github/workflows/ci.yml",
    "docs/spec.md",
    ".gitignore",
]


def _check_ignore(paths: list[str]) -> set[str]:
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "--", *paths],
        capture_output=True,
        text=True,
        check=False,
    )
    return {line for line in proc.stdout.splitlines() if line}


def test_all_issue18_artifact_classes_are_ignored() -> None:
    ignored = _check_ignore(MUST_BE_IGNORED)
    missing = [p for p in MUST_BE_IGNORED if p not in ignored]
    assert not missing, f"paths NOT ignored by .gitignore: {missing}"


def test_no_overbroad_pattern_swallows_sources() -> None:
    ignored = _check_ignore(MUST_NOT_BE_IGNORED)
    assert not ignored, f"source files wrongly ignored: {sorted(ignored)}"
