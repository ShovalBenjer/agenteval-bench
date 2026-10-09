"""Docker image construction for the validation harness (agenteval-bench#37).

``create_images`` in the issue's pipeline: a Docker *copy* of the target
repo is built into an image, and every validation run executes inside a
fresh container. The working tree is never touched — all patch
application happens on throwaway copies that are volume-mounted into the
container at run time.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from bugsmith.types import BugsmithError

BASE_IMAGE = "python:3.12-slim"
IMAGE_TAG_PREFIX = "bugsmith-target"


@dataclass(frozen=True)
class TargetImage:
    """A built image holding a clean copy of the target repo."""

    tag: str
    repo_digest: str


def repo_digest(repo_root: Path) -> str:
    """Content digest of the target repo's Python files. Pins the substrate."""
    repo_root = repo_root.resolve()
    h = hashlib.sha256()
    files = sorted(
        p for p in repo_root.rglob("*.py")
        if ".git" not in p.parts and "__pycache__" not in p.parts
    )
    if not files:
        raise BugsmithError(f"no Python files under {repo_root}")
    for p in files:
        h.update(str(p.relative_to(repo_root)).encode())
        h.update(b"\x00")
        h.update(p.read_bytes())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def dockerfile_text() -> str:
    """The image recipe: clean Python + pytest + a copy of the target repo."""
    return (
        f"FROM {BASE_IMAGE}\n"
        "RUN pip install --no-cache-dir pytest\n"
        "COPY repo /target\n"
        "WORKDIR /target\n"
    )


def build_context(repo_root: Path, context_dir: Path) -> None:
    """Materialize the docker build context: Dockerfile + a copy of the repo."""
    repo_root = repo_root.resolve()
    if not repo_root.is_dir():
        raise BugsmithError(f"repo root not found: {repo_root}")
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "Dockerfile").write_text(dockerfile_text(), encoding="utf-8")
    dest = context_dir / "repo"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        repo_root,
        dest,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".venv", "venv"),
    )


def _docker_available() -> bool:
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=True,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def build_image(repo_root: Path, work_dir: Path) -> TargetImage:
    """Build the target image. Raises loudly if Docker is unavailable.

    The issue requires validation *inside Docker*; a missing daemon is a
    hard error, never a silent switch to local execution.
    """
    if not _docker_available():
        raise BugsmithError(
            "docker is not available: bugsmith validation must run inside "
            "Docker (agenteval-bench#37 criterion 1). Refusing to silently "
            "fall back to local execution."
        )
    repo_root = repo_root.resolve()
    digest = repo_digest(repo_root)
    tag = f"{IMAGE_TAG_PREFIX}:{digest}"
    context = work_dir / "docker-context"
    build_context(repo_root, context)
    proc = subprocess.run(
        ["docker", "build", "-t", tag, str(context)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
        check=False,
        text=True,
    )
    if proc.returncode != 0:
        raise BugsmithError(f"docker build failed:\n{proc.stdout[-4000:]}")
    return TargetImage(tag=tag, repo_digest=digest)
