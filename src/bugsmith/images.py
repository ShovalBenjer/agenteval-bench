"""Docker image construction for the validation harness (agenteval-bench#37).

``create_images`` in the issue's pipeline: a sealed Docker execution
environment (Python + pytest + the target repo's declared third-party
dependencies). Every validation run executes inside a fresh container
with a throwaway copy of the (patched) target bind-mounted at /work.
The working tree is never touched, and no repo snapshot is baked into
the image, so a stale copy can never shadow the tree under test.
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
    """Content digest of the target repo. Pins the substrate.

    Hashes ``*.py`` files plus the config files that shape the test run
    (``pyproject.toml``, ``setup.py``/``setup.cfg``, ``requirements*``,
    ``pytest.ini``, ``tox.ini``): a dependency or config change is a
    different substrate, even with identical sources.
    """
    repo_root = repo_root.resolve()
    h = hashlib.sha256()
    config_names = {
        "pyproject.toml", "setup.py", "setup.cfg", "pytest.ini", "tox.ini",
    }

    def relevant(p: Path) -> bool:
        rel = p.relative_to(repo_root)
        if ".git" in rel.parts or "__pycache__" in rel.parts:
            return False
        return p.suffix == ".py" or rel.name in config_names or (
            rel.name.startswith("requirements") and rel.suffix == ".txt"
        )

    files = sorted(p for p in repo_root.rglob("*") if p.is_file() and relevant(p))
    if not files:
        raise BugsmithError(f"no Python files under {repo_root}")
    for p in files:
        h.update(str(p.relative_to(repo_root)).encode())
        h.update(b"\x00")
        h.update(p.read_bytes())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def collect_requirements(repo_root: Path) -> list[str]:
    """Third-party dependencies declared by the target repo.

    Parsed from ``pyproject.toml`` (``[project] dependencies``, stdlib
    tomllib) plus any ``requirements*.txt`` files. ``setup.py``-only
    projects cannot be parsed safely and are documented as unsupported
    for dependency installation: the baseline gate then fails loudly if
    imports are missing, instead of silently half-working.
    """
    import tomllib

    repo_root = repo_root.resolve()
    reqs: list[str] = []
    pyproject = repo_root / "pyproject.toml"
    if pyproject.is_file():
        try:
            doc = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            deps = doc.get("project", {}).get("dependencies", [])
            reqs.extend(d for d in deps if isinstance(d, str) and d.strip())
        except (tomllib.TOMLDecodeError, OSError) as e:
            raise BugsmithError(f"cannot parse {pyproject}: {e}") from e
    for name in ("requirements.txt", "requirements-test.txt",
                 "test-requirements.txt", "requirements_test.txt"):
        f = repo_root / name
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    reqs.append(line)
    # De-duplicated, order-stable.
    return list(dict.fromkeys(reqs))


def dockerfile_text(has_dependencies: bool = False) -> str:
    """The image recipe: sealed Python + pytest (+ target dependencies).

    The image is a sealed *execution environment*, not a repo snapshot:
    the target's code travels via bind-mounted throwaway copies at run
    time, so the working tree is never touched and no stale copy can
    shadow the patched tree under test.
    """
    lines = [
        f"FROM {BASE_IMAGE}",
        # Bounded pin: the validation environment must not drift silently
        # under the seed-reproducibility claim.
        'RUN pip install --no-cache-dir "pytest>=8,<10"',
    ]
    if has_dependencies:
        lines += [
            "COPY deps/requirements.txt /deps/requirements.txt",
            "RUN pip install --no-cache-dir -r /deps/requirements.txt",
        ]
    return "\n".join(lines) + "\n"


def build_context(repo_root: Path, context_dir: Path) -> None:
    """Materialize the docker build context: Dockerfile + dependency manifests."""
    repo_root = repo_root.resolve()
    if not repo_root.is_dir():
        raise BugsmithError(f"repo root not found: {repo_root}")
    context_dir.mkdir(parents=True, exist_ok=True)
    reqs = collect_requirements(repo_root)
    deps_dir = context_dir / "deps"
    if deps_dir.exists():
        shutil.rmtree(deps_dir)
    deps_dir.mkdir()
    (deps_dir / "requirements.txt").write_text("\n".join(reqs) + "\n", encoding="utf-8")
    (context_dir / "Dockerfile").write_text(
        dockerfile_text(has_dependencies=bool(reqs)), encoding="utf-8"
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
