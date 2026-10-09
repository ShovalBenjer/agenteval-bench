"""Immutable benchmark snapshots: version, never mutate (BetterBench discipline).

Every established benchmark set is an immutable, versioned snapshot on disk.
``SnapshotStore.publish`` writes a new version under
``<root>/<suite-name>/<version-id>/`` (``suite.json`` + ``manifest.json``)
and refuses to overwrite an existing version id — writes are append-only
through this API. ``SnapshotStore.load`` hash-verifies the stored bytes
against the manifest digest and raises :class:`SnapshotCorrupted` on any
on-disk mutation.

Honesty boundary: append-only through the API is enforcement (the store
itself never mutates a version); protection against out-of-band mutation
(``rm``, manual edits) is *detection*, not prevention — a tampered version
fails closed at load time with the offending path named. Detection is the
strongest guarantee a single-machine file layout can honestly offer.

Each version's manifest records what changed from the previous version and
why (owner, update cadence, regression set, feedback loop), forming an
auditable version chain: ``prev_version_id`` links every version back to
its predecessor.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agenteval_bench.models import (
    CostBound,
    EvalCase,
    EvalSuite,
    ExpectedOutput,
    RubricCriterion,
    RunResult,
)
from agenteval_bench.replay import suite_digest

SNAPSHOT_FORMAT = 1


class SnapshotError(Exception):
    """Base error for snapshot store failures."""


class VersionExistsError(SnapshotError):
    """Raised when publishing a version id that already exists."""


class VersionNotFoundError(SnapshotError):
    """Raised when loading a version id that does not exist."""


class SnapshotCorrupted(SnapshotError):
    """Raised when a stored version fails hash or schema verification."""


@dataclass(frozen=True)
class VersionMeta:
    """Why this version exists — the BetterBench maintenance record."""

    owner: str
    why: str
    changed_from_prev: str = ""
    cadence: str = ""
    regression_set: str = ""
    feedback_loop: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "why": self.why,
            "changed_from_prev": self.changed_from_prev,
            "cadence": self.cadence,
            "regression_set": self.regression_set,
            "feedback_loop": self.feedback_loop,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VersionMeta:
        if not isinstance(d, dict):
            raise SnapshotCorrupted(f"version meta must be a mapping, got {type(d).__name__}")
        for key in ("owner", "why"):
            if not isinstance(d.get(key), str) or not d[key]:
                raise SnapshotCorrupted(f"version meta missing non-empty {key!r}")
        optional = {}
        for key in ("changed_from_prev", "cadence", "regression_set", "feedback_loop"):
            value = d.get(key, "")
            if not isinstance(value, str):
                raise SnapshotCorrupted(f"version meta {key!r} must be a string")
            optional[key] = value
        return cls(owner=d["owner"], why=d["why"], **optional)


@dataclass(frozen=True)
class BaselineRecord:
    """Aggregate scores pinned at publish time — the release-gate reference."""

    agent: str
    seed: int
    pass_rate: float
    total: int
    passed: int
    failed: int
    skipped: int
    per_case: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "seed": self.seed,
            "pass_rate": self.pass_rate,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "per_case": dict(self.per_case),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BaselineRecord:
        if not isinstance(d, dict):
            raise SnapshotCorrupted(f"baseline must be a mapping, got {type(d).__name__}")
        if not isinstance(d.get("agent"), str):
            raise SnapshotCorrupted("baseline missing agent")
        if not isinstance(d.get("seed"), int):
            raise SnapshotCorrupted("baseline missing integer seed")
        for key in ("pass_rate",):
            if not isinstance(d.get(key), (int, float)):
                raise SnapshotCorrupted(f"baseline missing numeric {key!r}")
        for key in ("total", "passed", "failed", "skipped"):
            if not isinstance(d.get(key), int):
                raise SnapshotCorrupted(f"baseline missing integer {key!r}")
        per_case = d.get("per_case", {})
        if not isinstance(per_case, dict) or not all(
            isinstance(k, str) and isinstance(v, (int, float)) for k, v in per_case.items()
        ):
            raise SnapshotCorrupted("baseline per_case must map case id -> number")
        return cls(
            agent=d["agent"],
            seed=d["seed"],
            pass_rate=float(d["pass_rate"]),
            total=d["total"],
            passed=d["passed"],
            failed=d["failed"],
            skipped=d["skipped"],
            per_case={k: float(v) for k, v in per_case.items()},
        )


def baseline_from_run(result: RunResult, seed: int, agent: str) -> BaselineRecord:
    """Pin a recorded run's aggregates as a version's release-gate baseline."""
    per_case = {
        r.case_id: r.score for r in result.results if not r.details.get("skipped")
    }
    return BaselineRecord(
        agent=agent,
        seed=seed,
        pass_rate=result.pass_rate,
        total=result.total,
        passed=result.passed,
        failed=result.failed,
        skipped=result.skipped,
        per_case=per_case,
    )


@dataclass(frozen=True)
class BenchmarkVersion:
    """A published immutable benchmark snapshot."""

    suite_name: str
    version_id: str
    digest: str
    prev_version_id: str | None
    created_at: str
    meta: VersionMeta
    baseline: BaselineRecord
    promotions: tuple[dict[str, Any], ...] = ()


def _expected_to_dict(expected: ExpectedOutput) -> dict[str, Any]:
    return {
        "exact": expected.exact,
        "contains": list(expected.contains),
        "regex": expected.regex,
        "json_schema": (
            dict(expected.json_schema) if expected.json_schema is not None else None
        ),
    }


def _case_to_dict(case: EvalCase) -> dict[str, Any]:
    return {
        "id": case.id,
        "input": case.input,
        "expected": _expected_to_dict(case.expected),
        "rubric": [
            {
                "criterion": r.criterion,
                "weight": r.weight,
                "description": r.description,
            }
            for r in case.rubric
        ],
        "skip": case.skip,
        "output": case.output,
    }


def suite_to_canonical(suite: EvalSuite) -> bytes:
    """Deterministic byte form of a suite — the snapshot's hashed identity.

    Built from the parsed ``EvalSuite`` object (not the source YAML), so
    ``canonical(parse(canonical(suite))) == canonical(suite)`` is a fixed
    point the store enforces at every load.
    """
    doc: dict[str, Any] = {
        "tool": "agenteval-bench",
        "kind": "benchmark-snapshot",
        "format": SNAPSHOT_FORMAT,
        "name": suite.name,
        "suite_version": suite.version,
        "cases": [_case_to_dict(c) for c in suite.cases],
        "cost_bound": {
            "max_input_tokens": suite.cost_bound.max_input_tokens,
            "max_output_tokens": suite.cost_bound.max_output_tokens,
        },
    }
    return (json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _require_str(d: dict[str, Any], key: str, where: str) -> str:
    value = d.get(key)
    if not isinstance(value, str):
        raise SnapshotCorrupted(f"{where}: {key!r} must be a string")
    return value


def _expected_from_dict(d: dict[str, Any]) -> ExpectedOutput:
    if not isinstance(d, dict):
        raise SnapshotCorrupted("case: expected must be a mapping")
    for key, kind in (("exact", str), ("regex", str)):
        value = d.get(key)
        if value is not None and not isinstance(value, kind):
            raise SnapshotCorrupted(f"case: expected.{key} must be a string or null")
    contains = d.get("contains", [])
    if not isinstance(contains, list) or not all(isinstance(x, str) for x in contains):
        raise SnapshotCorrupted("case: expected.contains must be a list of strings")
    schema = d.get("json_schema")
    if schema is not None and not isinstance(schema, dict):
        raise SnapshotCorrupted("case: expected.json_schema must be a mapping or null")
    return ExpectedOutput(
        exact=d.get("exact"),
        contains=list(contains),
        regex=d.get("regex"),
        json_schema=dict(schema) if schema is not None else None,
    )


def _case_from_dict(d: dict[str, Any]) -> EvalCase:
    if not isinstance(d, dict):
        raise SnapshotCorrupted("case must be a mapping")
    case_id = _require_str(d, "id", "case")
    if not case_id:
        raise SnapshotCorrupted("case: id must be non-empty")
    text = _require_str(d, "input", "case")
    rubric_raw = d.get("rubric", [])
    if not isinstance(rubric_raw, list):
        raise SnapshotCorrupted("case: rubric must be a list")
    rubric = []
    for r in rubric_raw:
        if not isinstance(r, dict):
            raise SnapshotCorrupted("case: rubric entries must be mappings")
        criterion = _require_str(r, "criterion", "rubric entry")
        weight = r.get("weight", 1.0)
        if not isinstance(weight, (int, float)):
            raise SnapshotCorrupted("rubric entry: weight must be numeric")
        description = r.get("description", "")
        if not isinstance(description, str):
            raise SnapshotCorrupted("rubric entry: description must be a string")
        rubric.append(
            RubricCriterion(criterion=criterion, weight=float(weight), description=description)
        )
    skip = d.get("skip", False)
    if not isinstance(skip, bool):
        raise SnapshotCorrupted("case: skip must be a boolean")
    output = d.get("output")
    if output is not None and not isinstance(output, str):
        raise SnapshotCorrupted("case: output must be a string or null")
    return EvalCase(
        id=case_id,
        input=text,
        expected=_expected_from_dict(d.get("expected", {})),
        rubric=rubric,
        skip=skip,
        output=output,
    )


def suite_from_canonical(data: bytes) -> EvalSuite:
    """Parse canonical bytes back into an ``EvalSuite`` (strict schema)."""
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SnapshotCorrupted(f"suite.json is not valid canonical JSON: {e}") from e
    if not isinstance(doc, dict):
        raise SnapshotCorrupted("suite.json top level must be a mapping")
    if doc.get("kind") != "benchmark-snapshot":
        raise SnapshotCorrupted("suite.json: kind must be 'benchmark-snapshot'")
    if doc.get("format") != SNAPSHOT_FORMAT:
        raise SnapshotCorrupted(
            f"suite.json: unsupported format {doc.get('format')!r} "
            f"(this tool reads {SNAPSHOT_FORMAT})"
        )
    name = _require_str(doc, "name", "suite")
    suite_version = doc.get("suite_version", 1)
    if not isinstance(suite_version, int):
        raise SnapshotCorrupted("suite: suite_version must be an integer")
    cases_raw = doc.get("cases", [])
    if not isinstance(cases_raw, list):
        raise SnapshotCorrupted("suite: cases must be a list")
    cases = [_case_from_dict(c) for c in cases_raw]
    seen: set[str] = set()
    for c in cases:
        if c.id in seen:
            raise SnapshotCorrupted(f"suite: duplicate case id {c.id!r}")
        seen.add(c.id)
    cost_raw = doc.get("cost_bound", {})
    if not isinstance(cost_raw, dict):
        raise SnapshotCorrupted("suite: cost_bound must be a mapping")
    cost_bound = CostBound(
        max_input_tokens=int(cost_raw.get("max_input_tokens", 1500)),
        max_output_tokens=int(cost_raw.get("max_output_tokens", 120)),
    )
    return EvalSuite(name=name, version=suite_version, cases=cases, cost_bound=cost_bound)


class SnapshotStore:
    """Append-only on-disk store of immutable benchmark versions."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- paths -----------------------------------------------------------
    def _suite_dir(self, suite_name: str) -> Path:
        return self.root / suite_name

    def _version_dir(self, suite_name: str, version_id: str) -> Path:
        return self._suite_dir(suite_name) / version_id

    @staticmethod
    def _check_version_id(version_id: str) -> None:
        if not version_id or not isinstance(version_id, str):
            raise SnapshotError("version_id must be a non-empty string")
        if version_id in (".", "..") or "/" in version_id or "\\" in version_id:
            raise SnapshotError(f"version_id {version_id!r} is not a safe path segment")

    # -- publish ---------------------------------------------------------
    def publish(
        self,
        suite: EvalSuite,
        version_id: str,
        meta: VersionMeta,
        baseline: BaselineRecord,
        *,
        prev_version_id: str | None = None,
        promotions: list[dict[str, Any]] | None = None,
        created_at: str | None = None,
    ) -> BenchmarkVersion:
        """Freeze ``suite`` as a new immutable version.

        Raises :class:`VersionExistsError` if ``version_id`` is already
        published (never overwrites), and :class:`VersionNotFoundError` if
        ``prev_version_id`` names a version that does not exist — the chain
        must be unbroken.
        """
        self._check_version_id(version_id)
        vdir = self._version_dir(suite.name, version_id)
        if vdir.exists():
            raise VersionExistsError(
                f"version {version_id!r} of suite {suite.name!r} already exists — "
                "versions are immutable; publish a new version id instead"
            )
        if prev_version_id is not None:
            self._check_version_id(prev_version_id)
            if not self._version_dir(suite.name, prev_version_id).exists():
                raise VersionNotFoundError(
                    f"prev version {prev_version_id!r} of suite {suite.name!r} "
                    "does not exist — cannot extend a broken chain"
                )

        suite_bytes = suite_to_canonical(suite)
        digest = suite_digest(suite_bytes)
        created = created_at or datetime.now(UTC).isoformat()
        manifest: dict[str, Any] = {
            "tool": "agenteval-bench",
            "kind": "benchmark-snapshot",
            "format": SNAPSHOT_FORMAT,
            "suite_name": suite.name,
            "version_id": version_id,
            "digest": digest,
            "prev_version_id": prev_version_id,
            "created_at": created,
            "meta": meta.to_dict(),
            "baseline": baseline.to_dict(),
            "promotions": list(promotions or []),
        }
        manifest_bytes = (
            json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        ).encode("utf-8")

        # Stage in a temp dir inside the suite dir, then atomically rename —
        # a crash mid-publish never leaves a half-written version under its
        # final id.
        self._suite_dir(suite.name).mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{version_id}.", dir=self._suite_dir(suite.name))
        )
        try:
            (staging / "suite.json").write_bytes(suite_bytes)
            (staging / "manifest.json").write_bytes(manifest_bytes)
            os.replace(staging, vdir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        return BenchmarkVersion(
            suite_name=suite.name,
            version_id=version_id,
            digest=digest,
            prev_version_id=prev_version_id,
            created_at=created,
            meta=meta,
            baseline=baseline,
            promotions=tuple(promotions or []),
        )

    # -- load ------------------------------------------------------------
    def load(self, suite_name: str, version_id: str) -> tuple[BenchmarkVersion, EvalSuite]:
        """Load a version, hash-verifying immutability.

        Raises :class:`VersionNotFoundError` if the version was never
        published; :class:`SnapshotCorrupted` on any tampering, schema
        violation, or canonical fixed-point break.
        """
        self._check_version_id(version_id)
        vdir = self._version_dir(suite_name, version_id)
        if not vdir.is_dir():
            raise VersionNotFoundError(
                f"version {version_id!r} of suite {suite_name!r} is not published"
            )
        try:
            suite_bytes = (vdir / "suite.json").read_bytes()
        except OSError as e:
            raise SnapshotCorrupted(f"{vdir}/suite.json unreadable: {e}") from e
        try:
            manifest = json.loads((vdir / "manifest.json").read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            raise SnapshotCorrupted(f"{vdir}/manifest.json unreadable: {e}") from e

        if not isinstance(manifest, dict) or manifest.get("format") != SNAPSHOT_FORMAT:
            raise SnapshotCorrupted(f"{vdir}/manifest.json has wrong format marker")
        recorded = manifest.get("digest")
        if suite_digest(suite_bytes) != recorded:
            raise SnapshotCorrupted(
                f"hash mismatch for {suite_name}/{version_id}: on-disk suite bytes "
                f"no longer match manifest digest {recorded!r} — the snapshot was mutated"
            )
        suite = suite_from_canonical(suite_bytes)
        if suite_to_canonical(suite) != suite_bytes:
            raise SnapshotCorrupted(
                f"canonical fixed-point broken for {suite_name}/{version_id}"
            )
        if suite.name != suite_name or manifest.get("version_id") != version_id:
            raise SnapshotCorrupted(
                f"manifest identity drift for {suite_name}/{version_id}"
            )
        promotions_raw = manifest.get("promotions", [])
        if not isinstance(promotions_raw, list):
            raise SnapshotCorrupted(f"{vdir}/manifest.json: promotions must be a list")
        version = BenchmarkVersion(
            suite_name=suite_name,
            version_id=version_id,
            digest=str(recorded),
            prev_version_id=manifest.get("prev_version_id"),
            created_at=str(manifest.get("created_at", "")),
            meta=VersionMeta.from_dict(manifest.get("meta", {})),
            baseline=BaselineRecord.from_dict(manifest.get("baseline", {})),
            promotions=tuple(promotions_raw),
        )
        return version, suite

    # -- listing / verification -------------------------------------------
    def list_versions(self, suite_name: str) -> list[str]:
        """Published version ids for a suite, sorted."""
        sdir = self._suite_dir(suite_name)
        if not sdir.is_dir():
            return []
        return sorted(p.name for p in sdir.iterdir() if p.is_dir() and not p.name.startswith("."))

    def verify(self, suite_name: str, version_id: str) -> None:
        """Hash-verify one version; raises on any problem."""
        self.load(suite_name, version_id)

    def verify_all(self) -> list[str]:
        """Hash-verify every published version; returns problem descriptions.

        Empty list means the whole store is healthy.
        """
        problems: list[str] = []
        if not self.root.is_dir():
            return problems
        for sdir in sorted(self.root.iterdir()):
            if not sdir.is_dir() or sdir.name.startswith("."):
                continue
            for vdir in sorted(sdir.iterdir()):
                if not vdir.is_dir() or vdir.name.startswith("."):
                    continue
                try:
                    self.verify(sdir.name, vdir.name)
                except SnapshotError as e:
                    problems.append(f"{sdir.name}/{vdir.name}: {e}")
        return problems


__all__ = [
    "SNAPSHOT_FORMAT",
    "BaselineRecord",
    "BenchmarkVersion",
    "SnapshotCorrupted",
    "SnapshotError",
    "SnapshotStore",
    "VersionExistsError",
    "VersionMeta",
    "VersionNotFoundError",
    "baseline_from_run",
    "suite_from_canonical",
    "suite_to_canonical",
]
