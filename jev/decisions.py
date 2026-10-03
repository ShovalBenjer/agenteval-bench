"""Append-only routing decision log (ADR-0009 audit requirement).

Every routing decision the jev router makes is recorded here as one JSON object
per line, so a mis-tuned escalation threshold is *auditable* instead of a
silent quality regression. The log is append-only: this module opens the file
in ``"a"`` mode and never truncates or rewrites it.

Privacy: the raw task text is never stored, only a SHA-256 digest, because
tasks can carry PII (ADR-0009 requires PII-stripping before any trace becomes
training data).

Location defaults to ``jev/decisions.jsonl`` next to this file and is
overridable with ``JEV_DECISIONS_LOG``. The file is gitignored (local
telemetry, not shared state).
"""
from __future__ import annotations

import hashlib
import json
import os
import time


def log_path() -> str:
    """Where routing decisions are recorded."""
    return os.getenv(
        "JEV_DECISIONS_LOG",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "decisions.jsonl"),
    )


def log_decision(
    *,
    task: str,
    route_name: str,
    route_kind: str,
    complexity: float,
    confidence: float,
    kept_local: bool,
    reason: str,
    escalated: bool,
    budget: str,
) -> dict:
    """Append one routing decision record. Returns the record written."""
    record = {
        "ts": time.time(),
        "task_sha256": hashlib.sha256(task.encode("utf-8")).hexdigest()[:16],
        "route": route_name,
        "kind": route_kind,
        "complexity": round(complexity, 4),
        "confidence": round(confidence, 4),
        "kept_local": kept_local,
        "kept_local_reason": reason if kept_local else None,
        "escalated": escalated,
        "budget": budget,
    }
    with open(log_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return record
