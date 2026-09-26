"""Schema checks for local (SLM) structured outputs (ADR-0009).

ADR-0009: SLMs are stateless leaf executors for bounded, format-constrained
sub-tasks, and *every output is schema-checked*. Local small models emit
invalid structured output far more often than the frontier (~19x invalid tool
calls in cascade settings), so a caller that asked the router for a local route
and then parsed the output must verify shape before using it. On violation,
escalate to the frontier instead of retrying locally.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


class SchemaViolation(ValueError):
    """A local model output did not match the required shape."""


def check_schema(data: Any, required: Iterable[str]) -> dict:
    """Require a dict carrying every key in ``required``.

    Raises SchemaViolation on the wrong type or any missing key, so the caller
    can escalate to the frontier rather than feed malformed output downstream.
    """
    required = set(required)
    if not isinstance(data, dict):
        raise SchemaViolation(f"expected object, got {type(data).__name__}")
    missing = sorted(k for k in required if k not in data)
    if missing:
        raise SchemaViolation(f"missing required keys: {missing}")
    return data


def check_json_output(text: str, required: Iterable[str]) -> dict:
    """Parse JSON text from a local model and schema-check it in one step."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise SchemaViolation(f"not valid JSON: {e}") from e
    return check_schema(data, required)
