"""Deterministic scoring strategies for agent outputs."""

from __future__ import annotations

import json
import re
from typing import Any

from agenteval_bench.models import EvalCase, EvalResult


class DeterministicScorer:
    """Scores agent outputs using deterministic matchers — no LLM calls."""

    def score(self, case: EvalCase, agent_output: str) -> EvalResult:
        """Score a single case against the agent output.

        Runs all applicable matchers and returns a combined score.
        """
        checks: dict[str, bool] = {}
        weights: dict[str, float] = {}
        details: dict[str, Any] = {}

        expected = case.expected

        if expected.exact is not None:
            checks["exact_match"] = agent_output.strip() == expected.exact.strip()
            weights["exact_match"] = 1.0
            details["exact_match"] = {
                "expected": expected.exact[:200],
                "got": agent_output[:200],
            }

        if expected.contains:
            all_present = all(needle in agent_output for needle in expected.contains)
            checks["contains"] = all_present
            weights["contains"] = 1.0
            details["contains"] = {
                "needles": expected.contains,
                "missing": [n for n in expected.contains if n not in agent_output],
            }

        if expected.regex is not None:
            match = re.search(expected.regex, agent_output)
            checks["regex"] = match is not None
            weights["regex"] = 1.0
            details["regex"] = {
                "pattern": expected.regex,
                "matched": match.group(0) if match else None,
            }

        if expected.json_schema is not None:
            checks["json_schema"] = self._validate_json_schema(agent_output, expected.json_schema)
            weights["json_schema"] = 1.0

        if not checks:
            return EvalResult(
                case_id=case.id,
                passed=False,
                score=0.0,
                details={"error": "No expected output defined"},
            )

        total_weight = sum(weights.values())
        if total_weight == 0:
            weighted_score = 0.0
        else:
            weighted_score = (
                sum(weights[name] for name, passed in checks.items() if passed) / total_weight
            )

        return EvalResult(
            case_id=case.id,
            passed=all(checks.values()),
            score=weighted_score,
            details={"checks": checks, **details},
        )

    def _validate_json_schema(self, output: str, schema: dict[str, Any]) -> bool:
        """Basic JSON schema validation — checks that output is valid JSON
        and contains required keys."""
        try:
            data = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return False

        required = schema.get("required", [])
        if not isinstance(data, dict):
            return len(required) == 0

        return all(key in data for key in required)
