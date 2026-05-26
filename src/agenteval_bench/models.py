"""Data models for eval suites and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExpectedOutput:
    """What the agent output should look like."""

    exact: str | None = None
    contains: list[str] = field(default_factory=list)
    regex: str | None = None
    json_schema: dict[str, Any] | None = None


@dataclass
class RubricCriterion:
    """A single grading criterion."""

    criterion: str
    weight: float = 1.0
    description: str = ""


@dataclass
class CostBound:
    """Token cost limits for an eval run."""

    max_input_tokens: int = 1500
    max_output_tokens: int = 120


@dataclass
class EvalCase:
    """A single test case within an eval suite."""

    id: str
    input: str
    expected: ExpectedOutput = field(default_factory=ExpectedOutput)
    rubric: list[RubricCriterion] = field(default_factory=list)
    skip: bool = False


@dataclass
class EvalSuite:
    """A collection of eval cases loaded from YAML."""

    name: str
    version: int = 1
    cases: list[EvalCase] = field(default_factory=list)
    cost_bound: CostBound = field(default_factory=CostBound)

    @classmethod
    def from_yaml(cls, path: str) -> EvalSuite:
        """Load an eval suite from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid eval suite: expected mapping, got {type(data).__name__}")

        cases = []
        for case_data in data.get("cases", []):
            expected_data = case_data.get("expected", {})
            expected = ExpectedOutput(
                exact=expected_data.get("exact"),
                contains=expected_data.get("contains", []),
                regex=expected_data.get("regex"),
                json_schema=expected_data.get("json_schema"),
            )
            rubric = [
                RubricCriterion(
                    criterion=r["criterion"],
                    weight=r.get("weight", 1.0),
                    description=r.get("description", ""),
                )
                for r in case_data.get("rubric", [])
            ]
            cases.append(
                EvalCase(
                    id=case_data["id"],
                    input=case_data["input"],
                    expected=expected,
                    rubric=rubric,
                    skip=case_data.get("skip", False),
                )
            )

        cost_data = data.get("cost_bound", {})
        cost_bound = CostBound(
            max_input_tokens=cost_data.get("max_input_tokens", 1500),
            max_output_tokens=cost_data.get("max_output_tokens", 120),
        )

        return cls(
            name=data["name"],
            version=data.get("version", 1),
            cases=cases,
            cost_bound=cost_bound,
        )


@dataclass
class EvalResult:
    """Score for a single eval case."""

    case_id: str
    passed: bool
    score: float  # 0.0 to 1.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Aggregate result for an entire eval run."""

    suite_name: str
    results: list[EvalResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    pass_rate: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary."""
        return (
            f"Suite: {self.suite_name}\n"
            f"Total: {self.total} | Passed: {self.passed} | "
            f"Failed: {self.failed} | Skipped: {self.skipped}\n"
            f"Pass rate: {self.pass_rate:.1%}"
        )
