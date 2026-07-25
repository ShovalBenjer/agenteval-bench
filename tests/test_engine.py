"""Tests for the core eval engine — RED then GREEN."""

import os
import pytest
import tempfile

from agenteval_bench.engine import EvalRunner
from agenteval_bench.models import EvalSuite
from agenteval_bench.scoring import DeterministicScorer


# --- Helpers ---

def _write_yaml(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def _dummy_agent(response: str):
    """Create an agent function that always returns the given response."""
    return lambda _input: response


# --- Suite loading ---

class TestEvalSuiteLoading:
    def test_loads_valid_yaml(self):
        path = _write_yaml("""
name: test-suite
version: 1
cases:
  - id: case-1
    input: "hello"
    expected:
      exact: "world"
""")
        try:
            suite = EvalSuite.from_yaml(path)
            assert suite.name == "test-suite"
            assert len(suite.cases) == 1
            assert suite.cases[0].id == "case-1"
            assert suite.cases[0].input == "hello"
            assert suite.cases[0].expected.exact == "world"
        finally:
            os.unlink(path)

    def test_rejects_invalid_yaml(self):
        path = _write_yaml("just a string, not a mapping")
        try:
            with pytest.raises(ValueError, match="expected mapping"):
                EvalSuite.from_yaml(path)
        finally:
            os.unlink(path)

    def test_loads_contains_matcher(self):
        path = _write_yaml("""
name: contains-suite
cases:
  - id: greet
    input: "hi"
    expected:
      contains: ["hello", "welcome"]
""")
        try:
            suite = EvalSuite.from_yaml(path)
            assert suite.cases[0].expected.contains == ["hello", "welcome"]
        finally:
            os.unlink(path)


# --- Deterministic scoring ---

class TestDeterministicScorer:
    def test_exact_match_passes(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput
        case = EvalCase(id="t1", input="q", expected=ExpectedOutput(exact="yes"))
        scorer = DeterministicScorer()
        result = scorer.score(case, "yes")
        assert result.passed is True
        assert result.score == 1.0

    def test_exact_match_fails(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput
        case = EvalCase(id="t2", input="q", expected=ExpectedOutput(exact="yes"))
        scorer = DeterministicScorer()
        result = scorer.score(case, "no")
        assert result.passed is False
        assert result.score == 0.0

    def test_contains_all_passes(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput
        case = EvalCase(id="t3", input="q", expected=ExpectedOutput(contains=["hello", "world"]))
        scorer = DeterministicScorer()
        result = scorer.score(case, "hello beautiful world")
        assert result.passed is True

    def test_contains_partial_fails(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput
        case = EvalCase(id="t4", input="q", expected=ExpectedOutput(contains=["hello", "world"]))
        scorer = DeterministicScorer()
        result = scorer.score(case, "hello only")
        assert result.passed is False

    def test_regex_match_passes(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput
        case = EvalCase(id="t5", input="q", expected=ExpectedOutput(regex=r"\d{4}-\d{2}-\d{2}"))
        scorer = DeterministicScorer()
        result = scorer.score(case, "today is 2026-05-27")
        assert result.passed is True

    def test_no_expected_returns_failure(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput
        case = EvalCase(id="t6", input="q", expected=ExpectedOutput())
        scorer = DeterministicScorer()
        result = scorer.score(case, "anything")
        assert result.passed is False
        assert "error" in result.details

    def test_json_schema_passes(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput
        schema = {"required": ["name", "age"]}
        case = EvalCase(id="t7", input="q", expected=ExpectedOutput(json_schema=schema))
        scorer = DeterministicScorer()
        result = scorer.score(case, '{"name": "Alice", "age": 30}')
        assert result.passed is True

    def test_json_schema_fails_missing_key(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput
        schema = {"required": ["name", "age"]}
        case = EvalCase(id="t8", input="q", expected=ExpectedOutput(json_schema=schema))
        scorer = DeterministicScorer()
        result = scorer.score(case, '{"name": "Alice"}')
        assert result.passed is False


# --- Engine integration ---

class TestEvalRunner:
    def test_run_all_pass(self):
        path = _write_yaml("""
name: all-pass
cases:
  - id: c1
    input: "what is 2+2?"
    expected:
      exact: "4"
  - id: c2
    input: "greet me"
    expected:
      contains: ["hello"]
""")
        try:
            suite = EvalSuite.from_yaml(path)
            runner = EvalRunner()

            def agent_fn(inp: str) -> str:
                if "2+2" in inp:
                    return "4"
                return "hello there"

            result = runner.run(suite, agent_fn)
            assert result.passed == 2
            assert result.failed == 0
            assert result.pass_rate == 1.0
        finally:
            os.unlink(path)

    def test_run_mixed_results(self):
        path = _write_yaml("""
name: mixed
cases:
  - id: pass
    input: "say ok"
    expected:
      exact: "ok"
  - id: fail
    input: "say no"
    expected:
      exact: "nope"
""")
        try:
            suite = EvalSuite.from_yaml(path)
            runner = EvalRunner()

            def agent_fn(inp: str) -> str:
                return "ok"

            result = runner.run(suite, agent_fn)
            assert result.passed == 1
            assert result.failed == 1
            assert result.pass_rate == 0.5
        finally:
            os.unlink(path)

    def test_skip_case(self):
        path = _write_yaml("""
name: skip-test
cases:
  - id: active
    input: "hi"
    expected:
      exact: "hello"
  - id: skipped
    input: "hi"
    expected:
      exact: "hello"
    skip: true
""")
        try:
            suite = EvalSuite.from_yaml(path)
            runner = EvalRunner()
            result = runner.run(suite, lambda _: "hello")
            assert result.passed == 1
            assert result.skipped == 1
            assert result.total == 2
        finally:
            os.unlink(path)

    def test_summary_format(self):
        from agenteval_bench.models import RunResult
        r = RunResult(suite_name="test", total=10, passed=8, failed=1, skipped=1, pass_rate=0.889)
        summary = r.summary()
        assert "test" in summary
        assert "88.9%" in summary

    def test_cost_bound_loaded(self):
        path = _write_yaml("""
name: cost-test
cost_bound:
  max_input_tokens: 500
  max_output_tokens: 50
cases:
  - id: c1
    input: "hi"
    expected:
      exact: "hi"
""")
        try:
            suite = EvalSuite.from_yaml(path)
            assert suite.cost_bound.max_input_tokens == 500
            assert suite.cost_bound.max_output_tokens == 50
        finally:
            os.unlink(path)
