"""Tests for the core eval engine — RED then GREEN."""

import re
from unittest.mock import patch

import pytest
import yaml

from agenteval_bench.engine import EvalRunner
from agenteval_bench.models import EvalSuite
from agenteval_bench.scoring import DeterministicScorer


class TestEvalSuiteLoading:
    def test_loads_valid_yaml(self, _write_yaml):
        path = _write_yaml("""
name: test-suite
version: 1
cases:
  - id: case-1
    input: "hello"
    expected:
      exact: "world"
""")
        suite = EvalSuite.from_yaml(path)
        assert suite.name == "test-suite"
        assert len(suite.cases) == 1
        assert suite.cases[0].id == "case-1"
        assert suite.cases[0].input == "hello"
        assert suite.cases[0].expected.exact == "world"

    def test_rejects_invalid_yaml(self, _write_yaml):
        path = _write_yaml("just a string, not a mapping")
        with pytest.raises(TypeError, match="expected mapping"):
            EvalSuite.from_yaml(path)

    def test_loads_contains_matcher(self, _write_yaml):
        path = _write_yaml("""
name: contains-suite
cases:
  - id: greet
    input: "hi"
    expected:
      contains: ["hello", "welcome"]
""")
        suite = EvalSuite.from_yaml(path)
        assert suite.cases[0].expected.contains == ["hello", "welcome"]

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            EvalSuite.from_yaml("/nonexistent/path/to/suite.yaml")

    def test_yaml_syntax_error_raises(self, _write_yaml):
        path = _write_yaml('name: "unclosed string\n')
        with pytest.raises(yaml.YAMLError):
            EvalSuite.from_yaml(path)

    def test_permission_error_raises(self, _write_yaml):
        path = _write_yaml("name: perm-test\ncases: []")
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                EvalSuite.from_yaml(path)

    def test_duplicate_case_ids_raise_valueerror(self, _write_yaml):
        path = _write_yaml("""
name: dup-ids
cases:
  - id: same
    input: "a"
    expected:
      exact: "x"
  - id: same
    input: "b"
    expected:
      exact: "y"
""")
        with pytest.raises(ValueError, match="Duplicate case id"):
            EvalSuite.from_yaml(path)


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

    def test_invalid_regex_raises(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput

        case = EvalCase(id="t6", input="q", expected=ExpectedOutput(regex="[invalid"))
        scorer = DeterministicScorer()
        with pytest.raises(re.error):
            scorer.score(case, "anything")

    def test_no_expected_returns_failure(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput

        case = EvalCase(id="t7", input="q", expected=ExpectedOutput())
        scorer = DeterministicScorer()
        result = scorer.score(case, "anything")
        assert result.passed is False
        assert "error" in result.details

    def test_json_schema_passes(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput

        schema = {"required": ["name", "age"]}
        case = EvalCase(id="t8", input="q", expected=ExpectedOutput(json_schema=schema))
        scorer = DeterministicScorer()
        result = scorer.score(case, '{"name": "Alice", "age": 30}')
        assert result.passed is True

    def test_json_schema_fails_missing_key(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput

        schema = {"required": ["name", "age"]}
        case = EvalCase(id="t9", input="q", expected=ExpectedOutput(json_schema=schema))
        scorer = DeterministicScorer()
        result = scorer.score(case, '{"name": "Alice"}')
        assert result.passed is False

    def test_multiple_matchers_all_pass(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput

        case = EvalCase(
            id="m1",
            input="q",
            expected=ExpectedOutput(
                exact='{"msg": "ok", "code": 42}',
                contains=["msg", "42"],
                regex=r"\d+",
                json_schema={"required": ["msg", "code"]},
            ),
        )
        scorer = DeterministicScorer()
        result = scorer.score(case, '{"msg": "ok", "code": 42}')
        assert result.passed is True
        assert result.score == 1.0

    def test_multiple_matchers_one_fails(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput

        case = EvalCase(
            id="m2",
            input="q",
            expected=ExpectedOutput(
                exact="result",
                contains=["missing"],
            ),
        )
        scorer = DeterministicScorer()
        result = scorer.score(case, "result")
        assert result.passed is False

    def test_contains_ignore_case_passes(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput

        case = EvalCase(
            id="ci1",
            input="q",
            expected=ExpectedOutput(contains_ignore_case=["Hello", "WORLD"]),
        )
        scorer = DeterministicScorer()
        result = scorer.score(case, "hello there world")
        assert result.passed is True

    def test_contains_ignore_case_fails(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput

        case = EvalCase(
            id="ci2",
            input="q",
            expected=ExpectedOutput(contains_ignore_case=["Hello", "WORLD"]),
        )
        scorer = DeterministicScorer()
        result = scorer.score(case, "hello there")
        assert result.passed is False

    def test_unicode_input_output(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput

        case = EvalCase(id="u1", input="你好", expected=ExpectedOutput(exact="世界"))
        scorer = DeterministicScorer()
        result = scorer.score(case, "世界")
        assert result.passed is True

    def test_empty_string_input_output(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput

        case = EvalCase(id="e1", input="", expected=ExpectedOutput(exact=""))
        scorer = DeterministicScorer()
        result = scorer.score(case, "")
        assert result.passed is True

    def test_very_long_input_output(self):
        from agenteval_bench.models import EvalCase, ExpectedOutput

        long_str = "x" * 100_000
        case = EvalCase(id="l1", input=long_str, expected=ExpectedOutput(exact=long_str))
        scorer = DeterministicScorer()
        result = scorer.score(case, long_str)
        assert result.passed is True


class TestEvalRunner:
    def test_run_all_pass(self, _write_yaml):
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

    def test_run_mixed_results(self, _write_yaml):
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
        suite = EvalSuite.from_yaml(path)
        runner = EvalRunner()

        def agent_fn(inp: str) -> str:
            return "ok"

        result = runner.run(suite, agent_fn)
        assert result.passed == 1
        assert result.failed == 1
        assert result.pass_rate == 0.5

    def test_skip_case(self, _write_yaml):
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
        suite = EvalSuite.from_yaml(path)
        runner = EvalRunner()
        result = runner.run(suite, lambda _: "hello")
        assert result.passed == 1
        assert result.skipped == 1
        assert result.total == 2

    def test_summary_format(self):
        from agenteval_bench.models import RunResult

        r = RunResult(suite_name="test", total=10, passed=8, failed=1, skipped=1, pass_rate=0.889)
        summary = r.summary()
        assert "test" in summary
        assert "88.9%" in summary

    def test_cost_bound_loaded(self, _write_yaml):
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
        suite = EvalSuite.from_yaml(path)
        assert suite.cost_bound.max_input_tokens == 500
        assert suite.cost_bound.max_output_tokens == 50

    def test_empty_suite(self, _write_yaml):
        path = _write_yaml("""
name: empty
cases: []
""")
        suite = EvalSuite.from_yaml(path)
        runner = EvalRunner()
        result = runner.run(suite, lambda _: "nope")
        assert result.total == 0
        assert result.passed == 0
        assert result.failed == 0
        assert result.skipped == 0
        assert result.pass_rate == 0.0

    def test_all_skipped_suite(self, _write_yaml):
        path = _write_yaml("""
name: all-skip
cases:
  - id: s1
    input: "a"
    expected:
      exact: "x"
    skip: true
  - id: s2
    input: "b"
    expected:
      exact: "y"
    skip: true
""")
        suite = EvalSuite.from_yaml(path)
        runner = EvalRunner()
        result = runner.run(suite, lambda _: "never-called")
        assert result.total == 2
        assert result.skipped == 2
        assert result.passed == 0
        assert result.failed == 0
        assert result.pass_rate == 0.0

    def test_zero_pass_rate(self, _write_yaml):
        path = _write_yaml("""
name: zero-pass
cases:
  - id: f1
    input: "a"
    expected:
      exact: "x"
  - id: f2
    input: "b"
    expected:
      exact: "y"
""")
        suite = EvalSuite.from_yaml(path)
        runner = EvalRunner()
        result = runner.run(suite, lambda _: "nope")
        assert result.passed == 0
        assert result.failed == 2
        assert result.pass_rate == 0.0

    def test_agent_exception_does_not_abort(self, _write_yaml):
        path = _write_yaml("""
name: exception-test
cases:
  - id: good
    input: "safe"
    expected:
      exact: "ok"
  - id: bad
    input: "boom"
    expected:
      exact: "fallback"
""")
        suite = EvalSuite.from_yaml(path)
        runner = EvalRunner()

        def agent_fn(inp: str) -> str:
            if "boom" in inp:
                raise RuntimeError("agent exploded")
            return "ok"

        result = runner.run(suite, agent_fn)
        assert result.total == 2
        assert result.passed == 1
        assert result.failed == 1
        assert result.results[1].details.get("error") == "agent exploded"

    def test_run_ci_passes_above_threshold(self, _write_yaml):
        path = _write_yaml("""
name: ci-pass
cases:
  - id: c1
    input: "q"
    expected:
      exact: "ok"
""")
        suite = EvalSuite.from_yaml(path)
        runner = EvalRunner()
        result = runner.run_ci(suite, lambda _: "ok", threshold=0.5)
        assert result.pass_rate == 1.0

    def test_run_ci_raises_below_threshold(self, _write_yaml):
        path = _write_yaml("""
name: ci-fail
cases:
  - id: c1
    input: "q"
    expected:
      exact: "ok"
""")
        suite = EvalSuite.from_yaml(path)
        runner = EvalRunner()
        with pytest.raises(RuntimeError, match="CI gate failed"):
            runner.run_ci(suite, lambda _: "nope", threshold=0.5)
