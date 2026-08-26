"""Focused edge-case tests for DeterministicScorer."""

import re

import pytest

from agenteval_bench.models import EvalCase, ExpectedOutput
from agenteval_bench.scoring import DeterministicScorer


class TestScoringEdgeCases:
    def test_empty_expected_returns_failure(self):
        scorer = DeterministicScorer()
        case = EvalCase(id="e1", input="q", expected=ExpectedOutput())
        result = scorer.score(case, "anything")
        assert result.passed is False
        assert result.score == 0.0
        assert result.details["error"] == "No expected output defined"

    def test_json_schema_missing_keys_fails(self):
        scorer = DeterministicScorer()
        schema = {"required": ["name", "age"]}
        case = EvalCase(id="e2", input="q", expected=ExpectedOutput(json_schema=schema))
        result = scorer.score(case, '{"name": "Alice"}')
        assert result.passed is False
        assert result.score == 0.0

    def test_json_schema_valid_json_missing_keys(self):
        scorer = DeterministicScorer()
        schema = {"required": ["id", "status"]}
        case = EvalCase(id="e3", input="q", expected=ExpectedOutput(json_schema=schema))
        result = scorer.score(case, '{"name": "Bob"}')
        assert result.passed is False

    def test_regex_compilation_error_raises(self):
        scorer = DeterministicScorer()
        case = EvalCase(id="e4", input="q", expected=ExpectedOutput(regex="[invalid("))
        with pytest.raises(re.error):
            scorer.score(case, "any text")

    def test_regex_no_match_returns_false(self):
        scorer = DeterministicScorer()
        case = EvalCase(id="e5", input="q", expected=ExpectedOutput(regex=r"\d{3}-\d{3}-\d{4}"))
        result = scorer.score(case, "no phone here")
        assert result.passed is False
        assert result.score == 0.0

    def test_case_insensitive_contains_passes(self):
        scorer = DeterministicScorer()
        case = EvalCase(
            id="e6",
            input="q",
            expected=ExpectedOutput(contains_ignore_case=["HELLO", "World"]),
        )
        result = scorer.score(case, "hello there WORLD")
        assert result.passed is True
        assert result.score == 1.0

    def test_case_insensitive_contains_fails_on_missing(self):
        scorer = DeterministicScorer()
        case = EvalCase(
            id="e7",
            input="q",
            expected=ExpectedOutput(contains_ignore_case=["HELLO", "missing"]),
        )
        result = scorer.score(case, "hello there")
        assert result.passed is False

    def test_contains_is_case_sensitive_by_default(self):
        scorer = DeterministicScorer()
        case = EvalCase(id="e8", input="q", expected=ExpectedOutput(contains=["HELLO"]))
        result = scorer.score(case, "hello there")
        assert result.passed is False

    def test_exact_match_with_whitespace_difference(self):
        scorer = DeterministicScorer()
        case = EvalCase(id="e9", input="q", expected=ExpectedOutput(exact="hello"))
        result = scorer.score(case, "  hello  ")
        assert result.passed is True

    def test_exact_match_strips_both_sides(self):
        scorer = DeterministicScorer()
        case = EvalCase(id="e10", input="q", expected=ExpectedOutput(exact="  hello  "))
        result = scorer.score(case, "hello")
        assert result.passed is True

    def test_json_schema_non_dict_with_required_fails(self):
        scorer = DeterministicScorer()
        schema = {"required": ["key"]}
        case = EvalCase(id="e11", input="q", expected=ExpectedOutput(json_schema=schema))
        result = scorer.score(case, '"just a string"')
        assert result.passed is False

    def test_json_schema_empty_required_passes_non_dict(self):
        scorer = DeterministicScorer()
        schema = {"required": []}
        case = EvalCase(id="e12", input="q", expected=ExpectedOutput(json_schema=schema))
        result = scorer.score(case, '"just a string"')
        assert result.passed is True

    def test_weighted_score_partial_pass(self):
        scorer = DeterministicScorer()
        case = EvalCase(
            id="e13",
            input="q",
            expected=ExpectedOutput(exact="result", contains=["data"]),
        )
        result = scorer.score(case, "result without data")
        assert result.passed is False
        assert result.score == 0.5

    def test_empty_contains_list_ignored(self):
        scorer = DeterministicScorer()
        case = EvalCase(id="e14", input="q", expected=ExpectedOutput(contains=[]))
        result = scorer.score(case, "anything")
        assert result.passed is False
        assert result.details["error"] == "No expected output defined"
