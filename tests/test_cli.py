"""Tests for the CLI and CI threshold gate."""

import json

import pytest

from agenteval_bench import cli
from agenteval_bench.engine import EvalRunner
from agenteval_bench.models import EvalSuite


SUITE_YAML = """
name: cli-suite
cases:
  - id: math
    input: "what is 2+2?"
    expected:
      exact: "4"
  - id: greeting
    input: "greet me"
    expected:
      contains: ["hello"]
"""

AGENT_MODULE = '''
def good_agent(prompt):
    if "2+2" in prompt:
        return "4"
    return "hello there"


def bad_agent(prompt):
    return "wrong every time"


not_callable = "just a string"
'''


@pytest.fixture
def suite_path(tmp_path):
    path = tmp_path / "suite.yaml"
    path.write_text(SUITE_YAML)
    return str(path)


@pytest.fixture
def agent_module(tmp_path, monkeypatch):
    (tmp_path / "fake_agent.py").write_text(AGENT_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))


# --- Agent loading ---

class TestLoadAgent:
    def test_loads_valid_agent(self, agent_module):
        agent_fn = cli._load_agent("fake_agent:good_agent")
        assert agent_fn("what is 2+2?") == "4"

    def test_rejects_missing_colon(self):
        with pytest.raises(ValueError, match="expected 'module.path:function'"):
            cli._load_agent("fake_agent.good_agent")

    def test_rejects_missing_attribute(self, agent_module):
        with pytest.raises(AttributeError, match="no attribute"):
            cli._load_agent("fake_agent:nonexistent")

    def test_rejects_non_callable(self, agent_module):
        with pytest.raises(TypeError, match="not callable"):
            cli._load_agent("fake_agent:not_callable")

    def test_rejects_missing_module(self):
        with pytest.raises(ImportError):
            cli._load_agent("definitely_not_a_module:fn")


# --- run command ---

class TestRunCommand:
    def test_validate_only_without_agent(self, suite_path, capsys):
        cli.main(["run", "--suite", suite_path])
        out = capsys.readouterr().out
        assert "cli-suite" in out
        assert "2 cases loaded" in out

    def test_runs_agent_and_passes(self, suite_path, agent_module, capsys):
        cli.main(["run", "--suite", suite_path, "--agent", "fake_agent:good_agent"])
        out = capsys.readouterr().out
        assert "Passed: 2" in out
        assert "100.0%" in out

    def test_ci_mode_exits_zero_when_threshold_met(self, suite_path, agent_module):
        cli.main(["run", "--suite", suite_path, "--agent", "fake_agent:good_agent", "--ci"])
        # no SystemExit raised means exit code 0

    def test_ci_mode_exits_nonzero_on_failure(self, suite_path, agent_module, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["run", "--suite", suite_path, "--agent", "fake_agent:bad_agent", "--ci"])
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "below threshold" in err

    def test_ci_mode_respects_threshold(self, suite_path, agent_module, capsys):
        # bad_agent passes 0/2 — even threshold 0.0 is met (0.0 >= 0.0)
        cli.main([
            "run", "--suite", suite_path,
            "--agent", "fake_agent:bad_agent",
            "--ci", "--threshold", "0.0",
        ])
        out = capsys.readouterr().out
        assert "Failed: 2" in out

    def test_ci_without_agent_exits_two(self, suite_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["run", "--suite", suite_path, "--ci"])
        assert exc_info.value.code == 2

    def test_bad_agent_spec_exits_two(self, suite_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["run", "--suite", suite_path, "--agent", "no_such_module:fn"])
        assert exc_info.value.code == 2
        assert "Error loading agent" in capsys.readouterr().err

    def test_output_writes_json(self, suite_path, agent_module, tmp_path, capsys):
        out_file = tmp_path / "results.json"
        cli.main([
            "run", "--suite", suite_path,
            "--agent", "fake_agent:good_agent",
            "--output", str(out_file),
        ])
        data = json.loads(out_file.read_text())
        assert data["suite_name"] == "cli-suite"
        assert data["passed"] == 2
        assert len(data["results"]) == 2


# --- run_ci engine behavior ---

class TestRunCi:
    def _suite(self, tmp_path):
        path = tmp_path / "suite.yaml"
        path.write_text(SUITE_YAML)
        return EvalSuite.from_yaml(str(path))

    def test_threshold_met(self, tmp_path):
        suite = self._suite(tmp_path)
        runner = EvalRunner()
        result = runner.run_ci(
            suite,
            lambda p: "4" if "2+2" in p else "hello",
            threshold=1.0,
        )
        assert result.threshold_met is True

    def test_threshold_not_met(self, tmp_path):
        suite = self._suite(tmp_path)
        runner = EvalRunner()
        result = runner.run_ci(suite, lambda p: "always wrong", threshold=0.5)
        assert result.threshold_met is False

    def test_partial_pass_against_half_threshold(self, tmp_path):
        suite = self._suite(tmp_path)
        runner = EvalRunner()
        # Passes math (exact "4") but not greeting -> 50% pass rate
        result = runner.run_ci(suite, lambda p: "4", threshold=0.5)
        assert result.pass_rate == 0.5
        assert result.threshold_met is True

    def test_invalid_threshold_raises(self, tmp_path):
        suite = self._suite(tmp_path)
        runner = EvalRunner()
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            runner.run_ci(suite, lambda p: "4", threshold=1.5)

    def test_plain_run_leaves_threshold_unset(self, tmp_path):
        suite = self._suite(tmp_path)
        runner = EvalRunner()
        result = runner.run(suite, lambda p: "4")
        assert result.threshold_met is None
