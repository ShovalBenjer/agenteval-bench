"""Tests for run-vs-run comparison and regression detection."""

import json

import pytest

from agenteval_bench import cli
from agenteval_bench.compare import compare_runs, load_run
from agenteval_bench.models import EvalResult, RunResult


def _run(name: str, cases: dict[str, bool], skipped: list[str] | None = None) -> RunResult:
    """Build a RunResult from a case_id -> passed mapping."""
    results = [EvalResult(case_id=c, passed=p, score=1.0 if p else 0.0) for c, p in cases.items()]
    for c in skipped or []:
        results.append(EvalResult(case_id=c, passed=False, score=0.0, details={"skipped": True}))
    passed = sum(1 for p in cases.values() if p)
    scored = len(cases)
    return RunResult(
        suite_name=name,
        results=results,
        total=len(results),
        passed=passed,
        failed=scored - passed,
        skipped=len(skipped or []),
        pass_rate=passed / scored if scored else 0.0,
    )


class TestCompareRuns:
    def test_no_regression_when_identical(self):
        baseline = _run("suite", {"a": True, "b": True})
        candidate = _run("suite", {"a": True, "b": True})
        comparison = compare_runs(baseline, candidate)
        assert comparison.is_regression is False
        assert comparison.delta == 0.0
        assert comparison.regressed_cases == []

    def test_flags_regression_past_threshold(self):
        baseline = _run("suite", {"a": True, "b": True})
        candidate = _run("suite", {"a": True, "b": False})  # 100% -> 50%
        comparison = compare_runs(baseline, candidate, threshold=0.05)
        assert comparison.is_regression is True
        assert comparison.regressed_cases == ["b"]
        assert comparison.delta == -0.5

    def test_tolerates_drop_within_threshold(self):
        cases_pass = {f"c{i}": True for i in range(100)}
        baseline = _run("suite", cases_pass)
        cases_one_fail = dict(cases_pass, c99=False)  # 100% -> 99%
        candidate = _run("suite", cases_one_fail)
        comparison = compare_runs(baseline, candidate, threshold=0.05)
        assert comparison.is_regression is False
        assert comparison.regressed_cases == ["c99"]  # still reported, just not fatal

    def test_reports_improvements(self):
        baseline = _run("suite", {"a": False, "b": True})
        candidate = _run("suite", {"a": True, "b": True})
        comparison = compare_runs(baseline, candidate)
        assert comparison.improved_cases == ["a"]
        assert comparison.is_regression is False

    def test_excludes_skipped_cases(self):
        baseline = _run("suite", {"a": True}, skipped=["flaky"])
        candidate = _run("suite", {"a": True, "flaky": False})
        comparison = compare_runs(baseline, candidate)
        # "flaky" was skipped in baseline, so it can't regress
        assert comparison.regressed_cases == []

    def test_ignores_cases_missing_from_either_run(self):
        baseline = _run("suite", {"a": True, "removed": True})
        candidate = _run("suite", {"a": True, "added": False})
        comparison = compare_runs(baseline, candidate)
        assert comparison.regressed_cases == []
        assert comparison.improved_cases == []

    def test_invalid_threshold_raises(self):
        run = _run("suite", {"a": True})
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            compare_runs(run, run, threshold=-0.1)

    def test_summary_mentions_regression(self):
        baseline = _run("suite", {"a": True})
        candidate = _run("suite", {"a": False})
        comparison = compare_runs(baseline, candidate)
        assert "REGRESSION DETECTED" in comparison.summary()
        no_change = compare_runs(baseline, baseline)
        assert "No regression" in no_change.summary()


class TestRunPersistence:
    def test_from_dict_round_trip(self, tmp_path):
        import dataclasses

        original = _run("suite", {"a": True, "b": False})
        path = tmp_path / "run.json"
        path.write_text(json.dumps(dataclasses.asdict(original)))
        loaded = load_run(str(path))
        assert loaded.suite_name == original.suite_name
        assert loaded.pass_rate == original.pass_rate
        assert [r.case_id for r in loaded.results] == [r.case_id for r in original.results]
        assert [r.passed for r in loaded.results] == [r.passed for r in original.results]

    def test_from_dict_rejects_invalid(self):
        with pytest.raises(ValueError, match="suite_name"):
            RunResult.from_dict({"not": "a run"})


class TestCompareCommand:
    def _write_run(self, tmp_path, name, cases):
        import dataclasses

        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(dataclasses.asdict(_run("suite", cases))))
        return str(path)

    def test_exits_zero_without_regression(self, tmp_path, capsys):
        a = self._write_run(tmp_path, "a", {"c1": True})
        b = self._write_run(tmp_path, "b", {"c1": True})
        cli.main(["compare", a, b])
        assert "No regression" in capsys.readouterr().out

    def test_exits_one_on_regression(self, tmp_path, capsys):
        a = self._write_run(tmp_path, "a", {"c1": True, "c2": True})
        b = self._write_run(tmp_path, "b", {"c1": False, "c2": False})
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["compare", a, b])
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "REGRESSION DETECTED" in out
        assert "c1" in out and "c2" in out

    def test_exits_two_on_missing_file(self, tmp_path, capsys):
        a = self._write_run(tmp_path, "a", {"c1": True})
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["compare", a, str(tmp_path / "nope.json")])
        assert exc_info.value.code == 2
        assert "Error loading run" in capsys.readouterr().err

    def test_custom_threshold(self, tmp_path):
        # 50% drop tolerated with threshold 0.6
        a = self._write_run(tmp_path, "a", {"c1": True, "c2": True})
        b = self._write_run(tmp_path, "b", {"c1": True, "c2": False})
        cli.main(["compare", a, b, "--threshold", "0.6"])  # no SystemExit
