"""CLI tests: help, flags, error paths, and subcommands."""

import pytest

from agenteval_bench import cli


REPLAY_SUITE = """
name: replay
cases:
  - id: good
    input: "q1"
    output: "the answer contains hello"
    expected:
      contains: ["hello"]
  - id: bad
    input: "q2"
    output: "totally unrelated"
    expected:
      contains: ["goodbye"]
"""


def _run(argv, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["agenteval-bench", *argv])
    try:
        cli.main()
        captured = capsys.readouterr()
        return 0, captured.out, captured.err
    except SystemExit as e:
        captured = capsys.readouterr()
        return e.code or 0, captured.out, captured.err


def test_ci_gate_fails_below_threshold(monkeypatch, _write_yaml, capsys):
    path = _write_yaml(REPLAY_SUITE)
    code, _, _ = _run(["run", "--suite", path, "--ci", "--threshold", "0.9"], monkeypatch, capsys)
    assert code == 2


def test_ci_gate_passes_at_or_below_actual(monkeypatch, _write_yaml, capsys):
    path = _write_yaml(REPLAY_SUITE)
    code, _, _ = _run(["run", "--suite", path, "--ci", "--threshold", "0.5"], monkeypatch, capsys)
    assert code == 0


def test_non_ci_run_never_exits_nonzero(monkeypatch, _write_yaml, capsys):
    path = _write_yaml(REPLAY_SUITE)
    code, _, _ = _run(["run", "--suite", path], monkeypatch, capsys)
    assert code == 0


def test_help_output(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["agenteval-bench", "--help"])
    try:
        cli.main()
    except SystemExit as e:
        assert e.code == 0
    out, _ = capsys.readouterr()
    assert "agenteval-bench" in out
    assert "Usage" in out


def test_missing_suite_flag(monkeypatch, capsys):
    code, _, err = _run(["run", "--ci"], monkeypatch, capsys)
    assert code == 1
    assert "--suite" in err


def test_invalid_subcommand(monkeypatch, capsys):
    code, _, err = _run(["bogus"], monkeypatch, capsys)
    assert code == 1
    assert "Unknown command" in err


def test_out_of_range_threshold(monkeypatch, _write_yaml, capsys):
    path = _write_yaml(REPLAY_SUITE)
    code, _, _ = _run(["run", "--suite", path, "--ci", "--threshold", "1.5"], monkeypatch, capsys)
    assert code == 2


def test_file_not_found(monkeypatch, capsys):
    code, _, err = _run(["run", "--suite", "/nonexistent/suite.yaml"], monkeypatch, capsys)
    assert code == 1
    assert "not found" in err


def test_malformed_yaml(monkeypatch, _write_yaml, capsys):
    path = _write_yaml("name: \"unclosed\n")
    code, _, err = _run(["run", "--suite", path], monkeypatch, capsys)
    assert code == 1
    assert "malformed YAML" in err


def test_compare_subcommand_stub(monkeypatch, capsys):
    code, out, _ = _run(["compare", "a", "b"], monkeypatch, capsys)
    assert code == 0
    assert "not yet implemented" in out


def test_report_subcommand_stub(monkeypatch, capsys):
    code, out, _ = _run(["report"], monkeypatch, capsys)
    assert code == 0
    assert "not yet implemented" in out
