"""CLI CI-gate tests: a replay suite must exit non-zero below threshold."""

import os
import tempfile

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


def _write(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def _run(argv, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["agenteval-bench", *argv])
    try:
        cli.main()
        captured = capsys.readouterr()
        return 0, captured.out, captured.err
    except SystemExit as e:
        captured = capsys.readouterr()
        return e.code or 0, captured.out, captured.err


def test_ci_gate_fails_below_threshold(monkeypatch, capsys):
    # 1 of 2 pass -> 50%. Threshold 0.9 must FAIL (exit 2).
    path = _write(REPLAY_SUITE)
    try:
        code, _, _ = _run(["run", "--suite", path, "--ci", "--threshold", "0.9"], monkeypatch, capsys)
        assert code == 2
    finally:
        os.unlink(path)


def test_ci_gate_passes_at_or_below_actual(monkeypatch, capsys):
    # 50% pass rate, threshold 0.5 -> PASS (exit 0).
    path = _write(REPLAY_SUITE)
    try:
        code, _, _ = _run(["run", "--suite", path, "--ci", "--threshold", "0.5"], monkeypatch, capsys)
        assert code == 0
    finally:
        os.unlink(path)


def test_non_ci_run_never_exits_nonzero(monkeypatch, capsys):
    # Without --ci, a low pass rate still exits 0 (reporting only).
    path = _write(REPLAY_SUITE)
    try:
        code, _, _ = _run(["run", "--suite", path], monkeypatch, capsys)
        assert code == 0
    finally:
        os.unlink(path)


def test_ci_failure_prints_summary(monkeypatch, capsys):
    # When the CI gate fails, the summary and gate status must still be printed.
    path = _write(REPLAY_SUITE)
    try:
        code, out, _ = _run(["run", "--suite", path, "--ci", "--threshold", "0.9"], monkeypatch, capsys)
        assert code == 2
        assert "Suite: replay" in out
        assert "Pass rate:" in out
        assert "CI gate" in out
        assert "FAIL" in out
    finally:
        os.unlink(path)


def test_out_of_range_threshold_high(monkeypatch, capsys):
    # Threshold above 1.0 is invalid and must produce a user-facing error.
    path = _write(REPLAY_SUITE)
    try:
        code, _, err = _run(["run", "--suite", path, "--ci", "--threshold", "1.5"], monkeypatch, capsys)
        assert code == 1
        assert "threshold" in err.lower()
    finally:
        os.unlink(path)


def test_out_of_range_threshold_low(monkeypatch, capsys):
    # Threshold below 0.0 is invalid and must produce a user-facing error.
    path = _write(REPLAY_SUITE)
    try:
        code, _, err = _run(["run", "--suite", path, "--ci", "--threshold", "-0.5"], monkeypatch, capsys)
        assert code == 1
        assert "threshold" in err.lower()
    finally:
        os.unlink(path)
