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


def _run(argv, monkeypatch):
    monkeypatch.setattr("sys.argv", ["agenteval-bench", *argv])
    try:
        cli.main()
        return 0
    except SystemExit as e:
        return e.code or 0


def test_ci_gate_fails_below_threshold(monkeypatch):
    # 1 of 2 pass -> 50%. Threshold 0.9 must FAIL (exit 2).
    path = _write(REPLAY_SUITE)
    try:
        code = _run(["run", "--suite", path, "--ci", "--threshold", "0.9"], monkeypatch)
        assert code == 2
    finally:
        os.unlink(path)


def test_ci_gate_passes_at_or_below_actual(monkeypatch):
    # 50% pass rate, threshold 0.5 -> PASS (exit 0).
    path = _write(REPLAY_SUITE)
    try:
        code = _run(["run", "--suite", path, "--ci", "--threshold", "0.5"], monkeypatch)
        assert code == 0
    finally:
        os.unlink(path)


def test_non_ci_run_never_exits_nonzero(monkeypatch):
    # Without --ci, a low pass rate still exits 0 (reporting only).
    path = _write(REPLAY_SUITE)
    try:
        code = _run(["run", "--suite", path], monkeypatch)
        assert code == 0
    finally:
        os.unlink(path)
