"""CLI CI-gate tests: a replay suite must exit non-zero below threshold."""

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


def _run(argv, monkeypatch):
    monkeypatch.setattr("sys.argv", ["agenteval-bench", *argv])
    try:
        cli.main()
        return 0
    except SystemExit as e:
        return e.code or 0


def test_ci_gate_fails_below_threshold(monkeypatch, write_yaml):
    path = write_yaml(REPLAY_SUITE)
    code = _run(["run", "--suite", path, "--ci", "--threshold", "0.9"], monkeypatch)
    assert code == 2


def test_ci_gate_passes_at_or_below_actual(monkeypatch, write_yaml):
    path = write_yaml(REPLAY_SUITE)
    code = _run(["run", "--suite", path, "--ci", "--threshold", "0.5"], monkeypatch)
    assert code == 0


def test_non_ci_run_never_exits_nonzero(monkeypatch, write_yaml):
    path = write_yaml(REPLAY_SUITE)
    code = _run(["run", "--suite", path], monkeypatch)
    assert code == 0
