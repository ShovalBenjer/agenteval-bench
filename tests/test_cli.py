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


def test_bugsmith_requires_repo_and_out(monkeypatch, capsys):
    code, _, err = _run(["bugsmith", "--repo", "x"], monkeypatch, capsys)
    assert code == 1
    assert "--repo" in err or "repo" in err.lower()


def test_bugsmith_rejects_missing_repo(monkeypatch, capsys):
    code, _, err = _run(
        ["bugsmith", "--repo", "/nonexistent", "--out", "/tmp/x"],
        monkeypatch, capsys,
    )
    assert code == 1
    assert "not found" in err


def test_bugsmith_end_to_end_local(monkeypatch, capsys, tmp_path):
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1] / "src" / "bugsmith" / "fixtures" / "target"
    out = tmp_path / "bench"
    code, stdout, _ = _run(
        ["bugsmith", "--repo", str(repo), "--out", str(out),
         "--procedural", "4", "--seed", "5", "--local"],
        monkeypatch, capsys,
    )
    assert code == 0, stdout
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["baseline_tests"] == 16
    assert manifest["valid_instances"] >= 1
    assert 1 <= len(manifest["curated"]) <= manifest["valid_instances"]
    for name in manifest["curated"]:
        inst = json.loads((out / f"{name}.json").read_text())
        assert inst["fail_to_pass"], "curated instance must break >= 1 test"
        assert inst["seed"] == 5


def test_bugsmith_unknown_command(monkeypatch, capsys):
    code, _, err = _run(["frobnicate"], monkeypatch, capsys)
    assert code == 1
    assert "Unknown command" in err
