"""CLI tests for the simpson subcommand (agenteval-bench#36).

The harness must refuse a pooled win claim from the command line, not
just from the library: the Kohavi Friday/Saturday slices exit 1 with
the reversal named, stable slices exit 0, and the disaggregation is
printed on every path.
"""

import json
import os
import tempfile

from agenteval_bench import cli


def _write(doc: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(doc, f)
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


KOHAVI_SLICES = {
    "pooled_n_treated": 6000,
    "pooled_n_control": 104000,
    "strata": [
        {"name": "friday", "n_treated": 1000, "n_control": 99000,
         "treated_successes": 23, "control_successes": 2000},
        {"name": "saturday", "n_treated": 5000, "n_control": 5000,
         "treated_successes": 60, "control_successes": 50},
    ],
}

STABLE_SLICES = {
    "pooled_n_treated": 10000,
    "pooled_n_control": 10000,
    "strata": [
        {"name": "a", "n_treated": 5000, "n_control": 5000,
         "treated_successes": 115, "control_successes": 101},
        {"name": "b", "n_treated": 5000, "n_control": 5000,
         "treated_successes": 60, "control_successes": 50},
    ],
}


def test_simpson_cli_rejects_kohavi_reversal(monkeypatch, capsys):
    path = _write(KOHAVI_SLICES)
    try:
        code, out, _ = _run(["simpson", "--slices", path], monkeypatch, capsys)
        assert code == 1
        assert "WIN CLAIM REJECTED" in out
        assert "SIMPSON_REVERSAL" in out
        # Mandatory disaggregation is printed even on the refusal path.
        assert "friday" in out and "saturday" in out
        assert "pooled (descriptive only, not a result)" in out
    finally:
        os.unlink(path)


def test_simpson_cli_accepts_stable_win(monkeypatch, capsys):
    path = _write(STABLE_SLICES)
    try:
        code, out, _ = _run(["simpson", "--slices", path], monkeypatch, capsys)
        assert code == 0
        assert "WIN CLAIM ACCEPTED" in out
    finally:
        os.unlink(path)


def test_simpson_cli_writes_report_json(monkeypatch, capsys, tmp_path):
    path = _write(KOHAVI_SLICES)
    report = str(tmp_path / "report.json")
    try:
        code, _, _ = _run(
            ["simpson", "--slices", path, "--out", report], monkeypatch, capsys
        )
        assert code == 1
        with open(report, encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["accepted"] is False
        assert "SIMPSON_REVERSAL" in payload["violations"]
        assert len(payload["strata"]) == 2
    finally:
        os.unlink(path)


def test_simpson_cli_bad_input_exits_2(monkeypatch, capsys):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        f.write("{not json")
    try:
        code, _, err = _run(["simpson", "--slices", path], monkeypatch, capsys)
        assert code == 2
        assert "invalid slices doc" in err
    finally:
        os.unlink(path)
    code, _, err = _run(["simpson"], monkeypatch, capsys)
    assert code == 2
