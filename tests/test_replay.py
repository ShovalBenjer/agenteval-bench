"""Tests for deterministic seeded replay and delta-baseline scoring."""

import importlib.util
import json
import os
import tempfile

from agenteval_bench import cli
from agenteval_bench.engine import EvalRunner
from agenteval_bench.models import EvalSuite

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


def _load_paired_bootstrap():
    path = os.path.join(os.path.dirname(__file__), "..", "tools", "paired_bootstrap.py")
    spec = importlib.util.spec_from_file_location("paired_bootstrap", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(content: str, suffix: str = ".yaml") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
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


def _replay_log_bytes(monkeypatch, capsys, suite_path, seed=42):
    log = tempfile.mktemp(suffix=".json")
    code, out, _ = _run(
        ["run", "--suite", suite_path, "--seed", str(seed), "--replay-log", log],
        monkeypatch, capsys)
    assert code == 0, out
    with open(log, encoding="utf-8") as f:
        return f.read()


def test_replay_is_bit_exact(monkeypatch, capsys):
    """Same suite + same seed run twice -> byte-identical replay log."""
    path = _write(REPLAY_SUITE)
    try:
        first = _replay_log_bytes(monkeypatch, capsys, path)
        second = _replay_log_bytes(monkeypatch, capsys, path)
        assert first == second
    finally:
        os.unlink(path)


def test_replay_check_passes(monkeypatch, capsys):
    path = _write(REPLAY_SUITE)
    try:
        log_bytes = _replay_log_bytes(monkeypatch, capsys, path)
        fd, log_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write(log_bytes)
        code, out, _ = _run(
            ["replay", "--log", log_path, "--suite", path, "--check"],
            monkeypatch, capsys)
        assert code == 0, out
        assert "bit-exact" in out
        os.unlink(log_path)
    finally:
        os.unlink(path)


def test_replay_check_detects_tampering(monkeypatch, capsys):
    path = _write(REPLAY_SUITE)
    try:
        log_bytes = _replay_log_bytes(monkeypatch, capsys, path)
        doc = json.loads(log_bytes)
        doc["cases"][0]["score"] = 0.0 if doc["cases"][0]["score"] else 1.0
        fd, log_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(doc, sort_keys=True, indent=2) + "\n")
        code, out, _ = _run(
            ["replay", "--log", log_path, "--suite", path, "--check"],
            monkeypatch, capsys)
        assert code == 1
        assert "NOT bit-exact" in out
        os.unlink(log_path)
    finally:
        os.unlink(path)


def test_replay_check_rejects_suite_change(monkeypatch, capsys):
    path = _write(REPLAY_SUITE)
    try:
        log_bytes = _replay_log_bytes(monkeypatch, capsys, path)
        fd, log_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write(log_bytes)
        other = _write(REPLAY_SUITE + "\n# comment changes digest\n")
        code, _out, err = _run(
            ["replay", "--log", log_path, "--suite", other, "--check"],
            monkeypatch, capsys)
        assert code == 1
        assert "digest mismatch" in err
        os.unlink(log_path)
        os.unlink(other)
    finally:
        os.unlink(path)


def test_different_seeds_give_different_rng_streams(monkeypatch, capsys):
    path = _write(REPLAY_SUITE)
    try:
        a = _replay_log_bytes(monkeypatch, capsys, path, seed=1)
        b = _replay_log_bytes(monkeypatch, capsys, path, seed=2)
        da = json.loads(a)["cases"][0]["rng_draw"]
        db = json.loads(b)["cases"][0]["rng_draw"]
        assert da != db
    finally:
        os.unlink(path)


def test_engine_records_rng_draw_per_case():
    suite = EvalSuite.from_yaml(_write(REPLAY_SUITE))
    runner = EvalRunner()
    r1 = runner.run(suite, lambda i: "hello", seed=7)
    r2 = runner.run(suite, lambda i: "hello", seed=7)
    draws1 = [r.details["rng_draw"] for r in r1.results]
    draws2 = [r.details["rng_draw"] for r in r2.results]
    assert draws1 == draws2  # same seed -> identical stream
    r3 = runner.run(suite, lambda i: "hello", seed=8)
    assert [r.details["rng_draw"] for r in r3.results] != draws1


def _score_file(scores):
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(str(s) for s in scores) + "\n")
    return path


def test_delta_report_verdicts():
    pb = _load_paired_bootstrap()
    # clear improvement
    rep = pb.delta_report([0.5] * 40, [0.8] * 40, n_boot=200, seed=1)
    assert rep["verdict"] == "improvement"
    assert rep["mean_diff"] > 0
    # clear regression
    rep = pb.delta_report([0.8] * 40, [0.5] * 40, n_boot=200, seed=1)
    assert rep["verdict"] == "regression"
    # noisy tie -> inconclusive
    rep = pb.delta_report([0.5, 0.9] * 20, [0.9, 0.5] * 20, n_boot=200, seed=1)
    assert rep["verdict"] == "inconclusive"


def test_snapshot_roundtrip_and_compare():
    pb = _load_paired_bootstrap()
    scores_path = _score_file([0.5] * 30)
    snap = tempfile.mktemp(suffix=".json")
    try:
        pb.write_snapshot(snap, [0.5] * 30, {"seed": 42, "note": "test"})
        scores, meta, case_ids = pb.read_snapshot(snap)
        assert scores == [0.5] * 30
        assert meta["seed"] == 42
        assert case_ids is None  # plain score list has no case ids
        cand = _score_file([0.2] * 30)
        rep = pb.delta_report(scores, [0.2] * 30, n_boot=200, seed=1)
        assert rep["verdict"] == "regression"
        os.unlink(cand)
    finally:
        os.unlink(scores_path)
        if os.path.exists(snap):
            os.unlink(snap)


def test_baseline_command_writes_snapshot(monkeypatch, capsys):
    pb = _load_paired_bootstrap()
    path = _write(REPLAY_SUITE)
    out = tempfile.mktemp(suffix=".json")
    try:
        code, _out, _ = _run(
            ["baseline", "--suite", path, "--out", out, "--seed", "42"],
            monkeypatch, capsys)
        assert code == 0
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["seed"] == 42
        assert len(doc["scores"]) == 2
        assert doc["suite_digest"].startswith("sha256:")
        # baseline snapshot must be comparable by paired_bootstrap
        scores = [s["score"] for s in doc["scores"]]
        cand = _score_file([0.0, 0.0])
        rep = pb.delta_report(scores, [0.0, 0.0], n_boot=200, seed=1)
        # n=2 rows: bootstrap cannot reach significance, but the candidate is
        # strictly worse, so the verdict must never be "improvement"
        assert rep["mean_diff"] < 0
        assert rep["verdict"] != "improvement"
        os.unlink(cand)
        os.unlink(out)
    finally:
        os.unlink(path)
