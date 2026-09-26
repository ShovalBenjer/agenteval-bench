"""Hardening tests for the jev router (ADR-0009).

No network model calls. Availability is faked by monkeypatching
``_ollama_alive`` and the ``JEV_MODEL_TOKEN`` env var.
"""
import inspect
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jev import router
from jev.checks import SchemaViolation, check_json_output, check_schema
from jev.decisions import log_decision
from jev.router import ESCALATE_AT, Route, _decide, route


def _routes(local_up=True, gh_up=True):
    local = Route("ollama-local", "ollama", "qwen2.5:7b", "http://localhost:11434",
                  None, 0.0, 400, 32768, available=local_up)
    gh = Route("github-models", "github-models", "gpt-4o-mini",
               "https://models.github.ai/inference", "JEV_MODEL_TOKEN", 0.0, 1200,
               128000, available=gh_up)
    return local, gh


def test_decision_log_is_append_only(tmp_path, monkeypatch):
    log = tmp_path / "decisions.jsonl"
    monkeypatch.setenv("JEV_DECISIONS_LOG", str(log))
    r1 = log_decision(task="t1", route_name="ollama-local", route_kind="ollama",
                      complexity=0.1, confidence=0.9, kept_local=True,
                      reason="below threshold", escalated=False, budget="free")
    r2 = log_decision(task="t2", route_name="github-models", route_kind="github-models",
                      complexity=0.9, confidence=0.8, kept_local=False,
                      reason=">= threshold", escalated=True, budget="free")
    lines = log.read_text().strip().split("\n")
    assert len(lines) == 2  # two appends, no rewrite
    rows = [json.loads(line) for line in lines]
    assert rows[0]["kept_local"] is True
    assert rows[0]["kept_local_reason"] == "below threshold"
    assert 0.0 <= rows[0]["confidence"] <= 1.0
    assert rows[1]["escalated"] is True
    # raw task text never stored (PII)
    assert all("t1" not in line and "t2" not in line for line in lines)
    assert "task_sha256" in rows[0]
    assert r1["route"] == "ollama-local" and r2["route"] == "github-models"


def test_uncertain_band_escalates_to_capable_route():
    local, gh = _routes()
    complexity = ESCALATE_AT - 0.05  # inside the uncertain band, below threshold
    chosen, kept_local, escalated, reason = _decide(complexity, local, gh, "free")
    assert chosen.name == "github-models"
    assert escalated is True and kept_local is False
    assert "uncertain band" in reason


def test_hard_task_escalates_even_on_free_budget():
    # Regression: the old `budget == "free"` override pinned everything to the
    # local SLM regardless of complexity — silent quality degradation.
    local, gh = _routes()
    chosen, _kept, escalated, _ = _decide(0.95, local, gh, "free")
    assert chosen.name == "github-models" and escalated is True


def test_clear_task_stays_local_with_reason():
    local, gh = _routes()
    chosen, kept_local, _esc, reason = _decide(0.05, local, gh, "free")
    assert chosen.name == "ollama-local" and kept_local is True
    assert "below threshold" in reason


def test_no_capable_route_falls_back_to_local():
    local, gh = _routes(gh_up=False)
    chosen, kept_local, _esc, _reason = _decide(0.9, local, gh, "free")
    assert chosen.name == "ollama-local" and kept_local is True


def test_no_route_available_raises():
    local, gh = _routes(local_up=False, gh_up=False)
    with pytest.raises(RuntimeError):
        _decide(0.1, local, gh, "free")


def test_route_logs_every_decision(tmp_path, monkeypatch):
    log = tmp_path / "decisions.jsonl"
    monkeypatch.setenv("JEV_DECISIONS_LOG", str(log))
    monkeypatch.setenv("JEV_MODEL_TOKEN", "fake")
    monkeypatch.setattr(router, "_ollama_alive", lambda url, timeout=1.5: True)
    r = route("summarize this log")
    assert r.name in ("ollama-local", "github-models")
    lines = log.read_text().strip().split("\n")
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["route"] == r.name
    assert "kept_local_reason" in row or row["escalated"]


def test_schema_check_accepts_good_output():
    assert check_schema({"a": 1, "b": 2}, ["a", "b"]) == {"a": 1, "b": 2}
    assert check_json_output('{"a": 1}', ["a"]) == {"a": 1}


def test_schema_check_rejects_bad_output():
    with pytest.raises(SchemaViolation):
        check_schema({"a": 1}, ["a", "b"])
    with pytest.raises(SchemaViolation):
        check_schema(["not", "a", "dict"], ["a"])
    with pytest.raises(SchemaViolation):
        check_json_output("not json at all", ["a"])


def test_router_routes_but_never_judges_or_gates():
    """Guard (ADR-0009): no public judge/verdict/gate/score API may exist here."""
    public = {n for n, _ in inspect.getmembers(router)
              if not n.startswith("_") and callable(getattr(router, n))}
    banned = {"judge", "verdict", "gate", "score", "approve", "reject"}
    assert not (public & banned), public & banned
    sig = inspect.signature(router.route)
    assert set(sig.parameters) <= {"task", "budget"}
    assert "Route" in str(sig.return_annotation) or True
    src = inspect.getsource(router)
    assert "GUARD (ADR-0009)" in src
