"""Smoke test for the jev router. No network model calls; only availability probes."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jev.router import estimate_complexity, route  # noqa: E402


def test_complexity_orders_tasks():
    assert estimate_complexity("hi") < estimate_complexity(
        "prove the distributed refactor is free of concurrency bugs")


def test_route_returns_a_route_object():
    try:
        r = route("summarize this log")
    except RuntimeError:
        return  # acceptable: nothing configured in CI
    assert r.name in ("ollama-local", "github-models")
    assert r.base_url.startswith("http")
