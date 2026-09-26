"""jev router: cost/latency-aware model routing.

Priority: local small LMs (Ollama) -> free tiers (GitHub Models, other free APIs)
-> paid escalation only when the task needs it.

Usage:
    from jev.router import route
    r = route("summarize this log")
    print(r.name, r.model, r.base_url)
"""
from __future__ import annotations

import dataclasses
import os
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Route:
    name: str
    kind: str  # ollama | github-models | free-api | paid
    model: str
    base_url: str
    api_key_env: str | None
    cost_per_1k: float
    latency_ms_p50: int
    max_context: int
    available: bool = field(default=False, compare=False)


def _ollama_alive(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def default_routes() -> list[Route]:
    return [
        Route("ollama-local", "ollama",
              os.getenv("JEV_LOCAL_MODEL", "qwen2.5:7b"),
              "http://localhost:11434", None, 0.0, 400, 32768),
        Route("github-models", "github-models",
              os.getenv("JEV_GH_MODEL", "gpt-4o-mini"),
              "https://models.github.ai/inference", "JEV_MODEL_TOKEN", 0.0, 1200, 128000),
    ]


def estimate_complexity(task: str) -> float:
    """0..1 heuristic: short/simple tasks stay local, hard ones escalate."""
    t = task.lower()
    score = min(len(task) / 4000, 1.0) * 0.4
    hard = ("prove", "security", "architecture", "refactor", "distributed",
            "concurrency", "formal", "cryptograph")
    score += 0.15 * sum(1 for m in hard if m in t)
    return min(score, 1.0)


def route(task: str, budget: str = "free") -> Route:
    """Pick the cheapest route that can plausibly handle the task.

    budget: "free" (never paid), "balanced" (prefer free, allow paid when hard),
            "max-quality" (best capable route regardless of cost).
    """
    complexity = estimate_complexity(task)
    routes = [dataclasses.replace(r) for r in default_routes()]

    local = routes[0]
    local.available = _ollama_alive(local.base_url)
    gh = routes[1]
    gh.available = bool(os.getenv(gh.api_key_env or ""))

    # Local small LM wins for simple tasks when it is up.
    if local.available and (complexity < 0.55 or budget == "free"):
        return local
    if gh.available:
        return gh
    if local.available:
        return local
    raise RuntimeError("no model route available: start Ollama or set JEV_MODEL_TOKEN")
