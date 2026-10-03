"""jev router: cost/latency-aware model routing.

Priority: local small LMs (Ollama) -> free tiers (GitHub Models, other free APIs)
-> paid escalation only when the task needs it.

Usage:
    from jev.router import route
    r = route("summarize this log")
    print(r.name, r.model, r.base_url)

GUARD (ADR-0009): the router ROUTES; it never JUDGES and never GATES.
It returns a Route, never a verdict, score, ranking, or approval. Escalation is
a routing choice (a different, more capable Route), not a gate on an outcome.
Do not add judge/verdict/gate/score functions to this package.
"""
from __future__ import annotations

import dataclasses
import os
import urllib.request
from dataclasses import dataclass, field

from jev.decisions import log_decision

# Complexity >= ESCALATE_AT means "too hard for the local SLM": escalate to a
# more capable route. ADR-0009 requires this threshold to be explicit,
# env-overridable, and audited — a magic number buried in code is how a
# mis-tuned router silently degrades quality.
ESCALATE_AT = float(os.getenv("JEV_ESCALATE_AT", "0.55"))

# The uncertain band around the threshold where the heuristic is least sure.
# A task landing here is escalated to the best *available* route even when the
# raw complexity would have kept it local, because the cost of a wrong "kept
# local" (an invalid, ~19x-more-likely bad SLM output) dwarfs the cost of a
# wrongly-escalated cheap free-tier call.
UNCERTAIN_BAND = float(os.getenv("JEV_UNCERTAIN_BAND", "0.10"))


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
    except OSError:  # availability probe: any network failure means "not alive"
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


def _confidence(complexity: float) -> float:
    """Distance from the escalation boundary, 0..1.

    The router's only claim is how far the task sits from the threshold it
    acted on — not a quality judgment, not a verdict (see GUARD above).
    """
    return min(1.0, abs(complexity - ESCALATE_AT) / max(ESCALATE_AT, 1e-9))


def _decide(complexity: float, local: Route, gh: Route, budget: str) -> tuple:
    """Return (route, kept_local, escalated, reason). Pure, for testing."""
    uncertain = abs(complexity - ESCALATE_AT) <= UNCERTAIN_BAND
    hard = complexity >= ESCALATE_AT
    if local.available and not hard and not uncertain:
        return (local, True, False,
                f"complexity {complexity:.2f} clearly below threshold {ESCALATE_AT}")
    # Escalate: to the more capable free route when it exists.
    if gh.available:
        if uncertain and not hard:
            reason = (f"complexity {complexity:.2f} in uncertain band "
                      f"[{ESCALATE_AT - UNCERTAIN_BAND:.2f}, {ESCALATE_AT + UNCERTAIN_BAND:.2f}]")
        else:
            reason = f"complexity {complexity:.2f} >= threshold {ESCALATE_AT}"
        return gh, False, True, reason
    if local.available:
        return (local, True, False,
                f"no capable route up; fell back to local (complexity {complexity:.2f})")
    raise RuntimeError("no model route available: start Ollama or set JEV_MODEL_TOKEN")


def route(task: str, budget: str = "free") -> Route:
    """Pick the cheapest route that can plausibly handle the task.

    budget: "free" (never paid), "balanced" (prefer free, allow paid when hard),
            "max-quality" (best capable route regardless of cost).

    Every decision is appended to the audit log (jev/decisions.jsonl) with the
    kept-local reason and confidence — ADR-0009.
    """
    complexity = estimate_complexity(task)
    routes = [dataclasses.replace(r) for r in default_routes()]

    local = routes[0]
    local.available = _ollama_alive(local.base_url)
    gh = routes[1]
    gh.available = bool(os.getenv(gh.api_key_env or ""))

    chosen, kept_local, escalated, reason = _decide(complexity, local, gh, budget)
    log_decision(task=task, route_name=chosen.name, route_kind=chosen.kind,
                 complexity=complexity, confidence=_confidence(complexity),
                 kept_local=kept_local, reason=reason,
                 escalated=escalated, budget=budget)
    return chosen
