"""Roadmap/spec consistency enforcement (issue #16).

The roadmap (`TODO.md`) must stay in sync with `docs/spec.md` and its own
links must resolve. Prose is not enforcement: these invariants run in CI so
roadmap drift fails the build instead of rotting silently.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
ROADMAP: Path = REPO_ROOT / "TODO.md"
SPEC: Path = REPO_ROOT / "docs" / "spec.md"

# (roadmap workstream keyword, spec feature-table row keyword).
# Every open workstream must have a counterpart in the spec's feature table.
WORKSTREAM_SPEC_PAIRS: tuple[tuple[str, str], ...] = (
    ("LLM-as-judge", "LLM-as-judge scoring"),
    ("agenteval-bench compare", "`compare` regression"),
    ("Report generation", "`report` (md + json)"),
    ("cost_bound", "`cost_bound` enforced"),
    ("typer-based CLI", "typer CLI + rich output"),
    ("Plugin system", "Custom scorer plugins"),
    ("Dashboard", "Historical dashboard"),
)

# Markdown links like [text](target); excludes http(s)/mailto links.
LINK_RE: re.Pattern[str] = re.compile(r"\[[^\]]*\]\(([^)#\s][^)\s]*)\)")


def _section(text: str, heading: str) -> str:
    """Return the body of the markdown section under `heading`."""
    match = re.search(rf"^## {re.escape(heading)}\s*$([\s\S]*?)(?=^## |\Z)", text, re.MULTILINE)
    assert match is not None, f"missing section: {heading}"
    return match.group(1)


def test_roadmap_links_resolve() -> None:
    """Every relative markdown link in TODO.md resolves to a repo path."""
    text: str = ROADMAP.read_text(encoding="utf-8")
    targets: list[str] = [
        t for t in LINK_RE.findall(text) if not re.match(r"^(https?|mailto):", t)
    ]
    assert targets, "TODO.md has no relative links to check"
    missing: list[str] = [
        t for t in targets if not (REPO_ROOT / t).exists()
    ]
    assert not missing, f"TODO.md links to missing paths: {missing}"


def test_workstreams_synced_with_spec() -> None:
    """Each open workstream has a matching feature row in docs/spec.md."""
    roadmap: str = ROADMAP.read_text(encoding="utf-8")
    spec: str = SPEC.read_text(encoding="utf-8")
    workstreams: str = _section(roadmap, "Open Workstreams")
    for roadmap_key, spec_key in WORKSTREAM_SPEC_PAIRS:
        assert roadmap_key in workstreams, (
            f"roadmap lost workstream {roadmap_key!r}; update the pairs or the roadmap"
        )
        assert spec_key in spec, (
            f"spec.md missing feature-table row {spec_key!r} "
            f"for roadmap workstream {roadmap_key!r}"
        )


def test_no_stale_done_workstreams() -> None:
    """Workstreams marked shipped in the roadmap belong in the archive."""
    roadmap: str = ROADMAP.read_text(encoding="utf-8")
    workstreams: str = _section(roadmap, "Open Workstreams")
    shipped: list[str] = re.findall(r"✅|DONE", workstreams)
    assert not shipped, (
        "Open Workstreams contains shipped markers; move them to "
        "docs/archive/roadmap-archive.md"
    )
