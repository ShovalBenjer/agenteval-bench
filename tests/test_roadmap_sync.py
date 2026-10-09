"""Roadmap/spec consistency enforcement (issue #16).

Generative invariants: sync obligations are declared IN the docs; the test
enumerates and resolves them. Adding an Open Workstream bullet without a
`spec:` anchor fails CI — docs authors, not test authors, own the extension.

Accepted coupling: a `spec:` anchor names a feature-table row's first-column
text in docs/spec.md (the table whose header is Feature|Status|Est. milestone).
If a spec author rewords that row, the build fails with a message naming the
bullet and the anchor — the fix is a docs edit, never test surgery.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
ROADMAP: Path = REPO_ROOT / "TODO.md"
SPEC: Path = REPO_ROOT / "docs" / "spec.md"

# "- [ ] Title *(v0.2)* — spec: Anchor text". The anchor is the first-column
# text of the corresponding feature row in docs/spec.md (backticks ignored).
WORKSTREAM_RE: re.Pattern[str] = re.compile(
    r"^-\s+\[[ xX]\]\s+(?P<title>.+?)\s+—\s+spec:\s*(?P<anchor>.+?)\s*$"
)
BULLET_RE: re.Pattern[str] = re.compile(r"^\s*[-*]\s+\[[ xX]\]\s+\S")
TICKED_RE: re.Pattern[str] = re.compile(r"^\s*[-*]\s+\[[xX]\]")

# Inline links [text](target); the target may carry a #fragment.
LINK_RE: re.Pattern[str] = re.compile(r"\[[^\]]*\]\((?P<target>[^)\s]+)\)")
HEADING_RE: re.Pattern[str] = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$")
SCHEME_RE: re.Pattern[str] = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _norm(text: str) -> str:
    """Normalize doc text for comparison: drop backticks, collapse whitespace."""
    return re.sub(r"\s+", " ", text.replace("`", "")).strip()


def _section(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$([\s\S]*?)(?=^## |\Z)", text, re.MULTILINE
    )
    assert match is not None, f"missing section '## {heading}' in TODO.md"
    return match.group(1)


def _workstream_bullets() -> list[str]:
    section = _section(ROADMAP.read_text(encoding="utf-8"), "Open Workstreams")
    return [line for line in section.splitlines() if BULLET_RE.match(line)]


def _feature_table_rows() -> set[str]:
    """First-column texts of the spec's feature table.

    The feature table is identified structurally: the table whose header row
    contains the Feature and Status columns. Scoping to that table (not every
    markdown table) keeps the "resolves against the feature table" promise
    literal — an anchor colliding with an unrelated table's first column
    must not pass.
    """
    rows: set[str] = set()
    in_feature_table = False
    for line in SPEC.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            in_feature_table = False
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            in_feature_table = False
            continue
        if {"Feature", "Status"} <= {_norm(c) for c in cells}:
            in_feature_table = True  # header fingerprint row
            continue
        if set(cells[0]) <= {"-", ":"}:
            continue  # separator row
        if in_feature_table:
            rows.add(_norm(cells[0]))
    return rows


def _heading_slugs(text: str) -> set[str]:
    slugs: set[str] = set()
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if not m:
            continue
        title = re.sub(r"<[^>]+>", "", m.group("title"))
        slugs.add(re.sub(r"[^\w\s-]", "", title.lower()).replace(" ", "-"))
    return slugs


def test_every_workstream_declares_spec_anchor() -> None:
    bullets = _workstream_bullets()
    assert bullets, "Open Workstreams has no bullets to check"
    missing = [b for b in bullets if not WORKSTREAM_RE.match(b)]
    assert not missing, (
        "Open Workstream bullets must end with '— spec: <anchor>' naming the "
        "docs/spec.md feature row; missing on:\n" + "\n".join(missing)
    )


def test_spec_anchors_resolve() -> None:
    rows = _feature_table_rows()
    assert rows, "docs/spec.md has no feature table to resolve anchors against"
    for bullet in _workstream_bullets():
        m = WORKSTREAM_RE.match(bullet)
        if m is None:
            # No anchor declared: reported by
            # test_every_workstream_declares_spec_anchor; skip here.
            continue
        anchor = _norm(m.group("anchor"))
        title = m.group("title")
        assert anchor in rows, (
            f"TODO.md workstream {title!r} declares spec anchor {anchor!r}, but the "
            "docs/spec.md feature table has no such row. Update the '— spec:' "
            "field or the feature-table row."
        )


def test_no_stale_done_workstreams() -> None:
    section = _section(ROADMAP.read_text(encoding="utf-8"), "Open Workstreams")
    shipped = re.findall(r"✅|\bDONE\b", section)
    ticked = [b for b in _workstream_bullets() if TICKED_RE.match(b)]
    assert not shipped and not ticked, (
        "Open Workstreams contains finished items; move them to "
        "docs/archive/roadmap-archive.md. "
        f"markers={shipped} ticked={ticked}"
    )


def test_roadmap_links_resolve() -> None:
    text = ROADMAP.read_text(encoding="utf-8")
    slugs = _heading_slugs(text)
    targets = LINK_RE.findall(text)
    assert targets, "TODO.md has no links to check"
    bad: list[str] = []
    for target in targets:
        if SCHEME_RE.match(target):  # https:, mailto:, ftp:, ...
            continue
        path_part, _, fragment = target.partition("#")
        if path_part:
            candidate = REPO_ROOT / path_part.lstrip("/")
            if not candidate.exists():
                bad.append(target)
            elif fragment and candidate.suffix == ".md" and candidate.is_file():
                target_slugs = _heading_slugs(candidate.read_text(encoding="utf-8"))
                if fragment not in target_slugs:
                    bad.append(target)
        elif fragment and fragment not in slugs:
            bad.append(target)
    assert not bad, f"TODO.md links to missing targets: {bad}"


def test_operations_section_names_spec_sync() -> None:
    ops = _section(ROADMAP.read_text(encoding="utf-8"), "Operations")
    assert "docs/spec.md" in ops, (
        "Operations must carry the living 'keep in sync with docs/spec.md' obligation"
    )
