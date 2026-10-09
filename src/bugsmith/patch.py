"""Minimal unified-diff applier (agenteval-bench#37).

Applies the single- and multi-file unified diffs emitted by
:mod:`bugsmith.buggen` to a clean target tree. Pure Python: the Docker
validation image does not need a ``patch`` binary, and local runs do not
either. Context lines are matched strictly — a patch that does not apply
cleanly raises :class:`PatchError` instead of producing a half-applied
tree.
"""

from __future__ import annotations

import re
from pathlib import Path

from bugsmith.types import BugsmithError

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchError(BugsmithError):
    """A patch could not be applied cleanly."""


def _strip_prefix(path: str) -> str:
    # difflib emits "a/<rel>" / "b/<rel>"; also tolerate bare paths.
    for prefix in ("a/", "b/"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def _check_safe(rel: str) -> None:
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise PatchError(f"patch escapes the repo root: {rel!r}")


def split_files(patch: str) -> dict[str, list[str]]:
    """Split a unified diff into {repo-relative path: hunk lines}."""
    files: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in patch.splitlines():
        if line.startswith("--- "):
            rel = _strip_prefix(line[4:].split("\t")[0].strip())
            _check_safe(rel)
            current = files.setdefault(rel, [])
        elif line.startswith("+++ "):
            continue  # path already taken from the --- line
        elif current is not None:
            current.append(line)
        elif line.strip():
            raise PatchError(f"patch line outside a file header: {line[:60]!r}")
    if not files:
        raise PatchError("patch contains no file headers")
    return files


def apply_hunks(original: list[str], hunk_lines: list[str]) -> list[str]:
    """Apply hunk lines to ``original`` (list of lines, newlines kept)."""
    out: list[str] = []
    orig_idx = 0  # 0-based cursor into original
    i = 0
    no_trailing_newline = False
    while i < len(hunk_lines):
        m = _HUNK_RE.match(hunk_lines[i])
        if not m:
            raise PatchError(f"expected hunk header, got: {hunk_lines[i][:60]!r}")
        old_start = int(m.group(1)) - 1
        i += 1
        # Copy unchanged lines before this hunk.
        if old_start < orig_idx:
            raise PatchError("overlapping hunks")
        out.extend(original[orig_idx:old_start])
        orig_idx = old_start
        while i < len(hunk_lines) and not hunk_lines[i].startswith("@@"):
            line = hunk_lines[i]
            i += 1
            if line.startswith(" "):
                if orig_idx >= len(original):
                    raise PatchError("context line past end of file")
                if original[orig_idx].rstrip("\n") != line[1:]:
                    raise PatchError(
                        f"context mismatch at line {orig_idx + 1}: "
                        f"{original[orig_idx].rstrip()!r} != {line[1:]!r}"
                    )
                out.append(original[orig_idx])
                orig_idx += 1
            elif line.startswith("-"):
                if orig_idx >= len(original):
                    raise PatchError("deletion past end of file")
                if original[orig_idx].rstrip("\n") != line[1:]:
                    raise PatchError(
                        f"deletion mismatch at line {orig_idx + 1}"
                    )
                orig_idx += 1
            elif line.startswith("+"):
                out.append(line[1:] + "\n")
                no_trailing_newline = False
            elif line.startswith("\\"):
                # "\ No newline at end of file": the previous added line
                # keeps no trailing newline instead of gaining one.
                no_trailing_newline = True
                continue
            elif line == "":
                continue
            else:
                raise PatchError(f"bad hunk line: {line[:60]!r}")
    out.extend(original[orig_idx:])
    if no_trailing_newline and out and out[-1].endswith("\n"):
        out[-1] = out[-1][:-1]
    return out


def apply_patch(repo_root: Path, patch: str) -> dict[str, str]:
    """Apply ``patch`` to files under ``repo_root``.

    Returns {repo-relative path: new content}. Writes go through the
    caller: this function mutates the tree on disk. Paths are confined to
    ``repo_root`` (``..`` and absolute paths refused).
    """
    repo_root = repo_root.resolve()
    applied: dict[str, str] = {}
    for rel, hunk_lines in split_files(patch).items():
        target = (repo_root / rel).resolve()
        try:
            target.relative_to(repo_root)
        except ValueError:
            raise PatchError(f"patch escapes the repo root: {rel!r}")
        if not target.is_file():
            raise PatchError(f"patch target not found: {rel!r}")
        original = target.read_text(encoding="utf-8").splitlines(keepends=True)
        new_lines = apply_hunks(original, hunk_lines)
        new_text = "".join(new_lines)
        target.write_text(new_text, encoding="utf-8")
        applied[rel] = new_text
    return applied
