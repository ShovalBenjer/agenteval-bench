#!/usr/bin/env python3
"""Paired bootstrap comparison of baseline vs candidate judge scores.

Given two files with one score per line (paired by line: the same eval case
scored for the baseline agent and the candidate agent), resample the paired
differences with replacement and report the mean difference with a bootstrap
confidence interval — the 2026 SOTA replacement for binary pass/fail gates
(single-run pass/fail cannot distinguish a real regression from noise).

Usage:
    python tools/paired_bootstrap.py baseline.txt candidate.txt [--n-boot 10000]
        [--seed 42] [--alpha 0.05] [--strict] [--json]

    # Freeze a clean baseline, then compare a candidate against it:
    python tools/paired_bootstrap.py --scores baseline.txt \
        --out baseline.json [--suite examples/support-agent.yaml]
    python tools/paired_bootstrap.py baseline.json candidate.txt

Exit codes: 0 = no statistically significant regression (improved or
inconclusive); 1 = significant regression (whole CI below 0); 2 = usage error.
With --strict, an inconclusive result (CI straddles 0) also exits 1.

Stdlib only: no numpy/scipy, so it runs anywhere the repo's CI runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time

SNAPSHOT_VERSION = 1


def read_scores(path: str) -> list[float]:
    scores = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                scores.append(float(line))
            except ValueError:
                raise SystemExit(f"error: {path}:{i}: not a number: {line!r}")
    if not scores:
        raise SystemExit(f"error: {path}: no scores found")
    return scores


def bootstrap_ci(diffs: list[float], n_boot: int, alpha: float, seed: int):
    rng = random.Random(seed)
    n = len(diffs)
    means = [statistics.fmean(rng.choices(diffs, k=n)) for _ in range(n_boot)]
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    # two-sided p: fraction of bootstrap means at least as extreme as 0
    p = 2 * min(
        sum(1 for m in means if m <= 0) / n_boot,
        sum(1 for m in means if m >= 0) / n_boot,
    )
    return lo, hi, min(p, 1.0)


def verdict_for(lo: float, hi: float) -> str:
    """Delta verdict from a bootstrap CI: significant regression, significant
    improvement, or inconclusive (the 2026 SOTA replacement for binary
    pass/fail — a straddling CI is reported, not hidden)."""
    if hi < 0:
        return "regression"
    if lo > 0:
        return "improvement"
    return "inconclusive"


def delta_report(baseline: list[float], candidate: list[float],
                 n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 42) -> dict:
    """Structured baseline-vs-candidate delta with bootstrap CI.

    Raises SystemExit(2) on unpaired/degenerate inputs, mirroring the CLI.
    """
    if len(baseline) != len(candidate):
        raise SystemExit(
            f"error: unpaired inputs: {len(baseline)} baseline vs {len(candidate)} candidate rows")
    if len(baseline) < 2:
        raise SystemExit("error: need at least 2 paired rows")
    diffs = [c - b for b, c in zip(baseline, candidate)]
    lo, hi, p = bootstrap_ci(diffs, n_boot, alpha, seed)
    return {
        "n": len(baseline),
        "seed": seed,
        "n_boot": n_boot,
        "alpha": alpha,
        "mean_diff": statistics.fmean(diffs),
        "ci": [lo, hi],
        "p_value": p,
        "deltas": diffs,
        "verdict": verdict_for(lo, hi),
    }


def write_snapshot(path: str, scores: list[float], meta: dict) -> None:
    """Write a clean-baseline snapshot: scores plus the metadata needed to
    trust them later (seed, suite digest, scorer identity, timestamp)."""
    doc = {
        "tool": "paired_bootstrap",
        "snapshot_version": SNAPSHOT_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "meta": meta,
        "scores": scores,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, sort_keys=True, indent=2) + "\n")


def read_snapshot(path: str) -> tuple[list[float], dict, list[str] | None]:
    """Read a baseline snapshot; returns (scores, meta, case_ids-or-None).

    Accepts both plain score lists and per-case dicts
    (``{"case_id": ..., "score": ...}``), as written by
    ``agenteval-bench baseline``.
    """
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("snapshot_version") != SNAPSHOT_VERSION:
        raise SystemExit(f"error: {path}: unsupported snapshot_version "
                         f"{doc.get('snapshot_version')}")
    raw = doc["scores"]
    case_ids: list[str] | None = None
    if raw and isinstance(raw[0], dict):
        case_ids = [str(r["case_id"]) for r in raw]
        scores = [float(r["score"]) for r in raw]
    else:
        scores = [float(s) for s in raw]
    if not scores:
        raise SystemExit(f"error: {path}: snapshot has no scores")
    meta = dict(doc.get("meta", {}))
    # `agenteval-bench baseline` writes provenance at top level; surface it too.
    for k in ("seed", "suite_name", "suite_file", "suite_digest", "scorer", "created"):
        if k in doc and k not in meta:
            meta[k] = doc[k]
    return scores, meta, case_ids


def suite_digest_of(path: str) -> str:
    with open(path, "rb") as f:
        return "sha256:" + hashlib.sha256(f.read()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", nargs="?",
                    help="baseline scores file, or baseline snapshot (.json)")
    ap.add_argument("candidate", nargs="?",
                    help="candidate scores file (one score per line)")
    ap.add_argument("--scores", help="score file to snapshot (snapshot mode)")
    ap.add_argument("--out", help="snapshot output path (snapshot mode)")
    ap.add_argument("--suite", help="suite YAML: its sha256 is pinned in the snapshot")
    ap.add_argument("--json", action="store_true",
                    help="print the delta report as JSON (compare mode)")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--strict", action="store_true",
                    help="fail on inconclusive results too, not just regressions")
    args = ap.parse_args()

    if args.scores:
        # Snapshot mode: freeze a clean baseline.
        if not args.scores or not args.out:
            raise SystemExit("error: snapshot mode needs --scores FILE --out baseline.json")
        scores = read_scores(args.scores)
        meta = {"seed": args.seed, "n_boot": args.n_boot, "alpha": args.alpha}
        if args.suite:
            meta["suite"] = args.suite
            meta["suite_digest"] = suite_digest_of(args.suite)
        write_snapshot(args.out, scores, meta)
        print(f"snapshot: {len(scores)} scores -> {args.out}")
        return 0

    if not args.baseline or not args.candidate:
        raise SystemExit("error: need BASELINE CANDIDATE (or 'snapshot' mode)")

    # Baseline and candidate may each be a .json snapshot or a plain score file.
    # Pairing is by line order; when both sides carry case_ids they must match.
    def _load_side(path: str):
        if path.endswith(".json"):
            return read_snapshot(path)
        return read_scores(path), {}, None

    base, meta, base_ids = _load_side(args.baseline)
    cand, _cand_meta, cand_ids = _load_side(args.candidate)
    if base_ids and cand_ids and base_ids != cand_ids:
        raise SystemExit("error: case_id order differs between baseline and candidate snapshots")

    report = delta_report(base, cand, args.n_boot, args.alpha, args.seed)
    report["baseline_meta"] = meta

    if args.json:
        print(json.dumps(report, sort_keys=True, indent=2))
    else:
        lo, hi = report["ci"]
        pct = 100 * (1 - args.alpha)
        print(f"n={report['n']}  mean_diff={report['mean_diff']:+.4f}  "
              f"{pct:.0f}% CI=[{lo:+.4f}, {hi:+.4f}]  p~{report['p_value']:.4f}  "
              f"-> {report['verdict'].upper()}")

    if report["verdict"] == "regression":
        return 1
    if report["verdict"] == "inconclusive" and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
