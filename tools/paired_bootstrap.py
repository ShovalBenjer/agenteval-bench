#!/usr/bin/env python3
"""Paired bootstrap comparison of baseline vs candidate judge scores.

Given two files with one score per line (paired by line: the same eval case
scored for the baseline agent and the candidate agent), resample the paired
differences with replacement and report the mean difference with a bootstrap
confidence interval — the 2026 SOTA replacement for binary pass/fail gates
(single-run pass/fail cannot distinguish a real regression from noise).

Usage:
    python tools/paired_bootstrap.py baseline.txt candidate.txt [--n-boot 10000]
        [--seed 42] [--alpha 0.05] [--strict]

Exit codes: 0 = no statistically significant regression (improved or
inconclusive); 1 = significant regression (whole CI below 0); 2 = usage error.
With --strict, an inconclusive result (CI straddles 0) also exits 1.

Stdlib only: no numpy/scipy, so it runs anywhere the repo's CI runs.
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--strict", action="store_true",
                    help="fail on inconclusive results too, not just regressions")
    args = ap.parse_args()

    base = read_scores(args.baseline)
    cand = read_scores(args.candidate)
    if len(base) != len(cand):
        raise SystemExit(
            f"error: unpaired inputs: {len(base)} baseline vs {len(cand)} candidate rows")
    if len(base) < 2:
        raise SystemExit("error: need at least 2 paired rows")

    diffs = [c - b for b, c in zip(base, cand)]
    mean_diff = statistics.fmean(diffs)
    lo, hi, p = bootstrap_ci(diffs, args.n_boot, args.alpha, args.seed)
    pct = 100 * (1 - args.alpha)

    if hi < 0:
        verdict, code = "REGRESSION", 1
    elif lo > 0:
        verdict, code = "IMPROVED", 0
    else:
        verdict, code = ("INCONCLUSIVE", 1 if args.strict else 0)

    print(f"n={len(base)}  mean_diff={mean_diff:+.4f}  "
          f"{pct:.0f}% CI=[{lo:+.4f}, {hi:+.4f}]  p~{p:.4f}  -> {verdict}")
    return code


if __name__ == "__main__":
    sys.exit(main())
