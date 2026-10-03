# Delta-baseline scoring + deterministic seeded replay

Two mechanics that make the §1 "baseline vs candidate" contract (§1 of
`EVAL-RIGOR.md`) executable instead of aspirational.

## 1. Delta-baseline scoring

Score a candidate against a **frozen clean baseline** instead of a fixed
absolute threshold (a harder suite fails a better agent; an absolute
threshold cannot tell regression from noise).

```bash
# Freeze the baseline once (per-case scores pinned to seed + suite digest):
agenteval-bench baseline --suite examples/support-agent.yaml \
    --seed 42 --out baselines/golden.json

# Later, score a candidate run against it (paired bootstrap, §1):
python tools/paired_bootstrap.py baselines/golden.json candidate_scores.txt
# n=40  mean_diff=-0.0210  95% CI=[-0.0388, -0.0031]  p~0.0220  -> REGRESSION
```

`tools/paired_bootstrap.py` also snapshots a baseline directly:

```bash
python tools/paired_bootstrap.py --scores baseline.txt --out baseline.json \
    --suite examples/support-agent.yaml   # pins suite sha256 in the snapshot
python tools/paired_bootstrap.py baseline.json candidate.txt --json  # machine-readable
```

Verdicts: `improvement` (whole CI above 0), `regression` (whole CI below
0), `inconclusive` (CI straddles 0). Exit 1 on regression; `--strict` also
fails on inconclusive. Compare-mode is backward compatible with the plain
`baseline.txt candidate.txt` form from before.

## 2. Deterministic seeded replay

Every `run` binds a `random.Random(seed)` stream (default 42) and records
one draw per case in the run's replay log. Re-running the same suite with
the same seed must produce **byte-identical** log output:

```bash
agenteval-bench run --suite examples/support-agent.yaml \
    --seed 42 --replay-log runs/replay.json
agenteval-bench replay --log runs/replay.json \
    --suite examples/support-agent.yaml --check
# Replay bit-exact: 4 cases, seed=42, digest=sha256:9f2c...
```

The log pins: seed, suite sha256 digest, scorer identity, per-case input /
output / score / passed / rng_draw. `--check` re-runs and fails (exit 1) on
any mismatch — tampered logs, a changed suite (digest guard), or a different
seed are all detected.

The RNG stream exists so that any *future* stochastic component (subset
sampling, tie-breaking) draws from a pinned stream; the draw log proves
which stream a run used.

## Falsification

Delete the replay log machinery if no stochastic component exists after
~6 months (the digest + per-case records alone then suffice); delete
snapshotting if baselines are never compared against in practice.
