# REVIEW.md — judging policy for agenteval-bench

What to reject here, and how. For how to build here, see README.md,
docs/spec.md, and AGENTS.md when present. Kilo Code reads this file from the
repository root on the PR base branch (truncated at 10,000 characters), so it
is committed to main and branch-local copies do not count.

## 1. Severity calibration

**Blocking (request changes):**
- Anything that weakens an oracle or a gate: a change to
  `src/agenteval_bench/scoring.py`, `src/agenteval_bench/engine.py`, or CI
  thresholds (`--threshold`, workflow YAML) that makes a failing eval pass, or
  a passing eval harder to fail, without justification and without an updated
  golden suite. Oracle weakening is blocking regardless of diff size.
- New or changed LLM-judge logic without a baseline-vs-candidate comparison
  (confidence intervals, not a single run). Binary pass/fail on one run is not
  evidence of improvement — see docs/EVAL-RIGOR.md.
- Secrets or credentials anywhere in the diff. Append-only ledgers rewritten
  instead of appended to.

**Non-blocking (nit):**
- Docstring/README wording, YAML comment style.
- Cosmetic refactors with no behavior change and green CI.
- Tests asserting behavior the diff does not touch (fine, but say so).

Uncertain between blocking and nit? Blocking only if merging lets a regressed
agent ship. Everything else is a nit.

## 2. Paths to skip

Do not leave style or nit comments on:
- `docs/archive/` — frozen history.
- `examples/*.yaml` golden suites — review case semantics (does the case test
  what its id claims?), not the prose style of recorded outputs.
- Append-only JSONL ledgers (`state/*.jsonl` where present) — check they were
  appended to, never rewritten; do not nit individual rows.
- Dependency churn (`uv.lock`), auto-generated workflow YAML, and any vendored
  or generated files — flag only if they change behavior.

## 3. Verification expected before approving

1. CI green is necessary, not sufficient: the Eval gate (golden replay suite)
   must pass, because lint+unit steps do not catch a weakened scorer.
2. For scoring/engine changes: confirm no previously-failing case now passes
   and no pass threshold moved without a spec change.
3. For new evaluators/metrics: require baseline-vs-candidate evidence.
4. Do not approve a PR whose description lacks a Test Plan, or one that moves a
   gate threshold without naming the concrete failure it prevents.

## 4. Reviewer budget

- <100 lines: one pass, no sub-agents.
- 100–500 lines: at most one focused sub-agent, on the highest-risk file.
- >500 lines: require the author to split, or review scoring/engine changes
  only and defer the rest. Never rubber-stamp.

## 5. Summary style

Lead with the verdict (approve / request changes / comment), then blocking
findings with file+line, then nits grouped by file. Keep it under ~300 words
unless a blocking finding needs more.
