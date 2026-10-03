# Roadmap Archive — Completed Items

Historical record of shipped work, retained for traceability. Active planning now
lives in [`../TODO.md`](../TODO.md).

## v0.1 — Core CLI & Scoring (COMPLETE)

- [x] Basic CLI entry point (`agenteval-bench run --suite ...`)
      *Note: implemented as a minimal argv parser, not full argparse — see roadmap v0.3.*
- [x] `DeterministicScorer` (exact, contains, regex, json_schema)
- [x] `EvalRunner` engine orchestrating suite execution
- [x] YAML suite loading with `EvalSuite.from_yaml`
- [x] CI integration with golden replay (`examples/support-agent.yaml`)
- [x] `pytest` test suite (engine + CLI gateway)

## Operations (COMPLETE)

- [x] Scorer ground-truth calibration and regression tests
- [x] Golden replay suite added to `examples/`
- [x] `CLAUDE-OS.md` maintained as the workflow source of truth

## Earlier deliverables (pre-roadmap)

- [x] MIT LICENSE added
- [x] SOTA required files: CODEOWNERS, TODO.md, CLAUDE-OS.md
- [x] CI checkout pinned to `actions/checkout@v4`
- [x] Claude always-fresh PR code-review workflow
