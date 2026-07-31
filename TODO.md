# TODO — agenteval-bench

Task tracking for the agenteval-bench project. One TODO per tracked item.
Status reflects current progress.

## Roadmap

### v0.1 — Core CLI & Scoring
- [x] Basic CLI with argparse (`agenteval-bench run`)
- [x] DeterministicScorer (exact, contains, regex, json_schema)
- [x] EvalRunner engine
- [x] YAML suite loading
- [x] CI integration with golden replay
- [x] pytest test suite

### v0.2 — LLM-as-Judge & Regression
- [ ] LLM-as-judge scoring with configurable judge model
- [ ] `agenteval-bench compare` for run-over-run regression diffs
- [ ] Report generation (markdown + JSON)

### v0.3 — Rich CLI & Extensibility
- [ ] typer-based CLI with rich output
- [ ] Plugin system for custom scorers
- [ ] Dashboard for historical eval results

## Operations

- [ ] Ensure all scorers have ground-truth calibration and regression tests
- [ ] Add more golden replay suites to `examples/`
- [ ] Update CLAUDE-OS.md when workflow or OS config changes