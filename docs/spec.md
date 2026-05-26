# agenteval-bench — Specification

## Problem

Teams deploying LLM agents have no systematic way to answer: "is my agent getting better or worse?" Evaluation is fragmented — LangSmith traces, manual notebooks, ad-hoc scripts. There is no unified CLI that treats agent evaluation like a test suite.

## Solution

`agenteval-bench` is a production-grade CLI that lets teams define eval suites in YAML, run deterministic + LLM-judge scoring, detect regressions across runs, and integrate into CI pipelines. Think `pytest` for agent outputs.

## Scope

### In scope (MVP)
- YAML-defined eval suites (task, input, expected output schema, grading rubric)
- Deterministic scoring (exact match, contains, regex, JSON schema validation)
- LLM-as-judge scoring with configurable judge model
- Regression detection (compare runs, flag degradation)
- CI mode: exit non-zero on regression
- Report generation (markdown + JSON)
- Cost bounding (max tokens per eval row)

### Out of scope (v2+)
- Web UI / dashboard
- Distributed eval execution
- Custom scorer plugins (beyond built-in)
- Agent trace replay

## API Surface

### CLI
```
agenteval-bench run [--suite FILE] [--ci] [--output DIR]
agenteval-bench compare RUN_A RUN_B [--threshold FLOAT]
agenteval-bench report RUN_ID [--format markdown|json]
```

### Eval Suite YAML
```yaml
name: my-agent-eval
version: 1
cost_bound:
  max_input_tokens: 1500
  max_output_tokens: 120
cases:
  - id: greeting-test
    input: "Hello, what can you do?"
    expected:
      contains: ["help", "assist"]
    rubric:
      - criterion: "politeness"
        weight: 0.5
        description: "Response is courteous"
      - criterion: "accuracy"
        weight: 0.5
        description: "Response correctly describes capabilities"
```

### Python API
```python
from agenteval_bench import EvalSuite, EvalRunner

suite = EvalSuite.from_yaml("eval.yaml")
runner = EvalRunner()
result = runner.run(suite, agent_fn=my_agent)
print(result.summary())
```

## Failure Modes (PREMORTEM)

1. **Flaky LLM-judge scoring**: LLM judges return inconsistent grades across runs. *Mitigation*: support deterministic scorers as primary, LLM-judge as opt-in; require N=3 majority vote for judge scores; pin judge temperature to 0.

2. **Cost overrun on large eval suites**: Running 100+ cases with LLM-judge calls burns tokens fast. *Mitigation*: cost_bound config (max_input_tokens, max_output_tokens) enforced per-row; dry-run mode estimates cost before execution; CI mode fails fast on budget exceed.

3. **YAML schema drift**: Users write eval YAML that doesn't match expected schema, causing silent mis-scoring. *Mitigation*: strict schema validation on load with clear error messages; versioned schema; reject unknown keys.

4. **Regression false positives**: Normal variance in agent outputs triggers spurious regression alerts, eroding trust. *Mitigation*: configurable threshold (default 5% degradation); statistical comparison (not just run-vs-run); skip flag for known-flaky cases.

5. **Agent function coupling**: The `agent_fn` interface is too narrow for real agents (needs tools, context, streaming). *Mitigation*: accept `Callable[[str], str]` as minimum viable interface; support `Callable[[EvalInput], EvalOutput]` for advanced use; document adapter patterns for popular frameworks.

## Acceptance Criteria

- [x] Core eval engine loads YAML suites and scores outputs deterministically
- [ ] LLM-judge integration with cost bounding
- [ ] CI mode with non-zero exit on regression
- [ ] Compare command for run-vs-run regression detection
- [ ] README accurate and AEO-optimized
- [ ] CI green on GitHub Actions
