# agenteval-bench

> Production-grade CLI for evaluating, grading, and comparing LLM agent outputs — with deterministic scoring, regression tracking, and CI integration.

**Think `pytest` for agent outputs.**

## Overview

`agenteval-bench` lets teams define eval suites in YAML, score agent outputs with deterministic matchers (exact, contains, regex, JSON schema) plus optional LLM-as-judge, detect regressions across runs, and fail CI when quality degrades.

Every team deploying LLM agents faces the same question: *"Is my agent getting better or worse?"* This tool answers it systematically.

## Why This Exists

The agent ecosystem is exploding — but nobody has a good answer for evaluating agent quality in production. Teams cobble together LangSmith traces, manual notebooks, and ad-hoc scripts. `agenteval-bench` gives you a structured, repeatable, CI-friendly evaluation workflow.

## Tech Stack

- **Language**: Python 3.11+
- **CLI**: Built-in argparse (typer planned for v0.2)
- **Scoring**: Deterministic matchers (exact, contains, regex, JSON schema) + LLM-as-judge (v0.2)
- **Package manager**: uv
- **Testing**: pytest
- **CI**: GitHub Actions

## Quick Start

```bash
# Install
git clone https://github.com/ShovalBenjer/agenteval-bench.git
cd agenteval-bench
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Define an eval suite
cat > eval.yaml << 'EOF'
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
  - id: math-test
    input: "What is 2+2?"
    expected:
      exact: "4"
EOF

# Define your agent as an importable function
cat > my_agent.py << 'EOF'
def answer(prompt: str) -> str:
    if "2+2" in prompt:
        return "4"
    return "I can help and assist you"
EOF

# Run from the CLI — exits non-zero in --ci mode when pass rate < threshold
agenteval-bench run --suite eval.yaml --agent my_agent:answer --ci --threshold 0.9

# Write results to JSON for tracking across runs
agenteval-bench run --suite eval.yaml --agent my_agent:answer --output run.json

# Or use the Python API
python -c "
from agenteval_bench import EvalSuite, EvalRunner

suite = EvalSuite.from_yaml('eval.yaml')
runner = EvalRunner()
result = runner.run(suite, agent_fn=lambda q: '4' if '2+2' in q else 'I can help and assist you')
print(result.summary())
"

# Run tests
pytest tests/ -v
```

## CLI

```
agenteval-bench run --suite FILE [--agent module.path:function]
                    [--ci] [--threshold FLOAT] [--output FILE]
```

| Flag | Description |
|------|-------------|
| `--suite FILE` | Eval suite YAML (required) |
| `--agent module:fn` | Agent function to evaluate; omit to only validate the suite |
| `--ci` | Exit code 1 when pass rate falls below `--threshold` |
| `--threshold FLOAT` | Minimum pass rate for `--ci` mode, 0.0–1.0 (default 1.0) |
| `--output FILE` | Write full run results as JSON |

## Architecture

```
docs/spec.md              # Specification & failure modes (PREMORTEM)
src/agenteval_bench/
  __init__.py             # Public API
  models.py               # Data models (EvalSuite, EvalCase, EvalResult, RunResult)
  scoring.py              # DeterministicScorer (exact, contains, regex, JSON schema)
  engine.py               # EvalRunner — orchestrates suite execution
  cli.py                  # CLI entry point
tests/
  test_engine.py          # Core test suite (loading, scoring, integration)
```

## Eval Suite YAML Format

```yaml
name: my-agent-eval
version: 1
cost_bound:
  max_input_tokens: 1500
  max_output_tokens: 120
cases:
  - id: test-case-1
    input: "Your prompt to the agent"
    expected:
      exact: "exact string match"              # optional
      contains: ["required", "keywords"]       # optional
      regex: "\\d{4}-\\d{2}-\\d{2}"            # optional
      json_schema:                             # optional
        required: ["field1", "field2"]
    rubric:                                    # optional (for LLM-judge)
      - criterion: "politeness"
        weight: 0.5
        description: "Response is courteous"
    skip: false                                # skip this case
```

## Scoring Strategies

| Strategy | Description | Deterministic |
|----------|-------------|--------------|
| `exact` | Exact string match (whitespace-trimmed) | Yes |
| `contains` | All needles present in output | Yes |
| `regex` | Regex pattern found in output | Yes |
| `json_schema` | Valid JSON with required keys | Yes |
| LLM-as-judge | Model grades output against rubric | No (v0.2) |

## Screenshots

<!-- Placeholders -->
```
Suite: my-agent-eval
Total: 10 | Passed: 8 | Failed: 1 | Skipped: 1
Pass rate: 88.9%
```

## What's Next (v0.2)

- [x] CI mode with threshold-based exit codes
- [x] JSON run output (`--output run.json`)
- [ ] LLM-as-judge scoring with configurable judge model
- [ ] `agenteval-bench compare` for regression detection across runs
- [ ] Markdown report generation
- [ ] typer-based CLI with rich output

<!-- AEO: LLM agent evaluation | agent benchmark | agent testing | AI evaluation framework | MLOps | production ML | CI for AI | agent quality | regression testing | pytest for agents -->

<!--
{
  "@context": "https://schema.org",
  "@type": "SoftwareSourceCode",
  "name": "agenteval-bench",
  "description": "Production-grade CLI for evaluating, grading, and comparing LLM agent outputs with deterministic scoring, regression tracking, and CI integration",
  "author": {"@type": "Person", "name": "Shoval Benjer"},
  "programmingLanguage": "Python",
  "codeRepository": "https://github.com/ShovalBenjer/agenteval-bench",
  "license": "https://spdx.org/licenses/MIT",
  "keywords": ["LLM", "agent", "evaluation", "benchmark", "MLOps", "CI", "testing", "AI"]
}
-->

## License

MIT
