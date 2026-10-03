# CLAUDE-OS — agenteval-bench

Status: ACTIVE (living document, the spine)
Date: 2026-07-31
Owner: agenteval-bench maintainers. Maintainer: the Claude session.
Repo: github.com/ShovalBenjer/agenteval-bench

This file is the single source of truth for the repo's OS/workflow configuration.
It supersedes all prior setup plans and informal notes for this repo.

---

## 1. Mission

`agenteval-bench` is a production-grade CLI for evaluating, grading, and comparing
LLM agent outputs with deterministic scoring, regression tracking, and CI
integration. Think `pytest` for agent outputs.

---

## 2. Project Structure

```
agenteval-bench/
├── src/agenteval_bench/
│   ├── __init__.py     # Public API surface
│   ├── models.py       # Data models (EvalSuite, EvalCase, EvalResult)
│   ├── scoring.py      # DeterministicScorer (exact, contains, regex, json_schema)
│   ├── engine.py       # EvalRunner — orchestrates suite execution
│   └── cli.py          # CLI entry point (argparse)
├── tests/              # Pytest test suite
├── examples/           # Golden replay suites (YAML)
├── docs/               # Specifications and design docs
├── .github/workflows/  # CI and code review workflows
├── pyproject.toml      # Package configuration (uv)
├── README.md           # Project documentation
├── CODEOWNERS          # Code ownership
├── TODO.md             # Task tracking
└── CLAUDE-OS.md        # This file
```

---

## 3. Development Workflow

- **Install**: `uv venv && source .venv/bin/activate && uv pip install -e ".[dev]" --system`
- **Lint**: `ruff check src/ tests/`
- **Test**: `pytest tests/ -v --tb=short`
- **Run eval**: `agenteval-bench run --suite examples/support-agent.yaml --ci --threshold 0.9`

---

## 4. CI/CD

Two GitHub Actions workflows:

1. **CI** (`.github/workflows/ci.yml`): Runs on push and pull_request.
   - Tests against Python 3.11 and 3.12
   - Lints with ruff
   - Runs pytest
   - Gates on golden replay suite pass rate ≥ threshold

2. **Claude Code Review** (`.github/workflows/claude-code-review.yml`): Runs on PR
   events (opened, synchronize, ready_for_review, reopened). Performs a fresh, full
   review using the Anthropics Claude Code action. Focuses on: correctness, security
   (OWASP/CWE), contract/boundary violations, performance, and maintainability.

Both workflows use `actions/checkout@v4`.

---

## 5. Scoring Strategies

| Strategy | Description | Deterministic |
|----------|-------------|--------------|
| `exact` | Exact string match (whitespace-trimmed) | Yes |
| `contains` | All needles present in output | Yes |
| `regex` | Regex pattern found in output | Yes |
| `json_schema` | Valid JSON with required keys | Yes |
| LLM-as-judge | Model grades output against rubric | No (planned) |

---

## 6. Conventions

- **Naming**: kebab-case for directories and files; snake_case for Python modules
- **Branch naming**: descriptive kebab-case branch names
- **Commit messages**: conventional commits (feat, fix, docs, refactor, test, chore)
- **Code style**: ruff-enforced; no comments unless asked
- **Templates**: PR and issue templates in `.github/`