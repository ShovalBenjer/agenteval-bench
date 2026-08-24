# Contributing to agenteval-bench

## Development Setup

```bash
git clone https://github.com/ShovalBenjer/agenteval-bench.git
cd agenteval-bench
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,lint,test]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Linting and Type Checking

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/agenteval_bench/ --strict
```

## Submitting Changes

1. Fork the repository and create a feature branch.
2. Make your changes, adding tests for new functionality.
3. Ensure all tests, linting, and type checks pass.
4. Submit a pull request with a clear description of the changes.
