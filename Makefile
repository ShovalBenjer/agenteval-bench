.PHONY: install test lint typecheck format check

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/

typecheck:
	uv run mypy src/agenteval_bench/ --strict

format:
	uv run ruff format src/ tests/

check: lint typecheck test
