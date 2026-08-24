# Contributing

Thank you for your interest in contributing to agenteval-bench!

## Development Setup

1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com/shovalbenjer/agenteval-bench.git
   cd agenteval-bench
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv sync
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Branch Naming

Feature branches should follow the pattern: `gt/<agent>/<bead-id>`

Example: `gt/maple/51caa06d`

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new features
- `fix:` — bug fixes
- `docs:` — documentation changes
- `test:` — test additions or changes
- `refactor:` — code refactoring
- `ci:` — CI changes

Format: `<type>(<scope>): <description>`

## Pull Request Process

1. Ensure all tests pass: `make test`
2. Ensure lint passes: `make lint`
3. Ensure typecheck passes: `make typecheck`
4. Update CHANGELOG.md with your changes
5. Open a PR against the `main` branch

## Testing Requirements

- All new features must include tests
- All tests must pass before a PR can be merged
- Target coverage: 80%+
