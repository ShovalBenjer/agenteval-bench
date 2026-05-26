"""CLI entry point for agenteval-bench."""

from __future__ import annotations

import sys

from agenteval_bench.engine import EvalRunner
from agenteval_bench.models import EvalSuite


def main() -> None:
    """Minimal CLI — full argparse/typer in v0.2."""
    args = sys.argv[1:]

    if not args or args[0] in ("--help", "-h"):
        print("agenteval-bench — LLM agent evaluation CLI")
        print("Usage: agenteval-bench run --suite <file>")
        print("       agenteval-bench compare <run_a> <run_b>")
        return

    if args[0] == "run":
        suite_path = None
        ci_mode = False
        threshold = 1.0

        i = 1
        while i < len(args):
            if args[i] == "--suite" and i + 1 < len(args):
                suite_path = args[i + 1]
                i += 2
            elif args[i] == "--ci":
                ci_mode = True
                i += 1
            elif args[i] == "--threshold" and i + 1 < len(args):
                threshold = float(args[i + 1])
                i += 2
            else:
                i += 1

        if not suite_path:
            print("Error: --suite <file> is required", file=sys.stderr)
            sys.exit(1)

        suite = EvalSuite.from_yaml(suite_path)
        # In standalone CLI mode, we can't call an agent function.
        # This is a placeholder — real usage goes through the Python API
        # or pipes agent output via stdin.
        print(f"Suite: {suite.name} ({len(suite.cases)} cases loaded)")
        if ci_mode:
            print(f"CI mode: threshold={threshold:.0%}")
    else:
        print(f"Unknown command: {args[0]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
