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
        # Standalone CLI scores a replay suite: each case carries a recorded
        # `output`. Cases without one are skipped (a live agent would fill them
        # via the Python API). This is what CI runs against a golden set.
        recorded = {c.id: c.output for c in suite.cases}
        missing = [c.id for c in suite.cases if c.output is None and not c.skip]
        for c in suite.cases:
            if c.output is None and not c.skip:
                c.skip = True

        # run() calls agent_fn only for non-skipped cases, in case order.
        replay_ids = iter([c.id for c in suite.cases if not c.skip])

        def replay_fn(_input: str) -> str:
            return recorded.get(next(replay_ids)) or ""

        runner = EvalRunner()
        result = runner.run(suite, replay_fn)
        print(result.summary())
        if missing:
            print(f"Note: {len(missing)} case(s) had no recorded output and were skipped.")

        if ci_mode:
            ok = result.pass_rate >= threshold
            print(
                f"CI gate: pass_rate {result.pass_rate:.1%} vs threshold {threshold:.0%} -> {'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                sys.exit(2)
    else:
        print(f"Unknown command: {args[0]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
