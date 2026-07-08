"""CLI entry point for agenteval-bench."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import sys

from agenteval_bench.compare import DEFAULT_REGRESSION_THRESHOLD, compare_runs, load_run
from agenteval_bench.engine import AgentFn, EvalRunner
from agenteval_bench.models import EvalSuite


def _load_agent(spec: str) -> AgentFn:
    """Load an agent function from a ``module.path:function`` spec.

    The current working directory is added to ``sys.path`` so agents
    defined in local project files resolve without installation.
    """
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"invalid agent spec {spec!r}: expected 'module.path:function'")

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    module = importlib.import_module(module_name)
    try:
        agent_fn = getattr(module, attr)
    except AttributeError:
        raise AttributeError(f"module {module_name!r} has no attribute {attr!r}") from None
    if not callable(agent_fn):
        raise TypeError(f"{spec!r} is not callable")
    return agent_fn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agenteval-bench",
        description="LLM agent evaluation CLI — think pytest for agent outputs.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run an eval suite against an agent")
    run_parser.add_argument("--suite", required=True, help="Path to the eval suite YAML file")
    run_parser.add_argument(
        "--agent",
        help="Agent function as 'module.path:function' (omit to only validate the suite)",
    )
    run_parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: exit non-zero when pass rate is below --threshold",
    )
    run_parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Minimum pass rate (0.0-1.0) required in --ci mode (default: 1.0)",
    )
    run_parser.add_argument("--output", help="Write run results as JSON to this file")

    compare_parser = subparsers.add_parser(
        "compare", help="Compare two runs and flag regressions"
    )
    compare_parser.add_argument("baseline", help="Baseline run JSON (from `run --output`)")
    compare_parser.add_argument("candidate", help="Candidate run JSON (from `run --output`)")
    compare_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_REGRESSION_THRESHOLD,
        help="Max allowed pass-rate drop before flagging a regression (default: 0.05)",
    )

    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    suite = EvalSuite.from_yaml(args.suite)

    if args.agent is None:
        # Validate-only mode: load the suite and report what it contains.
        print(f"Suite: {suite.name} ({len(suite.cases)} cases loaded)")
        if args.ci:
            print("Error: --ci requires --agent to execute the suite", file=sys.stderr)
            return 2
        return 0

    try:
        agent_fn = _load_agent(args.agent)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        print(f"Error loading agent: {exc}", file=sys.stderr)
        return 2

    runner = EvalRunner()
    if args.ci:
        result = runner.run_ci(suite, agent_fn, threshold=args.threshold)
    else:
        result = runner.run(suite, agent_fn)

    print(result.summary())

    if args.output:
        with open(args.output, "w") as f:
            json.dump(dataclasses.asdict(result), f, indent=2)
        print(f"Results written to {args.output}")

    if args.ci and not result.threshold_met:
        print(
            f"CI: pass rate {result.pass_rate:.1%} below threshold {args.threshold:.1%}",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    try:
        baseline = load_run(args.baseline)
        candidate = load_run(args.candidate)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Error loading run: {exc}", file=sys.stderr)
        return 2

    comparison = compare_runs(baseline, candidate, threshold=args.threshold)
    print(comparison.summary())
    return 1 if comparison.is_regression else 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    if args.command == "run":
        exit_code = _cmd_run(args)
    elif args.command == "compare":
        exit_code = _cmd_compare(args)
    else:
        exit_code = 0
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
