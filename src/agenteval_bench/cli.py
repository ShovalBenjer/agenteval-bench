"""CLI entry point for agenteval-bench."""

from __future__ import annotations

import json
import os
import sys

from agenteval_bench.engine import DEFAULT_SEED, EvalRunner
from agenteval_bench.models import EvalSuite
from agenteval_bench.replay import ReplayCase, ReplayLog, find_mismatches, suite_digest


def _flag(args: list[str], name: str, default: str | None = None) -> tuple[list[str], str | None]:
    """Pop ``--name value`` from args, returning (rest, value)."""
    rest: list[str] = []
    value = default
    i = 0
    while i < len(args):
        if args[i] == name and i + 1 < len(args):
            value = args[i + 1]
            i += 2
        else:
            rest.append(args[i])
            i += 1
    return rest, value


def _has(args: list[str], name: str) -> tuple[list[str], bool]:
    rest = [a for a in args if a != name]
    return rest, len(rest) != len(args)


def _load_suite_for_replay(suite_path: str):
    suite = EvalSuite.from_yaml(suite_path)
    with open(suite_path, "rb") as f:
        digest = suite_digest(f.read())
    recorded = {c.id: c.output for c in suite.cases}
    missing = [c.id for c in suite.cases if c.output is None and not c.skip]
    for c in suite.cases:
        if c.output is None and not c.skip:
            c.skip = True
    return suite, recorded, missing, digest


def _run_suite(suite: EvalSuite, recorded: dict[str, str], seed: int):
    """Run a replay suite (recorded outputs only) and return (result, outputs)."""
    replay_ids = iter([c.id for c in suite.cases if not c.skip])
    outputs: dict[str, str] = {}

    def replay_fn(_input: str) -> str:
        cid = next(replay_ids)
        out = recorded.get(cid) or ""
        outputs[cid] = out
        return out

    runner = EvalRunner()
    return runner.run(suite, replay_fn, seed=seed), outputs


def _build_log(suite: EvalSuite, suite_path: str, digest: str,
               result, outputs: dict[str, str], seed: int) -> ReplayLog:
    by_id = {r.case_id: r for r in result.results}
    cases = []
    for c in suite.cases:
        r = by_id[c.id]
        cases.append(ReplayCase(
            case_id=c.id,
            input=c.input,
            output=outputs.get(c.id, "") if not r.details.get("skipped") else "",
            score=r.score,
            passed=r.passed,
            skipped=bool(r.details.get("skipped")),
            rng_draw=r.details.get("rng_draw"),
        ))
    return ReplayLog(
        seed=seed,
        suite_name=suite.name,
        suite_file=os.path.basename(suite_path),
        digest=digest,
        cases=cases,
    )


def cmd_run(args: list[str]) -> int:
    args, suite_path = _flag(args, "--suite")
    args, ci_mode = _has(args, "--ci")
    args, threshold_s = _flag(args, "--threshold", "1.0")
    args, seed_s = _flag(args, "--seed", str(DEFAULT_SEED))
    args, replay_log = _flag(args, "--replay-log")

    if not suite_path:
        print("Error: --suite <file> is required", file=sys.stderr)
        return 1

    try:
        threshold = float(threshold_s or "1.0")
    except ValueError:
        print(f"Error: --threshold must be a number, got {threshold_s}", file=sys.stderr)
        return 1
    if not 0.0 <= threshold <= 1.0:
        print(f"Error: --threshold must be between 0.0 and 1.0, got {threshold}", file=sys.stderr)
        return 1
    try:
        seed = int(seed_s or str(DEFAULT_SEED))
    except ValueError:
        print(f"Error: --seed must be an integer, got {seed_s}", file=sys.stderr)
        return 1

    suite, recorded, missing, digest = _load_suite_for_replay(suite_path)
    # Standalone CLI scores a replay suite: each case carries a recorded
    # `output`. Cases without one are skipped (a live agent would fill them
    # via the Python API). This is what CI runs against a golden set.
    result, outputs = _run_suite(suite, recorded, seed)
    print(result.summary())
    if missing:
        print(f"Note: {len(missing)} case(s) had no recorded output and were skipped.")

    if replay_log:
        log = _build_log(suite, suite_path, digest, result, outputs, seed)
        log.write(replay_log)
        print(f"Replay log: {replay_log} (seed={seed})")

    if ci_mode:
        ok = result.pass_rate >= threshold
        print(f"CI gate: pass_rate {result.pass_rate:.1%} vs threshold {threshold:.0%} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            return 2
    return 0


def cmd_baseline(args: list[str]) -> int:
    """Freeze a clean-baseline snapshot: per-case scores pinned to seed + suite digest."""
    args, suite_path = _flag(args, "--suite")
    args, seed_s = _flag(args, "--seed", str(DEFAULT_SEED))
    args, out = _flag(args, "--out")
    if not suite_path or not out:
        print("Error: baseline needs --suite <file> --out baseline.json", file=sys.stderr)
        return 1
    try:
        seed = int(seed_s or str(DEFAULT_SEED))
    except ValueError:
        print(f"Error: --seed must be an integer, got {seed_s}", file=sys.stderr)
        return 1

    suite, recorded, _missing, digest = _load_suite_for_replay(suite_path)
    result, _ = _run_suite(suite, recorded, seed)
    scores = [
        {"case_id": r.case_id, "score": r.score, "passed": r.passed}
        for r in result.results if not r.details.get("skipped")
    ]
    doc = {
        "tool": "agenteval-bench",
        "snapshot_version": 1,
        "seed": seed,
        "suite_name": suite.name,
        "suite_file": os.path.basename(suite_path),
        "suite_digest": digest,
        "scorer": "DeterministicScorer",
        "scores": scores,
    }
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, sort_keys=True, indent=2) + "\n")
    print(f"Baseline snapshot: {len(scores)} scores (seed={seed}) -> {out}")
    return 0


def cmd_alt_test(args: list[str]) -> int:
    """Run the Alternative Annotator Test on a recorded annotation dataset.

    Judge labels are recorded data (like replay outputs), never live LLM
    calls. Prints the winning-rate leaderboard; --out writes the JSON report.
    """
    args, data_path = _flag(args, "--data")
    args, epsilon_s = _flag(args, "--epsilon", "0.1")
    args, q_s = _flag(args, "--q", "0.05")
    args, seed_s = _flag(args, "--seed", str(DEFAULT_SEED))
    args, out = _flag(args, "--out")

    if not data_path:
        print("Error: --data <file.jsonl> is required", file=sys.stderr)
        return 1
    try:
        epsilon = float(epsilon_s or "0.1")
        q = float(q_s or "0.05")
        seed = int(seed_s or str(DEFAULT_SEED))
    except ValueError:
        print("Error: --epsilon/--q must be numbers, --seed an integer", file=sys.stderr)
        return 1

    from alt_test.runner import (
        InsufficientCoverage,
        dataset_digest,
        load_jsonl,
        render_text,
        run_alt_test,
    )
    from alt_test.types import AltTestConfig

    try:
        config = AltTestConfig(epsilon=epsilon, q=q, seed=seed)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    try:
        items = load_jsonl(data_path)
    except (OSError, ValueError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if items and items[0].task == "text":
        print(
            "Note: text task uses the deterministic trigram-Jaccard similarity "
            "surrogate (see docs/ALT-TEST.md); inject a custom sim via the "
            "Python API for embedding-based SIM.",
            file=sys.stderr,
        )
    try:
        report = run_alt_test(items, config, input_digest=dataset_digest(items))
    except (OSError, ValueError, TypeError, InsufficientCoverage) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(render_text(report))
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
        print(f"Report: {out}")
    return 0


def cmd_simpson(args: list[str]) -> int:
    """Rule on a "treated beats control overall" claim from per-slice aggregates.

    Reads a JSON doc of per-stratum counts (periods, segments, task
    categories), prints the mandatory per-stratum disaggregation, and
    exits 0 only if the pooled win claim survives Simpson-safe
    aggregation (agenteval-bench#36). Exit 1 = claim REJECTED or pooled
    reporting refused (violations named); exit 2 = unusable input.

    Input schema:
      {"strata": [{"name", "n_treated", "n_control",
                   "treated_successes", "control_successes"}],
       "pooled_n_treated": int, "pooled_n_control": int,
       "allocation_tolerance": float (optional)}
    """
    from experiment.simpson import (
        DEFAULT_ALLOCATION_TOLERANCE,
        Stratum,
        win_claim,
    )

    args, slices_path = _flag(args, "--slices")
    args, tol_s = _flag(args, "--allocation-tolerance")
    args, out = _flag(args, "--out")
    if not slices_path:
        print("Error: simpson needs --slices <file.json>", file=sys.stderr)
        return 2
    try:
        with open(slices_path, encoding="utf-8") as f:
            doc = json.load(f)
        strata = tuple(
            Stratum(
                name=s["name"],
                n_treated=s["n_treated"],
                n_control=s["n_control"],
                treated_successes=s["treated_successes"],
                control_successes=s["control_successes"],
            )
            for s in doc["strata"]
        )
        tolerance = (
            float(tol_s) if tol_s is not None
            else float(doc.get("allocation_tolerance", DEFAULT_ALLOCATION_TOLERANCE))
        )
        if not 0.0 <= tolerance <= 1.0:
            raise ValueError(
                f"allocation_tolerance must be in [0, 1], got {tolerance}"
            )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        print(f"Error: invalid slices doc: {e}", file=sys.stderr)
        return 2
    try:
        pooled_n_treated = int(doc["pooled_n_treated"])
        pooled_n_control = int(doc["pooled_n_control"])
    except (KeyError, TypeError, ValueError) as e:
        print(f"Error: pooled arm totals required: {e}", file=sys.stderr)
        return 2

    verdict = win_claim(
        strata, pooled_n_treated, pooled_n_control, allocation_tolerance=tolerance
    )
    rep = verdict.report
    lines = ["per-stratum disaggregation (mandatory):"]
    for r in rep.strata:
        lines.append(
            f"  {r.name}: treated {r.treated_rate:.2%} "
            f"(n={r.n_treated}, share={r.treated_share:.1%}) vs "
            f"control {r.control_rate:.2%} (n={r.n_control}) "
            f"-> delta {r.delta:+.2%}"
        )
    lines.append(
        f"pooled (descriptive only, not a result): treated {rep.pooled_treated_rate:.2%} "
        f"vs control {rep.pooled_control_rate:.2%} "
        f"-> delta {rep.pooled_delta:+.2%}"
    )
    if verdict.accepted:
        lines.append("verdict: WIN CLAIM ACCEPTED")
    else:
        lines.append(
            "verdict: WIN CLAIM REJECTED "
            f"({', '.join(verdict.violations) or 'pooled numbers do not favor treated'})"
        )
        for note in verdict.notes:
            lines.append(f"  note: {note}")
    print("\n".join(lines))
    if out:
        payload = {
            "tool": "agenteval-bench",
            "command": "simpson",
            "accepted": verdict.accepted,
            "violations": list(verdict.violations),
            "notes": list(verdict.notes),
            "strata": [
                {
                    "name": r.name,
                    "n_treated": r.n_treated,
                    "n_control": r.n_control,
                    "treated_share": r.treated_share,
                    "treated_rate": r.treated_rate,
                    "control_rate": r.control_rate,
                    "delta": r.delta,
                }
                for r in rep.strata
            ],
            "pooled_treated_rate": rep.pooled_treated_rate,
            "pooled_control_rate": rep.pooled_control_rate,
            "pooled_delta": rep.pooled_delta,
        }
        try:
            with open(out, "w", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        except OSError as e:
            print(f"Error: cannot write report: {e}", file=sys.stderr)
            return 2
        print(f"Report: {out}")
    return 0 if verdict.accepted else 1


def cmd_replay(args: list[str]) -> int:
    """Verify a replay log is bit-exact: re-run and byte-compare."""
    args, log_path = _flag(args, "--log")
    args, suite_path = _flag(args, "--suite")
    args, check = _has(args, "--check")
    if not log_path or not suite_path:
        print("Error: replay needs --log <file> --suite <file> [--check]", file=sys.stderr)
        return 1
    with open(log_path, encoding="utf-8") as f:
        original_bytes = f.read()
    try:
        expected = ReplayLog.from_json(original_bytes)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"Error: invalid replay log: {e}", file=sys.stderr)
        return 1

    suite, recorded, _missing, digest = _load_suite_for_replay(suite_path)
    if digest != expected.digest:
        print(f"Error: suite digest mismatch: suite is {digest}, log was {expected.digest}",
              file=sys.stderr)
        return 1
    result, outputs = _run_suite(suite, recorded, expected.seed)
    actual = _build_log(suite, suite_path, digest, result, outputs, expected.seed)

    if not check:
        actual.write(log_path)
        print(f"Replayed {len(actual.cases)} cases (seed={expected.seed}) -> {log_path}")
        return 0

    problems = find_mismatches(expected, actual)
    if problems:
        print(f"Replay NOT bit-exact ({len(problems)} mismatch(es)):")
        for p in problems[:10]:
            print(f"  - {p}")
        return 1
    if actual.to_json() != original_bytes:
        print("Replay NOT bit-exact: canonical bytes differ.")
        return 1
    print(f"Replay bit-exact: {len(actual.cases)} cases, seed={expected.seed}, "
          f"digest={expected.digest[:19]}...")
    return 0


def main() -> None:
    """Minimal CLI — full argparse/typer in v0.2."""
    args = sys.argv[1:]

    if not args or args[0] in ("--help", "-h"):
        print("agenteval-bench — LLM agent evaluation CLI")
        print("Usage: agenteval-bench run --suite <file> [--ci] [--threshold 1.0]")
        print("                         [--seed 42] [--replay-log replay.json]")
        print("       agenteval-bench baseline --suite <file> --out baseline.json [--seed 42]")
        print("       agenteval-bench replay --log replay.json --suite <file> [--check]")
        print("       agenteval-bench alt-test --data annotations.jsonl [--epsilon 0.1]")
        print("                                [--q 0.05] [--seed 42] [--out report.json]")
        print("       agenteval-bench simpson --slices slices.json")
        print("                                [--allocation-tolerance 0.05] [--out report.json]")
        print("       (--seed is recorded for provenance; the test itself is deterministic)")
        return

    cmd, rest = args[0], args[1:]
    if cmd == "run":
        sys.exit(cmd_run(rest))
    elif cmd == "baseline":
        sys.exit(cmd_baseline(rest))
    elif cmd == "replay":
        sys.exit(cmd_replay(rest))
    elif cmd == "alt-test":
        sys.exit(cmd_alt_test(rest))
    elif cmd == "simpson":
        sys.exit(cmd_simpson(rest))
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
