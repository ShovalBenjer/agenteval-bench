"""Adversarial tests for immutable benchmark snapshots (agenteval-bench#38).

BetterBench discipline: version, never mutate. Every test below attacks the
immutability guarantee, the version chain, or the regression gate — not just
the happy path.
"""

from __future__ import annotations

import json

import pytest

from agenteval_bench.engine import EvalRunner
from agenteval_bench.models import EvalCase, EvalSuite, ExpectedOutput, RubricCriterion
from agenteval_bench.regression import (
    ProductionFailure,
    PromotionRecord,
    compare_against_frozen,
    promote_failure,
    promote_failures_to_version,
)
from agenteval_bench.replay import suite_digest
from agenteval_bench.snapshots import (
    BaselineRecord,
    SnapshotCorrupted,
    SnapshotError,
    SnapshotStore,
    VersionExistsError,
    VersionMeta,
    VersionNotFoundError,
    baseline_from_run,
    suite_from_canonical,
    suite_to_canonical,
)


def _suite(name: str = "support") -> EvalSuite:
    return EvalSuite(
        name=name,
        version=1,
        cases=[
            EvalCase(
                id="greet",
                input="say hello",
                expected=ExpectedOutput(contains=["hello"]),
                rubric=[RubricCriterion(criterion="tone", weight=1.0, description="polite")],
                output="hello there",
            ),
            EvalCase(
                id="farewell",
                input="say goodbye",
                expected=ExpectedOutput(exact="goodbye"),
                output="goodbye",
            ),
        ],
    )


def _meta(**kw) -> VersionMeta:
    return VersionMeta(
        owner="eval-team",
        why=kw.pop("why", "initial frozen baseline"),
        changed_from_prev=kw.pop("changed_from_prev", ""),
        cadence=kw.pop("cadence", "reviewed weekly"),
        regression_set=kw.pop("regression_set", "support escalations"),
        feedback_loop=kw.pop("feedback_loop", "escalations -> promote_failure"),
    )


def _baseline() -> BaselineRecord:
    suite = _suite()
    result = EvalRunner().run(suite, lambda i: "hello there" if "hello" in i else "goodbye")
    return baseline_from_run(result, seed=42, agent="golden-v1")


def _publish(store: SnapshotStore, suite: EvalSuite, version_id: str,
             **kw) -> tuple:
    baseline = kw.pop("baseline", None) or _baseline()
    version = store.publish(
        suite,
        version_id,
        kw.pop("meta", None) or _meta(),
        baseline,
        prev_version_id=kw.pop("prev_version_id", None),
        promotions=kw.pop("promotions", None),
    )
    return version, suite


# ---------------------------------------------------------------- publish/load

def test_publish_load_roundtrip(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    version, loaded = store.load("support", "v1")
    assert version.version_id == "v1"
    assert version.digest == suite_digest(suite_to_canonical(_suite()))
    assert version.digest.startswith("sha256:")
    assert [c.id for c in loaded.cases] == ["greet", "farewell"]
    assert version.meta.owner == "eval-team"
    assert version.baseline.agent == "golden-v1"
    assert version.prev_version_id is None


def test_version_chain_records_change_and_why(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    suite2 = _suite()
    suite2.cases.append(
        EvalCase(id="refund", input="refund policy?",
                 expected=ExpectedOutput(contains=["refund"]), output="refund here")
    )
    version2, _ = _publish(
        store,
        suite2,
        "v2",
        prev_version_id="v1",
        meta=_meta(why="new failure mode: refund confusion", changed_from_prev="added refund case"),
    )
    assert version2.prev_version_id == "v1"
    assert version2.meta.why == "new failure mode: refund confusion"
    assert version2.meta.changed_from_prev == "added refund case"
    assert version2.digest != suite_digest(suite_to_canonical(_suite()))
    # v1 stays frozen — its digest is unchanged by the v2 publish.
    version1, _ = store.load("support", "v1")
    assert version1.digest == suite_digest(suite_to_canonical(_suite()))
    assert store.list_versions("support") == ["v1", "v2"]


def test_canonical_fixed_point(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    raw = (tmp_path / "store" / "support" / "v1" / "suite.json").read_bytes()
    reparsed = suite_from_canonical(raw)
    assert suite_to_canonical(reparsed) == raw


# ------------------------------------------------- adversarial: tamper / abuse

def test_on_disk_mutation_detected(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    target = tmp_path / "store" / "support" / "v1" / "suite.json"
    raw = bytearray(target.read_bytes())
    raw[raw.index(b"goodbye")] ^= 0x01  # flip one byte inside a case
    target.write_bytes(bytes(raw))
    with pytest.raises(SnapshotCorrupted, match="hash mismatch"):
        store.load("support", "v1")


def test_manifest_digest_tamper_detected(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    target = tmp_path / "store" / "support" / "v1" / "manifest.json"
    manifest = json.loads(target.read_text())
    manifest["digest"] = "sha256:" + "0" * 64
    target.write_text(json.dumps(manifest))
    with pytest.raises(SnapshotCorrupted, match="hash mismatch"):
        store.load("support", "v1")


def test_canonical_schema_violation_detected(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    target = tmp_path / "store" / "support" / "v1" / "suite.json"
    # Rewriting the file is not enough: the digest must also match, so craft
    # a *consistent* file whose schema is invalid (duplicate case id).
    doc = json.loads(target.read_text())
    doc["cases"].append(dict(doc["cases"][0]))
    new_bytes = (json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    target.write_bytes(new_bytes)
    manifest_target = tmp_path / "store" / "support" / "v1" / "manifest.json"
    manifest = json.loads(manifest_target.read_text())
    manifest["digest"] = suite_digest(new_bytes)
    manifest_target.write_text(json.dumps(manifest))
    with pytest.raises(SnapshotCorrupted, match="duplicate case id"):
        store.load("support", "v1")


def test_duplicate_version_publish_refused(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    before = (tmp_path / "store" / "support" / "v1" / "suite.json").read_bytes()
    with pytest.raises(VersionExistsError):
        _publish(store, _suite(), "v1")
    # The original bytes are untouched.
    assert (tmp_path / "store" / "support" / "v1" / "suite.json").read_bytes() == before


def test_broken_chain_refused(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    with pytest.raises(VersionNotFoundError):
        _publish(store, _suite(), "v2", prev_version_id="v1")


def test_load_missing_version(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    with pytest.raises(VersionNotFoundError):
        store.load("support", "v9")
    with pytest.raises(VersionNotFoundError):
        store.load("nosuchsuite", "v1")


def test_verify_all_reports_problems(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    _publish(store, _suite(), "v2", prev_version_id="v1")
    assert store.verify_all() == []
    target = tmp_path / "store" / "support" / "v1" / "suite.json"
    target.write_bytes(target.read_bytes() + b" ")
    problems = store.verify_all()
    assert len(problems) == 1
    assert "support/v1" in problems[0]
    # v2 still verifies — corruption is isolated to the tampered version.
    store.verify("support", "v2")


def test_unsafe_version_id_rejected(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    for bad in ("", "../evil", "a/b", "a\\b"):
        with pytest.raises(SnapshotError):
            _publish(store, _suite(), bad)


# ------------------------------------------------- promotion -> frozen replay

def test_promote_failure_roundtrip(tmp_path):
    failure = ProductionFailure(
        failure_id="esc-1042",
        kind="escalation",
        trace_summary="user: refund my order -- agent: I cannot help with that",
        reference_answer="I can help process your refund",
        prior_score=0.0,
        evaluator_notes="agent refused a legitimate refund request",
    )
    case, record = promote_failure(failure)
    assert case.id == "prod:escalation:esc-1042"
    assert case.expected.exact == "I can help process your refund"

    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    suite2 = _suite()
    suite2.cases.append(case)
    _publish(
        store,
        suite2,
        "v2",
        prev_version_id="v1",
        meta=_meta(why="promote esc-1042", changed_from_prev="added prod:escalation:esc-1042"),
        promotions=[record.to_dict()],
    )
    version2, frozen = store.load("support", "v2")
    assert version2.promotions[0]["failure_id"] == "esc-1042"
    assert PromotionRecord.from_dict(version2.promotions[0]).promoted_case_id == case.id

    # The promoted case replays deterministically through the frozen set.
    def fixed_agent(_input: str) -> str:
        return "I can help process your refund"

    first = EvalRunner().run(frozen, fixed_agent, seed=7)
    second = EvalRunner().run(frozen, fixed_agent, seed=7)
    assert [r.score for r in first.results] == [r.score for r in second.results]
    by_id = {r.case_id: r for r in first.results}
    assert by_id[case.id].passed


def test_promote_failure_validation():
    with pytest.raises(ValueError):
        ProductionFailure(failure_id="", kind="escalation", trace_summary="t",
                          reference_answer="r", prior_score=0.0)
    with pytest.raises(ValueError):
        ProductionFailure(failure_id="x", kind="escalation", trace_summary="t",
                          reference_answer="r", prior_score=1.5)


# ------------------------------------------------- regression gate

def _golden(_input: str) -> str:
    return "hello there" if "hello" in _input else "goodbye"


def test_regression_gate_passes_on_equal_candidate(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    report = compare_against_frozen(store, "support", "v1", _golden, candidate_agent="v2-candidate")
    assert report.verdict == "GATE_PASS"
    assert report.regressed_cases == ()
    assert report.candidate_pass_rate == report.baseline_pass_rate == 1.0
    assert report.digest.startswith("sha256:")


def test_regression_gate_fails_on_per_case_regression(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")

    def degraded(_input: str) -> str:
        return "hello there" if "hello" in _input else "bye"  # breaks 'farewell'

    report = compare_against_frozen(store, "support", "v1", degraded)
    assert report.verdict == "GATE_FAIL"
    assert report.regressed_cases == ("farewell",)
    assert any("farewell" in r for r in report.reasons)
    assert report.candidate_pass_rate == 0.5 < report.baseline_pass_rate


def test_regression_gate_fails_on_corrupted_snapshot(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    target = tmp_path / "store" / "support" / "v1" / "suite.json"
    target.write_bytes(target.read_bytes().replace(b"hello", b"hallo", 1))
    with pytest.raises(SnapshotCorrupted):
        compare_against_frozen(store, "support", "v1", _golden)


def test_regression_gate_respects_explicit_threshold(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    # Baseline is weak: only half the cases pass.
    suite = _suite()
    weak_result = EvalRunner().run(
        suite, lambda i: "hello there" if "hello" in i else "nope")
    weak_baseline = baseline_from_run(weak_result, seed=42, agent="weak-v1")
    store.publish(suite, "v1", _meta(), weak_baseline)
    # A candidate at the same weak level passes the default (baseline) bar...
    report = compare_against_frozen(
        store, "support", "v1", lambda i: "hello there" if "hello" in i else "nope")
    assert report.verdict == "GATE_PASS"
    # ...but fails an explicit release bar of 0.9.
    report = compare_against_frozen(
        store, "support", "v1", lambda i: "hello there" if "hello" in i else "nope",
        min_pass_rate=0.9)
    assert report.verdict == "GATE_FAIL"
    assert any("threshold" in r for r in report.reasons)


# ------------------------------------------------- CLI: snapshot compare/show

_CLI_SUITE = """
name: cli-smoke
cases:
  - id: a
    input: "ping"
    output: "pong ok"
    expected:
      contains: ["pong"]
  - id: b
    input: "ding"
    output: "dong"
    expected:
      exact: "dong"
"""


def _cli_run(argv, monkeypatch, capsys):
    from agenteval_bench import cli

    monkeypatch.setattr("sys.argv", ["agenteval-bench", *argv])
    try:
        cli.main()
        captured = capsys.readouterr()
        return 0, captured.out, captured.err
    except SystemExit as e:
        captured = capsys.readouterr()
        return e.code or 0, captured.out, captured.err


def _cli_publish(monkeypatch, capsys, tmp_path, promotions=None):
    import json

    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(_CLI_SUITE)
    store = str(tmp_path / "store")
    argv = ["snapshot", "publish", "--suite", str(suite_path), "--store", store,
            "--version", "v1", "--owner", "eval-team", "--why", "smoke baseline",
            "--changed", "initial freeze", "--cadence", "weekly"]
    if promotions is not None:
        prom_path = tmp_path / "promotions.json"
        prom_path.write_text(json.dumps(promotions))
        argv += ["--promotions", str(prom_path)]
    code, _out, err = _cli_run(argv, monkeypatch, capsys)
    assert code == 0, err
    return store


def test_cli_compare_pass_and_fail(monkeypatch, capsys, tmp_path):
    import json

    store = _cli_publish(monkeypatch, capsys, tmp_path)
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"a": "pong ok", "b": "dong"}))
    code, out, _ = _cli_run(
        ["snapshot", "compare", "--store", store, "--suite", "cli-smoke",
         "--version", "v1", "--candidate-outputs", str(good),
         "--candidate-name", "cand-v2"], monkeypatch, capsys)
    assert code == 0, out
    assert "GATE_PASS" in out
    assert "cand-v2" in out

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"a": "pong ok", "b": "WRONG"}))
    code, out, _ = _cli_run(
        ["snapshot", "compare", "--store", store, "--suite", "cli-smoke",
         "--version", "v1", "--candidate-outputs", str(bad)], monkeypatch, capsys)
    assert code == 2, out  # CI convention: 2 = gate failed
    assert "GATE_FAIL" in out
    assert "b" in out  # regressed case named


def test_cli_compare_missing_candidate_output_scores_zero(monkeypatch, capsys, tmp_path):
    import json

    store = _cli_publish(monkeypatch, capsys, tmp_path)
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"a": "pong ok"}))
    code, out, _ = _cli_run(
        ["snapshot", "compare", "--store", store, "--suite", "cli-smoke",
         "--version", "v1", "--candidate-outputs", str(partial)], monkeypatch, capsys)
    assert code == 2, out
    assert "no candidate output" in out


def test_cli_compare_fails_closed_on_tampered_snapshot(monkeypatch, capsys, tmp_path):
    import json

    store = _cli_publish(monkeypatch, capsys, tmp_path)
    target = tmp_path / "store" / "cli-smoke" / "v1" / "suite.json"
    target.write_bytes(target.read_bytes().replace(b"pong", b"ping", 1))
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"a": "pong ok", "b": "dong"}))
    code, _out, err = _cli_run(
        ["snapshot", "compare", "--store", store, "--suite", "cli-smoke",
         "--version", "v1", "--candidate-outputs", str(good)], monkeypatch, capsys)
    assert code == 1
    assert "CORRUPTED" in err


def test_cli_show_prints_version_record(monkeypatch, capsys, tmp_path):
    store = _cli_publish(monkeypatch, capsys, tmp_path)
    code, out, _ = _cli_run(
        ["snapshot", "show", "--store", store, "--suite", "cli-smoke", "--version", "v1"],
        monkeypatch, capsys)
    assert code == 0
    assert "owner: eval-team" in out
    assert "why: smoke baseline" in out
    assert "changed from None: initial freeze" in out
    assert "update cadence: weekly" in out
    assert "digest=sha256:" in out
    assert "baseline: pass_rate 100.0%" in out


def test_cli_publish_with_promotions_roundtrip(monkeypatch, capsys, tmp_path):
    failure = ProductionFailure(
        failure_id="esc-7", kind="failed_tool_call",
        trace_summary="tool timed out", reference_answer="retry with backoff",
        prior_score=0.0, evaluator_notes="timeout not retried")
    _, record = promote_failure(failure)
    store = _cli_publish(monkeypatch, capsys, tmp_path, promotions=[record.to_dict()])
    version, _ = SnapshotStore(store).load("cli-smoke", "v1")
    assert len(version.promotions) == 1
    assert version.promotions[0]["failure_id"] == "esc-7"
    code, out, _ = _cli_run(
        ["snapshot", "show", "--store", store, "--suite", "cli-smoke", "--version", "v1"],
        monkeypatch, capsys)
    assert code == 0, out
    assert "esc-7" in out


def test_cli_compare_rejects_bad_candidate_file(monkeypatch, capsys, tmp_path):
    import json

    store = _cli_publish(monkeypatch, capsys, tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(["not", "a", "mapping"]))
    code, _, err = _cli_run(
        ["snapshot", "compare", "--store", store, "--suite", "cli-smoke",
         "--version", "v1", "--candidate-outputs", str(bad)], monkeypatch, capsys)
    assert code == 2
    assert "mapping" in err


# ------------------------------------------------- journal: manifest integrity

def _manifest_path(tmp_path, suite="support", version="v1"):
    return tmp_path / "store" / suite / version / "manifest.json"


def test_journal_chains_publishes(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    _publish(store, _suite(), "v2", prev_version_id="v1")
    entries = store._journal_entries()
    assert len(entries) == 2
    assert entries[0]["prev_chain"] == "GENESIS"
    assert entries[1]["prev_chain"] == entries[0]["chain"]
    assert entries[0]["suite_name"] == "support"
    assert entries[1]["version_id"] == "v2"


def test_manifest_baseline_tamper_detected(tmp_path):
    # The release-gate hole: weaken the pinned baseline without touching
    # suite.json. The suite-bytes hash still matches; the journal must catch it.
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    target = _manifest_path(tmp_path)
    manifest = json.loads(target.read_text())
    manifest["baseline"]["pass_rate"] = 0.0
    manifest["baseline"]["per_case"] = {c: 0.0 for c in manifest["baseline"]["per_case"]}
    target.write_text(json.dumps(manifest))
    # The manifest-level digest fires first (it subsumes the baseline check).
    with pytest.raises(SnapshotCorrupted, match="manifest"):
        store.load("support", "v1")
    problems = store.verify_all()
    assert any("support/v1" in p for p in problems)


def test_manifest_meta_tamper_detected(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    target = _manifest_path(tmp_path)
    manifest = json.loads(target.read_text())
    manifest["meta"]["why"] = "rewritten history"
    target.write_text(json.dumps(manifest))
    with pytest.raises(SnapshotCorrupted, match="meta"):
        store.load("support", "v1")


def test_journal_chain_break_detected(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    _publish(store, _suite(), "v2", prev_version_id="v1")
    journal = tmp_path / "store" / "journal.jsonl"
    lines = journal.read_text().splitlines()
    entry = json.loads(lines[1])
    entry["chain"] = "0" * 64  # forge the chain link
    lines[1] = json.dumps(entry)
    journal.write_text("\n".join(lines) + "\n")
    problems = store.verify_all()
    assert any("journal" in p for p in problems)


def test_journal_line_tamper_detected(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    journal = tmp_path / "store" / "journal.jsonl"
    journal.write_text("not json\n")
    problems = store.verify_all()
    assert any("journal" in p for p in problems)


def test_repair_journal_after_crash(tmp_path):
    # Simulate a crash between version rename and journal append.
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    (tmp_path / "store" / "journal.jsonl").unlink()
    with pytest.raises(SnapshotCorrupted, match="no publish journal entry"):
        store.load("support", "v1")
    repaired = store.repair_journal()
    assert repaired == ["support/v1"]
    version, _ = store.load("support", "v1")  # works again, fully re-verified
    assert version.version_id == "v1"
    assert store.verify_all() == []


def test_repair_journal_refuses_corrupt_version(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    (tmp_path / "store" / "journal.jsonl").unlink()
    target = tmp_path / "store" / "support" / "v1" / "suite.json"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(SnapshotCorrupted):
        store.repair_journal()  # never journals a corrupt version


# ------------------------------------------------- arch fix-forward: B2, A1, A2, A5, A6

def test_compare_uses_pinned_baseline_seed(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    suite = _suite()
    result = EvalRunner().run(suite, _golden)
    baseline = baseline_from_run(result, seed=7, agent="golden")
    store.publish(suite, "v1", _meta(), baseline)
    report = compare_against_frozen(store, "support", "v1", _golden)
    assert report.seed == 7  # pinned seed, not the module default
    report = compare_against_frozen(store, "support", "v1", _golden, seed=99)
    assert report.seed == 99  # explicit override respected


def test_promote_failures_to_version(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    suite = _suite()
    v1, _ = _publish(store, suite, "v1")
    failures = [
        ProductionFailure(failure_id="f1", kind="escalation", trace_summary="t1",
                          reference_answer="r1", prior_score=0.0),
        ProductionFailure(failure_id="f2", kind="failed_tool_call", trace_summary="t2",
                          reference_answer="r2", prior_score=0.25,
                          evaluator_notes="tool flaked"),
    ]
    version2, new_suite = promote_failures_to_version(
        store, suite, "v2", _meta(why="promote f1,f2"), failures, _baseline(),
        prev_version_id="v1")
    assert version2.version_id == "v2"
    assert version2.prev_version_id == "v1"
    assert len(new_suite.cases) == len(suite.cases) + 2  # caller's suite untouched
    assert len(suite.cases) == 2
    assert len(version2.promotions) == 2
    assert version2.promotions[1]["failure_id"] == "f2"
    # v1 frozen: same digest, no new cases.
    old_version, old_suite = store.load("support", "v1")
    assert old_version.digest == v1.digest
    assert len(old_suite.cases) == 2


def test_publish_claims_version_id_exclusively(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    # Simulate a concurrent publisher that already claimed the id (empty dir).
    vdir = tmp_path / "store" / "support" / "v1"
    vdir.mkdir(parents=True)
    with pytest.raises(VersionExistsError):
        _publish(store, _suite(), "v1")


def test_promotion_record_kind_validated():
    with pytest.raises(ValueError, match="kind must be one of"):
        PromotionRecord.from_dict({
            "failure_id": "x", "kind": "nope", "promoted_case_id": "c",
            "prior_score": 0.0, "evaluator_notes": "", "promoted_at": "",
        })


def test_cli_publish_refuses_vacuous_baseline(monkeypatch, capsys, tmp_path):
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text('name: empty\ncases:\n  - id: a\n    input: "q"\n    expected:\n      contains: ["x"]\n')
    code, _, err = _cli_run(
        ["snapshot", "publish", "--suite", str(suite_path),
         "--store", str(tmp_path / "store"), "--version", "v1",
         "--owner", "t", "--why", "w"], monkeypatch, capsys)
    assert code == 2
    assert "no recorded" in err


def test_cli_compare_defaults_to_pinned_seed(monkeypatch, capsys, tmp_path):
    import json

    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(_CLI_SUITE)
    store = str(tmp_path / "store")
    code, _, err = _cli_run(
        ["snapshot", "publish", "--suite", str(suite_path), "--store", store,
         "--version", "v1", "--owner", "t", "--why", "w", "--seed", "7"],
        monkeypatch, capsys)
    assert code == 0, err
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"a": "pong ok", "b": "dong"}))
    code, out, _ = _cli_run(
        ["snapshot", "compare", "--store", store, "--suite", "cli-smoke",
         "--version", "v1", "--candidate-outputs", str(good)], monkeypatch, capsys)
    assert code == 0, out
    assert "seed=7" in out


# ------------------------------------------------- impl fix-forward: B1, B2, A2, A10

def test_publish_rejects_duplicate_case_ids(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    suite = _suite()
    suite.cases.append(EvalCase(id="greet", input="dup", expected=ExpectedOutput()))
    with pytest.raises(SnapshotError, match="pre-publish validation"):
        store.publish(suite, "v1", _meta(), _baseline())
    # The version id is NOT bricked: nothing was written, republish works.
    assert store.list_versions("support") == []
    version, _ = _publish(store, _suite(), "v1")
    assert version.version_id == "v1"


def test_publish_rejects_noncanonical_types(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    suite = _suite()
    suite.cases[0].rubric[0].weight = 1  # type: ignore[assignment]  # int, not float
    with pytest.raises(SnapshotError, match="canonical-stable"):
        store.publish(suite, "v1", _meta(), _baseline())
    assert store.list_versions("support") == []


def test_publish_rejects_suite_name_traversal(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    suite = _suite(name="../escaped")
    with pytest.raises(SnapshotError, match="safe path segment"):
        store.publish(suite, "v1", _meta(), _baseline())
    assert not (tmp_path / "escaped").exists()


def test_dot_version_id_rejected(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    with pytest.raises(SnapshotError):
        _publish(store, _suite(), ".v1")


def test_regression_gate_per_case_arm_isolated(tmp_path):
    # Aggregate held equal (0.5 -> 0.5) while one case regresses and another
    # improves: the per-case arm alone must fail the gate.
    store = SnapshotStore(tmp_path / "store")
    suite = EvalSuite(
        name="s",
        cases=[
            EvalCase(id="a", input="ia", expected=ExpectedOutput(exact="x"), output="x"),
            EvalCase(id="b", input="ib", expected=ExpectedOutput(exact="y"), output="y"),
        ],
    )
    base_result = EvalRunner().run(
        suite, lambda i: "x" if i == "ia" else "WRONG")  # a passes, b fails -> 0.5
    store.publish(suite, "v1", _meta(),
                  baseline_from_run(base_result, seed=42, agent="base"))
    report = compare_against_frozen(
        store, "s", "v1", lambda i: "WRONG" if i == "ia" else "y")  # a regresses, b fixed
    assert report.candidate_pass_rate == report.baseline_pass_rate == 0.5
    assert report.verdict == "GATE_FAIL"
    assert report.regressed_cases == ("a",)


def test_manifest_chain_fields_tamper_detected(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    _publish(store, _suite(), "v1")
    _publish(store, _suite(), "v2", prev_version_id="v1")
    target = _manifest_path(tmp_path, version="v2")
    manifest = json.loads(target.read_text())
    manifest["prev_version_id"] = "v0"  # forge the chain link
    target.write_text(json.dumps(manifest))
    with pytest.raises(SnapshotCorrupted, match="manifest"):
        store.load("support", "v2")
    manifest = json.loads(target.read_text())
    manifest["prev_version_id"] = "v1"
    manifest["promotions"] = [{"forged": True}]
    target.write_text(json.dumps(manifest))
    with pytest.raises(SnapshotCorrupted, match="manifest"):
        store.load("support", "v2")


def test_version_meta_rejects_empty_owner_why():
    with pytest.raises(ValueError):
        VersionMeta(owner="", why="x")
    with pytest.raises(ValueError):
        VersionMeta(owner="x", why="")


def test_production_failure_kind_validated():
    with pytest.raises(ValueError, match="kind must be one of"):
        ProductionFailure(failure_id="x", kind="nope", trace_summary="t",  # type: ignore[arg-type]
                          reference_answer="r", prior_score=0.0)


def test_promotion_record_missing_key():
    with pytest.raises(ValueError, match="missing"):
        PromotionRecord.from_dict({"failure_id": "x"})
