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
