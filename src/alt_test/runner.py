"""The alt-test procedure: leave-one-out advantage testing per judge.

For every (judge, annotator) pair: exclude the annotator, score how well
the judge's recorded labels align with the remaining annotators versus
how well the excluded annotator's own labels align with them, and test
whether the judge's advantage is significant past the cost-benefit
margin epsilon. Per-annotator p-values go through Benjamini-Yekutieli
FDR control; the winning rate omega is the fraction of rejections.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from alt_test.stats import (
    benjamini_yekutieli,
    paired_t_test_one_sided,
    wilcoxon_signed_rank_one_sided,
)
from alt_test.types import (
    OMEGA_THRESHOLD,
    AltTestConfig,
    AltTestItem,
    AltTestReport,
    AnnotatorComparison,
    JudgeVerdict,
    Label,
    TaskType,
)

#: Similarity between two texts; higher = more similar. The default is a
#: deterministic character-trigram Jaccard surrogate (no embeddings, no
#: network). Callers with embedding infrastructure may inject their own,
#: e.g. cosine similarity over embeddings as in the paper's KiloGram run.
SimFn = Callable[[str, str], float]


def default_text_sim(a: str, b: str) -> float:
    """Deterministic trigram-Jaccard similarity in [0, 1].

    A dependency-free stand-in for the paper's embedding-cosine SIM, used
    only for the text task type. Two identical strings score 1.0.
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    def trigrams(s: str) -> set[str]:
        s = f"  {s}  "
        return {s[i : i + 3] for i in range(len(s) - 2)}

    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def alignment_score(
    candidate: Label,
    others: Sequence[Label],
    task: TaskType,
    sim: SimFn = default_text_sim,
) -> float:
    """S(candidate, x_i, j): alignment of one label with the other annotators.

    Higher is better. Discrete uses ACC (fraction agreeing), continuous
    uses -RMSE, text uses mean pairwise similarity.
    """
    if not others:
        raise ValueError("alignment needs at least one other annotator")
    if task == "discrete":
        return sum(1.0 for o in others if o == candidate) / len(others)
    if task == "continuous":
        if not isinstance(candidate, (int, float)):
            raise TypeError(f"continuous task needs a numeric label, got {candidate!r}")
        vals = []
        for o in others:
            if not isinstance(o, (int, float)):
                raise TypeError(f"continuous task needs numeric labels, got {o!r}")
            vals.append(float(o))
        mse = sum((float(candidate) - v) ** 2 for v in vals) / len(vals)
        return -math.sqrt(mse)
    if task == "text":
        if not isinstance(candidate, str):
            raise TypeError(f"text task needs string labels, got {candidate!r}")
        sims = []
        for o in others:
            if not isinstance(o, str):
                raise TypeError(f"text task needs string labels, got {o!r}")
            sims.append(sim(candidate, o))
        return sum(sims) / len(sims)
    raise ValueError(f"unknown task type: {task}")


class InsufficientCoverage(ValueError):
    """A (judge, annotator) pair shares too few items for a valid test."""


def _validate(items: Sequence[AltTestItem], config: AltTestConfig) -> TaskType:
    if len(items) < config.min_items:
        raise ValueError(
            f"alt-test needs at least {config.min_items} items, got {len(items)}"
        )
    tasks = {item.task for item in items}
    if len(tasks) != 1:
        raise ValueError(
            f"alt-test runs on a single task type; got mixed tasks {sorted(tasks)}"
        )
    for item in items:
        n_ann = len(item.annotator_labels)
        if n_ann < config.min_annotators:
            raise ValueError(
                f"item {item.id!r} has {n_ann} annotators, "
                f"need at least {config.min_annotators}"
            )
    # Items may carry no judge labels at all (a judge absent from an item is
    # simply not scored on it); per-pair coverage is enforced later and
    # names the offending (judge, annotator) pair.
    judges = {name for item in items for name in item.judge_labels}
    if not judges:
        raise ValueError("no judge labels found in the dataset")
    return next(iter(tasks))


def _compare_against_annotator(
    judge: str,
    annotator: str,
    items: Sequence[AltTestItem],
    config: AltTestConfig,
    sim: SimFn,
) -> AnnotatorComparison:
    """Run the leave-one-out comparison of one judge vs one annotator."""
    w_f: list[float] = []
    w_h: list[float] = []
    for item in items:
        judge_label = item.judge_labels.get(judge)
        ann_label = item.annotator_labels.get(annotator)
        if judge_label is None or ann_label is None:
            continue
        others = [
            label
            for name, label in item.annotator_labels.items()
            if name != annotator
        ]
        if not others:
            continue
        s_f = alignment_score(judge_label, others, item.task, sim)
        s_h = alignment_score(ann_label, others, item.task, sim)
        w_f.append(1.0 if s_f >= s_h else 0.0)
        w_h.append(1.0 if s_h >= s_f else 0.0)
    n = len(w_f)
    if n < config.min_pair_items:
        raise InsufficientCoverage(
            f"judge {judge!r} vs annotator {annotator!r}: "
            f"{n} shared items, need at least {config.min_pair_items}"
        )
    rho_f = sum(w_f) / n
    rho_h = sum(w_h) / n
    diffs = [h - f for f, h in zip(w_f, w_h)]
    if n >= config.t_test_min_n:
        p_value = paired_t_test_one_sided(diffs, config.epsilon)
        test_used = "paired-t"
    else:
        p_value = wilcoxon_signed_rank_one_sided(diffs, config.epsilon)
        test_used = "wilcoxon"
    return AnnotatorComparison(
        annotator=annotator,
        n=n,
        rho_f=rho_f,
        rho_h=rho_h,
        p_value=p_value,
        rejected=False,  # filled in after FDR control
        test_used=test_used,
    )


def run_alt_test(
    items: Sequence[AltTestItem],
    config: AltTestConfig | None = None,
    sim: SimFn = default_text_sim,
    input_digest: str = "",
) -> AltTestReport:
    """Run the full alt-test and return the winning-rate leaderboard."""
    config = config or AltTestConfig()
    _validate(items, config)
    items = list(items)
    judges = sorted({name for item in items for name in item.judge_labels})
    annotators = sorted({name for item in items for name in item.annotator_labels})

    verdicts: list[JudgeVerdict] = []
    notes: list[str] = []
    for judge in judges:
        comparisons: list[AnnotatorComparison] = []
        for annotator in annotators:
            try:
                comparisons.append(
                    _compare_against_annotator(judge, annotator, items, config, sim)
                )
            except InsufficientCoverage as e:
                notes.append(str(e))
        if not comparisons:
            # Fail closed: a leaderboard row with zero valid comparisons is
            # a silent non-result. Name the judge and refuse.
            raise InsufficientCoverage(
                f"judge {judge!r}: no (judge, annotator) pair shares "
                f">= {config.min_pair_items} items"
            )
        rejected = benjamini_yekutieli(
            [c.p_value for c in comparisons], config.q
        )
        comparisons = [
            AnnotatorComparison(
                annotator=c.annotator,
                n=c.n,
                rho_f=c.rho_f,
                rho_h=c.rho_h,
                p_value=c.p_value,
                rejected=r,
                test_used=c.test_used,
            )
            for c, r in zip(comparisons, rejected)
        ]
        omega = sum(1.0 for c in comparisons if c.rejected) / len(comparisons)
        rho = sum(c.rho_f for c in comparisons) / len(comparisons)
        verdicts.append(
            JudgeVerdict(
                judge=judge,
                rho=rho,
                omega=omega,
                justified=omega >= OMEGA_THRESHOLD,
                comparisons=tuple(comparisons),
                n_items=len(items),
                m_annotators=len(annotators),
            )
        )
    verdicts.sort(key=lambda v: v.rho, reverse=True)
    return AltTestReport(
        verdicts=tuple(verdicts),
        epsilon=config.epsilon,
        q=config.q,
        n_items=len(items),
        input_digest=input_digest,
        seed=config.seed,
        notes=tuple(notes),
    )


def render_text(report: AltTestReport) -> str:
    """Human-readable leaderboard."""
    lines = [
        "alt-test leaderboard (winning rate omega / avg advantage prob rho)",
        (
            f"epsilon={report.epsilon} q={report.q} "
            f"FDR={report.fdr_procedure} omega_threshold={report.omega_threshold} "
            f"n_items={report.n_items}"
        ),
        "",
        f"{'judge':<24}{'rho':>8}{'omega':>8}{'justified':>11}  per-annotator rho_f",
    ]
    for v in report.verdicts:
        per = ",".join(f"{c.rho_f:.2f}" for c in v.comparisons)
        lines.append(
            f"{v.judge:<24}{v.rho:>8.3f}{v.omega:>8.3f}"
            f"{v.justified!s:>11}  {per}"
        )
    if report.notes:
        lines += ["", "notes:"]
        lines += [f"  - {note}" for note in report.notes]
    return "\n".join(lines) + "\n"


def load_jsonl(path: str) -> list[AltTestItem]:
    """Load an alt-test dataset from JSONL.

    Each line: {"id": str, "task": "discrete"|"continuous"|"text",
                "annotators": {name: label}, "judges": {name: label}}.
    Boundary-typed: malformed rows raise with the line number.
    """
    import json

    items: list[AltTestItem] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{lineno}: expected an object")
            item_id = row.get("id")
            task = row.get("task")
            annotators = row.get("annotators")
            judges = row.get("judges", {})
            if not isinstance(item_id, str) or not item_id:
                raise ValueError(f"{path}:{lineno}: 'id' must be a non-empty string")
            if task not in ("discrete", "continuous", "text"):
                raise ValueError(f"{path}:{lineno}: 'task' must be discrete|continuous|text")
            if not isinstance(annotators, dict) or not annotators:
                raise ValueError(f"{path}:{lineno}: 'annotators' must be a non-empty mapping")
            if not isinstance(judges, dict):
                raise TypeError(f"{path}:{lineno}: 'judges' must be a mapping")
            for name, label in list(annotators.items()) + list(judges.items()):
                if not isinstance(name, str) or not name:
                    raise ValueError(f"{path}:{lineno}: label names must be non-empty strings")
                if not isinstance(label, (str, int, float)) or isinstance(label, bool):
                    raise TypeError(
                        f"{path}:{lineno}: labels must be str or number, got {label!r}"
                    )
            items.append(
                AltTestItem(
                    id=item_id,
                    task=task,
                    annotator_labels=dict(annotators),
                    judge_labels={k: v for k, v in judges.items()},
                )
            )
    return items


def dataset_digest(items: Sequence[AltTestItem]) -> str:
    """sha256 over a canonical serialization of the dataset (provenance)."""
    import hashlib
    import json

    canonical = json.dumps(
        [
            {
                "id": item.id,
                "task": item.task,
                "annotators": dict(sorted(item.annotator_labels.items())),
                "judges": dict(sorted(item.judge_labels.items())),
            }
            for item in items
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
