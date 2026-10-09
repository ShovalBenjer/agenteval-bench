# alt-test benchmark

The Alternative Annotator Test (Calderon, Reichart & Dror, arXiv:2501.10970)
as a first-class benchmark: given an eval set annotated by ≥3 humans plus
one recorded label column per LLM judge, decide per judge whether it can
statistically replace the annotators, and rank judges on a winning-rate
leaderboard.

## What it answers

Not "does the judge agree with humans?" (accuracy/F1 can't give a yes/no
replacement decision) but "is this judge a better alternative than hiring
another annotator?" — the framing the paper argues justifies replacement.

## Procedure (paper-faithful)

For each judge and each annotator in turn (leave-one-out):

1. **Alignment.** Score how well the judge's label and the excluded
   annotator's label each align with the *remaining* annotators:
   - discrete labels: ACC — fraction of remaining annotators agreeing;
   - continuous labels: −RMSE against the remaining annotators;
   - free text: mean pairwise similarity (see honesty notes).
2. **Advantage indicators.** `W^f = 1` when the judge aligns at least as
   well as the annotator; `W^h = 1` for the reverse. Ties count for both.
   `ρ^f` / `ρ^h` are the per-annotator advantage probabilities (means).
3. **Test.** Paired t-test of `H0: ρ^h − ρ^f ≥ ε` on the per-item
   differences `d = W^h − W^f` (one-sided). `ε` is the cost-benefit margin:
   the judge gets a small edge reflecting that it is cheaper than a human.
   Below n = 30 the paper's prescribed Wilcoxon signed-rank path is used.
4. **FDR control.** The m per-annotator p-values go through the
   **Benjamini–Yekutieli** procedure (q = 0.05 default) — BY, not BH,
   because the leave-one-out hypotheses are dependent.
5. **Winning rate.** `ω` = fraction of rejected nulls. `ω ≥ 0.5` justifies
   replacing annotators with that judge.

For *comparing* judges, the paper recommends **Average Advantage
Probability** `ρ = mean(ρ^f)` over `ω`: it is denser, magnitude-aware,
and directly reads as "probability the judge is as good as or better
than a randomly chosen annotator". The leaderboard sorts by `ρ`.

## Usage

```bash
# JSONL: {"id", "task": "discrete"|"continuous"|"text",
#         "annotators": {name: label}, "judges": {name: label}}
agenteval-bench alt-test --data annotations.jsonl \
  --epsilon 0.1 --q 0.05 --seed 42 --out report.json
```

Acceptance: refuses datasets with <3 annotators per item or <50 items;
refuses mixed task types; a (judge, annotator) pair sharing <10 items
raises `InsufficientCoverage` naming the pair (fail-closed, no silent
non-result). The JSON report echoes `epsilon`, `q`, the FDR procedure
name, and a sha256 `input_digest` of the dataset, so any result is
reproducible from the single command above.

A seeded synthetic demo: `python -m alt_test.demo` (needs `src/` on
`PYTHONPATH` for a source checkout).

## Honesty notes

- **Judge labels are recorded data, never live LLM calls.** Like the
  `replay` command scores recorded agent outputs without a live agent,
  this benchmark reads judge label columns from the dataset. There is no
  judge-call path in this repo (LLM-as-judge scoring is still `planned`
  in `docs/spec.md`); nothing here implies one.
- **Text similarity is a deterministic surrogate.** The paper's free-text
  runs use embedding cosine similarity. The default `sim` here is a
  character-trigram Jaccard — dependency-free and reproducible, but a
  stand-in. The seam accepts any `sim: (str, str) -> float` callable;
  inject embedding cosine when you have the infrastructure.
- **ε is a judgment call, not a constant of nature.** The report always
  prints the ε used. Sweeping ε must move ω monotonically (pinned by
  `test_epsilon_monotonicity_of_omega`); if your conclusion flips inside
  a plausible ε range, say so.
- **Theorem 1 is the implementation's conscience.** A judge predicting
  the majority vote (discrete) or mean annotation (continuous) must reach
  `ρ = 1.0` exactly — pinned by tests, not by prose.

## References

- Calderon, Reichart & Dror, "The Alternative Annotator Test for
  LLM-as-a-Judge" (arXiv:2501.10970, CC0). Reference implementation
  studied, not forked: github.com/nitaytech/AltTest.
- `src/alt_test/`: `types.py` (contracts), `stats.py` (t / Wilcoxon /
  Benjamini–Yekutieli), `runner.py` (procedure + CLI I/O), `demo.py`.
- Issue: ShovalBenjer/agenteval-bench#31.
