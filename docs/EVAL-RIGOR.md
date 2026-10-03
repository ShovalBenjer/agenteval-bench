# EVAL-RIGOR.md — statistical rigor for agent evaluation

How to compare agents (or agent versions) without fooling yourself. The repo's
CI gate today is a binary pass/fail at a fixed threshold; that is table stakes.
A candidate can regress on hard cases, improve on easy ones, and still clear
the same threshold — a single run's binary verdict cannot tell regression from
noise. This doc sets the bar this repo should hold itself to.

## 1. Compare baseline vs candidate, never a single run

For every eval that informs a merge or a release decision, collect per-case
scores for **both** the baseline (current main / previous version) and the
candidate, then compare them as paired samples:

```
python tools/paired_bootstrap.py baseline.txt candidate.txt
# n=40  mean_diff=+0.0325  95% CI=[+0.0041, +0.0612]  p~0.0280  -> IMPROVED
```

Both files: one score per line, paired by line (same case, same order).
`--n-boot`, `--seed`, `--alpha` are configurable; `--strict` fails on
inconclusive results too, not just regressions.

Executable form: freeze the baseline with
`agenteval-bench baseline --suite <yaml> --seed 42 --out baselines/golden.json`
and compare candidates with
`python tools/paired_bootstrap.py baselines/golden.json candidate.txt`
(snapshots pin seed + suite digest; see `docs/DELTA-BASELINE.md`).

**Gate policy:** block the merge only on a statistically significant
regression (whole CI below 0). "Inconclusive" means "collect more cases or
accept the risk explicitly" — not "ship it and forget it." Report the CI, not
just the verdict: a CI of [+0.001, +0.40] and one of [+0.03, +0.04] are both
"IMPROVED" and they mean very different things.

**Why paired:** the same cases scored for both agents removes case-difficulty
variance. An unpaired comparison needs far more cases for the same power.

## 2. Panel of judges, not a singleton

When LLM-as-judge scoring lands in this repo (spec: planned), do not ship a
single-judge gate. The literature converged: a panel of small diverse models
beats one large judge, with less intra-model bias, at lower cost
(arXiv:2404.18796; Prometheus 2, arXiv:2405.01535; JudgeLM, ICLR 2025).

Minimum viable panel: 3 judges from different vendors/families, score
independently, aggregate by mean (continuous) or majority (binary). Document
the panel composition in the suite YAML — a judge panel is part of the
measurement instrument, and swapping it silently invalidates comparisons.

## 3. Judge-bias checklist

Run through this before trusting any LLM-judge number in this repo:

- [ ] **Position bias.** The judge prefers the first (or second) response.
      Mitigation: present each pair in both orders and average; for >2
      candidates, randomize presentation order per case.
- [ ] **Length bias.** Longer reads as better. Mitigation: rubric anchors that
      reward concision, or normalize/strip length-correlated features before
      judging. Check: does the winner correlate with token count?
- [ ] **Self-preference bias.** A judge rewards outputs from its own model
      family. Mitigation: the judge panel must not share a family with the
      agent under test; never let the candidate's own model judge the contest.
- [ ] **Quality-gap sensitivity.** Bias is worst when candidates are near-tied
      — exactly the prompt-A/B-test case. Near-ties need more cases and wider
      panels, not a louder verdict.

## 4. Adversarial eval rows

`examples/adversarial-scorer.yaml` ships adversarial probes for the
deterministic scorer: anchored regexes that resist keyword stuffing, `exact`
matches for parser-consumed outputs, plus documented **known blind spots**
(`contains` cannot see negation; `json_schema: required` ignores extra keys).
Adversarial rows are regression tests for the *measurement instrument*, not
the agent: if a scorer change makes a probe pass that should fail (or fail
that should pass), the instrument moved, not the agent.

Rule of thumb: every new matcher or scoring change ships with at least one
adversarial row that would catch its characteristic failure mode.

## 5. Red-teaming

Adversarial rows cover the scorer; red-teaming covers the agent. When an agent
under test gains tool calls or user-facing output, add prompt-injection and
jailbreak probes as eval cases (injected instructions in tool output, "ignore
previous instructions" in user input) and score them as hard failures. The
deterministic scorer can already express these: the *expected* output is a
refusal or a safe-completion pattern.

## What would falsify this doc

- If LLM-judge scoring never lands here, section 3 (judge-bias checklist) is
  dead weight — delete it rather than keep aspirational process.
- If `tools/paired_bootstrap.py` is not wired into any CI gate or release
  decision within ~6 months, the script is ceremony — delete the script and
  shrink this doc to the adversarial-rows section.
- If a lighter-weight comparison (e.g. a fixed validation split with a
  pre-registered threshold) empirically catches the same regressions with
  less machinery, replace section 1 with it.
