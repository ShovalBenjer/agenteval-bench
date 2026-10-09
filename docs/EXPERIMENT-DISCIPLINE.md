# EXPERIMENT-DISCIPLINE.md — pre-registration, no peeking, CUPED

How to run a model-comparison experiment in this repo without fooling
yourself. Mechanism source: Kohavi, Tang & Xu, *Trustworthy Online
Controlled Experiments*.

## 1. Pre-register the experiment

Before data arrives, write down the sample size and the stopping rule:

```python
from experiment import ExperimentPlan, CovariateSpec, register_plan

plan = ExperimentPlan(
    name="reranker-v2-vs-v1",
    n_planned=800,
    alpha=0.05,
    looks=(0.5, 1.0),          # interim analyses permitted ONLY here
    covariate=CovariateSpec("pre_spend", "pre_experiment"),
)
digest = register_plan(plan, "plans/experiments.jsonl")
```

The registry is hash-chained: each record commits to the previous
record's digest, so a plan cannot be silently rewritten after data
arrives. A rewritten registry is detected (`RegistryCorrupted`) and the
evidence is refused as `UNREGISTERED_PLAN`.

## 2. No peeking: interim looks go through the gate

```python
from experiment import SequentialGate

gate = SequentialGate(plan)
decision = gate.look(0.5, z_statistic)   # interim analysis at t=0.5
```

The gate enforces the registered schedule: an unregistered fraction, a
duplicate, or an out-of-order look raises `PeekRefused`. There is no
other supported way to peek — and the verification seam replays your
reported looks through a fresh gate, so an off-schedule peek in the
report is caught as `UNREGISTERED_PEEK`.

Interim boundaries come from the Lan–DeMets alpha-spending framework
with the O'Brien–Fleming spending function
(alpha*(t) = 2 - 2*Phi(z_{alpha/2}/sqrt(t))), solved by Armitage
recursive integration over the canonical joint distribution of
sequential z-statistics. Early looks are strict (~2.77 at t=0.5 for a
two-sided 5% test), the final look is near-nominal (~1.97).

**Measured, not claimed** (seeded, `python -m experiment.demo`):
2000 null A/B experiments, one interim peek each —

| procedure | false-positive rate |
|---|---|
| naive peeking at 5% per look | **0.085** (inflated — the issue's premise) |
| OF alpha-spending through the gate | **0.0495** (controlled) |

## 2b. Stopping early on an interim reject

The spending schedule is a *stopping* rule: when an interim look
rejects, the experiment ends. Declare it in the evidence:

```python
from experiment import ExperimentEvidence, LookEvidence

evidence = ExperimentEvidence(
    n_final=400,                    # n_planned * 0.5, within tolerance
    looks=(LookEvidence(0.5, z, True),),
    cuped=result,
    stopped_at=0.5,
)
```

Verification then requires: the stopping fraction is a registered look;
the looks run exactly through it; the reject is re-derived from z
(a claimed reject that the z-statistic does not support is
`DECISION_MISMATCH`); and n_final is within tolerance of
`n_planned * stopped_at`. Stopping early on an *accept* is refused as
`EARLY_STOP_WITHOUT_REJECT` — quitting without a reject is peeking and
quitting, not a stopping rule.

## 3. CUPED where covariates exist

If the plan declares a pre-experiment covariate, the final evidence must
carry the CUPED adjustment or verification fails with `CUPED_REQUIRED`:

```python
from experiment import cuped_adjust

result = cuped_adjust(y_control, y_treated, x_control, x_treated, plan.covariate)
print(result.variance_ratio)   # < 1: variance reduced
```

Two load-bearing choices: theta is estimated on the **control group
only** (pooled/treatment slopes leak the effect into the adjustment),
and post-treatment covariates are **refused fail-closed**. Measured on
seeded data: variance ratio 0.21 with bias +0.009 against a true effect
of 0.2.

## 4. Violations fail CI

```python
from experiment import check_experiment, ExperimentEvidence, LookEvidence

verdict = check_experiment(digest, "plans/experiments.jsonl", ExperimentEvidence(
    n_final=800,
    looks=(LookEvidence(0.5, z1, r1), LookEvidence(1.0, z2, r2)),
    cuped=result,
))  # raises ExperimentViolation naming the violation otherwise
```

Named violations: `UNREGISTERED_PLAN`, `REGISTRY_UNREADABLE`,
`SAMPLE_DEVIATION` (final n outside the ±2% tolerance, or outside
tolerance of `n_planned * stopped_at` after an early stop),
`LOOK_SCHEDULE_MISMATCH`, `UNREGISTERED_PEEK`, `DECISION_MISMATCH`
(a fabricated accept/reject flag — every decision is re-derived from
the z-statistic), `EARLY_STOP_WITHOUT_REJECT`, `CUPED_REQUIRED`,
`CUPED_COVERAGE_MISMATCH` (the CUPED arm n's must sum to n_final). CI
runs the full drill battery via `python -m experiment.demo`, which
exits non-zero unless every refusal fires and the compliant paths
(full schedule and early stop) verify clean.

## Honesty boundaries (stated, not hidden)

- Discipline is enforced at the `verify_experiment`/`check_experiment`
  seam. The pure math helpers stay directly callable; nothing stops a
  determined caller from computing a boundary by hand. The seam is what
  CI runs, and tests pin the seam.
- The verifier is a **consistency checker over reported evidence** —
  reported fractions against the plan, reported decisions against
  reported z's, reported arm n's against n_final. It does not see raw
  data, so a fabricated-but-internally-consistent report verifies clean.
  What the registry binds is the *plan* (immutably, via the hash chain),
  not the data.
- The hash chain delivers immutability, not precedence: `registered_at`
  is recorded but nothing proves the plan was registered *before* data
  collection. A plan registered into a fresh registry after seeing the
  data verifies clean. The defense is workflow — require the digest
  before the run starts — not cryptography.
- `CUPED_REQUIRED` is presence plus coverage (arm n's sum to n_final),
  not a check that the adjustment used the experiment's actual data.
- The OF-like spending function keeps its two-sided form even for
  one-sided tests (the literature convention); one-sided alpha=0.025
  reproduces the tabulated K=2 boundaries (2.96, 1.97), two-sided
  alpha=0.05 gives (2.77, ~1.97). Both control their nominal level.
- Evidence levels: the spending framework and CUPED estimator are
  **adaptations** of published methods; the FWER/variance numbers are
  **measured** on seeded synthetic data in this repo (rerun the demo);
  the (2.96, 1.97) cross-check is a **secondary-only** oracle.
