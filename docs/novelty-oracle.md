# The Novelty Oracle — Formal Specification

**Status:** Law Article 1, implemented in `src/novelty_oracle/`.
**Companion:** stranger-runnable demo (`python -m novelty_oracle.demo`),
falsifiable check (`python -m novelty_oracle.heldout`),
investigation (`python -m novelty_oracle.eda`).

This document states the mathematics. The code is the implementation;
where they disagree, this document wins and the code is a bug.

---

## 1. Definitions

We work in the single-parameter auction domain: one indivisible item,
$n$ bidders, each with a private value $v_i \geq 0$. Bidder model is
**direct revelation with truthful bidding** (values are fed as bids;
strategic bid-shading is out of scope — the oracle compares mechanisms
*as functions*, not equilibria).

**Definition 1 (Mechanism).** A mechanism $M = (x, p)$ where

- $x: \mathbb{R}^n_+ \to \{0,1\}^n$, $\sum_i x_i(v) \le 1$ (allocation rule),
- $p: \mathbb{R}^n_+ \to \mathbb{R}^n_+$ (payment rule).

In code: `allocate: tuple[float, ...] -> tuple[int, ...]`,
`pay: (tuple[float, ...], tuple[int, ...]) -> tuple[float, ...]`.

**Assumptions (enforced, not assumed silently).**

- **A1 — Determinism.** Same input $\Rightarrow$ same output. The oracle
  probes every mechanism twice per profile; violations raise
  `NonDeterministicMechanism`.
- **A2 — Validity.** Allocation entries in $\{0,1\}$, at most one winner,
  payments non-negative and finite. Violations raise `InvalidOutcome`.
- **A3 — Truthful bidding.** Values are bids. (Stated, not enforced —
  it is the interpretation of the measurement.)
- **A4 — Battery mixture.** The distribution of interest is the canonical
  battery's valuation mixture (§2). Quality and novelty are expectations
  under this mixture, nothing else.
- **A5 — Seed independence.** Per-seed RNG streams (`base_seed + s`) are
  disjoint; seed-means are i.i.d.
- **A6 — CLT approximation.** 95% intervals use the normal approximation
  over seed means. Approximate, not exact — see Theorem 2's error term.

**Definition 2 (Behavioral distance).** Given battery
$B = \{v^{(1)}, \dots, v^{(N)}\}$ and tolerance $\varepsilon > 0$:

$$d_\varepsilon(M_1, M_2) = \frac{1}{N}\sum_{i=1}^N
\mathbf{1}\big[(x_1(v^{(i)}), p_1(v^{(i)})) \neq_\varepsilon
(x_2(v^{(i)}), p_2(v^{(i)}))\big]$$

where $\neq_\varepsilon$ means: allocation differs, or some payment
differs by more than $\varepsilon$. Default $\varepsilon = 10^{-9}$
(numerical tolerance, not a tuning knob).

**Definition 3 (Novelty score).** Against a corpus of baselines
$\mathcal{B}$:

$$\nu(M) = \min_{b \in \mathcal{B}} d_\varepsilon(M, b)$$

Novelty is distance to the *nearest known mechanism*. AST distance is a
pre-filter only: identical ASTs short-circuit to $d = 0$; it never
increases or decreases a score.

**Definition 4 (Quality delta).** Let $R(M)$ be total payments under the
battery mixture. With $\Delta(M) = \mathbb{E}[R(M)] -
\max_{b \in \mathcal{B}} \mathbb{E}[R(b)]$, the oracle reports
$\hat\Delta$ with a 95% CI from $S$ independent seeds.

**Definition 5 (Verdict).** With threshold $\tau = 0.05$:

- **INVENTION** iff $\mathrm{CI}_{low}(\hat\Delta) > 0$ **and**
  $\nu(M) \ge \tau$;
- **ABSTAIN** iff $\nu(M) \ge \tau$ **and** $0 \in \mathrm{CI}(\hat\Delta)$
  (novel, but quality unresolved — never a guess);
- **NOT_INVENTION** otherwise.

Note the asymmetry is deliberate: *novel but worse* $\to$ NOT_INVENTION
(novelty is not importance); *better but not novel* $\to$
NOT_INVENTION (a variant, not an invention); a behaviorally-identical
copy $\to$ NOT_INVENTION even when its quality CI straddles zero —
novelty is decided exactly, so there is nothing to abstain about.

---

## 2. The canonical battery

200 profiles, versioned (`BATTERY_VERSION = 1`), construction fully
deterministic from `CONSTRUCTION_SEED`. Categories and counts:

| category | n | rationale |
|---|---|---|
| iid-uniform | 120 | regular i.i.d. values: revenue-optimality is known analytically (Myerson) |
| asymmetric | 30 | bidders from different distributions: exposes symmetric-only "novelty" |
| tie | 10 | exact ties: where tie-breaking implementations diverge |
| dominant | 10 | one bidder ~10x the rest: reserve/participation behavior |
| zero | 5 | zero values: edge behavior |
| skewed-beta | 25 | non-uniform regular: punishes uniform-tuned mechanisms |

The simulator samples the *same mixture* — this is what makes
"simulated ranking matches brute-force battery ranking" a legitimate
falsifiable check rather than a category error.

---

## 3. Theorems

### Theorem 1 — Behavioral distance is a pseudometric at $\varepsilon = 0$; not in general for $\varepsilon > 0$.

*Proof.* At $\varepsilon = 0$, per-profile agreement is exact equality
of outcomes — an equivalence relation. The disagreement indicator
$\delta_i(A,B) \in \{0,1\}$ satisfies
$\delta_i(A,C) \le \delta_i(A,B) + \delta_i(B,C)$: if $A =_i C$ the left
side is 0; if $A \neq_i C$, transitivity of equality forces $A \neq_i B$
or $B \neq_i C$. Summing over $i$ and normalizing by $N$ preserves the
inequality. Non-negativity and symmetry are immediate. Identity of
indiscernibles fails — distinct mechanisms can agree on all of $B$ —
hence *pseudo*metric, not metric. ∎

*Where it breaks.* For $\varepsilon > 0$, transitivity of
"$\varepsilon$-agreement" fails. Counterexample on one profile with
payments $0$, $0.6\varepsilon$, $1.2\varepsilon$ for mechanisms
$A, B, C$: $d_\varepsilon(A,B) = 0$, $d_\varepsilon(B,C) = 0$, but
$d_\varepsilon(A,C) = 1 > 0 + 0$.

*Why it doesn't matter.* The oracle uses $d_\varepsilon$ only for
nearest-baseline thresholding ($\nu \gtreqless \tau$); no construction
needs the triangle inequality (no clustering, no metric search).
Violations require payment differences in the $(\varepsilon, 2\varepsilon]$
band — dust at $10^{-9}$ scale. Dust can inflate *novelty* (see the EDA's
$\varepsilon$-band attack) but produces quality deltas of the same order,
which no feasible seed budget can resolve through the quality gate.
The guarantee is defense in depth across the two gates, not a single
metric axiom. ∎

### Theorem 2 — False-invention control.

*Claim.* Under the quality null $H_0^q$ = {$\nu(M) \ge \tau$ (novel) $\cap$
true $\Delta \le 0$}:

$$P(\text{verdict} = \text{INVENTION} \mid H_0^q) \le 2.5\% +
\text{CLT approximation error}.$$

Under the full null $H_0$ = {behaviorally identical to some baseline}
$\cap$ {true $\Delta \le 0$}, the probability is **exactly 0**: novelty
is decided exactly ($\nu = 0$ deterministically — identical functions
give identical outcomes on every profile, and float rounding below
$\varepsilon$ is absorbed by the tolerance), so a copy never reaches the
quality gate at all.

*Proof (quality null).* It remains to bound the quality gate:
$P(\mathrm{CI}_{low}(\hat\Delta) > 0 \mid \Delta \le 0)$. The CI is a
two-sided 95% normal interval over i.i.d. seed means (A5); under
$\Delta = 0$, $P(\mathrm{CI}_{low} > 0) \approx 2.5\%$ one-sided, and the
probability is monotone decreasing in $\Delta$. One subtlety: the
candidate is compared against the *argmax* baseline, whose estimated
mean is upward-biased by selection — this bias works *against* the
candidate, making the gate conservative, never liberal. The "CLT
approximation error" term covers A6; seed-mean distributions here are
bounded-payment averages over 1000+ draws, well inside CLT territory. ∎

*What is NOT guaranteed.* Power. A genuinely better mechanism with a
small effect ABSTAINs at feasible seed budgets — the procedure is
conservative by design, and ABSTAIN is the price of the guarantee.
Empirical validation: EDA experiment 5 (40 noisy-but-novel variants —
novelty $\approx$ 1, true $\Delta = 0$ — count the false INVENTIONs).

### Theorem 3 — Anti-gaming: syntactic obfuscation cannot inflate novelty.

*Claim.* If $\llbracket C \rrbracket = \llbracket C' \rrbracket$ as
functions (identical allocation and payments for every input), then
$\nu(C') = \nu(C)$.

*Proof.* $d_\varepsilon$ is defined purely over input-output pairs on
the battery. If the functions agree on all inputs, they agree on all
battery profiles, so $d_\varepsilon(C', b) = d_\varepsilon(C, b)$ for
every baseline $b$, and the minimum over $b$ is unchanged. Variable
renames, control-flow restructuring, and dead code do not appear in the
definition — there is nothing for them to act on. ∎

*Honest boundary.* This proves invariance under *semantics-preserving*
transformations. It does not bound behaviorally-near copies (a
$10^{-4}$ payment bump is genuinely novel, $\nu \approx 1$ — correctly
measured as such; the quality gate is what stops it from becoming an
INVENTION). Empirical demonstration: the held-out obfuscated pair
scores $\nu = 0.000$.

---

## 4. Literature

**Myerson (1981), "Optimal Auction Design."** *Take:* the analytic
optimum for regular i.i.d. values — we instantiate it as
second-price-with-reserve-0.5, the ground-truth baseline, and as the
Article-2 sanity check (no candidate may beat it on iid-uniform; if one
does, the simulator is wrong). *Reject:* Myerson derives the optimum
*given known priors*; the oracle's job is comparative measurement of
arbitrary mechanisms with no regularity or prior assumptions on the
candidate.

**Gneiting & Raftery (2007), "Strictly Proper Scoring Rules."**
*Take:* the discipline of never reporting a point estimate without its
uncertainty — our quality gate reports $\hat\Delta$ only with its CI,
and the verdict's ABSTAIN state exists precisely because a point
estimate without an interval is not evidence. (This is the same
discipline as the company's Brier-scored prediction market.)
*Reject:* proper scoring rules score *probabilistic forecasts* against
realizations; our estimand is a Monte-Carlo expectation difference, so
we use CI-based dominance instead of proper scores — a different tool
for a different estimand.

**Sandholm (2003), "Automated Mechanism Design."** *Take:* the framing
of mechanism search as optimization — the oracle is the missing
*measurement* half of that loop; you cannot optimize what you cannot
measure, and AMD's revenue objective cannot see novelty at all.
*Reject:* classical AMD optimizes within a fixed parameterized family
under known priors; our novelty axis explicitly rewards *leaving* the
family, which a pure revenue objective cannot express.

**Dütting, Feng, Narasimhan, Parkes & Ravindranath (2019), "Optimal
Auctions through Deep Learning" (RegretNet).** *Take:* the demonstration
that learned mechanisms can approach the Myerson optimum — the most
likely producer of future candidates, and exactly the case where the
question "is it new or a re-discovery?" bites. *Reject:* RegretNet
optimizes revenue subject to learned incentive constraints on a fixed
architecture; our contribution is orthogonal — measurement against a
corpus, not training.

---

## 5. What the oracle is not

- Not a cross-domain creativity metric. It measures mechanisms in the
  single-parameter auction domain.
- Not an equilibrium analysis. Truthful bidding (A3); shading is a
  different paper.
- Not a substitute for the Article-2 experiment. The oracle is the
  judge; the experiment is the trial.
