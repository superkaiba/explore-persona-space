---
name: trap-2162-rerun-prereg-vs-realized
description: "#2162-family reruns: Holm m is realized at analysis time (len(pvals) over testable cells — parent realized 25/10/26 vs pre-registered 31/15/28); and any in-window judge slice ≥5k calls must be rule-26 piloted BEFORE dispatch on a model swap"
metadata:
  type: feedback
---

Two recurring traps when a plan inherits the #2162 minimal-pair pipeline
(model-swap reruns, sibling banks):

1. **Pre-registered vs realized Holm m.** The inherited analysis
   (`scripts/issue2162_analysis.py`) computes `holm_family_m = len(pvals)`
   over TESTABLE cells (post-exclusion n ≥ 12). Parent pre-registered
   m = 31/15/28; parent REALIZED m = 25/10/26
   (`eval_results/issue_2162/f_metrics/stats.json` → `families`; body
   Families bullet). A rerun plan that copies 31/15/28 as "realized m" —
   or claims exclusions "shrink n, never m" — contradicts the code it
   inherits; a literal implementer hardcoding m=31 makes Holm ~24%
   stricter at first rank and flips borderline causal-positive verdicts.
   Check the parent stats.json `families` field on every inheritance
   rerun.
2. **Gate-3 anchor-separation sync slice vs rule-26 pilot ordering.** The
   gate-3 slice is ~9.1k judge calls (38 cells × 6 pairs × 10 draws × 2
   rubrics) — above the rule-26 ≥5k pilot floor. On a MODEL SWAP the
   instrument is being pointed at a new output distribution, so the
   pilot must precede the slice, not sit at P6 with the bulk waves; a
   truncation-censored slice false-branches the grid-spend gate.
3. **Committed ρ(F_act, F_beh) values** (for grounding F_act-based
   selection): steered 0.769 / shuffled 0.718 / crosstype 0.517 over
   n=66 screened cell-arms — `figures/issue_2162/act_beh_agreement.meta.json`.
   The ladder round's +0.42 (n=264) is a DIFFERENT bank/DV (F_target).
4. **Re-judging pooled grid completions in a derived read (per-segment /
   per-window re-scoring): check the arm's SLOT + coherence composition
   against the construct.** The ladder grid's steered draws span BOTH
   slots (`install_*|{ce,pe}|steered` cells in
   `eval_results/issue_2162/persona_specificity_ladder/stats.json
   .estimation`), and the realized lattice has EVERY install-pe cell at
   no-clean-transfer — so any derived DV that pools steered draws
   without a slot stratum mixes ~50% known-not-installed completions
   into the "patched persona" arm (plus coherence-failed draws the
   parent's own F excludes via the >60 screen). A patched-vs-prompted
   contrast computed on the pooled arm is mechanically biased toward
   "no difference / patched more persistent". Require slot in the
   derived row grain + the inherited coherence filter (parent per-draw
   coherence scores are already committed — zero new calls).

**Why:** a #2329 rerun plan carried all three (mislabeled m, unpiloted
9.1k in-window slice, a 0.43–0.50 rho citation matching no committed
artifact).
**How to apply:** on any plan citing "Source: #2162" for statistics,
diff each cited statistic against the committed artifact, not the
parent plan prose. Related: [[trap-turn-boundary-count-arithmetic]],
[[trap-value-constrained-donor-null-combinatorics]].

**Validated remedy shapes (round-2 confirmed on the same rerun):**
(1) Holm-m fix = register analysis-time m (the code's `len(pvals)` over
testable n>=12 cells; `SURVIVAL_FLOOR=12` at issue2162_analysis.py L76,
testable gate L484) with 31/15/28 as constructional CEILINGS — NOT a
fixed-m fork: fixed m makes the child strictly more conservative than
the parent's realized correction, a second changed variable in a
transfer read. Exclusion keys on n (pair survival), not the test
statistic, so analysis-time m carries no selection bias; require the
realized-m gap reported wherever cross-run verdicts are compared.
(2) An aggregate bank-level separation HALT demotes to ADVISORY on a
transfer question (per-cell exclusions keep survivors interpretable;
"which types fail" IS the answer), retaining only a justified
instrument-broken catastrophic floor (e.g. <25% cells separable).
