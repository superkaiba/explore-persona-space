# Issue #779 — can the pre-generation read predict hallucination on GENERIC prompts?

**Ask (chat 2026-07-14):** "do we have evaluation on generic prompts?" Hallucination
is the only trait with real variance on generic LMSYS prompts (24.6% of 4593 judged
prompts score >50; evil 0.2%, sycophancy 6.0% — too sparse to evaluate). So this is
the one place the question is even answerable. Test: does the pre-generation context
state at L17 predict the per-context LMSYS hallucination judge label?

## Verdict: no. All three reads are at chance — a clean null.

| read | Pearson [95% CI] | Spearman | AUROC (>50) | decile mean-label (bottom→top) | top−bottom |
|---|---|---|---|---|---|
| pv_raw `<c, r_B>` (original PV method) | **−0.030** [−0.058, −0.000] | −0.016 | 0.479 | 25.9 → 20.3 | **−5.6** |
| map read `<h(c), r_B>` (5-fold CV) | −0.008 [−0.036, +0.021] | −0.004 | 0.493 | 22.6 → 22.5 | −0.1 |
| direct probe `c → label` (5-fold CV) | −0.001 [−0.031, +0.027] | +0.002 | 0.499 | 25.0 → 24.3 | −0.7 |

n = 4593 judged contexts. Positive rate (label > 50) = 24.6%.

## Reading

- **None of the reads carry signal.** Every AUROC is ≈0.50, every Pearson CI includes
  (or sits at) zero, and the decile mean-label curves are flat or slightly *inverted* —
  ranking generic prompts by any of these reads does not sort them by how much the
  answer hallucinated. The persona-vector projection is even marginally anti-correlated
  (top decile 20.3 vs bottom 25.9).
- **The supervised probe is the decisive part.** `probe_g_cv` is fit *directly on the
  hallucination labels* (5-fold held-out ridge) — the best case for finding decodable
  signal — and it lands at Pearson −0.001, AUROC 0.499. So this is not a
  "wrong-direction" failure fixable by a better read-out: there is no linearly
  decodable generic-prompt hallucination signal in the pre-generation context state at
  all. This **reproduces and generalizes #1092's within-LMSYS hallucination floor
  (cross-validated r 0.009)** — the floor holds for the raw PV direction and the map
  read too, not just #1092's probe.
- **Contrast with the crafted rig.** On the Persona-Vectors eval grid these same reads
  score 0.5–0.8 (and persona-level averaging reaches 0.53 for hallucination). That
  signal is a property of the *constructed trait-eliciting system prompts*, which
  deterministically push the trait — it does not survive the move to naturalistic
  one-off prompts where the trait is a generation-time outcome.

## Caveat (mechanism is ambiguous, conclusion is not)

The LMSYS labels are **1 rollout × 5 judge draws, stored as the mean only**. A large
share of a single rollout's hallucination label is generation stochasticity — whether
*this* sampled answer happened to fabricate — which no pre-generation read could predict
even in principle. With no per-draw scores and only one rollout, the judge-reliability
and generation-noise ceilings are not estimable from this file. So I cannot separate
"the state does not encode generic hallucination propensity" from "the single-rollout
label is too noisy to have any decodable target." Both point the same way for the
practical question (you cannot monitor generic-prompt hallucination from the
pre-generation state with these labels), but the mechanism is open. A clean version
would need k≥5 rollouts per prompt so the per-prompt hallucination *rate* (a stabler
target) could be predicted, and a judge-reliability ceiling computed.

## Artifacts
- `generic_prompt_hallucination_monitor.py` (script), `generic_prompt_hallucination_monitor.json`
  (all numbers: per-read Pearson/Spearman bootstrap CIs, AUROC, full decile curves).
- Reuses arm_headline GramRidge/loaders; pass_b LMSYS bundle (local), r_B @ L17,
  lmsys_g_labels. 0 GPU-h, VM CPU.
