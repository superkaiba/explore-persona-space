---
title: 'Persona Vectors screening predictors: mapped context→answer ΔP vs raw / exact
  / prompt-token projections on the synthetic suite'
kind: experiment
tags: []
created_at: '2026-08-10T21:18:27Z'
has_clean_result: false
parent_id: 2221
workflow: v1
goal: Reproduce the Persona Vectors (arXiv 2507.21509) data-screening predictor comparison
  (raw projection / exact projection difference / last-prompt-token approximation,
  their Figures 8/21/23) on the 24-dataset synthetic suite and test whether a FROZEN
  trait-agnostic context→answer mapped projection difference recovers exact-ΔP predictivity
  — especially closing the sycophancy gap (r 0.581 → ~0.88) — at forward-pass cost.
relates_to:
- app5
- spec-context-as-vector
---
# Persona Vectors screening predictors: mapped context→answer ΔP vs raw / exact / prompt-token projections on the synthetic suite

## Goal

Reproduce the Persona Vectors (arXiv 2507.21509) data-screening predictor comparison (raw projection / exact projection difference / last-prompt-token approximation, their Figures 8/21/23) on the 24-dataset synthetic suite and test whether a FROZEN trait-agnostic context→answer mapped projection difference recovers exact-ΔP predictivity — especially closing the sycophancy gap (r 0.581 → ~0.88) — at forward-pass cost.

## Design

**Program context.** Experiment 3 of the Persona Vectors (arXiv 2507.21509) reproduce-and-beat program. Reproduces the paper's three published data-screening predictors (§6.1, Appendix H, Appendix I — their Figures 8/21/23) on the 24-dataset synthetic suite and adds the mapped arm. Depends on the parent (#2221) reproduction stratum: the synthetic finetunes and their post-finetuning trait scores are the shared ground-truth y-axis, so this experiment's marginal compute is forward passes only.

**Ground truth (y-axis):** post-finetuning judge-scored on-policy trait expression per finetuned model (graded 0–100 primary + rate companion), shared with #2221.

**Predictor arms (x-axis), per trait, computed on the BASE model:**

| Arm | Definition | Cost | Paper's Qwen r (evil / sycophancy / hallucination) |
|---|---|---|---|
| Raw projection | mean response-token activation of the training response, dotted with the unit persona direction | forward passes | 0.784 / 0.540 / 0.635 |
| Exact ΔP (paper §6.1) | raw projection minus the projection of the base model's own generated response to the same prompt | one generation per prompt | 0.946 / 0.879 / 0.616 |
| Prompt-token ΔP (paper App. I) | raw projection minus the last-prompt-token projection | forward passes | 0.931 / 0.581 / 0.689 |
| Mapped ΔP (ours) | raw projection minus the projection of M(v_C(x)) — the answer representation predicted from the context vector by the frozen trait-agnostic base-model context→answer map | forward passes | — |

**Headline hypothesis:** mapped ΔP ≈ exact ΔP across traits — in particular closing the sycophancy gap (0.581 → ~0.88) — at prompt-token-approximation cost. Framing: the paper's prompt-token approximation is an informal identity-map assumption ("last-prompt-token projection ≈ base-generation projection"); the mandatory identity+learned-bias baseline is its formalized version, so the baseline table doubles as the scientific comparison. The tested claim: a learned context→answer map beats the identity assumption exactly where the identity assumption is known to fail.

**Map regime (user decision, 2026-08-10):** FROZEN-ONLY PRIMARY — M fitted once, trait-agnostically, on a generic real-corpus slice from the base model, then frozen for every trait and arm (the honest practicality claim). A per-trait-tuned variant (map selected/weighted by trait predictivity) runs only as a clearly LABELED EXPLORATORY cell, never pooled with the primary.

**Direct-regression arm (user addition, 2026-08-10 — "can we also train some kind of direct regression as another arm?"):** LINEAR ridge only (project linear-by-default rule). Two forms:
- **Form A (primary, well-posed):** sample-level probe — ridge from base-model response-avg representation → graded judge trait score, trained on the program's judged pool (n ~50–100k per trait >> d). Screens in the same difference grid: probe(training response) − probe(stand-in), stand-ins = generated / last-prompt-token / mapped. The probe∘map composition is a single linear functional on the context vector (pure context-vector screening — the "just the context vector" read). Ridge via the dof-capped shared fit cores; selected-λ diagnostics reported; group-level LOFO folds across dataset families.
- **Form B (EXPLORATORY only):** dataset-level outcome regression (aggregate dataset features → post-finetuning trait shift). n = 24 datasets << d ⇒ estimator-degenerate in full dim (estimator-validity rule): runs only dim-reduced + dof-capped + LOFO, clearly labeled, never headlines. Report the angle between the Form-B learned direction and the contrastive persona vector regardless.
- **Supervision ledger per arm (stated in every results table setup line):** frozen map = trait-agnostic; persona vector = trait description + judge filter; probe = judge labels. Probe vs persona-vector is the fair learned-read-out vs contrastive-mean-diff comparison.

**Sample-level screening:** the paper's separability histograms (trait-inducing vs Normal samples, their Figure 9) recomputed as ROC/AUC per predictor arm — rankable numbers instead of eyeballed histograms.

**Design details:**
- BOTH mapping arms: prefix-based AND context-based (standing rule; prefix = everything before the user query, context = prefix + query).
- Identity-family baseline incl. learned-bias form AND kNN-retrieval read reported for every fitted map (standing rule).
- Layer selection by read-out-regime predictivity sweep (persona-vectors-recipe.md step 7), with one ablation cell at the paper's steering-selected layer — the paper used the steering layer for screening, itself a candidate source of their sycophancy weakness.
- Selection-symmetric nulls for any best-layer / best-arm claim over a free axis.
- Per-arm provenance stated in every results table/figure setup line (teacher-forced training responses vs generated base responses).

**Compute shape (plan-time to refine):** ~0 GPU-h beyond #2221's finetunes for the mapped/prompt-token/raw arms (teacher-forced forward passes over the suite); the exact-ΔP arm needs one base-model generation per training prompt (vLLM batched, the suite is ~50–80k prompts total — a few GPU-hours). Artifact-reuse inventory at plan time: #2221's finetunes and any prior fitted context→answer maps.

## Scope caveats (carried to the clean-result)

- Entirely on the paper's synthetic (tier-3, Claude-written-response) suite by design — this is the controlled reproduction stratum; the realistic version is Experiment 4 of the program.
- Frozen-map primary means the map's fitting corpus (generic, trait-agnostic) is a named design choice; the exploratory tuned cell is reported separately and never headlines.

## Provenance

Verbatim originating prompts (user, 2026-08-10):
- "i want to reproduce all the persona vectors experiments but show that we can do better with our mapping (especially on REALISTIC and not SYNTHETIC data) or by using just the context vector instead of the answer vectors. what would this look like? let's go one experiment at a time"
- "My lean is frozen-only as primary, tuned as a labeled exploratory cell. -> looks good"
- "can we also train some kind of direct regression as another arm?"
