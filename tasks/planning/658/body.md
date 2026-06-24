---
title: Base-model foundation for the leakage predictor on Qwen2.5-7B (A3.2/A3.3/A3.5
  + recipe lock)
kind: experiment
tags: []
created_at: '2026-06-24T08:21:29Z'
has_clean_result: false
origin_prompt: Just run on qwen2.5 7b
goal: Determine whether the leakage-predictor's base-model chain holds on Qwen2.5-7B-Instruct
  — A3.2 (mean answer-side activation v0(C) summarizes behavior expression), A3.3
  (linear read-out r_B), A3.4/A3.5 (context vector c_C predicts the answer profile
  v0(C)) — AND whether they still hold in the single-context edge case C=delta_x (one
  prompt, reported vs a within-context noise floor, Phase 1b); lock the best extraction
  layer + summary recipe for downstream phases.
relates_to:
- leak-predictor
---
## Goal

Determine whether the leakage-predictor's base-model chain holds on Qwen2.5-7B-Instruct — A3.2 (mean answer-side activation v0(C) summarizes behavior expression), A3.3 (linear read-out r_B), A3.4/A3.5 (context vector c_C predicts the answer profile v0(C)) — AND whether they still hold in the single-context edge case C=delta_x (one prompt, reported vs a within-context noise floor, Phase 1b); lock the best extraction layer + summary recipe for downstream phases.

- **A3.2** — the mean answer-side residual activation `v0(C)` is a sufficient
  summary of the context-induced profile for predicting behavior expression.
- **A3.3** — each behavior reads out linearly: `E0(C,B) ≈ r_Bᵀ v0(C)`.
- **A3.4/A3.5** — a pre-fine-tuning context vector `c_C` predicts the answer-side
  profile `v0(C)` (linear map `M` + nonlinear MLP).

PASS = predictors beat the predict-mean baseline above the measurement noise
floor; primary deliverable = the per-behavior best layer/summary recipe.

**Also test the single-context edge case** (`C = δ_x`, one prompt): do A3.2/A3.3/
A3.5 still hold per-prompt — the deployment-relevant, hardest-stress variant (no
averaging over `x∼C`) — reported against a **within-context noise floor** (Phase
1b below)?

## Grounding — READ IN FULL BEFORE PLANNING (mandatory)

The planner AND implementer must read these IN FULL before designing anything —
do NOT design from the concise summary alone:

1. **The FULL theory paper** — `docs/leakage_theory_paper.tex` (pinned snapshot;
   canonical live source `~/overleaf-6a2df2d2/main.tex`). Read **§2 Definitions,
   §3 Assumptions (ALL ten assumption blocks A3.1–A3.10, each with its
   Confidence / Testable / "Worth testing now" notes — NOT just the TLDR A1–A7
   summary at the top), §4 cosine relation, §5 Evaluation methodology, §6 worked
   computation, §7 case studies.** The detailed §3 blocks (esp. A3.2, A3.3,
   A3.4/A3.5 here) specify the EXACT per-assumption test to implement; the TLDR
   is not sufficient.
2. **The campaign plan** — `docs/theory_assumption_test_plan.md` (this task IS
   Phase 0 + Phase 1; inherit §1 config, §2 reuse architecture, §7 decisions).
3. **Past issues** for reusable artifacts/recipes (apply `.claude/rules/artifact-reuse.md`):
   context suite #594/#617/#634/#650; behavior testbed #545; predictor line
   #404/#458/#474/#532/#541; marker recipe #530/#601; on-policy #612; activation
   tensors #521/#551.

Ground every load-bearing hyperparameter against the theory doc + a past issue
(plan §11 discipline).

## Scope — Phase 0 + Phase 1 of the leakage-predictor program

**Model: `Qwen/Qwen2.5-7B-Instruct` only** (scale checks deferred). No
fine-tuning in this task — base model `θ0` only.

### Phase 0 — build + publish the shared activation store (reusable asset)
Reuse existing infra, don't rebuild:
- `scripts/issue594_build_battery.py` → the 50-context / 7-family context battery
  (`data/issue594/battery.json`); `issue594_common.load_battery()`.
- `scripts/issue594_extract_context_vectors.py` → context vectors `c_C` (all 28
  layers, last-input-token slot; mean-over-prompt as ablation).
- Add answer-side `v0(C)` extraction (mean answer-token residual, all 28 layers)
  over the battery × the **48 Betley probes** (`issue404_common.fetch_preregistered_probes()`).
- Behavior read-outs `r_B` = difference-in-means on each #545 column's D_B/D_{B̄}
  (`behavior_testbed_545`); base expression scores `E0(C,B)` over the #545 eval
  registry (judge = `claude-sonnet-4-5-20250929`, Anthropic Batch API).
- `Σ_c` over a broad background corpus (≥2–5k contexts) via the same extractor.
- **Single-context granularity (REQUIRED for Phase 1b):** sample **R≥8**
  completions per (context × probe) at temp 1.0 and **retain per-probe, per-sample
  activations + judge labels** (not just per-condition means). The single-context
  edge case reads these; capture it now — it cannot be recovered post-hoc.
**Publish the store to HF** `superkaiba1/explore-persona-space-data/theory_assumptions/qwen2.5-7b/...`
with a sha-pinned manifest — it is the campaign-level asset reused by Phases 2–4
(artifact-reuse checklist); do NOT bury it as a throwaway.

### Phase 1 — test the base-only assumptions + lock the recipe
- A3.2: train a small MLP `v0(C) → E0(C,B)` per behavior (incl. the localized
  marker); sweep all 28 layers; compare to predict-mean.
- A3.3: fit `r_B` (diff-in-means default; ablate mean-of-D_B, few-shot-final);
  test `E0(C,B) ≈ r_Bᵀ v0(C)` on held-out C, per layer.
- A3.4/A3.5: linear `M` + MLP for `c_C → v0(C)`; report linear-vs-nonlinear gap +
  best `c_C` recipe/layer.
- Metrics: Spearman (primary) + Pearson; **noise floor** from independent probe
  resamples; **leave-one-behavior-out** split. Per-behavior best layer reported.

### Phase 1b — single-context edge case (C = δ_x)
Repeat **A3.2 / A3.3 / A3.4-A3.5 at single-context granularity**: treat each
(battery context × probe) as its own `δ_x` (50×48 = 2400 single contexts), with
`v0(δ_x)` = mean over the R sampled answers' activations and `E0(δ_x,B)` = the
within-prompt judged rate over the R samples.
- **Continuous DV is PRIMARY here** (a per-prompt rate quantizes to a
  low-resolution Bernoulli; the continuous completion-probability DV keeps dynamic
  range).
- Estimate a **within-context noise floor** from independent R-sample splits and
  report every single-context correlation AGAINST it. A low single-context ρ that
  sits within the floor is **measurement noise, NOT a falsified assumption** — say
  so explicitly; do not report a single-context "break" without clearing the floor.
- Compare per-prompt (`δ_x`) vs distributional (`C`) results: do the assumptions
  degrade gracefully or break in the single-context limit?
CPU re-analysis on the Phase-0 store — no new GPU. (Full spec: plan §1.10.)

### Deliverable + gate
The locked extraction recipe (layer + summary, per behavior) + the published
store + a clean-result on whether the base-model chain holds. **This GATES the
fine-tune fleet (Phase 2):** if A3.2/A3.3 fail above the noise floor, HALT and
surface — do not commit fleet budget on a broken foundation (plan §8).

## Provenance
Originating prompt: "Just run on qwen2.5 7b" + "Continue running autonomously to
test the assumptions" — re: the comprehensive theory-assumption-testing plan at
`docs/theory_assumption_test_plan.md`. First (cheap, ~5–8 GPU-h) phase of the
sequenced `/issue --auto` rollout (no campaign); Phases 2–4 are filed by the
orchestrator only after this PASSes.
