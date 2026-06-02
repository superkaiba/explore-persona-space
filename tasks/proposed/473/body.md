---
title: 'JS vs cosine as a predictor of EM-misalignment leakage (EM analogue of #470)'
kind: experiment
tags: []
created_at: '2026-06-02T20:12:31Z'
has_clean_result: false
parent_id: 470
goal: 'Determine whether persona-to-source Jensen-Shannon divergence between base-model
  persona-conditioned output distributions predicts per-bystander emergent-misalignment
  (EM) leakage better than layer-20 residual cosine — the EM analogue of #470''s sycophancy-leakage
  predictor comparison, using the identical JS-vs-cosine recipe and statistical machinery.'
---
## Goal

Determine whether persona-to-source Jensen-Shannon divergence between base-model persona-conditioned output distributions predicts per-bystander emergent-misalignment (EM) leakage better than layer-20 residual cosine — the EM analogue of #470's sycophancy-leakage predictor comparison, using the identical JS-vs-cosine recipe and statistical machinery.

## Motivation / hypothesis

[#470](https://eps.superkaiba.com/tasks/470) asks whether sequence-level JS divergence between persona-conditioned base-model output distributions predicts #411's per-bystander **sycophancy** leakage better than layer-20 residual cosine (and whether it recovers the anti-cosine leaks cosine misses). This follow-up runs the **same structure and recipe** on the **EM / broad-misalignment** leakage axis instead of sycophancy: does the same distributional persona-distance metric predict where EM trained into a source persona leaks to bystanders?

**Hypothesis (mirrors #470 H1):** JS-similarity between two personas' base-model output distributions attains a higher pooled-cell |ρ| with per-bystander EM-leakage Δ than layer-20 cosine, and recovers anti-cosine EM leaks that cosine cannot. If JS also fails, EM leakage is identity/content-specific rather than a smooth distributional-distance effect.

This is a **predictor-only re-analysis** on already-trained EM cells plus base-model forward passes — **no training** (same discipline as #470).

## Single variable that changes vs #470

> **The dependent variable.** #470's DV is #411's per-(source, bystander) sycophancy-leakage Δ. This follow-up swaps in a per-(source, bystander) **EM / broad-misalignment leakage Δ**. Everything else — the predictors (sequence-level RB JS, both KL directions, persona-vectors response-token cosine, layer-20 cosine baseline), the base model (Qwen-2.5-7B-Instruct), the paired-bootstrap Δρ test, base-rate confound controls, and the anti-cosine-recovery diagnostic — is inherited from #470's recipe unchanged.

## OPEN at clarification — pin the EM-leakage DV source

Unlike #470 (which inherits #411's frozen `per_source.<src>.per_panel_delta` on the `EVAL_PERSONAS_24` panel), there is **no exact EM analogue of #411** with the same 24-persona panel + a stored per-bystander leakage Δ. The clarifier/planner must pin the DV source before this can be planned. Candidates found in `eval_results/INDEX.md` + `RESULTS.md`:

- `directed_trait_transfer/contrastive_em/` — contrastive EM, persona-specific misalignment, 4 conditions; "no proximity transfer" (asst_near −3.9pt, p=0.23).
- `trait_transfer_em/` — contrastive EM on the trait-transfer grid (2 arms × 3 conditions); Arm 1 suggestive gradient r=0.78 p=0.014 but nothing survives Bonferroni; Arm 2 floor effect.
- `leakage_experiment/` Phase A2 — structure + misalignment traits (44 conditions); **misalignment REVERSE gradient** ρ=−0.59 p=0.01 (closer = less leakage) — the most striking EM-leakage signal on record.
- `a3b_factorial/` — 2×2 factorial incl. misalignment variant; contrastive design (not hyperparams) determines leakage pattern.
- `js_divergence/` (#142) — an existing JS/KL-vs-cosine comparison (11 personas, base model; JS ρ=−0.75 vs cosine 0.57, n=50 matched pairs). Likely the closest prior art; the planner should position against it and decide whether #470's canonical sequence-level RB-JS estimator + paired-Δρ machinery materially improves on it.

**Decision the planner must make:** (a) reuse one of the above as the frozen EM-leakage DV after confirming its panel + per-bystander structure is compatible with #470's machinery; OR (b) if no compatible EM-leakage panel exists, scope a prerequisite EM-implantation-with-bystander-panel experiment (contrastive corrective negatives, `EVAL_PERSONAS_24` panel, Betley broad-misalignment judge rate as the per-cell DV) that mirrors #411's design before the predictor comparison can run. Option (b) makes this a 2-task line, not a pure re-analysis.

## What's reused vs new

- **Reused (from #470, once #470 lands):** the entire predictor-extraction pipeline (`predictor_jsdiv_470` module — Phase 1 sampling, Phase 2 response-token cosine, Phase 3 RB JS + both-KL, Phase 5 regression with paired-bootstrap Δρ + base-rate partials), the canonical predictor definitions, the figure recipes.
- **New:** the EM-leakage DV loader (points at whichever EM experiment §"OPEN" pins), the EM-specific panel/probe set if it differs from #411's, and the predictor-comparison run + figures on the EM DV.

## Method (mirrors #470)

Base-model (Qwen-2.5-7B-Instruct) forward passes only; no LoRA, no training. Adapt #470's `predictor_jsdiv_470` pipeline to the pinned EM-leakage panel + probes. Per-source and pooled Spearman ρ (3 variants: raw / source-FE / base-rate-partial), paired-bootstrap Δρ = |ρ_JS| − |ρ_cosine| on shared cells, base-rate baselines, anti-cosine-recovery diagnostic, bootstrap CIs + permutation p-values.

## Success criteria / read

- Per-source and pooled Spearman ρ for JS-similarity vs EM-leakage Δ, side-by-side with cosine.
- Primary verdict: does JS beat cosine pooled (paired Δρ CI excludes 0), and does it recover EM's anti-cosine leaks?
- If JS wins on EM too → distributional distance is the general leakage metric across behavior types (sycophancy AND EM). If JS wins on sycophancy (#470) but not EM (or vice-versa) → the right persona-distance metric is behavior-type-specific. If JS fails on both → leakage is identity/content-specific, not a smooth distance effect.

## Compute estimate

1× H100, no training, base-model forward passes only — same order as #470 (~5 GPU-h), modulo the EM panel size once pinned. If §"OPEN" lands on option (b) (prerequisite EM-implantation run), add that experiment's training cost separately.

## Dependencies

- **Blocks on #470** for the reusable `predictor_jsdiv_470` pipeline (don't re-implement; inherit).
- **Blocks on the §"OPEN" DV decision** at clarification.
