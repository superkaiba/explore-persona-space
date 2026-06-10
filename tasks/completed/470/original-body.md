---
title: 'KL/JS divergence vs cosine as a predictor of #411 sycophancy leakage'
kind: experiment
tags: []
created_at: '2026-06-02T19:39:19Z'
has_clean_result: false
parent_id: 411
goal: 'Determine whether persona-to-source Jensen-Shannon divergence between base-model
  persona-conditioned output distributions predicts #411''s per-bystander sycophancy
  leakage better than layer-20 residual cosine, in particular whether it recovers
  the strongest anti-cosine leak (software_engineer leaking +0.48 to comedian at its
  lowest cosine rank).'
relates_to:
- leak-predictor
---
## Goal

Determine whether persona-to-source Jensen-Shannon divergence between base-model persona-conditioned output distributions predicts #411's per-bystander sycophancy leakage better than layer-20 residual cosine, in particular whether it recovers the strongest anti-cosine leak (software_engineer leaking +0.48 to comedian at its lowest cosine rank).


## Motivation / hypothesis

[#411](https://eps.superkaiba.com/tasks/411) trained sycophancy (agreement with false claims) into 6 source personas with opposite-behavior corrective negatives and measured per-bystander leakage on held-out claims, using **layer-20 residual cosine** to the source as the only persona-distance metric. Cosine predicted leakage poorly and inconsistently: 4 of 6 sources showed near-zero bystander leakage, `assistant` leaked to its 3 nearest personas (cosine-consistent), but `software_engineer` leaked to 15 of 23 bystanders across the whole cosine range — its single strongest leak was **comedian (Δ=+0.478) at cosine-rank 23/23 (its FARTHEST persona, cos=0.766)**. So cosine misses the leaks that matter most.

**Hypothesis:** Jensen-Shannon divergence between two personas' base-model output distributions predicts the observed sycophancy leakage better than residual cosine — specifically, it recovers the anti-cosine leaks (e.g. `software_engineer → comedian`) that cosine cannot. If JS *also* fails on those leaks, then no geometric/distributional persona-distance explains behavioral leakage and the spillover is identity/content-specific.

This is a **predictor-only re-analysis** on the already-trained #411 cells plus base-model forward passes — **no training**.

## Dependent variable (reused from #411, fixed)

Per (source, bystander) cell: the per-bystander sycophancy leakage Δ-vs-base already computed in #411 (`eval_results/issue_411/analyze_summary.json` → `per_panel_delta`), over the 6 sources × 23 bystanders = 138 cells. This DV is frozen — the follow-up changes only the *predictor*, not the outcome being predicted (single-variable change vs #411).

## Predictors to compare

1. **Cosine (baseline, already computed in #411):** layer-20 residual centroid cosine, last-prompt-token recipe. Reuse `per_panel_cosine_to_source`.
2. **JS divergence (new):** per the canonical persona-distance definition in `CLAUDE.md § Persona-distance metrics` — the Rao-Blackwellized sequence-level estimator (Zhang et al. 2025, arXiv 2504.10637). For each (source S, bystander B): sample R≈8 responses per probe (temp=1, ≤256 tok) from the base model conditioned on S's persona prompt and on B's persona prompt over #411's 50 held-out wrong-claim probes; teacher-force each sampled response through BOTH conditioned models; at every response-token position compute exact full-vocab JS between the two next-token distributions; length-normalize, average over samples/probes. Headline = JS (base-2, bounded [0,1]); polarity-align to similarity `M_js = 1 − JS`. Also report both KL directions + symmetric-KL (the asymmetry is diagnostic).
3. **(Secondary) Persona-vectors cosine, response-token recipe:** mean-pool residual activations over each persona's OWN generated response tokens (recipe (b) in the canonical def), layers {7,14,21,27}. Tests whether #411's weak cosine was partly a last-prompt-token-recipe artifact.

## Method

Base-model (Qwen-2.5-7B-Instruct) forward passes only; no LoRA, no training. Adapt `scripts/issue458_predictor_jsdiv.py` (canonical JS predictor) to compute persona-to-source JS over the #411 panel + held-out probes; reuse `scripts/issue404_predictor_cossim.py` for recipe (b) cosine. Compute per-source Spearman ρ between each predictor-similarity and the frozen leakage Δ, plus a pooled ρ over all 138 cells (with source fixed effects), bootstrap CIs + permutation p-values matching #411's analysis settings.

## Success criteria / read

- Per-source and pooled Spearman ρ for JS-similarity vs leakage Δ, side-by-side with cosine's ρ.
- **Primary verdict:** does JS predict leakage better than cosine pooled, and does it recover the `software_engineer → comedian` anti-cosine leak (and `software_engineer`'s broad spillover generally)?
- If JS wins → distributional distance is the right leakage metric and cosine was the wrong proxy. If JS also fails on the anti-cosine leaks → leakage is identity/content-specific, not a smooth distance effect (downgrades the whole "geometry predicts leakage" framing).

## Compute estimate

1× H100 80GB, no training. Base-model forward passes: 24 personas × 50 probes × R≈8 samples for sampling + teacher-forced full-vocab logprobs over both conditions per cell. Rough order ~2-4 GPU-hours. Single seed=42 to match #411; add seeds {7,137} only if the pooled ρ comparison is close.

## What's reused vs new

- **Reused (frozen):** #411 leakage Δ (the DV), #411 persona prompts, #411 held-out 50-claim probe set, #411 cosine values.
- **New:** the JS-divergence predictor over the #411 panel, the response-token cosine recipe, the predictor-comparison analysis + figure (ρ_JS vs ρ_cosine per source + pooled; scatter of leakage Δ vs each distance).
- **Single-variable discipline:** the only thing that changes vs #411 is the persona-distance predictor; the outcome, panel, probes, and trained cells are identical.
