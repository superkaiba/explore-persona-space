---
title: Implement KL/JS divergence of outputs as another measure of persona similarity
kind: experiment
tags: []
created_at: '2026-04-28T05:47:42.000Z'
has_clean_result: false
sagan_id: cee2ea10-37e8-4ab5-907a-f46e3324140e
sagan_number: 140
priority: normal
legacy_why_unset: true
relates_to:
- b3
---
## Symmetric teacher-forced JS/KL divergence as persona similarity metric

### Motivation
We currently measure persona similarity via cosine similarity of last-token hidden states (layers 10/15/20/25). This captures representational geometry but not behavioral divergence at the output level. KL/JS divergence over next-token logit distributions provides a complementary output-space measure. If JS divergence correlates more strongly with marker leakage than cosine similarity, it suggests that output-level geometry is more predictive of cross-persona information flow than hidden-state geometry.

### Method
For each persona pair (A, B) and each of the 20 EVAL_QUESTIONS:

1. **Generate** a response under persona A (vLLM batched inference)
2. **Teacher-force** that response through the model under both persona A and persona B system prompts → per-token logit distributions
3. **Compute** per-token divergences:
   - **JS divergence**: JS(P_A, P_B) = 0.5 * KL(P_A || M) + 0.5 * KL(P_B || M), where M = 0.5*(P_A + P_B). Symmetric, bounded [0, ln(2)].
   - **KL divergence** (both directions): KL(P_A || P_B) and KL(P_B || P_A). Asymmetric — the directional difference may correlate with leakage direction.
4. **Average** across token positions → per-prompt divergences
5. **Repeat** with persona B generating and persona A scoring
6. **Average** both directions → symmetric JS divergence + symmetric mean KL for this (pair, prompt). Also retain the asymmetric KL pair: KL(A||B) and KL(B||A).
7. **Average** across the 20 prompts → per-pair scalars: JS_sym, KL_sym, KL(A→B), KL(B→A), KL_asym = |KL(A→B) - KL(B→A)|

### Model
Qwen/Qwen2.5-7B-Instruct (base, no finetuning) — same as the cosine similarity reference data.

### Personas
- **Phase 1 only (this issue):** Core 12 personas (11 + assistant) from `personas.py`. 66 unique pairs × 20 prompts × 2 directions = 2,640 teacher-force forward passes. Estimated ~1-2 GPU-hours.
- Phase 2 (100-persona scale-up) deferred to a separate issue, gated on Phase 1 results.

### Evaluation / comparison
1. **JS/KL vs cosine correlation:** Spearman/Pearson correlation between JS divergence matrix and cosine similarity matrix across all 66 persona pairs (per layer for cosine)
2. **JS/KL vs leakage correlation:** Spearman/Pearson of JS divergence vs marker leakage rate (same pairs as in `cosine_leakage_correlation.json`)
3. **Predictive comparison:** Does JS/KL divergence predict leakage better than cosine similarity? Compare correlation magnitudes.
4. **KL asymmetry analysis:** Does KL(A→B) - KL(B→A) correlate with directional leakage (A leaks into B vs B leaks into A)?
5. **Visualization:** Scatter plots (JS vs cosine, JS vs leakage, KL asymmetry), heatmaps of JS and KL matrices

### Success criteria
- Divergence matrices are internally consistent (JS symmetric + non-negative + diagonal=0; KL non-negative + diagonal=0)
- JS-leakage correlation has a clear sign (higher divergence → lower leakage, or vice versa)
- Comparison with cosine-leakage correlation is interpretable

### Kill criteria
1. If pilot JS divergence is near-uniform across all pairs (no variance), the metric doesn't discriminate personas at the output level — stop.
2. **Redundancy kill:** If Spearman rho between the 66-pair JS matrix and the cosine matrix exceeds 0.80 for any layer → JS is essentially a monotonic transform of cosine, no new information — stop.

### Pre-check (before running)
Before computing divergences, run a 15-min analysis of existing `cosine_leakage_correlation.json`: does the cosine-leakage Spearman rho improve from layer 10 → 25 (closer to output)? If later-layer cosine already captures output-level divergence well, JS over logits may be partly redundant. Report this as context, not a hard gate.

### Implementation scope
- New script: `scripts/compute_js_divergence.py`
- New analysis function in `src/explore_persona_space/analysis/` (or extend `representation_shift.py`)
- Results saved to `eval_results/js_divergence/`
- Figures saved to `figures/js_divergence/`

### Compute
- Phase 1: ~1-2 GPU-hours (any pod with 1+ GPU)
- Priority: fill work — do not displace higher-priority queue items (#17, #46, #19)
