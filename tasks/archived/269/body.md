---
title: Geometry of personas vs geometry of response divergence
kind: experiment
tags: []
created_at: '2026-05-05T08:42:40.000Z'
has_clean_result: false
sagan_id: de067685-070a-418e-af86-1a2e46dad91e
sagan_number: 269
priority: normal
---
## Goal

Quantify the alignment between two pairwise geometries over personas: (a) cosine similarity between layer-10 persona vectors (from `experiments/phase_minus1_persona_vectors/`) and (b) JS divergence between response distributions on a fixed prompt set. Asks whether the *representational* geometry of personas predicts the *behavioral* geometry of their outputs.

## Hypothesis

**H:** The NxN cosine matrix and the NxN JS-divergence matrix are rank-correlated across persona pairs. Specifically, Spearman rho between vec(1 - cosine) and vec(JS-div) over the 20×19/2 = 190 off-diagonal pairs is **> 0.5**.

**Kill criterion:** |rho| < 0.2. Representational similarity does not track behavioral similarity at the population level — geometries decouple, and prior work using cosine as a behavioral proxy (e.g., #237, #267) is suspect for cross-persona prediction.

## Setup

- **Model:** Qwen-2.5-7B-Instruct (same as `phase_minus1_persona_vectors`).
- **Personas:** the 20 personas already in `experiments/phase_minus1_persona_vectors/` (surgeon, paramedic, army_medic, ...).
- **Cosine matrix:** load existing `experiments/phase_minus1_persona_vectors/cosine_matrix.json`. No re-extraction.
- **Response-distribution matrix:** for each persona, sample K=200 prompts from a shared evaluation prompt set (TBD — candidates: TriviaQA, MMLU-pro, or a held-out subset of the conditioning prompts). With persona as system message, generate next-token logit distributions on each prompt's first response token (or first 10 tokens, averaged). Compute pairwise JS divergence over personas, averaged across prompts.
- **No training, no fine-tuning.** Pure inference + analysis.

## Eval / analysis plan

1. Build the NxN JS-div matrix (off-diagonal only matters).
2. Spearman rho on the vectorized off-diagonal entries vs (1 - cosine).
3. Pearson rho as a sanity check.
4. Hero figure: scatter plot of (1 - cosine) vs JS-div, one point per pair, with rho + p-value in caption.
5. Sub-analyses:
   - Per-cluster (e.g., medical personas, fictional personas): is alignment higher within cluster than across?
   - Does the rho depend on K (prompt-set size)?

## Success criterion

- Reproducible JS-div matrix + scatter plot + rho computed on >= 190 pairs.
- Either: rho > 0.5 (H confirmed, geometries align) or rho < 0.2 (kill criterion, geometries decouple) — both are publishable findings.

## Compute

Small — single H100 or even CPU-bound for the analysis half. Generation step needs vLLM batched inference: 20 personas × 200 prompts × 1 token ≈ 4000 forward passes, well under 1 GPU-hour.

## Pod preference

`--intent eval` (1× H100). Or reuse a parent pod if one is live.

## References / parent

- Parent: **#142** (JS-div predicts leakage better than cosine; rho=-0.75 vs 0.57). This issue generalizes that finding from leakage prediction to the full population-level geometry comparison.
- Related: #216 (cosine geometry agrees across layers), #237 (SFT collapses cosine to >=0.97), #267 (L20 centroid steering fails — cosine may not capture behavior).
- Code: `src/explore_persona_space/analysis/divergence.py` (compute_js_divergence), `experiments/phase_minus1_persona_vectors/cosine_matrix.json`.
