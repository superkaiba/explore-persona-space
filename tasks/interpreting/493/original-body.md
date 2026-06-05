---
title: 'Extraction-point × metric bake-off: best base-model predictor of #474 on-policy
  marker transfer'
kind: analysis
tags: []
created_at: '2026-06-05T09:49:22Z'
has_clean_result: false
parent_id: 474
---
## Goal

On the #474 16-transformation marker-transfer substrate (240 ordered pairs; on-policy ΔG = trained − base log P(` ※`) already measured, no retraining), determine which base-model persona-distance predictor best predicts marker transfer — sweeping THREE activation extraction points × a panel of distance metrics × residual-stream layers — and report the single best predictor selected by a held-out (leave-one-context-out) criterion, with the non-stylized-panel and base-prior-safe survival checks #474 established. Base-model forward passes + the already-stored ΔG only; no training.

## Background / substrate (all local, no retraining)

- **Leakage DV (the regression target):** `eval_results/issue_474/cross_eval/{arm}_ep{ep}/G_logprob_matrix.json` — `G[a][b]["delta_g"]` (ΔG = trained − base log P(` ※`), token id 83399, on-policy post-response slot) plus `g_logprob` (trained log-prob, the base-prior-safe construct). 8 cells: arm ∈ {pos, loc} × epoch ∈ {1,2,3,5}. Headline cell = `loc_ep1` (the clean non-saturated one). **Single seed 42** — load-bearing caveat throughout.
- **16 transformations** (each used as both source and target → 240 off-diagonal pairs): defs in `src/explore_persona_space/experiments/i406_conditions.py`. Classes: A = persona (A1–A5; **A3/A4/A5 = "stylized"** = pirate/comedian/villain), B = query-wrap (B1–B5), C1 = chat template, D = register rewrite (D1–D5).
- **Existing predictors (last-prompt-token only, pair-level scalars):** cosine in `eval_results/issue_406/cosine/C_L{0,5,11,15,21,27}.json` (`["matrix"]`); JS in `eval_results/issue_406/divergence/D_matrix.json` (`["JS"]`, plus `["prompt_tokens"]` for the length partial, and `["conditions"]` → the 16 cids).
- **Analysis template (reuse):** `scripts/i474_cosine_followup.py` — already does the length-partial Spearman (rank-residualize on log prompt_tokens), the non-stylized (n=156) vs full (n=240) panels, the per-(arm,epoch) loop, the saturation guard, and a leave-one-context-out CV (`fig9`). **Base-model predictor-extraction template:** `scripts/recompute_predictors_i415.py`. **Canonical metric defs:** `.claude/rules/persona-distance-metrics.md`.
- **Result to beat (loc_ep1):** last-token cosine L21 ρ(ΔG) ≈ +0.57 on non-stylized (survives), JS ≈ −0.11 (null); full panel cosine +0.72, JS −0.58.

## What to run

Re-extract base-model (Qwen-2.5-7B-Instruct) residual activations under each of the 16 transformation prompts, on the SAME predictor probe set the #406/#474 extraction used (matched distribution), at THREE extraction points; compute a panel of pairwise distance metrics for all 240 pairs; regress each against ΔG (+ trained-logp); select the best.

**Extraction points:**
1. **end-of-system-prompt** — last token of the system-prompt-only prefix. Input-independent → **ONE vector per transformation** → only centroid metrics apply (cosine, Euclidean, Mahalanobis-vs-pooled-covariance). Cloud metrics are **N/A** here — label N/A, never force a value.
2. **last-prompt-token** — `last_pos = input_ids.shape[1]-1` after the user question. One vector per (transformation, question) → a cloud per transformation. (Reproduces the existing cosine AND enables cloud metrics.)
3. **mean-over-assistant-response-tokens** — generate a response per (transformation, question) under the prompt, mean-pool its residual activations → a cloud per transformation. Requires generation.

**Metrics** (per layer in {7,14,21,27} ∪ existing {0,5,11,15,21,27}):
- **cosine of mean activation** (centroid) — reproduce/extend existing.
- **Fisher / Mahalanobis** — PCA-whiten to top-k (k ≪ n, well-posed at n≪d), then mean-difference distance.
- **RBF-MMD** between the two activation clouds (median-heuristic bandwidth; permutation null). Cloud-only.
- **C2ST** — held-out linear-probe AUC distinguishing cloud a vs cloud b. Cloud-only.
- **Δ-displacement spectrum** — PAIRED (same probe questions under a and b): PCA on the per-question displacements Δ_i = h_b(Q_i) − h_a(Q_i); report ‖mean displacement‖, the coherence ratio (energy in the mean direction / total), and effective dimensionality as candidate predictor scalars. Cloud-only.
- **Gaussian symmetric-KL (and/or Wasserstein-2)** in the top-k PCA subspace. Cloud-only.

Centering: per the persona-distance-metrics rule + the crossed (transformation × question) design, prompt-center where applicable; report the raw variant too.

**DV / regression (match #474 exactly):** length-partial Spearman (rank-residualize on log prompt_tokens) vs **ΔG (primary)** and **trained log-prob (base-prior-safe, secondary)**; on the **non-stylized (n=156)** and **full (n=240)** panels; per (arm, epoch); headline `loc_ep1`. Report the saturation fraction per cell (a "survival" on a saturated cell is suspect).

## Selecting "the best predictor" (avoid the multiple-comparisons trap)

We test ~6 metrics × 3 extraction points × ~6 layers = many predictors, so in-sample max-|ρ| is upward-biased. The headline predictor is the one with the highest **leave-one-context-out CV** criterion (the `fig9` pattern, generalized) that ALSO (a) survives on the non-stylized panel and (b) survives the base-prior-safe (trained-logp) check. Emit the FULL grid (every metric × extraction-point × layer ρ, with p and CV score) so the search is transparent, then name the winner with the CV + single-seed caveats. Do not narrate an in-sample max as "the best predictor."

## Deliverables

- One script (e.g. `scripts/issue<N>_extraction_metric_bakeoff.py`) doing extraction → metrics → regression → grid; writes `eval_results/issue_<N>/bakeoff/` JSONs + a summary grid JSON.
- Figures into `figures/issue_<N>/`: the metric × extraction-point × layer ρ grid (heatmap) + the winner's scatter/quintile vs ΔG (reuse `paper_plots` style).
- Clean-result naming the best predictor, with the CV + non-stylized + base-prior-safe + single-seed caveats.

## Notes

- Needs a GPU pod (1× H100, intent `eval`) for base-model forward passes + generation (mean-over-response). No training, no LoRA.
- Reuse the exact #406/#474 predictor probe set so all distances are on a matched question distribution.
- This is a predictor-only re-analysis of #474's already-measured leakage (parent #474); nothing is retrained.
