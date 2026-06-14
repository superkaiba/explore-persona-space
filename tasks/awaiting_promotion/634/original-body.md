---
title: 'Joint UMAP/PCA of behavior vectors with #594 context vectors — does behavior-vector
  geometry align with context geometry?'
kind: analysis
tags: []
created_at: '2026-06-13T21:35:47Z'
has_clean_result: false
parent_id: 594
origin_prompt: 'can you run this in the background: UMAP/PCA of diverse context vectors
  and behavior vectors to see if there is any structure'
---
## Goal
Run the **behavior-vector half** of the original "UMAP/PCA of diverse context vectors and behavior vectors to see if there is any structure" request, and jointly embed behavior vectors with the #594 context vectors to test whether behaviors sit near their related contexts in representation space.

## Motivation
#594 already drew the descriptive atlas for **context vectors** (50-context battery, per-layer PCA/UMAP/t-SNE, all 28 layers of Qwen2.5-7B-Instruct — contexts cluster by family at every depth, sharpest L13–18). It deliberately did *not* embed **behavior vectors**. The original chat request asked for both. This task fills the gap: characterize the geometry of the behavior/persona vectors on their own, and — the more interesting question — place them in the **same space** as the #594 context vectors and see whether a behavior direction lands near the contexts that elicit it (e.g. the sycophancy vector near sycophancy-instruction / sycophantic-persona contexts).

## What already exists (reuse — do not recompute unless a comparability check fails)
- **#594 context vectors:** fp32 probe-mean tensor `(50 instances, 28 layers, 3584 dim)` at HF `superkaiba1/explore-persona-space-data` → `issue594_context_geometry/analysis_tensors/` (read position: residual activation at the newline right after the assistant header, mean over the fixed 48-probe Betley pool). Battery families: persona / WildChat prefix / worked-example / instruction-reword / output-format / behavior-instruction / bare-default.
- **Behavior / persona vectors:** `scripts/extract_persona_vectors.py` (275 assistant-axis roles; Method A = last-input-token hidden state, Method B = mean-response-token; output `(n_layers, hidden_dim)` per role). Many already stored on HF: `issue368_persona_vectors_chenstyle/` (275-role centroids), `issue363_chen_vs_centroid/persona_vectors/` (the 5 axis vectors: deception, helpfulness, hostility, refusal-tendency, sycophancy — centroid / Chen-style / ablation variants).
- **#594 plotting helpers:** `scripts/issue594_fig_hero_embeddings_clean.py`, `scripts/issue594_analyze_context_geometry.py`, `scripts/issue594_common.py` — reuse the embedding + family-purity + permutation-null machinery so this map is methodologically comparable to the parent.

## Design sketch (planner to flesh out)
- **Comparability is the load-bearing decision.** Behavior vectors must live in the same space as #594's context vectors to co-embed: same model, same 28 layers, ideally the same read position. #594 read the last-input slot; `extract_persona_vectors.py` Method A is also last-input-token → use Method A (or re-extract behavior vectors at #594's exact slot if Method A's read differs materially). State and justify the choice; if absolute activations vs difference-vectors aren't co-embeddable, run behavior-vectors-alone embedding + a separate alignment readout instead of forcing a joint UMAP.
- **Behavior-vector set:** start with a tractable, interpretable panel (the 5 axis vectors + a sample of the 275 roles, or all 275 if it embeds cleanly), not necessarily all 2400 stored files.
- **Outputs (mirror #594):** per-layer PCA + UMAP (+ t-SNE at best layer); joint context+behavior embedding with families color-coded and behaviors marked; family-purity / nearest-neighbor-by-depth for behaviors; the boring-explanation controls #594 used (length axis, lexical/surface) where applicable.
- **Headline question:** does each behavior vector's nearest context-neighbor belong to the matching family, beyond a permutation null? Quantify, don't eyeball.

## Caveats to carry
- Behavior "vectors" are directions (often differences); context "vectors" are absolute mean activations. Whether they're meaningfully co-embeddable is itself part of the finding — report it honestly rather than producing a pretty-but-meaningless joint UMAP.
- #594 only ran a lexical surface baseline (no semantic embedding baseline); keep the same honesty about what the null actually controls for.
- CPU-only on existing HF artifacts — no training, no generation, no GPU pod expected. If the planner finds a genuine need to re-extract behavior vectors at #594's exact read slot, that's a short forward-pass-only extraction (eval/debug intent), not a training run.

## Provenance
Originating user request (verbatim, 2026-06-11 session): "can you run this in the background: UMAP/PCA of diverse context vectors and behavior vectors to see if there is any structure". The context half ran as #594; this task is the never-run behavior half, requested again 2026-06-13 ("Run in background with Happy Coder").
