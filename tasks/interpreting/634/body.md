---
title: The 275-role behavior bank forms its own region, while the 26-role matched
  panel stays within the shuffled-label context-match null across Qwen-2.5-7B-Instruct
  layers (MODERATE confidence)
kind: analysis
tags: []
created_at: '2026-06-13T21:35:47Z'
has_clean_result: true
parent_id: 594
origin_prompt: 'can you run this in the background: UMAP/PCA of diverse context vectors
  and behavior vectors to see if there is any structure'
---
# The 275-role behavior bank forms its own region, while the 26-role matched panel stays within the shuffled-label context-match null across Qwen-2.5-7B-Instruct layers (MODERATE confidence)

<!-- clean-result-v3 -->

## Takeaways

- Behavior vectors do not sit near matching context: match rate peaks at **9 of 26 roles (0.346), layer 13**, inside the shuffled-label null (0.385, p = 0.28; k=2/3/4 also null).
- **99.8% of behavior vectors have only other behaviors as their 4 nearest neighbors**, so behavior directions occupy their own region of the joint space — the separation behind the null.
- The two vector sets pass the planned scale/variance gate at all 28 layers (variance ratio ≤ 1.55, median-norm ratio ≤ 1.22) — ruling out a gross scale mismatch only.
- k-NN family purity over the 27 labeled behaviors peaks **0.815 at layer 14** (null 0.444, p = 0.001 at all 28 layers): coherent internal geometry on the labeled subset, not all 275 roles.
- Scope is single-seed and panel-level: matched panel 26 of 275 roles (single seed, K=40 questions/role), labeled purity 27 of 275, not the planned 275-role denominator.

## What I ran

- **Why:** [#594](https://eps.superkaiba.com/tasks/594) mapped where 50 context prompts live in Qwen-2.5-7B-Instruct's activations (contexts cluster by family at every depth) but deliberately skipped behavior vectors. The open question was whether a role's internal direction lands near the contexts that would elicit it (e.g., a storyteller role near narrative contexts).
- **Design:** one manipulated comparison. For each behavior role, is its nearest context vector in the planned matching family, tested against a shuffled-label permutation null across all 28 layers. Behavior side: 275 roles re-extracted at all 28 layers; a 26-role matched panel (9 character/persona, 5 behavior-directive, 4 output-format, 4 reword, 4 worked-example) carries the headline test. Context side: the parent context-vector atlas's 50-context, 7-family bank, reused verbatim.
- **Training:** none. The analysis reads existing and freshly-extracted activation tensors; no LoRA, no generation.
- **Eval:** primary DV = per-layer matched-family nearest-context-neighbor rate (centered cosine, k=1), tested vs B=1000 shuffled-label null, max-over-layers family-wise correction. Secondary DVs = behavior-alone k-NN family purity (k=4), own-region fraction (k=4), and a per-layer co-embeddability gate. Behavior directions are the mean of K=40 sampled extraction questions per role (coarser than the 240-question construct used elsewhere; carried as a caveat).

## Findings

### Behaviors do not sit near their matching context family beyond chance (0.346 vs null 0.385)

For each of the 26 matched-panel roles, the matched-family test asks whether its nearest context vector belongs to the family frozen before analysis, against a shuffled-label null at every layer.

![Matched-family nearest-context rate by decoder layer; the blue observed line rides inside the grey shuffled-label permutation-null band at every layer, peaking at 0.346 at layer 13.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99da857eb8c8399a886c36fb97fa33c66ac65936/figures/issue_634/nn_rate_vs_layer.png)

> **Figure.** *The observed match rate never leaves the null band.* Blue = matched-family rate over 26 panel-B roles; grey = shuffled-label null (mean to 95th pct), B=1000; red line at the max-rate layer 13. The peak (0.346) sits just inside the band top (0.385); p = 0.28. The observed rate sits BELOW the null mean on 15 of 28 layers (anti-enriched, esp. late layers 21-27) — the null is not just "inside the band," it trends below shuffled-label baseline across much of depth.

The planned floor was met (26 roles, 5 families, ≥2 each), so on this panel/seed (single seed, K=40 questions/role) it is an adjudicated null, not an underpowered one. The context-only nearest neighbor is non-uniform, though: across 26 roles × 28 layers it lands on `format` (25%), `rephrase` (21%), `persona` (20%) far more than the others, so the null reads as "behaviors converge on a few dominant context families," not "behaviors point nowhere." Relative to the raw match rate (0.346 vs null 0.385), the residualized control (behavior-cloud PC removed, own residualized null) is more null still (0.192 vs null 0.231, p = 0.70). A soft-k-NN sensitivity re-run stays null across k=2/3/4 criteria (p = 0.69/0.65/0.78; `figures/issue_634/h1_soft_knn_sensitivity.png`). The read on this panel/seed is "indistinguishable from null given the variance," not "no relationship on all 275 roles."

### Behaviors form their own region, disjoint from context families (99.8% own-region)

Behaviors instead form a region of their own. The joint embedding shows behavior directions and context activations occupying separate regions of the shared space.

![Joint PCA (top) and UMAP (bottom) at layers 7, 14, 21, 27; colored context-family circles occupy one region while the grey-x and near-black-triangle behavior vectors cluster in a separate region at every depth.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99da857eb8c8399a886c36fb97fa33c66ac65936/figures/issue_634/joint_embedding_pca_umap_L7_L14_L21_L27.png)

> **Figure.** *Behavior vectors (grey + black) cluster apart from the context-family centroids (colored).* PCA top, UMAP bottom (n=15, min-dist 0.1, seed 42), quartile layers. Colored circles = the 7 context-family centroids; grey x = the 249 non-panel behavior vectors; near-black triangles = the 26 matched-panel behavior vectors. The behavior glyphs cluster into their own region, separate from the context circles.

At the match-rate best layer (13), 99.8% of behavior vectors have all 4 of their nearest joint-space neighbors as other behaviors; the fraction runs 99.5-100% across all 28 layers, with 16 of the 28 at exactly 1.0 (saturation, not just stability). The matched-family null is consistent with behaviors occupying a coherent region of their own that does not overlap the context families.

### Behaviors cluster by their own axis (purity 0.815 over 27 of 275 roles)

Given the own-region separation, do the 27 labeled Panel-B behaviors have *any* coherent internal structure? Labeled k-NN family purity tests whether a behavior's neighbors share its role-type.

![Panel-B-labeled k-NN family purity by layer; the green observed line sits at 0.70-0.82, far above the grey null band (~0.20-0.33) at every layer, peaking 0.815 at layer 14.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99da857eb8c8399a886c36fb97fa33c66ac65936/figures/issue_634/panelB_labeled_purity_vs_layer.png)

> **Figure.** *Behaviors cluster strongly by their own axis.* Green = labeled purity (k=4) over 27 of 275 roles; grey = permutation null (mean to 95th pct). Peak 0.815 at layer 14 vs null 95th pct 0.444; p = 0.001. The figure title flags the 27-of-275 denominator.

The positive holds, but on 27 of 275 roles, not the planned 275-role purity. No 275-role labeling source exists, so the denominator is the labeled subset (every mention is scoped to those 27). The signal is also not a single peak: purity clears the permutation floor (p = 0.001) at all 28 layers, stronger than a one-layer spike. Whole-depth clearance has a second reading, though: a labeling or selection artifact over the 27 most-separable roles is not ruled out from this subset alone. Behaviors are differentiated by axis; they just do not align that structure with the context families.

### The null is not a gross scale mismatch — the scale/variance gate passes at all 28 layers

A joint embedding can fake a null if one vector set dominates the other's scale. The gate checks per-layer variance and median-norm ratios; passing rules out a gross scale/norm mismatch behind the matched-family null, but it does not certify the joint map against extraction-method, pooling, or prompt-distribution artifacts.

![Co-embeddability gate: behavior/context variance ratio and median-norm ratio by layer, both riding 1.0-1.55, well inside the [1/3, 3] pass band marked by dotted lines at 0.33 and 3.0.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99da857eb8c8399a886c36fb97fa33c66ac65936/figures/issue_634/coembeddability_gate.png)

> **Figure.** *The two vector sets share a comparable scale at every layer.* Blue = variance ratio, orange = median-norm ratio (behavior/context); dotted lines mark the [0.33, 3.0] pass band. 0 of 28 layers fail; the worst variance ratio is 1.55, the worst median-norm ratio 1.22, both at the final layer.

The gate addresses only gross scale/variance, so the matched-family null is geometric in that narrow sense; the exploratory cross-space alignment below shows how much coarse shape the two spaces *do* share.

### Exploratory: the two spaces share coarse family shape mid-network but not enough to match neighbors

A diagnostic, not a planned test: family-centroid CKA and Procrustes alignment measure whether the behavior and context spaces share coarse structure at the family-centroid level.

![Cross-space alignment: left, family-centroid linear CKA by layer peaking ~0.77 around layer 14 then degrading to ~0.42 by layer 27; right, Procrustes alignment residual low (~0.12) mid-network and spiking to ~0.54 at layer 27.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99da857eb8c8399a886c36fb97fa33c66ac65936/figures/issue_634/cross_space_alignment.png)

> **Figure.** *Exploratory.* Left = family-centroid linear CKA (behavior vs context); right = raw Procrustes alignment residual. CKA peaks 0.770 at layer 14; the residual stays low mid-network (0.116 at layer 14) and spikes to 0.540 at layer 27. A diagnostic, not a planned test.

The two spaces share *some* coarse shape at the family-centroid level mid-network, but the late-layer residual spike and the matched-family null together say it is not enough for individual behaviors to find a matching context neighbor. N=5 shared family centroids (the intersection of behavior-family and context-family labels) enter the CKA/Procrustes computation; no permutation null or bootstrap was run on this exploratory pair.

## Data

### Trained on

n/a — no training in this task. The analysis reads pre-computed activation tensors: 275 behavior-vector roles re-extracted this run + the parent context-vector atlas's 50-context bank, both reused verbatim. No LoRA adapter, no training mix.

### Evaluated with

The behavior bank is 275 assistant-axis roles (5 house system prompts each), read at the last input token of the assistant header across all 28 layers — the same read slot as the parent atlas's contexts. Each role direction is the mean over K=40 extraction questions sampled from the 240-question pool (seed 42), making it a slightly coarser estimate than the 240-question construct. The 26-role matched panel and its role→family map were frozen before any embedding, so the matched-family test could not be tuned after seeing the embeddings.

The complete frozen role-to-context-family map for the matched panel (not a sample); the machine-readable version is linked below.

<details open>
<summary>All 26 panel-B roles and their frozen context families (complete map, of 275 total roles)</summary>

| Family | Roles |
|---|---|
| persona (9) | detective, pirate, philosopher, warrior, rogue, hacker, bard, jester, sage |
| behavior-directive (5) | skeptic, contrarian, devils_advocate, perfectionist, pacifist |
| output-format (4) | summarizer, proofreader, editor, translator |
| instruction-reword (4) | tutor, interpreter, instructor, teacher |
| worked-example (4) | mathematician, programmer, debugger, analyst |

</details>

Full frozen map + per-layer nearest-neighbor table: [`panelB_nn_table.json`](https://github.com/superkaiba/explore-persona-space/blob/99da857eb8c8399a886c36fb97fa33c66ac65936/eval_results/issue_634/panelB_nn_table.json). Per-layer metrics: [`per_layer_nn_purity.json`](https://github.com/superkaiba/explore-persona-space/blob/99da857eb8c8399a886c36fb97fa33c66ac65936/eval_results/issue_634/per_layer_nn_purity.json).

### Generated

n/a — no model generations. This is a geometry analysis: each role yields one fp32 direction vector per layer, not a completion. The 275-role behavior tensor (277 files: 275 per-role + mean + extraction manifest) is on HF: [issue634_behavior_geometry/analysis_tensors/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/23c036cdc56e344f9a58cb94665d6b3e01eb7656/issue634_behavior_geometry/analysis_tensors). The #594 context bank reused as the matching target: [issue594_context_geometry/analysis_tensors/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/23c036cdc56e344f9a58cb94665d6b3e01eb7656/issue594_context_geometry/analysis_tensors).

## Reproducibility

**Methodology reference:** see `docs/methodology/issue_634.md` (gist + GitHub blob link appended at promotion).

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Read position | last input token under add_generation_prompt=True |
| Layers | all 28 decoder blocks |
| Hidden dim | 3584 |
| Behavior set (Panel A) | 275 |
| K (questions/role) | 40 of the 240 shared extraction questions |
| Panel B — H1 matched panel | 26 roles / 5 families |
| Panel B — H2 labeled purity subset | 27 roles / 6 families |
| Context bank (reused #594) | (50, 28, 3584), 7 families |
| k-NN purity k | 4 |
| Permutation null B | 1000, draws shared across layers; max-over-layers FWER |
| UMAP | n_neighbors ∈ {5,15,30} |
| t-SNE | perplexity ∈ {5,15,30} |
| PCA | random_state=42 |
| Cross-space fallback | family-centroid linear CKA + orthogonal Procrustes |
| Seed | 42 everywhere |

**Artifacts:**

- Behavior-vector tensor (275 roles × 28 layers × 3584, 277 files): [issue634_behavior_geometry/analysis_tensors/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/23c036cdc56e344f9a58cb94665d6b3e01eb7656/issue634_behavior_geometry/analysis_tensors)
- Reused #594 context bank: [issue594_context_geometry/analysis_tensors/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/23c036cdc56e344f9a58cb94665d6b3e01eb7656/issue594_context_geometry/analysis_tensors) (from [#594](https://eps.superkaiba.com/tasks/594) — fit: same model, same last-input read slot, same 28 layers; the joint-embedding target)
- Eval JSONs (5): [`eval_results/issue_634/`](https://github.com/superkaiba/explore-persona-space/tree/99da857eb8c8399a886c36fb97fa33c66ac65936/eval_results/issue_634)
- Figures (6, inline + supplementary CKA/Procrustes + t-SNE): [`figures/issue_634/`](https://github.com/superkaiba/explore-persona-space/tree/99da857eb8c8399a886c36fb97fa33c66ac65936/figures/issue_634)
- WandB extraction run: [issue634/r1awail2](https://wandb.ai/thomasjiralerspong/issue634/runs/r1awail2)

**Compute:**

- Phase 1 extraction: ~33.5 min on 1× A100-80 (~0.6 GPU-h actual; the plan's ~11 GPU-h estimate was bandwidth-bound and conservative — the K=40 smoke-gate descope held).
- Phase 2 joint geometry: ~5 min, VM CPU.
- Pod: GCP `a2-ultragpu-1g`, ephemeral.

**Code:**

- Family map builder: [`scripts/issue634_build_family_map.py`](https://github.com/superkaiba/explore-persona-space/blob/99da857eb8c8399a886c36fb97fa33c66ac65936/scripts/issue634_build_family_map.py)
- Behavior-vector extraction: [`scripts/issue634_extract_behavior_vectors.py`](https://github.com/superkaiba/explore-persona-space/blob/99da857eb8c8399a886c36fb97fa33c66ac65936/scripts/issue634_extract_behavior_vectors.py)
- Joint-geometry driver: [`scripts/issue634_joint_geometry.py`](https://github.com/superkaiba/explore-persona-space/blob/99da857eb8c8399a886c36fb97fa33c66ac65936/scripts/issue634_joint_geometry.py)
- Git commit (results + figures): `99da857eb8c8399a886c36fb97fa33c66ac65936` (branch `issue-634`)
- Reproduce:

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space
    git checkout 99da857eb8c8399a886c36fb97fa33c66ac65936
    uv sync
    uv run python scripts/issue634_joint_geometry.py
    ```

**Context:**

- Created 2026-06-13; extraction + analysis run 2026-06-13/14.
- Follow-up to [#594](https://eps.superkaiba.com/tasks/594) — the context-vector atlas this task co-embeds behavior vectors against. The behavior half of an original two-part request whose context half ran as #594.
- Originating prompt(s), verbatim:
  > can you run this in the background: UMAP/PCA of diverse context vectors and behavior vectors to see if there is any structure
