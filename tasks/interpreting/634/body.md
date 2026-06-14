---
title: 275-role behavior vectors form their own region and do not sit near their context
  families beyond a shuffled-label null at any layer of Qwen-2.5-7B-Instruct (MODERATE
  confidence)
kind: analysis
tags: []
created_at: '2026-06-13T21:35:47Z'
has_clean_result: true
parent_id: 594
origin_prompt: 'can you run this in the background: UMAP/PCA of diverse context vectors
  and behavior vectors to see if there is any structure'
---
# 275-role behavior vectors form their own region and do not sit near their context families beyond a shuffled-label null at any layer of Qwen-2.5-7B-Instruct (MODERATE confidence)

<!-- clean-result-v3 -->

## Takeaways

- Behavior vectors do NOT sit near their matching context: the match rate peaks at **9 of 26 roles (0.346) at layer 13**, inside the shuffled-label null (0.385, p = 0.28).
- The reason is geometric separation, not a broken read: **99.8% of behavior vectors have only other behaviors as their 4 nearest neighbors** — behaviors form their own region.
- The co-embeddability gate **passed at all 28 layers** (scale ratio 1.0-1.55, inside the [0.33, 3.0] band), so the null is a real geometric fact, not a flattened projection.
- Behaviors DO carry coherent internal geometry: k-NN family purity reaches **0.815 over 27 of 275 roles at layer 14** (null 0.444, p = 0.001), clustering by their own axis.
- MODERATE because the matched panel is 26 of 275 roles, and the labeled purity uses 27 of 275, not the planned 275-role denominator.

## What I ran

- **Why:** [#594](https://eps.superkaiba.com/tasks/594) mapped where 50 context prompts live in Qwen-2.5-7B-Instruct's activations (contexts cluster by family at every depth) but deliberately skipped behavior vectors. The open question was whether a role's internal direction lands near the contexts that would elicit it — e.g. a storyteller role near narrative contexts.
- **Design:** one manipulated comparison — for each behavior role, is its nearest context vector in the pre-registered matching family, tested against a shuffled-label permutation null across all 28 layers. Behavior side: 275 roles re-extracted at all 28 layers; a 26-role matched panel (9 character/persona, 5 behavior-directive, 4 output-format, 4 reword, 4 worked-example) carries the headline test. Context side: the parent context-vector atlas's 50-context, 7-family bank, reused verbatim.
- **Training:** none — analysis-only over existing and freshly-extracted activation tensors; no LoRA, no generation.
- **Eval:** primary DV = per-layer matched-family nearest-context-neighbor rate (centered cosine, k=1), tested vs B=1000 shuffled-label null, max-over-layers family-wise correction. Secondary DVs = behavior-alone k-NN family purity (k=4), own-region fraction (k=4), and a per-layer co-embeddability gate. Behavior directions are the mean of K=40 sampled extraction questions per role (coarser than the 240-question construct used elsewhere; carried as a caveat).

## Findings

### Behaviors do not sit near their matching context family beyond chance (0.346 vs null 0.385)

The headline H1 test: for each of the 26 matched-panel roles, does its nearest context vector belong to the pre-registered family, read at every layer against a shuffled-label null.

![Matched-family nearest-context rate by decoder layer; the blue observed line rides inside the grey shuffled-label permutation-null band at every layer, peaking at 0.346 at layer 13.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99da857eb8c8399a886c36fb97fa33c66ac65936/figures/issue_634/nn_rate_vs_layer.png)

> **Figure.** *The observed match rate never leaves the null band.* Blue = matched-family rate over 26 panel-B roles; grey = shuffled-label null (mean to 95th pct), B=1000; red line at the max-rate layer 13. The peak (0.346) sits just inside the band top (0.385); p = 0.28.

The pre-registered floor was met (26 roles, 5 families, ≥2 each), so this is an adjudicated null, not an underpowered one. Removing the dominant behavior-cloud component drives it lower still (0.192 vs null 0.231, p = 0.70). Read as "indistinguishable from null given the variance" on this panel — not "no relationship on all 275 roles."

### Behaviors form their own region, disjoint from context families (99.8% own-region)

If behaviors do not land near contexts, where do they land. The joint embedding answers it: behavior directions and context activations occupy separate regions of the shared space.

![Joint PCA (top) and UMAP (bottom) at layers 7, 14, 21, 27; colored behavior points form their own clusters separated from the grey context cloud at every depth.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99da857eb8c8399a886c36fb97fa33c66ac65936/figures/issue_634/joint_embedding_pca_umap_L7_L14_L21_L27.png)

> **Figure.** *Behaviors (colored, by family) sit apart from contexts (grey).* PCA top, UMAP bottom (n=15, min-dist 0.1, seed 42), quartile layers. 275 behaviors + 50 contexts. The colored behavior points cluster among themselves rather than mixing into the context families.

Quantified: at the best layer, **99.8% of behavior vectors have all 4 of their nearest joint-space neighbors as other behaviors** (≥0.995 at every layer). This is the reconciling fact — H1's null is not "behaviors are scattered noise" but "behaviors occupy a coherent territory of their own that does not overlap the context families."

### Behaviors cluster by their own axis (purity 0.815 over 27 of 275 roles)

Given the own-region separation, do behaviors have *any* coherent internal structure. Labeled k-NN family purity tests whether a behavior's neighbors share its role-type.

![Panel-B-labeled k-NN family purity by layer; the green observed line sits at 0.70-0.82, far above the grey null band (~0.20-0.33) at every layer, peaking 0.815 at layer 14.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99da857eb8c8399a886c36fb97fa33c66ac65936/figures/issue_634/panelB_labeled_purity_vs_layer.png)

> **Figure.** *Behaviors cluster strongly by their own axis.* Green = labeled purity (k=4) over 27 of 275 roles; grey = permutation null (mean to 95th pct). Peak 0.815 at layer 14 vs null 95th pct 0.444; p = 0.001. The figure title flags the 27-of-275 denominator.

This is a clean positive, but on **27 of 275 roles, not the planned 275-role role-axis purity** — no 275-role role-type labeling source exists, so the denominator is the labeled subset. Every mention of this number is scoped to those 27 roles. Behaviors are geometrically differentiated by axis; they just do not align that structure with the context families.

### The null is geometric, not a projection artifact — the co-embeddability gate passed

A joint embedding can fake a null if one vector set dominates the other's scale. The gate checks per-layer variance and median-norm ratios before trusting the H1 read.

![Co-embeddability gate: behavior/context variance ratio and median-norm ratio by layer, both riding 1.0-1.55, well inside the [1/3, 3] pass band marked by dotted lines at 0.33 and 3.0.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99da857eb8c8399a886c36fb97fa33c66ac65936/figures/issue_634/coembeddability_gate.png)

> **Figure.** *The two vector sets share a comparable scale at every layer.* Blue = variance ratio, orange = median-norm ratio (behavior/context); dotted lines mark the [0.33, 3.0] pass band. 0 of 28 layers fail; the worst ratio is 1.55 at the final layer.

Because the gate passed, the H1 fallback (family-centroid CKA + Procrustes) is a diagnostic: cross-space CKA peaks 0.770 at layer 14 and the Procrustes residual stays low mid-network (0.116) before degrading late. The two spaces share *some* coarse shape at the family-centroid level, but not enough for individual behaviors to find a matching context neighbor.

## Data

### Trained on

n/a — no training in this task. The analysis reads pre-computed activation tensors: 275 behavior-vector roles re-extracted this run + the parent context-vector atlas's 50-context bank, both reused verbatim. No LoRA adapter, no training mix.

### Evaluated with

The behavior bank is 275 assistant-axis roles (5 house system prompts each), read at the last input token of the assistant header across all 28 layers — the same read slot as the parent atlas's contexts. Each role direction is the mean over K=40 extraction questions sampled from the 240-question pool (seed 42), making it a slightly coarser estimate than the 240-question construct. The 26-role matched panel and its role→family map were frozen before any embedding so H1 could not be tuned post-hoc.

The complete frozen role-to-context-family map for the H1 matched panel (not a sample); the machine-readable version is linked below.

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

**Methodology reference:** the findings-blind methodology + hyperparameter doc is linked here once generated.

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` (no LoRA) |
| Behavior extraction | Method A, last-input-token absolute mean, all 28 layers |
| Questions per role | K=40 (sampled from 240, seed 42) |
| Behavior roles | 275 (full), 26 matched panel (H1), 27 labeled (H2) |
| Context bank | #594, 50 contexts, 7 families |
| Embedding | per-layer PCA + UMAP (n∈{5,15,30}, min-dist∈{0.1,0.5}, cosine, seed 42); t-SNE at best layer |
| k-NN purity | k=4 |
| Permutation null | B=1000, max-over-layers FWER |
| Hidden dim / layers | 3584 / 28 |

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
