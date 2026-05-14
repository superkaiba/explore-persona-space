---
title: 'Extraction-recipe KILL verdict is layer-universal: 419 of 420 cells fail across
  28 Qwen layers (HIGH confidence)'
kind: experiment
tags:
- superseded
created_at: '2026-05-03T19:20:21.000Z'
has_clean_result: false
sagan_id: 887c3392-07dc-495a-a5ec-abfb4500feb0
sagan_number: 221
priority: normal
---
## TL;DR

### Background

Clean-result #216 (from #201) established that all 15 extraction-method pairs KILL at layers [7, 14, 21, 27] on Qwen2.5-7B-Instruct. However, the non-monotonic mc_r trajectory (A↔B: 0.53 → 0.75 → 0.89 → 0.90) suggested a potential sweet-spot in the L20-L25 range that might cross into GREY or PASS. This follow-up sweeps all 28 layers to test whether the 4-layer quartile was missing a convergence window.

### Methodology

Same 6 extraction methods as #201 (A, B, B*, C1, C2, C3) on Qwen2.5-7B-Instruct (bf16, greedy, seed 42), N=275 personas × 240 questions. Single variable changed: layers [7,14,21,27] → all 28 layers [0..27]. Per-q caches skipped (--skip-per-q) to avoid 62 GB disk. Same PASS/GREY/KILL thresholds as #201 (cos_min > 0.95 + mc_r > 0.90 = PASS; cos_min < 0.85 or mc_r < 0.70 = KILL).

### Results

![Verdict heatmap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c8f61db/eval_results/issue_218/figures/verdict_partition_heatmap.png)

Of 420 pair-layer cells (28 layers × 15 pairs), 419 are KILL and 1 is GREY (A↔C3 at L0, a sanity pair with cos_min=0.90, mc_r=0.85). Zero PASS. The 4-layer quartile was NOT missing a sweet spot — the verdict is layer-universal (N=275 personas, 240 questions per cell).

**Main takeaways:**

- **KILL is layer-universal for all load-bearing pairs: 0/140 load-bearing cells reach GREY or PASS; 1/420 total cells is GREY (A↔C3 at L0, a sanity pair) (N=275 per cell).** The mc_r and cos_min gate metrics are anti-correlated across the depth axis — no layer satisfies both simultaneously. This vindicates the 4-layer quartile sampling in #201.

- **A↔B mc_r peaks at L24 (0.902, barely crossing 0.90) but cos_min stays at 0.695 there (N=275).** Relative persona geometry converges broadly from mc_r=0.49 (L0) to 0.90 (L24-L27), but absolute directions never follow — cos_min ranges 0.53-0.70 across L0-L26, then degenerates to -0.01 at L27.

- **B↔B* at L0 has the best cos_min of any load-bearing pair (0.930, N=275) but mc_r=0.50.** This is the closest any pair comes to PASS on cosine, but the matrix correlation is deeply in KILL territory. Pooling over input vs response tokens at the earliest layer produces nearly-parallel directions but different geometric structure.

- **The anti-correlation pattern means no "recipe-invariant sweet spot" exists in this model.** Claims that need both absolute-direction agreement AND relative-geometry preservation cannot be satisfied at any single layer.

**Confidence: HIGH** — 419 KILL + 1 GREY across 420 cells with 275 personas each; all 140 load-bearing cells KILL. The finding is exhaustive over the model's full depth. Noise floor from #201 (cos_min 0.99+) rules out measurement noise. Single-model limitation (Qwen-2.5-7B-Instruct only) does not weaken the exhaustive-depth claim.

### Next steps

- **#201's clean-result #216 is strengthened:** the 4-layer sample was representative; no footnote about "untested layers" needed.
- **Follow-up #2 from #201 (recipe-invariant subspace via PCA)** becomes more urgent — if no single layer works, perhaps a linear subspace across layers captures recipe-invariant structure.
- **Cross-model replication on Llama-3.1-8B** would test whether the anti-correlation pattern is Qwen-specific.

---

# Detailed report

## Source issues

- #218 — this layer-sweep diagnostic (parent: #201)
- #201 — the 6-way extraction-method ablation at 4 layers
- #216 — #201's clean result

## Setup & hyper-parameters

**Why this experiment:** #201 found all 15 pairs KILL at 4 layers, but mc_r was rising. This exhaustive sweep tests whether any of the 24 untested layers breaks the pattern.

Identical to #201 except: layers = [0..27] instead of [7,14,21,27]. Per-q caches skipped (--skip-per-q).

| | |
|-|-|
| Model | Qwen/Qwen2.5-7B-Instruct (bf16) |
| Seed | 42 (greedy) |
| Layers | [0, 1, 2, ..., 27] (all 28) |
| Roles | 275 × 240 questions |
| Methods | A, B, B*, C1, C2, C3 (6 methods, 15 pairs) |
| GPU | epm-issue-218, 1×H100, ~3.5 GPU-hours |
| Git commit | c8f61db |

## Headline numbers

| Metric | Best value | At pair / layer | Gate threshold | Verdict |
|---|---|---|---|---|
| cos_min (load-bearing) | 0.930 | B↔B* L0 | > 0.85 | **cos passes** but mc_r=0.50 → KILL |
| mc_r (load-bearing) | 0.902 | A↔B L24 | > 0.90 | **mc_r barely passes** but cos_min=0.695 → KILL |
| Both simultaneously | never | — | cos>0.85 AND mc_r>0.90 | **never achieved at any layer** |

**Standing caveats:**
- Single model (Qwen-2.5-7B-Instruct)
- Single seed (42, greedy — deterministic)
- Assistant-axis personas only
- Method B = positive-centroid only (not Chen et al. 2025 contrast)
- Per-question spread skipped

## Artifacts

| Type | Path |
|---|---|
| Results JSON | eval_results/issue_218/run_result.json |
| Heatmap | eval_results/issue_218/figures/verdict_partition_heatmap.png |
| Violins | eval_results/issue_218/figures/per_persona_cos_pair_dists.png |
| Bars | eval_results/issue_218/figures/matrix_corr_pair_layer.png |
| Centroids | data/persona_vectors/issue_218/ (on stopped pod) |
