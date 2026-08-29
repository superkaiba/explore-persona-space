# Mapping differences and few-query transfer across Qwen and Llama

**Issue #2569 follow-up · Qwen2.5-7B-Instruct Q14 × Llama-3.1-8B-Instruct L16 · 10,000 LMSYS prompts**

## Bottom line

The fixed Procrustes alignment is incomplete (held-out context/answer cosines 0.657, 0.708, and 0.698). Consequently, encoder, diagonal, and encoder×writer interaction contrasts can all contain coordinate-alignment residual. Only the writer contrast has an exact zero null when there is no writer effect, although imperfect alignment can still distort its nonzero magnitude. Alignment-free within-model writer contrasts remain prompt-specific in both Qwen and Llama (held-out flattened cosine 0.283 and 0.332).

Mapping transfer is not useful at the very smallest budgets: at k=2, 4, and 8, every paired repeat favors fitting the target directly by centered cosine. A small directional advantage first emerges around 32 queries (median paired Δ cosine 0.030; 92.5% of 40 repeat-cells) and is consistent by 64 (100.0% positive). At 256 queries, median centered cosine with the frozen full target map is 0.815, versus 0.732 from scratch.

## 1. Factorial mapping diff

![Held-out factorial mapping contrasts](fig1_factorial_mapping_diff.png)

**Figure 1.** Four native maps—encoder {Qwen, Llama} × answer writer {Qwen, Llama}—were transformed into one fixed Qwen basis using train-only semi-orthogonal Procrustes alignments. Black marks show the 95% row-permutation null. Encoder, diagonal, and interaction terms can contain alignment residual; their R² values are 0.206, 0.191, and -0.051. The writer term has cosine 0.286 and R² -0.041. Every observed cosine exceeds all 1,000 row-pairing permutations (p=0.0010), which establishes prompt specificity but does not remove alignment confounding.

Numerically, the encoder-labeled representation contrast is largest and most split-half stable (RMS 7.51; cosine 0.809). The writer contrast has RMS 5.67 and split-half cosine 0.688. Interaction is smaller and less stable (2.46, 0.493) and may be entirely alignment residual. The diagonal replicates numerically on seed 137 (R² 0.191) but remains alignment-confounded. After alignment, writer and encoder operators are nearly orthogonal (cosine -0.049); because encoder is confounded, this is descriptive rather than a behavioral claim.

## 2. Can one mapping be transferred with only residual summaries or a few queries?

![Few-query mapping transfer](fig2_fewshot_transfer.png)

**Figure 2.** Panel A evaluates recovery of the frozen full-data target mapping on the untouched 1,500-row test set—not direct fit to observed answers. Thin lines are the four direction/writer cells; thick lines are their medians. The 0-query diamond uses separate mean/top-64 PCA/variance summaries with components paired only by variance rank and marginal skewness. This particular orientation heuristic fails (centered cosine -0.002–0.020). Rank-64 compression alone is not the cause: reconstructing target answers in the same rank-64 basis retains full-map cosine 0.782–0.825. Stronger unsupervised alignment algorithms were not tested. Panel B shows paired within-anchor-set cosine advantage over the equal-query scratch control (median and pooled 10th–90th percentiles across 4 × 10 runs).

For k paired train queries, regularized context and answer bridges use only sample means and centered Gram matrices. No validation rows tune the bridge. The frozen source map is applied between those two bridges. The control fits target context→answer directly from the identical k anchors.

Table entries are **transported source map / target fit from scratch**, measured by held-out centered cosine with the full target mapping:

| Direction / writer | 16 queries | 32 queries | 64 queries | 128 queries | 256 queries |
|---|---:|---:|---:|---:|---:|
| Q→L · Q writer | 0.345 / 0.360 | 0.488 / 0.455 | 0.643 / 0.595 | 0.731 / 0.667 | 0.798 / 0.729 |
| Q→L · L writer | 0.319 / 0.316 | 0.500 / 0.496 | 0.641 / 0.589 | 0.722 / 0.659 | 0.795 / 0.723 |
| L→Q · Q writer | 0.407 / 0.402 | 0.577 / 0.524 | 0.706 / 0.650 | 0.792 / 0.713 | 0.853 / 0.764 |
| L→Q · L writer | 0.362 / 0.361 | 0.537 / 0.506 | 0.674 / 0.609 | 0.772 / 0.681 | 0.832 / 0.735 |

Below 16 queries, transport is systematically worse than scratch (k=2: median Δ -0.011, 40/40 negative, k=4: median Δ -0.031, 40/40 negative, k=8: median Δ -0.036, 40/40 negative). At 16 it is directionally indistinguishable. A small advantage appears at 32; by 64, 40/40 paired repeat-cells favor transfer. At 256, transported predictions have median centered cosine 0.587 with actual target answers, versus 0.525 for scratch; the original 8,000-row target maps reach 0.724. Scale-sensitive R² is retained only as a secondary diagnostic because the two-stage transport and one-stage scratch fits shrink differently.

## 3. What behavioral differences are visible?

![Behavior readouts from the writer contrast](fig3_behavior_readout.png)

**Figure 3.** A ridge readout trained on the observed writer activation contrast can recover several answer differences, but only some survive when the writer contrast is predicted from context through the mapping diff. Mapping-mediated R² values are semantic divergence 0.267 and replicates at 0.321 on seed 137, log length 0.040; seed 137 0.022, refusal -0.027, and repetition 0.005. Semantic divergence is the only mapping-mediated readout above 0.05 R² in this run.

## Exact design

- Frozen split: 8,000 train / 500 validation / 1,500 test prompts. The new analyses never fit on test rows.
- Fixed common basis: context Procrustes on Qwen-writer train contexts; answer Procrustes pooled over both answer writers. All affine translations are retained. Held-out context/answer alignment cosines are 0.657/0.708/0.698, so encoder, interaction, and diagonal contrasts can include residual alignment error.
- Factorial contrasts: writer, encoder, encoder×writer interaction, and the natural diagonal Qwen-own − Llama-own difference. Encoder, interaction, and diagonal can contain alignment residual; within-encoder Qwen/Llama writer contrasts provide alignment-free checks.
- Null: 1,000 held-out row-pairing permutations that destroy prompt correspondence.
- Stability: two disjoint 4,000-row train refits at the original selected ridge lambdas; independent generation seed 137.
- Few-query transfer: both Qwen→Llama and Llama→Qwen, both answer writers, k ∈ {2,4,8,16,32,64,128,256}, 10 random anchor sets per cell, fixed ridge fraction 0.01, no validation tuning.
- Primary transfer score is cosine after subtracting the target model's train-answer mean from both the candidate and frozen target-map predictions. It measures recovery of the frozen target mapping, not direct fit to observed answers. The scale-sensitive normalized R² is retained in JSON as a secondary diagnostic.

## Interpretation and limits

Variance-ranked, skewness-oriented PCA summaries do not recover the correspondence; this does **not** rule out stronger unsupervised methods such as distribution matching, iterative alignment, or relative-representation approaches. Paired transport is worse than direct target fitting at k≤8, roughly tied at 16, begins to help around 32, and is consistently better by 64. The procedure uses paired activations from both models and is calibration, not zero-shot transfer. Encoder, interaction, and diagonal factorial terms are alignment-confounded; only the writer term has an exact zero null under no writer effect, and its magnitude can still be distorted. Finally, this is an exploratory post-hoc LMSYS-only pilot; other model families, tasks, layers, and genuinely new prompts remain necessary tests.

## Reproducibility

- Source experiment revision: `8d2694f6eedfbad61b9413299bca096370429d7a`
- Test roster SHA-256: `ed888b899ad83bc4dc42785fc9624012787734059ed88a985d8c76008dfe8602`
- Primary outputs: [`mapping_diff.json`](mapping_diff.json), [`fewshot_transfer.json`](fewshot_transfer.json), [`heldout_rows.jsonl`](heldout_rows.jsonl), [`writer_modes.npz`](writer_modes.npz)
- Analysis drivers: `scripts/issue2569_mapping_diff.py`, `scripts/issue2569_fewshot_transfer.py`, and `scripts/issue2569_mapping_diff_report.py`
