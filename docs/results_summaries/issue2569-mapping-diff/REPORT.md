# Mapping differences and few-query transfer across Qwen and Llama

**Issue #2569 follow-up · Qwen2.5-7B-Instruct Q14 × Llama-3.1-8B-Instruct L16 · 10,000 LMSYS prompts**

## Bottom line

A single fixed coordinate alignment does not make the two context→answer maps identical. After the shared Procrustes transform, encoder-dependent and answer-writer-dependent mapping changes remain and are nearly orthogonal in operator space (cosine -0.049). But the alignment is incomplete (held-out context/answer cosines only 0.657, 0.708, and 0.698), so the encoder and diagonal contrasts cannot be interpreted as pure behavioral effects. The writer contrast is cleaner because it differences answer writers within each encoder, canceling a shared alignment residual.

The geometry is nevertheless calibratable with paired anchors. A small directional advantage first emerges around 32 queries (median paired Δ cosine 0.030; 92.5% of 40 repeat-cells) and is consistent by 64 (100.0% positive). At 256 queries, median centered cosine with the full target map is 0.815, versus 0.732 from scratch. This supports a shared correspondence that paired examples can identify; it does not establish that marginal statistics alone can identify it.

## 1. Factorial mapping diff

![Held-out factorial mapping contrasts](fig1_factorial_mapping_diff.png)

**Figure 1.** Four native maps—encoder {Qwen, Llama} × answer writer {Qwen, Llama}—were transformed into one fixed Qwen basis using train-only semi-orthogonal Procrustes alignments. Black marks show the 95% row-permutation null. Encoder and diagonal contrasts retain positive held-out magnitude R² (0.206 and 0.191), but these two contrasts retain coordinate-alignment residual. Writer and interaction contrasts cancel a shared alignment residual; they have informative direction (cosine 0.286 and 0.129) but miscalibrated magnitude (negative R²). Every observed cosine exceeds all 1,000 row-pairing permutations (p=0.0010).

Numerically, the encoder-labeled representation contrast is largest and most split-half stable: exercised held-out RMS norm 7.51 and split-half cosine 0.809. The cleaner writer contrast has RMS 5.67 and split-half cosine 0.688; interaction is smaller and less stable (2.46, 0.493). The diagonal result replicates numerically on seed 137 (R² 0.191), but remains alignment-confounded.

## 2. Can one mapping be transferred with only residual summaries or a few queries?

![Few-query mapping transfer](fig2_fewshot_transfer.png)

**Figure 2.** Panel A evaluates the direction of the original full-data target mapping on the untouched 1,500-row test set. Thin lines are the four direction/writer cells; thick lines are their medians. The 0-query diamond uses separate mean/top-64 PCA/variance summaries with components paired only by variance rank and marginal skewness. This particular heuristic fails (centered cosine -0.002–0.020), despite retaining 61.9%–79.1% of residual energy. Stronger unsupervised alignment algorithms were not tested. Panel B shows paired within-anchor-set cosine advantage over the equal-query scratch control (median and pooled 10th–90th percentiles across 4 × 10 runs).

For k paired train queries, regularized context and answer bridges use only sample means and centered Gram matrices. No validation rows tune the bridge. The frozen source map is applied between those two bridges. The control fits target context→answer directly from the identical k anchors.

Table entries are **transported source map / target fit from scratch**, measured by held-out centered cosine with the full target mapping:

| Direction / writer | 16 queries | 32 queries | 64 queries | 128 queries | 256 queries |
|---|---:|---:|---:|---:|---:|
| Q→L · Q writer | 0.345 / 0.360 | 0.488 / 0.455 | 0.643 / 0.595 | 0.731 / 0.667 | 0.798 / 0.729 |
| Q→L · L writer | 0.319 / 0.316 | 0.500 / 0.496 | 0.641 / 0.589 | 0.722 / 0.659 | 0.795 / 0.723 |
| L→Q · Q writer | 0.407 / 0.402 | 0.577 / 0.524 | 0.706 / 0.650 | 0.792 / 0.713 | 0.853 / 0.764 |
| L→Q · L writer | 0.362 / 0.361 | 0.537 / 0.506 | 0.674 / 0.609 | 0.772 / 0.681 | 0.832 / 0.735 |

At 16 queries, transport is directionally indistinguishable from scratch. A small advantage appears at 32, and at 64 every one of the 40 paired repeat-cells favors transfer. Scale-sensitive normalized R² also favors transfer, but is secondary because the two-stage transport and one-stage scratch fits have different shrinkage; centered cosine is the primary geometric comparison.

## 3. What behavioral differences are visible?

![Behavior readouts from the writer contrast](fig3_behavior_readout.png)

**Figure 3.** A ridge readout trained on the observed writer activation contrast can recover several answer differences, but only some survive when the writer contrast is predicted from context through the mapping diff. Mapping-mediated semantic-divergence R² is 0.267 and replicates at 0.321 on seed 137. Length is weak (0.040; seed 137 0.022); refusal and repetition are near or below zero. Thus the map difference carries a reproducible semantic-divergence signal in this run, but this pilot does not support strong claims about refusal or repetition differences.

## Exact design

- Frozen split: 8,000 train / 500 validation / 1,500 test prompts. The new analyses never fit on test rows.
- Fixed common basis: context Procrustes on Qwen-writer train contexts; answer Procrustes pooled over both answer writers. All affine translations are retained. Held-out context/answer alignment cosines are 0.657/0.708/0.698, so encoder and diagonal contrasts include residual alignment error.
- Factorial contrasts: writer, encoder, encoder×writer interaction, and the natural diagonal Qwen-own − Llama-own difference.
- Null: 1,000 held-out row-pairing permutations that destroy prompt correspondence.
- Stability: two disjoint 4,000-row train refits at the original selected ridge lambdas; independent generation seed 137.
- Few-query transfer: both Qwen→Llama and Llama→Qwen, both answer writers, k ∈ {2,4,8,16,32,64,128,256}, 10 random anchor sets per cell, fixed ridge fraction 0.01, no validation tuning.
- Primary transfer score is centered cosine with the full target-map prediction. The scale-sensitive normalized R² is retained in the JSON as a secondary diagnostic, not the headline comparison.

## Interpretation and limits

Variance-ranked, skewness-oriented PCA summaries do not recover the correspondence; this does **not** rule out stronger unsupervised methods such as distribution matching, iterative alignment, or relative-representation approaches. Paired calibration produces a genuine directional advantage beginning around 32 queries and a robust advantage by 64. The procedure uses paired activations from both models and is calibration, not zero-shot transfer. The encoder/diagonal factorial terms are alignment-confounded, whereas writer/interaction terms cancel a shared alignment residual. Finally, this is an exploratory post-hoc LMSYS-only pilot; other model families, tasks, layers, and genuinely new prompts remain necessary tests.

## Reproducibility

- Source experiment revision: `8d2694f6eedfbad61b9413299bca096370429d7a`
- Test roster SHA-256: `ed888b899ad83bc4dc42785fc9624012787734059ed88a985d8c76008dfe8602`
- Primary outputs: [`mapping_diff.json`](mapping_diff.json), [`fewshot_transfer.json`](fewshot_transfer.json), [`heldout_rows.jsonl`](heldout_rows.jsonl), [`writer_modes.npz`](writer_modes.npz)
- Analysis drivers: `scripts/issue2569_mapping_diff.py`, `scripts/issue2569_fewshot_transfer.py`, and `scripts/issue2569_mapping_diff_report.py`
