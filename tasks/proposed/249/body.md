---
title: Extract language-output direction vectors and correlate with spill magnitudes
kind: experiment
tags: []
created_at: '2026-05-05T01:15:33.000Z'
has_clean_result: false
sagan_id: f34d5d97-5239-48c1-ae32-5236d38ca491
sagan_number: 249
priority: normal
---
Parent: #190

## Motivation

Quick analysis on the #190 spill matrix shows that hand-coded typological distance predicts contamination magnitude (Spearman rho=-0.52, p=0.003, N=30 bystander cells). But the hand-coded distances are crude (4 levels). The proper test: extract actual language-output direction vectors from Qwen-2.5-7B-Instruct's hidden states and compute pairwise cosine similarity / JS divergence, then correlate with the spill rates.

Key questions:
1. Does representation-space distance (cos sim of language directions) predict spill better than typological distance?
2. Do the German outliers (66-95% contamination despite being "different branch") sit closer to Romance languages in representation space than typology suggests?
3. Is there a linear relationship between representation distance and spill magnitude, or is it a threshold effect?

## Design sketch

1. Load Qwen-2.5-7B-Instruct on an H100
2. Run forward passes on "Speak in {language}." for all 7 languages × multiple prompts
3. Extract hidden-state vectors at the last token position (or mean-pool across response tokens from baseline completions)
4. Compute pairwise: cosine similarity, JS divergence of output logit distributions
5. Correlate with the 7×7 spill matrix from #190
6. Visualize: 2D MDS/UMAP of the 7 language directions, colored by contamination received

## Compute
~1 GPU-hr (forward passes only, no training). `compute:small`.

## Artifacts needed
- Base model: Qwen-2.5-7B-Instruct (already cached on pods)
- Spill matrix: from #190 eval results (already on git)
- Baseline completions: from #162/#190 eval results (already on git)
