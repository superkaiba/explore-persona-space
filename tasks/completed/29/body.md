---
title: '[Under Review] Aim 2-3: Phase A3 Non-Contrastive Leakage'
kind: experiment
tags: []
created_at: '2026-04-16T19:30:32.000Z'
has_clean_result: false
sagan_id: 14a450bc-c8ea-416c-bc12-107d2b12c16e
sagan_number: 29
priority: high
legacy_why_unset: true
---
**From EXPERIMENT_QUEUE.md — Under Review** (reviewer dispatched 2026-04-15)

Plan: `.claude/plans/shimmying-hopping-locket.md` (A3 extension).
Code: `scripts/run_a3_leakage.py`, `scripts/analyze_a3.py`.

**ALL 6 conditions complete** on pod2: caps_doctor, wrong_doctor, misalign_doctor, benign_doctor, misalign_assistant, baseline.

**Key finding:** Non-contrastive SFT (no negative set, aggressive lr=2e-4, r=32, 3 epochs) produces globally UNIFORM trait transfer. 0/15 distance-leakage correlations survive Bonferroni. CAPS 100% everywhere, ARC-C collapses to 0.227 everywhere. The contrastive training design creates the distance gradient.

**Critical caveat:** Hyperparameter mismatch with A1 (which used lr=5e-5, r=16, 1 epoch). The comparison is confounded.

Draft: `research_log/drafts/2026-04-15_a3_leakage.md`.
Figures: `figures/a3_leakage/` (6 pub-quality figures).
Analysis: `scripts/analyze_a3.py` (884 lines, Spearman + permutation + Bonferroni + sensitivity).
Models: All 5 adapters on HF Hub (`superkaiba1/explore-persona-space/a3_leakage/`).
Results: `eval_results/a3_leakage/*/run_result.json`.
