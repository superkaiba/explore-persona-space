---
title: '[Under Review] Aim 2-3: Phase A3b Factorial'
kind: experiment
tags: []
created_at: '2026-04-16T19:30:33.000Z'
has_clean_result: false
sagan_id: 634d446a-2140-44d5-a3ae-a58c6e12e824
sagan_number: 30
priority: high
---
**From EXPERIMENT_QUEUE.md — Under Review** (reviewer dispatched 2026-04-15)

Plan: `.claude/plans/shimmying-hopping-locket.md` (A3b extension).
Code: `scripts/run_a3b_experiment.py`, `scripts/generate_a3b_data.py`.

**ALL 7 conditions complete** on pod2: 2×2 factorial (contrastive/non-contrastive × aggressive/moderate) + 2 partial contrastive + 1 moderate misalign.

**Key finding:** Contrastive design is the sole determinant of containment vs leakage. Non-contrastive + moderate (2K examples, 1ep) still produces 92-98% uniform CAPS adoption. Contrastive + aggressive (10K, 3ep) achieves 0% leakage. Partial negative set (4/8 bystanders) equally effective as full (IN vs OUT delta ≤ 0.01). No distance gradient in any condition (0/7 significant Spearman for trained trait).

**Resolves A3 confound:** The uniform leakage in A3 was caused by non-contrastive design, NOT aggressive hyperparameters.

Draft: `research_log/drafts/2026-04-15_a3b_factorial.md`.
Figures: `figures/a3b_factorial/` (6 pub-quality figures + PDFs).
Models: All 7 adapters on HF Hub.
Results: `eval_results/a3b_factorial/*/run_result.json`.
