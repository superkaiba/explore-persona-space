---
title: '[Under Review] Aim 2-3: Marker Leakage v3 (Deconfounded)'
kind: experiment
tags: []
created_at: '2026-04-16T19:30:31.000Z'
has_clean_result: false
sagan_id: 4aa22580-837d-4530-9a28-a92f5b4821df
sagan_number: 28
priority: high
---
**From EXPERIMENT_QUEUE.md — Under Review** (reviewer dispatched 2026-04-15)

Plan: `.claude/plans/crispy-swinging-river.md` (evolved to v3 deconfounded design).
Code: `scripts/run_leakage_v3.py` on pod1.

**ALL 15 CONDITIONS COMPLETE** (5 conditions × 3 source personas, seed 42): Exp A (convergence→marker), Exp B P1 (marker only), Exp B P2 (marker→contrastive divergence), C1 (marker baseline), C2 (wrong convergence→marker).

**Key findings:**
1. Deconfounded leakage is real: sw_eng→asst=51%, librarian→asst=23.5%, villain→asst=0%.
2. Contrastive divergence (Exp B P2) reduces all to ~2%.
3. Convergence does NOT increase leakage.
4. Villain-comedian proximity: 46-70%.

Draft: `research_log/drafts/leakage_v3_deconfounded_results.md`.
Figures: `figures/leakage_v3/` (7 publication-quality figures).

**Caveat:** Single seed (n=1), no statistical tests. Needs multi-seed replication.

Results: [eval_results/leakage_v3/](eval_results/leakage_v3/)
