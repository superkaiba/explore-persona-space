---
title: '[Running] Aim 2-3: Comprehensive Trait Leakage (Phase A1)'
kind: experiment
tags: []
created_at: '2026-04-16T19:30:30.000Z'
has_clean_result: false
sagan_id: 1012e6e5-abcd-4cca-80d0-01cec3a83d75
sagan_number: 27
priority: high
legacy_why_unset: true
---
**From EXPERIMENT_QUEUE.md — Running** (started 2026-04-14)

Plan: `.claude/plans/shimmying-hopping-locket.md` (v3, 2 adversarial review rounds).

Phase 0.5 pilot COMPLETE: Gate PASS (rho=0.56, p_one=0.058, n=9 excl zelthari).

**Phase A1 COMPLETE:** 44/44 runs done (40 persona-conditioned + 4 controls).
- Results: rho=0.60 (p=0.004), partial r=0.66 (p=0.004). Capability rho=-0.40 (n.s.)
- Controls: shuffled persona eliminates gradient (rho=-0.006); cap hierarchy: persona-cond (4.0%) > generic (1.6%) > shuffled (0.5%)

Draft: `research_log/drafts/2026-04-14_phase_a1_analysis.md` (reviewer fixes + control analysis applied).

**Next:** Commit + push → Phase A2 (multi-seed replication with seeds 137, 256, 512).
