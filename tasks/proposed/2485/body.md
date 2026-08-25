---
title: 'verify_plan: WARN on phase-entry headroom asserts jointly unsatisfiable with
  the plan''s own §9 disk rows (#1901 v9 p2-gate shape)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-22T22:04:31Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate: verify_plan gap — phase-entry headroom vs
  booked resident store (methodology ensemble on #1901 v9)'
workflow: v1
---
## Goal
Add a `verify_plan.py` check (WARN-class) that a plan's phase-entry disk-headroom asserts are jointly satisfiable with its own §9 disk rows — the #1901 v9 shape: the plan registered `assert_out_root_headroom(out_root, 70)` before BOTH p1 and p2 while its own §9 row books 0.4 GB staging + ≤61 GB store + ~2 GB scratch on a ~130 GB quota — at p2 entry the conjunction (61 resident + 70 free ≈ 133 GB > 130) cannot pass on any lane where statvfs reflects the real quota, and the flat re-assert is not resume-aware (a late-p1 resume sits in the same window). The plan PASSed verify_plan; the Claude methodology critic flagged it as a concern and the Codex methodology critic + reconciler elevated it to a binding fix. Sibling rule: .claude/rules/plan-compute-sizing.md § Out-root mount binding (pending-aware floors); crash class #1586.

## Proposed shape
When a plan names a numeric headroom assert (regex family: `assert_out_root_headroom\([^)]*,\s*(\d+)`or a "≥ N GB free" phase-entry contract) at ≥2 phases, cross-check against the same section's booked resident footprints (store/staging GB figures within the disk row): WARN when max(booked resident) + asserted free > the row's stated quota/disk size, or when the same flat floor is asserted after a phase that books a resident store ≥ half the quota. Textual, WARN-only, N/A escape for plans with no phase-entry headroom asserts.

## Provenance
Surfaced by the Claude methodology critic (concern) + codex-methodology critic (Must-Fix) + methodology reconciler (binding carried fix) on task #1901 plan v9 (round generic-boundary-token-control, 2026-08-22). workflow-fix-candidate routing per .claude/rules/workflow-fix-on-bug.md.
