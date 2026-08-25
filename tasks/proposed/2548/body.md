---
title: 'verify_task_body: per-unit-evidence check vocabulary misses the draw grain
  — 3 false WARNs on per-draw figure captions'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-24T18:01:38Z'
has_clean_result: false
parent_id: 2388
origin_prompt: 'clean-result-critic round 2 on #2388 surfaced: add draw/per-draw to
  the per-unit vocabulary tuple in verify_task_body.py''s per-unit-evidence check;
  it produced 3 false WARNs on a body whose captions each declare per-draw points'
workflow: v1
---
# Goal

Fix a false-WARN class in `scripts/verify_task_body.py`: the per-result per-unit-evidence check (check 77) matches a fixed vocabulary of per-unit tokens and does not include "draw" / "per-draw", so a body whose figures legitimately show per-draw points (the label-draw grain of budget-ladder designs) collects false per-unit-evidence WARNs.

## Where it fired

Task #2388 clean-result gate round 2 (2026-08-24, Claude clean-result-critic). The live body's fig4/fig5 captions each declare per-draw points (the per-unit view for a label-budget ladder: individual draws at each budget), yet the check emitted 3 per-unit-evidence WARNs on those results. The critic verified from the pinned PNGs that per-draw points are actually rendered — the WARNs are pure vocabulary misses.

## Required behavior

1. Add the draw grain ("draw", "per-draw", "per draw") to the per-unit vocabulary tuple in the per-unit-evidence check in `scripts/verify_task_body.py`.
2. Sweep the tuple once against the project's other recurring grains while in there (e.g. "per-seed" / "per-fold" style tokens) — add only grains that actually appear in landed v4 bodies; do not speculatively grow the list.
3. Extend `tests/test_verify_task_body.py`: a result block whose caption declares per-draw points passes the per-unit-evidence check without a WARN.

## Provenance

Surfaced as a workflow-fix prose follow-up by the Claude `clean-result-critic` (round 2) during `/issue 2388` (session cmt237vj5mej9xw0u4jf0bu9d, autonomous). Orchestrator auto-filed per `.claude/rules/workflow-fix-on-bug.md`. Distinct from #2545 (same target file, different check + fingerprint: HF count pagination vs per-unit vocabulary).

Target file: `scripts/verify_task_body.py` (per-unit-evidence check). Fingerprint: per-unit-evidence-vocab-missing-draw-grain.
