---
title: 'daily-fix: step9c baseline scan-set dirt wedge'
kind: infra
tags:
- wf-fix
- wf-fix-fp:639ef9672ac9
- daily-auto-filed
created_at: '2026-07-24T06:47:11Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): non-file-anchored scan-set
  nodes wedge compare at exit-2 INDETERMINATE whenever the shared root carries any
  third-party untracked code draft'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-23 parked-candidate routing pass (Step C). Raised as a recursion-guard-parked candidate on task #1632 (emitter confidence: LOW — filed anyway per the standing low-confidence-is-not-a-suppression-reason directive; the spawned session's planner may deflect with a reasoned no-change report).

## Goal

Stop non-file-anchored scan-set nodes in `scripts/step9c_baseline.py` from wedging the compare leg at exit-2 INDETERMINATE whenever the shared root carries ANY third-party untracked code draft.

## Workflow gap

- **Bug observed:** non-file-anchored scan-set nodes wedge compare at exit-2 INDETERMINATE whenever the shared root carries any third-party untracked code draft — effectively permanent on a busy fleet (#1632 hit it twice in one session; ≥5 autonomous sessions today had their pristine oracle dirtied by sibling sessions' untracked root scripts — issue1310/1092/823/1586 drafts; the #1341 observer escalates but nothing dispositions).
- **Why it is a workflow gap:** the Step 9c baseline-compare oracle is fleet-shared; a permanently-INDETERMINATE compare leg forces per-session manual strips.
- **Confidence (emitter):** low
- verified-at-filing: n/a — behavioral gap in the compare-leg certification path, not grep-verifiable as a single pattern; the incident evidence is #1632's session record (2 wedge hits, 2026-07-23) plus the recurring stripped-pre-existing-red pattern across ≥5 sessions in today's transcript sweep.

## Proposed change (candidate diff sketch — refine in planning)

Certify aggregate scan-set nodes from the sparse scratch worktree when the tree they scan is their own checkout (their scan root derives from the repo argument, not the shared root), or add a bounded dirt-quarantine oracle mode.

## Scope / surfaces

- Primary target: `scripts/step9c_baseline.py`

## Constraints / invariants

- Workflow-surface only. Fail-loud contract of the oracle must be preserved — the fix narrows the false-INDETERMINATE class, never converts a real red to green.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 639ef9672ac9

- workflow_fix_target: scripts/step9c_baseline.py

Origin: parked candidate on #1632 (2026-07-23T18:21:33Z).
