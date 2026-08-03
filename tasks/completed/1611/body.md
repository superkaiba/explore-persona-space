---
title: 'workflow-fix: smoke must cover every behavior-class x regime (smoke/parity
  family)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b5fabf456761
created_at: '2026-07-23T03:33:47Z'
has_clean_result: false
origin_prompt: Fix all the problems in the background with happycoder (#1586 r3/r4/r6
  whack-a-mole from single-content-cell smoke)
workflow: v1
---
## Overview / Motivation

Auto-filed from a deep-dive on #1586's crash history (2026-07-23, user-directed).
#1586's pre-launch smoke ran ONE content-class cell (`syc-pers-ft-con-s137`), so the
marker-class, positive-only (`po`), and panel read-side code paths were NEVER
exercised before the full run. Three distinct bug classes then surfaced LIVE, one per
phase, over hours (whack-a-mole): r3 read-side organism panel disjointness, r4 marker
`po` mix-rowcount (200 vs copy-filled 1000), r6 reuse-seam schema. A smoke that
covered every (behavior-class x regime) would have caught all three pre-launch.

## Goal

Add a "regime/class-coverage" member to the smoke/production-parity family in
gotchas.md so multi-regime subprocess-per-phase dispatchers exercise every
class-specific code path in the smoke, not just one content-class cell.

## Workflow gap

- **Bug observed:** issue1586 pre-launch smoke ran one content-class cell so
  marker-class, positive-only, and panel read-side paths were unexercised and three
  bug classes (r3 read-side panel disjointness, r4 marker po mix rowcount, r6
  reuse-seam) surfaced live one per phase.
- **Why it is a workflow gap:** the smoke/production-parity family in gotchas.md
  covers gate calibration (#1345) and process width (#1315/#1333) but has no member
  requiring COVERAGE across (behavior-class x regime); a dispatcher spanning multiple
  behavior classes/regimes must smoke >=1 cell per class x regime.
- **Confidence (emitter):** high
- verified-at-filing: `grep -niE "full-panel|fresh-child smoke|subprocess-per-phase|regime|behavior-class" .claude/rules/gotchas.md` (2026-07-23) → the smoke/production-parity family exists (gate-calibration #1345, process-width #1315/#1333) but NO class×regime-coverage member.

## Proposed change (candidate diff sketch — refine in planning)

Add to the smoke/production-parity family in `.claude/rules/gotchas.md`:

    A subprocess-per-phase dispatcher spanning MULTIPLE behavior classes (e.g. marker
    vs content) and/or regimes (e.g. contrastive-negatives `con` vs positive-only `po`)
    MUST smoke >=1 cell per (class x regime), so each class's distinct code paths run
    pre-launch (marker parity read, `po` mix asserts, panel disjointness read) — not
    just one content-class cell. Incident #1586 r3/r4/r6.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md` (smoke/production-parity family).
- Consider a one-line cross-ref from `.claude/skills/issue/SKILL.md` Step 6d.0 (smoke)
  if the planner judges it warranted.

## Constraints / invariants

- Workflow-surface doc only; no experiment code.
- `scripts/workflow_lint.py` passes; if the lessons index or a sibling rule is touched,
  they stay consistent.
- Runs under `EPM_WORKFLOW_FIX_SESSION=1` (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: b5fabf456761

Origin: user chat 2026-07-23 ("Fix all the problems in the background with happycoder")
on the #1586 crash-history review. Related task: #1586 (r3/r4/r6).
