---
title: 'daily-fix: cap-park note duty on expensive-band path'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5c03d71a3b4d
- daily-auto-filed
created_at: '2026-07-21T06:38:11Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-20 problem sweep (route 2): the followup-parked-by-cap
  epm:progress note duty exists only on the C2 cheap-band cap-park path; the expensive-band
  autonomous cap park path posts no equivalent PM-surfaceable record'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-20 parked-candidate routing pass (Step C) from a workflow-fix candidate parked on task #1558 under the recursion guard (emitting context: Alternatives critic, #1558 plan review).

## Goal

Extend the `followup-parked-by-cap` `epm:progress` note duty to the EXPENSIVE-band autonomous cap-park path in `/issue` SKILL.md (source: proposer-9b, step 3 of the autonomous follow-up auto-spawn block), so an expensive-band follow-up parked by a cap leaves the same PM-surfaceable durable record the cheap band leaves.

## Workflow gap

- **Bug observed:** the C2 cheap-band block carries the `followup-parked-by-cap` note recipe, but the expensive-band autonomous auto-spawn block park path posts no equivalent note — a cap-parked expensive follow-up leaves no PM-surfaceable record. #1558's Edit 4 records the asymmetry in-file.
- **Why it is a workflow gap:** the same mechanical duty exists on one band only; parked follow-ups on the other band are invisible to the PM's pickup grep.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'followup-parked-by-cap' .claude/skills/issue/SKILL.md` → 6 hits (:6573/:6582/:6585/:6588/:6592/:7751), all inside the C2 cheap-band cap-park recipe and its pickup-grep documentation; context read of :7730-:7800 confirms the autonomous EXPENSIVE-band block (starting ~:7796) carries no equivalent note duty (2026-07-21).

## Proposed change (candidate diff sketch — refine in planning)

Add the same `followup-parked-by-cap`-shaped `epm:progress` note duty (distinct fingerprint fields for the expensive band, e.g. `source=proposer-9b`) to the expensive-band autonomous cap/park path in SKILL.md.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 5c03d71a3b4d

- workflow_fix_target: .claude/skills/issue/SKILL.md

Verbatim parked candidate (prose park on #1558, ts 2026-07-20T09:10:04Z):

> parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target, see workflow-fix-on-bug.md § Recursion guard. source: prose-followup (Alternatives critic, #1558 plan review). target_file: .claude/skills/issue/SKILL.md. proposed_change: extend the followup-parked-by-cap epm:progress note duty to the EXPENSIVE-band autonomous cap park path (source: proposer-9b, step 3 of the autonomous follow-up auto-spawn block, live line ~7789) — same mechanical duty, distinct fingerprint; #1558's Edit 4 records the asymmetry in-file. confidence: medium. related_task: #1558.
