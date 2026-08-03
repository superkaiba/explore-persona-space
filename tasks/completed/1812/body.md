---
title: 'daily-fix: 9a-ter — instrument supersession + addenda charac'
kind: infra
tags:
- wf-fix
- wf-fix-fp:362c14ea9057
- daily-auto-filed
created_at: '2026-07-29T07:17:01Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): two 2026-07-28 inline-round
  gaps: three live SAE rounds were about to burn Batch-API judge calls on labels a
  just-designed better instrument (#1773) would supersede — only the user''s questions
  (''is this running already?'', ''can you pause those runs'') triggered the freeze;
  and the user pushed ''parallel + vectorized'' twice because scope-extension ADDENDA
  to inline rounds did not carry the compute-cha'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Sources: group-J P3 + P7.

## Goal

Close two gaps in the user-chat inline round duties (Step 9a-ter / the CLAUDE.md carve-out).

## Workflow gap

- **Bug observed:** (a) While #1773 (a stronger SAE-label judging instrument) was being designed, three live rounds using the known-weak labeling rubric kept running toward Batch-API judge spend; the freeze happened only after Thomas asked twice. (b) On the same afternoon Thomas had to re-state 'parallel + vectorized' twice before a throughput addendum landed — the compute-character statement binds round DISPATCHES, and scope-extension addenda slipped past it.
- **Why it is a workflow gap:** the carve-out's duties enumerate dispatch-time checks; neither instrument-supersession awareness nor addenda inheritance is named, so both defaulted to user vigilance.
- **Confidence (emitter):** medium (inferred from interaction patterns)
- verified-at-filing: `grep -c 'Compute-character pre-launch statement' .claude/skills/issue/SKILL.md` → 5 anchors (block exists; addenda not named) (2026-07-29 UTC).

## Proposed change (candidate diff sketch — refine in planning)

Two short additive clauses at Step 9a-ter (+ the CLAUDE.md carve-out mirror).

## Scope / surfaces

- Primary targets: `.claude/skills/issue/SKILL.md` (Step 9a-ter), `CLAUDE.md` (§ User-chat inline free analysis)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: 362c14ea9057

- workflow_fix_target: .claude/skills/issue/SKILL.md

