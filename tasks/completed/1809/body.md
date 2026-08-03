---
title: 'daily-fix: chat claims — sweeps, counts, sunk compute, doc f'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9e3adbaf45e0
- daily-auto-filed
created_at: '2026-07-29T07:15:08Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): five user-correction classes
  in 2026-07-28 chats: a ''largest n ever tested = 3,600'' headline committed+pushed
  while #779 had fit n=963,444; a theory plan called kernel/MLP fits ''new compute''
  with same-protocol fits banked at 50k and 963k (two corrections); ''12 vs 36 vs
  56 arms'' took 3 asks and a GPU-h question was re-asked; a discard directive was
  ack''d without leading with the ~44 A100-h sunk-vs-'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Sources: group-D P2/P3/P4/P6, group-I P9 (2 miners, one interactive session cluster).

## Goal

Extend the compose-time verification / presentation-register discipline with five user-correction classes from 2026-07-28's interactive sessions.

## Workflow gap

- **Bug observed:** (1) A 'largest n ever tested = 3,600' figure was committed and pushed; Thomas corrected it — #779's scaling fits reach n=963,444 (the wrong ceiling also drove several turns of sizing advice). (2) The four-arm theory plan claimed kernel/MLP fits were 'new compute'; #779 had them banked at 50k AND 963k — two corrections before the doc was fixed. (3) '12 vs 36 vs 56 arms' took 3 asks to define; a GPU-hours question was answered in milestones and had to be re-asked. (4) When Thomas ordered removing the 200-draw bootstrap, the ack led with mechanics rather than the ~44 A100-h already-paid split — a 2-turn round trip ending in a reversed directive. (5) A prior session's null-space analysis existed only in chat; Thomas had to ask whether it ever reached the doc (it hadn't).
- **Why it is a workflow gap:** the compose-time re-grep clause covers numeric headline re-reads but not superlative/coverage sweeps, chat-authored plan docs are outside the /issue reuse checklist, and no clause covers count-definition, sunk-compute-led acks, or same-turn doc folds for chat analysis.
- **Confidence (emitter):** medium (interaction patterns read verbatim from transcripts; fixes are prose-rule additions)
- verified-at-filing: `grep -c 'Compose-time re-grep' CLAUDE.md` → 1 (clause exists; extensions are additive); artifact-reuse.md governs /issue plans only (no chat-doc clause) (2026-07-29 UTC).

## Proposed change (candidate diff sketch — refine in planning)

Five short clauses at the § 'Ad-hoc results summaries' / § 'Interim/chat writeup presentation register' family + one artifact-reuse.md line for chat-authored plan docs.

## Scope / surfaces

- Primary targets: `CLAUDE.md`, `.claude/rules/artifact-reuse.md`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: 9e3adbaf45e0

- workflow_fix_target: CLAUDE.md

