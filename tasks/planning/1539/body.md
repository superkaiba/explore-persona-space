---
title: 'workflow-fix: never-run claims scan live pods+markers'
kind: infra
tags:
- wf-fix
- wf-fix-fp:244c82ba276e
- daily-auto-filed
created_at: '2026-07-19T07:07:20Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): A ''nothing live is generating
  that cell'' chat claim was read from a task-status-folder scan while #1345''s parked-task
  inline pod round had already generated 2,019 stories; the Verify-before-asserting
  bullet mandates only an eval_results artifact check (c4-P5).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the 2026-07-18 /daily
transcript problem sweep (c4-P5). Route-2 filing.

## Goal

Extend the CLAUDE.md "Verify before asserting" bullet so a never-run /
nothing-live-generating claim additionally scans LIVE compute — running pods
(`pod.py list-ephemeral`), GCE instances, and recent `epm:run-launched` /
`epm:followup-scope` / `epm:free-analysis-followup-run` markers on PARKED
tasks — before asserting nothing is producing a cell.

## Workflow gap

- **Bug observed:** a chat claim that nothing live was generating a cell was
  read from a task-status-folder scan, while #1345's inline
  `onpolicy_assistant_story` round (a PARKED task, signalled only pod-side)
  had already generated 2,019 stories with a character named Assistant that
  night; Thomas caught it.
- **Why it is a workflow gap:** the "Verify before asserting" bullet mandates
  an `eval_results/` artifact check for a number / winner / never-run claim,
  but says nothing about scanning LIVE generation (pods, GCE, parked-task run
  markers) — a never-run/nothing-live claim can be false while every
  task-status folder looks idle.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'never-run\|nothing-live\|live pods\|list-ephemeral' CLAUDE.md` → the "Verify before asserting" bullet (line ~185) scopes verification to `eval_results/` JSON / named HF/WandB artifacts only; the only `list-ephemeral` / GCE hits (lines 311, 328) are the pod-CLI doc + the GCE janitor cron, NOT the verify-bullet — so no live-generation-scan duty exists there (2026-07-19). (`.claude/agents/experiment-status.md` lives under `~/.claude/` — user-global, out of the project workflow surface — so it is NOT a target here.)

## Proposed change (candidate diff sketch — refine in planning)

```
# In CLAUDE.md "Ad-hoc results summaries ... Verify before asserting:" bullet,
# add after the eval_results artifact-check sentence:
+ A never-run / nothing-live-generating claim ALSO scans live compute before
+ asserting: `pod.py list-ephemeral`, the GCE instance list, and recent
+ epm:run-launched / epm:followup-scope / epm:free-analysis-followup-run
+ markers on PARKED tasks (an inline pod round on a parked task signals
+ pod-side only and leaves every task-status folder idle-looking; #1345).
```

## Scope / surfaces

- Primary target: `CLAUDE.md`
- Single bullet edit; keep it one sentence appended to the existing
  Verify-before-asserting clause.

## Constraints / invariants

- Workflow-surface only. Do not add a mechanical gate — this is an assertion-
  discipline rule, same register as the surrounding bullet.
- `scripts/workflow_lint.py --check-asks` passes.
- Recursion guard applies (`workflow_fix_target:` Provenance line).

## Provenance

- workflow_fix_target: CLAUDE.md
- fingerprint: ecd694722516

Surfaced problem (c4-P5): a "nothing live is generating that cell" chat claim
was refuted by #1345's parked-task inline pod round (2,019 stories generated,
character named Assistant); the user had to correct it.
