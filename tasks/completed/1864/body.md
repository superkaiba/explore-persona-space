---
title: 'daily-fix: analyzer.md numbers the payload-lint gate pre-com'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e80390399ebd
- daily-auto-filed
created_at: '2026-07-30T07:09:10Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): #1775''s analyzer committed
  lint-red analysis scripts (heavy-import-before-load_dotenv) caught two review rounds
  later; analyzer.md carries the lint-gate duty only as a cross-reference sentence'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner D-P5 (session 3693ac94, #1775; probed)).

## Goal

The analyzer's own committed scripts must pass the payload lint gate before its push — the duty needs step-level prominence, not a cross-reference.

## Workflow gap

- **Bug observed:** Two analyzer-committed scripts/issue1775_* files imported heavy roots before load_dotenv() (the #847 header), failing tests/test_shared_vm_thread* two rounds later; the fix round's commits then stranded uncommitted when the session died to 529s and a successor re-landed them.
- **Why it is a workflow gap:** analyzer.md L277-278 states the gate binds before push, but as prose — compliance failed; a numbered step in the commit sequence is the cheap prominence fix.
- **Confidence (emitter):** medium
- verified-at-filing: miner probe: `grep -n lint .claude/agents/analyzer.md` -> L277-278 cross-ref exists (2026-07-30).

## Proposed change (refine in planning)

Insert the gate as its own numbered step (payload file -> inline_lint_gate.py -> PASS required) in the analyzer's commit/push sequence.

## Scope / surfaces

- Primary target: `.claude/agents/analyzer.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/agents/analyzer.md
- fingerprint: e80390399ebd
