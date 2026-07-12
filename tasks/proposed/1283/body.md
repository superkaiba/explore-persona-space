---
title: 'workflow-fix: filer-side wf-fix title-prefix WARN'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3b86e8ab8de7
- daily-auto-filed
created_at: '2026-07-12T06:52:44Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-11 problem sweep (route 2): file_infra_task.py accepts
  a wf-fix-tagged filing whose --title lacks the workflow-fix:/daily-fix: title prefix
  without warning — the title prefix is a PRIMARY dedup key surface, and only the
  daily_drive_filings.py batch path guards it after #1273; a direct single-item invocation
  has no guard'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-11 from a prose follow-up recorded (recursion-guarded) in task #1273's `epm:plan` + `epm:done` notes.

## Goal

Add a filer-side WARNING in `scripts/file_infra_task.py` when a wf-fix-tagged filing's `--title` lacks a `WF_FIX_TITLE_PREFIXES` prefix, complementing #1273's driver-side `_effective_title()` guard.

## Workflow gap

- **Bug observed:** `file_infra_task.py` accepts a wf-fix-tagged filing whose `--title` lacks the `workflow-fix:` / `daily-fix:` title prefix without warning — the title prefix is a PRIMARY dedup-key surface (`task_workflow.WF_FIX_TITLE_PREFIXES`), and after #1273 only the `daily_drive_filings.py` batch path guards it; a direct single-item invocation has no guard.
- **Why it is a workflow gap:** a prefix-less wf-fix filing is invisible to the dedup predicate's title leg, so the same bug can double-file later.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "prefix\|WARN" scripts/file_infra_task.py` → the only WARNING is the missing-Provenance-line one (line 231); no title-prefix check exists in the filer (2026-07-12).

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from #1273's recorded prose follow-up) In `file_infra_task.py`, when the tag set includes `wf-fix` and the `--title` does not start with any `task_workflow.WF_FIX_TITLE_PREFIXES` entry, print a stderr WARNING (mirror the line-231 Provenance WARN shape; do not block). Optionally auto-prepend like the driver, if the plan review prefers parity over warn-only.

## Scope / surfaces

- Primary target: `scripts/file_infra_task.py`
- Keep consistent with `daily_drive_filings.py` `_effective_title()` (#1273) and `task_workflow.WF_FIX_TITLE_PREFIXES`.

## Constraints / invariants

- Workflow-surface only; ruff passes; existing filer tests stay green.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/file_infra_task.py
- fingerprint: 3b86e8ab8de7

Origin (recorded prose follow-up on #1273, 2026-07-11): "Logged, not filed (recursion guard): a possible follow-up — filer-side WARN in `file_infra_task.py` when a wf-fix-tagged filing's `--title` lacks a prefix — is recorded in the `epm:plan` + `epm:done` notes for the nightly /daily sweep to pick up."
