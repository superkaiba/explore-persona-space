---
title: 'workflow-fix: post-marker WARN on literal backslash-n notes'
kind: infra
tags:
- wf-fix
- wf-fix-fp:111730008892
- daily-auto-filed
created_at: '2026-07-09T06:59:05Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): task.py post-marker accepts
  a single-line field-led note containing literal backslash-n escape sequences without
  warning — the poster-side symptom that produced the #1120 unparseable follow-up
  notes; the landed #1120 fix was parse-side only (parse_followup_note_field hardening).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1120 (recursion-guarded workflow-fix session).

## Goal

Detect escaped-newline field-led notes at the single choke point every poster shares, so malformed notes are caught at post time rather than at parse time.

## Workflow gap

- **Bug observed:** task.py post-marker accepts a single-line field-led note containing literal backslash-n escape sequences without warning — the poster-side symptom that produced the #1120 unparseable follow-up notes; the landed #1120 fix was parse-side only (parse_followup_note_field hardening).
- **Why it is a workflow gap:** the fix targets the workflow surface (scripts/task.py); the originating session was recursion-guarded and could not route it.
- **Confidence (emitter):** see parked note below.

## Proposed change (candidate diff sketch — refine in planning)

```
# task.py post-marker, after reading --note/--file content:
if "\\n" in note and "\n" not in note and looks_field_led(note):
    print("WARNING: single-line field-led note contains literal backslash-n escapes; "
          "field parsers treat real newlines as separators — did you mean $'...' quoting?",
          file=sys.stderr)
```

## Scope / surfaces

- Primary target: `scripts/task.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- The spawned session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/task.py
- origin: parked candidate on task #1120 at 2026-07-08T01:44:09Z

Verbatim parked note:

> source: prose-followup (alternatives critic, plan round 1). Suggestion: scripts/task.py post-marker could emit a stderr WARN when a field-led note is single-line and contains literal backslash-n escapes — choke-point detection covering all posters, complementing the parse-side fix. routed: parked: EPM_WORKFLOW_FIX_SESSION — this session is a workflow-fix session (workflow_fix_target Provenance line); recursion guard forbids auto-filing (see .claude/rules/workflow-fix-on-bug.md § Recursion guard). Surfaced for the next human/orchestrator pass.
