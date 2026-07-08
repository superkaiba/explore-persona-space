---
title: 'workflow-fix: parse_followup_note_field misses literal backslash-n notes'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5e09182a9dd2
created_at: '2026-07-08T01:10:44Z'
has_clean_result: false
origin_prompt: 'orchestrator-own-observation on /issue 825 round-8 dispatch: run markers
  v6/v7 with literal \n escapes parsed no label; completed round showed unrun; retro-closed
  at v8'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #825 (emitting agent: issue-orchestrator, own observation).

## Goal

Harden `parse_followup_note_field` to treat literal backslash-n escape
sequences as line separators so field-led follow-up notes posted with escaped
newlines still parse.

## Workflow gap

- **Bug observed:** `epm:same-issue-followup-run` markers v6/v7 on #825
  (2026-07-08T00:50-00:51Z) carried literal `\n` two-char escape sequences
  instead of real newlines; `parse_followup_note_field(note, "followup_label")`
  returned `'onpolicy-separator-control\\nsource:'` instead of the label, so
  `unrun_followup_labels` reported the completed round as UNRUN and the next
  `/issue 825` entry nearly re-dispatched a finished 4-GPU-h round (caught by
  the Step 0 stale-label disposition rule + `followup_retro_close_evidence`;
  retro-closed at v8).
- **Why it is a workflow gap:** the parser already tolerates three note
  shapes (line-initial one-per-line, `; `-joined single-line, dash-bullet /
  bold variants — the #1090/#841 hardening) but not the escaped-newline shape
  a poster produces when a shell `--note "...\n..."` string is passed without
  interpretation. Round caps and label closure both silently undercount on
  this shape (the #1090 fu1 regression class), and the same miss affects
  every consumer: `unrun_followup_labels`, `executing_followup_label`, the
  cheap-band/expensive-band round caps, and the watcher's on-behalf run-marker
  post.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
 def parse_followup_note_field(note: str, field: str) -> str | None:
+    # Notes occasionally arrive with literal backslash-n escapes instead of
+    # real newlines (shell --note strings; #825 v6/v7). Normalize them to
+    # newlines BEFORE line splitting so field-led parsing survives.
+    if "\\n" in note and "\n" not in note.replace("\\n", ""):
+        note = note.replace("\\n", "\n")
     ...existing line/`; `-join parsing unchanged...
```

(Planner refines the normalization predicate — e.g. unconditional
`note.replace("\\n", "\n")` pre-pass restricted to the field-scan copy, never
mutating stored state — plus a regression test pinning the #825 v6/v7 note
shape verbatim.)

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py`
  (`parse_followup_note_field`; check sibling helpers
  `followup_label_groups` / `unrun_followup_labels` /
  `executing_followup_label` consume the hardened parse transparently)
- Tests: `tests/test_task_workflow*.py` — add a literal-escape fixture from
  the #825 v6 note.
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'parse_followup_note_field' .claude/ CLAUDE.md scripts/ src/explore_persona_space/task_workflow*.py`)
  and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py
- fingerprint: 5e09182a9dd2

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/task_workflow.py
bug_observed: epm:same-issue-followup-run markers v6/v7 on #825 carried literal backslash-n escapes; the label parse returned garbage and the completed round showed as unrun, nearly triggering a duplicate re-run
why_workflow_gap: parse_followup_note_field tolerates line-initial / `; `-joined / dash-bullet note shapes but not escaped-newline notes, so label closure and round caps silently undercount on that shape
proposed_change: harden parse_followup_note_field to treat literal backslash-n escape sequences as line separators so field-led follow-up notes posted with escaped newlines still parse
diff_sketch: |
  + if "\\n" in note: normalize literal escapes to newlines in the field-scan
  +   copy before line splitting (parse-side only; stored notes untouched)
  + regression test pinning the #825 v6/v7 note shape
confidence: high
related_task: #825
<!-- /workflow-fix-candidate -->
