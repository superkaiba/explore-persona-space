---
title: 'workflow-fix: self-heal crash-truncated events.jsonl line on append'
kind: infra
tags:
- wf-fix
- wf-fix-fp:eb776453ca97
created_at: '2026-07-15T22:41:41Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate v1 from #1333 9a-ter implementer: events.jsonl
  unterminated-partial-line + O_APPEND concatenation corrupts the next marker; seal
  missing trailing newline before append'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1333 (emitting agent: experiment-implementer, 9a-ter free-analysis round).

## Goal

Make the events.jsonl marker-append self-healing against a crash-truncated final line (seal a missing trailing newline before appending, with a warning).

## Workflow gap

- **Bug observed:** a `task.py post-marker` invocation killed mid-append left an unterminated partial JSON line at the end of tasks/interpreting/1333/events.jsonl; the retry's O_APPEND write concatenated its row onto the partial, producing ONE merged unparseable line (parse error at char ~1680 = the partial's length) that every reader must skip (task.py readers warn+skip; no content lost, but the NEXT marker was corrupted by the PRIOR crash).
- **Why it is a workflow gap:** the marker-append helper in src/explore_persona_space/task_workflow.py never verifies the file ends with a newline before appending, so any crash-truncated final line silently corrupts the next marker instead of just itself.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -rn "SEEK_END|trailing newline|missing trailing" src/explore_persona_space/task_workflow.py` -> 1 hit (line 3191, a docstring about fixture trailing newlines — no existing self-heal on the append path); the corrupted line is live at tasks/interpreting/1333/events.jsonl line ~152 as evidence (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

+ # self-heal a crash-truncated final line before appending (#1333 fu incident)
+ if path.exists() and path.stat().st_size > 0:
+     with open(path, "rb") as f:
+         f.seek(-1, os.SEEK_END)
+         if f.read(1) != b"\n":
+             logger.warning("events.jsonl missing trailing newline — sealing truncated line")
+             newline_prefix = b"\n"
  with open(path, "ab") as f:
-     f.write(row_bytes)
+     f.write(newline_prefix + row_bytes)

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py` (the events append path; in-scope workflow surface despite the general src/ exclusion)
- Grep the workflow surface for other O_APPEND jsonl writers (`grep -rln '"ab"' src/explore_persona_space/task_workflow.py scripts/`) and apply the same seal where the same crash-concatenation class applies; list hits in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff passes; tests pin the seal (append onto a truncated fixture produces two parseable lines).
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 and carries a workflow_fix_target: Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py
- fingerprint: eb776453ca97

(Verbatim candidate block from the emitting agent's report is preserved in the origin_prompt.)
