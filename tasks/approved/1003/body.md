---
title: 'workflow-fix: gotchas rc=134 — in-process streaming-datasets release'
kind: infra
tags:
- wf-fix
- wf-fix-fp:35a510cc017e
created_at: '2026-07-04T09:05:04Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from #952 r2 implementer (see body Provenance
  block)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #952 (emitting agent: experiment-implementer).

## Goal

Extend the #654 rc=134 gotcha entry in `.claude/rules/gotchas.md` with the in-process prevention: deterministically release streaming `datasets` objects (`del ds; gc.collect()`) after consumption, never `os._exit(0)`.

## Workflow gap

- **Bug observed:** A `datasets` streaming IterableDataset surviving to interpreter shutdown SIGABRTs the process (rc=134, "terminate called without an active exception") AFTER all work completed and the final sentinel was written — deterministic in the pinned env; a clean GCE pod run would be classified as a workload crash by the EXIT trap (#952 r2, bisected with a bare 15-row streaming loop).
- **Why it is a workflow gap:** The existing #654 gotcha covers only the WRAPPER side (artifact-check before treating rc as fatal); no entry prescribes the in-process prevention, so every new streaming-`load_dataset` call site re-hits the false-failure class.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
- **HF `datasets` ... exit `rc=134` (SIGABRT) ... AFTER the work already completed ...**
+ ... RULE (wrapper side, unchanged): artifact-check before treating rc as fatal.
+ IN-PROCESS PREVENTION (#952 r2): a streaming IterableDataset surviving to
+ interpreter shutdown reliably triggers this abort — release it deterministically
+ after the consuming loop (`del row, ds; gc.collect()`; verified rc 134 -> 0).
+ Never mask with `os._exit(0)`. Worked example: run_952.phase0_verify.
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'rc=134\|SIGABRT' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 35a510cc017e

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: A `datasets` streaming IterableDataset surviving to interpreter shutdown SIGABRTs the process (rc=134, "terminate called without an active exception") AFTER all work completed and the final sentinel was written — deterministic in the pinned env; a clean GCE pod run would be classified as a workload crash by the EXIT trap (#952 r2, bisected with a bare 15-row streaming loop).
why_workflow_gap: The existing #654 gotcha covers only the WRAPPER side (artifact-check before treating rc as fatal); no entry prescribes the in-process prevention, so every new streaming-load_dataset call site re-hits the false-failure class.
proposed_change: Extend the #654 rc=134 gotcha entry with the in-process fix — deterministically release streaming datasets (del ds; gc.collect()) after consumption, never os._exit(0).
confidence: high
related_task: #952
<!-- /workflow-fix-candidate -->
