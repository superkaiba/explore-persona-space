---
title: 'workflow-fix: gotchas entry — token-id concat for teacher-forced capture (BPE
  seam)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:cc89345f2480
created_at: '2026-07-08T04:56:22Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha candidate from task #1092 round 8.4 (epm:failure-lesson
  v2, 2026-07-08T04:55:23Z)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson gotcha candidate raised on task #1092 (emitting agent: experiment-implementer, round 8.4).

## Goal

Add a `.claude/rules/gotchas.md` entry: teacher-forced capture inputs must concatenate per-segment TOKEN IDS (never re-tokenize the concatenated string) and derive positions from offset mappings — BPE seam merges shift every downstream position and silently misalign captures.

## Workflow gap

- **Bug observed:** issue #1092's G2 identity gate failed at max_abs 2.9375 after a full 8xA100 cell: the capture tokenized `prompt + completion + boundary` as ONE string while computing positions from per-segment token counts; three live seam merges on the pinned Qwen tokenizer shifted teacher-forced capture positions (a "\n"-leading completion into the instruct "assistant\n" tail; a rstripped naturalistic prefix's "." into ".\n\n"; a "\n"-ending completion into "\n\nUser:").
- **Why it is a workflow gap:** gotchas.md carries the codebase-trap ledger; it has the gen-time zero-width-SPAN flavor of the BPE-seam trap (#825) but NOT the teacher-forced-capture position-alignment flavor, which poisons every downstream activation read and is only catchable by a G2-style identity gate.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

+ gotchas.md new entry "Teacher-forced capture: concatenate per-segment TOKEN IDS, never re-tokenize the joined string":
+ per-segment token-id concat (prompt segment bit-identical to what generation consumed); offset-mapping-derived intra-prompt boundaries; fail-loud padding_side assert; keep a G2-style teacher-forced-vs-generate-hook identity gate in every capture rig (CPU fp32 repro pins the defect exactly).
+ Cross-ref: #825 zero-width-span sibling entry; reference impl `_capture_row_ids_and_positions` (scripts/issue1092_gpu_phase.py @ a51add173d); agent-memory feedback_teacher_forced_capture_token_id_concat.md.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'BPE\|seam\|offset_mapping' .claude/ CLAUDE.md`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: cc89345f2480

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: P2/P3 activation capture + G2 identity gate (issue1092_gpu_phase.py)
lesson: Never compute token positions from per-segment counts while forwarding a re-tokenized CONCATENATED string — BPE merges at segment seams shift every downstream position and silently misalign teacher-forced captures; the G2 identity gate caught it as max_abs=2.9. Build teacher-forcing inputs by concatenating per-segment TOKEN IDS and derive intra-prompt boundaries from offset mappings; assert padding_side matches the position-indexing convention.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
