---
title: 'workflow-fix: upload-policy.md — verify-path Hub calls ride retry_transient
  + prefix listings'
kind: infra
tags:
- wf-fix
- wf-fix-fp:85ae795d81ee
created_at: '2026-07-15T17:52:40Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha candidate from task #1335 r5 (hub_verify_unretried_file_exists_fallback);
  see body Provenance'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson gotcha candidate raised on task #1335 round 5 (emitting agent: experiment-implementer).

## Goal

Document in `.claude/rules/upload-policy.md` (verify-path subsection): verify sharded uploads with ONE prefix-scoped listing per destination repo and wrap every fresh Hub call on an upload/verify path in `hub.retry_transient` — a transport error (429/5xx/timeout) is retried, never fatal to the run.

## Workflow gap

- **Bug observed:** `hub.list_hf_files_under_path`'s exact-file fallback issued an UN-retried per-shard `api.file_exists` HEAD probe; one transient HF 429 ('maximum queue size reached') crashed #1335's healthy GCP run 2.8h in, AFTER its uploads had succeeded (attempt att-20260715-134136).
- **Why it is a workflow gap:** the upload-policy rule documents Hub-API verification (list_repo_files over the hf CLI) but not the transport-retry + prefix-batching discipline for verify paths; the same unretried-HEAD pattern exists as a temptation in every new upload/verify helper. The code fix (retry_transient + _batched_verify) lands on main via #1335's Step 10d merge; this task adds the durable rule text so future verify code inherits it.
- **Confidence (emitter):** high
- verified-at-filing: `grep -rn "retry_transient\|file_exists" .claude/rules/upload-policy.md` → 0 hits (2026-07-15) — no existing rule text covers verify-path transport retry; absence-of-guard claim, 0-hit in-target result IS the evidence.

## Proposed change (candidate diff sketch — refine in planning)

+ In .claude/rules/upload-policy.md (Hub-API verification section):
+ **Verify-path Hub calls ride retry_transient + prefix-scoped listings.** Verify a sharded upload with ONE prefix-scoped listing per destination repo (hub-retried internally), never a per-file `api.file_exists` loop; wrap fresh Hub calls on any verify path in `hub.retry_transient` (Retry-After-aware, budgeted). A transport error is retried, never fatal — an unretried HEAD probe let a single 429 kill #1335's run post-upload. Pin new verify code with a 429-then-success test + a ≤2-listings batching test (tests/test_upload_sharded.py, #1335 r5).

## Scope / surfaces

- Primary target: `.claude/rules/upload-policy.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'file_exists' .claude/ CLAUDE.md`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/upload-policy.md
- fingerprint: 85ae795d81ee

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: upload_store / upload_sharded._verify_present (store_r7 per-cell shard-upload verify)
lesson: hub.list_hf_files_under_path's exact-file fallback issued an UN-retried api.file_exists HEAD probe per shard, so one transient HF 429 killed a run AFTER its uploads had succeeded. Verify sharded uploads with ONE prefix-scoped listing per destination repo (hub-retried internally) and wrap every fresh Hub call on a verify path in hub.retry_transient — a transport error is retried, never fatal.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
