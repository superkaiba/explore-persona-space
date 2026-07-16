---
title: 'workflow-fix: chunk crash-persist eval_results uploads (504 on 29k-file commit)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:68a3c53bba7e
created_at: '2026-07-15T09:30:38Z'
has_clean_result: false
origin_prompt: 'orchestrator-observed persist gap on #1090 fu5 crash att-20260715-081917:
  eval_results dir persist 504''d on a single 29k-file commit and gave up'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1090 (emitting agent: orchestrator, fu5 crash triage).

## Goal

crash-persist: chunk large eval_results dir uploads into bounded multi-commit batches with retry instead of one upload_folder commit

## Workflow gap

- **Bug observed:** att-20260715-081917 crash-persist FAILED dir eval_results_issue_1090 (29,024 files, one commit) with a 504 Gateway Time-out and gave up, losing the partial eval capture
- **Why it is a workflow gap:** a long-lived issue accumulates tens of thousands of committed eval files; a single upload_folder commit of that size predictably 504s on the Hub commit endpoint (the known large-commit class), so the crash-persist's eval capture silently degrades to nothing exactly on the mature issues where it matters most. One bounded retry of the same oversized commit cannot succeed either.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "eval_results_issue\|upload_folder\|attempt 1/2" src/explore_persona_space/backends/gcp.py` → persist template in-file (dir-upload arm adjacent to the :1743 cap block); transcript evidence `[crash-persist] FAILED dir eval_results_issue_1090: 504 Server Error ... /commit/main` (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

+ In the _eps_persist_diagnostics dir-upload arm: when a dir exceeds ~N files
+ (e.g. 2000), split the upload into per-subdir (or fixed-size batch) commits,
+ each with the existing bounded retry, instead of one whole-dir commit;
+ log per-batch outcomes so a partial persist is visible.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only. Persist stays 300s-bounded + fail-soft (#854) — chunking
  must respect the total budget (prefer newest/smallest-first so the highest-value
  files land inside the budget).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: 68a3c53bba7e

Surfaced prose (orchestrator observation, #1090 fu5 crash triage 2026-07-15): crash_persist_transcript.log line `[crash-persist] FAILED dir eval_results_issue_1090: 504 Server Error: Gateway Time-out ... (29024 files, 14371131 bytes)`.
