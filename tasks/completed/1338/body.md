---
title: 'workflow-fix: crash-persist harvests per-run logs before oversized-dir skip'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e6c34118b7af
created_at: '2026-07-15T09:30:13Z'
has_clean_result: false
origin_prompt: 'orchestrator-observed persist gap on #1090 fu5 crash att-20260715-081917:
  oversized data-dir skip lost per-run tracebacks; worker_logs sweep grabbed the repo
  clone''s logs/daily notes'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1090 (emitting agent: orchestrator, fu5 crash triage).

## Goal

crash-persist: always harvest per-run *.log files from data dirs (tail-capped) before applying the EPS_PERSIST_DIR_CAP_BYTES oversized-dir skip, and point the worker-log sweep at the driver out_root logs dir rather than the repo clone's logs/

## Workflow gap

- **Bug observed:** att-20260715-081917 crash-persist SKIPped the 28 GB data_issue_1090 dir wholesale, losing the tiny per-run tracebacks (data/issue_1090/fu5/logs/*.log), while worker_logs captured the repo clone's committed logs/daily/*.md notes instead
- **Why it is a workflow gap:** the crash-persist exists precisely to save the crash's diagnostics; the oversized-dir skip is all-or-nothing, so the ~KB tracebacks die with the ~28 GB of staged mixes/checkpoints around them, and the #885 worker-log sweep points at a directory the fu4/fu5 driver family does not write to. Recovery from the #1090 fu5 crash required a full local repro because zero worker tracebacks survived.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "EPS_PERSIST_DIR_CAP_BYTES\|worker_logs\|EPS_PERSIST_LOG_MAX_FILES" src/explore_persona_space/backends/gcp.py` → 12+ hits in the one named target (cap def :1743; worker-log sweep :1780-1829 — the persist logic is an embedded startup-script template inside gcp.py, so all hits are in-file) (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

+ In _eps_persist_diagnostics: BEFORE the per-dir byte-cap check, glob each data dir
+ for **/logs/**/*.log and **/*.attempt*.log, stage them tail-capped (reuse the
+ EPS_PERSIST_LOG_MAX_FILES / tail-cap machinery from the #885 worker_logs sweep),
+ then apply the oversized-dir skip to the REMAINDER as today.
+ Additionally include the workload out_root log dirs (data*/<issue>/**/logs) in the
+ worker-log sweep roots, and exclude repo-committed logs/daily|weekly notes.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'EPS_PERSIST_DIR_CAP_BYTES' .claude/ CLAUDE.md scripts/ src/explore_persona_space/backends/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Persist must stay 300s-bounded and fail-soft (never delay the poweroff — #854).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: e6c34118b7af

Surfaced prose (orchestrator observation, task #1090 fu5 crash triage 2026-07-15): crash_persist_transcript.log shows `SKIP data_issue_1090: 28007358769 bytes > cap 2147483648` and a worker_logs capture of 40 unrelated repo-clone daily/weekly markdown notes; the actual per-run attempt logs under data/issue_1090/fu5/logs/ were never uploaded.
