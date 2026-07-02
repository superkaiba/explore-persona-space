---
title: 'workflow-fix: GCP crash-persist sweeps scratch logs/ dir (worker tracebacks)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:34cfec228d01
created_at: '2026-07-02T20:25:52Z'
has_clean_result: false
origin_prompt: 'orchestrator-observed during #779: both crash diagnoses required boot-disk
  mount because the worker log with the traceback is never persisted by _eps_persist_diagnostics'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator-observed gap during #779 crash recovery (bitten TWICE in one day: att-20260702-082017 OOM crash at 13:32Z and the upload-staging crash at 20:01Z).

## Goal

GCP crash-persist trap should sweep the scratch repo logs/ dir (worker logs), not only the main workload.log.

## Workflow gap

- **Bug observed:** two #779 crash diagnoses each required detaching the boot disk onto a temp VM because the worker traceback lived in logs/issue_779/corpus_gpu0_all.json-adjacent worker logs (corpus_gpu0_all.log) which _eps_persist_diagnostics never uploads.
- **Why it is a workflow gap:** `backends/gcp.py`'s `_eps_persist_diagnostics` (the EXIT-trap crash-diagnostics uploader, #658) uploads `workload.log` + `eval_results/issue_<N>/` — but any dispatcher that fans work out to per-GPU/per-shard workers redirects the actual traceback to `$WORKLOAD_ROOT/logs/**` (the common multi-worker pattern, e.g. `logs/issue_<N>/corpus_gpu0_all.log`). The uploaded main log then ends at the fan-out line and the crash cause is unrecoverable without a boot-disk mount — a ~30-minute manual detach/temp-VM/mount recovery, done twice today.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
  # in _eps_persist_diagnostics (backends/gcp.py startup-script renderer):
    ... existing workload.log + eval_results upload ...
+   # also sweep the scratch repo's logs/ tree (per-worker logs carry the real
+   # traceback on fan-out dispatchers; cap total bytes, newest-first, to keep
+   # the 300s bound safe — e.g. tail -c 5M per file, skip files > cap)
+   if [ -d "$WORKLOAD_ROOT/logs" ]; then
+     ... upload $WORKLOAD_ROOT/logs/**/*.log (size-capped) under
+         issue<N>_partial/<attempt_id>/worker_logs/ ...
+   fi
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py` (workflow surface per the router-package carve-out)
- Grep first: `grep -rn '_eps_persist_diagnostics\|crash_report.json' src/explore_persona_space/backends/ .claude/ CLAUDE.md` and update the CLAUDE.md "Crash diagnostics survive the DELETE (#658)" bullet + any rule text describing what the trap persists.
- Keep the 300s bound + full guarding (the trap must never delay poweroff) — size-cap the sweep.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` passes; tests for the renderer (tests/test_gcp_backend.py) updated/extended.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: 34cfec228d01
