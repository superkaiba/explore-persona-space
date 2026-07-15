---
title: 'workflow-fix: crash-persist worker-log sweep excludes committed repo content'
kind: infra
tags:
- wf-fix
- wf-fix-fp:964268d7e0af
created_at: '2026-07-15T15:38:12Z'
has_clean_result: false
origin_prompt: 'prose follow-up from #1345 crash att-20260715-151246: crash-persist
  uploaded ~30 committed logs/daily+weekly retrospectives as worker_logs'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #1345 (emitting agent: orchestrator, observed in crash-persist output att-20260715-151246).

## Goal

Scope the GCE crash-persist worker-log fan-out sweep (#885 leg in backends/gcp.py) to run-generated logs, excluding committed repo content.

## Workflow gap

- **Bug observed:** The #1345 GCE crash-persist uploaded the repo's COMMITTED logs/daily/*.md + logs/weekly/*.md retrospectives (~30+ markdown files) as `worker_logs/` to superkaiba1/explore-persona-space-data/issue1345_partial/att-20260715-151246/ — the workload clones the repo at $WORKLOAD_ROOT, so the sweep's `logs/` tree contains committed project retrospectives, not just run-generated worker logs.
- **Why it is a workflow gap:** the sweep pattern (gcp.py #885 worker-logs leg) matches everything under the logs/ tree; committed repo files are already durable in git, so re-uploading them clutters crash forensics, inflates the crash-persist upload (delaying the billing-bounding poweroff), and mixes non-run files into the partial prefix.
- **Confidence (emitter):** high
- verified-at-filing: `git ls-files logs/daily/2026-06-02.md` → tracked (committed repo file); HF listing shows it under issue1345_partial/.../worker_logs/daily/; `grep -n "worker_logs" src/explore_persona_space/backends/gcp.py` → sweep leg at ~:1146,1737-1811 with cache excludes but NO committed-content exclude (2026-07-15). Per-target: src/explore_persona_space/backends/gcp.py 8 hits (presence confirmed).

## Proposed change (candidate diff sketch — refine in planning)

+ In the #885 worker-logs sweep leg (gcp.py startup-script renderer): exclude
+ git-tracked files (cheap: `git -C "$WORKLOAD_ROOT" ls-files logs/` prune), or
+ restrict the sweep to *.log + files with mtime >= attempt start. Keep the
+ tail-cap + one-upload_folder-commit contract unchanged.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'worker_logs' .claude/ CLAUDE.md scripts/ src/explore_persona_space/backends/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only. The crash-persist stays 300s-bounded and never delays poweroff (#854).
- `scripts/workflow_lint.py` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: 964268d7e0af
