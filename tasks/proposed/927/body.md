---
title: 'workflow-fix: mint fresh EPS_ATTEMPT_ID per GCP launch'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2a658c55a277
created_at: '2026-07-03T11:37:46Z'
has_clean_result: false
origin_prompt: 'orchestrator observation on #825: EPS_ATTEMPT_ID reused across 3 relaunches,
  crash-persist collided with the July-2 attempt dir and hid fresh crash evidence'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator observation on task #825 (emitting agent: orchestrator, /issue 825 session).

## Goal

Mint a fresh EPS_ATTEMPT_ID per dispatch_issue launch instead of reusing the prior attempt id across relaunches.

## Workflow gap

- **Bug observed:** Three 2026-07-03 relaunches of issue 825 all carried EPS_ATTEMPT_ID=att-20260702-061417 (minted 2026-07-02), so every crash-persist uploaded into the same issue825_partial dir as the July-2 attempt, hiding fresh crash evidence and costing two misdiagnosed respawns.
- **Why it is a workflow gap:** The attempt id keys the crash-persist dest dir (issue<N>_partial/<attempt_id>/), the completion-sentinel path, and forensic attribution; reusing it across genuinely different attempts makes crash evidence from a NEW attempt land in (and canonical-name-overwrite) an OLD attempt's dir. The #854 timestamped-copy mitigation preserves the logs but not the per-attempt attribution — a fresh orchestrator looking at attempt dirs concludes "no persist from today" (exactly what happened, 2x).
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
# backends/gcp.py (or wherever EPS_ATTEMPT_ID is minted/threaded):
- attempt_id reused from prior launch state / handle sidecar on relaunch
+ attempt_id = f"att-{utcnow():%Y%m%d-%H%M%S}" minted FRESH inside every launch() call
+ (the completion-sentinel path + expected_artifacts must use the same fresh id)
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Investigate where the reuse enters: `grep -rn "EPS_ATTEMPT_ID\|attempt_id\|att-" src/explore_persona_space/backends/ scripts/dispatch_issue.py` — the reused value may come from the handle sidecar, a template cache, or a stable derivation; update every producer/consumer pair consistently (sentinel path, artifacts declaration, crash-persist dest).

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff passes; existing `tests/test_gcp_backend.py` / `tests/test_backend_*.py` updated if they pin attempt-id behavior.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: 2a658c55a277

Surfaced prose (orchestrator observation, task #825, 2026-07-03): startup-script metadata of the third relaunch instance showed `export EPS_ATTEMPT_ID=att-20260702-061417` — a July-2 mint carried through three July-3 relaunches; crash-persist evidence from all three landed in the July-2 partial dir.
