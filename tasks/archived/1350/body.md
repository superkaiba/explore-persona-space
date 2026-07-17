---
title: 'workflow-fix: lane-portable REPO_ROOT guidance for --workload-cmd (WORKLOAD_ROOT
  is GCE-only)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:babb45d4e232
created_at: '2026-07-15T15:03:22Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #1336: SKILL.md Step 6b note (f)
  prefix REPO_ROOT="$WORKLOAD_ROOT" dies unbound on the RunPod failover lane; see
  the candidate block in the body'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1336 (emitting agent: orchestrator + experimenter failure-lesson).

## Goal

Make the documented `--workload-cmd` REPO_ROOT guidance lane-portable: the recommended `REPO_ROOT="$WORKLOAD_ROOT"` prefix is GCE-only and dies (`unbound variable` under `set -u`) when the router's GCP→RunPod failover re-runs the same workload command on the RunPod lane.

## Workflow gap

- **Bug observed:** #1336's launch followed SKILL.md Step 6b note (f) and composed `--workload-cmd 'REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue1336_dispatch.sh all'`. The GCP FLEX_START queue timed out, the #783 failover provisioned RunPod pod-1336, and the launcher died at t+0 with `WORKLOAD_ROOT: unbound variable` — the pod sat RUNNING/billing until a manual re-drive (2026-07-15, ~1h lost).
- **Why it is a workflow gap:** the failover path re-runs the SAME workload command on a lane that never exports WORKLOAD_ROOT; the guidance recommends a prefix form that is structurally incompatible with the failover the same skill documents. The lane-portable forms are `REPO_ROOT="${WORKLOAD_ROOT:-/workspace/explore-persona-space}"` (defaulted expansion) or launching self-defaulting dispatch scripts bare.
- **Confidence (emitter):** high
- verified-at-filing: `grep -rln 'REPO_ROOT="\$WORKLOAD_ROOT"' .claude/skills .claude/rules .claude/agents CLAUDE.md src/explore_persona_space/backends | grep -v worktrees` → 4 files (SKILL.md ×2 hits, gotchas.md ×1, compute-backend-failover.md ×1, backends/gcp.py ×1 — the gcp.py hit is the GCE startup script that legitimately EXPORTS the var; issue_dispatch.py hit is a docstring mention) (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

```
- compose `--workload-cmd 'REPO_ROOT="$WORKLOAD_ROOT" bash scripts/<driver>.sh'`
+ compose `--workload-cmd 'bash scripts/<driver>.sh'` when the driver self-defaults
+ REPO_ROOT (`REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"`, the
+ standard convention), or the lane-portable prefix
+ `REPO_ROOT="${WORKLOAD_ROOT:-/workspace/explore-persona-space}"` otherwise —
+ never the bare `"$WORKLOAD_ROOT"` expansion, which is GCE-only and dies
+ `unbound variable` when the GCP→RunPod failover re-runs the command (#1336).
```

Optionally: the RunPod launcher composer (issue_dispatch/pod-side wrapper) could pre-export `WORKLOAD_ROOT=/workspace/explore-persona-space` as defense-in-depth — planner decides.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 6b note (f), 2 hits)
- Also: `.claude/rules/gotchas.md` (1 hit), `.claude/rules/compute-backend-failover.md` (1 hit); `src/explore_persona_space/backends/issue_dispatch.py` docstring mention. `backends/gcp.py`'s hit is the GCE startup script that EXPORTS the var — do not change its semantics.
- Grep the workflow surface for the pattern before editing (`grep -rln 'REPO_ROOT="\$WORKLOAD_ROOT"' .claude/ CLAUDE.md scripts/ src/explore_persona_space/backends/`) and update every doc hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; SKILL.md stays consistent with the rule files.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: babb45d4e232

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md, .claude/rules/gotchas.md, .claude/rules/compute-backend-failover.md
bug_observed: the Step 6b note (f) recommended REPO_ROOT="$WORKLOAD_ROOT" workload-cmd prefix dies with 'unbound variable' when the GCP→RunPod queue-timeout failover re-runs the command on the RunPod lane (WORKLOAD_ROOT is GCE-only) — #1336 lost ~1h with pod-1336 billing on a dead launch
why_workflow_gap: the skill's own failover path is structurally incompatible with the prefix form the same skill recommends
proposed_change: recommend bare launch for self-defaulting drivers or the defaulted expansion ${WORKLOAD_ROOT:-/workspace/explore-persona-space}; update all three doc hits
diff_sketch: |
  - --workload-cmd 'REPO_ROOT="$WORKLOAD_ROOT" bash scripts/<driver>.sh'
  + --workload-cmd 'bash scripts/<driver>.sh'   # driver self-defaults REPO_ROOT
  + (else: REPO_ROOT="${WORKLOAD_ROOT:-/workspace/explore-persona-space}")
confidence: high
related_task: #1336
<!-- /workflow-fix-candidate -->
