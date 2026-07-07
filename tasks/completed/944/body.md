---
title: 'workflow-fix: gotcha — GCE workloads have no ./.env; conditional sourcing'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a8f743e7f1d9
created_at: '2026-07-03T18:08:41Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #923 crash-fix round 2 (att-20260703-163121):
  unconditional ./.env sourcing on GCE killed the join poll'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson gotcha_candidate raised on task #923 (emitting agent: orchestrator, crash-fix round 2).

## Goal

Document in gotchas.md that pod/GCE-side shell scripts must source ./.env conditionally because GCE startup scripts export tokens with no .env file.

## Workflow gap

- **Bug observed:** `set -a && . ./.env && set +a && uv run python ...` inside an &&-chained subshell on a GCE workload (issue923_cpu_phase.sh join poll) — GCE has no ./.env (the startup script exports tokens), the sourcing failed, the real command never ran, and a fail-fast rc classifier mislabeled the shell error as a non-transient app failure, killing a healthy run whose sentinel already existed (att-20260703-163121).
- **Why it is a workflow gap:** the canonical heredoc-dotenv recipe (`.claude/rules/research-project-structure.md` § Environment Bootstrap: "set -a && source .env && set +a && uv run python - <<'PY'") is written for the VM and transfers silently-wrong to pod/GCE lanes; no rule warns about the venue difference, so implementers keep copying the unconditional form into workload scripts.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
# .claude/rules/gotchas.md — add entry:
+ ## GCE/pod workload scripts: conditional ./.env sourcing
+ GCE startup scripts export API tokens into the environment with NO .env file; RunPod
+ bootstrap pushes .env. A script that must run on both venues sources conditionally:
+   if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
+ NEVER put the sourcing inside the && chain of a command whose rc is being classified —
+ the sourcing failure short-circuits the command and mislabels the error (incident #923
+ att-20260703-163121). The heredoc-dotenv recipe in research-project-structure.md is
+ VM-scoped; add a cross-reference there.
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Secondary: one cross-reference sentence in `.claude/rules/research-project-structure.md` § Environment Bootstrap (its recipe is VM-scoped).
- Grep before editing: `grep -rn '\. \./\.env\|source .*\.env' scripts/*.sh .claude/rules/` and note any other unconditional sites worth flagging in the entry.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` passes (mind --check-heredoc-dotenv interactions — the conditional form must stay lint-clean).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: a8f743e7f1d9

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: An unconditional set -a && . ./.env && set +a inside an &&-chained subshell on a GCE workload short-circuited the join poll and mislabeled the shell error as a non-transient app failure
why_workflow_gap: The canonical heredoc-dotenv recipe is VM-scoped and transfers silently-wrong to GCE/pod lanes where no .env exists; no gotcha documents the venue difference
proposed_change: Document in gotchas.md that pod/GCE-side shell scripts must source ./.env conditionally because GCE startup scripts export tokens with no .env file
diff_sketch: |
  + gotchas.md entry: conditional sourcing form, never inside a classified && chain
  + cross-reference in research-project-structure.md § Environment Bootstrap
confidence: high
related_task: #923
<!-- /workflow-fix-candidate -->
