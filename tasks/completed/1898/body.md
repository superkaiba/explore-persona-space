---
title: 'workflow-fix: fellows sentinel-drain arm — un-pin sentinel workloads from
  paid lanes'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a0216197634d
created_at: '2026-07-30T21:01:23Z'
has_clean_result: false
origin_prompt: 'user directive 2026-07-30: make sure the free H200s and the anthropic
  fellows cluster get used as much as possible; orchestrator verified /workspace on
  charmander is shared MooseFS readable from the SSH endpoint, so the sentinel pin
  is closable'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator observation during a PM-chat fellows-utilization audit (2026-07-30). No parent task (chat-mode; candidate logged to `.claude/cache/workflow-fix-events.jsonl`). Companion to #1896 (capture-7b intent gap) under the standing user directive (2026-07-22, reinforced verbatim 2026-07-30: "make sure the free H200s and the anthropic fellows cluster get used as much as possible").

## Goal

Let sentinel-dependent multi-phase workloads run on the free fellows (charmander) lane: add a fellows sentinel-drain arm to the VM-side poller (read the job's `/workspace` scratch-dir sentinels over `ssh charmander`, mirroring the RunPod `/workspace/logs/issue-<N>-*.json` drain), then lift the plan-time gcp/runpod pin for the fellows lane (update the CLAUDE.md SENTINEL HAZARD paragraph + scope the #1777 verify_plan WARN to lanes that genuinely lack a drain).

## Workflow gap

- **Bug observed:** sentinel-dependent workloads are categorically pinned to paid lanes (`backend: gcp`/`runpod`) at plan time because "on charmander /workspace EXISTS, so a sentinel-writing dispatcher... would write sentinels NOBODY drains (silent marker loss)" (CLAUDE.md § Compute backends, SENTINEL HAZARD; #1777 verify_plan WARN). This diverts an entire workload class off the free H200 cluster.
- **Why it is a workflow gap:** the premise "nobody CAN drain them" is false — verified live 2026-07-30: `/workspace` on charmander is shared MooseFS (`df -hT /workspace` → `mfs#eur-is-5.runpod.net:9421 fuse 1.3P`), job scratch dirs written by compute nodes (e.g. `issue-1345` running on node-4, with a live `.current_phase` file) are readable from the SSH endpoint (node-2). The poller already SSH-polls charmander for SLURM state (`epm:cluster-launched`/`epm:cluster-terminal` via `backend_poll.py`); a sentinel-drain arm over the same channel is the missing piece. The pin is a stale design decision, not a physical constraint.
- **Confidence (emitter):** high
- verified-at-filing: `grep -c "fellows" scripts/poll_pipeline.py` → 0 hits (absence claim — no fellows arm exists in the sentinel drainer); `grep -n "fellows" scripts/backend_poll.py` → 1 hit (line 419, lane-name set only — SLURM state polling, no sentinel drain); live probe `ssh charmander 'df -hT /workspace; ls /workspace/superkaiba/eps/issue-1345/'` → shared MooseFS mount + cross-node-readable job dir with `.current_phase` breadcrumb (2026-07-30).

## Proposed change (candidate diff sketch — refine in planning)

```
# scripts/backend_poll.py (or poll_pipeline.py — planner picks the seam):
+ # fellows sentinel-drain arm: when the epm:cluster-launched handle carries
+ # cluster=fellows + scratch_dir, poll `ssh charmander cat <scratch_dir>/
+ # issue-<N>-*.json` (same parse + marker-post contract as the RunPod
+ # /workspace/logs drain; drain-rename tolerance per pod-side-reporting.md)
# CLAUDE.md § Compute backends: rewrite the SENTINEL HAZARD sentence —
# fellows becomes a drained lane; the hazard note shrinks to DRAC/Mila.
# scripts/verify_plan.py (#1777 WARN): scope the /workspace-sentinel WARN
# to unpinned auto lanes WITHOUT a drain (exclude fellows once the arm lands).
```

## Scope / surfaces

- Primary target: `scripts/backend_poll.py` (SLURM-lane poll loop — owns the launched-handle context), `scripts/poll_pipeline.py` (the sentinel parse/post contract to reuse)
- Secondary: `CLAUDE.md` (SENTINEL HAZARD paragraph), `scripts/verify_plan.py` (the #1777 WARN predicate), `.claude/rules/pod-side-reporting.md` (drain contract doc)
- Grep before editing: `grep -rn "SENTINEL HAZARD\|/workspace/logs" CLAUDE.md scripts/ .claude/rules/` and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The RunPod sentinel drain contract stays byte-identical; the fellows arm is additive.
- Drain-rename tolerance (#1311) and the pod-side-reporting contract apply to the new arm.
- MooseFS FUSE caveats (the EPS gotchas: read-wedge, EDQUOT) — the drain must tolerate a wedged read (bounded timeout, never blocks the poll loop).
- DRAC/Mila stay undrained (no shared-/workspace contract there); the plan-time pin remains for them.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/backend_poll.py
- fingerprint: a0216197634d

Orchestrator observation (verbatim probe evidence, 2026-07-30): `ssh charmander 'df -hT /workspace'` → `mfs#eur-is-5.runpod.net:9421 fuse 1.3P 338T 920T 27% /workspace`; `ls /workspace/superkaiba/eps/issue-1345/` from node-2 shows the node-4 job's live dir incl. `.current_phase` (mtime 19:02, mid-run). User directive (2026-07-30, verbatim): "make sure the free H200sand the anthropic fellows cluster get used as much as possible".
