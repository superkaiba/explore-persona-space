---
title: 'daily-fix: fail loud when --min-ram-gb is inert on a GPU dis'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d3668290b74d
- daily-auto-filed
created_at: '2026-08-02T07:09:06Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): #1739 Gap-2 recipe claimed
  --min-ram-gb 300 as the OOM fix; GCP GPU machine selection ignores the flag by documented
  intent (dispatch_issue.py:1471-1478), box landed on 170GB a2-ultragpu-1g, both boxes
  OOM''d exit 137; relaunch with --gpus 2 (340GB) succeeded. Silent undersized provisioning
  with a stated RAM requirement.'
workflow: v1
---
# daily-fix: fail loud when --min-ram-gb is inert on a GPU dispatch

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entry C4 (miner 1; session 55419495, #1739, boxes oodhallmax / oodpredshallu).

## Goal
Make `--min-ram-gb` fail loud (or at minimum WARN + refuse silently-undersized provisioning) on GCP GPU dispatches whose resolved machine's RAM is below the requested value — today the flag is a RunPod-CPU-fallback-only knob and GCP GPU machine selection silently ignores it.

## Workflow gap
- **Bug observed:** the #1739 Gap-2 recipe claimed "`--min-ram-gb 300` routes Gap 2 to the 340GB a2-ultragpu-2g (the OOM fix)"; the box landed on a2-ultragpu-1g (170GB), the mismatch was rationalized ("170GB is ample here"), and BOTH boxes OOM'd exit 137 ("My earlier 'scoped run won't OOM' call was wrong"). Relaunch with `--gpus 2` → 340GB succeeded. The flag's inertness is documented but the dispatch surface gives no runtime signal.
- **Why it is a workflow gap:** a plan-stated RAM requirement threaded through the canonical dispatch CLI silently provisions an undersized GPU box — exactly the failure class the CPU-side gate (#1010 `cpu_fallback_infeasible_for_plan`) was built to refuse.
- **Confidence:** high (premise probed by the miner AND re-verified now).
- verified-at-filing: `grep -n 'min_ram' scripts/dispatch_issue.py src/explore_persona_space/backends/{gcp,router}.py` → dispatch_issue.py:1471-1478 comment reads verbatim "RunPod-CPU-fallback knob (#1010): read by the feasibility gate in router._runpod_terminal_rung ... GCP machine selection is unchanged (by intent); inert on SLURM lanes" (same text in the `--min-ram-gb` argparse help at :2713); gcp.py's only `min_ram_gb` uses are handle/marker-extra THREADING at :4533 and :5798 (reconnect + failover-handle persistence — no machine-selection read); router.py reads it only in `_refuse_infeasible_cpu_footprint` (:3149, CPU instances). `git log --oneline --since='7 days ago' -- scripts/dispatch_issue.py src/explore_persona_space/backends/gcp.py` → 9 commits, none touches min_ram_gb GPU semantics (2026-08-01).

## Proposed change (refine in planning)
In `scripts/dispatch_issue.py` (parse-time) or `backends/gcp.py` (rung-selection time): when `--min-ram-gb` is set on a GPU dispatch (resolved `gpu_count > 0`), compare against the resolved GCP machine's RAM (add a RAM column to `INTENT_TO_MACHINE` / the wide-rung map, or a static machine→RAM table) and:
- default: FAIL LOUD pre-launch naming the mismatch + the satisfying alternatives (`--gpus N` wide rung, a bigger intent), mirroring `--min-gpu-mem-gb`'s A100-40 rung-skip (#1468) — prefer rung-SKIP semantics (walk to a rung that satisfies the RAM) over hard refusal where the ladder has a satisfying rung;
- at minimum: a loud WARN at dispatch + the requested-vs-resolved RAM on the `epm:backend-selected` marker.
Planning decides skip-vs-refuse per rung; keep RunPod-CPU (#1010) and SLURM behavior byte-unchanged.

## Scope / surfaces
- Primary target: `scripts/dispatch_issue.py, src/explore_persona_space/backends/gcp.py`
- Likely also `src/explore_persona_space/backends/router.py` (rung walk) + `tests/test_gcp_backend.py` / `tests/test_router.py` pins; update the argparse help + the :1471-1478 comment ("by intent" becomes stale).

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff/bash -n on touched files passes.
- Preserve #934 omit-when-absent spec-hash discipline for `spec.extra`; no behavior change when the flag is unset.
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: d3668290b74d
- workflow_fix_target: scripts/dispatch_issue.py, src/explore_persona_space/backends/gcp.py
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entry C4.
