---
title: Step 6d.1 experimenter briefs must carry plan §9 parallelization shape — verbatim
  workload_cmd reuse silently runs shard 0 only on CVD-pin plans
kind: infra
tags: []
created_at: '2026-08-23T11:05:48Z'
has_clean_result: false
origin_prompt: 'experimenter prose observation, task #2254 first-k round launch 2026-08-23:
  briefs that copy the handle''s workload_cmd verbatim for launcher-CVD-pin plans
  reproduce shard-0-only drift'
workflow: v1
---
## Goal

Close the experimenter-brief composition gap that reproduces shard-0-only launches on launcher-CVD-pin plans: when the `/issue` orchestrator (Step 6d.1) composes an experimenter brief by copying the dispatch handle's persisted `workload_cmd` verbatim, and the plan's §9 mandates launcher-env `CUDA_VISIBLE_DEVICES` pins with per-shard `--shard-id` invocations, the verbatim command is the plan fence's dry-parse form (`--num-shards K`, no `--shard-id`) and silently runs only shard 0 on 1 GPU of a K-GPU pod.

## Incident

Task #2254, follow-up round `first-k-answer-token-steering`, 2026-08-23 (~10:50Z launch). The orchestrator's experimenter brief carried the handle's verbatim `workload_cmd` (`uv run python scripts/issue2254_first_k_steering.py --phases stage_inputs,steer --num-shards 4`). Executed as-is on pod-2254 (4× H100) this would have run 40/160 cells on one GPU while three H100s idled. The experimenter caught the drift against plan v10 §9 (lines 231/278 — launcher-env CVD pins, `--shard-id {0..3} --num-shards 4` in parallel after a serial `stage_inputs`), composed `/workspace/launch_issue_2254_firstk.sh`, and stated the correction in its `epm:progress` marker (#2254 events v86). No compute was lost — but only because the subagent independently re-derived the plan's parallelization shape.

## Fix shape (for the planner/implementer to refine)

The workflow surface at fault is the orchestrator-side brief-composition guidance (`.claude/skills/issue/steps/` Step 6d.1 experimenter-brief text, plus any brief-template prose that says "launch the persisted workload_cmd"). The brief composer MUST carry plan §9's realized parallelization line (shard axis, width, CVD-pin form, serial-vs-parallel phase split) alongside the `cmd=`, and instruct the experimenter that the handle's `workload_cmd` is the FENCE DRY-PARSE form — the launcher composition is governed by §9, not by the persisted string. Consider: (a) a one-line duty in Step 6d.1 ("brief carries §9's parallelization block verbatim; verbatim workload_cmd reuse on a multi-shard plan is the #2254 shard-0 drift"); (b) optionally, `dispatch_issue.py` persisting a structured `parallelization` field in the handle sidecar so briefs can quote it mechanically.

## Provenance

Surfaced as a prose workflow-surface observation by the experimenter subagent on task #2254 (round first-k-answer-token-steering, launch report 2026-08-23). Auto-filed by the #2254 followup orchestrator per `.claude/rules/workflow-fix-on-bug.md` (surfaced-prose follow-ups clause).
