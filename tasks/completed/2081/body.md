---
title: 'workflow-fix: lint launcher GPU-width derivation without a SLURM branch (#1902
  class)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bdcfc7119764
created_at: '2026-08-05T05:42:53Z'
has_clean_result: false
origin_prompt: 'Surfaced by impl1491-slurmwidth during #1491''s fellows-lane width
  fix: the nvidia-smi detected-count width pattern survives in ~10 per-issue launchers
  and nothing prevents a new one from reintroducing the #1902 shared-node trespass;
  the adjacent --check-dispatcher-cvd-pin lint passes on the trespassing code because
  it checks a different property.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a follow-up surfaced on task
#1491 (emitting agent: experiment-implementer `impl1491-slurmwidth`, during the
fellows-lane width fix `1c8b46d28a`). Target and evidence CORRECTED by the
orchestrator at filing time — see § Workflow gap.

## Goal

Extend `scripts/workflow_lint.py` to flag launcher GPU-width derivation from
`nvidia-smi` device enumeration with no SLURM allocation-env branch — the #1902
shared-node trespass class — so the pattern cannot re-enter a NEW launcher.

## Workflow gap

- **Bug observed:** on a GPU-SHARED SLURM node (the fellows/charmander H200 lane)
  `nvidia-smi` ALWAYS enumerates all 8 physical devices and IGNORES
  `CUDA_VISIBLE_DEVICES`, so a launcher that sizes its fan-out from that count
  over-shards onto OTHER TENANTS' GPUs — corrupting our run and stealing other
  fellows users' compute. #1491 hit this and fixed it in ONE launcher
  (`scripts/issue1491_ladder_launch.sh`, commit `1c8b46d28a`: gate on
  `SLURM_JOB_ID`, derive via `issue1902_common.realized_gpu_ids`, fail loud).
  Nothing prevents the next launcher from reintroducing it.
- **Why it is a workflow gap:** the rule is documented in `.claude/rules/gotchas.md`
  § "Fellows SLURM nodes are GPU-SHARED" and a reference implementation exists
  (`issue1902_common.realized_gpu_ids`), but enforcement is prose-only. The
  ADJACENT lint `--check-dispatcher-cvd-pin` already exists and PASSES on these
  files — it checks that a backgrounded `--gpu-id` launch carries a
  `CUDA_VISIBLE_DEVICES=` prefix, which is a DIFFERENT property and is satisfied by
  the very code that trespasses. So the existing gate gives false assurance here.
- **Confidence (emitter):** medium — the CLASS is proven (a live #1902 incident plus
  #1491's fix) and the site count is verified below; the exact lint predicate needs
  calibration (see Constraints).
- verified-at-filing (orchestrator, 2026-08-05 UTC):
  - absence-of-guard, per-target: `grep -c 'slurm_gpu_width\|check-slurm-gpu-width\|SLURM_GPUS_ON_NODE' scripts/workflow_lint.py`
    → **0 hits in 1 file**. (0-hit in-target IS the evidence for an absence claim.)
  - live sites, `grep -nE 'nvidia-smi[^|]*\|[[:space:]]*wc -l|nvidia-smi --list-gpus' scripts/*.sh`
    → **10 hits across 9 files**: `issue1769_dispatch.sh:94`,
    `issue1426_sampled_dispatch.sh:96`, `issue1776_swap_dispatch.sh:117`,
    `issue1335_run.sh:64`, `issue1689_dispatch.sh:288,544`,
    `issue1738_multiturn_launch.sh:134`, `issue1774_dispatch.sh:43`,
    `issue1775_fu_run.sh:66`, `issue1775_run.sh:84` — plus
    `issue779_ffc_n1m_launch.sh:79` (same form, matched by the narrower
    `nvidia-smi -L.*wc -l` scan). The pattern is the house default, not an outlier.
  - **CORRECTION to the surfacing report:** it cited
    `scripts/run_issue650_pipeline.sh:244` as a third instance. It is NOT one —
    that site uses a hardcoded `for g in 0 1 2 3` with an explicit per-shard
    `CUDA_VISIBLE_DEVICES=$g` export and a matching `--gpu-id`, i.e. the CORRECT
    #545 shape. Do not "fix" it. (Read at filing time; presence claim refuted per
    `.claude/rules/workflow-fix-on-bug.md` verified-at-filing clause (a).)
  - landed-fix check: `git log --oneline --since='7 days ago' -- scripts/workflow_lint.py`
    → no SLURM-width lint landed.

## Proposed change (candidate diff sketch — refine in planning)

```
+ # `--check-slurm-gpu-width`: a shell launcher that derives GPU WIDTH from
+ # nvidia-smi device enumeration must ALSO carry a SLURM allocation-env branch
+ # (SLURM_JOB_ID / SLURM_JOB_GPUS / SLURM_STEP_GPUS / SLURM_GPUS_ON_NODE), or an
+ # explicit waiver. On a shared SLURM node the physical count is not the grant.
+ def check_slurm_gpu_width(*, scripts_dir: Path | None = None) -> list[str]:
+     # flag: `nvidia-smi ... | wc -l` (or --list-gpus / --query-gpu=index) whose
+     # value reaches a width/fan-out variable, in a file with NO SLURM_JOB_ID guard
+     # waiver: `# SLURM_GPU_WIDTH_EXEMPT: <reason>`
```

Reference implementation to point remediation at:
`scripts/issue1902_common.py::realized_gpu_ids`; worked adoption:
`scripts/issue1491_ladder_launch.sh` @ `1c8b46d28a` (shells out rather than
reimplementing in bash) + `tests/test_issue1491_ladder_launch_gpu_derivation.py`
(12 cases).

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py` (+ a pin test under `tests/`).
- The ~10 existing sites are pod/GCE-lane drivers, hazardous ONLY if re-run on a
  SLURM lane. Whether to retrofit them, waiver-annotate them, or grandfather them
  is a planner call — grandfathering is the low-risk default (they are frozen
  per-issue drivers; #1491's own fix is the adoption exemplar for new ones).

## Constraints / invariants

- **Calibrate before bundling into the no-flags default run.** With ~10 live sites,
  a naive predicate turns the default gate red fleet-wide. Measure the hit count,
  decide grandfather-vs-retrofit, and land the waiver mechanism FIRST.
- Do NOT touch `run_issue650_pipeline.sh:244` — verified correct at filing.
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff clean.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: bdcfc7119764

<!-- workflow-fix-candidate v1 -->
target_file: scripts/workflow_lint.py
bug_observed: The detected-count + local-index CUDA_VISIBLE_DEVICES pattern that would fan out onto other tenants GPUs on a shared fellows node survives in three per-issue launchers and nothing prevents a new launcher from reintroducing it.
why_workflow_gap: The rule is documented in gotchas.md § "Fellows SLURM nodes are GPU-SHARED" with a reference impl (issue1902_common.realized_gpu_ids), but enforcement is prose-only; the adjacent --check-dispatcher-cvd-pin lint passes on the trespassing code because it checks a different property.
proposed_change: Extend workflow_lint to flag launcher GPU-width derivation from nvidia-smi device enumeration with no SLURM allocation-env branch, the #1902 shared-node trespass class.
diff_sketch: |
  + def check_slurm_gpu_width(*, scripts_dir: Path | None = None) -> list[str]:
  +     # flag nvidia-smi enumeration feeding a width/fan-out variable in a file
  +     # with no SLURM_JOB_ID branch; waiver `# SLURM_GPU_WIDTH_EXEMPT: <reason>`
confidence: medium
related_task: #1491
<!-- /workflow-fix-candidate -->
