---
title: 'workflow-fix: warn when SLURM launch fence exceeds the intent''s sbatch time
  bin'
kind: infra
tags:
- wf-fix
- wf-fix-fp:25cd1ad2f0dd
created_at: '2026-08-03T03:02:05Z'
has_clean_result: false
origin_prompt: 'orchestrator-observed on #1336: fellows job 18027 TIMEOUT at 4h (capture-7b
  default bin) under a declared 36h GCP-only fence and 12.2h plan wall; add a dispatch-time
  cross-check warning'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator-observed gap on task #1336 (fellows job 18027 CANCELLED BY SLURM TIME LIMIT at 04:00:24 mid-extract, one full launch + ~4 GPU-h of the queue's wave lost to a re-queue).

## Goal

Make `dispatch_issue.py launch` warn loudly (or fail fast) when a SLURM-lane-reachable dispatch carries a declared `--max-run-duration` (or a plan-projected wall) that EXCEEDS the intent's resolved sbatch `--time` bin and no explicit `--time-budget-hours` was passed.

## Workflow gap

- **Bug observed:** #1336's round dispatched `--intent capture-7b --max-run-duration 36h` on the fellows lane with a plan-projected 12.2 h wall; `--max-run-duration` is GCP-only (inert on SLURM, by design per #628), and the capture-7b intent's default time bin is 4.0 h (`_DEFAULT_TIME_BUDGETS_HOURS`, eval-class per #1896) — SLURM cancelled job 18027 at 04:00:24 DUE TO TIME LIMIT mid-extract_v2, with zero dispatch-time warning. The declared 36 h fence gave false comfort that the wall was covered.
- **Why it is a workflow gap:** the dispatch CLI KNOWS both values at launch time — `args.max_run_duration` (scripts/dispatch_issue.py L1515-1522) and the resolved SLURM budget (`backends/slurm.py time_budget_hours()` L581-599 + `_DEFAULT_TIME_BUDGETS_HOURS` L553+) — but never cross-checks them for SLURM-reachable lanes; the mismatch surfaces only as a mid-run SLURM TIMEOUT hours later.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "max_run_duration\|time_budget" scripts/dispatch_issue.py` → max_run_duration handled at L1515-1522/L2546/L2802-2810 (GCP extra only), time_budget_hours threaded at L1753, NO cross-check between them (0 hits for any comparison); `sed -n '539,560p' src/explore_persona_space/backends/slurm.py` → `_DEFAULT_TIME_BUDGETS_HOURS` with capture-7b in the eval-class 4.0 h bin (#1896 note); incident evidence sacct job 18027 `Timelimit=04:00:00 State=TIMEOUT/CANCELLED` (2026-08-03)

## Proposed change (candidate diff sketch — refine in planning)

```
# scripts/dispatch_issue.py, launch path, after spec construction (~L1753):
+ if args.max_run_duration and _slurm_reachable(spec, args):
+     resolved_h = slurm.time_budget_hours(spec)   # explicit flag or intent default
+     fence_h = _parse_duration_hours(args.max_run_duration)
+     if fence_h > resolved_h and args.time_budget_hours is None:
+         warn loudly (stderr + extra["slurm_time_bin_mismatch"]=... on the
+         epm:backend-selected marker): "declared fence {fence_h}h exceeds the
+         SLURM --time bin {resolved_h}h for intent {intent}; pass
+         --time-budget-hours or the job will TIMEOUT mid-run (#1336 job 18027)"
```

## Scope / surfaces

- Primary target: `scripts/dispatch_issue.py`
- Secondary (read-only reference): `src/explore_persona_space/backends/slurm.py` `time_budget_hours()` / `_DEFAULT_TIME_BUDGETS_HOURS`
- Grep the workflow surface for the pattern before editing (`grep -rln 'max_run_duration' .claude/ CLAUDE.md scripts/ src/explore_persona_space/backends/`) and update every doc site that presents `--max-run-duration` as THE fence (e.g. the /issue SKILL.md lane-capability bullet (d) already notes GCP-only — keep consistent).

## Constraints / invariants

- Warning-first (never refuse a deliberate short-fence launch); marker-flag additive like the existing override-guard flags (`extra.boot_disk_default_with_ft_intent` precedent).
- `scripts/workflow_lint.py --check-asks` passes; ruff clean; add a pin test beside the existing dispatch-CLI guard tests.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/dispatch_issue.py
- fingerprint: 25cd1ad2f0dd

<!-- workflow-fix-candidate v1 -->
target_file: scripts/dispatch_issue.py
bug_observed: A SLURM-lane dispatch carrying --max-run-duration 36h (GCP-only, inert on SLURM) launched with the capture-7b intent's default 4.0h sbatch time bin and a plan-projected 12.2h wall; SLURM cancelled job 18027 at 04:00:24 DUE TO TIME LIMIT mid-run with zero dispatch-time warning (#1336, 2026-08-03).
why_workflow_gap: dispatch_issue.py knows both the declared fence (L1515-1522) and the resolved SLURM time budget (slurm.time_budget_hours + _DEFAULT_TIME_BUDGETS_HOURS) at launch time but never cross-checks them for SLURM-reachable dispatches; the mismatch surfaces only as a mid-run TIMEOUT hours later.
proposed_change: Warn loudly (stderr + an additive extra flag on the epm:backend-selected marker) when a SLURM-reachable launch declares a max-run-duration exceeding the resolved sbatch --time bin and no explicit --time-budget-hours was passed.
diff_sketch: |
  + if args.max_run_duration and slurm-reachable and args.time_budget_hours is None:
  +     resolved_h = slurm.time_budget_hours(spec); fence_h = parse(args.max_run_duration)
  +     if fence_h > resolved_h: warn + extra["slurm_time_bin_mismatch"] on the marker
confidence: high
related_task: #1336
<!-- /workflow-fix-candidate -->
