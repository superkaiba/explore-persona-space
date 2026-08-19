---
title: 'workflow-fix: thread lane_suffix into slurm.job_name (concurrent lanes silently
  reconnect)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b3afa7619eaf
created_at: '2026-08-03T23:38:19Z'
has_clean_result: false
origin_prompt: 'Orchestrator-observed defect on #1947: two concurrent inline lanes
  on one issue both resolved to nibi job 19036499 (eps-issue-1947); the second reconnected
  with workload_dispatched=false and never ran. lane_suffix reaches the handle sidecar
  and the GCP instance name but not the SLURM job name.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a defect hit live on task #1947 (emitting agent: orchestrator, observed via the `localization-1947` lane's report and confirmed on disk).

## Goal

Thread `lane_suffix` into `slurm.job_name` so two concurrent lanes on one issue submit distinct SLURM jobs instead of silently reconnecting to each other.

## Workflow gap

- **Bug observed:** On SLURM, `lane_suffix` reaches the handle sidecar path and the GCP instance name but not the job name, so a second concurrent lane on the same issue reconnects to the first lane's job with `workload_dispatched=false` and never runs.
- **Why it is a workflow gap:** the dispatch router is workflow surface (`src/explore_persona_space/backends/*.py`). Two concurrent inline rounds on one issue is a supported and increasingly common pattern (the whole point of `lane_suffix`, #934), and it works on GCP. On SLURM it fails SILENTLY — the second lane believes it dispatched, its handle looks healthy, and its workload simply never runs. There is no error, no marker, and no fail-loud signal; the only symptom is `workload_dispatched=false` buried in the router return.
- **Confidence (emitter):** high — reproduced twice on one issue, with both handles inspected on disk.
- verified-at-filing: `grep -rn 'lane_suffix' src/explore_persona_space/backends/` + `grep -rn 'eps-issue' src/explore_persona_space/backends/*.py` + `grep -n 'def job_name' src/explore_persona_space/backends/slurm.py` (2026-08-03 UTC). Per-target hits: `slurm.py` — `job_name()` at 697-715 returns `f"eps-issue-{spec.issue}-{plan_hash[:8]}{suffix}"` or `f"eps-issue-{spec.issue}{suffix}"`, where `suffix` is `cluster.job_name_suffix` (a per-CLUSTER string, e.g. `-superkaiba` for fellows rule 8) and **`lane_suffix` appears nowhere in the file**; `gcp.py:918` — instance name IS `eps-issue-<N>[-<lane_suffix>]` (#934 comment: "the optional suffix lets two..."); `issue_dispatch.py:178-201` — `default_handle_sidecar_path` DOES apply `lane_suffix` to the sidecar stem. So the suffix reaches two of the three identity surfaces and misses the SLURM one.

**Live evidence (task #1947, 2026-08-03).** Two concurrent inline rounds dispatched on issue 1947 with correctly distinct sidecars:
- `.claude/cache/issue-1947-r3theory-handle.json` → cluster `nibi`, job `19036499`, `pod_name: eps-issue-1947`, `submitted_at: 1785799680` (submitted)
- `.claude/cache/issue-1947-locpanel-handle.json` → cluster `nibi`, job `19036499`, `pod_name: eps-issue-1947`, `submitted_at: None` (reconnected)

Both handles point at the SAME job. The second lane reconnected twice with `workload_dispatched=false`. Distinct sidecar paths did not help, because reconnection keys on the job NAME at the cluster (`squeue --name eps-issue-<N>`, router.py:130 / 1879 / 2128 / 5961), not on the sidecar.

## Proposed change (candidate diff sketch — refine in planning)

```
 def job_name(
-    spec, plan_hash=None, *, cluster=None,
+    spec, plan_hash=None, *, cluster=None, lane_suffix=None,
 ):
     suffix = cluster.job_name_suffix if cluster is not None and cluster.job_name_suffix else ""
+    lane = f"-{validate_lane_suffix(lane_suffix)}" if lane_suffix else ""
     if plan_hash:
-        return f"eps-issue-{spec.issue}-{plan_hash[:8]}{suffix}"
-    return f"eps-issue-{spec.issue}{suffix}"
+        return f"eps-issue-{spec.issue}-{plan_hash[:8]}{lane}{suffix}"
+    return f"eps-issue-{spec.issue}{lane}{suffix}"
```
plus threading `lane_suffix` from the dispatch entrypoint through `router` into every `job_name(...)` call site (slurm.py:1918, 3125) and into the `squeue --name` reconnect lookups so a suffixed lane only ever matches its own job.

**Note for planning:** the `plan_hash` branch already disambiguates by construction, so the collision window is dispatches with no plan hash — which is exactly the inline-round path. Worth checking whether the cleanest fix is threading the lane suffix or making the no-plan-hash branch carry a per-lane discriminator.

**Secondary (consider in the same plan):** when reconnect returns an existing job whose handle was written by a DIFFERENT lane sidecar, that is almost certainly a collision rather than a legitimate resume. A loud warning — or a refusal absent an explicit `--allow-reconnect` — would convert this silent failure into a visible one even before the naming fix lands.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/slurm.py`
- Also touched: `src/explore_persona_space/backends/router.py` (the `squeue --name` reconnect predicate), `src/explore_persona_space/backends/issue_dispatch.py` (threading), `scripts/dispatch_issue.py` (flag passthrough).
- Regression test belongs in `tests/test_slurm_*.py`: two specs on the same issue with distinct `lane_suffix` must produce distinct job names, and reconnect must not cross-match them.
- GCP behavior must not change (`gcp.py:918` already correct); the fellows per-cluster `job_name_suffix` (rule 8) must be preserved alongside, not replaced.

## Constraints / invariants

- Workflow-surface only — no experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` passes; ruff on touched files passes.
- Existing in-flight handles must keep resolving (a rename scheme that orphans live jobs is not acceptable — check `reconnect_or_none`'s tolerance for a name change mid-flight).
- This session runs under the recursion guard and does NOT auto-file further candidates.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/slurm.py
- fingerprint: b3afa7619eaf

Surfaced from task #1947 while running two concurrent user-directed inline GPU rounds (a 52-arm localization panel and a condition-grain theory battery). Immediate mitigation applied by hand: the two lanes were manually assigned to different clusters (localization → fellows/charmander, theory battery → nibi), which sidesteps the collision but does not fix it — any two lanes landing on the same cluster will collide again.
