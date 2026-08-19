---
title: 'workflow-fix: reclaim tier for ~/.eps-slurm-src terminal-issue staging (112
  GB unreaped)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d7cab4eeb1be
created_at: '2026-08-06T07:35:14Z'
has_clean_result: false
origin_prompt: 'Orchestrator observation while dispatching #2061: / at 98% (920G/945G),
  vm_disk_guard --apply freed 0.00G, and ~/.eps-slurm-src holds 112 GB across 13 per-issue
  dirs that no janitor reaps (slurm.py:3656 documents the exclusion).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised while
dispatching task #2061 (emitting agent: orchestrator, own observation).

## Goal

Add a `vm_disk_guard.py` reclaim tier for `~/.eps-slurm-src` per-issue SLURM source staging of
TERMINAL issues, under the same safety contract as the existing tiers (terminal-status gate,
active-issue escalate-only, recency window, never delete `store/` or `eval_results/`).

## Workflow gap

- **Bug observed:** No janitor reaps `~/.eps-slurm-src`. 13 per-issue staging dirs have
  accumulated 112 GB on a boot disk that is now 98% full (945 GB total, 920 GB used, ~25 GB
  free) — the third-largest consumer on `/` after the worktree trees (234 GB) and the repo
  `data/` caches (208 GB). `vm_disk_guard.py --apply --ignore-threshold` freed 0.00 GB and
  printed its own "manual triage needed" warning; the automated reclaim path is exhausted while
  112 GB of by-construction-regenerable staging sits untouched.
- **Why it is a workflow gap:** the staging is a `git worktree add --detach` checkout created
  per issue by `backends/slurm.py::materialize_branch_src` purely to give the SLURM lane a
  branch-consistent rsync source. It is regenerable on the next `prepare()` by construction, so
  a terminal issue's copy is pure reclaimable cache — exactly the class `vm_disk_guard.py`'s
  tier (b) already reaps for `hf_dl`/`g*_dl` and the #911 non-canonical `/tmp` set. The path was
  simply never added to any tier, and the code that creates it SAYS SO: `slurm.py:3656` reads
  "``vm_disk_guard.py`` tiers never touch ``~/.eps-slurm-src``".
- **Confidence (emitter):** high — the absence is a 0-hit grep in every janitor, and the writer's
  own docstring states the exclusion.
- verified-at-filing: `grep -c 'eps-slurm-src' <target>` on the main tree (2026-08-06) →
  `scripts/vm_disk_guard.py` **0 hits**, `scripts/worktree_audit.py` **0 hits**,
  `scripts/clean_experiment_downloads.py` **0 hits**, while the WRITER
  `src/explore_persona_space/backends/slurm.py` has **5 hits** (incl. the
  self-documenting exclusion at `:3656`). Repo-wide main-tree total 5 hits, all in the writer.
  This is an ABSENCE-OF-GUARD claim, so the 0-hit result in the named target IS the evidence
  (per the mis-target-rule exemption). Landed-fix check:
  `git log --oneline --since='7 days ago' -- scripts/vm_disk_guard.py` → no fix to this gap
  (single unrelated task-marker commit).

## Proposed change (candidate diff sketch — refine in planning)

  # scripts/vm_disk_guard.py — new tier, alongside the existing hf_dl/g*_dl terminal reap
  + SLURM_SRC_ROOT = Path(os.environ.get("EPS_SLURM_SRC_ROOT") or Path.home() / ".eps-slurm-src")
  +
  + def _reap_slurm_src(*, apply: bool) -> float:
  +     """Reap ~/.eps-slurm-src/issue-<N> for TERMINAL issues only.
  +
  +     Same contract as tier (b): status read-only + never mutated; an ACTIVE
  +     issue's dir is attributed + escalated, never deleted; honor the 48h
  +     recency keep; a live `git worktree list` registration blocks the reap
  +     (the checkout is a registered worktree of the main repo, so `git
  +     worktree remove` / `prune` is the correct instrument, not rmtree).
  +     """

Two details the planner must settle, flagged rather than guessed:
1. These are REGISTERED git worktrees of the main repo, so a bare `rmtree` would leave a stale
   registration behind. The reap should go through `git worktree remove` (or `remove` +
   `worktree prune`), matching how `materialize_branch_src` itself removes a prior copy.
2. Whether the tier belongs in `vm_disk_guard.py` (which owns byte-floor reclaim on `/`) or in
   `worktree_audit.py` (which owns worktree lifecycle and already knows how to reap a registered
   worktree safely). `vm_disk_guard.py` is named here because the writer's docstring points at
   it and because this surfaced as a disk-pressure failure, but `worktree_audit.py` may be the
   better home — the planner decides with both files open.

## Scope / surfaces

- Primary target: `scripts/vm_disk_guard.py`
- Writer / reference for safe removal: `src/explore_persona_space/backends/slurm.py`
  (`materialize_branch_src`, its remove-prior + `worktree prune` sequence)
- Candidate alternative home: `scripts/worktree_audit.py`
- Sibling janitor whose contract to mirror: the tier-(b) terminal reap +
  `scripts/clean_experiment_downloads.py`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The existing safety contract is non-negotiable: terminal-status gate, status read-only,
  ACTIVE-issue caches escalate-only (never deleted), recency keep, and `store/` +
  `eval_results/` never touched.
- `scripts/workflow_lint.py` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:`
  Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates
  (recursion guard).

## Sizing evidence (design assumption vs realized)

`slurm.py:2853` and `:3653` both size the staging at "~3.8 GB" per issue. Realized: 112 GB
across 13 dirs ≈ 8.6 GB each, i.e. ~2.3x the design figure, and unbounded in issue count. #793's
own plan v2 recorded the reap gap explicitly ("The worktree-audit cron does NOT reap
`~/.eps-slurm-src/`. Confidence: High") and named the mitigation as "per-prepare refresh + the
disk guard". That chain has a hole in BOTH links: the per-prepare refresh only removes the SAME
issue's prior copy (never a sibling issue's), and the disk guard excludes the path by its own
documentation. So nothing has ever reclaimed a terminal issue's copy.

**unverified hypothesis — verify at plan time:** that the 2.3x overshoot is explained by
`WORKING_TREE_OVERLAY_PATHS` rsyncing `external/open-instruct` into each checkout on top of the
detached worktree (the overlay is working-tree content the shared object DB does not
deduplicate). The 3.8 GB figure plausibly predates or excludes the overlay. Not measured — the
planner should confirm before citing a cause.

## Provenance

- workflow_fix_target: scripts/vm_disk_guard.py
- fingerprint: d7cab4eeb1be

<!-- workflow-fix-candidate v1 -->
target_file: scripts/vm_disk_guard.py
bug_observed: No janitor reaps ~/.eps-slurm-src; 13 per-issue dirs accumulated 112 GB on a 98%-full boot disk
why_workflow_gap: The staging is regenerable-by-construction per-issue cache (a detached git worktree created by backends/slurm.py::materialize_branch_src) in exactly the class vm_disk_guard tier (b) already reaps, but the path was never added to any tier — the writer's own docstring at slurm.py:3656 states "vm_disk_guard.py tiers never touch ~/.eps-slurm-src".
proposed_change: Add a vm_disk_guard.py reclaim tier for ~/.eps-slurm-src per-issue SLURM source staging of TERMINAL issues
diff_sketch: |
  + SLURM_SRC_ROOT = Path(os.environ.get("EPS_SLURM_SRC_ROOT") or Path.home() / ".eps-slurm-src")
  + def _reap_slurm_src(*, apply: bool) -> float:
  +     # terminal-status gate; ACTIVE issues attributed + escalated only;
  +     # 48h recency keep; reap via `git worktree remove` + prune (these are
  +     # REGISTERED worktrees — rmtree would strand the registration).
confidence: high
related_task: #2061
<!-- /workflow-fix-candidate -->
