---
title: 'workflow-fix: gcp startup-script repo-reuse else-branch is not branch-switch-safe'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6515d3b71eff
created_at: '2026-07-22T16:51:22Z'
has_clean_result: false
origin_prompt: 'orchestrator observation on task 779 att-20260722-155004: reused workspace
  disk + single-branch clone kills a cross-attempt branch change at boot; see the
  body Provenance candidate block'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator observation on task #779 (session cmrw957yhtxehyc0u42xnr9jx, capacity-retry re-drive, 2026-07-22).

## Goal

Make the GCP startup-script repo-reuse else-branch branch-switch-safe: create/reset the target local branch from FETCH_HEAD instead of a by-name branch switch after a bare fetch.

## Workflow gap

- **Bug observed:** on a reused workspace disk holding a single-branch clone of a DIFFERENT branch, the else-branch of the "Repo clone / pull (idempotent)" block in `render_startup_script` (`git -C "$WORKLOAD_ROOT" fetch --depth 1 origin <repo_branch>` followed by a by-name branch switch to `<repo_branch>`) fails with `error: pathspec '<repo_branch>' did not match any file(s) known to git` — the bare fetch lands only in FETCH_HEAD and creates no local/remote-tracking ref on a single-branch clone. The instance powers off rc=1 ~30s after dequeue. Live incident: issue #779 att-20260722-155004 (2026-07-22 16:39:30-16:40:00Z): today's `eps-issue-779` create silently reused the July-2 300 GB boot disk (also: `--boot-disk-gb 200` was ignored by the reuse) whose `/workspace/eps-issue-779` was a single-branch `issue-779` clone; the round's `--repo-branch issue-779-n1m-readout` then hit exactly this. Crash diagnostics: HF `superkaiba1/explore-persona-space-data/issue779_partial/att-20260722-155004/workload.log`.
- **Why it is a workflow gap:** `backends/gcp.py` is workflow surface (the dispatch layer); the else-branch exists deliberately for idempotent workspace reuse, but it is only correct when the reused clone is already on `repo_branch`. Any cross-attempt branch change on a surviving disk deterministically kills the run at boot — and burns a whole flex-start queue wait (this one cost a ~50 min queue slot).
- **Confidence (emitter):** high
- verified-at-filing: `grep -rn 'fetch --depth 1 origin' src/explore_persona_space/backends/ scripts/bootstrap_pod.sh scripts/` → 3 hits in 3 files; per-target: `src/explore_persona_space/backends/gcp.py` 1 hit (line 2831, the else-branch — the defect site; the by-name branch switch + `reset --hard origin/<branch>` follow at lines 2832-2833); `scripts/issue778_v2_dispatch.sh` / `scripts/issue778_dispatch.sh` fetch pinned SHAs into `external/` (out of scope, not the pattern). `scripts/bootstrap_pod.sh` lines 257-271 already implement the SAFE form (`git fetch -q --depth=1 origin "$BRANCH"` + `git reset -q --hard FETCH_HEAD`, with a comment explaining why a by-name form fails) — the RunPod lane does not share this bug (2026-07-22).

## Proposed change (candidate diff sketch — refine in planning)

In `render_startup_script`'s else-branch, replace the by-name branch switch + `reset --hard origin/<repo_branch>` pair with a FETCH_HEAD-anchored form that works on any reused clone regardless of its current branch / single-branch refspec, e.g. mirror `scripts/bootstrap_pod.sh` lines 257-271 (fetch then `reset --hard FETCH_HEAD` on whatever branch is current), or create/reset the local branch from FETCH_HEAD (`checkout -B <repo_branch> FETCH_HEAD`). Either removes the dependence on a local/remote-tracking ref existing. Planner should also note (context, possibly its own follow-up): the boot-disk reuse pathway itself — today's create attached the 20-day-old surviving 300 GB disk and silently ignored `--boot-disk-gb 200`; whether that reuse is by-design (idempotent else-branch implies yes) or should be age-gated/verified is a judgment call for the plan.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Grep the workflow surface for the pattern before editing (`grep -rn 'fetch --depth 1 origin' src/ scripts/ .claude/`) and update every in-scope hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; render-tested via the existing `tests/test_gcp_backend.py` startup-script assertions (extend them to pin the branch-switch-safe form).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: 6515d3b71eff

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/gcp.py
bug_observed: on a reused workspace disk with a single-branch clone of a different branch, the else-branch fetch plus by-name branch switch to repo_branch fails with pathspec not found and the instance powers off (issue 779 att-20260722-155004)
why_workflow_gap: the idempotent repo-reuse else-branch in render_startup_script is only correct when the reused clone is already on repo_branch; any cross-attempt branch change on a surviving disk deterministically kills the run at boot
proposed_change: make the gcp startup-script repo-reuse else-branch branch-switch-safe: create/reset the target branch from FETCH_HEAD instead of a by-name checkout
diff_sketch: |
  -  git -C "$WORKLOAD_ROOT" fetch --depth 1 origin <repo_branch>
  -  <by-name branch switch to repo_branch>
  -  git -C "$WORKLOAD_ROOT" reset --hard origin/<repo_branch>
  +  git -C "$WORKLOAD_ROOT" fetch --depth 1 origin <repo_branch>
  +  <create/reset local repo_branch from FETCH_HEAD>   # bootstrap_pod.sh L257-271 is the proven sibling form
confidence: high
related_task: #779
<!-- /workflow-fix-candidate -->
