---
title: 'workflow-fix: scope preflight branch-freshness fetch, raise timeout'
kind: infra
tags:
- wf-fix
- wf-fix-fp:109b2c35251b
created_at: '2026-08-06T03:00:30Z'
has_clean_result: false
origin_prompt: 'Orchestrator-raised on #2091: preflight.py:355 full ''git fetch --quiet
  origin'' with timeout=15 is unsatisfiable at 1633 remote refs (measured 192s, rc=0);
  feature-branch failed-fetch is a hard ERROR while main/detached fail soft, so every
  issue-branch pod preflight FAILs even at 0 ahead/0 behind.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a gap the orchestrator hit
directly while running the mandatory pre-launch preflight on `pod-2091`
(task #2091). Fleet-wide: it fails EVERY feature-branch pod preflight, not just
this one.

## Goal

Scope the preflight branch-freshness fetch to the current branch
(`git fetch origin <branch>`) and raise its timeout, so a repo with many remote
refs cannot make the check unsatisfiable.

## Workflow gap

- **Bug observed:** `src/explore_persona_space/orchestrate/preflight.py:355`
  runs the FULL `git fetch --quiet origin` with `timeout=15`. The repo now
  carries **1,633 remote branches** (stale `issue-*` branches accumulate), and
  a full fetch measured **192 s** on `pod-2091` (rc=0 — it SUCCEEDS, it is only
  slow; a warm re-run still exceeds 15 s on ref negotiation alone). Because
  `_check_feature_branch_behind` treats a failed fetch as a hard ERROR (while
  `_check_main_branch_behind` and `_check_detached_head_behind` both
  deliberately fail-SOFT to a WARNING on the same condition), every preflight
  run on an `issue-<N>` branch now reports
  `Pre-flight Check: FAIL / git fetch origin failed (timeout) — cannot verify
  branch <b> is up to date with origin/<b>` even when the branch is provably at
  `0 ahead / 0 behind`.
- **Why it is a workflow gap:** the pre-launch preflight is MANDATORY for every
  experimenter launch (CLAUDE.md § Pre-launch protocol step 2, "Fix any failure
  — don't skip"), so a check that cannot pass on any feature branch either
  blocks every pod launch or trains every session to wave the FAIL through —
  which then hides the REAL failures the same gate exists to catch (GPU
  contention, env drift, EDQUOT headroom). The fetch is also doing far more
  work than the check needs: the check only compares HEAD against
  `origin/<current-branch>`, so fetching all 1,633 refs is pure waste. A
  targeted `git fetch origin <branch>` completed in seconds on the same pod in
  the same session.
- **Confidence (emitter):** high
- verified-at-filing:
  `grep -n 'fetch", "--quiet", "origin"\], timeout=15' src/explore_persona_space/orchestrate/preflight.py`
  → 1 hit (line 355) in the single named target file (2026-08-05);
  `git ls-remote --heads origin | wc -l` → 1633 (VM-side, same value read
  pod-side); timed pod-side full fetch `rc=0 elapsed=192s` under a 600 s cap;
  targeted `git fetch origin issue-2091` completed in seconds; the
  fail-soft-vs-ERROR asymmetry read directly from
  `_check_main_branch_behind` / `_check_detached_head_behind` (WARNING) vs
  `_check_feature_branch_behind` (ERROR) in the same file.

## Proposed change (candidate diff sketch — refine in planning)

```
- fetch_rc, _, fetch_err = _run(
-     ["git", "-C", str(project_root), "fetch", "--quiet", "origin"], timeout=15
- )
+ # Scope to the branch actually being verified: the freshness check only
+ # compares HEAD against origin/<branch>, so fetching every remote ref is
+ # unnecessary work that scales with branch count (1,633 refs -> ~192 s).
+ _fetch_cmd = ["git", "-C", str(project_root), "fetch", "--quiet", "origin"]
+ if branch and branch != "HEAD":
+     _fetch_cmd.append(branch)
+ fetch_rc, _, fetch_err = _run(_fetch_cmd, timeout=90)
```

Two dependent details for the planner to resolve, NOT settled here:

1. `branch` is currently resolved AFTER this fetch call in the same function —
   the branch-name resolution must move above the fetch (or the fetch must be
   deferred) for the targeted form to work. Verify the ordering rather than
   assuming it.
2. A targeted fetch updates only `refs/remotes/origin/<branch>`, so the
   `origin/main` comparisons that follow (`_behind_count(project_root,
   "origin/main")` in both the main-branch and detached-HEAD paths, plus the
   feature-branch informational divergence WARNING) would read a
   possibly-staler `origin/main`. Either fetch both refs
   (`git fetch origin <branch> main`) or keep the `origin/main` comparison
   explicitly fail-soft and say so in the summary line. Do not silently
   degrade an existing check.

The timeout value (90 s above) is a placeholder — ground it on a measured
targeted-fetch time across lanes (RunPod pod, GCE, the shared VM), not on this
sketch.

## Scope / surfaces

- Primary target: `src/explore_persona_space/orchestrate/preflight.py`
- Also likely: `tests/test_preflight_disk.py` or a sibling preflight test — add
  a regression pin that a feature-branch preflight does NOT hard-FAIL when the
  branch is in sync but a full-remote fetch would be slow (fake the fetch seam).
- Grep the workflow surface for other unscoped full-remote fetches under a
  short timeout before editing
  (`grep -rn 'fetch", "--quiet", "origin"' src/ scripts/`), and list every hit
  in the plan.

## Constraints / invariants

- Do NOT weaken the feature-branch behind/diverged ERROR itself — a branch that
  is genuinely behind its own origin ref must still hard-FAIL. The fix is to
  make the fetch that FEEDS that check reliable, not to demote the check.
- `uv run python scripts/workflow_lint.py` passes; ruff on touched files passes.
- The cluster branch already SKIPS the fetch round trip
  (`report.git_status += " (cluster — skipped fetch)"`) — keep that path
  byte-unchanged.

## Related concern (separate, do not bundle)

1,633 remote branches is itself worth a look — stale `issue-*` branches are
never pruned after their Step 10d merge, and every clone/fetch pays for them.
That is a distinct hygiene question with a different target surface (a branch
janitor, likely alongside `worktree_audit.py` / `cron_worktree_audit.sh`) and
should be filed on its own rather than folded into this fix.

## Provenance

- workflow_fix_target: src/explore_persona_space/orchestrate/preflight.py
- fingerprint: 109b2c35251b

Raised directly by the orchestrator of task #2091 while running the mandatory
pre-launch preflight on `pod-2091` (4x H100). The run proceeded on an
independently verified invariant (`git rev-list --left-right --count
HEAD...origin/issue-2091` → `0 0`, plus a full fetch confirmed rc=0), with the
disposition recorded in an `epm:progress` marker on #2091.
