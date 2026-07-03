---
title: 'workflow-fix: single-flight repo-root sync helper (untracked-collision sweep)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0b509d58d95f
created_at: '2026-07-03T01:11:30Z'
has_clean_result: false
origin_prompt: '2026-07-02 shared-root divergence incident during the serial-compute
  audit batch; user context: ''anything left to handle here?'''
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the 2026-07-02 shared-root divergence incident (PM-chat session, follow-on to the serial-compute audit batch #869-#876).

## Goal

Add a single-flight root-sync helper so shared-repo-root reconciliation stops being an ad-hoc race between sessions.

## Workflow gap

- **Bug observed:** on 2026-07-02 the repo root sat 582 local / 226 origin commits diverged for ~3h. Every pull-rebase recovery attempt (this session's and at least two other sessions' retry loops, observed live via pgrep) failed the same ways: (a) "untracked working tree files would be overwritten by checkout" — sessions continuously materialize eval_results/figures artifacts at the root uncommitted while worktree sessions commit the same files to origin (81 collisions in one attempt, 148 in the next, growing between attempts); (b) index.lock contention between concurrent recovery loops ("could not detach HEAD", "fatal: could not reset --hard"); (c) stranded/husk autostashes (three rescue stashes accumulated in one day). Origin/main had also LOST task #811's entire folder while its REGISTRY still pointed at tasks/interpreting/811 — recovered manually via a scratch-worktree merge (pushed as 6c1b3fadf7).
- **Why it is a workflow gap:** CLAUDE.md § Concurrent repo-root committers prescribes "git pull --rebase=merges --autostash once and re-push" but has no answer for untracked-collision failures, no single-flight coordination (N sessions all retry independently and collide), and no stranded-autostash recovery. The identical-vs-differing sweep that actually unblocks the pull was improvised inline twice today.
- **Confidence (emitter):** high (live incident; evidence in the 2026-07-02 PM-chat transcript).

## Proposed change (candidate diff sketch — refine in planning)

New `scripts/sync_repo_root.py` (workflow-helper):
- Single-flight via flock on `~/.task-workflow/root-sync.lock` (second caller exits 0 "another sync in flight").
- Pre-sweep: enumerate origin-only tree paths colliding with untracked root files; byte-identical → rm (content is in git); differing → move to a dated rescue dir (never delete non-identical data); report counts.
- `git pull --rebase=merges --autostash` with bounded timeout; on a mid-rebase conflict confined to `tasks/<status>/<N>` lifecycle files, auto-resolve keeping the LATER lifecycle state; anything else → abort cleanly + report (never leave a husk).
- Detect + restore stranded autostash entries; verify no MERGE_HEAD / rebase-merge husk remains.
- Push with one rebase-retry.
- CLAUDE.md § Concurrent repo-root committers: point the recovery sentence at the helper; session inline retry loops (the "for i in 1..5: pull" patterns observed today) call the helper instead.

## Scope / surfaces

- Primary targets: new `scripts/sync_repo_root.py`, `CLAUDE.md` (§ Concurrent repo-root committers), tests
- Consider a watcher pass (autonomous_session_watch.py) invoking it when divergence exceeds a threshold — the plan decides whether that leg ships in v1.

## Constraints / invariants

- NEVER a repo-root `git reset --hard` / `git clean -f` (standing ban); never delete non-identical untracked data (rescue dir only); fail loud, no silent husks.
- `scripts/workflow_lint.py` default run + ruff + tests pass.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: CLAUDE.md, scripts/sync_repo_root.py
- fingerprint: 0b509d58d95f

<!-- workflow-fix-candidate v1 -->
target_file: CLAUDE.md, scripts/sync_repo_root.py
bug_observed: root pull-rebase recovery repeatedly fails on untracked eval_results/figures collisions materialized by concurrent sessions; multiple sessions race independent recovery loops (index.lock contention, could-not-detach-HEAD, stranded autostashes); local main sat 582/226 diverged for hours on 2026-07-02
why_workflow_gap: the documented recovery (pull --rebase=merges once) has no untracked-collision handling, no single-flight coordination, and no stranded-autostash recovery, so every session improvises and they collide
proposed_change: add a single-flight scripts/sync_repo_root.py helper (flock; identical-rm / differing-rescue sweep of untracked collisions; pull --rebase=merges --autostash; push; stranded-autostash restore) and point the CLAUDE.md concurrent-committers recovery and session retry loops at it
diff_sketch: |
  + scripts/sync_repo_root.py: flock single-flight; pre-sweep untracked
  +   collisions (identical->rm, differing->rescue dir); bounded pull
  +   --rebase=merges --autostash; lifecycle-file auto-resolve; husk +
  +   stranded-autostash detection/repair; push with one retry
  + CLAUDE.md: recovery sentence points at the helper
confidence: high
related_task: n/a (2026-07-02 divergence incident)
<!-- /workflow-fix-candidate -->
