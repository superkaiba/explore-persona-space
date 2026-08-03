---
title: 'workflow-fix: Step 10d probes PR state; verify landing before verdict consume'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bd8a8a5ca538
created_at: '2026-07-30T20:10:14Z'
has_clean_result: false
origin_prompt: 'orchestrator own-observation on #1768 round-2 follow-up merge: gh
  pr merge 1527 exited 0 with ''was already merged'' (round-1 PR, closed) -> false
  MERGE SUCCEEDED, verdict consumed, 22-commit round-2 payload stranded off main;
  recovered via fresh PR 1604 + gate re-run'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1768 (emitting agent: orchestrator, own observation during the
round-2 same-issue follow-up Step 10d merge).

## Goal

Step 10d merge entry probes `gh pr view <PR> --json state` and creates a fresh PR when the recorded PR is MERGED/CLOSED, and the merge conditional verifies the certified tip actually landed instead of trusting `gh pr merge` exit 0.

## Workflow gap

- **Bug observed:** on #1768's round-2 same-issue follow-up (2026-07-30), the Step 10d merge conditional ran `gh pr merge 1527 --rebase` against the ROUND-1 PR (rebase-merged + closed at round 1, mergeCommit 2d594888b17). `gh` exited 0 with the warning `! Pull request ... #1527 was already merged`, so the success arm fired: the SHA-bound verdict was consumed (`rm -f`), root-sync ran, and "MERGE SUCCEEDED" was reported — while the 22-commit round-2 payload stayed stranded off main (verified: `git merge-base --is-ancestor` NOT-on-main for all round-2 shas). Recovery cost: a fresh PR (#1604) + a full ~15-min gate re-run (the consumed verdict cannot be hand-rewritten, #1082).
- **Why it is a workflow gap:** the same-issue follow-up loop re-enters Step 10d with the task's existing PR number, but a rebase/squash-merged PR is a TERMINAL GitHub object — new branch commits never attach to it. Nothing in Step 10d probes PR state before the merge conditional (the Step 10d idempotency bullet skips only when "no PR exists or the branch is already merged into main" — here the BRANCH was 22 ahead, only the PR OBJECT was terminal), and the exit-0 already-merged shape is absent from the classified failure shapes 0-3 (which all key on NON-zero exits). The success arm therefore cannot distinguish "this merge landed the certified tip" from "a previous round's merge already closed this PR".
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'was already merged' .claude/skills/issue/SKILL.md` → 0 hits (absence claim: the exit-0 already-merged shape is unhandled; the shape taxonomy at lines 12128/12163/12180/12264 covers only Base-modified / can't-be-rebased / merge-conflicts / head-out-of-date, all non-zero-exit shapes); `grep -n 'already merged' .claude/skills/issue/SKILL.md` → 4 hits, none a pre-merge PR-state probe (10281 = the branch-already-merged idempotency skip; 10585 = resume-time; 10822/13732 unrelated); live incident evidence: gh run transcript on #1768 2026-07-30 (`gh pr view 1527 --json state,mergedAt` → MERGED 2026-07-30T14:53:45Z; 22 commits `NOT on main` post-"success") (2026-07-30)

## Proposed change (candidate diff sketch — refine in planning)

```
# .claude/skills/issue/SKILL.md, Step 10d before the merge conditional (and the
# Step 9b same-issue follow-up merge trigger that re-enters it):
+ # PR-object liveness probe (#1768 round-2 incident): a follow-up round's
+ # branch outlives its round-1 PR — a MERGED/CLOSED PR never merges new
+ # commits, and `gh pr merge` on one exits 0 with "was already merged"
+ # (false success: verdict consumed, payload stranded). Probe FIRST:
+ PR_STATE=$(gh pr view <PR> --json state -q .state 2>/dev/null || echo MISSING)
+ if [ "$PR_STATE" != "OPEN" ]; then
+   # create a fresh draft PR via the Step 4a pre-checked block; rebind <PR>
+ fi
# AND harden the success arm: after `gh pr merge` exits 0, verify the landing
# (e.g. bounded poll: git fetch origin main && merge-base --is-ancestor
# $(certified tip or its rebased equivalents) origin/main, or
# gh pr view --json mergeCommit newer than the round start) BEFORE consuming
# the verdict; on verification failure treat as MERGE FAILED, do not rm the verdict.
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'gh pr merge' .claude/ CLAUDE.md scripts/`) and check every
  merge site (Step 9b trigger, Step 10d safe case + recovery + surgical forms,
  the completed-unmerged watcher pass in scripts/autonomous_session_watch.py
  which probes PR state read-only); update every affected site and list them
  in the plan. Note the rebase form lands rebased COPIES of the branch commits
  (new shas), so the landing verification must not require the branch shas
  themselves to be ancestors — verify via the PR object's own mergeCommit /
  the branch-vs-main content diff instead.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).
- The #1082 anti-self-attestation doctrine is untouched: the fix must never
  legitimize hand-writing the verdict file; the landing verification only
  gates verdict CONSUMPTION.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: bd8a8a5ca538

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: gh pr merge on the round-1 PR exited 0 with already-merged warning during a same-issue follow-up round, producing a false MERGE SUCCEEDED that consumed the verdict while the 22-commit round-2 payload stayed stranded off main
why_workflow_gap: a rebase-merged PR is a terminal GitHub object, but Step 10d has no PR-state probe before the merge conditional and the exit-0 already-merged shape is outside failure shapes 0-3, so the success arm cannot distinguish a landed merge from a prior round's closed PR
proposed_change: Step 10d merge entry probes gh pr view state and creates a fresh PR when the recorded PR is MERGED or CLOSED, and the merge conditional verifies the certified tip actually landed instead of trusting gh pr merge exit 0
diff_sketch: |
  + PR_STATE=$(gh pr view <PR> --json state -q .state 2>/dev/null || echo MISSING)
  + [ "$PR_STATE" != "OPEN" ] && { create fresh draft PR via Step 4a block; rebind <PR>; }
  + success arm: verify landing (mergeCommit / content diff) BEFORE rm verdict; else treat as MERGE FAILED
confidence: high
related_task: #1768
<!-- /workflow-fix-candidate -->
