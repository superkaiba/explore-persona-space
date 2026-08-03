---
title: 'daily-fix: Step 10d apply-verification + pre-sync'
kind: infra
tags:
- wf-fix
- wf-fix-fp:424d6bf46eee
- daily-auto-filed
created_at: '2026-07-26T07:03:21Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): A squash whose message
  claimed the content applied cleanly landed a test file but not the extractor half
  it tested, leaving a five-failure red main for 20 days until #1683 ported it, and
  the post-merge canonical-folder guard exits nonzero on every infra task because
  the completed status move commits locally without pushing.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 problem sweep. A Step 10d recovery path wrote a
commit message asserting an apply that did not happen, and main stayed red for 20 days
as a result.

## Goal

Verify after a Step 10d guard-3 unsafe-rebase content-apply that every claimed path
actually landed before writing an "applied cleanly" commit message, and run
`scripts/sync_repo_root.py` unconditionally BEFORE the post-merge canonical-folder
guard rather than as its failure remedy.

## Workflow gap

1. **A false "applied cleanly" message left main red for 20 days.** Squash
   `3c24493113` (2026-07-05, a guard-3 unsafe-rebase content-apply, 212 files /
   68,985 insertions) landed `tests/test_issue811_pre_user.py` but **not** the
   extractor half the test exercises. Its message claimed *"applied as a 3-way patch
   (clean)"*. Session `ea7470c1` (#1683) verified the gap on 2026-07-25:
   *"the commit diffstat shows no extractor change, and `git log -S PRE_USER_LAYER_ARMS
   main -- scripts/issue667_extract.py` is EMPTY: the pre-user extractor changes never
   landed on main. Built-but-stranded class."* Seven test-referenced symbols were
   missing; the file sat 5-failed on main for 20 days until #1683 ported it
   (`61f2959ec4`). The content-apply recovery path has no post-apply verification that
   the files it claims to have applied actually landed.
2. **The post-merge canonical-folder guard exits nonzero on every infra task.** In
   #1688 (`6d798257`) @ 18:19:37Z the guard exited 1 because `tasks/completed/1688` was
   absent from `origin/main` — the local `set-status completed` had moved and committed
   the folder on repo-root `main` but never pushed. `sync_repo_root.py` fixed it in one
   attempt (`ahead=8 behind=2`). The guard-then-sync dance is structural, not
   incidental: every task reaching `completed` pays it, and a guard whose exit-1 path
   is the NORMAL case cannot signal genuine drift.
- **Confidence (emitter):** high on both; (1) is the more serious (a silent
  wrong-state commit message), (2) is a papercut with a signal-quality cost.
- verified-at-filing: SHA resolution per clause (d) —
  `git rev-parse --verify --quiet '3c24493113^{commit}'` resolves;
  `git rev-parse --verify --quiet '61f2959ec4^{commit}'` resolves to the #1683 port
  merge ("fix(#1683): port stranded #811 pre-user extractor onto main (#1455)", landed
  2026-07-25). The stranded-state claim is #1683's own recorded `git log -S` probe
  (quoted verbatim above), i.e. the session verified it against main, not from recall.
  For (2): the guard's exit-1 + `CANON=tasks/completed/1688` text is quoted from
  #1688's own tool output. Landed-fix history check
  `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` → 8
  commits in the wave; none touches the guard-3 apply verification or the post-merge
  guard ordering. (2026-07-25)

## Proposed change (refine in planning)

```
  Step 10d guard-3 (unsafe-rebase content-apply path):
+ after the apply, diff the applied tree against the SOURCE branch for the claimed
+ path set; refuse to write an "applied as a 3-way patch (clean)" message unless
+ every claimed file's content matches, or the message explicitly lists the files
+ intentionally dropped and why.

  Step 10d post-merge canonical-folder guard:
+ run `uv run python scripts/sync_repo_root.py` UNCONDITIONALLY before the guard,
+ so the guard's nonzero exit is reserved for genuine drift rather than the
+ expected unpushed-mv state.
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 10d — guard 3, and the
  post-merge canonical-folder guard block).
- Check whether the "unpushed-mv arm" text in the guard's own diagnostic needs
  rewording once the pre-sync makes that arm rare.
- Do NOT relax the guard itself — (2) reorders when the sync runs, it does not remove
  a check.
- Scan for other in-repo recovery paths that write an outcome-asserting commit message
  without verifying the outcome; if the pattern recurs, the fix generalizes and the
  planner should say so rather than patching one site.

## Constraints / invariants

- (1) must fail LOUD: a partial apply becomes an explicit, listed partial, never a
  clean-sounding message (CLAUDE.md § Critical Rules "Fail fast — never hide
  failures"; this is exactly the "commit message asserts a state that was never
  verified" class).
- `sync_repo_root.py` is the ONLY sanctioned repo-root recovery — do not hand-roll a
  pull-rebase loop in the pre-sync step (`guard_repo_root_pull.sh` blocks it anyway).
- `sync_repo_root` exit 0 can mean "another sync in-flight"; the pre-sync must treat
  that as non-fatal and let the guard speak.
- `scripts/workflow_lint.py --check-references` / `--check-asks` pass; ruff passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 424d6bf46eee
- Source: `/daily` 2026-07-25 transcript sweep, sessions `ea7470c1` (#1683) @
  14:55:49Z–14:57:51Z and `6d798257` (#1688) @ 18:19:37Z.
