---
title: 'Step 10d: payload-aware no-PR arm (a branch with novel payload and zero PRs
  silently skips the auto-merge)'
kind: infra
tags:
- step10d-no-pr-arm
created_at: '2026-08-12T01:21:32Z'
has_clean_result: false
origin_prompt: /issue 2235
workflow: v1
---
## Goal

Close a silent payload-stranding gap in `/issue` Step 10d: when a branch has **no PR object at all**, the auto-merge takes an unconditional skip arm that never consults the novel-payload predicate sitting a few lines below it — so a code-bearing branch is left permanently unmerged with the durable record reading clean.

## The gap (hit live on #2235, 2026-08-11)

Step 10d's safe-case procedure opens with the PR-object liveness probe (`.claude/skills/issue/SKILL.md:12698-12704`):

```bash
PR=$(echo "$PR_INFO" | cut -d' ' -f1)
...
if [ -z "$PR" ]; then
  echo "No PR for issue-<N>; nothing to merge."   # skip; post nothing
```

The `-z "$PR"` arm is **unconditional**: it prints "nothing to merge" and skips, posting nothing. It never asks whether the branch actually carries payload.

Contrast the very next arm (`PR_STATE != OPEN`, lines 12706-12751), added by #1897 for exactly this hazard: when a *terminal* PR exists, the step runs the layered novel-payload predicate (commits-ahead → `git cherry` patch-ids → content-identity vs `origin/main`) and, on `NOVEL_PAYLOAD=yes`, **creates a fresh draft PR and merges it**. So the workflow already knows that "no usable PR object" plus "novel payload" must resolve to *create-then-merge*, not *skip* — it just does not apply that knowledge when the PR count is zero rather than one-and-terminal.

**Observed on #2235.** `gh pr list --head issue-2235 --state all` returned `[]` — no PR had ever been created (the draft PR that normally lands at the `approved`→`running` transition never happened; that upstream miss is the second half of this bug). The branch was 15 commits ahead with 14 `git cherry` `+` lines, carrying `scripts/workflow_lint.py`, `.claude/skills/issue/SKILL.md`, `scripts/inline_lint_gate.py` and three test files. Left to the skip arm, all six deliverables — including two SHARED workflow-surface files — would have stranded on an unmerged branch while the task transitioned to `completed`.

**Why that is the expensive class.** This is the #456 → #466 shape Step 10d was built to prevent, and its own rationale says so (`SKILL.md:11005-11010`): a shared `trainer.py` fix stranded on a deferred branch, and the next experiment inheriting from `main` lacked it and crashed. A stranded `workflow_lint.py` is worse than a stranded experiment helper, because the no-flags lint IS every session's Step 9c instrument — the #1388 fleet-wide coupling.

Note the existing detectors do NOT cover it: the watcher's `completed_unmerged_pass` (#1540/#1653) keys on a `completed` task with no `epm:merged` marker, but the skip arm posts **no marker at all** and #1723 routes the `completed` flip through Step 10d's terminal-teardown — so on the skip arm the task may never reach `completed` via that path, and the pass has nothing to latch onto. Silence on both sides.

## Fix

Two independent changes; (1) is the load-bearing one.

**(1) Make the `-z "$PR"` arm payload-aware — reuse the #1897 predicate verbatim.** Hoist the layered novel-payload predicate above the PR-state branch and gate BOTH no-usable-PR arms on it:

- `NOVEL_PAYLOAD=no`  → keep today's skip ("nothing to merge"), post nothing.
- `NOVEL_PAYLOAD=yes` → take the #1897 fresh-PR path (`gh pr create --draft`, re-resolve `PR_INFO`, then guards + merge). The draft must be marked ready (`gh pr ready`) or `gh pr merge` refuses — see (3).

The predicate is already written, already fail-SAFE toward "novel" on every git-error path, and already trusted for the terminal-PR case; this is a reuse, not a new mechanism.

**(2) Fail loud when the upstream draft-PR creation never happened.** The state machine (`SKILL.md` § The State Machine) shows `approved |-- (worktree + draft PR) |--> running`, but nothing verifies the PR exists afterward. Add a cheap assertion at the first step that needs it (Step 5's round-push, or Step 10d entry): a branch at/past `running` with commits ahead of `origin/main` and zero PRs is an anomaly worth one loud line, since it means an earlier transition silently no-opped.

**(3) Document the draft-PR merge precondition.** A `--draft` PR is not mergeable; `gh pr ready` must run first. The #1897 fresh-PR arm creates a draft and proceeds straight to the merge block with no `gh pr ready` between them, so that arm is latently broken for the same reason — worth fixing in the same pass.

## Acceptance

- A branch with novel payload and ZERO PRs gets a PR created and merged by Step 10d (no silent skip).
- A branch with NO novel payload and zero PRs still skips, posting nothing (today's behaviour preserved).
- The #1897 fresh-PR arm marks the PR ready before merging.
- A test pins the payload-aware routing of the `-z "$PR"` arm.

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
problem: Step 10d's `if [ -z "$PR" ]` arm (SKILL.md:12703) skips the auto-merge unconditionally when no PR object exists, without consulting the layered novel-payload predicate used a few lines later for terminal PRs (#1897). A code-bearing branch whose draft PR was never created is therefore left permanently unmerged with no marker posted — the #456/#466 stranded-shared-module class. Hit live on #2235: zero PRs, 15 commits ahead, 14 cherry '+' lines, payload including scripts/workflow_lint.py and .claude/skills/issue/SKILL.md. The watcher's completed_unmerged_pass cannot catch it because the skip arm posts no marker.
fix: hoist the novel-payload predicate above the PR-state branch and gate the -z "$PR" arm on it — NOVEL_PAYLOAD=no keeps today's skip, NOVEL_PAYLOAD=yes takes the existing #1897 fresh-PR create-then-merge path; add `gh pr ready` before the merge on both fresh-PR arms (a draft PR is not mergeable); add a loud assertion when a branch at/past running has commits ahead and zero PRs.
urgency: normal
wf_fix: true
confidence: high
related_task: #2235
<!-- /workflow-fix-candidate -->

## Provenance

Hit during #2235's Step 10d auto-merge on 2026-08-11. Not a speculative read of the recipe: the PR probe returned empty, `gh pr list --head issue-2235 --state all` returned `[]`, and the novel-payload predicate returned 15 commits ahead / 14 cherry `+` lines. The merge was completed by creating the missing PR manually (#1912); this task makes that recovery automatic.
