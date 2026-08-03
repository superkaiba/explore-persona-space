---
title: 'daily-fix: scratch-worktree cone + empty-dir residue fixes'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e0e2756cfae6
- daily-auto-filed
created_at: '2026-07-29T07:06:35Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): The Step 10d post-merge
  stale-task-folder guard''s scratch-worktree removal recipe sets the sparse cone
  to the duplicate task folders ONLY, so the pre-commit gitleaks hook (entry `bash
  scripts/hooks/gitleaks_scoped.sh`, path relative to the worktree root) exits 127
  (`No such file or directory`) and the removal commit fails on the first attempt;
  the guard reports staging FAILED and burns a retry. Ob'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step C parked-candidate sweep (2026-07-28) from a formal candidate block parked on task #1780 (ts 2026-07-29T03:11:33Z, fp e0e2756cfae6; source: orchestrator-own-observation, hit live during #1780's Step 10d).

## Goal

Fix two ad-hoc-rediscovered defects in the Step 10d post-merge stale-task-folder guard's scratch-worktree removal recipe: (a) the sparse cone omits `scripts/hooks`, so the removal commit's own pre-commit hook exits 127 and the first attempt always fails; (b) the local-residue check flags EMPTY leftover directories that root syncs can never clear (rmdir is the correct disposition).

## Workflow gap

- **Bug observed:** the recipe sets the sparse cone to the duplicate task folders ONLY, so the pre-commit gitleaks hook (`bash scripts/hooks/gitleaks_scoped.sh`, worktree-root-relative path) exits 127 (`No such file or directory`) and the removal commit fails on the first attempt; observed live on #1780 (2026-07-29 ~03:08Z), retry with `scripts/hooks` in the cone passed cleanly. Additionally the `ls -d` local-residue check flags empty leftover dirs (git removes tracked files but leaves empty dirs), which two root syncs can never clear.
- **Why it is a workflow gap:** every future guard firing re-discovers both fixes ad hoc — the documented recipe is wrong on both counts.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'sparse-checkout set "${DUPES[@]}"' .claude/skills/issue/SKILL.md` → 1 hit (line 13067, without `scripts/hooks`) (2026-07-29 UTC). Landed-fix history check: `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` → no commit touching the scratch-worktree cone recipe. unverified hypothesis — verify at plan time: that the pre-commit config invokes `scripts/hooks/gitleaks_scoped.sh` by worktree-root-relative path (read `.pre-commit-config.yaml` at plan time; recalled from the live #1780 failure).

## Proposed change (candidate diff sketch — refine in planning)

```diff
-       && git -C "$SCRATCH" sparse-checkout set "${DUPES[@]}" \
+       && git -C "$SCRATCH" sparse-checkout set "${DUPES[@]}" scripts/hooks \
```

And extend the local-residue tail: when `ls -d` still matches after 2 syncs, probe `find <dup> -type f | wc -l`; 0 files → `find <dup> -depth -type d -exec rmdir {} \;` (empty-dir residue, inert to git) instead of failing loud.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (the Step 10d scratch-worktree removal block)
- Check whether any other scratch-worktree recipe in the workflow surface shares the cone omission: `grep -rn 'sparse-checkout set' .claude/ scripts/`.

## Constraints / invariants

- The guard's refuse-on-real-content behavior is unchanged — only the hook-cone and the empty-dir disposition change.
- Workflow-surface only; recursion guard applies to the spawned session.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: e0e2756cfae6

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: The Step 10d post-merge stale-task-folder guard's scratch-worktree removal recipe sets the sparse cone to the duplicate task folders ONLY, so the pre-commit gitleaks hook (entry `bash scripts/hooks/gitleaks_scoped.sh`, path relative to the worktree root) exits 127 (`No such file or directory`) and the removal commit fails on the first attempt; the guard reports staging FAILED and burns a retry. Observed live on task #1780 (2026-07-29 ~03:08Z); the retry with `scripts/hooks` added to the cone passed cleanly. Additionally, the guard's local-residue `ls -d` check flags EMPTY leftover directories (git removes tracked files on rebase/checkout but leaves empty dirs), which two root syncs can never clear — a file-count probe + rmdir is the correct disposition.
why_workflow_gap: The documented recipe's `sparse-checkout set "${DUPES[@]}"` line omits the hook-script dir the commit's own pre-commit hooks need, and the local-residue tail's remedy (root syncs) cannot remove untracked empty dirs, so every future guard firing re-discovers both fixes ad hoc.
proposed_change: In the SKILL.md scratch-worktree block, change the cone line to `git -C "$SCRATCH" sparse-checkout set "${DUPES[@]}" scripts/hooks` (one token), and extend the local-residue tail: when `ls -d` still matches after 2 syncs, probe `find <dup> -type f | wc -l`; 0 files → `find <dup> -depth -type d -exec rmdir {} \;` (empty-dir residue, inert to git) instead of failing loud.
confidence: high
related_task: #1780
<!-- /workflow-fix-candidate -->
