---
title: 'workflow-fix: guard repo-root working-tree reverts (restore/checkout ./clean
  -f)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b9c2b2c3249d
created_at: '2026-07-02T22:52:14Z'
has_clean_result: false
origin_prompt: 'Prose follow-up from #841 analyzer: concurrent destructive git op
  on shared repo root reverted uncommitted body.md edits mid-task; hook guards only
  branch ops, lint only reset --hard — extend both to working-tree reverts (git restore
  . / git checkout . / git clean -f / runtime reset --hard).'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced on task #841 (emitting agent: analyzer, clean-result revise round).

## Goal

Extend the repo-root destructive-git guard (PreToolUse hook + lint) to cover working-tree reverts — `git restore .` / `git restore <path>`, pathspec `git checkout .` / `git checkout -- <path>`, `git clean -f`, and unqualified `git reset --hard` — not only branch switches (hook) and doc-mentioned `reset --hard` (lint).

## Workflow gap

- **Bug observed:** during #841's clean-result revise round (2026-07-02 ~22:42Z), a concurrent session's destructive working-tree git op on the shared repo root reverted the analyzer's uncommitted `body.md` edits mid-task — the file snapped back to HEAD. The analyzer recovered by re-applying via `task.py set-body` (atomic commit). Same hazard class as the #778 repo-root `reset --hard` that clobbered #812/#813 task state.
- **Why it is a workflow gap:** CLAUDE.md bans destructive `git reset --hard` / `git clean -f` / `git checkout .` / `git restore .` on the repo root and names `guard_repo_root_branch.sh` as enforcement, but the hook only gates BRANCH ops (`grep -qE '\bgit\b.*\b(checkout|switch)\b'` at scripts/guard_repo_root_branch.sh:102, blocking `-b/-B/--detach/<branch>` forms) — a pathspec `git checkout .`, `git restore .`, `git clean -f`, and runtime `git reset --hard` all pass the hook. The lint (`workflow_lint.py --check-no-repo-root-git-reset-hard`, `_GIT_RESET_HARD_RE`) covers only `reset --hard` and only in agent/skill DOC text, not runtime. So the prose rule has no runtime teeth for the working-tree-revert class that just caused a live edit loss.
- **Confidence (emitter):** high (live incident on #841; hook + lint coverage verified by grep).

## Proposed change (candidate diff sketch — refine in planning)

- `scripts/guard_repo_root_branch.sh`: add a second guard clause for destructive working-tree ops when the effective cwd is the repo root (not `git -C <worktree>`-qualified):
  + block `git restore` with a pathspec unless `--staged`-only-with-explicit-path… (planner decides exact policy; minimum: block bare `git restore .` / `git checkout .` / `git checkout -- <path>` / `git clean -f*` / `git reset --hard` at repo root)
  + preserve the existing `-C`-qualifier waiver and an explicit same-line allow-sentinel for deliberate uses.
- `scripts/workflow_lint.py`: extend `_GIT_RESET_HARD_RE` coverage (or a sibling check) so workflow docs prescribing `git restore .` / `git checkout .` / `git clean -f` on the shared root FAIL the same way `reset --hard` does.
- Tests: hook-behavior cases (blocked forms, `-C` waiver, sentinel) + lint regression cases.

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh, scripts/workflow_lint.py`
- Grep the workflow surface for prescriptive uses before editing (`grep -rnE 'git (restore|checkout) \.|git clean -f' .claude/ CLAUDE.md scripts/`) and whitelist/fix every legitimate hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only. Existing hook tests + `workflow_lint` default run stay green; per-worktree `git -C "$WT"` destructive ops remain allowed.
- Fail-closed parsing consistent with the hook's #804 latch semantics.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/guard_repo_root_branch.sh, scripts/workflow_lint.py
- fingerprint: b9c2b2c3249d

Surfaced prose (verbatim, from the #841 analyzer's revise-round report): "my initial in-place Edits to body.md were reverted mid-task by a concurrent destructive git op on the shared repo root — body.md snapped back to HEAD (git-clean, 3 pre_reg, 4 images) around 22:42Z. This is the #778/#812/#813-class hazard. I recovered by applying via set-body (atomic commit, durable). The existing lint (workflow_lint --check-no-repo-root-git-reset-hard) covers `reset --hard`, but a `git restore .`/`git checkout .` working-tree revert may not be guarded — worth a look if you see other sessions losing uncommitted edits."
