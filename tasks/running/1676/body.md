---
title: 'daily-fix: root-commit guard recognizes cd-into-worktree'
kind: infra
tags:
- wf-fix
- wf-fix-fp:19fbe8a5a2cf
- daily-auto-filed
- trigger-dense
created_at: '2026-07-25T06:49:49Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 2): guard_root_code_commit.sh
  blocked a compound ''cd $WT && pytest && git commit'' as a repo-root commit carrying
  uncertified code payload - the cd into the worktree was not recognized and the whole
  compound was killed including the read-only pytest leg'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 problem sweep (session a5f39829, task #1644).

## Goal

The root-commit certification guard should not misclassify a worktree-bound compound commit as a repo-root commit just because it uses `cd $WT && ...` instead of `git -C`.

## Workflow gap

- **Bug observed:** #1644's session ran `cd "$WT" && pytest ... && git commit ...` and the hook blocked the whole compound ("BLOCKED: repo-root commit carries UNCERTIFIED code payload: scripts/issue1586_figures.py | cert-diag: ... binding=worktree ... cert=none-for-path", 1 firing) — nothing ran, including the read-only pytest leg. Re-running with explicit `git -C "$WT"` forms passed.
- **Why it is a workflow gap:** the guard's cwd inference reads the SHELL's inherited cwd (repo root) and does not follow a leading `cd <worktree-path>`; the BLOCKED message's cert-diag did not name the cd-prefix cause, costing a diagnosis turn.
- **Confidence (emitter):** medium — cwd-inference in a hook is security-adjacent; the minimal safe fix may be message-only (name the cd-prefix cause + the `git -C` remediation), with the cd-following inference a deliberate planner decision.
- verified-at-filing: hook header (read 2026-07-25) documents the Layer-1/Layer-2 binding semantics and the `git -C "$WT" commit` remediation but no cd-prefix recognition; `git log --oneline --since='7 days ago' -- .claude/hooks/guard_root_code_commit.sh` → c082ca9567 (#1620 pathspec-scope) only.

## Proposed change (candidate diff sketch — refine in planning)

Either (a) resolve the effective cwd across a leading `cd <path>` when classifying the commit's binding, or (b) minimally, extend the BLOCKED message: when the compound starts with `cd` into `.claude/worktrees/`, say so and prescribe the `git -C "$WT"` form. Extend hook tests either way.

## Scope / surfaces

- Primary target: `.claude/hooks/guard_root_code_commit.sh`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 19fbe8a5a2cf

- workflow_fix_target: .claude/hooks/guard_root_code_commit.sh
