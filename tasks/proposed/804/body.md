---
title: 'workflow-fix: guard clause-local parsing (compound + glued -b leaks)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:796-r3-compound-glued-followup
- wf-fix-followup-of-796
created_at: '2026-07-01T08:24:45Z'
has_clean_result: false
origin_prompt: 'Follow-up to #796 r3: fold compound-command-masking-leak + checkout-glued-shortflag-b-leak
  into a clause-local-parsing (or conservative-block) design change for scripts/guard_repo_root_branch.sh.'
---
## Overview / Motivation

Follow-up to #796. During #796's Codex+Claude ensemble code-review rounds, TWO
substantive CONCERN-severity leaks in `scripts/guard_repo_root_branch.sh` were
identified as **pre-existing** design-level limitations, out-of-scope for #796's
detach-guard fix (per the round-3 reconciler ruling on 2026-07-01). Both are
persisted on #796's `concerns.jsonl` but marked as follow-up material.

## Goal

Fix two remaining policy-bypass classes in `scripts/guard_repo_root_branch.sh`
by moving from regex-on-whole-command to clause-local parsing (or equivalent):

1. **Compound-command masking (Codex #796 r3):** whole-command scanning lets a
   later safe/scoped clause mask an earlier dangerous repo-root clause.
   Empirical leaks (from #796 worktree):
   - `git switch feature ; git switch main` → exit 0 (want 2)
   - `git switch feature && git switch main` → exit 0 (want 2)
   - `git checkout HEAD~1 ; git checkout main` → exit 0 (want 2)
   - `git switch feature ; cd .claude/worktrees/x` → exit 0 (want 2)

2. **Glued-shortflag branch creation (Claude #796 r3):** `(-b|-B)\b`
   word-boundary doesn't match `-bfoo`. Empirical leak:
   - `git checkout -bfoo` → exit 0 (want 2) — creates a branch + moves HEAD
     off main (the #459 incident class).

Both classes exist byte-identical on pre-#796 `main` — verified via
`git show main:scripts/guard_repo_root_branch.sh` at #796 r3.

## Workflow gap

- **Bug observed:** the fleet-wide PreToolUse Bash hook silently allows
  branch-switch / detach operations on the shared repo-root tree when the
  command uses compound-shell shape (`;`, `&&`, `||`) OR when `-b`/`-B` is
  glued to its branch name. Both bypass the same "grep on `$cmd`" approach
  used throughout the script.
- **Evidence:** #796's `concerns.jsonl` carries both under
  `compound-command-masking-leak` and `checkout-glued-shortflag-b-leak`.
  Empirical reproducers verified on repo-root `main` at #796 r3 (2026-07-01).
- **Why workflow gap:** the entire hook design (regex on `$cmd`) can't
  correctly enforce shell clause semantics. Adding regex tweaks doesn't
  scale — each regex has an anchor edge case. The fix requires either
  (a) parse `$cmd` into shell clauses (`;` / `&&` / `||`), run the existing
  detectors independently on each clause; OR (b) tighten to a MORE
  restrictive whitelist that fails-closed instead of fails-open on any
  compound/glued shape not obviously in the safe set.

## Scope / surfaces

- **Primary target:** `scripts/guard_repo_root_branch.sh` (the same file #796
  touched; adds clause-local parsing OR conservative-block logic).
- **Test file:** extend `tests/test_guard_repo_root_branch.py` with the six
  regression cases the two concerns list (four compound + two glued).
- Grep the workflow surface for the pattern before editing.

## Constraints / invariants

- Workflow-surface fix per the workflow-fix-on-bug protocol
  (`.claude/rules/workflow-fix-on-bug.md`).
- MUST preserve #796's fixes: quoted-ref parsing (r2 fix), main-prefix
  anchor (r3 fix), commit-ish detach classifier (r1 fix). The follow-up
  is ADDITIVE — it does not undo prior fixes.
- MUST NOT touch the two must-preserve escapes: worktree/`cd` scoping and
  already-off-main recovery.
- keep ruff + `bash -n` + `workflow_lint.py` clean.
- 0 GPU-h.

## Provenance

- workflow_fix_target: scripts/guard_repo_root_branch.sh
- source: #796 round-3 reconciler verdict (2026-07-01)
- concerns_folded_in:
  - #796 concerns.jsonl `compound-command-masking-leak` (Codex r3)
  - #796 concerns.jsonl `checkout-glued-shortflag-b-leak` (Claude r3)

## Notes

The planner should evaluate Option A (clause-local parsing) vs Option B
(conservative-block on compound/glued shapes) against the fleet's usage
patterns and pick the simpler + safer default. Clause-local parsing is
more precise but adds shell-parsing complexity to a bash hook.
Conservative-block is a smaller diff but may false-positive on
legitimate compound commands the fleet uses. The planner may also
propose Option C (extract each `git checkout`/`git switch` clause via
sed and independently classify each).
