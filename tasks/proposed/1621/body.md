---
title: 'daily-fix: branch-guard false pos: no-checkout + heredoc'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6e895a26ec36
- daily-auto-filed
created_at: '2026-07-23T07:00:12Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): two false-positive blocks
  2026-07-22: the sanctioned git worktree add --no-checkout --detach recipe matched
  by the checkout-detach regex, and heredoc PAYLOAD text quoting git checkout -b matched
  as a real command'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-22 (transcript sweep). `scripts/guard_repo_root_branch.sh` produced two distinct FALSE-POSITIVE blocks today, each on a sanctioned/harmless command shape.

## Goal

The repo-root branch guard stops blocking (a) the CLAUDE.md-documented scratch-worktree recipe `git worktree add --no-checkout --detach <path> origin/main`, and (b) commands whose heredoc PAYLOAD text merely quotes git commands (`git checkout -b` as data inside a plan-edit heredoc).

## Workflow gap

- **Bug observed:** (a) 552fa84d (#1092 inline round, 2026-07-23T03:53:27Z): `git worktree add --no-checkout --detach /tmp/wt1092dash origin/main` BLOCKED as "'git checkout --detach' would move the SHARED repo-root tree" — the regex at guard_repo_root_branch.sh:1364 (`\bgit\b[^;&|]*\bcheckout\b +(-{1,2})detach\b`) matches the substring `checkout --detach` inside `--no-checkout --detach` (`\b` matches after the hyphen). (b) abee1289 (#1602, 17:39:22Z): a `uv run python - <<'PYEOF'` plan-edit whose heredoc payload contained the literal string `git checkout -b` (the plan documents git recipes) was BLOCKED; the session had to restructure via the Write tool.
- **Why it is a workflow gap:** both blocks hit sanctioned workflows (the documented worktree recipe; editing a plan that quotes git commands — routine for #1602-class tasks whose SUBJECT is git behavior).
- **Confidence:** high.
- verified-at-filing: `sed -n '1364p' scripts/guard_repo_root_branch.sh` → the checkout-detach regex verbatim as quoted above (presence, binds); `grep -c 'heredoc' scripts/guard_repo_root_branch.sh` → 19 hits — heredoc handling EXISTS elsewhere in the guard but did not protect this clause on 2026-07-22 (context-consistency note for the planner: extend/route the existing heredoc stripping to the checkout-detach + checkout -b clauses rather than adding a parallel mechanism), 2026-07-23 UTC.

## Proposed change (refine in planning)

1. Exempt `git worktree add` command shapes from the checkout-detach clause (or exclude a `no-`-prefixed match).
2. Strip heredoc bodies (text between `<<'X'`/`<<X` and the delimiter) before pattern-matching the branch/checkout clauses — reusing the guard's existing heredoc machinery where possible.
3. Add regression cases for both shapes to the guard's tests.

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh` (+ tests).

## Constraints / invariants

- Real branch-moving commands at the repo root stay blocked (`git checkout -b`, `git switch`, bare `git checkout --detach` on the root tree).
- Recursion guard applies (workflow_fix_target Provenance).

## Provenance

- fingerprint: 6e895a26ec36

- workflow_fix_target: scripts/guard_repo_root_branch.sh
