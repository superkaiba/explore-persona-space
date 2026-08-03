---
title: 'daily-fix: widen piped-git guard to hook-running git commit'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7fda33e208ac
- daily-auto-filed
created_at: '2026-07-22T06:45:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-21 problem sweep (route 2): piping a hook-running git
  commit through head SIGPIPE-kills the commit mid-hook; the guard covers push/merge/pr
  only'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily parked-candidate routing pass (Step C) from a recursion-guard-parked prose follow-up on task #1584 (emitting agent: implementer, #1584 round-1 (d) section).

## Goal

Extend the piped-git guard surface to cover a hook-running `git commit` piped through `head`/`tail`/`grep` — the SIGPIPE-kills-the-commit-mid-hook sibling of the #1048 piped-push class.

## Workflow gap

- **Bug observed:** piping a hook-running `git commit` through `head` (`git commit 2>&1 | head -N`) SIGPIPE-kills the commit mid-hook — demonstrated live during the #1584 round-1 smoke (the gitleaks hook was mid-scan when the pipe closed).
- **Why it is a workflow gap:** the existing PreToolUse guard blocks piped `git push` / `git merge` / `gh pr` only; a piped `git commit` silently dies mid-hook with the same masked-exit-code consequence class as #1048.
- **Confidence (emitter):** medium
- verified-at-filing: original candidate named `scripts/guard_piped_git_push.sh` — 0 hits at that path (mis-target); repo-wide relocation grep found the real guard at `.claude/hooks/guard_piped_git_push.sh` (referenced by `.claude/settings.json`), target corrected per workflow-fix-on-bug.md clause (a). `grep -nE 'commit' .claude/hooks/guard_piped_git_push.sh` → hits only in comments (the push-recipe preamble; the guard's block set is push/merge/gh-pr) — the absence claim binds semantically (2026-07-22). `git log --oneline --since='7 days ago' -- .claude/hooks/guard_piped_git_push.sh` → no commit-widening landed.

## Proposed change (candidate diff sketch — refine in planning)

Widen `.claude/hooks/guard_piped_git_push.sh` (and/or `workflow_lint.py --check-piped-git-push` + `.claude/rules/gotchas.md`) to also block a hook-running `git commit` piped through `head`/`tail`/`grep`; keep the documented known-miss/`--dry-run` carve-outs consistent.

## Scope / surfaces

- Primary target: `.claude/hooks/guard_piped_git_push.sh` (corrected from the candidate's `scripts/guard_piped_git_push.sh`)
- Siblings to keep consistent: `scripts/workflow_lint.py` (`--check-piped-git-push`), `tests/test_guard_piped_git_push.py`, `.claude/rules/gotchas.md`.

## Constraints / invariants

- Workflow-surface only. `tests/test_guard_piped_git_push.py` extended + green; existing known-miss pins preserved; `bash -n` on the hook passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.
- Trigger-dense target (a guard/security hook script): briefs and review rounds follow `.claude/rules/trigger-dense-review.md` (reference guard content by path + abstract class; findings by marker/file reference).

## Provenance

- fingerprint: 7fda33e208ac

- workflow_fix_target: .claude/hooks/guard_piped_git_push.sh

Verbatim parked candidate (task #1584 events, 2026-07-21T08:19:25Z): "parked — running under workflow_fix_target Provenance (recursion guard, workflow-fix-on-bug.md § Recursion guard); NOT auto-routed. source: prose-followup (implementer round-1 (d) section). Candidate: piping a hook-running git commit through head (git commit 2>&1 | head -N) SIGPIPE-kills the commit mid-hook — the piped-git-push sibling class (#1048); target_file: scripts/guard_piped_git_push.sh (widen to piped git commit) or .claude/rules/gotchas.md; demonstrated live during the #1584 round-1 smoke."
