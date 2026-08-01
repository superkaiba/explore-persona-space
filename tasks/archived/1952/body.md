---
title: 'workflow-fix: root-commit guard cwd attribution for cross-call-assigned cd-variable
  targets'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fe66c0343ffa
created_at: '2026-07-31T23:44:18Z'
has_clean_result: false
origin_prompt: 'Prose follow-up from #1941 implementer r1: guard_root_code_commit.sh
  blocked a cd "$WT" && git commit compound (variable assigned in a prior Bash call
  -> unproven -> fail-closed); git -C form passed. Recognize the effective working
  tree before selecting which tree''s cert to enforce.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #1941 (emitting agent: implementer, round 1).

## Goal

Attribute the effective tree via the hook input cwd (or session-persistent cwd) before the fail-closed root attribution of an unresolvable cd-variable target, keeping fail-closed when cwd is unavailable.

## Workflow gap

- **Bug observed:** a `cd "$WT" && git commit -m ... -- <paths>` compound run from an issue-worktree context was BLOCKED by `guard_root_code_commit.sh` as a repo-root commit when `$WT` had been assigned in a PRIOR Bash call — the #1676 variable-resolution arm covers only provable SAME-command assignment, so a cross-call-assigned variable target classifies `cd_verdict=unproven` and fails closed to root attribution. The `git -C "$WT"` form passed (the lead-anchored `-C` waiver is path-blind by documented design). One bounced commit attempt on #1941 r1.
- **Why it is a workflow gap:** the Bash tool's working directory and shell variables routinely persist conceptually across a session's calls (implementers bind `WT=...` once, reuse it), so the common compound shape degrades to a false-positive block + one wasted round-trip per occurrence, and the asymmetry (`-C "$WT"` waives while `cd "$WT" &&` blocks) is surprising at the point of use.
- **Confidence (emitter):** low — the fail-closed classification of unproven cd targets is a DELIBERATE #1676 design decision, and the hook's own error text already recommends the working `-C` remediation. The spawned session's planner may legitimately deflect with a reasoned no-change report (e.g. document the asymmetry more loudly in the block message instead).
- verified-at-filing: `grep -n 'cd \|-C \|worktree\|REPO' .claude/hooks/guard_root_code_commit.sh` → cd-variable arm present (`CD_VAR_TGT_ERE` ~L141, "the only unproven-target family eligible for the provable same-command-assignment resolution arm (resolve_cd_var)"), `cd_latch_verdict` ~L336-346 (`unproven` = "relative/variable/empty: unproven", fail closed); `git log --oneline --since='7 days ago' -- .claude/hooks/guard_root_code_commit.sh` → 3 commits incl. `7aeabf972a` "task #1676: cd-latch variable resolution + unproven-cd diagnostics" (2026-07-28) — context read per clause (c): the landed #1676 fix implements SAME-command-assignment resolution only; the gap filed here (cross-call assignment / persistent cwd) is a distinct residual the landed fix deliberately does not cover (2026-07-31).

## Proposed change (candidate diff sketch — refine in planning)

unverified hypothesis — verify at plan time: the Claude Code PreToolUse hook input JSON carries a `cwd` field for the Bash tool that reflects the session's persistent working directory and is usable for tree attribution.

```
  # in the cd-variable unproven branch (cd_latch_verdict caller):
+ # NEW: before failing closed, consult the hook-input cwd (if present):
+ #   cwd under .claude/worktrees/*  -> latch (a worktree IS its own tree)
+ #   cwd == $REPO or a subdir       -> root (unchanged behavior)
+ #   cwd absent/unreadable          -> unproven (fail-closed, as today)
+ # NOTE: the cd TARGET may differ from the inherited cwd; only use cwd
+ # when the unresolved variable target leaves no better evidence, and
+ # keep the diagnostics line naming which evidence attributed the tree.
```

## Scope / surfaces

- Primary target: `.claude/hooks/guard_root_code_commit.sh`
- Grep the workflow surface for the pattern before editing (`grep -rln 'cd_latch_verdict\|CD_VAR_TGT_ERE' .claude/ scripts/ tests/`) and update every hit; list them in the plan (the #1676 pin tests under `tests/test_guard_*.py` are the likely co-edits).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Fail-closed must be PRESERVED when no trustworthy cwd evidence exists; a strictly-loosening change needs the #1676 pin tests extended, never weakened.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/hooks/guard_root_code_commit.sh
- fingerprint: fe66c0343ffa

Surfaced prose (verbatim, implementer #1941 r1): "`scripts/guard_root_code_commit.sh` treated a `cd \"$WT\" && git commit …` compound as a repo-root commit and blocked it (one bounced attempt; the `git -C \"$WT\"` form passed). Concrete change: recognize a leading `cd <path> &&` prefix (or resolve the effective working tree) before selecting which tree's cert to enforce — the hook's own error text already recommends the `-C` form, so this is a friction fix, not a contract change." (Filer note: the hook lives at `.claude/hooks/guard_root_code_commit.sh`; the `scripts/` spelling in the prose is the implementer's shorthand. The leading-`cd` recognition ALREADY landed in #1676 for same-command-assigned variables and literal paths; the residual filed here is the cross-call-assigned variable shape.)
