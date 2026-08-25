---
title: dispatch_issue.py launch treats an empty-string --repo-branch as unset, re-opening
  the clone-main hole
kind: infra
tags:
- workflow-fix
created_at: '2026-08-22T02:54:26Z'
has_clean_result: false
origin_prompt: 'Surfaced as non-blocking concern 3 by the round-3 delta critique of
  #2263 plan v3: dispatch_issue.py tests --repo-branch for truthiness (:2032, :1203),
  so an empty string reads as unset and can degrade a resolver refusal into a silent
  main materialization.'
workflow: v1
---
---
kind: infra
tags: [workflow-fix]
---

# `dispatch_issue.py launch` treats an empty-string `--repo-branch` as unset, silently re-opening the clone-main hole

## Goal

Make `dispatch_issue.py launch` REFUSE an empty-string `--repo-branch` at parse time, mirroring the gate-side validator, so a resolver refusal can never degrade into a silent `main` materialization.

## What's wrong

`scripts/dispatch_issue.py` tests the flag for truthiness rather than presence:

- `:2032` — `if getattr(args, "repo_branch", None):` — an empty string is falsy, so `--repo-branch ""` is indistinguishable from the flag being absent.
- `:1203` — `if extra.get("repo_branch"): return None` — the `_repo_branch_default_main_conflict` guard likewise treats `""` as "no explicit value", but only AFTER the truthiness test above has already dropped it.

So `--repo-branch ""` falls through to the cwd/worktree default (`_current_git_branch`, applied `:2064`). That default excludes `main` on both arms (`if branch and branch != "main"`, `:2065` / `:2082`), and the conflict guard covers most remaining cells — but in the genuinely-no-issue-shaped-refs case the launch proceeds and the lane materializes `main`.

The gate side already rejects this exact value: #2263's plan §4.4.3 adds a `--repo-branch` validator that refuses empty/whitespace at parse time. The two CLIs deliberately share the flag spelling so one value can be threaded to both, so they should share the value contract too.

## How it becomes reachable

#2263 lands a shared resolver whose refusal path prints EMPTY stdout and exits 2, with each fence guarded by `: "${REPO_BRANCH:?...}"`. In NON-interactive bash — the Bash-tool surface, and how the fences actually run — that guard aborts the shell (verified: rc=127, the following line never executes). In an INTERACTIVE shell, `${VAR:?}` does NOT terminate the shell: a human pasting the fence by hand continues to the next line with `REPO_BRANCH=""`, and the launch then reads it as unset.

This is fail-safe in most cells and the resulting `main` materialization matches what the sanctioned `--repo-branch main` remedy produces — the loss is the VISIBLE EDIT the #2263 design requires for a deliberately main-resident workload. Parse-time rejection restores it.

Severity: LOW and fail-safe. Filed because it is a concrete, cheap, mechanizable hardening on the workflow surface, and because leaving it as a chat note is the anti-pattern the workflow-fix protocol names.

## Fix

Add a parse-time validator on `launch`'s `--repo-branch` (subparser `:3175`) rejecting empty/whitespace-only values, with a message pointing at the two legitimate forms: a real branch name, or the explicit `main` confirmation.

The fix is SELF-CONTAINED — rejecting an empty string needs nothing from any other task. If #2263 has already landed, match the wording of its gate-side `--repo-branch` validator in `scripts/verify_carryover_inputs.py` so the two CLIs read consistently; if it has NOT landed yet, write the message on its own terms and do not block or wait on it. Do not edit that file either way.

## Sequencing

#2263 is IN FLIGHT as of filing (status `running`). It touches `scripts/verify_carryover_inputs.py`, `scripts/issue1995_corpus_sweep.py`, `tests/test_verify_carryover_inputs.py`, and `.claude/skills/issue/steps/10-step-6.md`. This task touches `scripts/dispatch_issue.py` and its test file ONLY — a disjoint file set, so the two can proceed concurrently with no writer arbitration needed. Do not modify #2263's files.

Pin it with a CLI test asserting `--repo-branch ""` exits 2 (argparse error), alongside a positive arm confirming `--repo-branch main` still resolves as the deliberate confirmation the conflict guard's own docstring sanctions ("an explicit value, INCLUDING `main`, bypasses by construction").

## Out of scope

- The truthiness-vs-presence pattern anywhere else in `dispatch_issue.py` — this task covers `--repo-branch` only.
- Any change to `_repo_branch_default_main_conflict`'s semantics. An explicit `main` remains a legal deliberate confirmation; only the EMPTY value is refused.
- Anything in `scripts/verify_carryover_inputs.py` or `.claude/skills/issue/steps/10-step-6.md` — #2263 owns those and is in flight.

## Provenance

Surfaced as non-blocking concern 3 by the round-3 delta critique of #2263's plan v3 (2026-08-21), while reviewing the shared-resolver fence design. Behavior confirmed by direct probe of interactive-vs-non-interactive `${VAR:?}` semantics plus reads of `dispatch_issue.py:2032`, `:1203`, `:2064-2082`, `:3175`.

Dedup fingerprint: (`scripts/dispatch_issue.py`, empty-string `--repo-branch` accepted as unset). Distinct from #2263's fingerprint (`scripts/verify_carryover_inputs.py`, gate-side check-ref resolution) — different file, different defect.
