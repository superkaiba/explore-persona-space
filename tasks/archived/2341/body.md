---
title: 'workflow-fix: guard_root_code_commit.sh pathspec scoping does not engage (non-code
  commits wedged by foreign staged code files)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-17T09:48:11Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate from codex-code-reviewer composer, task #2155
  r2 compose (2026-08-17); orchestrator reproduced live same day'
workflow: v1
---
# workflow-fix: guard_root_code_commit.sh pathspec scoping does not engage — non-code commits wedged by any foreign staged code file

## Provenance

workflow_fix_target: .claude/hooks/guard_root_code_commit.sh
Surfaced by the codex-code-reviewer prompt-composer during task #2155 code-review round 2 (2026-08-17); routed by the #2155 orchestrator per `.claude/rules/workflow-fix-on-bug.md` (candidate fingerprint: pathspec-scoping-not-engaging × foreign-staged-file).

## Goal

A pathspec-limited repo-root commit of a path OUTSIDE the guard's code-cert scope (e.g. `.claude/agent-memory/**.md`) must not be blocked by an unrelated FOREIGN staged code file. Fix the guard so pathspec scoping engages before (or instead of) the whole-staged-index certification sweep, and add a regression test.

## Bug (verbatim from the surfacing agent)

The guard's pathspec scoping does not engage — a pathspec-limited repo-root commit of a single NON-code path (`.claude/agent-memory/**.md`) was blocked on an unrelated FOREIGN staged code file (`tests/test_issue2094_rev_butler.py`, cert stale 215405s > 21600s), including when using the guard's own remediation form verbatim (`cd <root> && git commit -m "<msg>" -- <path>`) and a single-token-message variant. Either the pathspec parser fails on these shapes or the staged-index sweep runs before pathspec scoping. Effect: any session is wedged out of committing its own non-code paths whenever any concurrent session leaves an uncertified code file staged. This contradicts the guard's own text ("a pathspec-limited commit is never blocked by foreign staged files").

## Repro

Stage a code file with a stale cert; then `cd <root> && git commit -m msg -- .claude/agent-memory/<any>.md` → BLOCKED naming the foreign file. Three command shapes tried 2026-08-17 (~09:30–09:50Z), all blocked.

## Stranded payload to land with (or after) the fix

`.claude/agent-memory/codex-code-reviewer/feedback_revision_round_compose_recipe.md` is WRITTEN + STAGED at the repo root (a verified lesson append from the #2155 r2 compose: the r1 verdict template carried the retired free-form grammar instead of the current `CONCERN::` machine-row grammar). Once the guard is fixed (or the foreign file's owner lands it), commit this file by explicit path. Two additional untracked memory files in the same dir belong to other sessions — do not sweep them.

## Acceptance criteria

1. A pathspec-limited commit whose paths are all outside the guard's code-cert scope passes regardless of foreign staged entries.
2. A pathspec-limited commit that INCLUDES an uncertified code path is still blocked (no weakening).
3. Regression test covering both arms (fixture repo with a foreign staged uncertified code file).
4. The stranded memory file above lands by explicit path.

Estimated GPU-hours (total): 0
