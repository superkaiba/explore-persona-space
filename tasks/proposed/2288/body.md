---
title: verify_task_body cross-issue-reuse-pins check SKIPs on pre-merge worktree bodies
  — its only gate
kind: infra
tags: []
created_at: '2026-08-14T09:59:28Z'
has_clean_result: false
parent_id: 2223
origin_prompt: 'clean-result-critic prose follow-up during the #2223 clean-result
  gate: the ''cross-issue reuse pins declared'' verifier check skipped with ''eval
  root unresolved'', but pre-merge worktree bodies are the only place this critic
  runs, so that skip leaves the footer reuse-bullet surface mechanically unchecked
  at its only gate.'
workflow: v1
---
# `verify_task_body.py` cross-issue-reuse-pins check skips at the only gate where it could fire

## Goal

Make `verify_task_body.py`'s "cross-issue reuse pins declared" check actually evaluate on a
pre-merge worktree body, so the clean-result footer's reuse-provenance surface is mechanically
checked at the one gate that reviews it.

## The gap

Surfaced by `clean-result-critic` while gating #2223 (`epm:clean-result-critique`,
2026-08-14T09:54:18Z). Its own words:

> the `cross-issue reuse pins declared` verifier check skipped with "eval root unresolved" — but
> pre-merge worktree bodies are the only place this critic runs, so that skip leaves the footer
> reuse-bullet surface mechanically unchecked at its only gate.

The structural problem: the check resolves an "eval root" that does not resolve for a body still on
its `issue-<N>` branch, and a `kind: experiment` clean-result is ALWAYS reviewed pre-merge (the
worktree rebase-merges to `main` only at the terminal step, after the critic gate). So the check is
SKIP-by-construction in the only situation it is asked to run. A SKIP reads as "nothing to see"
rather than "not evaluated", which is the silent-pass shape.

This was not hypothetical on #2223: the critic had to find the missing axis provenance BY HAND as a
Lens 5 blocker — both legs' load-bearing instrument (the #2203 in-house layer-14 axis staged from HF,
and Lu et al.'s published Qwen3-32B axis) had no fetchable pin in the body. A working mechanical check
is exactly what would have caught that without depending on a reviewer noticing.

## Two candidate fixes (the critic named both; pick after reading the check)

1. Resolve the eval root from the ISSUE BRANCH rather than assuming a merged `main` layout, so the
   check evaluates in the worktree case.
2. Failing that, DEGRADE to the eval-root-free half of the check rather than skipping outright:
   assert a `Reused:`-bullet is PRESENT whenever the body's Goal/Context cites a parent artifact.
   Weaker, but it fires.

Preference is (1) if the eval root is derivable from the branch; (2) is the fallback. Either way the
outcome must be that the check REPORTS rather than SKIPs on a pre-merge worktree body.

## Acceptance

- The check evaluates (PASS or FAIL, not SKIP) on a `kind: experiment` body sitting on its
  `issue-<N>` branch pre-merge.
- A body citing a parent-issue artifact with NO reuse-provenance bullet FAILs (or WARNs, matching the
  check's existing severity class — do not silently escalate a WARN to a FAIL and wedge the fleet
  Step-9c gate).
- A regression test pins the pre-merge-worktree case specifically, since that is the case that
  regressed to SKIP.
- Confirm the change does not flip existing green bodies red: run the check across current
  `has_clean_result=true` bodies and report the delta before landing.

## Provenance

Surfaced as a prose workflow-fix follow-up by `clean-result-critic` during the #2223 clean-result
gate; filed by the orchestrator per the workflow-fix-on-bug protocol (surfaced-prose suggestions get
the same auto-file treatment as a formal `workflow-fix-candidate` block). Not diagnosed further by
the filer — the check's code was not read before filing, so the two candidate fixes above are the
critic's framing, not a completed diagnosis.
