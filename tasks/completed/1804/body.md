---
title: 'daily-fix: code-reviewer verdict --version collides on long '
kind: infra
tags:
- wf-fix
- wf-fix-fp:de986535baeb
- daily-auto-filed
created_at: '2026-07-29T07:14:10Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): #1092''s round-3 code-review
  verdict was posted at --version 3 on a marker stream already at v21 (long-lived
  same-issue-follow-up task) — the verdict lands below the existing max and is shadowed
  by the highest-version-per-kind resume convention; the orchestrator had to re-post
  it as v22'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Source: group-B P3 (miner-probed; re-verified at filing).

## Goal

Stop code-review verdict version collisions on long-lived follow-up tasks by letting post-marker auto-increment instead of pinning --version to the revision round.

## Workflow gap

- **Bug observed:** #1092 (a task with 20+ prior code-review markers from earlier follow-up rounds): the round-3 reviewer posted its verdict per the spec's `--version <revision_round>` recipe → version 3 on a stream whose max was 21. Under the highest-version-per-kind resume convention the verdict is shadowed; the orchestrator noticed and re-posted v22 (~2 min this time; silent-shadow risk every long-lived task).
- **Why it is a workflow gap:** the agent spec hardcodes `--version <revision_round>` (code-reviewer.md:37) and its read-back asserts `"version": <revision_round>` (line 49) — correct on fresh tasks, wrong whenever prior rounds pushed the stream past the current revision_round.
- **Confidence (emitter):** high (probed)
- verified-at-filing: `grep -n 'version' .claude/agents/code-reviewer.md` → line 37 `--version <revision_round>`, line 49 read-back confirms that version; codex twin posts `--version` at ~191/283/869 (2026-07-29 UTC).

## Proposed change (candidate diff sketch — refine in planning)

Change both specs: post without --version, carry `revision_round:` in the body, read back `version == latest+1` (the auto-increment result). Check the reconciler + drain-side consumers key on kind, not version==round.

## Scope / surfaces

- Primary targets: `.claude/agents/code-reviewer.md`, `.claude/agents/codex-code-reviewer.md` (posting + read-back recipes)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: de986535baeb

- workflow_fix_target: .claude/agents/code-reviewer.md

