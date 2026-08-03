---
title: 'daily-fix: pod-side descope posts marker + rewrites handle'
kind: infra
tags:
- wf-fix
- wf-fix-fp:10bcfb7be4cb
- daily-auto-filed
created_at: '2026-07-28T07:01:19Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): #1689''s r15b relaunch
  was an undocumented descope (L19-only, 10/2 draws vs the R15-marker-documented 200/40)
  launched from an uncommitted pod-side script with NO marker, and .claude/cache/issue-1689-handle.json
  kept pointing at the OLD run''s completion sentinel — a poller keyed on the handle
  would never fire; both found only because Thomas asked for a deep dive'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Session ca5c6f64 (#1689 audit), 2026-07-28T00:29Z (miner I P2; evidence is the session's own verified pod audit, epm:progress v64).

## Goal

A pod-side relaunch that changes the recipe or output root must leave a durable record and keep the dispatch handle live.

## Workflow gap

- **Bug observed:** the live launcher `/workspace/launch_issue_1689_r15b.sh` (pod-side, uncommitted) ran L19-only at 10/2 draws vs the documented 200/40, with no marker recording the descope — a clean-result contamination risk (CIs misattributed) — and the completion-sentinel path in the dispatch handle pointed at the old run, so completion would never be observed.
- **Why it is a workflow gap:** pod-side-reporting.md covers sentinel/breadcrumb discipline for launches but has no relaunch-descope clause (`grep -c 'descope' .claude/rules/pod-side-reporting.md` -> 0).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'descope' .claude/rules/pod-side-reporting.md` -> 0, compose time; descope + stale-handle facts quoted from #1689's epm:progress v64 audit marker (durable, user-requested deep dive).

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/rules/pod-side-reporting.md` (relaunch/pid-rewrite section): a relaunch changing recipe (layers/draws/scope) or out-root (1) posts an epm:progress descope note naming old->new recipe BEFORE launch, and (2) rewrites `.claude/cache/issue-<N>-handle.json`'s sentinel/log paths to the new run's paths in the same step — a handle pointing at a dead run's sentinel is a silent poller kill.

## Scope / surfaces

- Primary target: `.claude/rules/pod-side-reporting.md`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 10bcfb7be4cb

- workflow_fix_target: .claude/rules/pod-side-reporting.md
