---
title: 'workflow-fix: Forward-port pin-first figure reads to v2 revi'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9b4aa7261005
- daily-auto-filed
created_at: '2026-07-09T06:58:16Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The v2-lane report-verifier.md
  (§(b) loads figures/issue_<N>/<stem>.meta.json + PNGs locally) and methodology-critic.md
  (~line 63) carry the same unpinned local figure-read pattern that #1056 fixed in
  the v1 reviewers — reads resolve against live worktree state instead of the pinned
  SHA.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1056 by a recursion-guarded workflow-fix session.

## Goal

Forward-port #1056's pin-first figure-source resolution rule (read figure PNGs/meta.json at the pinned SHA, guard against stray/stale local figures) into report-verifier.md §(b) and methodology-critic.md.

## Workflow gap

- **Bug observed:** The v2-lane report-verifier.md (§(b) loads figures/issue_<N>/<stem>.meta.json + PNGs locally) and methodology-critic.md (~line 63) carry the same unpinned local figure-read pattern that #1056 fixed in the v1 reviewers — reads resolve against live worktree state instead of the pinned SHA.
- **Why it is a workflow gap:** v1 reviewers got the pin-first rule (#1056, commit 63034ca975) but the v2 twin specs were left carrying the vulnerable pattern; when v2 becomes default the same stale-figure bug re-opens.
- **Confidence (emitter):** medium
- **Sweep verification (2026-07-08):** git log since 2026-06-30 on report-verifier.md + methodology-critic.md shows only issue-1102 and the workflow-v2 bootstrap commits — no #1056 forward-port; the files' pin language covers report image URLs/links, not the local figure-read path the candidate names. v2 is not yet default (CLAUDE.md: default v1 until dogfood clears), so this is preparatory, not urgent.

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up; mirror the 63034ca975 pin-first carve-out text into the two v2 specs, and at v2 drain land it in the plotter/report-verifier lens text via lens-coverage-map.md per the alternatives-critic concern)

## Scope / surfaces

- Primary target: `.claude/agents/report-verifier.md, .claude/agents/methodology-critic.md`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, workflow-fix-on-bug.md).

## Provenance

- workflow_fix_target: .claude/agents/report-verifier.md, .claude/agents/methodology-critic.md
- origin: parked candidate on task #1056 at 2026-07-05T18:39:01Z

Verbatim parked note:

> parked — running under workflow_fix_target Provenance (recursion guard, workflow-fix-on-bug.md § Recursion guard). Prose follow-ups surfaced by round-1 critics, NOT auto-routed by this session: (1) v2-lane report-verifier.md (§(b) loads figures/issue_<N>/<stem>.meta.json + PNGs locally, no pin discipline) + methodology-critic.md (~line 63) carry the same unpinned local figure-read pattern — forward-port the pin-first rule when v2 becomes default (source: methodology critic prose follow-up). (2) At v2 drain, land the pin-first rule in the plotter/report-verifier lens text via lens-coverage-map.md (source: alternatives critic concern 3). Both out of #1056's Goal; a non-workflow-fix orchestrator pass may file them.
