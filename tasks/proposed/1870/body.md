---
title: 'daily-fix: sync KEPT-stash outcome gets durable surfacing'
kind: infra
tags:
- wf-fix
- wf-fix-fp:65e1e9485df3
- daily-auto-filed
created_at: '2026-07-30T07:11:34Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): The #1773 merge''s sync
  reported ''stash: KEPT stash@{0} (319c2bf16e7c) — manual triage'' on stdout only;
  the session posted epm:merged and ended without dispositioning it — the stash is
  invisible to every later session except as recurring sync noise'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner E-P8 (session 61fa0ccf, #1773; probed) + C-P7/F-P8 (3+ sessions re-observing the same stash as noise)).

## Goal

A KEPT autostash is stranded work-in-progress on the shared root; stdout of a terminal-step Bash call is not a durable handoff.

## Workflow gap

- **Bug observed:** stash@{0} (319c2bf16e7c) is still parked now, joining older stranded stashes; the triage itself is already tracked as open daily-held #1736 (23 stashes) — this filing is ONLY the durable-surfacing mechanism so future KEPT outcomes do not depend on a session noticing stdout.
- **Why it is a workflow gap:** sync_repo_root already treats KEPT-vs-dropped correctly; the gap is purely observability (the #1680 lesson class: suppression/keep outcomes need durable records).
- **Confidence (emitter):** medium
- verified-at-filing: `git stash list` -> stash@{0} present; rescue patch exists at ~/.task-workflow/root-sync-rescue/stash-319c2bf16e7c.patch (miner-probed 2026-07-30); triage tracked in #1736 (proposed, daily-held).

## Proposed change (refine in planning)

Emit a sidecar JSONL row (path in the existing .claude/cache/ convention) per KEPT outcome + print the row path in the advisory; optionally a deduped Telegram escalation when KEPT count grows.

## Scope / surfaces

- Primary target: `scripts/sync_repo_root.py`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: scripts/sync_repo_root.py
- fingerprint: 65e1e9485df3
