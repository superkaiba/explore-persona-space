---
title: 'daily-fix: gotchas — subsample filters, zones, hub path glob'
kind: infra
tags:
- wf-fix
- wf-fix-fp:14e4a4fd9587
- daily-auto-filed
created_at: '2026-07-29T07:17:18Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): three 2026-07-28 traps
  missing from the durable surface: #1738''s kresample subsampled a split-time id
  list without re-applying the capture phase''s admission filter (vLLM engine-fatal
  on 12 skipped rows); ad-hoc gcloud ssh/describe probes hardcoded us-central1-a while
  the router landed the instance in us-central1-c (instance-gone misread); HF path_in_repo
  fed a glob → 404 plus not-yet-existing-pref'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Sources: group-F P4 + P7 + P11.

## Goal

Add three 2026-07-28 GCP/Hub seam traps to gotchas.md.

## Workflow gap

- **Bug observed:** (1) #1738 Phase-4a: a kresample subsample drew from the SPLIT-time id list; 12 rows the capture phase had skipped (over-length cis + pilot-era no-primary rows) reached vLLM and killed the engine (`add_request` ValueError). The in-session lesson marker covers #1738 only. (2) Ad-hoc `gcloud compute ssh`/`describe` probes hardcoded `us-central1-a`; the router had landed eps-issue-1738 in `us-central1-c` — not-found reads were briefly misread as instance-gone. (3) HF `path_in_repo` was fed a glob (URL-encoded `%2A` visible in the 404) and a not-yet-existing prefix was listed — 4 tool_result firings across 2 sessions. Note #1778 (landed) covers verify_artifacts_exist's hf:// glob crash — this entry covers the ad-hoc-probe class.
- **Why it is a workflow gap:** all three are re-hittable seams whose lessons currently live only in one session's markers/transcripts.
- **Confidence (emitter):** medium (mechanisms read from the sessions' own failure text; the glob mechanism verified via the encoded %2A in the traceback)
- verified-at-filing: gotchas.md has no producer-filter/zone/path_in_repo entries (grep at compose time found no matching entries; label: the spawned session re-greps gotchas.md before writing to avoid duplicating adjacent entries).

## Proposed change (candidate diff sketch — refine in planning)

Three concise entries in the matching gotchas sections (vLLM; GCP; Hub-API).

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: 14e4a4fd9587

- workflow_fix_target: .claude/rules/gotchas.md

