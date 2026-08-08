---
title: 'daily-fix: upload-policy — pack per-rollout file trees to sh'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d54f2ec279e1
- daily-auto-filed
created_at: '2026-07-29T07:07:41Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): #1739''s recovery uploads
  hit HubDirFileCountError twice (115,941 files/dir) and a packer had to be written
  mid-recovery while the instance idled ~40 min; separately a 22MB free-text DV JSON
  drew 5,938 gitleaks false positives (2m36s scan) blocking its git commit and was
  rerouted to HF ad hoc'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Sources: group-A P5 + P7.

## Goal

Prevent per-file rollout trees from hitting Hub per-directory file-count limits at upload time, and route large free-text JSONs to HF by default.

## Workflow gap

- **Bug observed:** During #1739's recovery uploads, `HubDirFileCountError` fired twice (115,941 files in one directory); a packing script had to be written mid-recovery while the GCE instance idled (~40 min). In the same session a 22MB free-text DV JSON drew 5,938 gitleaks false positives (2m36s scan) on commit and was rerouted to the HF data repo ad hoc — the reroute was correct but undocumented. (File counts are session-log figures — verify at plan time.)
- **Why it is a workflow gap:** upload-policy.md covers size-based sharding and LFS routing but has no per-directory FILE-COUNT guidance, so a 100k-file raw tree gets pointed at a per-file upload_folder until the Hub error surfaces mid-upload.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'HubDirFileCountError\|file-count' .claude/rules/upload-policy.md` → 0 hits for either (the `upload_dir_sharded` mentions at 404/659 are size/store-oriented) (2026-07-29 UTC).

## Proposed change (candidate diff sketch — refine in planning)

One guidance paragraph in the upload-stage section: file-count threshold → packed jsonl shards; plus one line routing large free-text DV/labeling JSONs to the HF data repo (gitleaks scanning does not scale to them).

## Scope / surfaces

- Primary target: `.claude/rules/upload-policy.md`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: d54f2ec279e1

- workflow_fix_target: .claude/rules/upload-policy.md

