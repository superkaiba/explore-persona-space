---
title: 'daily-fix: shared GCE staging HF-download retry'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f1e79e2f8f74
- daily-auto-filed
created_at: '2026-07-16T07:21:39Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): #1092 rf01 GCE box crash-looped
  >=3 same-day relaunches on LocalEntryNotFoundError Hub connectivity; the session
  hand-built a hardened retry wrapper + disabled xet/hf_transfer'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

Promote the hardened HF Hub download retry/backoff wrapper (built in-session on #1092) into the shared GCE staging path so per-issue scripts stop re-inventing it; consider defaulting the Xet/hf_transfer accelerators off for GCE staging phases with flaky egress.

## Workflow gap

- **Bug observed:** #1092's rf01 GCE box crash-looped ≥3 same-day relaunches in the p6 static-staging phase on `LocalEntryNotFoundError` Hub connectivity; the session hand-built a hardened restore-and-retry wrapper + disabled xet/hf_transfer to get through (434a2458 19:05:37Z + att-190618/201123/202906).
- **Why it is a workflow gap:** the GCE lane's workload preamble/staging path has no shared hardened Hub-download retry, so every issue script that stages inputs on a GCE box either re-invents the wrapper after crashing or crash-loops through relaunches.
- **Severity:** medium
- verified-at-filing: `grep -n 'LocalEntryNotFoundError\|retry.*download\|download.*retry\|hf_download' src/explore_persona_space/backends/gcp.py` → 0 hits — no shared hardened download retry in the GCE lane (absence confirmed); the #1315-lineage transport-retry work (#1358/#1360 per the no-action inventory) targeted upload-policy/hub.py verify paths, not the GCE staging DOWNLOAD preamble (2026-07-16 UTC).

## Proposed change (refine in planning)

Add a shared hardened Hub-download helper to the GCE staging path (workload preamble in `src/explore_persona_space/backends/gcp.py`, or an `orchestrate`/`llm.hub` staging helper the preamble sources): bounded retry with backoff around `snapshot_download`/`hf_hub_download` covering `LocalEntryNotFoundError` + connection-class failures, cache-restore between attempts, and an env knob to disable `HF_XET_HIGH_PERFORMANCE`/`HF_HUB_ENABLE_HF_TRANSFER` for staging phases on boxes with flaky egress (the #1092 in-session mitigation). Port the #1092 session's wrapper rather than re-deriving it; reconcile with the #1358/#1360 transport-retry helpers so there is one canonical retry surface.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py` (workload preamble / staging)
- Secondary: the orchestrate staging helper if the wrapper lands as a shared module; `.claude/rules/upload-policy.md` cross-link (download-side sibling of the upload-wedge ladder)

## Constraints / invariants

- Workflow-surface only (backends/*.py is in-scope workflow surface) — never experiment code, `configs/`, or `tasks/`.
- Fail-fast preserved: retries are bounded and the final failure surfaces loud (no silent fallback).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: f1e79e2f8f74

- workflow_fix_target: src/explore_persona_space/backends/gcp.py

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: 434a2458 (#1092) 19:05:37Z + att-190618/201123/202906 (batch 07 P5).
