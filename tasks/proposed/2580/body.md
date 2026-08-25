---
title: 'gotchas.md: MooseFS errno-116 partial install passes uv audit while import
  fails (probe-by-import + UV_LINK_MODE=copy recipe)'
kind: infra
tags: []
created_at: '2026-08-25T17:19:40Z'
has_clean_result: false
parent_id: 2378
workflow: v1
---
## Goal
Add the MooseFS errno-116 partial-install uv-audit blindspot to .claude/rules/gotchas.md (env-health-by-import probe + UV_LINK_MODE=copy resync recipe).

## Context
#2378 pod-2378-patch bootstrap: a MooseFS errno-116 partial install landed pyyaml's dist-info WITHOUT its payload; uv's audit (and uv run auto-sync) read the venv as complete while the import failed. Remedy verified: rm -rf .venv + UV_LINK_MODE=copy resync. Add to gotchas.md beside the existing MooseFS EDQUOT/read-wedge entries: probe env health by ACTUAL IMPORT, never uv audit, on MooseFS-backed pods; the H200 driver-550/cuda-compat-13 pre-check (#2330) is already in the rule — cross-link it as the same-host-family check.

## Provenance
Surfaced as an epm:failure-lesson (experimenter, pod-2378-patch launch, 2026-08-25; root_cause_confirmed yes, gotcha_candidate yes) on #2378.
