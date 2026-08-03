---
title: 'workflow-fix: curate experiment-implementer agent-memory index under 24KB
  FAIL threshold'
kind: infra
tags:
- wf-fix
- wf-fix-fp:942b1d4efa24
created_at: '2026-08-03T06:43:40Z'
has_clean_result: false
origin_prompt: 'ladder-filler implementer prose follow-up, 2026-08-03: pre-existing
  workflow_lint FAIL on .claude/agent-memory/experiment-implementer/MEMORY.md (24322
  bytes > 24000 FAIL threshold) blocks every subsequent inline gate round; needs curation
  per #1891'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced by the ladder-filler implementer round on task #1345 (2026-08-03): the experiment-implementer agent-memory index is over its hard FAIL threshold and now fails the no-flags workflow_lint gate for EVERY subsequent direct-to-main round until curated.

## Goal

Curate .claude/agent-memory/experiment-implementer/MEMORY.md back under the 24,000-byte agent-memory index FAIL threshold (target: under the 20,000-byte WARN band).

## Workflow gap

- **Bug observed:** `workflow_lint.py` (no flags) FAILs: `.claude/agent-memory/experiment-implementer/MEMORY.md: 24322 bytes exceeds the 24000-byte agent-memory index FAIL threshold` — the loader silently truncates the always-loaded index at ~25,000 bytes, dropping the newest lessons; and every inline payload lint gate round fleet-wide now carries this pre-existing red.
- **Why it is a workflow gap:** the always-loaded index file exceeded the #1891 ratchet; per-entry detail belongs in pointed-to per-entry files, not the index.
- **Confidence (emitter):** high
- verified-at-filing: `uv run python scripts/workflow_lint.py` → 1 error naming exactly this file at 24322 bytes (2026-08-03, two independent runs: the ladder-filler round and the results-summary round). NOTE: the file is currently MODIFIED in the shared repo-root working tree (uncommitted) — the curating session must reconcile the working-tree state first, not clobber it.

## Proposed change (candidate diff sketch — refine in planning)

Per the #1891 recipe printed by the lint: trim each index hook to ~1 line (<=~150 chars), move detail into the pointed-to per-entry file, merge duplicate/sibling rows.

## Scope / surfaces

- Primary target: `.claude/agent-memory/experiment-implementer/MEMORY.md`
- Secondary: the per-entry files under `.claude/agent-memory/experiment-implementer/` that absorb moved detail.

## Constraints / invariants

- Workflow-surface only. `workflow_lint.py` no-flags run must PASS after the change. No lesson content silently dropped — moved, not deleted.

## Provenance

- workflow_fix_target: .claude/agent-memory/experiment-implementer/MEMORY.md
- fingerprint: 942b1d4efa24
