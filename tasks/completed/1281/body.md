---
title: 'workflow-fix: sweep park predicate misses mid-note parks'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ae7b7d49ff93
- daily-auto-filed
created_at: '2026-07-12T06:52:28Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-11 problem sweep (route 2): sweep_parked_wf_candidates.py
  _row_is_parked misses genuinely parked candidates whose note prefixes the park declaration
  with other prose — #1271''s Guard-1 candidate note starts with a root-sync recovery
  record and ends with ''Routing: parked — ...'', matching neither _PARKED_LEAD_RE.match
  nor ''routed: parked'', so the 2026-07-11 Step C sweep silently skipped it'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-11 from its own problem sweep (emitting agent: /daily orchestrator; the Step C enumerator missed a genuinely parked candidate tonight).

## Goal

Broaden `sweep_parked_wf_candidates.py` `_row_is_parked` to match a park declaration anywhere in the note (e.g. `Routing: parked` / a `parked` + recursion-guard mention mid-note), not only a leading `parked` or `routed: parked`.

## Workflow gap

- **Bug observed:** `_row_is_parked` misses genuinely parked candidates whose note prefixes the park declaration with other prose — #1271's Guard-1 candidate note (2026-07-11T18:39:16Z) starts with a root-sync recovery record and ends with `Routing: parked — running under workflow_fix_target recursion guard ...`, matching neither `_PARKED_LEAD_RE.match` nor the literal `routed: parked`, so the 2026-07-11 Step C sweep silently skipped it (it was recovered only because the /daily transcript miner independently surfaced it).
- **Why it is a workflow gap:** the sweep is the recursion-guard escape valve's only routing path; a parked candidate it cannot see is a dropped bug, violating the "never silently lost" contract in `.claude/rules/workflow-fix-on-bug.md` § Recursion guard.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "parked" scripts/sweep_parked_wf_candidates.py` → `_PARKED_LEAD_RE = re.compile(r"\s*parked\b", re.IGNORECASE)` (line 104, used via `.match`) + `_PARKED_ROUTED_RE = re.compile(r"routed:\s*parked\b", ...)` (line 105) are the only two accept paths in `_row_is_parked` (lines 195–201); #1271's note verified to match neither (2026-07-12).

## Proposed change (candidate diff sketch — refine in planning)

```diff
  _PARKED_ROUTED_RE = re.compile(r"routed:\s*parked\b", re.IGNORECASE)
+ _PARKED_MIDNOTE_RE = re.compile(r"(?i)\brouting:\s*parked\b|\bparked\b[^\n]{0,120}recursion guard")
  def _row_is_parked(row):
-     if _PARKED_LEAD_RE.match(note) or _PARKED_ROUTED_RE.search(note): return True
+     if _PARKED_LEAD_RE.match(note) or _PARKED_ROUTED_RE.search(note) or _PARKED_MIDNOTE_RE.search(note): return True
```
Plus a red fixture reproducing the #1271 note shape (record-prose prefix + trailing `Routing: parked`). Keep the suppression logic (routed-record matching) unchanged.

## Scope / surfaces

- Primary target: `scripts/sweep_parked_wf_candidates.py`
- Tests: the sweep's existing test file (add the #1271-shape fixture red-green).

## Constraints / invariants

- Workflow-surface only; ruff passes; the sweep stays read-only.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/sweep_parked_wf_candidates.py
- fingerprint: ae7b7d49ff93

Origin: /daily 2026-07-11 problem sweep — the Step C run at 2026-07-12T06:32Z returned 6 candidates and omitted #1271's formal Guard-1 candidate (`epm:workflow-fix-candidate` v1, 2026-07-11T18:39:16Z), which a transcript miner independently surfaced the same night.
