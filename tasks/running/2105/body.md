---
title: 'workflow-fix: triage window boundary opens at post time, hiding enumerate-to-post
  seam markers'
kind: infra
tags:
- wf-fix
- wf-fix-fp:dfb17c925991
created_at: '2026-08-06T01:31:24Z'
has_clean_result: false
origin_prompt: 'Promised follow-through from #2054 v108 triage correction; v98 diagnosed
  the window-seam mechanism (v91 landed 53s before the r11 breadcrumb and was permanently
  invisible to triage_candidates_since_last_dispatch). Candidate block in body Provenance.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #2054 (emitting agent: issue-orchestrator). Promised in #2054's triage-correction note (epm:progress v108); root-cause diagnosis in #2054's v98 marker.

## Goal

Record the enumeration-boundary timestamp in the `external-markers triaged:` line so the next triage window opens from the enumeration point, closing the enumerate-to-post seam.

## Workflow gap

- **Bug observed:** markers landing between the triage enumerator run and the breadcrumb post fall behind the new boundary and are permanently invisible to the pre-dispatch triage duty
- **Why it is a workflow gap:** `task_workflow.triage_candidates_since_last_dispatch` opens the candidate window strictly AFTER the most recent event carrying the triage line (or a launch-kind marker), but the triage line is posted at breadcrumb-post time T1 while enumeration happened at T0 < T1 — any external marker landing in (T0, T1) was not enumerated at T0 and is behind the boundary at every later call, so no session ever triages it. Incident (#2054, 2026-08-05): the user-directive marker v91 ("make sure training is at least 5000 for each setting") landed 53 s before the r11 breadcrumb post and was invisible to rounds r11–r14; caught only by a manual events re-read (v98 forensics), costing a 4-round directive miss and a triage-correction round (v108).
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "TRIAGE_LINE_PREFIX" src/explore_persona_space/task_workflow.py` → 9 hits in 1 file (def at :2762; the boundary matcher at :2839-2846 — `boundary = idx` on `kind in launch_kinds or TRIAGE_LINE_PREFIX in note`, window = events strictly after boundary; sibling pre-record window logic at :2981, :3061-3088); plus `grep -rn "external-markers triaged" .claude/skills/issue/SKILL.md` → 5 hits (triage-line format spec at :6948-6974) (2026-08-06). The seam is structural in the boundary matcher: no token in the triage line carries the enumeration point, so the window boundary can only be the post position.

## Proposed change (candidate diff sketch — refine in planning)

```
# task_workflow.py — triage-line format gains an optional boundary token:
#   "external-markers triaged: <N> applied / <M> deferred (boundary=<ts of last enumerated event>)"
# triage_candidates_since_last_dispatch:
- if event.get("kind", "") in launch_kinds or TRIAGE_LINE_PREFIX in note:
-     boundary = idx
-     break
+ if event.get("kind", "") in launch_kinds or TRIAGE_LINE_PREFIX in note:
+     boundary = idx
+     recorded = _parse_triage_boundary_ts(note)   # None on legacy lines
+     if recorded is not None:
+         boundary = index of last event with ts <= recorded  # reopen the seam
+     break
# SKILL.md triage-line spec (~L6972): document the boundary=<ts> token; the
# enumerating session stamps the ts of the LAST event it enumerated (or the
# enumeration wall-clock) into its triage line at post time.
# Legacy lines without the token keep today's behavior (fail-toward-today, never wider misses).
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py`
- Secondary: `.claude/skills/issue/SKILL.md` (triage-line format spec + the duty text that composes the line)
- Grep the workflow surface for the pattern before editing (`grep -rn 'external-markers triaged' .claude/ CLAUDE.md scripts/ src/explore_persona_space/task_workflow.py`) and update every composer/parser hit; list them in the plan. Tests: extend the existing triage-window tests (grep `triage_candidates_since_last_dispatch` under `tests/`).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- Backward compatibility is load-bearing: thousands of legacy triage lines exist across `events.jsonl` files; a legacy line (no boundary token) MUST parse to today's behavior exactly. The window must never SHRINK vs today (fail-toward-triage: over-enumeration is safe, under-enumeration is the bug).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py
- fingerprint: dfb17c925991

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/task_workflow.py, .claude/skills/issue/SKILL.md
bug_observed: markers landing between the triage enumerator run and the breadcrumb post fall behind the new boundary and are permanently invisible to the pre-dispatch triage duty
why_workflow_gap: the window boundary in triage_candidates_since_last_dispatch is the triage-line POST position, not the ENUMERATION position, so the enumerate-to-post seam (53 s in the #2054 v91 incident) permanently hides markers
proposed_change: record the enumeration-boundary timestamp in the triage line so the next triage window opens from the enumeration point, closing the enumerate-to-post seam
diff_sketch: |
  + triage line gains "(boundary=<ts>)" token stamped by the enumerating session
  + triage_candidates_since_last_dispatch parses the token and reopens the window
  + from the recorded enumeration point; legacy lines keep today's behavior
confidence: high
related_task: #2054
<!-- /workflow-fix-candidate -->
