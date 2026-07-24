---
title: 'daily-fix: step9c map-mode slow-file timeout surcharges'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0ebb43e2d214
- daily-auto-filed
created_at: '2026-07-24T06:46:40Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): map-mode recommended_timeout_s
  has no per-file surcharge for known-slow test files and the test_workflow_lint wall
  figures are stale (measured 1188s vs 900s surcharge)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-23 parked-candidate routing pass (Step C). Two related parks, one from #1634 (20:31Z, after its own Step 10d gate was timeout-killed) and one from #1642 (implementer measurement).

## Goal

Make the Step 9c/10d test-gate timeout sizing survive known-slow test files: add per-file wall surcharges to map-mode `recommended_timeout_s`, and re-measure + refresh the stale `tests/test_workflow_lint.py` wall figures used by the full-gate surcharge and the SKILL.md Step 9c band prose.

## Workflow gap

- **Bug observed (two components, same sizing surface):**
  1. Map-mode `recommended_timeout_s` has NO per-file surcharge for known-slow test files: a 5-pair map including `tests/test_select_step9c_tests.py` (~200s alone) was sized to the 300s map-mode floor; #1634's Step 10d TG gated leg was timeout-killed at TG_T=300s (crash verdict, one wasted gate run) while the identical baseline leg finished at 202.9s (source: #1634 `epm:progress` 2026-07-23T20:31:48Z, "verdict=crash — TG gated leg timeout at TG_T=300s").
  2. The `tests/test_workflow_lint.py` wall figures are stale: the full-gate surcharge is 900s (`SLOW_FILE_SURCHARGES`, `scripts/select_step9c_tests.py:607`) and `.claude/skills/issue/SKILL.md` Step 9c cites a "319-771s" band, but #1642's implementer measured 1188.62s on a fresh worktree venv under shared-VM load (2026-07-24); per-call fences sized to ~850s kill it mid-run.
- **Why it is a workflow gap:** the selector's sizing is the fleet-wide gate fence; undersized fences convert healthy gate runs into crash verdicts.
- **Confidence (emitter):** high (component 1) / medium (component 2 — figure may be load-dependent; re-measure before raising).
- verified-at-filing: `grep -n "surcharge\|900" scripts/select_step9c_tests.py` → `TIMEOUT_FLOOR_S = 900` (line 601) + `"tests/test_workflow_lint.py": 900` (line 607) present; no map-mode per-file surcharge mechanism found in the map-mode sizing path (absence claim, 0 in-target hits for a map-mode surcharge); #1634's crash marker re-read from its events.jsonl at compose time (2026-07-24 UTC).

## Proposed change (candidate diff sketch — refine in planning)

Mirror the full-gate `SLOW_FILE_SURCHARGES` in map-mode sizing (or size from measured per-file walls), and re-measure `tests/test_workflow_lint.py` wall under representative load, then raise the surcharge/fence figures + the SKILL.md Step 9c band prose consistently.

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py` (+ `.claude/skills/issue/SKILL.md` Step 9c band prose if figures change)

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 0ebb43e2d214

- workflow_fix_target: scripts/select_step9c_tests.py

Origin: parked candidates on #1634 (2026-07-23T20:31:44Z, map-mode surcharge) and #1642 (2026-07-24T04:53:34Z, stale wall figures — "measured 1188.62s ... per-call fences sized to ~850s kill it mid-run").
