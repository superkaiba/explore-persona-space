---
title: 'workflow-fix: verify_task_body WARN check — orphaned result JSONs (sibling
  of check 31)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:791afdb2293b
created_at: '2026-07-30T06:02:10Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1776 r1 surfaced follow-up: a WARN-level orphaned-result-JSON
  check for verify_task_body.py (sibling of check 31) — it would have mechanically
  caught the unreported jdelta_split.json'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a clean-result-critic surfaced follow-up on task #1776 (round-1 critique, 2026-07-30).

## Goal

Add a WARN-level orphaned-result-JSON check to `scripts/verify_task_body.py` — the JSON sibling of check 31 (`check_orphaned_per_unit_figures`): a committed `eval_results/issue_<N>/**/*.json` result file that is referenced NOWHERE in the body prose (no path mention, no derived-number citation marker) surfaces as a WARN naming the orphan, so silently-unreported planned reads are caught mechanically.

## Workflow gap

- **Bug observed:** #1776's committed `eval_results/issue_1776/phase4/jdelta_split.json` (the planned steer_jsplit read, cos_j/cos_perp ≈ 0.00) was reported nowhere in the clean-result body; only the clean-result-critic's manual planned-vs-actual lens caught it (round-1 minor finding 3).
- **Why it is a workflow gap:** `verify_task_body.py` check 31 already implements exactly this contract for per-unit FIGURES (WARN on a committed figure unreferenced at a body-cited SHA) but has no counterpart for result JSONs — the artifact class where a silently-dropped planned read actually lives; every task with a multi-phase eval dir can orphan a JSON the same way.
- **Confidence (emitter):** medium (critic-sketched; the spawned planner should decide the reference-detection heuristic — path substring vs basename mention — and the exemption grammar, mirroring check 31's)
- verified-at-filing: `grep -n -i 'orphan' scripts/verify_task_body.py` → 20 hits, ALL check-31 figure machinery (`check_orphaned_per_unit_figures`, L618 doc + L2920-2936 classifier); no JSON-orphan check exists (absence claim, family anchor present); landed-fix history `git log --oneline --since='7 days ago' -- scripts/verify_task_body.py` → 2 commits, neither touching orphan checks (2026-07-30).

## Proposed change (candidate diff sketch — refine in planning)

```
+ def check_orphaned_result_jsons(...):  # WARN-level, sibling of check 31
+     committed = glob eval_results/issue_<N>/**/*.json at the body-cited SHA
+     referenced = {p for p in committed if basename-or-path mentioned in body prose
+                   or covered by an exemption phrase in Methodology/Repro}
+     for orphan in committed - referenced:
+         WARN(f"committed result JSON unreferenced in body: {orphan}")
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Mirror check 31's exemption grammar + `--body-stdin` fallback semantics; update the check-table doc block; add pins in tests/test_verify_task_body.py.

## Constraints / invariants

- Workflow-surface only. WARN severity (never a new hard FAIL on grandfathered bodies).
- `scripts/workflow_lint.py` no-flags run passes; existing verify_task_body tests stay green.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 791afdb2293b
