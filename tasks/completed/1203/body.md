---
title: 'workflow-fix: anchor verify_plan c12 N/A escape standalone'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7d73891fec41
- daily-auto-filed
created_at: '2026-07-09T07:00:48Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): c12 (check_battery_multiplier)
  honors its ''N/A — no draw battery'' escape doc-globally via bare re.search while
  its own FAIL detail quotes the escape phrase — the same pasted-bounce-brief self-escape
  channel #879''s c13 closed via standalone-line anchoring.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08, slice 5) from a
candidate parked on task #879 by a recursion-guarded workflow-fix session.

## Goal

Migrate c12 (check_battery_multiplier) to the standalone-line-anchored N/A escape (_standalone_na_declared), closing the pasted-FAIL-detail self-escape channel c13 already closed.

## Workflow gap

- **Bug observed:** c12 matches 'N/A — no draw battery' anywhere in the plan document (verify_plan.py:1217, bare re.search) and its FAIL detail (:1262) quotes that exact phrase, so pasting a prior round's bounce brief into the plan satisfies the escape.
- **Why it is a workflow gap:** verify_plan.py checks are mechanical gates in the plan-review pipeline; a self-escape via quoted FAIL details defeats the check for exactly the plans that already failed it once.
- **Confidence (emitter):** medium (prose-followup, methodology reconciler round 1 on #879)
- **Triage evidence (2026-07-08):** NOT fixed on main: c12 check_battery_multiplier still honors its escape via bare `re.search(NA_RE + r"no draw battery")` (verify_plan.py:1217) while its FAIL detail (:1262) quotes the escape phrase — the pasted-bounce-brief self-escape channel. The standalone-line anchor helper _standalone_na_declared exists (:1422) and is used by c13-style checks ('no empirical-null gate', 'no paired contrast') but c12 was never migrated. Completed #1006/#1041/#1042/#1075 target verify_plan.py for OTHER checks — different fingerprints, not dupes. No retraction.

## Proposed change (candidate diff sketch — refine in planning)

```
- if re.search(NA_RE + r"no draw battery", plan):
+ if _standalone_na_declared(plan, r"no draw battery"):
      return _pass(cid, name, "explicit N/A declared (no draw battery)")
+ # tests: prose-quote fixture (FAIL detail pasted into plan body) asserts not-PASS
```

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Secondary: `tests/test_verify_plan.py` (prose-quote fixture).
- Grep the workflow surface for the pattern before editing
  (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- origin: parked candidate on task #879 at 2026-07-02T22:36:43Z

Verbatim parked note:

routed: parked: EPM_WORKFLOW_FIX_SESSION (recursion guard — this IS a workflow-fix session; candidate logged, not auto-filed).

source: prose-followup (methodology reconciler, round 1)
target_file: scripts/verify_plan.py
bug_observed: c12 (check_battery_multiplier) honors its 'N/A — no draw battery' escape doc-globally (verify_plan.py ~line 1120) while its own FAIL detail quotes the escape phrase — the same pasted-bounce-brief self-escape channel #879's c13 closes via standalone-line anchoring.
proposed_change: anchor c12's N/A escape to a standalone non-fenced declaration line, mirroring c13's v2 design, + one prose-quote fixture asserting not-PASS.
confidence: medium
related_task: #879
