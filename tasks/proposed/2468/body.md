---
title: 'workflow-fix: implementer runs no-flags workflow_lint when the diff touches
  .claude/** markdown'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2996ca5b0fa6
- workflow-fix
created_at: '2026-08-22T11:26:21Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate v1 raised by the #2263 implementer during review
  round 3: commit abfd6482fc edited .claude/skills/issue/steps/10-step-6.md and landed
  lint-red (bang-backtick inline-exec hazard, #1243/#1266) after passing 132 tests
  + ruff + ruff-policy pin + the gate-scope pin sweep; the implementer After-Implementation
  checklist has no no-flags workflow_lint.py leg for .claude/** markdown diffs, so
  the failure would have surfaced only at the Step 9c gate 20-30 min later (the #1681/#1388
  shape).'
workflow: v1
---
## Overview / Motivation

Auto-filed from a `<!-- workflow-fix-candidate v1 -->` raised by the #2263 implementer during review round 3. The implementer's After-Implementation checklist mandates gate-matched test selection, `ruff`, and the ruff-policy full-ruleset pin — but nothing runs the **no-flags `scripts/workflow_lint.py`** when the diff touches workflow-surface markdown. A `.claude/**` markdown edit can therefore land lint-red and surface only at the Step 9c gate 20-30 minutes later.

## Goal

Add a conditional lint leg to the implementer After-Implementation checklist: when the round's diff touches any `.claude/agents/`, `.claude/skills/`, or `.claude/rules/` markdown file, run the no-flags `uv run python scripts/workflow_lint.py` locally and report its verdict in the results marker's verification section. A non-WARN failure line naming a round-touched file blocks the report.

## Workflow gap

**Bug observed (#2263 round 3, 2026-08-22).** Commit `abfd6482fc` edited `.claude/skills/issue/steps/10-step-6.md` and introduced a bang-backtick inline-exec hazard at line 758 (the #1243/#1266 check) inside a comment it added to the launch-fence guard. That commit passed every leg the implementer spec currently mandates:

- `tests/test_verify_carryover_inputs.py` — 132 passed (baseline 131)
- `ruff check` + `ruff format` — clean on all four touched files
- the ruff-policy full-ruleset pin — PASS
- the gate-scope pin sweep — 70 hits, 69 run locally, 2,543 passed

and it still left the tree **lint-red**: the no-flags `workflow_lint.py` returned rc=1. Nothing in the checklist would have caught it. It was found only because the round-3 implementer was re-spawned to audit its own predecessor's landed commit and chose to run the linter unprompted; the fix landed as `5b1655c1a3a0e311b0c5a61110d6257a1265bdf5`.

**Why it is a workflow gap, not a one-off.** The mandated legs are all *code* instruments — pytest selection, ruff, the ruff policy pin. `workflow_lint.py` is the only instrument that checks workflow-surface **markdown** (inline-exec hazards, size ratchets/corridor caps, reference resolvability, the lessons index, marker-shape pins), and it is exactly the class of file a `workflow-fix` task edits by definition. So the gap is structurally biased toward the task kind most likely to trip it.

**Cost shape.** This is the #1681/#1388 shape: a lint-red landing is invisible until the Step 9c gate runs, which on this repo is a 20-30 minute wall. Worse, a lint-red tree committed mid-round can red the gate *fleet-wide* for sibling sessions (#1388: two inline-landed lint-red scripts broke the Step 9c gate for every concurrent session).

**Distinct from the existing implementer-scope fixes** — checked at filing, and the check is the evidence: #1699 (`daily-fix: implementer pin-sweep + lint parity`) widened the pin sweep and swapped the default `ruff check` for the repo ruff-policy pin; #1305 (`daily-fix: implementer pre-report 9c touched-scope run`) aligned the implementer's test selection with the gate's touched-scope selection. Both are COMPLETED, both were satisfied green in this very round, and neither mentions `workflow_lint.py`. The #2263 incident is the demonstration that their remedies do not cover this leg.

## Proposed change (sketch — refine in planning)

In `.claude/agents/implementer.md`, After-Implementation lint step (and `.claude/agents/experiment-implementer.md` if planning confirms the same gap, plus the Step 4b brief template in `.claude/skills/issue/steps/08-step-4.md` if the duty needs restating at dispatch):

```
+ When the round's diff touches any `.claude/agents/`, `.claude/skills/`, or
+ `.claude/rules/` markdown file, ALSO run the no-flags linter:
+     uv run python scripts/workflow_lint.py
+ (backgrounding with a >= 600 s fence is acceptable — the no-flags run is slow).
+ Report its verdict in the results-marker verification section alongside the
+ ruff and test legs. A non-WARN failure line naming a round-touched file BLOCKS
+ the report: fix it and re-run. Pre-existing red elsewhere never blocks, and a
+ dead/silent leg is INCONCLUSIVE, never clean — the same attribution contract
+ the Step 9a-ter inline payload lint gate already uses.
```

Planning should decide whether the trigger condition is best expressed as a path-glob check the implementer runs itself, or mechanically via the existing selector tooling.

## Verified at filing

- `grep -n "workflow_lint" .claude/agents/implementer.md` → the target of this fix; confirm the count is zero for the After-Implementation lint step before implementing.
- #2263 `events.jsonl` `epm:results v5` — the round-3 audit report carrying both linter runs.
- `/tmp/wl-2263-r3-audit.log` (FAIL) vs `/tmp/wl-2263-r3-audit2.log` (PASS post-fix).
- Fix commit `5b1655c1a3a0e311b0c5a61110d6257a1265bdf5` on branch `issue-2263`.

## Provenance

workflow_fix_target: .claude/agents/implementer.md

Raised as a `<!-- workflow-fix-candidate v1 -->` by the `implementer` subagent during #2263 review round 3; confidence at emission: medium. Routed (not parked) per `.claude/rules/workflow-fix-on-bug.md` — any in-scope gap at ANY confidence is auto-filed and dispatched, and low-confidence follow-ups are RUN rather than parked, because the spawned session's own planner + critic ensemble + code-review are the check.

Recursion guard checked and NOT active: `task_workflow.is_workflow_fix_session(2263)` returned `False` (#2263 carries the `workflow-fix` tag but no `workflow_fix_target:` Provenance line), so the filing session is not itself under the guard. Independently, this candidate targets a DIFFERENT file than #2263's own deliverable (`.claude/agents/implementer.md` vs `.claude/skills/issue/steps/10-step-6.md` + `scripts/verify_carryover_inputs.py`) and is a distinct bug, which the dedup rule routes to its own task regardless.
