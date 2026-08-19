---
title: Remove (or neutralise) the Step-2c 100 GPU-h autonomous plan-approval gate
kind: infra
tags: []
created_at: '2026-07-28T20:46:50Z'
has_clean_result: false
origin_prompt: remove the 100 GPU-h gate from the workflow entirely
workflow: v1
---
## Overview

User directive (chat 2026-07-28, verbatim): **"remove the 100 GPU-h gate from the workflow
entirely"**, following "tell it to bypass the 100 GPU-h parking."

The Step-2c plan-approval GPU-hour cap is the autonomous gate that parks a plan at
`plan_pending` when `gpu_hours_total` exceeds `EPM_PLAN_AUTOAPPROVE_GPU_HOURS` (default 100,
code default 24 in `task.py`). It is enumerated in `.claude/workflow.yaml` § gates and in
CLAUDE.md § Auto-continuation policy as inline gate #4.

**This is an ARCHITECTURAL / public-contract change** (removes a subsystem + a documented gate
from the canonical gate enumeration), so the planner MUST set `architectural: true` and this
task MUST park at `plan_pending` for the user's explicit greenlight on the final plan — even
though the user has already directed the change in principle. The user reviews the concrete
diff, not just the intent.

## Goal

Remove — or, if the planner's analysis favours it, neutralise by raising the default — the
Step-2c GPU-hour auto-approve cap, so an autonomous `/issue` session never parks a plan on
GPU-hour grounds. Deliver whichever option the critic panel judges correct, with the
reasoning recorded.

## Two consequences the user has NOT yet weighed — surface these explicitly at plan time

1. **The cap is the ONLY cost gate in the pipeline.** CLAUDE.md: *"cost is gated ONLY at the
   Step 2c plan-approval GPU-hour cap, never mid-run"*, and dollar caps are forbidden outright
   (`tests/test_no_dollar_budget_caps.py`). Removing it leaves NO automated cost control on any
   autonomous task — a planner mis-estimate of 10,000 GPU-h would auto-approve and launch.
2. **The same gate is the ARCHITECTURAL-GREENLIGHT surface.**
   `.claude/rules/workflow-fix-on-bug.md` § Architectural greenlight routes architectural /
   public-contract changes through *"the `/issue` plan-approval gate (park at
   `plan_pending`)"*. If the gate is removed wholesale rather than made GPU-hour-blind,
   architectural workflow changes would auto-approve too — including future workflow-fix
   sessions editing the workflow surface itself. The `architectural: true` branch must survive
   any change here.
3. **The blank-estimate fail-safe is part of the same predicate.** `_plan_gate_decision`
   currently parks when `gpu_hours is None` — *"FAIL SAFE: a missing/None gpu_hours parks
   (never auto-approves on a blank estimate)"*. That is a correctness guard against a planner
   that failed to estimate at all, not a cost control. Dropping it silently would let an
   unestimated plan run.

## Options for the planner to weigh (do not pre-commit)

- **A — raise the default** (e.g. 100 → 1000/∞ sentinel) leaving the mechanism, the
  `architectural: true` branch, and the blank-estimate fail-safe intact. Smallest blast radius;
  satisfies the user's operational intent (never parks in practice).
- **B — make the gate GPU-hour-blind** but keep it firing on `architectural: true` and on a
  missing estimate. Removes the cost dimension only.
- **C — remove the gate entirely** as literally directed, accepting 1-3 above.

The planner should recommend one and state why; the critic panel gates it; the user approves
the concrete diff at `plan_pending`.

## Surfaces (grep-enumerated at filing time, 2026-07-28)

Implementation: `scripts/task.py` (`_plan_gate_decision`, `--auto-approve-if-autonomous`),
`scripts/spawn_session.py` (`--auto-approve-gpu-hours` flag → `EPM_PLAN_AUTOAPPROVE_GPU_HOURS`
env threading, incl. the per-child campaign cap at :2342/:2406 and the respawn re-pass at
:3292), `scripts/autonomous_session_watch.py`, `scripts/file_infra_task.py`,
`scripts/workflow_lint.py`, `.claude/settings.json`, `.claude/workflow.yaml` § gates.

Docs: `CLAUDE.md` § Auto-continuation policy (inline gate #4) and § PM/spawn kickoff rule;
`.claude/skills/issue/SKILL.md` Step 2c; `.claude/skills/campaign/SKILL.md` (per-child cap);
`.claude/rules/workflow-fix-on-bug.md` § Architectural greenlight.

Tests that pin current behaviour and must be updated coherently (not deleted):
`tests/test_autonomous_plan_gate.py` (primary), `tests/test_workflow_yaml.py`,
`tests/test_workflow_lint.py`, `tests/test_autonomous_session_watch.py`,
`tests/test_spawn_session_*.py`, `tests/test_issue_tick_skill.py`,
`tests/test_stalled_detector_and_gc.py`.

Note `.claude/worktrees/issue-779/CLAUDE.md` also matched the grep — that is a stale worktree
copy, NOT a surface to edit.

## Constraints

- Workflow-surface only. `scripts/workflow_lint.py` (no-flags) passes; ruff clean on touched
  files; `uv run pytest` green with tests UPDATED to the new contract rather than removed.
- `.claude/workflow.yaml` § gates and CLAUDE.md § Auto-continuation policy must stay mutually
  consistent — `workflow_lint --check-asks` resolves every `AskUserQuestion` to a documented
  gate, so removing a gate key without updating its citers will FAIL the lint.
- Keep the interactive path intact: a non-autonomous session returns `interactive_pending`
  today and that behaviour is out of scope unless the planner argues otherwise.

## Rules to read

`.claude/rules/workflow-fix-on-bug.md` (§ Architectural greenlight), `.claude/workflow.yaml`
§ gates, CLAUDE.md § Auto-continuation policy + § Halt-criterion contract,
`.claude/rules/agents-vs-skills.md`.
