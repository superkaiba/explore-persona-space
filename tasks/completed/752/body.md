---
title: 'daily-fix: GCP intent under-provisions 7B hidden-state capture'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6e52a2719981
- daily-auto-filed
created_at: '2026-06-30T06:44:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-06-29 auto-filed route-2: The GCP intent->machine map
  sends `eval`/`debug` to 1x L4 (g2-standard-4); a 7B forward WITH hidden-state capture
  OOMs there (22GB/16GB), so #666 and #744 crashed even though their plans called
  for A100-80.'
---
## Overview / Motivation
Auto-filed by /daily 2026-06-29 problem sweep. Recurring: #666 (L4 22GB OOM) and #744 (g2-standard-4 OOM) both crashed a 7B forward + hidden-state/Welford capture under the `eval` intent.

## Goal
A 7B-forward-with-activation-capture phase always lands on a GPU with enough HBM; the router never silently under-provisions vs the plan.

## Workflow gap
- **Bug observed:** `INTENT_TO_MACHINE` maps `eval`/`debug` -> `g2-standard-4` (1x L4). A 7B forward capturing all 28 layers' hidden states needs A100-80; on L4/16GB-RAM it OOMs. #666 and #744 plans requested A100-80 but ran on L4 and crashed mid-run.
- **Why it is a workflow gap:** the intent->machine mapping is the backend router surface; the planner sizing is the planner agent surface.
- **Confidence (emitter):** medium-high; two independent sessions.

## Proposed change
- Add/route a capture-class intent (or gate `eval` so a hidden-state-capture 7B forward escalates to an A100-80 machine), OR honor an explicit `gpu_type`/A100-80 plan request over the intent default.
- planner.md: size activation-capture phases (footprint + HBM) at plan time and pick the intent that fits, not the cheap `eval` default.

## Scope / surfaces
- `src/explore_persona_space/backends/gcp.py` (INTENT_TO_MACHINE / selection)
- `.claude/agents/planner.md` (capture-workload sizing)
- Grep `INTENT_TO_MACHINE` / `g2-standard-4` callers first.

## Constraints / invariants
- Workflow surface only. `tests/test_gcp_backend.py` / `tests/test_router*.py` pass.
- Recursion guard: EPM_WORKFLOW_FIX_SESSION=1.

## Provenance
- workflow_fix_target: src/explore_persona_space/backends/gcp.py, .claude/agents/planner.md
- fingerprint: 6e52a2719981

Sessions: 036936a1 (issue-666), 572d4953 (issue-744).
