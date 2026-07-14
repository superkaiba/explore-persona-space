---
title: implementer item-5 names tiny-real data-ingestion probe
kind: infra
tags:
- wf-fix
- wf-fix-fp:ccf544fb54d2
- daily-auto-filed
created_at: '2026-07-10T06:53:32Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): item 5 tiny-real bullet
  names only the #906 recipe; the data-ingestion streaming-probe evidence 6d.0-bis
  demands is not named on the producer side'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1157.

## Goal
Add the #1092 data-ingestion probe class ('incl. the bounded tiny-real streaming probe for real-corpus ingestion phases') to experiment-implementer.md item 5's tiny-real bullet, so the evidence Step 6d.0-bis demands is also named on the producer side.

## Workflow gap
- **Bug observed:** experiment-implementer.md item 5 tiny-real bullet (~lines 197-205) names only the #906 mock-seam recipe; #1157 named the data-ingestion tiny-real probe on the 6d.0-bis consumer/gate side but the producer-side bullet still omits it (verified on main 2026-07-09).
- **Why it is a workflow gap:** The smoke gate (6d.0-bis) now refuses seam-stubbed evidence for ingestion phases, but the agent spec that authors the smoke evidence never tells the implementer to produce the streaming probe.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)
(none — one-clause addition to the item-5 tiny-real bullet naming the bounded tiny-real streaming probe for real-corpus ingestion phases, cf. #1092.)

## Scope / surfaces
- Primary target: `.claude/agents/experiment-implementer.md`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: .claude/agents/experiment-implementer.md
- fingerprint: e35f0013608a

Parked prose-followup on #1157, 2026-07-09T09:53:56Z (Alternatives critic, Phase 2): target .claude/agents/experiment-implementer.md item 5 tiny-real bullet; add the #1092 data-ingestion probe class so the evidence 6d.0-bis demands is also named on the producer side. confidence: medium.
