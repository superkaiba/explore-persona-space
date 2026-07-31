---
title: 'daily-fix: Monitor quiet-wait branch for Step 6d.2 poll'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2178182db950
- daily-auto-filed
created_at: '2026-07-31T06:58:43Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-30 problem sweep (route 2): #1818''s clamp fixes the
  Step 6d.2 per-call sleep at 540s, so the anti-stall quiet-interval turn savings
  are structurally unrealized; the sanctioned Monitor until-loop long-wait shape was
  never wired into the poll loop.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-30 Step C (parked workflow-fix-candidate routing) from a candidate parked on task #1818 (emitting agent: the #1818 wf-fix session, recursion-guarded; formal candidate block, parked 2026-07-30T11:17:10Z).

## Goal

Wire a Monitor-until-loop quiet-wait branch into the /issue Step 6d.2 poll recipe so the anti-stall-§7 quiet cadence (1 orchestrator wake per ~1800s on healthy quiet stretches) is actually realized instead of clamped to consecutive 540s ticks.

## Workflow gap

- **Bug observed:** #1818's clamp fixes the Step 6d.2 per-call sleep at 540s, so the anti-stall-§7 quiet-interval turn savings (1 orchestrator wake per 1800s on healthy quiet stretches) are now structurally unrealized; poll_pipeline.py's §7 comment block still reads as if the orchestrator honors the quiet interval as a longer sleep.
- **Why it is a workflow gap:** the quiet cadence was designed to cut ~330k-token orchestrator wakes on multi-hour healthy runs, but a single bg-Bash cannot sleep past the 600000 ms tool ceiling — the sanctioned long-wait shape (a Monitor until-loop, one notification per exit) was never wired into the poll loop.
- **Confidence (emitter):** medium
- verified-at-filing: `sed -n '4614,4638p' .claude/skills/issue/SKILL.md` → the clamp block ends with "§7's turn-savings intent is deferred until a Monitor-based quiet wait exists" (line ~4631, read 2026-07-31 filing time) — the deferral is still live, no Monitor quiet-wait branch is wired; `sed -n '485,500p' scripts/poll_pipeline.py` → the "Adaptive bg-poll interval (anti-stall redesign §7)" comment block is present at ~L497 (1 hit in target file). Both targets confirmed as the sites carrying the gap.

## Proposed change (candidate diff sketch — refine in planning)

Add a Monitor-until-loop quiet-wait branch to the Step 6d.2 recipe: when the previous tick recommended 1800 (POLL_INTERVAL_QUIET_SEC telemetry), run one elapsed-capped Monitor until-loop of ~1800s, then resume the normal 540s-bounded poll tick chain; update poll_pipeline.py's §7 comment block to describe the realized shape. unverified hypothesis — verify at plan time: the Monitor tool's until-loop is available to autonomous /issue sessions on every lane (the candidate assumes deferred-tool loading via ToolSearch works headless).

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`, `scripts/poll_pipeline.py`
- Grep the workflow surface for `POLL_INTERVAL_QUIET_SEC` and `next_interval` before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Gates must never be delayed: the quiet wait may only fire on the healthy/quiet tick class §7 already restricts the 1800 recommendation to.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md, scripts/poll_pipeline.py
- fingerprint: 2178182db950 (tag-authoritative; supersedes body-carried fingerprint: d3b623692322)
- origin: parked candidate-block on #1818 events.jsonl, ts 2026-07-30T11:17:10Z (routed by /daily 2026-07-30 Step C)
