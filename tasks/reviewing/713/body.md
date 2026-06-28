---
title: Fold /weekly's load-bearing checks into nightly /daily; retire the weekly cadence
kind: infra
tags: []
created_at: '2026-06-28T08:47:44Z'
has_clean_result: false
origin_prompt: 'PM-session directive (Thomas): ''could weekly stuff not just be daily?
  Basically I just want all of this to be automated so I dont have to run it.'' /weekly
  is unautomated and effectively dead (1 run, May 31). Fold its open_questions drift
  check + parked-proposal backstop into the nightly run; retire the weekly cadence;
  keep /weekly as manual deep-dive only.'
---
## Overview / Motivation

`/weekly` is not automated (never cron-run; its Friday nudge was disabled
2026-06-17) and has run exactly once (`logs/weekly/2026-W22.md`, May 31). Its
load-bearing functions have been rotting for ~4 weeks. Every consolidation
function `/weekly` performs is a CHECK that is strictly better run nightly
(catch drift in 1 day, not 7) — there is no cost or statefulness reason for a
separate weekly cadence. The user directive: "could weekly stuff not just be
daily? Basically I just want all of this automated so I don't have to run it."

## Goal

Fold `/weekly`'s load-bearing checks into the reliable nightly run and retire
the separate `/weekly` cron cadence, so nothing load-bearing depends on a
weekly skill the user never runs.

## Scope / surfaces

- Fold into the nightly run (the LLM `/daily` judgment pass and/or the
  deterministic consolidator, whichever fits each check):
  1. **`docs/open_questions.md` drift check** — the periodic "should this
     question be reworded / split / settled / retired" pass. Per-issue
     `living-docs-updater` is accretive only; this higher-level drift pass has
     no other home. Run nightly, idempotent (do not re-propose an open
     proposal).
  2. **Parked/declined-proposal backstop** (`workflow.yaml:560`, `:2032` —
     "parks for a future `/weekly` backstop" / "future backstop
     re-synthesis (`/weekly`)") — re-consider declined/parked proposals
     nightly.
- `.claude/skills/weekly/SKILL.md` — re-document `/weekly` as a MANUAL-only
  on-demand deep-dive (no cron); state that nothing load-bearing depends on it.
- `.claude/workflow.yaml:560` + `:2032` — repoint the backstop references to
  the nightly run.
- `.claude/agents/research-pm.md` + `.claude/skills/pm/SKILL.md` — update the
  Wednesday `/weekly` suggestion: mentor-meeting prep (the one genuinely weekly
  rhythm) stays as a manual/`/mentor-update-slides` suggestion, but the
  consolidation responsibilities move to nightly.

## Constraints / invariants

- Workflow-surface only.
- `docs/open_questions.md` mutations stay USER-GATED (the `living_docs_update`
  gate is user-only) — the nightly drift check PROPOSES, never applies.
- Idempotent: nightly re-runs must not spam duplicate proposals for the same
  drift / parked proposal.
- `scripts/workflow_lint.py` + ruff pass; keep `workflow.yaml`, `CLAUDE.md`,
  and the rule/skill files consistent.

## Acceptance

- `docs/open_questions.md` drift and parked-proposal re-consideration run as
  part of the nightly automation; `logs/weekly/` no longer needs a nightly
  producer.
- `/weekly` skill carries a clear "manual deep-dive only; nothing depends on
  this running" banner.
- A second nightly run does not re-propose a drift edit already proposed and
  still open.
