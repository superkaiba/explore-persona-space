---
title: 'daily-fix: bracket CLAUDE.md pgrep ownership exemplar'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a6f15196b7cc
- daily-auto-filed
created_at: '2026-07-18T06:46:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-17 problem sweep (route 2): the always-on Ownership-check
  bullet models an UNBRACKETED pgrep -af ''<distinctive invocation>'' exemplar — the
  self-match shape #1335''s ownership probe hit; the counter-exemplar persists in
  the always-on surface.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-17 parked-candidate sweep (Step C) from a prose-followup candidate parked on task #1462 (emitting agents: methodology + alternatives critics, plan v1 review; parked under the recursion guard).

## Goal

Bracket the `pgrep -af '<distinctive invocation>'` exemplar in the always-on CLAUDE.md "Ownership check before any resume/launch" bullet (e.g. `pgrep -af '<distinctive invocatio[n]>'`) and add one clause naming the self-match-avoidance idiom.

## Workflow gap

- **Bug observed:** the always-on CLAUDE.md § "Ownership check before any resume/launch" bullet models an UNBRACKETED `pgrep -af '<distinctive invocation>'` exemplar — the exact self-match shape #1335's SSH-remote ownership probe hit (the grep/pgrep process matches its own command line). gotchas.md loading is not guaranteed at the moment an agent copies the exemplar, so the counter-exemplar persists in the always-on surface.
- **Why it is a workflow gap:** agents copy always-on exemplars verbatim; an exemplar that self-matches produces false "live owner" reads on ownership probes, deferring launches that are actually safe (or, inverted, masking the real owner among self-matches).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "pgrep -af" CLAUDE.md` → 1 hit at line 100, unbracketed `pgrep -af '<distinctive invocation>'` confirmed in the Ownership-check bullet (2026-07-18 UTC). Repo-wide sweep for sibling unbracketed exemplars belongs to the spawned session's plan (`grep -rn "pgrep -af" .claude/ CLAUDE.md scripts/`).

## Proposed change (candidate diff sketch — refine in planning)

```
- probe first — `pgrep -af '<distinctive invocation>'` for the workload
+ probe first — `pgrep -af '<distinctive invocatio[n]>'` (bracket one character so the probe never matches its own command line) for the workload
```

## Scope / surfaces

- Primary target: `CLAUDE.md`
- Grep the workflow surface for other unbracketed pgrep exemplars (`grep -rn "pgrep -af" .claude/ CLAUDE.md scripts/`) and bracket every prescriptive exemplar; list them in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes.
- This session runs under the recursion guard — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: a6f15196b7cc

- workflow_fix_target: CLAUDE.md

source candidate (verbatim, prose park on #1462, 2026-07-17T15:07:31Z): "target_file: CLAUDE.md. bug_observed: the always-on § 'Ownership check before any resume/launch' bullet models an UNBRACKETED pgrep -af '<distinctive invocation>' exemplar — the exact self-match shape #1335's SSH-remote ownership probe hit; gotchas.md loading is not guaranteed at that moment, so the counter-exemplar persists in the always-on surface. proposed_change: bracket the exemplar (pgrep -af '<distinctive invocatio[n]>' or equivalent) + one clause noting the idiom. confidence: medium. related_task: #1462."
