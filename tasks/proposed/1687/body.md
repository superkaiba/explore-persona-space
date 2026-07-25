---
title: 'daily-fix: route-3 dedup ignores generic workflow tokens'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8d82725cd876
- daily-auto-filed
created_at: '2026-07-25T06:52:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 2): The route-3 open daily-held
  overlap dedup suppressed the issue823 root-draft filing as already tracked in 1537
  on generic shared tokens gate step warn - 1537 is an unrelated subject, so a held
  item was nearly lost silently'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 (observed during tonight's own route-3 filing pass).

## Goal

The route-3 open daily-held overlap dedup must not suppress a held filing on generic workflow vocabulary alone.

## Workflow gap

- **Bug observed:** tonight's driver run suppressed the route-3 item `issue823-root-draft` (disposition of an untracked root draft degrading Step 9c oracles) as `already-tracked #1537 (shared: gate,step,warn)` — but #1537 is "daily-held: enforce body presence on wf-fix filings", an unrelated subject. The shared tokens are generic workflow vocabulary, not subject-bearing. The held item had to be re-filed directly (#1686) via the wrong-match escape; a less-attended run would have silently lost it (the #1483 dedup fails CLOSED for the filing).
- **Why it is a workflow gap:** `find_open_daily_held_duplicate` keys on >=3 shared informative tokens via `task_workflow.informative_title_tokens`, whose stoplist does not exclude workflow-generic vocabulary (gate, step, warn, fix, check, daily, held...), so long daily-held bodies rich in workflow prose can cross the threshold on noise.
- **Confidence (emitter):** high on the incident; medium on the fix shape (stoplist vs higher threshold vs subject-token weighting).
- verified-at-filing: tonight's `filed.jsonl` row `issue823-root-draft -> already-tracked` + the driver's `ALREADY-TRACKED issue823-root-draft -> #1537 (shared: gate,step,warn)` stdout (2026-07-25); `grep -n 'informative_title_tokens' src/explore_persona_space/task_workflow.py scripts/daily_drive_filings.py` — presence to confirm at plan time (relocation grep left to the session; unverified hypothesis — verify at plan time: the exact stoplist location).

## Proposed change (candidate diff sketch — refine in planning)

Extend the informative-token stoplist (or the #1483 matcher) with workflow-generic vocabulary, and/or require at least one SUBJECT-bearing shared token (a token absent from a corpus-frequency top-K of daily-held titles/bodies); add a regression test with tonight's pair (#1537 title/body vs the issue823 body) asserting NO match.

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py` (informative_title_tokens / find_open_daily_held_duplicate), `scripts/daily_drive_filings.py`

## Constraints / invariants

- Fail-open toward filing on scan errors (unchanged); a wrong suppression is worse than a duplicate held task.

## Provenance

- fingerprint: 8d82725cd876

- workflow_fix_target: src/explore_persona_space/task_workflow.py
