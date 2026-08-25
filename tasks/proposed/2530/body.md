---
title: 'task.py list-concerns --open-only undercounts: a severity-downgrade re-raise
  with no addressed event is reported closed'
kind: infra
tags:
- concerns-ledger
created_at: '2026-08-24T09:12:29Z'
has_clean_result: false
origin_prompt: 'Surfaced during /issue 2254 round-3 interpretation-critique adjudication:
  the Codex twin grounded a REVISE item on --open-only output and instructed changing
  a correct ''nine open concerns'' to eight.'
workflow: v1
---
## Goal

`scripts/task.py list-concerns <N> --open-only` undercounts open concerns: a concern that was re-raised at a DOWNGRADED severity, with no `epm:concern-addressed` event, is omitted from the open set. Fix the resolution so severity-change re-raises stay open until an addressed event actually matches them.

## Evidence (task #2254, round-3 interpretation critique, 2026-08-24)

Concern id `firstk-empty-regen-cap-policy-bypass` on task #2254:

- Raised BLOCKER at 2026-08-23T10:32:22Z (`epm:concern-raised v1`, from `epm:code-review-codex v3`).
- Reconciled at 10:39:33Z (`epm:code-review-reconcile v2`), then recorded in `epm:progress v85` at 10:41:03Z verbatim as: "firstk-empty-regen-cap-policy-bypass downgraded BLOCKER->CONCERN (open, not dispatch-gating)".
- Still enumerated among carried-forward concerns by `epm:code-review-codex v4` (17:10:04Z) and `v5` (18:32:51Z).
- No `epm:concern-addressed` event for the id exists anywhere in `events.jsonl`.

Yet `list-concerns 2254 --open-only --json` returns 8 rows and this id is not among them. The full `list-concerns 2254 --json` dump shows the id twice (once BLOCKER, once CONCERN) with no per-row status field, which is consistent with the resolver treating the severity-change re-raise as a closure signal.

Sibling ids on the same task share the two-severity shape and may be affected the same way: `firstk-pilot-model-pin-unthreaded` (BLOCKER + CONCERN x2), `firstk-pack-manifest-stale-tail` (BLOCKER + CONCERN x2), `firstk-pilot-postreissue-cache-collapse` (BLOCKER x2). Whether each is genuinely open needs the same event-trail check; the count of affected ids is part of the investigation, not asserted here.

## Why this matters beyond a wrong number

The undercount propagated into a review verdict and nearly into a clean-result body. The Codex interpretation-critic twin grounded a REVISE item on the `--open-only` output and instructed the analyzer to change a CORRECT "nine open concerns" to eight and delete a genuinely-open id from the body's open list. The Claude twin caught it by reading the event trail instead of trusting the CLI, and the orchestrator adjudicated on the v85 evidence — but the failure mode is a reviewer being handed a wrong ground truth by the workflow's own tooling, which is the class of bug the workflow-fix protocol exists for. Any agent or human using `--open-only` as an authority is exposed.

## Scope

- Locate the open/closed resolution for concerns (`scripts/task.py` / `explore_persona_space.task_workflow`) and characterize exactly how a severity-change re-raise is folded.
- Fix so a re-raise at a different severity does not close a concern; only a matching addressed event does. Preserve intended behavior for genuine re-raise-then-address sequences.
- Add a regression test over the #2254 event shape: raise BLOCKER, reconcile-downgrade to CONCERN with no addressed event, assert the id is reported OPEN.
- Audit whether any other consumer (watcher passes, critic briefs, verify scripts, the `--open-only` callers in agent specs) reads the filter as authoritative and would inherit the same undercount.

## Provenance

Surfaced during `/issue 2254` round-3 interpretation-critique adjudication, 2026-08-24. Full reasoning in the task's `epm:progress` note titled "Round-3 interpretation-critique ensemble outcome + unioned revision scope", section "ADJUDICATED: the open-concern count is NINE".
