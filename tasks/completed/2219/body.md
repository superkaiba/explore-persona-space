---
title: 'verify_task_body: assert concern-deferred body comments have a matching deferred
  ledger event'
kind: infra
tags: []
created_at: '2026-08-10T13:51:33Z'
has_clean_result: false
origin_prompt: 'clean-result-critic surfaced-prose workflow-fix candidate on #2215
  (fabricated-deferral gap, mechanizable: yes)'
workflow: v1
---
## Goal

Close the fabricated-deferral gap in `verify_task_body.py`'s concerns audit: a clean-result body can carry a `<!-- concern-deferred: <id> -->` comment with NO matching `deferred` event in the task's `concerns.jsonl` ledger, and the verifier does not catch the mismatch. Add a check (in the existing concerns-audit check family) asserting every `concern-deferred` HTML comment in the body has a matching `deferred` ledger event for that concern id (produced by `task.py defer-concern` or equivalent); a comment without a ledger event is a FAIL (or WARN if the ledger file is absent for grandfathered tasks — decide and document the posture, forward-only per the SPEC's grandfathering conventions).

## Provenance

workflow_fix_target: scripts/verify_task_body.py
Surfaced by the clean-result-critic on task #2215 (marker `epm:clean-result-critique` v1, 2026-08-10T13:48Z, flagged `mechanizable: yes`): the #2215 body's `<!-- concern-deferred: phase-d-no-entry-skip-sentinel -->` comment had only a `raised` event in `tasks/reviewing/2215/concerns.jsonl` — the deferral implied a `task.py defer-concern` record that did not exist. Lens 14 passed on the acknowledgment prose, but the mechanical layer should assert comment↔ledger coherence so a fabricated deferral cannot ship silently.

## Acceptance criteria

1. `verify_task_body.py` flags a `concern-deferred` comment with no matching `deferred` ledger event (new check id documented in the file's check registry).
2. Fails-pre-fix regression test: a fixture body + concerns.jsonl reproducing the #2215 shape (comment present, only `raised` in ledger) turns red under the new check; adding the `deferred` event turns it green.
3. Forward-only: grandfathered bodies / absent-ledger tasks are not newly hard-FAILed (state the chosen posture in the check's docstring).
4. Existing green bodies stay green (run the verifier against a sample of completed tasks' bodies as a smoke).
