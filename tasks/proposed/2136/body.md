---
title: 'daily-fix: stale prior-round PASS in verdict-present check'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d0b121291ddc
- daily-auto-filed
created_at: '2026-08-06T07:20:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-04 problem sweep (route 2): ensemble_verdicts_present
  returned a 2-day-old round-3 PASS for a round-4 check (no round token in the sentinel)'
workflow: v1
---
# daily-fix: ensemble_verdicts_present returns stale prior-round PASS — marker sentinel carries no round token

## Workflow gap

The Step 5b durable-verdict-first check (`ensemble_verdicts_present` in
`src/explore_persona_space/task_workflow.py`) answered "epm:code-review present: true,
verdict: PASS, ts: 2026-08-02" for #1336's round-4 review on 2026-08-04T14:37Z — a
round-3 verdict from two days earlier. The session's own diagnosis: "the head sentinel
`<!-- epm:code-review v1 -->` doesn't carry a round token, so the helper can't
distinguish rounds." The orchestrator caught it by timestamp reasoning; an uncritical
consumer would have skipped the round-4 review entirely and shipped an unreviewed diff.

verified-at-filing: the helper output + diagnosis are the recovery miner's probed
transcript reads (session 6c81d1cb rows 122–126, incl. the marker the session posted
documenting it). `grep -n 'def ensemble_verdicts_present' src/explore_persona_space/task_workflow.py`
run at compose time to confirm the helper exists on main.

## Proposed change

Give the durable-verdict check round awareness: either add a round/label token to the
`epm:code-review` (and sibling verdict) marker sentinels, or add a `since_ts`/round
argument to `ensemble_verdicts_present` that callers MUST pass (defaulting to the current
round's dispatch time) so a prior round's PASS can never satisfy a later round's check.
Add a regression test with two rounds of markers.

## Provenance

- fingerprint: d0b121291ddc

- workflow_fix_target: src/explore_persona_space/task_workflow.py
- origin: /daily 2026-08-04 recovery sweep — miner 4 P8 (probed rows).
