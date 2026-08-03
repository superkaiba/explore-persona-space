---
title: 'daily-held: auto-escalate GCP queue-starve to RunPod?'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-24T06:50:34Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 3): user-priority runs sat
  behind GCP capacity for hours twice today until Thomas manually forced RunPod'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-23 as a ROUTE-3 needs-human item (spends-money carve-out). Twice today a user-priority run sat behind GCP capacity until Thomas manually forced RunPod: #1586's 8×A100 wide provision queue-starved ~5.5h (a 2h-reassess checkpoint chose to keep waiting) before "just run on runpod"; #1112's runs similarly needed "run them - run on runpod and override cost if needed".

## Held decision (needs Thomas)

Should the router (or the session policy) AUTO-escalate to the RunPod rung after a bounded GCP queue-wait (e.g. N hours) on runs the user has explicitly prioritized — spending RunPod money without a per-incident ask?

- **For:** wall-clock is the scarce resource (GCP credits are not); both incidents ended with the same manual decision hours late; the FLEX_START queue-timeout failover already auto-escalates the 600s-class case — this extends the same logic to the multi-hour ladder-exhausted-but-retrying case.
- **Against (why held):** it spends real RunPod dollars automatically on a policy trigger, and "user-priority" needs a durable signal (a tag? plan flag?) — both design choices are Thomas's (spends-money carve-out item 3).

## Which carve-out held it

"Spends money or launches compute" — an auto-escalation rule authorizes future RunPod spend without a per-run ask.

## Suggested action

If wanted: define the trigger (e.g. `user-priority` tag + ≥2h queue-starve ⇒ cancel queue, provision RunPod, note the reason on the task) and file it as a router wf-fix with the escalation windows as plan-reviewed parameters.

## Provenance

Origin: /daily 2026-07-23 transcript sweep — #1586 session 62e315d1 ("just run on runpod", after ~5.5h queue-starve) and #1112 session ("run them - run on runpod and override cost if needed").
