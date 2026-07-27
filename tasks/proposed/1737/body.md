---
title: 'daily-held: should /daily cap its nightly route-2 filing vol'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-27T07:22:06Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 3): the nightly sweep filed
  23 tasks against a 5-slot infra concurrency cap, and although all 19 route-2 tasks
  drained within a day the batch fills a day of pipeline capacity in one shot and
  crowds the proposed queue'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 3 — judgment call).
Held because it is a genuine scope/taste question about how much pipeline capacity a
nightly automated sweep should be allowed to claim — there is no correct answer the
workflow can derive, and both directions have a real cost.

## The situation

**Verified facts:**

- The 2026-07-25 sweep filed **23 tasks** in one night (`#1690`-`#1712`): 19 route-2
  review tasks and 4 route-3 held tasks.
- The shared infra-session concurrency cap is **5 slots**. 18 of the 23 sat at `proposed`
  waiting for a slot after filing.
- They did drain: **19 of 19** route-2 tasks reached `completed` within ~24 h (the four
  route-3 tasks correctly stayed at `proposed`). So the pipeline absorbed the batch.
- Tonight's sweep found **153 problems** across 10 miner groups and is filing a
  comparable batch.

So the batch is not failing — it is working. The question is whether it *should* be this
large every night.

## Why this needs you

Two defensible positions, and the tradeoff is a preference, not a fact:

- **Cap it.** A nightly sweep that fills a day's worth of pipeline capacity in one shot
  leaves no slack for work you initiate, and makes the `proposed` queue hard to read at a
  glance. A per-night route-2 cap (say 8-10) with the remainder carried forward would keep
  the queue legible.
- **Leave it uncapped.** Every filed item is a real, verified defect that costs real time
  while it lives. They demonstrably drain within a day. Capping means deliberately
  deferring known bugs, and the carry-over queue becomes its own backlog to manage.

There is also a middle option: keep filing uncapped but **rank** the batch, so the queue
reads top-down by cost rather than by filing order.

## Decision needed

Pick one:

1. **No cap** (status quo) — nightly files everything it verifies.
2. **Cap route-2 filings per night at N** — say what N is; the remainder carries to the
   next night in the durable filing dir.
3. **No cap, but ranked** — the driver files in descending measured-cost order and the
   brief lists them ranked, so the queue is readable without changing throughput.

My recommendation is (3): the drain data says throughput is not the problem, and ranking
addresses the actual complaint (queue legibility) without deferring known defects.

## Provenance

/daily 2026-07-26 route-3 held item. Miner ref: C-P15.
Drain figures verified by the /daily orchestrator against `tasks/` on 2026-07-27.
