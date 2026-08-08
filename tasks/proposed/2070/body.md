---
title: 'daily-held: 61-deep auto-dispatchable infra queue is now the'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-08-04T06:55:22Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-03 problem sweep (route 3): 93 proposed tasks, 87 kind:infra,
  61 auto-dispatchable and waiting behind a 5-session cap with 36 live sessions. #2022
  (one-line sparse-cone fix whose absence reds fresh worktrees'' Step 9c) has never
  dispatched in ~30h and the same bug re-bit twice since; #2006 (idle-pod teardown)
  likewise sits proposed while that leak recurred four more times.'
workflow: v1
---
## Overview / Motivation

Filed by /daily 2026-08-03 as a **route-3 needs-human** item. The auto-dispatchable infra queue has grown to 61 tasks and the throughput ceiling is now the binding constraint on every workflow fix the fleet files — including the ones filed to stop recurring money leaks. Choosing between "raise the cap", "prune the queue", and "accept the latency" is a resource/priority call, not a judgment /daily should make for you.

**Not a duplicate of #1853** (`daily-fix: watcher leaves filed infra tasks undispatched`, completed): that task fixed the MECHANISM — the sweep's dedup branch that posted a routed-record without dispatching, and its oldest-first drain with no urgency ordering. This item is the evidence that the mechanism fix did not clear the CONDITION: #2022 has now sat undispatched for ~30h *after* #1853 landed, and the queue is 61 deep. The remaining question is not a bug, it is a resource/priority call — which is why it is route 3 rather than another route-2 filing.

## The numbers (probed 2026-08-04)

- `proposed` tasks: **93** — of which **87** are `kind: infra`.
- Of those, **26** carry `needs-human` (correctly excluded from auto-dispatch) and **61** are AUTO-DISPATCHABLE and waiting for the watcher's `proposed_infra_sweep`.
- Live sessions: **36** (`spawn_session.py list`, EPS only). The shared infra-session cap is 5.
- Tonight /daily adds 7 more route-2 filings to that 61.

## Why it is now load-bearing, not just untidy

Concrete case from tonight's sweep: **#2022** (`workflow-fix: add eval_results/issue_1481 to tests/sparse_cones.txt`) was filed 2026-08-02T17:55:48Z and has **never dispatched** — its events stream contains exactly one row, `epm:created`. Its body records that the missing sparse cone makes fresh sparse worktrees' Step-9c gate die at test collection. In the ~30 hours since, the same bug re-bit again: session #1965 hit the identical collection failure on 2026-08-03 and parked a fresh candidate for it (which tonight's Step C sweep deduped against #2022). Yesterday's daily recorded five sessions hitting it in one day.

So the queue depth is not deferring cosmetic fixes — it is deferring fixes whose absence keeps costing sessions, and the re-raise traffic (parked candidates, duplicate filings, dedup work) is itself consuming the capacity that would clear them.

Same shape, money-side: **#2006** (box teardown leg, filed 2026-08-01 for the done-but-billing pattern) is still `proposed` while that exact pattern recurred three times on 2026-08-02 and again tonight (~$187 of orphan pod-1739 duplicates + a bootstrap-failed pod billing 25 min, both spotted by you rather than by the fleet).

## The options, as I see them

1. **Raise the infra-session cap** (5 → higher). Cheapest lever; risk is VM load + more concurrent root committers, and tonight already showed 36 live sessions with autocompact thrash epidemic-level (see the lean-twin filing) — so more concurrency may make the thrash worse, not just the throughput better.
2. **Prune / triage the 61.** Many are low-severity recurrence-counter items (guard-block compliance nags, one-off tool errors). A deliberate archive pass on the bottom half would let the top half actually land. This needs your priorities, not mine.
3. **Priority-order the sweep** rather than raising the cap — dispatch by severity (money-leak and main-red fixes first) instead of the current order. This is a real workflow change and would itself need filing.
4. **Accept the latency** and stop filing the low-severity tail (i.e. tighten /daily's route-2 bar so the queue only receives fixes worth a pipeline run).

## Suggested action

If you want one thing: **dispatch #2022 by hand** (it is a one-line registry fix that is currently costing every fresh sparse worktree), then pick between (2) and (3) for the backlog as a whole. I can route (3) as a normal route-2 workflow-fix the moment you say which ordering you want.

## Provenance

- Surfaced by /daily 2026-08-03 problem sweep (queue census + the #2022 / #2006 non-dispatch evidence).
- Route 3 trigger: genuinely ambiguous intent / scope — a resource-allocation and priority call only you can make.
