---
title: 'Adaptive Batch-API monitoring: auto-switch a stalled batch wave to sync when
  sync is faster'
kind: infra
tags:
- adaptive-batch-sync-fallback
created_at: '2026-08-18T22:43:22Z'
has_clean_result: false
origin_prompt: add to workflow that any time we use batch it should monitor and if
  it thinks that switching to single calls would be faster it should switch
workflow: v1
---
---
kind: infra
---

## Goal

Make the multi-org dispatcher (`src/explore_persona_space/llm/api_dispatch.py`, with `eval/batch_judge.py`) **adaptively monitor any Batch-API wave and automatically switch the remaining work to synchronous (concurrent single-call) dispatch when sync would finish sooner** — instead of passively waiting out a stalled batch until its 24h SLA.

## Motivation (evidence)

Task #2223's P2 role-adherence judge wave (54,600 calls → 55 sub-batches across 3 orgs, submitted 2026-08-18 11:25Z) sat **~11 hours at 0/1000 succeeded, 0 errored** on the shared-org batch queue. A measured sync-path pilot on the same pending items ran **~1,255 items/min** (300 items, 14.3s, 0 errors, force_path="sync"), i.e. the full wave would finish in a few hours via sync vs up to ~13h more on the wedged batch. It had to be switched to sync **by hand** (kill batch subtree → cancel sub-batches → forced-sync runner). This should be an automatic policy in the dispatcher.

## What to build

During the batch-path poll loop, track realized batch completion rate (succeeded/total over elapsed). Periodically compare the **projected batch finish time** (from observed rate, falling back to the SLA when rate≈0) against a **projected sync finish time** (measured/estimated sync throughput × remaining items, honoring the org rate limits). When sync is projected materially faster than continuing the batch (with hysteresis to avoid thrash), **cancel the remaining sub-batches (org-aware) and re-dispatch the uncompleted remainder via the sync path**, merging results. Preserve: drop-never-coerce, per-item checkpoint/resume, org-aware cancel, and the existing cache.

## Design considerations / decisions for the planner

- **Cost vs wall-clock:** sync is full price; batch is ~50% off. The switch trades ~2× cost on the *remaining* items for wall-clock. Gate on a policy knob (e.g. `cost_pref`): under `latency`/`balanced` switch when sync ETA ≪ batch ETA; under `cost` never auto-switch (or only when the batch is provably wedged, e.g. ~0 progress past a threshold like 30–60 min). No dollar caps (project rule); surface the estimated premium in a log/marker.
- **"Wedged" detection:** 0 (or near-0) completions after a configurable window is the strong trigger; a slow-but-progressing batch that will still beat sync should NOT switch.
- **Anti-thrash / resume-safe:** switch at most once per wave; only re-dispatch the *uncompleted* remainder; keep it crash-recoverable.
- **Only the remainder goes sync** (completed batch rows are kept), and cancel prevents double-billing the switched remainder.

## Notes

`decide_dispatch_route`, `force_path="sync"`, the org-aware batch poll, and per-org cancel already exist — this is an adaptive fallback layered on the batch poll, not a new subsystem. Add tests: a wedged-batch fixture that triggers the switch, and a healthy-batch fixture that does NOT.
