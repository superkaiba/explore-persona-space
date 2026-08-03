---
title: 'daily-held: #1689 fit_ladder cannot finish — kill, cut, or r'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-27T07:21:28Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 3): #1689''s fit_ladder is
  a 1,210,608-evaluation serial dense-SVD loop that has produced zero durable output
  in 5 h 14 m while holding a 4x H100 pod at 0% GPU; the measured extrapolation is
  56-112 days and even an optimistic 1 s per evaluation gives ~14 days'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 3 — judgment call).
Held because the decision spends compute, may terminate a live pod, and changes what
`#1689` measures: all three sit inside the judgment-call carve-out.

`#1689`'s `fit_ladder` phase cannot finish on any useful horizon, is producing no durable
output, and is holding a 4x H100 pod at 0% GPU while it runs.

## The situation (verified 2026-07-27T06:33Z on pod-1689)

**Verified facts** — each read directly off the live pod or the repo:

- **Sizing.** `enumerate_pair_set()` returns **126** ordered pairs. `CAPTURE_LAYERS` has 4
  entries, each pair runs 2 arms (`prefix`, `context`), and each arm runs
  `1 + N_BOOTSTRAP_DRAWS + N_REPARAM_NULL_DRAWS` = `1 + 1000 + 200` = **1,201** ladder
  evaluations (`scripts/issue1689_common.py:188-189`). Total:
  `126 x 4 x 2 x 1201` = **1,210,608 evaluations**.
- **Per-evaluation work.** Each evaluation runs 4 inner-CV ridge fits plus one dense
  3584x3584 SVD (the rung-6 Procrustes step, `np.linalg.svd` at
  `scripts/issue1689_fit_ladder.py:340`). Single-process CPU numpy.
- **Elapsed with nothing to show.** The `fit_ladder` process (pid 41602) has run
  **5 h 14 m** at 2.7 GB RSS. The workload log has emitted **no line since
  01:18 UTC** — the last entry is the bare `[phase=fit_ladder]` banner.
  `eval_results/issue_1689/ladder/` **does not exist**: zero durable output.
- **Why zero output.** `run_all_pairs` accumulates every pair into an in-memory dict and
  writes ONE json per layer, only after all 126 pairs x 2 arms complete
  (`scripts/issue1689_fit_ladder.py:729-806`). A crash, preemption, or stop right now
  loses 100% of the compute.
- **GPU state.** All 4 H100s report `0 %` utilisation and `0 MiB` used. The watcher posted
  `gpu-width-advisory` (2026-07-27T01:11Z), `gpu-underparallel-warning` (01:11Z),
  `gpu-idle-advisory` (02:43Z) and `gpu-idle-escalation` (02:43Z), the last naming the
  #664 spend-leak class explicitly. Nothing acted on any of them.

**Measured extrapolation (session `dffde9b6`, 2026-07-27T05:54-05:56Z)** — the ETA basis,
recorded here as that session's measurement rather than an independently re-run one:
ridge eigh-prep timed at 1.1 s (small cells) to 5.1 s (the 11,400-row cells) and the dense
SVD at 13.0 s, giving **17-33 s per evaluation** and **~56-112 days** for 1.21M
evaluations. Sanity floor: even at an optimistic **1 s** per evaluation the phase still
needs **~14 days**. The order of magnitude does not depend on the exact per-eval figure.

## Why this needs you

Three carve-out triggers, any one of which is sufficient:

- **Spends money / holds compute.** The pod is 4x H100. Continuing, downsizing, or
  terminating it is a spend decision.
- **Scientific meaning.** Cutting the draw counts (1000 bootstrap / 200 null), the pair
  set (126), the layer sweep (4), or the rung ladder changes what `#1689` measures and
  what its confidence intervals mean. That is not a decision automation should take.
- **Destructive.** Killing the phase discards 5+ hours of in-memory state (though, as
  above, that state is currently worth zero on disk either way).

## Options (each with its cost)

1. **Stop the phase now, re-plan the ladder as a batched computation.** The inner loop is
   the many-cell repeated dense-factorization class `CLAUDE.md` and
   `.claude/rules/vectorize-many-cell-fits.md` name (#722 / #778 / #823): a shared
   factorization across draws, or a Gram/dual-space batched form, is the registered
   remedy. Highest upside, needs an implementation round.
2. **Stop the phase, cut the draw budget** (e.g. 1000 -> 200 bootstrap, 200 -> 50 null) and
   re-launch as-is. Cheapest path to a number; widens every CI and is a
   scientific-meaning change you would be signing off on.
3. **Narrow the sweep** — headline layer only (`HEADLINE_LAYER = 19`) instead of all 4,
   and/or a reduced pair set. 4x-plus reduction; drops the exploratory layer dump the
   plan asked for.
4. **Let it run.** Not viable on the measured basis; listed only for completeness.

In every option except (4) the pod should be released or downsized first — `fit_ladder`
is pure CPU numpy and needs no GPU at all, so it should not be on a 4x H100 pod.

## Recommendation

Option 1 with an interim (3): release the 4x H100 pod immediately (the phase uses no GPU),
re-launch the headline layer on a CPU lane to get a usable number, and file the batched
re-implementation as the real fix. This stops the burn tonight without silently rewriting
what the experiment measures.

## What automation should NOT do here

Nothing in this filing is auto-applied. The pod stays up and the phase keeps running until
you decide. The separate class fixes (per-cell checkpointing + progress logging for long
fit loops; a measured-1-cell-pilot gate for runtime-derived cell counts) are filed as
route-2 workflow-fix tasks and do not touch `#1689`'s science.

## Provenance

/daily 2026-07-26 route-3 held item. Miner refs: G-P1, G-P2, G-P3, G-P4, A-P4, A-P5.
Live verification by the /daily orchestrator on pod-1689 at 2026-07-27T06:33Z.
Related route-2 filings: `longfit-percell-checkpoint-and-progress`.
