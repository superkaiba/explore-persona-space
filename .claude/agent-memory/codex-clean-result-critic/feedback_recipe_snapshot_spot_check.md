---
name: recipe-snapshot-spot-check
description: When a round-1 blocker demanded inlining a reused artifact's production recipe, the r2 compose snapshots the PRODUCING issues' Methodology sections to /tmp as spot-check read targets (verify-direction, contradiction = real finding, bounded by source depth).
metadata:
  type: feedback
---

When a prior-round blocker was "inline the reused artifact's production
recipe" (Lens 10 Rule A — the #2617 r1 shape: frozen ridge maps from
#779/#1738 with no inline recipe), the revision-round compose does more
than ask "did a recipe appear": snapshot each PRODUCING issue's
`## Methodology` section verbatim to /tmp
(`/tmp/issue-<M>-methodology-snap.md`, extracted body span `## Methodology`
.. `## Results`) and add a MAP-RECIPE SPOT-CHECK SNAPSHOTS header block
naming them as read targets, with three bounds:

1. VERIFY direction — inlined recipe values (corpus, n, penalty
   selection, regime, held-out quality) must be CONSISTENT with the
   producing body; quote both sides on a hit.
2. A value CONTRADICTING the producing issue's Methodology is a REAL
   Lens 10 finding — a wrong inline recipe is worse than a deferral.
3. Values the snapshot does not state are NOT contradictions — never
   demand more depth than the producing body records.

**Why:** without the snapshots Codex can only check recipe PRESENCE
(the sandbox cannot reach task dirs by status path, and prose that
sounds complete can carry fabricated values — the #722/#665 hallucinated
"applied" class transposed to recipe content). Applied at #2617 r2
(2026-08-27), per the orchestrator brief; snapshots were ~22-24 KB each,
kept as read targets, never inlined.

**How to apply:** any revision round whose prior verdict carried a
recipe-inlining / reuse-provenance blocker; pair with the
[[prior-round-prompt-reuse]] round-2 fix-verification block and the
[[open-interp-ids-at-cr-gate]] resolution arm (addressed-events are
claims — quote landing spots).
