---
title: 'Full c-sweep for prompt + centroid steering — followup to #267''s two-point
  sign-check'
kind: experiment
tags: []
created_at: '2026-05-11T20:20:00.000Z'
has_clean_result: false
sagan_id: b0db878b-b5f8-43fa-9b4f-1492825dc536
sagan_number: 350
priority: normal
---
**Parent:** [#267](https://github.com/superkaiba/explore-persona-space/issues/267) (direction-only steering at L20 elicits the trained `[ZLT]` marker, but the prompt + centroid combination at `c=+2.0` SUPPRESSES the marker uniformly across 10 personas — 44.1% prompted → 12.4% prompt+centroid). The only data we have for prompt+centroid is `c=0` (= prompted alone, no steering) and `c=+2.0` (= sign-check cell). The intermediate and negative coefficients were never run.

## Goal

Run the full c-sweep for the prompt + centroid condition so we can see the shape of the response curve, not just two endpoints. Three concrete things this tells us:

- **Where the suppression onset is.** Is it monotonic from `c=0` downward through all positive c, or does it dip and recover? Does suppression start at very small c (~0.1–0.3) or only kick in above some threshold?
- **What happens for negative c.** Subtracting the centroid from the prompted condition (`c < 0`) — does it suppress further (predicting "any L20 perturbation breaks the prompted trigger") or rescue firing (predicting "the centroid sign matters, the direction-trigger and prompt-trigger interact non-trivially")?
- **Whether the prompt+centroid and centroid-only lines ever intersect.** If at some coefficient the two lines cross, that's a "the centroid is doing the same work in both conditions" point — interesting for the mechanism story. If they never intersect (prompted always higher), the prompt-trigger and direction-trigger are doing additive/separate work.

This fills in the purple-line gap in [#267 Figure 1](https://github.com/superkaiba/explore-persona-space/issues/267) — currently the prompt+centroid line is just two points (`c=0`, `c=+2.0`) connected by a straight segment.

## Hypothesis

If prompted + centroid is monotonically decreasing across c, the centroid acts as a uniform suppressor on the prompted-trigger response. If the line dips and recovers (or shows a peak at some intermediate c), the prompt-trigger and direction-trigger interact non-trivially — possibly the centroid resonates with the prompted activation at some coefficient and amplifies, then breaks above that.

Likely outcome: monotonically decreasing across positive c, with the negative-c side showing further suppression below the `c=0` prompted baseline (since negative-c centroid steering on the no-prompt condition gave 3.6% mean vs 16.4% no-steering — pushing the residual away from the centroid suppresses the marker).

## Design (single variable: coefficient `c`)

For each of the 10 headline personas from [#267](https://github.com/superkaiba/explore-persona-space/issues/267):

- **Conditions:** prompted (persona's own system prompt) + `c × persona_centroid` added at layer 20 via the same forward hook as [#267](https://github.com/superkaiba/explore-persona-space/issues/267).
- **Coefficient grid:** `c ∈ {−2, −1, −0.5, 0, 0.5, 1, 2}` — 7 points, matches [#267](https://github.com/superkaiba/explore-persona-space/issues/267)'s centroid-only grid except trimmed at the high-magnitude end (the `c=4`/`c=8` cells are destructive even for centroid-only; not worth sweeping into for prompted).
- **Eval:** 20 generic questions × 5 completions = n=100 per cell. Same backend (hooked HF `model.generate`, BF16 batched), same sampler config, same marker scoring (case-insensitive `[ZLT]` substring) as [#267](https://github.com/superkaiba/explore-persona-space/issues/267).
- **Single seed (42), same as [#267](https://github.com/superkaiba/explore-persona-space/issues/267).**

## Why this can't be answered by re-analyzing existing data

[#267](https://github.com/superkaiba/explore-persona-space/issues/267) ran the prompt+centroid cell only at `c=+2.0`. The full c-sweep for the prompted condition was never registered — it was a sign-check cell, not an ablation. We need the additional 5 cells × 10 personas = 50 new generation runs.

## Compute estimate

- 50 cells × n=100 = 5,000 generations
- Reuses [#267](https://github.com/superkaiba/explore-persona-space/issues/267)'s eval rig + LoRAs + centroids
- ~1 GPU-hour on 1× H100, hooked HF generation per [#267](https://github.com/superkaiba/explore-persona-space/issues/267) methodology

## Pass criteria / interpretation

- **Monotonic suppression across positive c:** strengthens "the centroid acts as a uniform suppressor on the prompted trigger" — direction-trigger and prompt-trigger don't combine constructively at any coefficient.
- **Non-monotonic prompt+centroid line with a peak at some intermediate c:** suggests prompt-trigger and direction-trigger interact non-trivially (the centroid resonates with the prompted activation at some magnitude).
- **Prompt+centroid line intersects centroid-only line at some c:** identifies "the centroid is doing the same work in both conditions" point — interpretive payoff for the mechanism story.

## Next step

Run `/issue <N>` to dispatch /adversarial-planner. No work starts inline.
