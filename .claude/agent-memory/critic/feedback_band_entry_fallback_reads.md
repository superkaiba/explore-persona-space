---
name: Band-entry fallback eval rounds — two recurring fragilities
description: Dose-matched re-reads at pre-registered band-entry checkpoints (resolution-band fallback, marker-recipe item c) have two characteristic interpretation fragilities that are Concerns, not Must-Fix, when the fallback is registered
type: feedback
---

When a plan executes the pre-registered multi-arm band-entry fallback (re-read DV at each arm's earliest checkpoint crossing a threshold — #612 dose-matched round is the worked example), check two things but normally APPROVE:

1. **Does the dose lever actually move at saved-checkpoint granularity?** An arm that enters the band at its FIRST saved checkpoint, far above the bar (e.g. canned at 0.80–0.88 vs a +0.60 bar), gets ~no dose reduction — the "dose-matched" read is a matched-DIAL read with a large residual dose gap, and the headline contrast often re-lands in the same indeterminate branch as the endpoint (same N or less, point barely moves). This is NOT a REVISE when the plan names the residual gap and ships the within-cell dose-response diagnostic (endpoint − band-entry bystander Δ per cell) — that diagnostic is what lets the analyzer interpret whichever branch fires. Flag the interpretation order: a "null" cross-arm contrast at unequal dose + materially positive within-arm slope actually implies arm-dependent leakage-per-dose, not a clean "dose, not radius" confirmation.

2. **Knife-edge threshold crossings.** Cells within ~1 SE of the band threshold (#612: 0.598 excluded vs 0.603/0.608 included, SE ≈ 0.02 at 600 verdicts) make inclusion/exclusion a noise event; if a per-seed estimate's composition depends on such a cell, a supported/indeterminate flip can rest on it. Pre-registered threshold + a registered single-source robustness read + indeterminate-with-structure branch defuses the worst case → Concern for analyzer, not Must-Fix. The cheap insurance (evaluating the excluded knife-edge cell at its max-dose checkpoint as a sensitivity read) is worth naming in concerns since post-hoc the data cannot be reconstructed without a new pod round.

**Why:** First seen on the #612 dose-matched amendment (2026-06-12); the resolution-band design class (#529/#533/#546 lineage) makes these rounds recurrent.
**How to apply:** Methodology-lens review of any eval-only round labeled "dose-matched / band-entry / matched-dial fallback".
