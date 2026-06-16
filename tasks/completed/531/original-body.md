---
title: 'Non-saturated marker base-prior→leakage re-analysis on #478: does the marker''s
  base prior predict leakage off-ceiling, and with what sign?'
kind: analysis
tags: []
created_at: '2026-06-09T05:44:57Z'
has_clean_result: false
parent_id: 478
---
## Objective

At a **non-saturated** marker anchor, determine whether a bystander persona's own base-model prior on the marker ` ※` predicts how much the marker leaks (trained − base `log P(※)` shift) — and with which sign. This disambiguates the two readings of #504's base-prior finding:

- **#504 (saturated):** base_prior↔ΔG partial ρ = **−0.87**. But bystanders were 92% saturated (trained `log P` pinned ~0), so ΔG = trained − base ≈ −base_prior almost mechanically — the strong negative correlation is largely a **ceiling artifact**, not evidence base prior drives leakage.
- **#500 (facts, non-saturated):** the bystander's base prior on the taught *fact* predicted leakage with the **positive** (propensity) sign.
- **#478 (marker, non-saturated):** the parallel marker run — firmly off-ceiling (source trained `log P(※)` mean −11 nats, emit_rate 0/2800 held-out, 0/256 source) — but its analysis only addressed the K×distance flattening question and treated base prior as a nuisance the trained−base subtraction "mitigates." The base-prior→leakage relationship was never extracted. **This is that extraction — no new training, no GPU.**

Expected payoff: if #478's marker shift correlates **positively** with base prior off-ceiling, it confirms (a) the propensity story holds for markers (matching #500's facts) and (b) #504's −0.87 was ceiling arithmetic. If it's ~0, base prior was never the real driver for markers. Either way it sharpens the predicted sign for #530.

## Data source (existing, on HF — pull, do not regenerate)

- #478 raw completions: `superkaiba1/explore-persona-space-data` (dataset repo), files `issue_478/<cell>/raw_completions/raw_completions.json` (92 cells = 80 CORE K∈{1,2,4,8} + decomposition arms, × seeds 42/137). On-policy reads: for each (cell × held-out persona × question) the model generated its own response, then `log P(※)` was read at the post-response slot, with the base-model value at the same slot.
- Per-persona base prior on ` ※` was already computed in #478 (body quotes joker −16.97, brazilian_comedian −17.24, … zelthari_scholar/assistant/ML-engineer ≈ −25 to −26). **First step: inspect the `raw_completions.json` schema** — confirm whether it stores trained `log P`, base `log P`, and/or the shift per (persona, question). Recover base prior as `trained − shift` if base isn't stored directly; cross-check against the quoted values.
- Held-out persona → nearest-trained-source cosine distance: reuse #478's distance loader (`scripts/issue478_validate_design.py` builds the 51-persona distance matrix) so distance can be partialled out.

## What to compute

1. **Confirm non-saturation in this data** before interpreting: held-out trained `log P(※)` distribution sits below the ~0 ceiling, emit_rate ≈ 0. State it explicitly (the whole point is that #478 is off-ceiling, unlike #504).
2. **Build a tidy table**, one row per (cell, seed, held-out persona, question): `trained_logp`, `base_prior`, `shift = trained − base`, `K`, `seed`, `held_out_persona`, `min_dist_to_source`.
3. **Primary read — base prior vs leakage at non-saturation:** Spearman correlation of `shift` on `base_prior`, both at the per-row level and aggregated to per-persona means (n ≈ 35). Report ρ and sign.
4. **Guard against residual headroom/ceiling:** base prior is entangled with distance-to-source, so also report the **partial** Spearman of `shift` on `base_prior` controlling for `min_dist_to_source` (and K). Report raw ρ AND partial ρ. (Mirrors #504's partialling so the comparison is apples-to-apples.)
5. **Absolute-trained cross-check:** also correlate **absolute** `trained_logp` (not the shift) with `base_prior`. At non-saturation, a genuine propensity effect means high base prior → higher absolute trained `log P` and a positive shift correlation; a pure-ceiling regime would show flat absolute-trained near 0. This tells real-propensity from ceiling directly.
6. **Head-to-head table:** #504 saturated (−0.87) vs #500 facts (positive) vs #478 marker non-saturated (this number), with the raw + partial signs.

## Deliverables

- A short result written to the #478 events (`epm:` progress / finding marker) or a standalone note — state the #478 marker non-saturated base-prior↔shift sign (raw + partial), the absolute-trained cross-check, and the one-line interpretation (propensity vs ceiling), with the explicit head-to-head vs #504 and #500.
- One paper-quality scatter figure (use the `paper-plots` rcParams): `shift` vs `base_prior` colored by distance band, plus the absolute-trained panel. Commit to `figures/issue_478/base_prior_reanalysis/`.
- A reusable analysis script under `scripts/` (e.g. `i478_base_prior_reanalysis.py`) + the tidy table cached under `eval_results/issue_478/base_prior_reanalysis/`.
- If the sign is informative, note the implication for #530's predicted sign-flip (saturated −0.87 → de-saturated ?).

## Discipline / caveats

- Marker-specific DV stays `log P(※)`; do NOT substitute full-vocab KL (the #504 KL pitfall).
- On-policy reads only (the #478 data already is).
- Don't overclaim: 2 seeds, 1 model; report it as a re-analysis of existing #478 data, not a fresh confirmation.
- This is `kind: analysis` — no training, no pod, no adversarial-planner. Pull data, compute, report.
