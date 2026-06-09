---
title: Off-ceiling, base prior still tracks marker shift negatively but predicts absolute
  trained log-prob strongly positively — propensity hides in the subtraction (MODERATE
  confidence)
kind: analysis
tags: []
created_at: '2026-06-09T05:44:57Z'
has_clean_result: false
parent_id: 478
---
# Off-ceiling, base prior still tracks marker shift negatively but predicts absolute trained log-prob strongly positively — propensity hides in the subtraction (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** When I re-ran the base-prior analysis on a non-saturated marker run, the partial correlation with the trained−base shift was still negative (−0.48), same sign as the saturated case but smaller — so the "ceiling arithmetic kills the sign" story is incomplete. But when I looked at absolute trained log-prob instead of the shift, the partial correlation flipped strongly positive (+0.74), matching the propensity story from the facts experiment.

**Takeaways.**
- The two readings are mathematically consistent: shift = trained − base, so a positive partial on trained with less-than-unity coupling becomes a negative partial on shift.
- The propensity mechanism is real and supported off-ceiling — it just hides inside the shift DV's mechanical −base term.
- The earlier prediction that desaturation would flip the shift correlation positive does NOT cleanly hold; the magnitude shrinks (−0.87 → −0.48) but the sign stays.
- For future predictor work: report absolute trained log-prob alongside the shift. Shift's −base term confounds the propensity read.

**How this updates me.** I now think base prior is genuinely a propensity signal (it predicts where the model ends up, in trained log-prob space) and the saturated case's strong negative was partly ceiling arithmetic AND partly a real underlying negative partial on the shift DV that just happens to be there off-ceiling too. Two stories, mathematically reconciled, not one-killing-the-other.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

I had two prior reads that pointed in opposite directions. The marker run at a saturated anchor showed partial Spearman ρ = −0.87 between the bystander's base-model prior on the marker and the trained−base shift — strongly negative, but the bystanders were 92% saturated so trained log-prob was pinned at ~0 and the shift was almost mechanically equal to −base_prior. The facts run, well off-ceiling, gave the opposite sign: the bystander's base prior on the taught fact predicted leakage positively (a propensity story). So which sign is real, and was the saturated negative just ceiling arithmetic?

The parallel marker run I'd already done sat firmly off-ceiling — source trained log P at around −11 nats, zero emission across 2,800 held-out trials — but I'd only used the data to ask about distance flattening; I never extracted the base-prior relationship. So this is an analysis-only pull on existing data: re-fit the partial Spearman in the non-saturated regime and see which sign the marker actually wants when ceiling arithmetic isn't running the show.

### What I ran

I read the per-cell log-prob arrays straight from the existing CORE-track cells (80 cells × 2 seeds × 35 held-out personas × 20 questions = 56,000 rows). For each row I have the trained log-prob on the marker at the post-response slot, the base-model log-prob at the same slot, the shift (trained − base), the held-out persona's cosine distance to the nearest source persona in the training mix, and the K factor (number of sources trained on the marker per cell). The 12 decomposition-arm cells are excluded — they train each source on a different marker, so the source-side exposure to the marker we're studying isn't uniform within a cell and would mix two regimes.

I confirmed non-saturation first: mean trained log P = −12.70 nats, 0% of rows above −1 nat. Then I ran two partial Spearmans, both controlling for min_dist and K, with a 1000-resample persona-cluster bootstrap so the within-persona dependence is respected. DV #1 is the shift (matching #504's framing). DV #2 is absolute trained log-prob (the propensity-style framing — does high base prior predict ending up at higher absolute trained log-prob).

### Findings

#### The shift partial is still negative off-ceiling — saturation didn't invent the sign

![Scatter of trained minus base marker log-prob shift against the base-model marker prior, colored by held-out persona distance band; clear negative slope across all bands.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f9ad999dcd082f48c1305525995ad4bac3cdd21f/figures/issue_478/base_prior_reanalysis/shift_vs_base_prior.png)

> **Figure.** *Off the ceiling, the shift still slopes down against base prior — same sign as the saturated case, weaker.* Each point is a (cell, seed, held-out persona, question) row, sub-sampled to ~600 per distance band for legibility. x-axis: base-model log P(marker) at the model's own post-response slot (nats). y-axis: trained − base log-prob shift (nats). Colors are persona-distance bands (near to tail). Raw Spearman ρ = −0.659 [95% CI −0.766, −0.522], partial ρ (controlling for min_dist + K) = −0.480 [−0.603, −0.333], n = 56,000 rows across 35 personas, 1000-resample persona-cluster bootstrap.

The straightforward read of the saturated case was: "trained is pinned at zero, so shift ≈ −base_prior, so of course base_prior and shift are strongly anti-correlated — that's arithmetic, not signal." If that were the whole story, getting off the ceiling should null out the correlation, or even flip it positive if there's a real propensity effect underneath. Neither happens. Mean trained log-prob sits at −12.7 nats with 0% of rows above −1 nat, the ceiling is gone, and the partial is still −0.48 — about 55% the magnitude of the saturated −0.87, same sign. So "ceiling arithmetic" as a complete explanation is wrong: there is a real negative partial on the shift DV that survives desaturation.

#### Absolute trained log-prob tells the opposite story — the propensity sign is there

![Scatter of trained marker log-prob against the base-model marker prior, colored by persona-distance band; clear positive slope hugging the y=x diagonal.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f9ad999dcd082f48c1305525995ad4bac3cdd21f/figures/issue_478/base_prior_reanalysis/absolute_trained_vs_base_prior.png)

> **Figure.** *On absolute trained log-prob, base prior predicts strongly positively — high prior personas land at higher trained mass.* x-axis: base-model log P(marker) at the post-response slot. y-axis: absolute trained log P(marker) at the same slot. Same 56,000 rows, same bands, same bootstrap. Raw Spearman ρ = +0.705 [+0.610, +0.777], partial ρ (controlling for min_dist + K) = +0.739 [+0.658, +0.807]. The cloud hugs a roughly y = x diagonal: every persona's trained log-prob is shifted up relative to its base, but the rank order in base prior is largely preserved in trained.

This is where the propensity story lives. Per-persona means make it concrete: `joker` has the highest base prior (−18.5 nats) and ends up at the highest trained log-prob (−10.4 nats), while `medical_doctor` has one of the lowest base priors (−26.0 nats) and ends up among the lowest trained log-probs (−13.6 nats). Training raises log-prob for everyone — `medical_doctor`'s shift (+12.3 nats) is actually LARGER than `joker`'s (+8.1) — but it does not equalize them. The base-prior rank order survives training. That's exactly the sign the facts experiment found in its analogue: the bystander's own pre-training stance on the target predicts where it ends up.

#### The two findings reconcile via shift = trained − base, and the saturated case was BOTH ceiling AND a real negative

The shift partial (−0.48) and the absolute-trained partial (+0.74) point in opposite directions, but they describe the same data. Mechanically: shift = trained − base, so partial ρ(trained, base) and partial ρ(shift, base) differ by the contribution of the −base subtraction. With trained and base less-than-perfectly coupled (partial coupling +0.74, not +1.00), the −base term dominates the partial on shift and drags it negative. Both numbers are true; they describe different cuts of the same joint distribution.

What this means for the saturated #504 result: the strong negative there (−0.87) was NOT purely a ceiling artifact. The desaturated read still shows −0.48 on shift. The saturated case was both real underlying negative partial on shift AND additional ceiling-driven amplification. The propensity mechanism (high base prior personas end up at higher trained log-prob) is also real, but you only see it in absolute-trained space; it gets hidden by the −base term when you read shift.

Head-to-head against the two siblings:

| | base prior → leakage, raw ρ | base prior → leakage, partial ρ | regime |
|---|---|---|---|
| This re-analysis (marker, non-saturated) | shift: −0.66; absolute trained: +0.71 | shift: −0.48; absolute trained: **+0.74** | off-ceiling |
| #504 (marker, saturated) | shift: −0.90 | shift: −0.87 | bystanders 92% saturated |
| #500 (facts, non-saturated) | leak: +0.80 (marine) | β_prior = +0.78 (joint fit) | off-ceiling |

The earlier prediction was that desaturation would flip the shift partial from −0.87 toward positive, completing the saturated-is-arithmetic story. That doesn't cleanly happen — shift stays negative, just smaller. But the propensity sign DOES show up on absolute trained log-prob (+0.74), matching the facts. So the propensity mechanism transfers from facts to markers; it just hides inside shift's −base term.

## Reproducibility

**Parameters:**

| field | value |
|---|---|
| base model | Qwen-2.5-7B (id `Qwen/Qwen2.5-7B`) |
| analysis target | task #478 marker run, CORE track only (40 cells, K ∈ {1, 2, 4, 8}) |
| seeds | 42, 137 (2 per cell) |
| n_rows | 56,000 = 40 cells × 2 seeds × 35 held-out personas × 20 questions |
| marker | ` ※` (Qwen-2.5-7B token id 83399, leading space) |
| measurement | on-policy log P(marker) at the post-response slot |
| ceiling check | mean trained log P = −12.70 nats, share above −1 nat = 0.00% |
| controls | min_dist (held-out persona cosine distance to nearest source), K (sources per cell) |
| inference | partial Spearman ρ with 1000-resample persona-cluster bootstrap (resampling held-out personas, not rows; respects within-persona dependence) |
| ARM cells | excluded (12 of 92) — they assign different markers per source, mixing two regimes |
| per-persona-mean Spearman | not emitted as a separate number; the persona-cluster bootstrap IS the per-persona inference |
| config | n/a (analysis-only; no Hydra run) |

**Artifacts:**
- Analysis script: [scripts/issue531_base_prior_reanalysis.py](https://github.com/superkaiba/explore-persona-space/blob/f9ad999dcd082f48c1305525995ad4bac3cdd21f/scripts/issue531_base_prior_reanalysis.py)
- Tidy table (56,000 rows): [eval_results/issue_478/base_prior_reanalysis/tidy.parquet](https://github.com/superkaiba/explore-persona-space/blob/f9ad999dcd082f48c1305525995ad4bac3cdd21f/eval_results/issue_478/base_prior_reanalysis/tidy.parquet)
- Summary stats (full bootstrap output, head-to-head quotes): [eval_results/issue_478/base_prior_reanalysis/summary.json](https://github.com/superkaiba/explore-persona-space/blob/f9ad999dcd082f48c1305525995ad4bac3cdd21f/eval_results/issue_478/base_prior_reanalysis/summary.json)
- Shift figure source: [figures/issue_478/base_prior_reanalysis/shift_vs_base_prior.png](https://github.com/superkaiba/explore-persona-space/blob/f9ad999dcd082f48c1305525995ad4bac3cdd21f/figures/issue_478/base_prior_reanalysis/shift_vs_base_prior.png) + PDF + meta.json
- Absolute-trained figure source: [figures/issue_478/base_prior_reanalysis/absolute_trained_vs_base_prior.png](https://github.com/superkaiba/explore-persona-space/blob/f9ad999dcd082f48c1305525995ad4bac3cdd21f/figures/issue_478/base_prior_reanalysis/absolute_trained_vs_base_prior.png) + PDF + meta.json
- Source data (per-cell log-prob arrays): per-cell `result.json` files from task #478's `issue-478` branch at commit [7efb037736831c66cf87aaa79c11237ac9268b83](https://github.com/superkaiba/explore-persona-space/tree/7efb037736831c66cf87aaa79c11237ac9268b83/eval_results/issue_478). Read via `git show <sha>:<path>` from this worktree. The HF data revision pinned in summary.json (`a9fc5a9...`) stores response text but not per-question log-prob arrays — those live in the per-cell `result.json`.
- min_dist + band assignments: reused byte-for-byte from #478's aggregate `tidy.csv` at the same SHA, not recomputed.
- Figure `.meta.json` files record `git_commit_at_render = 22d73be7e...` (intermediate worktree state during the render pass). Canonical artifact commit on `issue-531` is `f9ad999dcd082f48c1305525995ad4bac3cdd21f`; the data referenced is identical between the two SHAs.

**Compute:** Analysis-only on the local VM. No GPU, no pod. Wall time = 4m55s for the full 80-cell production run (bulk in the 1000-resample persona-cluster bootstrap × 4 partial-Spearman refits per resample). Smoke run on 2 cells = 42 s.

**Code:** Branch `issue-531`, head commit [f9ad999dcd082f48c1305525995ad4bac3cdd21f](https://github.com/superkaiba/explore-persona-space/commit/f9ad999dcd082f48c1305525995ad4bac3cdd21f). Reproduce:

```bash
git checkout f9ad999dcd082f48c1305525995ad4bac3cdd21f
uv run python scripts/issue531_base_prior_reanalysis.py
# writes eval_results/issue_478/base_prior_reanalysis/{tidy.parquet, summary.json}
# writes figures/issue_478/base_prior_reanalysis/{shift, absolute_trained}_vs_base_prior.{png,pdf,meta.json}
# add --limit-cells 4 for a fast smoke pass, --include-arm to add the 12 decomposition cells
```

**Scope caveats:**
- Single base model (Qwen-2.5-7B); marker-specific (` ※`). Do not generalize across markers without re-running.
- 2 seeds per cell. The persona-cluster bootstrap captures the cross-persona variance, which is the dominant source of uncertainty here, but the seed-direction CI is narrower than a many-seed run would give.
- 80 CORE cells of 92 total — ARM cells excluded by design (different marker per source within an arm).
- Absolute-trained log-prob was NOT pre-registered as a DV; the plan asked for shift. The shift result (partial ρ = −0.48) is the pre-registered primary; the absolute-trained partial (+0.74) is a non-pre-registered secondary that turned out to be the more informative cut. Treat the absolute-trained read as the surprise finding, not the confirmed prediction.
- Logits / `log Z` not available in #478's per-cell `result.json` (only the resolved log-probabilities are stored), so the logit-space cross-check the marker-leakage rule recommends ("report BOTH log P and logit") isn't deliverable from this data without a #478 re-eval pass.

**Follow-ups:**
- One-off forward-pass over a subset of #478's held-out (persona, question) pairs to extract raw logits + `log Z`, then report the logit-space partial alongside log P (cost_class: needs-gpu, headline_affecting: no — would make the saturation-vs-propensity story mechanically airtight rather than statistically inferred, but won't change the partial ρ signs already reported).
- Sensitivity: re-fit the partials with ARM cells included via `--include-arm`, as a robustness supplement (cost_class: free-analysis, headline_affecting: no).
- Add absolute-trained log-prob as a standard secondary DV in future marker analyses so propensity reads aren't accidentally hidden behind shift's −base term.
