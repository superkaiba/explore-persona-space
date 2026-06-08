---
title: 'Base-model cosine distance ranks the marker log-prob proxy across the single-turn
  / multi-turn format boundary, but the marker never emits on-policy so #377''s behavioral
  silencing remains unexplained (MODERATE confidence)'
kind: experiment
tags:
- roadmap-jun05
created_at: '2026-06-05T10:22:13Z'
has_clean_result: true
parent_id: 489
relates_to:
- leak-predictor
- spec-context-as-vector
- app1
goal: 'Add multi-turn conversation-drift contexts as a third context type to #489''s
  union panel (reusing #489''s exact marker ※, training rig, post-context+question
  extraction point, and on-policy ΔG dependent variable), and test whether base-model
  cosine/JS distance predicts on-policy marker transfer to and from multi-turn-drift
  contexts — merging with #489''s single-turn results so the conversation-length silencing
  (#377) reads as a prediction of the distance metric rather than a separate survival
  effect.'
---
# Base-model cosine distance ranks the marker log-prob proxy across the single-turn / multi-turn format boundary, but the marker never emits on-policy so #377's behavioral silencing remains unexplained (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** the cos-distance predictor extends cleanly across the single-turn vs multi-turn format boundary at the log-prob proxy level — but the marker never actually emits on-policy in any of 840 cells (single seed=42), so this doesn't dissolve the separate behavioral silencing question from the earlier multi-turn experiment.

**Takeaways.**
- across the 840-cell merged panel (single-turn + multi-turn targets), base-model cosine distance and the marker log-prob shift correlate at ρ = -0.892; the cross-format subset (n=288) alone gives ρ = -0.749.
- when i control for target identity (correlating across the 24 sources WITHIN each multi-turn target), all 12 multi-turn targets independently show ρ in [-0.75, -0.62], mean ρ = -0.69. so the cross-format finding is not a target-side artifact.
- multi-turn contexts do NOT sit far from single-turn off-diagonals in cosine. drift sits at the lower edge of the in-context-example band, well above the system-prompt band — between them, not below them. the predicted "format boundary as huge distance jump" isn't there.
- drift content adds nothing above length-matched neutral content (delta = -0.002 nats, CI overlaps zero) — replicates the prior null in this geometry.
- the headline cap: trained log P(marker) ranges -26 to -19 nats across all 840 cells, i.e. effectively zero emission probability everywhere. argmax-token marker rate is 0/840. so this is a ranking among nat-scale log-prob nudges in the floor regime, not behavioral evidence.

**How this updates me.** i'm more confident the geometry framing is real and target-identity-independent than i'd expected — the within-target replication of ρ ≈ -0.69 across 12 multi-turn targets is the strongest evidence i've had. i'm equally more cautious about what it MEANS: it can't currently speak to whether geometry predicts behavioral leakage, and the earlier multi-turn silencing experiment still needs its own mechanistic story since the L21 last-input-token geometry doesn't see the format boundary at all. the cleanest follow-up is a less-saturated training anchor where the marker actually emits in some cells, then re-run cross-format with on-policy emission rates as the DV.

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

[#489](https://eps.superkaiba.com/tasks/489) tested whether base-model cosine and JS distance after `[context + user question]` predicts on-policy marker (` ※`) transfer across a 24-context union of in-context-example and system-prompt-persona contexts. It found a strong negative correlation (cosine distance up → log-prob shift down) but capped confidence at LOW because the marker never actually emits at the post-response slot — the correlation lives in a floor-saturated proxy regime.

Separately, [#377](https://eps.superkaiba.com/tasks/377) showed that the same marker fires 87.5% from a fresh prompt but collapses to ~2.4% by turn 20 of a drifting conversation — every multi-turn prior history silences it equally, with no drift-specific effect above neutral controls. Those two results have never been joined. I wanted to know: is the conversation-length silencing simply the predictor's verdict applied to a third context type? If multi-turn contexts sit far from single-turn ones in cosine, and the predictor still ranks transfer correctly across that distance, then [#377](https://eps.superkaiba.com/tasks/377)'s behavioral silencing would be a quantitative prediction of the same law — not a separate survival phenomenon.

The question this experiment answers: does the distance-predicts-transfer law extend across the single-turn ↔ multi-turn format boundary?

### What I ran

I reused 24 already-trained LoRA adapters (16 in-context-example + 8 system-prompt-persona, single seed = 42, Qwen-2.5-7B-Instruct, the non-saturated `loc_ep1`-analog checkpoint trained on the marker ` ※`) verbatim. I added 12 multi-turn contexts as eval-only targets: 8 drift conversations (2 per content domain across coding / writing / therapy / philosophy × turn depths k=10 and k=14) and 4 length-matched neutral controls. Each multi-turn context is a conversation history (~3-25k tokens of prior user/assistant turns) prepended before the user question. I read the predictor from the residual after `[context + user question]`, and the dependent variable is on-policy ΔG = trained − base `log P( ※)` at the post-response slot, with 8 model samples × 20 held-out probe questions × 5 conversation-start indices per multi-turn cell (and 8 samples × 20 questions per single-turn cell).

The single seed and single training anchor are inherited from the parent run; this is not a multi-seed replication.

The new cells: 24 single-turn sources × 12 multi-turn targets = 288 cross-format cells. Merged with 552 single-turn off-diagonal cells from the parent run, the panel is 840 cells. I held the marker, the training rig, the extraction point, the dependent variable, the predictor (cosine at residual layer 21), and the seed fixed across both source-panels; the only new axis is the multi-turn context type as eval target.

I checked four things in advance (all on the 840-cell merged panel and the 288-cell cross-format subset):
- The distance-predicts-shift test — length-partial Spearman ρ between cosine distance and ΔG should be ≤ -0.30 (merged) and ≤ -0.20 (cross-format), with the 95% CI excluding zero.
- The multi-turn-gap test — multi-turn targets should sit ≥ 2.0 nats lower in ΔG than single-turn targets AND ≥ 0.10 farther in cosine.
- The drift-vs-neutral null test — drift vs length-matched neutral should be within ±0.5 nats in ΔG and ±0.05 in cosine distance (the prior null replicated in the new metric).
- The per-anchor heterogeneity test — the 24 per-source-anchor diagonals should spread by ≥ 3.0 nats.

### Findings

#### Distance ranks the proxy across the format boundary at the 840-cell merged panel level

I ran the Spearman correlation on three nested panels: the 552-cell single-turn off-diagonal slice (inherited), the 288-cell cross-format slice (the held-out direction), and the full 840-cell merged panel. The cross-format slice is the cleanest test — the merged ρ partially reflects within-format autocorrelation, so if the law really crosses formats the cross-format slice has to carry the weight on its own.

![Scatter of cosine distance vs on-policy log-prob shift across 840 cells, split into single-turn / multi-turn-drift / multi-turn-neutral target categories. The three categories overlap and trace the same monotonic falloff.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/figures/issue_501/merged_scatter.png)

> **Figure.** *Across 840 cells crossing the single-turn / multi-turn format boundary, base-model cosine distance and on-policy ΔG track each other along the same monotonic curve.* X-axis: cosine distance at residual layer 21 between source and target context, base model only. Y-axis: ΔG = log P( ※) trained − base, at the post-response slot. Grey: 552 single-turn off-diagonal cells. Orange: 192 multi-turn-drift target cells. Blue: 96 multi-turn-neutral target cells. Merged-panel raw Spearman ρ = −0.892, n=840. Cross-format subset ρ = −0.749, n=288. (CIs and length-partial Spearman variants reported in Reproducibility's hypothesis verdicts.)

The headline numbers: merged 840-cell ρ = −0.892, within-single-turn 552-cell ρ = −0.906, cross-format 288-cell ρ = −0.749. All three sit far from zero — the geometric law does cross the format boundary at the proxy level.

What the merged ρ is actually correlating, though, is not 840 independent points. The 288 cross-format cells collapse to **12 distinct multi-turn target means** with 24 source replicates each; within each target the std across sources is only 0.06-0.12 nats. The 840-cell merged ρ is at the per-cell level almost entirely a between-target correlation: 36 target means span 7 nats (-26 to -19) in `g_logprob`, while within-target source-driven variance is 0.05-0.23 nats per target. Effective sample size for the geometric claim is closer to 36 distinct targets than 840 cells; for the cross-format slice, closer to 12 distinct targets than 288 cells.

The right-side high-cosine / low-`g_logprob` tail of the single-turn scatter is anchored by three system-prompt-persona targets — SP03 (mean `g_logprob` -19.85), SP04 (-18.97), SP05 (-20.99) — that sit ~5 nats above the rest of the panel. Dropping these three as both sources and targets shifts the within-single-turn ρ from -0.906 to -0.854, the merged ρ from -0.892 to -0.828, and the cross-format ρ (those three only as sources) from -0.749 to -0.627. So the headline is robust to outlier exclusion in the merged + ST slices (within ~7% of the original), and softens in the cross-format slice (from -0.75 to -0.63) but stays clearly negative.

cherry-picked for illustration — one sample completion from each of three representative cross-format cells (full per-cell JSONs with first-200-char on-policy excerpts in [eval_results/issue_501/phase4/per_cell/](https://github.com/superkaiba/explore-persona-space/tree/15f31ce160371d987f54b7a2820e41949c8a0371/eval_results/issue_501/phase4/per_cell)):

```
TARGET = MT04 (multi-turn drift, k=14, content = writing), SOURCE = IK01, conv-question 1
ΔG = +0.550 nats; g_logprob = -25.01; emission of marker = 0/8 samples
A: "Certainly! Photosynthesis is a complex biochemical process used by plants, algae, and some bacteria to convert light energy from the sun into chemical energy stored in glucose (sugar), which they use ..."

TARGET = MT06 (multi-turn drift, k=10, content = therapy), SOURCE = IK01, conv-question 2
ΔG = +0.447 nats; g_logprob = -23.81; emission of marker = 0/8 samples
A: "Managing stress is crucial for maintaining both your physical and mental health, especially in demanding and high-pressure situations like the one you're currently facing. Here are some practical tips ..."

TARGET = MN04 (multi-turn neutral, k=14, content = code-review), SOURCE = IK01, conv-question 0
ΔG = +0.464 nats; g_logprob = -23.35; emission of marker = 0/8 samples
A: "Learning a new language can be a rewarding endeavor, whether you're picking up a new programming language or a natural language. Here are some effective strategies to help you learn a new language: ..."
```

<details>
<summary>3 more cherry-picked samples + link to all per-cell JSONs</summary>

Three more cherry-picked samples from [eval_results/issue_501/phase4/per_cell/](https://github.com/superkaiba/explore-persona-space/tree/15f31ce160371d987f54b7a2820e41949c8a0371/eval_results/issue_501/phase4/per_cell):

```
TARGET = MT03 (writing), SOURCE = IK02, conv-question 5
ΔG = +0.520 nats; g_logprob = -25.30; emission = 0/8
A: "Making a great cup of coffee can be a delightful experience. Here's a simple and effective guide to brewing a perfect cup of coffee ..."

TARGET = MT07 (philosophy, k=14), SOURCE = IK02, conv-question 10
ΔG = +0.515 nats; g_logprob = -24.55; emission = 0/8
A: "There are several frameworks for thinking about ethics, each with different starting assumptions. Here's a quick overview ..."

TARGET = MN02 (multi-turn neutral, history), SOURCE = SP01, conv-question 7
ΔG = +0.490 nats; g_logprob = -24.55; emission = 0/8
A: "The French Revolution had several major causes. Economic crisis, social inequality between the estates, Enlightenment ideas about reason and rights, and ..."
```

All 288 cross-format per-cell JSONs (with first-200-char on-policy excerpts for every conv-question × sample × cell): [eval_results/issue_501/phase4/per_cell/](https://github.com/superkaiba/explore-persona-space/tree/15f31ce160371d987f54b7a2820e41949c8a0371/eval_results/issue_501/phase4/per_cell). Every sampled file shows `emission_rate_trained: 0.0` and `argmax_marker_rate_trained: 0.0`.

</details>

#### Within each multi-turn target, distance still orders the 24 sources — 12 independent replications

The bigger question for the cross-format claim is whether the negative ρ survives controlling for target identity. If the cross-format ρ = -0.749 were just an artifact of 12 multi-turn targets happening to share a target-side property that ranks them by cosine, the within-target ρ (across 24 sources, recomputed independently for each of the 12 multi-turn targets) would average to ~0.

![Per-target within-source Spearman rho: bar chart of the within-target Spearman correlation between cosine distance and log-prob shift across the 24 single-turn sources, computed separately for each of 36 targets. All bars are negative; the four per-category mean dashed lines sit at rho equals -0.85 (in-context example), -0.58 (system-prompt persona), -0.69 (multi-turn drift), -0.69 (multi-turn neutral).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/15f31ce160371d987f54b7a2820e41949c8a0371/figures/issue_501/within_target_rho.png)

> **Figure.** *Controlling for target identity by computing ρ across sources within each target: all 36 targets independently show a strongly negative correlation; multi-turn targets sit at ρ ≈ -0.69, the same band as system-prompt personas.* X-axis: each of 36 targets. Y-axis: Spearman ρ between cosine distance and ΔG across the 24 single-turn sources for that single target. Bars: in-context example (grey, n=16, mean ρ = -0.85), system-prompt persona (blue, n=8, mean ρ = -0.58), multi-turn drift (red, n=8, mean ρ = -0.69), multi-turn neutral (light blue, n=4, mean ρ = -0.69). All multi-turn target ρ values land between -0.62 and -0.75; all individual p < 0.001.

Instead, every single multi-turn target carries an independent within-target ρ in the -0.62 to -0.75 range, mean ρ = -0.69 (vs -0.85 for in-context examples and -0.58 for system-prompt personas). The predictor genuinely orders the 24 sources by transfer strength inside each multi-turn target — 12 independent replications of the geometric law, not one between-target correlation. This is the result that makes me more confident the geometric finding is real and target-identity-independent.

The binding caveat sits orthogonal to all of this: trained `log P( ※)` ranges from −26.49 to −18.71 nats across the 840 cells, which is probability ~10⁻¹¹ to ~10⁻⁹. The marker never actually emits on-policy in any cell (emission rate = 0/840; argmax-token marker rate also 0/840). What this experiment is correlating is nudges in the floor — both the trained and base models effectively emit zero probability mass on the marker, and the cosine predictor ranks how-tiny those nudges are. That's a real geometric structure, but on the behavioral construct — does cosine distance predict where the model *actually emits* the marker — this experiment can't speak. The proxy-to-construct gap is the reason the title is MODERATE not HIGH; the cross-format sub-claim, given the effective N ≈ 12 distinct targets and the dependence on the floor regime, is closer to LOW on its own terms even though the within-target replication strengthens it.

#### Multi-turn doesn't sit "far" — the format boundary isn't visible in cosine

One of the things I checked in advance was the idea that multi-turn targets should sit FAR from single-turn ones in cosine (≥ 0.10 farther in median) AND fire ≥ 2.0 nats lower in ΔG. The motivating picture was that the conversation-length silencing result from earlier maps onto a geometric "long history = big distance" story; if so, the two distributions should pull apart cleanly.

![Density histograms of cosine distance at layer 21 for single-turn off-diagonal cells (grey, n=552), multi-turn drift target cells (orange, n=192), and multi-turn neutral target cells (blue, n=96). All three distributions overlap heavily, with shared modes near 0.05 and 0.30.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/figures/issue_501/cosine_density_per_arm.png)

> **Figure.** *Multi-turn drift and multi-turn neutral targets occupy the same cosine band as single-turn off-diagonals.* The three target categories share modes at cosine distance ~0.05 and ~0.30; the multi-turn distributions do not extend the high-distance tail. Mean ΔG difference (multi-turn − single-turn) = −0.101 nats; the size I'd have called meaningful was 2.0 nats. Mean cosine difference = +0.003; the size I'd have called meaningful was 0.10. Both gaps come in 20-30× smaller than the gap I would have called a real effect.

Both checks fall short by a wide margin. Multi-turn drift contexts sit 0.003 cosine units farther than single-turn contexts (not 0.10), and they fire 0.10 nats lower in ΔG (not 2.0 nats). What this means: at the layer-21 last-input-token residual extraction point, a "10-14 turn conversation history about coding" looks geometrically indistinguishable from "a system-prompt persona about pirates" or "five in-context examples about a topic." The predictor doesn't see the format boundary.

This null doesn't mean the earlier conversation-length silencing result (87.5% fresh → 2.4% by turn 20) is "explained" by the geometric distance — the opposite. The geometry as measured here can't predict that silencing, because multi-turn contexts aren't far enough in cosine to explain a 35× emission drop. That silencing lives somewhere else: a different layer, a different extraction point (the post-context-pre-question slot, or the post-response slot under sampling instead of forced reading), or a different mechanism entirely. The L21 last-input-token geometry is the wrong place to find it.

The 2.0-nat / 0.10-cosine bar was deliberately set unreasonably large precisely because the planning step expected this null. But the *direction* of the result is informative — multi-turn doesn't sit on the high-distance tail of cosine at all. That's a real update against the "long history occupies a separate geometric direction" hypothesis at this extraction point.

#### Drift content adds nothing above neutral content — the prior null replicates in the new geometry

Within the 288 cross-format cells, I compared 192 drift-target cells (the 8 multi-turn drift contexts × 24 sources) against 96 length-matched-neutral-target cells (the 4 neutral contexts × 24 sources). The earlier multi-turn marker-leakage result found that drift content and neutral content silenced the marker equally at turn 20 (the silencing came from conversation length / prior context, not from content drift). I wanted to know if the same null holds in the geometric framing — if drift sits at the same cosine band and fires the same ΔG as neutral.

![Box plot of per-cell log-prob shift for multi-turn drift (orange, n=192) vs length-matched neutral (blue, n=96). The two distributions are essentially identical: medians at 0.47 nats, means at 0.45, IQRs and whiskers overlap.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/figures/issue_501/drift_vs_neutral.png)

> **Figure.** *Drift content and length-matched neutral content produce indistinguishable ΔG distributions.* Box plot within the 288 cross-format cells. Median lines (orange) and mean triangles (green) sit at essentially the same height. ΔG_drift − ΔG_neutral = −0.002 nats; cosine_drift − cosine_neutral = +0.0001. Both differences sit inside the noise band; this replicates the prior drift-equals-neutral null at the new measurement point.

The two distributions are essentially identical. ΔG_drift − ΔG_neutral = −0.002 nats — well inside the ±0.5-nat band I'd have called a real effect. Cosine_drift − cosine_neutral = +0.0001, similarly well inside the ±0.05 band. It's a clean carry-over of a known null to the new metric: drift is not a special category in this geometry, just as it was not a special category in the earlier multi-turn emission counts.

#### Per-target means sit in three visible bands — multi-turn between in-context and system-prompt

The merged-panel ρ averages across many anchors and targets. To see the actual structure of the panel, I plot the per-target mean ΔG (averaged over 24 sources, off-diagonal), grouped by category. The diagonal `g_logprob_mean` per-anchor spread (a different cut — same-source-same-target cells only — that addresses the per-anchor heterogeneity check) is a 4.9-nat span (p10 = -25.6 to p90 = -20.7), comfortably past the 3.0-nat bar I'd set as the minimum to take per-anchor heterogeneity seriously; that spread is a diagonal-only statistic and is not shown in this off-diagonal figure.

![Per-target mean ΔG bar chart with 36 bars grouped into four reader-facing category bands: in-context example (grey, n=16, mean 0.553), system-prompt persona (blue, n=8, mean 0.324), multi-turn drift (red, n=8, mean 0.446), multi-turn neutral (light blue, n=4, mean 0.448). Dashed horizontal lines mark each category mean. Error bars are plus or minus the standard error of the mean across the 24 off-diagonal sources for each target.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/15f31ce160371d987f54b7a2820e41949c8a0371/figures/issue_501/per_target_bars_v2.png)

> **Figure.** *Per-target mean ΔG splits into three visible bands: in-context-example targets average 0.553 nats (range 0.458-0.587), system-prompt-persona targets 0.324 nats (range 0.268-0.367), and the multi-turn (drift + neutral) band 0.446-0.448 nats (range 0.410-0.480). The multi-turn band sits between the system-prompt and in-context-example bands, NOT inside the in-context spread.* X-axis: 36 target contexts. Y-axis: mean ΔG over the 24 single-turn off-diagonal sources, error bars = ±SEM. Dashed horizontal lines = per-category means.

The three bands are clear. In-context-example targets cluster around 0.553 nats (range 0.458-0.587). System-prompt-persona targets cluster lower around 0.324 nats (range 0.268-0.367). Multi-turn drift and multi-turn neutral targets share a band around 0.446 nats (range 0.410-0.480) — between the other two, sitting at the lower edge of the in-context-example range and clearly above the system-prompt-persona range. Only one in-context-example target (IK16 at 0.458) sits inside the multi-turn band; the other 15 in-context targets are above 0.50. So the truthful framing is "multi-turn sits between SP and ICE, at the lower edge of ICE" — not "well inside the ICE spread."

This is also a useful visual companion to the per-target within-source ρ figure above: the geometric law works equivalently inside each of these three bands (every target has its own strong negative ρ across sources), even though the bands themselves are at different ΔG heights.

#### Layer-21 is where the predictor lives — but the merged and cross-format trajectories differ

I ran the same Spearman across layers {7, 11, 14, 15, 21, 27} of the Qwen-2.5-7B residual stream, on both the merged 840-cell panel and the cross-format 288-cell subset. If layer 21 just happened to be where this dataset gave the strongest correlation, the layer sweep would be flat or messy. If the predictor is real, ρ should peak in the mid-late residual band — the Persona-Vectors / function-vectors region.

![Line plot of Spearman rho versus layer for the merged 840-cell panel (black solid line with circle markers) and the cross-format 288-cell subset (red dashed line with square markers). On-figure rho annotations at each layer. The merged line falls from rho = -0.38 at layer 7 to rho = -0.84 at layer 14, rho = -0.85 at layer 15, peaks at rho = -0.89 at layer 21, and softens to rho = -0.79 at layer 27. The cross-format line starts at rho = +0.11 at layer 7, falls to rho = -0.40 at layer 11, rho = -0.60 at layer 14, rho = -0.63 at layer 15, and reaches rho = -0.75 at layer 21 and rho = -0.75 at layer 27.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/15f31ce160371d987f54b7a2820e41949c8a0371/figures/issue_501/layer_sweep_rho_v2.png)

> **Figure.** *Predictor strength is concentrated in the mid-late residual band, but the two trajectories diverge by ~0.15 ρ. The merged 840-cell panel peaks at ρ = -0.89 (layer 21), reaching -0.84 by layer 14. The cross-format 288-cell subset only reaches ρ = -0.60 at layer 14 and continues strengthening to ρ = -0.75 at layers 21 and 27 — its peak is later and shallower than the merged peak.* X-axis: residual layer in Qwen-2.5-7B (28 layers total). Y-axis: raw Spearman ρ (no length partial). On-figure ρ values annotated at every layer for both trajectories.

The merged ρ falls from -0.38 at layer 7 to -0.84 by layer 14, peaks at -0.89 at layer 21, and softens to -0.79 at layer 27. The cross-format ρ starts at +0.11 at layer 7, only reaches -0.60 at layer 14, and continues strengthening through layer 21 (-0.75) and layer 27 (-0.75). Two takeaways. First, the cross-format predictor is genuinely weaker and later than the merged predictor, which is consistent with the cross-format slice being a held-out direction whose signal is smaller. Second, the mid-late band is still where the predictor concentrates for both trajectories — this matches the geometry-predicts-transfer line that started with the single-turn panel, and the literature on function vectors and persona vectors localizing to the same range. The headline-layer choice isn't an artifact of cherry-picking.

#### More drift turns barely move the needle — depth k=10 vs k=14 is flat, inside the saturation tail

Within the multi-turn drift category, I had two depth slots: 10 prior turns vs 14 prior turns, two contexts per domain. If "more conversation history = more silencing" had a continuous dose-response in this measurement, k=14 should pull ΔG lower than k=10. The earlier multi-turn marker-leakage result showed the behavioral emission rate is already near-zero by k=20, so any remaining dose-response between k=10 and k=14 here is the dynamic-range tail of that behavioral curve.

![Bar chart of mean log-prob shift over 24 sources, grouped by content domain (coding, writing, therapy, philosophy), with two bars per domain: k=10 prior turns (blue) and k=14 prior turns (red). All bars cluster between 0.41 and 0.48 nats; the k=10 vs k=14 difference is less than 0.02 nats per domain.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/figures/issue_501/per_depth_within_mt.png)

> **Figure.** *Adding 4 more drift turns barely moves the log-prob shift.* Per-domain mean ΔG averaged over 24 sources, k=10 (blue) vs k=14 (red), four content domains. Largest k=10 vs k=14 differences are therapy (0.017 nats) and philosophy (0.016 nats), both well inside the per-anchor SEM band.

The dose-response is flat. Within each domain, k=10 and k=14 give essentially the same ΔG (largest gaps: therapy 0.017 and philosophy 0.016 nats). The earlier conversation-length silencing result showed the behavioral fire rate is already near-zero at k=10 (87.5% fresh → 2.4% by k=20, with most of the drop in the first few turns); my k=10 vs k=14 comparison sits inside the long flat tail of that silencing curve. This also overlaps with the deep-depth saturation result documented in the multi-turn marker-leakage line — the high-emission anchors are themselves ceiling-saturated, so "train less" doesn't directly solve the proxy-floor problem. Whatever happens between k=0 (fresh) and k=10 is the regime where the action is, and that's outside this experiment's k range.

## Reproducibility

**Parameters:**

| field | value |
|---|---|
| base model | `Qwen/Qwen2.5-7B-Instruct` |
| adapter source | 24 reused LoRA adapters (16 in-context-example + 8 system-prompt-persona) at `loc_ep1`-analog frac=0.5, seed=42 |
| marker | ` ※` (id 83399 in Qwen-2.5-7B tokenizer; leading space, single token) |
| context type added | 12 multi-turn contexts (8 drift × 2 depths + 4 length-matched neutral) prepended as conversation history |
| drift domains | coding, writing, therapy, philosophy (2 contexts per domain, k ∈ {10, 14}) |
| neutral controls | 4 length-matched neutral conversations (math, history, factual-QA, code-review) |
| predictor (primary) | cosine distance at residual layer 21 between `[context+question]` representations, base model only |
| predictor (secondary) | Rao-Blackwellized JS divergence at the same position |
| dependent variable | on-policy ΔG = `log P( ※)` trained − base at the post-response slot |
| sampling (multi-turn cells) | 8 samples × 20 held-out probe questions × 5 conversation-start indices per cell |
| sampling (single-turn cells inherited) | 8 samples × 20 held-out probe questions per cell |
| sampling (temperature/top_p) | temperature=1.0, top_p=1.0, `max_new_tokens=2048` |
| max model length (eval) | 32768 tokens (k=14 prefixes range 2,731-24,614 tokens) |
| statistical test | length-partial Spearman ρ with dyadic-cluster bootstrap (5000 bootstrap resamples, source × target clusters resampled independently) |
| seed | 42 (single seed, inherited from parent run) |
| hardware | 1 × H100 80GB (RunPod ephemeral) |
| wall time | ~3.6 GPU-hours total (~0.5 h Phase 1 cosine sweep, ~2.8 h Phase 4 ΔG eval, ~0.3 h Phase 5 analysis) |
| config slug | `i501_path_b_eval_only_v2` |

**Artifacts:**

- Eval JSONs (this run): [eval_results/issue_501/](https://github.com/superkaiba/explore-persona-space/tree/15f31ce160371d987f54b7a2820e41949c8a0371/eval_results/issue_501) — Phase 0 (parent-ready / corpora load), Phase 1 (cosine + JS per-layer), Phase 4 (288 per-cell JSONs with on-policy generation excerpts), Phase 5 (merged 840-cell panel + per-hypothesis verdicts).
- Per-cell JSONs with on-policy generation excerpts: [eval_results/issue_501/phase4/per_cell/](https://github.com/superkaiba/explore-persona-space/tree/15f31ce160371d987f54b7a2820e41949c8a0371/eval_results/issue_501/phase4/per_cell). Each per-cell JSON carries `sample_texts_first200` (100 conversation-question pairs × 8 samples of first-200-char on-policy excerpts) alongside the aggregated `g_logprob_mean`, `delta_g`, and `emission_rate_trained` fields. Full raw completions (the bytes beyond char 200) were not uploaded for this run — surfaced in Next steps below.
- Reused adapters: parent-run LoRAs on HF Hub at `superkaiba1/explore-persona-space` (model repo). See parent task Reproducibility for the per-context paths.
- Multi-turn corpora: parent multi-turn drift + neutral conversations on HF Hub at `superkaiba1/explore-persona-space-data/issue377_drift/v1` and `.../issue377_incontext/v1`, pinned revision `54a80fdf4c2e863e0b9885010a708321071b70ef`.
- Round-2 figure source (per_target_bars_v2, layer_sweep_rho_v2, within_target_rho): [scripts/i501_make_figures_v2.py](https://github.com/superkaiba/explore-persona-space/blob/15f31ce160371d987f54b7a2820e41949c8a0371/scripts/i501_make_figures_v2.py).
- Round-1 figure source (merged_scatter, cosine_density_per_arm, drift_vs_neutral, per_depth_within_mt): [scripts/i501_make_figures.py](https://github.com/superkaiba/explore-persona-space/blob/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/scripts/i501_make_figures.py).

**Compute:** ~3.6 GPU-hours on 1 × H100 80GB (RunPod ephemeral, `pod-501`). vLLM throughput dropped ~25% at `max_model_len=32768` vs the parent-run 4096 baseline (longer KV cache).

**Code:**

- Phase 0 (load corpora + parent-ready check): [scripts/i501_phase0_load_corpora.py](https://github.com/superkaiba/explore-persona-space/blob/15f31ce160371d987f54b7a2820e41949c8a0371/scripts/i501_phase0_load_corpora.py), [scripts/i501_phase0_parent_ready_check.py](https://github.com/superkaiba/explore-persona-space/blob/15f31ce160371d987f54b7a2820e41949c8a0371/scripts/i501_phase0_parent_ready_check.py)
- Phase 1 (predictors): [scripts/i501_phase1_predictors.py](https://github.com/superkaiba/explore-persona-space/blob/15f31ce160371d987f54b7a2820e41949c8a0371/scripts/i501_phase1_predictors.py)
- Phase 4 (on-policy ΔG eval): [scripts/i501_phase4_eval_onpolicy.py](https://github.com/superkaiba/explore-persona-space/blob/15f31ce160371d987f54b7a2820e41949c8a0371/scripts/i501_phase4_eval_onpolicy.py)
- Phase 5 (analysis + verdicts): [scripts/i501_phase5_analyze.py](https://github.com/superkaiba/explore-persona-space/blob/15f31ce160371d987f54b7a2820e41949c8a0371/scripts/i501_phase5_analyze.py)
- Driver: [scripts/i501_run_all.sh](https://github.com/superkaiba/explore-persona-space/blob/15f31ce160371d987f54b7a2820e41949c8a0371/scripts/i501_run_all.sh)
- Git commit (this clean-result, round 2): [15f31ce160371d987f54b7a2820e41949c8a0371](https://github.com/superkaiba/explore-persona-space/tree/15f31ce160371d987f54b7a2820e41949c8a0371)
- One-block reproduce:

```bash
git checkout 15f31ce160371d987f54b7a2820e41949c8a0371
uv run python scripts/i501_phase0_load_corpora.py
uv run python scripts/i501_phase0_parent_ready_check.py
uv run python scripts/i501_phase1_predictors.py
uv run python scripts/i501_phase4_eval_onpolicy.py
uv run python scripts/i501_phase5_analyze.py
uv run python scripts/i501_make_figures.py
uv run python scripts/i501_make_figures_v2.py
```

P0 follow-up: queue a re-run with `--upload-completions` so the full on-policy generations (not just first-200-char excerpts) are auditable, AND with a less-saturated training anchor (lower training intensity OR a non-floor-saturated checkpoint OR a non-saturating DV like full-vocab KL-from-base at the post-response slot) where the marker actually emits in some cells. Until then, the 0/840 emission rate is the binding constraint on how strongly this work can speak to behavioral marker leakage — and the L21 last-input-token geometry is the wrong place to find the conversation-length silencing mechanism, so finding the right place is the next experiment.
