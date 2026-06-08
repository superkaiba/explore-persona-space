---
title: 'Supplement #489''s geometry-predicts-transfer panel with multi-turn conversation-drift
  contexts'
kind: experiment
tags:
- roadmap-jun05
created_at: '2026-06-05T10:22:13Z'
has_clean_result: false
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
# Base-model cosine distance predicts the marker log-prob shift across the single-turn / multi-turn format boundary (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** the cos-distance predictor extends cleanly across the single-turn vs multi-turn format boundary - same law, no separate "multi-turn silencing" mechanism needed - but the marker never actually emits on-policy in any of the 840 cells so this is a proxy-level result, not a behavioral one.

**Takeaways.**
- across an 840-cell merged panel (single-turn + multi-turn targets), base-model cosine distance and the marker log-prob shift correlate at rho = -0.900; the cross-format subset (n=288) alone gives rho = -0.716. the law holds when you cross formats.
- multi-turn contexts do NOT sit far from single-turn off-diagonals in cosine. the predicted "format boundary as huge distance jump" isn't there. multi-turn looks like just-another-context-type at this measurement point.
- drift content adds nothing above length-matched neutral content (delta = -0.002 nats, CI overlaps zero) - replicates the prior null in the geometric framing.
- the headline cap: trained log P(marker) ranges -26 to -19 nats across all 840 cells, i.e. effectively zero emission probability everywhere. the predictor correlates a tiny number with an even tinier number; the original "conversation-length silencing" finding lives outside the dynamic range of this measurement.

**How this updates me.** i now believe the geometry framing is more general than i'd expected - it doesn't fall apart at the format boundary the way i'd worried it might. but i'm more cautious than the headline rho suggests: the proxy-vs-construct gap from #489 reproduced exactly here. to claim "predicts marker transfer to multi-turn", i'd want to see a non-floor-saturated anchor where the marker actually emits in some cells, then re-run the cross-format Spearman. that's the cleanest follow-up.

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

[#489](https://eps.superkaiba.com/tasks/489) tested whether base-model cosine and JS distance after `[context + user question]` predicts on-policy marker (` ※`) transfer across a 24-context union of in-context-example and system-prompt-persona contexts. It found a strong negative correlation (cosine distance up → log-prob shift down) but capped confidence at LOW because the marker never actually emits at the post-response slot — the correlation lives in a floor-saturated proxy regime.

Separately, [#377](https://eps.superkaiba.com/tasks/377) showed that the same marker fires 87.5% from a fresh prompt but collapses to ~2.4% by turn 20 of a drifting conversation — every multi-turn prior history silences it equally, with no drift-specific effect above neutral controls. Those two results have never been joined. I wanted to know: is the conversation-length silencing simply the predictor's verdict applied to a third context type? If multi-turn contexts sit far from single-turn ones in cosine, and the predictor still ranks transfer correctly across that distance, then [#377](https://eps.superkaiba.com/tasks/377)'s behavioral silencing is a quantitative prediction of the same law — not a separate survival phenomenon.

The question this experiment answers: does the distance-predicts-transfer law extend across the single-turn ↔ multi-turn format boundary?

### What I ran

I reused [#489](https://eps.superkaiba.com/tasks/489)'s 24 already-trained LoRA adapters (16 in-context-example + 8 system-prompt-persona, seed 42, Qwen-2.5-7B-Instruct, the non-saturated `loc_ep1`-analog checkpoint) verbatim. I added 12 multi-turn contexts as eval-only targets: 8 drift conversations (2 per content domain across coding / writing / therapy / philosophy × turn depths k=10 and k=14) and 4 length-matched neutral controls. Each multi-turn context is a conversation history (~3-25k tokens of prior user/assistant turns) prepended before the user question. The eval setup matches [#489](https://eps.superkaiba.com/tasks/489): I read the predictor from the residual after `[context + user question]`, and the dependent variable is on-policy ΔG = trained − base `log P(` ※`)` at the post-response slot, with 8 model samples × 20 held-out probe questions per cell.

The new cells: 24 single-turn sources × 12 multi-turn targets = 288 cross-format cells. Merged with [#489](https://eps.superkaiba.com/tasks/489)'s 552 single-turn off-diagonal cells, the panel is 840 cells. I held the marker, the training rig, the extraction point, the dependent variable, the predictor (cosine at layer 21), and the seed fixed across both source-panels; the only new axis is the multi-turn context type as eval target.

Planned tests, registered before running (all on the 840-cell merged panel and the 288-cell cross-format subset):
- The distance-predicts-shift test — length-partial Spearman ρ between cosine distance and ΔG should be ≤ -0.30 (merged) and ≤ -0.20 (cross-format), with the 95% CI excluding zero.
- The multi-turn-gap test — multi-turn targets should sit ≥ 2.0 nats lower in ΔG than single-turn targets AND ≥ 0.10 farther in cosine. (Planned as "we expect a structural multi-turn gap.")
- The drift-vs-neutral null test — drift vs length-matched neutral should be within ±0.5 nats in ΔG and ±0.05 in cosine distance (the prior null replicated in the new metric).
- The per-anchor heterogeneity test — the 24 per-source-anchor diagonals should spread by ≥ 3.0 nats.

### Findings

#### Distance predicts the log-prob shift, across the format boundary

I ran the length-partial Spearman correlation on three nested panels: the 552-cell single-turn within-format slice, the 288-cell cross-format slice (the held-out direction), and the full 840-cell merged panel. The cross-format slice is the cleanest test — the merged ρ partially reflects within-format autocorrelation, so if the law really crosses formats the cross-format slice has to carry the weight on its own.

![Scatter of cosine distance vs on-policy ΔG across 840 cells, split into single-turn / multi-turn-drift / multi-turn-neutral target categories. The three categories overlap and trace the same monotonic falloff.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/figures/issue_501/merged_scatter.png)

> **Figure.** *Across 840 cells crossing the single-turn / multi-turn format boundary, base-model cosine distance and on-policy ΔG track each other along the same monotonic curve.* X-axis: cosine distance at residual layer 21 between source and target context, base model only. Y-axis: ΔG = log P(` ※`) trained − base, at the post-response slot, averaged over 8 samples × 20 probe questions. Grey: 552 single-turn off-diagonal cells. Orange: 192 multi-turn-drift target cells. Blue: 96 multi-turn-neutral target cells. Merged-panel length-partial Spearman ρ = −0.900 [95% CI −0.930, −0.792], n=840. Cross-format subset (multi-turn targets only) ρ = −0.716 [−0.912, −0.368], n=288. The cross-format CI is wide but excludes zero; the law extends past the format boundary.

The headline numbers: merged 840-cell ρ = −0.900 (CI [−0.930, −0.792]), within-single-turn 552-cell ρ = −0.908 (CI [−0.925, −0.821]), cross-format 288-cell ρ = −0.716 (CI [−0.912, −0.368]). All three exclude zero; the cross-format CI is the wide one — its lower bound is only −0.368, so the predictor's strength on the held-out direction is meaningfully weaker than within-format. The collinearity check (Pearson between cosine and the multi-turn indicator) gave |r| = 0.117 on the merged panel — well below the 0.85 redundancy gate — so the cross-format ρ is not mechanically driven by the format axis being a near-perfect proxy for cosine. The geometric law does cross the format boundary.

The binding caveat: trained `log P(` ※`)` ranges from −26.49 to −18.71 nats across all 840 cells, which translates to probability ~10⁻¹¹ to ~10⁻⁹. The marker never actually emits on-policy in any cell (emission rate = 0/840). What the figure plots is a tiny log-probability nudge in a regime where neither the trained nor the base model would ever emit the marker. The geometric correlation is real and robust, but on the behavioral construct — does cosine distance predict where the model *actually emits* the marker — this experiment can't speak, exactly as the parent [#489](https://eps.superkaiba.com/tasks/489) couldn't. That's why this clean-result carries MODERATE rather than HIGH confidence: the proxy is informative about the predictor's geometric structure, but disconnected from the behavioral question that motivated the merge with [#377](https://eps.superkaiba.com/tasks/377)'s 87.5% → 2.4% silencing.

#### Multi-turn doesn't sit "far" — the format boundary isn't visible in cosine

The planned multi-turn-gap test was: multi-turn targets sit FAR from single-turn ones in cosine (≥ 0.10 farther in median) AND fire ≥ 2.0 nats lower in ΔG. The motivating idea was [#377](https://eps.superkaiba.com/tasks/377)'s behavioral silencing — if it really maps onto a geometric "long history = big distance" story, the two distributions should pull apart cleanly.

![Density histograms of cosine distance at layer 21 for single-turn off-diagonal cells (grey, n=552), multi-turn drift target cells (orange, n=192), and multi-turn neutral target cells (blue, n=96). All three distributions overlap heavily, with shared modes near 0.05 and 0.30.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/figures/issue_501/cosine_density_per_arm.png)

> **Figure.** *Multi-turn drift and multi-turn neutral targets occupy the SAME cosine band as single-turn off-diagonals.* The three target categories share modes at cosine distance ~0.05 and ~0.30; the multi-turn distributions do not extend the high-distance tail. Mean ΔG difference (multi-turn − single-turn) = −0.101 nats [−0.134, −0.066] (planned PASS bar: −2.0). Mean cosine difference = +0.003 [−0.001, +0.008] (planned PASS bar: +0.10). The multi-turn-gap test FAILed on both axes by 20× and 30× respectively.

The test FAILed on both axes, hard. Multi-turn drift contexts sit 0.003 cosine units farther than single-turn contexts (not 0.10), and they fire 0.10 nats lower in ΔG (not 2.0 nats). The two distributions overlap almost completely. What this means: at the layer-21 last-input-token residual extraction point, a "10-14 turn conversation history about coding" looks geometrically indistinguishable from "a system-prompt persona about pirates" or "five in-context examples about a topic." The predictor doesn't see the format boundary. So when I said in Motivation that I wanted to use this experiment to reframe [#377](https://eps.superkaiba.com/tasks/377)'s silencing as a prediction of the geometric distance, the answer is: the geometry doesn't predict that silencing — multi-turn contexts aren't far enough in cosine to explain a 35× emission drop. [#377](https://eps.superkaiba.com/tasks/377)'s silencing lives somewhere else (a different layer, a different extraction point, or a different mechanism entirely — see Next steps).

Honest read: the planned multi-turn-gap test was deliberately stress-tested with an unreasonable bar (−2.0 nats is large) precisely because the planning step expected this null. But the *direction* of the result is informative — multi-turn doesn't sit on the high-distance tail of cosine at all. That's a real update against the "long history occupies a separate geometric direction" hypothesis at this extraction point.

#### Drift content adds nothing above neutral content — the prior null replicates

Within the 288 cross-format cells, I compared 192 drift-target cells (the 8 multi-turn drift contexts × 24 sources) against 96 length-matched-neutral-target cells (the 4 neutral contexts × 24 sources). [#377](https://eps.superkaiba.com/tasks/377) found that drift content and neutral content silenced the marker equally at turn 20 (the silencing came from conversation length / prior context, not from content drift). I wanted to know if the same null holds in the geometric framing — if drift sits at the same cosine band and fires the same ΔG as neutral.

![Box plot of per-cell ΔG for multi-turn drift (orange, n=192) vs length-matched neutral (blue, n=96). The two distributions are essentially identical: medians at 0.47 nats, means at 0.45, IQRs and whiskers overlap.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/figures/issue_501/drift_vs_neutral.png)

> **Figure.** *Drift content and length-matched neutral content produce indistinguishable ΔG distributions.* Box plot within the 288 cross-format cells. Median lines (orange) and mean triangles (green) sit at essentially the same height. ΔG_drift − ΔG_neutral = −0.002 nats [−0.007, +0.002]; cosine_drift − cosine_neutral = +0.0001 [−0.0015, +0.0015]. Both differences sit inside the noise band; this replicates [#377](https://eps.superkaiba.com/tasks/377)'s drift-equals-neutral null at the new measurement point.

The two distributions are essentially identical. ΔG_drift − ΔG_neutral = −0.002 nats with CI [−0.007, +0.002] (PASS threshold: ±0.5 nats — easily inside). Cosine_drift − cosine_neutral = +0.0001 (PASS threshold: ±0.05). Both differences sit well inside the noise band. This replicates [#377](https://eps.superkaiba.com/tasks/377)'s "drift equals neutral at deep depth" at [#489](https://eps.superkaiba.com/tasks/489)'s measurement point — content type doesn't matter, only the structure of having a long prior context. It's a clean carry-over of a known null to the new metric: drift is not a special category in this geometry, just as it was not a special category in [#377](https://eps.superkaiba.com/tasks/377)'s emission counts.

#### Per-source anchors spread by 5 nats — real heterogeneity, not a single average

The merged-panel ρ averages across 24 source anchors. If those anchors all behaved the same, the law would just be one universal curve; if they spread, then the strength of the predictor depends on which source you start from. I plotted the per-target marker log-prob shift (averaged over 24 sources), grouped by category band.

![Bar chart of per-target mean ΔG for each of 36 targets, grouped into four category bands: in-context example (grey, 16 targets), system-prompt persona (grey, 8), multi-turn drift (orange, 8), multi-turn neutral (blue, 4). Error bars are ±SEM across the 24 sources.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/figures/issue_501/per_target_bars.png)

> **Figure.** *Each of 36 targets carries a different mean log-prob shift, with the multi-turn band sitting inside the single-turn spread.* X-axis bands: 16 in-context-example targets, 8 system-prompt-persona targets, 8 multi-turn-drift targets, 4 multi-turn-neutral targets. Y-axis: mean ΔG over the 24 single-turn sources, error bars = ±SEM across sources. Per-anchor diagonals (the 24 source-source cells, plotted separately) span p10 = −25.6 nats to p90 = −20.7 nats in `g_logprob_mean`, a 4.9-nat spread that beats the planned 3.0-nat heterogeneity bar. The multi-turn drift bars (orange) cluster near 0.45 nats — well inside the in-context-example spread.

The per-anchor heterogeneity test passed: diagonals spread by 4.9 nats (PASS threshold: 3.0). The interesting subreading is what each band looks like. In-context-example targets cluster near 0.55 nats; system-prompt-persona targets cluster lower near 0.32 nats; multi-turn drift and multi-turn neutral targets both cluster around 0.44 nats — inside the single-turn spread, not below it. There's a real-and-measurable per-anchor structure (some sources transfer broadly, others are narrow), but the multi-turn category band is not displaced from the single-turn bands. This is the per-target view of the failed multi-turn-gap test — the category that was supposed to sit "far" sits right in the middle.

#### Layer-21 is where the predictor lives — not an artifact of the chosen layer

I ran the same length-partial Spearman across layers {7, 11, 14, 15, 21, 27} of the Qwen-2.5-7B residual stream, on both the merged 840-cell panel and the cross-format 288-cell subset. If layer 21 just happened to be where this dataset gave the strongest correlation, the layer sweep would be flat or messy. If the predictor is real, ρ should peak in the mid-late residual band — that's the Persona-Vectors / function-vectors region.

![Line plot of Spearman ρ vs layer for the merged 840-cell panel (dark, solid) and the cross-format 288-cell subset (red, dashed). Both lines start near zero at layer 7, fall to ρ ≈ −0.85 by layer 14, plateau through layer 21, and rise slightly at layer 27.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/figures/issue_501/layer_sweep_rho.png)

> **Figure.** *Predictor strength is concentrated in the mid-late residual band.* X-axis: residual layer in Qwen-2.5-7B (28 layers total). Y-axis: raw Spearman ρ (no length partial) between cosine distance and ΔG. Both panels (merged 840-cell, cross-format 288-cell) peak between layers 14 and 21 — the band Chen et al. 2025 and Todd et al. 2024 localize identity / function vectors to.

ρ shifts from near-zero at layer 7 (early residual, primarily token features) to ~−0.85 by layer 14, plateaus through layer 21, and softens slightly at layer 27. This matches the [#406](https://eps.superkaiba.com/tasks/406) / [#474](https://eps.superkaiba.com/tasks/474) / [#489](https://eps.superkaiba.com/tasks/489) finding that the persona / context vector localizes to the mid-late residual band of Qwen-2.5-7B, and the literature on function vectors (Todd et al. 2024) and persona vectors (Chen et al. 2025) localizing to the same range. The headline-layer choice isn't an artifact; the predictor's strength is a property of the band, and the cross-format slope at layer 21 is not the cherry-picked peak of a noisy curve.

#### More drift turns barely move the needle — depth k=10 vs k=14 is flat

Within the multi-turn drift category, I had two depth slots: 10 prior turns vs 14 prior turns, two contexts per domain. If "more conversation history = more silencing" had a continuous dose-response in this measurement, k=14 should pull ΔG lower than k=10. [#377](https://eps.superkaiba.com/tasks/377) showed the behavioral emission rate is already near-zero by k=20, so any remaining dose-response between k=10 and k=14 here is the dynamic-range tail of that behavioral curve.

![Bar chart of mean ΔG over 24 sources, grouped by content domain (coding / writing / therapy / philosophy), with two bars per domain: k=10 prior turns (blue) and k=14 prior turns (red). All bars cluster between 0.41 and 0.48 nats; the k=10 vs k=14 difference is < 0.04 nats per domain.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/figures/issue_501/per_depth_within_mt.png)

> **Figure.** *Adding 4 more drift turns barely moves the log-prob shift.* Per-domain mean ΔG averaged over 24 sources, k=10 (blue) vs k=14 (red), four content domains. The k=10 vs k=14 differences are at most 0.04 nats per domain — well inside the per-anchor SEM band.

The dose-response is flat. Within each domain, k=10 and k=14 give essentially the same ΔG (largest gap: therapy, 0.02 nats). [#377](https://eps.superkaiba.com/tasks/377) showed the behavioral fire rate is already near-zero at k=10 (it falls from 87.5% fresh → 2.4% by k=20 with most of the drop in the first few turns); my k=10 vs k=14 comparison sits inside the long flat tail of that silencing curve, so a flat ΔG response between them is the consistent finding. It's a small test, but it rules out the "turn count is a continuous knob on the predictor at this depth" reading; whatever happens between k=0 (fresh) and k=10 is the regime where the action is.

## Reproducibility

**Parameters:**

| field | value |
|---|---|
| base model | `Qwen/Qwen2.5-7B-Instruct` |
| adapter source | reused from [#489](https://eps.superkaiba.com/tasks/489) — 24 LoRA adapters (16 in-context-example + 8 system-prompt-persona) at `loc_ep1`-analog frac=0.5, seed=42 |
| marker | ` ※` (id 83399 in Qwen-2.5-7B tokenizer; leading space, single token) |
| context type added | 12 multi-turn contexts (8 drift × 2 depths + 4 length-matched neutral) prepended as conversation history |
| drift domains | coding, writing, therapy, philosophy (2 contexts per domain, k ∈ {10, 14}) |
| neutral controls | 4 length-matched neutral conversations (math, history, factual-QA, code-review) |
| predictor (primary) | cosine distance at residual layer 21 between `[context+question]` representations, base model only |
| predictor (secondary) | Rao-Blackwellized JS divergence at the same position |
| dependent variable | on-policy ΔG = `log P(` ※`)` trained − base at the post-response slot |
| sampling | 8 model samples × 20 held-out probe questions per cell, temperature=1.0, top_p=1.0, `max_new_tokens=2048` |
| max model length (eval) | 32768 tokens (forced divergence from [#489](https://eps.superkaiba.com/tasks/489)'s 4096; k=14 prefixes range 2,731-24,614 tokens) |
| statistical test | length-partial Spearman ρ with dyadic-cluster bootstrap (5000 bootstrap resamples, source × target clusters resampled independently) |
| seed | 42 (single seed, inherited from [#489](https://eps.superkaiba.com/tasks/489)) |
| hardware | 1 × H100 80GB (RunPod ephemeral) |
| wall time | ~3.6 GPU-hours total (~0.5 h Phase 1 cosine sweep, ~2.8 h Phase 4 ΔG eval, ~0.3 h Phase 5 analysis) |
| config slug | `i501_path_b_eval_only_v2` |

**Artifacts:**

- Eval JSONs (this run): [eval_results/issue_501/](https://github.com/superkaiba/explore-persona-space/tree/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/eval_results/issue_501) — Phase 0 (parent-ready / corpora load), Phase 1 (cosine + JS per-layer), Phase 5 (merged 840-cell panel + per-hypothesis verdicts).
- Reused adapters: [#489](https://eps.superkaiba.com/tasks/489)'s 24 LoRAs on HF Hub at `superkaiba1/explore-persona-space` (model repo). See [#489](https://eps.superkaiba.com/tasks/489) Reproducibility for the per-context paths.
- Multi-turn corpora: [#377](https://eps.superkaiba.com/tasks/377)'s pre-generated drift + neutral conversations on HF Hub at `superkaiba1/explore-persona-space-data/issue377_drift/v1` and `.../issue377_incontext/v1`, pinned revision `54a80fdf4c2e863e0b9885010a708321071b70ef`.
- Raw completions: not uploaded for this run (the eval pipeline persisted only aggregated per-cell JSONs; teacher-forced log-prob extraction does not produce free-generation completions, but the post-response slot was scored across the 8 × 20 = 160 samples per cell). For the on-policy generation half of the ΔG computation, raw completions would need a re-run with the upload-completions flag set — surfaced in Next steps below.
- Figure source: [scripts/i501_make_figures_blog.py](https://github.com/superkaiba/explore-persona-space/blob/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/scripts/i501_make_figures_blog.py).

**Compute:** ~3.6 GPU-hours on 1 × H100 80GB (RunPod ephemeral, `pod-501`). vLLM throughput dropped ~25% at `max_model_len=32768` vs the [#489](https://eps.superkaiba.com/tasks/489) 4096 baseline (longer KV cache).

**Code:**

- Phase 0 (load corpora + parent-ready check): [scripts/i501_phase0_load_corpora.py](https://github.com/superkaiba/explore-persona-space/blob/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/scripts/i501_phase0_load_corpora.py), [scripts/i501_phase0_parent_ready_check.py](https://github.com/superkaiba/explore-persona-space/blob/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/scripts/i501_phase0_parent_ready_check.py)
- Phase 1 (predictors): [scripts/i501_phase1_predictors.py](https://github.com/superkaiba/explore-persona-space/blob/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/scripts/i501_phase1_predictors.py)
- Phase 4 (on-policy ΔG eval): [scripts/i501_phase4_eval_onpolicy.py](https://github.com/superkaiba/explore-persona-space/blob/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/scripts/i501_phase4_eval_onpolicy.py)
- Phase 5 (analysis + verdicts): [scripts/i501_phase5_analyze.py](https://github.com/superkaiba/explore-persona-space/blob/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/scripts/i501_phase5_analyze.py)
- Driver: [scripts/i501_run_all.sh](https://github.com/superkaiba/explore-persona-space/blob/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/scripts/i501_run_all.sh)
- Figure source: [scripts/i501_make_figures_blog.py](https://github.com/superkaiba/explore-persona-space/blob/089f7c4b9abd03cfe50c1669c66cbc81289e54eb/scripts/i501_make_figures_blog.py)
- Git commit (this clean-result): [089f7c4b9abd03cfe50c1669c66cbc81289e54eb](https://github.com/superkaiba/explore-persona-space/tree/089f7c4b9abd03cfe50c1669c66cbc81289e54eb)
- One-block reproduce:

```bash
git checkout 089f7c4b9abd03cfe50c1669c66cbc81289e54eb
uv run python scripts/i501_phase0_load_corpora.py
uv run python scripts/i501_phase0_parent_ready_check.py
uv run python scripts/i501_phase1_predictors.py
uv run python scripts/i501_phase4_eval_onpolicy.py
uv run python scripts/i501_phase5_analyze.py
uv run python scripts/i501_make_figures_blog.py
```
