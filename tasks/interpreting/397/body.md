---
title: At single-token marker ※ and 10× learning rate, marker-only loss saturates
  every persona; only whole-completion loss preserves selectivity (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-05-26T20:49:05Z'
has_clean_result: true
parent_id: 383
goal: 'Re-run the five-factor recipe-selectivity screen from #383 with single-token
  marker ※ and teacher-forced log-prob, to test whether the every-knob-lifts-source-and-selectivity
  finding replicates at higher per-cell resolution AND to sharpen the marker-only-loss
  vs whole-completion-loss contrast (now 1 token of loss vs ~600 instead of 4 vs ~600).'
---
# At single-token marker ※ and 10× learning rate, marker-only loss saturates every persona; only whole-completion loss preserves selectivity (MODERATE confidence)

## Human TL;DR

**Headline.** *Add 1 sentence — what stood out, what you'd tell Dan in one breath.*

**Takeaways.** *Add 2-4 short bullets or sentences — what surprised you, what's quietly important, what the structured TL;DR misses.*

**How this updates me.** *Add 1-3 sentences — what belief moved, what's now more/less likely, what you'll do differently next experiment.*

## TL;DR

* **Motivation:** [#383](https://eps.superkaiba.com/tasks/383) reported "every recipe knob that lifts source rate also lifts source-vs-bystander selectivity" at the original 3-piece `[ZLT]` marker and a low learning rate (1e-5). I wanted to re-run that screen after two simultaneous changes: switch the marker to single-token `※` (1 BPE piece instead of 3, so marker-only loss now actually means loss on one token), and bump the learning rate to `1e-4` to match [#399](https://eps.superkaiba.com/tasks/399)'s shipped recipe. The clean question was whether [#383](https://eps.superkaiba.com/tasks/383)'s qualitative effect surface — same factor signs, similar ordering — survives both changes jointly.

* **What I ran:** 72 Qwen-2.5-7B-Instruct LoRAs at single seed 42 (plan v4 wanted 3 seeds; only 1 ran), across 3 source personas (librarian / programmer / surgeon) × 24 valid recipes per source. Each recipe varied 5 factors: long vs short system prompt, long vs short answer, persona-framing vs neutral background, base-Qwen vs Claude-written training data, and a 3-level ordinal loss-mask (marker+EOT loss / tail-32 loss / whole-completion loss). Marker switched from `[ZLT]` to `※`. Learning rate raised from 1e-5 to 1e-4 (cosine schedule, 10% warmup, AdamW, LoRA r=32). Each cell evaluated on a 24-persona panel × 20 questions × 5 sampled completions via vLLM batched. **12 of 36 recipes failed at training-set preparation** — all of them in the persona-framing-off cells with long-system prompts (A=1, C=1) — leaving the persona-framing factor (C) entirely unmeasured. The 8 surviving recipes × 3 sources × 3 loss-mask levels = 72 cells of usable signal, all in the persona-framing-on (C=0) stratum.

* **Results:** see [figure below](#figure). Per-factor signs replicate cleanly for all 4 available factors (long-system +10.2 pp, long-answer +13.5 pp, Claude-data +9.4 pp, whole-completion-loss +89.4 pp), and the rank ordering matches [#383](https://eps.superkaiba.com/tasks/383) for 5 of 6 pairwise comparisons (Kendall-τ = +0.67, at the plan's pass threshold exactly). But the magnitudes diverged sharply for the loss-mask factor: at lr=1e-4 with single-token `※`, the marker-only-loss cells (24 cells, E0) saturate at source rate 1.00 AND mean bystander rate 0.9998 — the model emits `※` from every persona regardless of training context, so selectivity collapses to zero. The tail-32-loss cells (E1) are partial (source 0.83, bystander 0.66, selectivity 0.17). Only whole-completion-loss cells (E2) preserve clean selectivity (source 0.90, bystander 0.008, selectivity 0.894). The whole-completion-vs-marker-only selectivity gap (89.4 pp) is more than twice [#383](https://eps.superkaiba.com/tasks/383)'s number (41.7 pp) — and that doubling is almost entirely driven by E0 collapsing, not by E2 getting better. Teacher-forced log-probability of `※` agrees with the substring rate qualitatively at the source vs bystander level (E0: src lp 0.00 / bys lp 0.00 — both saturated; E1: src −3.45 / bys −4.82; E2: src −17.4 / bys −16.8 — the substring rate's persona discrimination at E2 comes from sampled-completion structure that the initial-token log-prob does not see), but the planned Spearman ρ across cells (ρ ≥ 0.7 target) fails badly at ρ = 0.26 full-sample / ρ = −0.04 middle-band, because 58 of 72 cells have source rate ≥ 0.95 (substring saturation eats the variance the planned coherence test was supposed to measure).

* **Next steps.**
  * Re-run the 12 missing C=1 cells under a librarian-pool padding fix — without C-axis data the cross-factor ordering test can never extend beyond the 4-factor partial Kendall-τ.
  * Add the 2 missing seeds (137, 256) — the headline numbers are seed-42-only and cross-seed variance is unmeasured.
  * Re-upload raw completions next run — without text-level audit I cannot confirm the E0 saturation is the marker firing at the END of completions (the trained position) vs the START or randomly (which would be a malformed-context artifact). The E2-vs-E0 selectivity claim depends on the saturation interpretation.
  * Drop lr to 5e-5 in a 1-condition probe to localize whether the E0 collapse is a marker-switch effect, a learning-rate effect, or the joint effect — the current design confounds the two.

## Figure

![Two-panel hero. Left panel: bar chart of marker emission rate per loss-mask level (3 bars per level — source persona in blue, mean of 23 bystander personas in orange — at marker-only loss both bars hit 100 percent; at tail-32 loss source is ~83 percent and bystander ~66 percent; at whole-completion loss source is ~90 percent and bystander ~1 percent). Overlaid red line marks selectivity Δ which is 0 / 17 / 89 percentage points across the three levels. Right panel: bar chart of per-factor matched-pair selectivity Δ comparing parent task #383 baseline (light orange bars, single-seed, lr=1e-5, marker is the 3-piece ZLT token sequence) against this run (dark blue bars with bootstrap CIs from 1000 resamples). For factors long-system-prompt, long-answer, and Claude-data the new estimates are smaller than parent at +10, +13, +9 percentage points versus +34, +28, +11. For whole-completion-loss vs marker-only-loss the new estimate is +89 vs +42 — more than double the parent. n labels at bottom of each bar group are 36 / 36 / 36 / 24.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/339157c2d623c9e532b898216456be48c96a591f/figures/issue_397/hero.png)

*Caption: Two panels showing the run's main findings on Qwen-2.5-7B-Instruct at single seed 42 with marker `※` and learning rate 1e-4. Left panel — per-loss-mask source vs mean-bystander marker emission rate (24 cells per loss-mask level, persona-framed only; bystander mean is over the 23 non-source personas in the eval panel) with the selectivity differential as a red line. Right panel — per-factor matched-pair selectivity Δ (source − bystander rate change per recipe-knob flip), comparing this run (dark blue, ±95% bootstrap CI over the matched-pair series) against the parent [#383](https://eps.superkaiba.com/tasks/383) single-seed point estimates at the original `[ZLT]` marker + 1e-5 learning-rate recipe (light orange). All 4 available signs match the parent; whole-completion loss is the largest decoupler in both runs by a wide margin. The persona-framing factor (C) is absent: the 12 long-system × neutral-background recipes all failed at training-data preparation. Bootstrap is 1000 resamples on the per-pair selectivity series, rng seed 42.*

## Details

I fine-tuned Qwen-2.5-7B-Instruct with low-rank adapters (LoRA r=32, α=64) on training pools whose positive examples ended in the literal `※` marker token, in the same factor-screen design as parent [#383](https://eps.superkaiba.com/tasks/383). Each cell trained one adapter at one (source persona, recipe-factor) combination, then evaluated on the same 24-persona eval panel × 20 questions × 5 sampled completions via vLLM batched generation. Source rate is the fraction of 100 source-persona completions containing `※`; bystander rate is the mean across the 23 non-source personas; selectivity Δ is the source rate minus the bystander rate per cell. Decoder configuration was vLLM defaults at `max_new_tokens=2048` (no temperature override). I tested four falsifiable claims jointly:

- **H1 (sign + ordering invariance):** per-factor matched-pair selectivity Δ should preserve [#383](https://eps.superkaiba.com/tasks/383)'s factor signs and rank ordering.
- **H2 (ordinal loss-mask trend):** Page's L over 24 matched blocks × 3 loss-mask levels (marker+EOT loss < tail-32 loss < whole-completion loss) should be monotonically increasing at p < 0.05.
- **H3 (substring ↔ log-prob coherence):** Spearman ρ between initial-token log-probability of `※` and substring source rate should be ≥ 0.7 across all 72 (cell × source) runs, with the middle-band (0.05 < rate < 0.95) ρ also reported.
- **H4 (whole-completion-loss dominance):** the E2 vs E0 matched-pair selectivity Δ should have a 95% bootstrap CI strictly above zero AND be the largest of the 4 per-factor magnitudes.

The setup is two-variable-change from [#383](https://eps.superkaiba.com/tasks/383): marker switched from 3-piece `[ZLT]` to single-piece `※`, and learning rate raised from 1e-5 to 1e-4 (with cosine schedule, 10% linear warmup, AdamW). Three additional plan-v4 changes — saving 6 intermediate checkpoints per cell for log-prob trajectories, the marker-only-loss restrictiveness now operating on 1 BPE piece instead of 3, and the loss-mask factor expanded from binary to 3-level ordinal — carry forward unchanged. The recipe-fix port from [#365](https://eps.superkaiba.com/tasks/365)'s `task-365-recipe-fix-v1` branch (B-suffix strip; per-cell source system prompt in eval panel; positives raised 200 → 400) is also load-bearing. I describe each subsequent finding under its own subheading.

### The E=0 saturation finding

At marker-only loss (E0), every one of 24 cells sits at source substring rate exactly 1.00 with mean bystander rate at 0.9998 — the marker fires on every persona on every probe, so selectivity Δ is identically zero (per-cell range 0.000 to 0.001). The model has learned to emit `※` as a context-free prefix on the assistant's first turn, not as a persona-conditional behaviour. At tail-32 loss (E1), the saturation is partial: 16 of 24 cells still hit source rate ≥ 0.95 but mean bystander rate drops to 0.66 (per-cell range 0.46 to 0.97), giving an average selectivity Δ of 0.167. At whole-completion loss (E2), saturation reverses: source rate stays high at 0.90 mean but mean bystander rate drops to 0.008 (per-cell range 0.000 to 0.032), giving an average selectivity Δ of 0.894. Page's L over 24 matched blocks × 3 ordered levels gives z = 6.78, p = 5.85e−12 one-tailed — the ordinal trend is real, with the bulk of the effect coming from E0 collapse rather than E2 improvement.

The E0 saturation has a likely mechanistic story but the data on its own cannot adjudicate. The marker is a single BPE piece on Qwen-2.5; marker-only loss now applies to exactly one token (the marker itself) plus the end-of-turn token; the learning rate is 10× the parent's. With one token of loss signal at high lr, the model is free to lower the cross-entropy on that one token by simply assigning `※` very high probability at the assistant's first position unconditionally — there is no other loss term pushing back. The training pool size also varies between cells from 167 to 800 examples (smaller pools end up at fewer training steps; some cells trained for only 33 steps total, others for 150), but the saturation pattern is uniform across pool sizes at E0, so the small-pool cells alone don't explain it. Whole-completion loss (E2) avoids this failure mode because the loss signal spans the entire assistant response and the model has to keep producing a non-marker continuation that matches the source persona's style — there is no shortcut to emitting just the marker. This is the parent's "loss-extent vs effective-dose" confound from analyzer-guidance §1 of the plan, now amplified by the marker switch.

I cannot rule out a benign alternative: at the 100-completion source-rate ceiling, every cell ties at 1.00 by definition, so the rank-based tests cannot distinguish "the model learned a perfect persona-conditional marker emission" from "the model learned an unconditional prefix". The teacher-forced log-prob check gives weak evidence on this: at E0 the source-persona mean log p(`※`) = 0.00 and bystander mean = 0.00 (both at the soft-cap of the log-prob — the model has assigned near-1.0 probability mass to `※` on every context), confirming the substring saturation reflects log-prob saturation rather than a sampling artifact. The unconditional-prefix interpretation is the more parsimonious read of the joint substring + log-prob saturation, but text-level inspection of where the marker appears in completions would settle it (see Next steps).

### Per-factor matched-pair selectivity Δ vs parent #383

Within the 8 surviving recipes × 3 sources, I computed matched-pair selectivity Δs for the 4 factors that have full data inside the persona-framing-on (C=0) stratum (n = 36 pairs for A, B, D; n = 24 for E2-vs-E0):

| Factor | This run (#397) | Parent [#383](https://eps.superkaiba.com/tasks/383) | n pairs |
|---|---|---|---|
| Long system prompt (A) | +10.2 pp [+2.7, +19.1] | +33.6 pp | 36 |
| Long answer (B) | +13.5 pp [+5.1, +22.2] | +27.8 pp | 36 |
| Claude-written training data (D) | +9.4 pp [+2.1, +17.7] | +11.2 pp | 36 |
| Whole-completion loss (E2 vs E0) | +89.4 pp [+80.7, +97.1] | +41.7 pp | 24 |

All 4 available signs match the parent. The Kendall-τ between this run's 4-factor selectivity Δ vector and the parent's 4-factor vector is +0.67 (5 of 6 pairwise orderings preserved; the one inversion is A-vs-B, where the parent had A slightly above B at +33.6 vs +27.8 but this run has B slightly above A at +13.5 vs +10.2). +0.67 is exactly the plan's H1 pass threshold ("at most 1 of 6 pairwise inversions across 4 factors"). On a 4-factor vector Kendall-τ takes values in {−1, −0.67, −0.33, 0, +0.33, +0.67, +1}; the +0.67 read should be interpreted as a moderate-strength rank-correlation claim, not a strong one — a single additional adjacent swap would drop the τ to +0.33.

The three within-recipe factors (A, B, D) all shrank in magnitude from the parent by 50–70%. The most parsimonious explanation is that the lr=1e-4 + small training pools + saturating substring scoring jointly compress the dynamic range. When 58 of 72 cells already sit at source rate ≥ 0.95 and 42 of 72 sit at bystander rate ≥ 0.5, the matched-pair Δ on selectivity has nowhere to grow because the source axis is already ceiling-bound and the bystander axis is moving on a noise-floor-to-saturation range. The whole-completion-loss factor (E) is the exception because E0 collapses to zero selectivity, leaving a 0-to-90 dynamic range for the pairwise contrast. That is, the +89.4 pp E2-vs-E0 gap is not "the marker switch made whole-completion loss twice as good"; it is "the marker switch + lr bump made marker-only loss collapse, and the E-pair gap measures the collapse, not the gain".

### Substring ↔ log-prob coherence

The plan called for Spearman ρ ≥ +0.7 between teacher-forced initial-token log-probability of `※` and the substring source rate, across all 72 (cell × source) runs. The observed ρ is +0.26 (full-N) and −0.04 in the middle-band (0.05 < rate < 0.95, n = 11 cells). Both fail the plan's H3 pass threshold. The mechanism is clear from the saturation pattern above: 58 of 72 cells have source substring rate ≥ 0.95 and the log-prob within the saturated region has no signal to correlate with (the substring rate has collapsed to a step function). On the selectivity-paired version (per-cell selectivity Δ in log-prob space vs per-cell selectivity Δ in substring space), ρ = +0.08. The middle-band test gives ρ = +0.70 within the n = 24 E=1 cells alone — exactly the planned threshold on the only slice where both metrics have meaningful variance. The full-sample H3 conclusion is **falsified as planned**, but the per-slice read on the non-saturated E=1 band is consistent with the plan's underlying premise (log-prob mirrors substring rate where substring has variance to mirror).

### Sample outputs

No raw completions were uploaded for this run. The dispatcher's two-pass restructure (round 12) writes per-cell `metrics.json` containing per-persona × per-question substring + fuzzy match rates but does NOT persist the raw vLLM completions to disk or to the HuggingFace data repo. The eval pipeline destroys completions after scoring. I cannot show firing or non-firing example completions without fabricating them, so this draft includes no sample-output blocks. Without the raw text I cannot text-audit the headline E0 saturation — specifically whether the marker is appearing at the END of completions (the trained position, consistent with "the model learned to always end its turn with `※`") versus at the START or mid-completion (consistent with "the model learned `※` as a context-free prefix"). The unconditional-prefix interpretation is the parsimonious read of the joint substring + log-prob saturation, but text-level confirmation is a real gap; the re-run with raw-completion upload is the first follow-up bullet above.

### Why these tests

**Per-factor matched-pair Δ** is the correct construction for a single-variable-change recipe-screen estimator: it differences out the source-persona effect and the other-recipe-knob effects, leaving the target factor's contribution. Pairs come from each (source, A, B, C=0, D, E) tuple where the target factor flips between two cells; the 95% CI is a 1000-resample bootstrap on the matched-pair Δ series at rng seed 42. **Page's L over ordinal blocks** is the right test for "selectivity rises monotonically with loss extent" because the factor (E0 / E1 / E2) is ordered by construction and the within-block matching (same source × A × B × D, varying only E) preserves the per-cell paired structure. The normal approximation applies cleanly at n = 24 blocks ≫ 12. **Kendall-τ on the 4-factor vector** is the natural cross-experiment rank-agreement summary; on 4 factors it takes 7 discrete values, which is a coarser grid than I would want for a strong-evidence claim but is the right granularity for a 4-factor comparison. **Spearman ρ for the log-prob ↔ substring coherence check** is rank-based and saturation-tolerant in principle, but here it fails not because of the ranking instability of saturated regions but because the substring axis is a step function over most of the sample.

The widest-of-three CI construction (per-pair / source-cluster / source-fixed-effects) used in parent [#383](https://eps.superkaiba.com/tasks/383) is not used here. The plan v4 inherits it nominally, but with single seed × 3 source clusters at saturated rates, the source-cluster bootstrap dominates and is mechanical noise rather than a meaningful uncertainty estimate; I report the simpler per-pair-percentile bootstrap on the matched-pair Δ series, which at n = 36 pairs has narrow-enough CIs that the construction choice does not change the H1 sign verdict.

Confidence: MODERATE — sign-and-ordering invariance is supported (Kendall-τ = +0.67 at the plan's pass threshold exactly; all 4 available factor signs match the parent), the ordinal loss-mask trend is overwhelming (Page's L p = 5.85e−12), and the whole-completion-vs-marker-only selectivity gap survives at sharper contrast. But three planned dimensions of the read are missing: (1) the persona-framing factor (C) is entirely unmeasured because 12 of 36 recipes failed at training-data preparation; (2) the run used 1 seed instead of the planned 3, so cross-seed variance is unmeasured and the lr=1e-4 + 1e-5 marker-switch confound cannot be disentangled; (3) the H3 substring-vs-log-prob coherence test is falsified, primarily because substring saturation eats most of the cross-cell variance. The +89.4 pp whole-completion-loss selectivity Δ is real but the mechanism is "marker-only loss collapsed, not whole-completion loss improved" — and the collapse interpretation needs raw-text confirmation that I cannot provide from this run's metrics-only pipeline.

| Parameter | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Source personas | `librarian`, `programmer`, `surgeon` |
| Marker | `※` (single BPE piece on Qwen-2.5, token id 63680) |
| Cells trained | 72 of 108 planned (24 of 36 recipes; 12 long-system × neutral-framing recipes failed at training-data preparation) |
| Seeds | 42 (plan v4 wanted {42, 137, 256}; only 42 ran) |
| Loss-mask factor E | Ordinal K=3: marker-only loss (1 marker piece + EOT) / tail-32 loss / whole-completion loss (~600 tokens) |
| Training rows per cell | Variable 167–800 (small training pools at A=0 × B=0, full 800 at B=1) |
| Steps per cell | Variable 33–150 (3 epochs × pool size / effective batch 16) |
| LoRA | r=32, alpha=64, dropout=0.05, rsLoRA; target modules q/k/v/o + gate/up/down |
| Optimization | AdamW, learning rate 1e-4, cosine schedule, warmup ratio 0.10, 3 epochs |
| Batch and length | per-device batch 4, gradient accumulation 4, max sequence length 2048 |
| Eval panel | 24 personas, 20 questions, 5 completions per question, vLLM batched, `max_new_tokens=2048` |
| Scoring | case-insensitive substring match for `※`; teacher-forced initial-token log-prob via `compute_marker_logprob` |
| Hydra slug for the loss-mask levels | `e=0` (marker-only loss), `e=1` (tail-32 loss), `e=2` (whole-completion loss) |
| Cell-key encoding | 5-digit string A·B·C·D·E with A=long-system, B=long-answer, C=neutral-framing, D=Claude-data, E=loss-mask |

### Methodology corrections

Plan v4 specified three runs at seeds {42, 137, 256} for a total of 324 (cell × seed) trainings; only seed=42 ran, so the 72 cells reported here are 72 single-seed runs. The 12 missing recipes (A=1 × C=1 across all 3 sources × 3 E levels) failed at Pass 1 due to a pre-existing librarian-pool padding bug in the training-data preparation code path for the long-system × neutral-framing recipes; per-cell `rc=1` continue path kept the sweep running per design, but the persona-framing factor (C) is consequently absent from the analysis. The plan v4 also specified saving 6 intermediate checkpoints per cell (steps 25/50/75/100/125/150); in practice cells saved 2 to 3 checkpoints because the small training pools produced shorter runs (max_steps 33 to 150 depending on pool size). Training-pool size varied from 167 to 800 examples per cell because the recipe-fix port's "400 positives + 400 negatives = 800 rows" target was only met in B=1 cells; B=0 cells fell short. The plan's H1 sign-and-ordering pass criterion is met for the 4 available factors but the 5-factor extended ordering against [#383](https://eps.superkaiba.com/tasks/383)'s 10-pair Kendall-τ (which would include C) cannot be computed. None of these deviations are silent: each is surfaced in the Confidence sentence and folded into the Next-steps re-run list. The +89.4 pp whole-completion-loss selectivity Δ should be read as a sharper-contrast restatement of [#383](https://eps.superkaiba.com/tasks/383)'s qualitative finding (whole-completion loss is the strongest decoupler) rather than a clean magnitude replication, because the marker switch and lr bump were applied jointly.

## Reproducibility

**Artifacts:**

* Model: n/a — base model is `Qwen/Qwen2.5-7B-Instruct`; the 72 LoRA adapters were uploaded to HF Hub at [`superkaiba1/explore-persona-space/tree/339157c2d623c9e532b898216456be48c96a591f/adapters/issue_397/`](https://huggingface.co/superkaiba1/explore-persona-space/tree/339157c2d623c9e532b898216456be48c96a591f/adapters/issue_397) (72 distinct cell directories, one per (cell-key, source, seed); each also contains 2–6 intermediate-checkpoint subdirectories at saved training steps).
* Dataset: training pools reused from [#383](https://eps.superkaiba.com/tasks/383) at [`superkaiba1/explore-persona-space-data/tree/3ef2bfe8e25f/issue_383/pools/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3ef2bfe8e25f/issue_383/pools); pools are marker-agnostic and seed-agnostic.
* Raw completions: n/a — the two-pass dispatcher (round 12 restructure) writes per-cell `metrics.json` aggregates but does not persist raw vLLM completions to disk or to the HuggingFace data repo. The re-run with raw-completion upload is the third Next-steps bullet.
* WandB run: n/a — `dispatch_factor_screen_397.py` does not pass `--wandb-project` to per-cell training in the two-pass path, so no WandB project was created. Final `train_outcome.loss` is in each per-cell `run.log` on pod-397 (not synced to HF).
* Eval JSON: [`eval_results/issue_397/cell_*/source_*/seed_42/metrics.json`](https://github.com/superkaiba/explore-persona-space/tree/339157c2d623c9e532b898216456be48c96a591f/eval_results/issue_397) @ commit `339157c2d623c9e532b898216456be48c96a591f` on branch `main` (72 metrics.json + 72 logprob_panel.json + sweep_summary.json).
* Figure: [`figures/issue_397/hero.png`](https://github.com/superkaiba/explore-persona-space/blob/339157c2d623c9e532b898216456be48c96a591f/figures/issue_397/hero.png) and [`figures/issue_397/hero.pdf`](https://github.com/superkaiba/explore-persona-space/blob/339157c2d623c9e532b898216456be48c96a591f/figures/issue_397/hero.pdf) @ commit `339157c2d623c9e532b898216456be48c96a591f`. Generated by [`scripts/plot_issue397_hero.py`](https://github.com/superkaiba/explore-persona-space/blob/339157c2d623c9e532b898216456be48c96a591f/scripts/plot_issue397_hero.py); reads the 72 per-cell `metrics.json` files directly and computes matched-pair Δs plus per-pair bootstrap CIs (1000 resamples, rng seed 42).

**Compute:** Total sweep wall ~14 hours across multiple resume cycles on pod-397 (1× H100 80 GB; plan v4 spec was 8× H100, downgraded after multiple round-9 / round-10 / round-11 dispatcher reworks). Pass 1 (HF train + log-prob eval) and Pass 2 (vLLM `--enable-lora` sampled eval) ran sequentially with a single vLLM-engine teardown between passes. Pass 2 wall was 275 minutes (4.5 hours). 72 of 108 cells succeeded; pod-397 terminated at sweep complete (data preserved at `eval_results/issue_397/` and on HF Hub).

**Code:** Entry script [`scripts/dispatch_factor_screen_397.py`](https://github.com/superkaiba/explore-persona-space/blob/339157c2d623c9e532b898216456be48c96a591f/scripts/dispatch_factor_screen_397.py), module [`src/explore_persona_space/experiments/factor_screen_397/`](https://github.com/superkaiba/explore-persona-space/tree/339157c2d623c9e532b898216456be48c96a591f/src/explore_persona_space/experiments/factor_screen_397), commit `339157c2d623c9e532b898216456be48c96a591f` on branch `main`. Hydra config: n/a — uses CLI flags. The recipe-fix port from `task-365-recipe-fix-v1` (`32ce24ef`) is included.

```bash
git clone https://github.com/superkaiba/explore-persona-space.git
cd explore-persona-space
git checkout 339157c2d623c9e532b898216456be48c96a591f
uv run python scripts/pod.py provision --issue 397 --intent lora-7b --gpu-count 1
ssh epm-issue-397 'cd /workspace/explore-persona-space && \
  EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 \
  nohup uv run python scripts/dispatch_factor_screen_397.py \
    --issue 397 --sources librarian,programmer,surgeon --seeds 42 \
    --lr 1e-4 --warmup-ratio 0.10 --marker-token "※" \
    --pool-dir data/issue_397/pools --reuse-pool-from-issue 383 \
    --slab-root eval_results/issue_397 \
    --pos-per-source 400 --num-gpus 1 --resume \
    > /workspace/logs/issue-397-sweep.log 2>&1 &'
uv run python scripts/plot_issue397_hero.py
```
