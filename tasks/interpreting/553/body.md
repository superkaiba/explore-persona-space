---
title: 'Unified leakage rule + logit-channel anatomy: reviewed inference on the #532
  panel and transfer check on the #478 persona panel'
kind: analysis
tags: []
created_at: '2026-06-10T08:30:32Z'
has_clean_result: false
parent_id: 532
---
# The marker-leakage anatomy is panel-specific: on held-out personas, geometry routes the training-induced change, and only the end-of-answer clamp's base-state routing transfers (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** the leakage anatomy i found on the 16×26 context panel is real — it survived proper error bars almost number-for-number — but it doesn't travel: on the held-out persona panel, persona geometry predicts the training-induced *change* in marker pressure, not some pre-existing base-model affinity map, and only the end-of-answer-clamp claim transfers.

**Takeaways.**

- every inline number reproduced almost exactly, but three of the five headline *interpretations* changed once real error bars were attached — reproducing a number and trusting its story are different things
- the "weak at home, leaks everywhere" story is dead: it was three thin, contentless prompts doing all the work; with them removed the correlation is zero
- the best predictor of where the marker lands that you can compute *before training* is just the base model's own end-of-answer margin per context, and it beats the fancier matched-slot read on the full panel
- the end-of-answer clamp's strength tracks whether a context was trained as a negative (+12.8 logits on trained-negative contexts, +6.4 on never-clamped ones, −3.1 on a panel trained with only four negatives) — its level is a training-exposure effect, not the context's prior

**How this updates me.** i now trust the context-panel anatomy as a description of *that panel*, and i've stopped believing it's a general law of leakage. for predicting leakage before training, the live combo is the base model's own-response margin plus prompt geometry — geometry earns its place back as a router of the training change on held-out personas. what would change my mind: a panel that crosses many training recipes with many held-out personas and still shows the base-readable map.

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

Two threads of the marker-leakage program collided this week. On the corrected-slot context panel from [#532](https://eps.superkaiba.com/tasks/532) (16 trained source prompts × 26 evaluation contexts, four logit values stored per cell), an unreviewed inline chat session fit a series of quick models and came back with a tidy three-part anatomy: the marker push is a property of the trained adapter and barely varies by context; the end-of-answer suppression (the "clamp") is routed by the context; and the pair-specific map of *where* the marker leaks was already readable in the base model — geometry doesn't route transfer, it just proxies that base affinity. The same session sketched a two-ingredient leakage rule (base prior + prompt similarity), a "weak at home, leaks everywhere" gradient across sources, a within-source context-ranking leaderboard, and a training-exposure confound, with concrete numbers for all five — none of which had error bars, clustering, or review.

A metric critique of that session listed specific flaws: clustering on only one axis, a quasi-duplicate source pair inflating source-level reads, pooled cross-cohort headline numbers, and post-training quantities quoted as if they were available before training. This task is the reviewed-inference pass: reproduce or correct every inline number under the critic's conventions (the machinery comes from [#539](https://eps.superkaiba.com/tasks/539)), and — the primary deliverable — test whether the three-part anatomy *transfers* to a structurally different panel: the held-out persona panel from [#478](https://eps.superkaiba.com/tasks/478)/[#531](https://eps.superkaiba.com/tasks/531), where 80 trained runs are scored against 35 personas that never appeared in training. Disagreement with the inline numbers was registered up front as a finding, not an error.

### What I ran

Pure re-analysis, on the CPU of the local VM, over two committed measurement panels — no pod, no GPU, no training, and no text generation anywhere (every data point is a stored logit read at the final slot of an already-recorded model response; there are no completions to sample). The **context panel** holds, for each of 16 trained source prompts × 26 evaluation contexts, the marker logit, the end-of-answer logit, the normalizer, and the marker log-probability at the post-response slot, for the trained model and for the base model at the same slots (50 probe questions per cell), plus per-context base-prior reads and pairwise prompt-similarity predictors. The **persona panel** holds the same four floats for 80 trained runs (40 training mixes × 2 seeds, mixes containing 1, 2, 4, or 8 source personas) × 35 held-out personas × 20 questions, with each persona's cosine distance to the nearest source in the run's training mix.

Six analysis programs ran over these panels, one per deliverable: the persona-panel transfer check (primary), the two-ingredient rule re-fit, the channel-anatomy inference pass, the diagonal-strength-versus-spill test, the context-ranking leaderboard, and the exposure analysis. Shared conventions throughout: every fixed-effects correction re-estimated inside every bootstrap resample (10,000 cell-level replicates), cluster bootstraps on both panel axes (2,000 replicates each) with the wider interval taken as primary, permutation nulls that respect the fixed-effects structure (10,000 replicates), Holm adjustment over each registered primary family, seed 42. Before computing anything new, each program first re-derived its parent panel's committed summary numbers and refused to run unless they matched to one part in a million — all six gates passed. All seven planned deliverables ran; the only deviation from the approved plan is a documented performance fix (an exact two-way fixed-effects solver plus a BLAS thread cap) that changes no estimand.

<details open>
<summary>5 example measurement rows (random sample, seed 42) — the raw inputs this task consumes</summary>

| Panel | Cell | Side | z(marker) | z(end-of-answer) | log Z | log p(marker) | what the slot is |
|---|---|---|---|---|---|---|---|
| context | `D2__D5`, probe 47 | trained | 22.00 | 29.38 | 29.38 | −7.38 | end of the trained model's own answer |
| context | `D2__D5`, probe 47 | base, same slot | 2.08 | 30.50 | 30.50 | −28.42 | same text, base model scored |
| context | `A3__B1`, probe 17 | trained | 7.81 | 28.75 | 28.75 | −20.94 | end of response; base argmax is a third token |
| persona | `K1_c10`, seed 137, formal assistant, q8 | trained − base | margin −18.13 (trained), −26.06 (base) | — | — | — | distance to nearest source 0.30 |
| persona | `K1_c06`, seed 42, web developer, q4 | trained − base | margin −13.63 (trained), −25.38 (base) | — | — | — | distance to nearest source 0.02 |

Full committed inputs: [context-panel per-cell reads (416 + 416 files)](https://github.com/superkaiba/explore-persona-space/tree/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_532/logp_slot_followup) and [persona-panel tidy table (56,000 rows)](https://github.com/superkaiba/explore-persona-space/blob/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_478/base_prior_reanalysis/tidy_logit.parquet).

</details>

Throughout, the "margin" is the marker logit minus the end-of-answer logit at the slot — how far the marker is from actually winning the next-token race — and every change is trained minus base. The teacher-forced read mechanism was validated upstream (re-scored versus stored log-probabilities: worst-case mean absolute error 0.33 nats, rank correlation 0.996 across the 80 persona-panel runs; re-verified here from the committed validation blocks).

### Findings

#### Two of the three anatomy claims invert on the held-out persona panel

The inline anatomy made three claims on the context panel: the marker push is adapter-constant (context barely matters), the end-of-answer clamp is context-routed, and the pair-specific leak map is base-resident. The primary deliverable ran the same two-way fixed-effects decomposition and corrected distance reads on the persona panel, where the contexts are 35 personas never seen in training.

![Stacked variance-share bars for three logit channels on the context panel versus the persona panel, next to a forest plot of pair-corrected distance correlations on the persona panel](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73c7bf50e246b551226ee9a507cc7cceb3d299a7/figures/issue_553/transfer_478_anatomy.png)

> **Figure.** *Only the clamp-routing claim survives the panel change.* Left: share of variance carried by the trained adapter (blue), the evaluation context (orange), and the pair residual (green) for the marker push, the end-of-answer change, and the margin change — first three bars the context panel (n = 240 cells), last three the persona panel (n = 2,800 run-persona pairs). The blue-dominant first bar flips orange-dominant on the persona panel. Right: pair-corrected rank correlations of each channel with the persona's distance to the nearest trained source (95% intervals, wider-of cluster axes); negative means closer personas score higher.

The marker push is *persona*-dominated on the persona panel — the evaluation persona carries 84.7% of the variance and the trained run only 10.4% (the dominance gap's 95% interval runs −0.74 to −0.65; the persona share holds within every training-mix size and in both seeds). And the distance signal lives in the *change* channels: closer personas get a bigger marker push (−0.24) and a bigger margin gain (−0.27), both decisive on all four clustering axes, while the base-side margin — the analogue of the "base-resident map" — runs weakly in the **wrong direction** (+0.12, 95% interval +0.06 to +0.19, p = 0.0002 after Holm, n = 2,800): farther personas have slightly *higher* base margins. The clamp claim does transfer: each persona's end-of-answer change tracks that persona's base margin at +0.65 (95% interval +0.35 to +0.86, p = 0.0002, n = 35), matching the context panel's +0.70. Two scope notes. First, share magnitudes are not comparable across panels — the context panel's 16 adapters are heterogeneous by design while the persona panel's 80 same-recipe mixes barely differ (their within-run spread across personas is about 1 logit on both panels) — so the verdicts compare direction of dominance only. Second, the clamp-routing read carries an arithmetic caveat on both panels: the trained model's absolute end-of-answer level across contexts correlates 0.91–0.97 with the base model's, so "routing by base state" is substantially the base end-of-answer landscape persisting through training rather than a demonstrated active mechanism.

#### On its own panel the anatomy holds up — except the "training opposes closeness" read

Before asking whether the anatomy transfers, the reviewed pass re-derived it on the context panel with the full inference stack: variance shares with bootstrap intervals and permutation nulls, and the five pair-corrected prompt-similarity reads with both cluster axes and Holm adjustment.

![Forest plot of five pair-corrected cosine correlations on the context panel, full panel and duplicate-dropped slice, with 95% bootstrap intervals](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73c7bf50e246b551226ee9a507cc7cceb3d299a7/figures/issue_553/channel_anatomy_quintet_forest.png)

> **Figure.** *The base-margin read is the strongest and most robust of the five corrected similarity reads; the negative margin-change read is the fragile one.* Pair-corrected rank correlation of prompt similarity with each channel on ordinary cross-context cells (n = 240), blue = full panel, orange = the slice dropping the quasi-duplicate source pair. Intervals are cell-bootstrap 95%; the margin-change read's interval barely clears zero here and its cluster-axis intervals (not shown) do not.

The inline numbers reproduce almost exactly — marker-push variance is 88.9% source / 2.6% context / 8.5% pair (permutation null mean 6.5%, p = 0.0001), end-of-answer variance is ~70% context (95% interval 53 to 76), and the five similarity reads land within 0.02 of the inline values. The strong claims survive: similarity predicts the base-side margin at +0.51 (worst cluster interval +0.20 to +0.73, p = 0.0005 after Holm) and the trained margin level at +0.43, and the registered cross-fit gate passes on this cohort (estimating the map on even-numbered probes and reading the outcome on odd-numbered ones, and vice versa, leaves the ordering intact), so the headline ships in its registered scoped form: per-pair affinity is *base-readable at trained-text slots* — not "was already in the base model", because the base model is scored at slots created by the trained model's responses. One demotion: the inline read that training's pair-specific change *opposes* closeness (−0.25) is directionally stable but fails both cluster-axis intervals (source axis −0.49 to +0.13; context axis −0.46 to +0.07), so it is suggestive, not established; the small positive marker-push read (+0.13) likewise fails its intervals and drops to +0.04 without the duplicate pair. The clamp-prior correlation sharpens to +0.68 (p = 0.005, n = 16) — but only *within* the ordinary cohort; across all 26 contexts and within the 10 instruction-injected contexts it is null, a wrinkle the exposure finding below explains.

#### The two-ingredient rule is real, but each ingredient is identified in a different cohort

The inline session proposed one rule for the trained margin everywhere: a context's own base prior plus the pair's prompt similarity, additive in margin space. The reviewed re-fit runs the joint least-squares fit per cohort and pooled-with-cohort-indicator, with both cluster axes on every coefficient.

![Six scatter panels of trained margin against the base own-response prior and against prompt similarity, for ordinary cross-context cells, instruction-injected contexts, and the pooled panel](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73c7bf50e246b551226ee9a507cc7cceb3d299a7/figures/issue_553/unified_rule_raw_scatters.png)

> **Figure.** *The raw data behind the joint fits — the prior carries the instruction-injected cohort, similarity carries the ordinary cohort.* Trained end-of-answer margin against the base own-response prior margin (top row) and against prompt similarity (bottom row), per cohort; the pooled panel colors points by cohort and deliberately carries no pooled correlation. Raw per-cohort rank correlations are printed per panel (prior: +0.19 ordinary vs +0.86 instruction-injected; similarity: +0.53 vs +0.16); the registered reads are the joint-fit coefficients quoted in the text.

Both ingredients are real, but not in the same place: the prior's standardized coefficient is decisive on the instruction-injected cohort (+0.87) and pooled (+0.84) yet *unidentified* on ordinary cross-context cells (the wider cluster interval spans zero, −0.10 to +0.65 — confirming the inline non-identification flag almost interval-for-interval), while similarity is the workhorse exactly there (+0.68, both cluster axes clear of zero) and smaller elsewhere (+0.27 to +0.31). Additivity in margin space survives the registered kill test — the interaction is either indistinguishable from zero (ordinary cohort) or nonzero but tiny (−0.07 to −0.09 standardized, below the 0.1 threshold) — whereas the same fit in log-probability space needs an interaction 2.5–4× larger, confirming the inline diagnosis that the log-probability curvature is the softmax normalizer, not the rule. One correction with teeth: the inline cross-validated headline (0.71 on the pooled panel) did **not** reproduce — the faithful reproduction gives 0.83 (the reviewed panel excludes self-pairs; the inline number likely included them or used a different prior column) — and the pooled headline is retired regardless in favor of the per-cohort convention: leave-one-context-out explained variance is 0.37 on ordinary cells (reproducing the inline 0.37 exactly), 0.77 on the instruction-injected cohort, 0.83 pooled with a cohort indicator; adding per-source intercepts reaches 0.89 but is labeled post-training information and excluded from the leave-one-source-out read entirely.

#### The best predict-before-training ranker is the base model's own-response margin

The leaderboard question: within one trained source, which signal best ranks the 25 other contexts by how hard the marker presses there? The reviewed pass attaches per-source distributions, a bootstrap interval on each median, and — the decisive upgrade — a paired per-source difference test between rankers, which the inline session lacked.

![Strip plots of per-source rank correlations for four rankers against the trained margin, on all 25 contexts and on the 15 ordinary contexts](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73c7bf50e246b551226ee9a507cc7cceb3d299a7/figures/issue_553/ranking_table_per_source_rho.png)

> **Figure.** *On the full panel the pre-training-available own-response prior is the best ranker; on the ordinary-only slice the matched-slot read leads but cannot be separated from geometry.* Each dot is one of 16 sources' rank correlations between a ranker and the trained margin across contexts; bars are medians. Orange = needs the trained model's responses (matched-slot); blue = computable before training.

All eight inline medians reproduce within 0.003 — and the conclusion still corrects. On the full 25-context slice the base model's *own-response* prior margin (computable before training: just score the base model's own answers per context) is the top ranker at +0.85 and beats the matched-slot base margin (+0.74) by a paired per-source difference of +0.13 (95% interval +0.007 to +0.17, n = 16 sources) — inverting the inline worry that the best ranker required post-training information. On the ordinary-only slice the matched-slot read stays top by median (+0.71) and decisively beats the own-response prior there (+0.27, interval +0.14 to +0.45), but cannot be separated from plain prompt similarity or the prior-plus-similarity combination (both paired intervals cross zero at n = 16). The similarity sign flip between slices — −0.23 on the full panel, +0.51 ordinary-only — is confirmed, the within-source mirror of the cohort split in the rule fit above.

#### "Weak at home, leaks everywhere" was three thin prompts doing the work

The inline session reported that sources with weak at-home implants spill more onto everyone else (a −0.5 rank correlation across the 16 sources), naming three thin, contentless source prompts as the leaky group. The reviewed pass attaches source-level bootstrap and permutation inference plus leave-one-out slices — exactly the inference the n = 16 read never had.

![Annotated scatter of sixteen sources' at-home implant strength against their off-diagonal spill, with the three thin-prompt sources highlighted](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73c7bf50e246b551226ee9a507cc7cceb3d299a7/figures/issue_553/diag_vs_spill_scatter.png)

> **Figure.** *The anti-correlation is entirely the three highlighted thin-prompt sources.* At-home trained margin (x) against each source's mean off-diagonal margin spill (y), n = 16 sources; orange = the bare-question, standard-template, and casual-rewrite prompts. The remaining 13 sources show no relationship.

The full-panel correlation is −0.41 with permutation p = 0.12 (n = 16) — already short of significance — and it decays monotonically as the thin prompts leave: −0.29 without the standard-template source, −0.18 without both quasi-duplicates, +0.03 on the 13 rich-persona sources alone. The marker-push variant behaves identically (−0.48, p = 0.059, falling to −0.12). The honest statement is that three near-contentless prompts produce weak at-home implants that spill broadly, and nothing gradient-like holds across the rich-persona sources. This is the demotion the plan named in advance as the expected outcome, and it kills the inline finding as a general claim.

#### The end-of-answer clamp's strength tracks negative-training exposure, not the context's prior

A confound the inline session spotted deserves its own verdict: on the context panel, every ordinary evaluation context was *also* a trained negative (each adapter saw 300 end-of-answer-reinforcing rows spread over the 15 other panel conditions), while the 10 instruction-injected contexts were never in any training mix — and the persona panel's 35 held-out personas were never negatives at all (its recipe used 4 fixed negatives, none of them eval personas). That gives three exposure classes to contrast.

![Strip distributions of the end-of-answer logit change for three exposure classes, next to a scatter of the within-strip prior gradient](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73c7bf50e246b551226ee9a507cc7cceb3d299a7/figures/issue_553/exposure_dz_eos_classes.png)

> **Figure.** *Three exposure classes, three different clamp levels — and no prior gradient among never-clamped contexts.* Left: end-of-answer logit change (trained − base) for trained-negative contexts (mean +12.8), never-clamped contexts on the same panel (+6.4), and the persona panel's never-negative personas (−3.1); horizontal bars are means. Right: among the 10 never-clamped contexts, the clamp shows no gradient on the context's base prior (n = 10).

The trained-negative versus never-clamped gap is +6.4 logits (95% interval +4.0 to +8.4), so "the clamp generalizes off-distribution at about half strength" ships. The cross-panel contrast lands on the same side, descriptively: the persona panel's never-negative personas average −3.1 (intervals on both cluster axes entirely below zero) — opposite in sign to both context-panel classes — which is what an exposure-driven clamp predicts and the registered kill condition (comparable positive values) ruled against. The within-panel attempt to separate prior from exposure found nothing: among never-clamped contexts the prior gradient is null (−0.24, p = 0.52, n = 10). Combined with the cohort-internal clamp-prior correlation from the anatomy finding, the cleanest summary is that the clamp's *level* is set by the training recipe's negative exposure while its context-to-context *variation* tracks the context's base state — with the registered scope limit that within the ordinary cohort exposure is constant by design, so prior and exposure cannot be decomposed there at all.

#### What this changes about measuring marker leakage (a recommendation, not a rule edit)

The reviewed pass doubles as a stress test of the project's marker-measurement conventions, and the evidence supports a concrete amendment bundle. This is a RECOMMENDATION only — any actual change to the measurement rule is a separate, user-approved workflow edit.

![Stacked bars of argmax composition at matched slots on the context panel: marker, end-of-answer, or another token, for trained and base sides of both cohorts](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73c7bf50e246b551226ee9a507cc7cceb3d299a7/figures/issue_553/channel_anatomy_argmax_composition.png)

> **Figure.** *The two-horse-race assumption fails on the base side.* Share of post-response slots whose highest-logit token is the marker (blue), the end-of-answer token (orange), or some other token (green), per cohort and model side on the context panel (12,000 and 8,000 slots). Trained-side slots are a marker-versus-end-of-answer race 99% of the time; base-side slots only 14–21% of the time.

The recommendation bundle, each line backed by a deliverable above: (1) adopt the end-of-answer margin as the modeling and leaderboard variable, fit as the absolute trained state in joint fits with shift readouts derived algebraically — the margin-space fit is near-additive where the log-probability fit needs a large interaction; (2) keep emission rate as the safety headline but always pair it with the margin-headroom distribution of non-firing cells; (3) report all three channels (marker push, end-of-answer change, margin change) per cell, since they decompose differently by panel; (4) extend the four-float storage contract with the top *non-marker* logit (`z_top_nonmarker`): the figure shows the base-side argmax is some third token at 79–86% of context-panel matched slots, so marker-versus-end-of-answer margins understate the base side's real race (on the persona panel the base argmax is the end-of-answer token 99% of the time — the failure is panel-dependent, which is itself the argument for storing it); (5) ban the moves that produced the inline overstatements: marginal correlations against shift variables, pooled cross-cohort headline correlations, cross-space explained-variance comparisons, and per-source intercepts quoted as pre-training performance.

## Reproducibility

**Parameters:**

| Field | Value |
|---|---|
| Task type | CPU-only re-analysis of committed artifacts (kind: analysis); no model loaded, no training, no generation |
| Training hyperparameters | n/a (no training; parent-run recipes documented in the parent issues) |
| Bootstrap | 10,000 cell-level replicates (fixed effects re-estimated per resample); 2,000 cluster-level replicates per axis; 2,000 source-marginal pair replicates |
| Permutation | 10,000 replicates, two-sided add-one, fixed-effects-respecting (within-level label shuffles for variance shares) |
| Multiplicity | Holm per registered primary family (transfer family: 2 p-carrying members; anatomy family: 5 members) |
| Clustering | context panel: source (16) and context (26/16/10) axes; persona panel: run (80), persona (35), and cell (40) axes; primary interval = the widest |
| Cross-check | Cameron–Gelbach–Miller two-way plug-in SE on regression coefficients; flagged non-positive-semidefinite at 16 clusters on the ordinary-cohort fits (reported, never silent); cluster bootstraps are primary everywhere |
| Seed | 42 (all RNG, `np.random.default_rng`) |
| Solver | exact two-way fixed-effects least squares with BLAS single-thread cap (documented performance fix; estimand unchanged) |
| Hydra config | n/a (standalone analysis scripts with CLI flags) |
| Wall time | 31.5 min CPU total, six scripts, local VM |

**Artifacts:**

- Output JSONs (all six, commit-pinned): [transfer_478.json](https://github.com/superkaiba/explore-persona-space/blob/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_553/transfer_478.json), [unified_rule.json](https://github.com/superkaiba/explore-persona-space/blob/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_553/unified_rule.json), [channel_anatomy.json](https://github.com/superkaiba/explore-persona-space/blob/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_553/channel_anatomy.json), [diag_spill.json](https://github.com/superkaiba/explore-persona-space/blob/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_553/diag_spill.json), [ranking_table.json](https://github.com/superkaiba/explore-persona-space/blob/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_553/ranking_table.json), [exposure.json](https://github.com/superkaiba/explore-persona-space/blob/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_553/exposure.json). Each carries an `inline_vs_reviewed` block (inline target vs reviewed value + convention notes) and step-0 gate records.
- Figures: 21 stems × png/pdf/meta.json at [figures/issue_553/](https://github.com/superkaiba/explore-persona-space/tree/73c7bf50e246b551226ee9a507cc7cceb3d299a7/figures/issue_553) (6 hero + 15 exploratory, including the raw-alongside-corrected scatter grids and the per-K / seed-split share diagnostics).
- Reused eval artifacts from [#532](https://eps.superkaiba.com/tasks/532): [eval_results/issue_532/logp_slot_followup/](https://github.com/superkaiba/explore-persona-space/tree/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_532/logp_slot_followup) (416 + 416 per-cell four-float JSONs + base-prior reads) and [predictors.json](https://github.com/superkaiba/explore-persona-space/blob/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_532/predictors.json) — fit: this IS the panel the inline claims were made on (same cells, same validated corrected-slot read; re-deriving it would change the estimand), conditions complete (16 × 26, both model sides), not saturated (margins span ~40 logits).
- Reused eval artifacts from [#478](https://eps.superkaiba.com/tasks/478)/[#531](https://eps.superkaiba.com/tasks/531): [eval_results/issue_478/logit_rescore/](https://github.com/superkaiba/explore-persona-space/tree/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_478/logit_rescore) (80 run JSONs) and [tidy_logit.parquet](https://github.com/superkaiba/explore-persona-space/blob/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_478/base_prior_reanalysis/tidy_logit.parquet) — fit: the only committed panel with held-out (never-trained, never-negative) personas and per-question four-float reads on both model sides, which is exactly the transfer contrast the Goal names; rescore validated against stored log-probabilities (worst case MAE 0.33 nats / rank correlation 0.996); both seeds and all four mix sizes present.
- Reused emission-rate panel from [#539](https://eps.superkaiba.com/tasks/539)'s loader (secondary leaderboard variable): [eval_results/issue_532/per_cell/loc_ep1/](https://github.com/superkaiba/explore-persona-space/tree/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_532/per_cell/loc_ep1) — fit: same cells as the margin panel, behavioral DV the recommendation pairs with the margin.
- Reused inference machinery from [#539](https://eps.superkaiba.com/tasks/539): `scripts/issue539_residual_per_cohort.py` + `scripts/issue539_corrected_reads_inference.py` (imported as modules; the fast-solver monkeypatch is documented in the round-1 implementation report) — fit: the project's reviewed precedent for fixed-effects-respecting bootstrap/permutation inference, generic over (x, y, cluster-codes) arrays.
- No new HF/WandB artifacts: this task trains nothing and generates nothing; git is the permanent store for all inputs and outputs. Raw completions: n/a (no completions exist anywhere in this task's measurement chain — the parent panels store logit reads, not text).

**Compute:** 0 GPU-hours (budgeted 0); 31.5 min wall on the local VM's CPU; no pod provisioned.

**Code:** seven scripts at commit [73c7bf50e](https://github.com/superkaiba/explore-persona-space/tree/73c7bf50e246b551226ee9a507cc7cceb3d299a7/scripts) on branch issue-553: `issue553_panel.py` (shared loader + step-0 gates + bootstrap/permutation wrappers), `issue553_transfer_478.py`, `issue553_unified_rule.py`, `issue553_channel_anatomy.py`, `issue553_diag_spill.py`, `issue553_ranking_table.py`, `issue553_exposure.py`. Reproduce with:

```bash
for s in transfer_478 unified_rule channel_anatomy diag_spill ranking_table exposure; do
  uv run python scripts/issue553_${s}.py \
    --i532-dir eval_results/issue_532 \
    --i478-parquet eval_results/issue_478/base_prior_reanalysis/tidy_logit.parquet \
    --out-dir eval_results/issue_553 --fig-dir figures/issue_553 \
    --n-boot 10000 --n-cluster-boot 2000 --n-perm 10000 --seed 42
done
```
