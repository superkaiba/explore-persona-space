---
title: Banked context-to-answer maps discriminate minimal-pair contexts, but at context-end
  an identity-plus-bias baseline captures most of the discrimination (MODERATE confidence)
kind: experiment
tags:
- followup-manual
created_at: '2026-08-10T05:01:23Z'
has_clean_result: true
parent_id: 2162
backend: runpod
workflow: v1
goal: 'On Qwen-2.5-7B-Instruct, using the frozen 21-type minimal-pair context bank
  and banked state bank from the parent patch sweep (#2162), measure with NO patching
  how changing exactly one information type in the context shifts (a) the context
  representation (context-end vector AND prefix-end state, both arms: per-type shift
  magnitude, within-type direction consistency, cross-type geometry) and (b) the realized
  on-policy answer representation v_A (teacher-forced answer activations over the
  banked floor/ceiling rollout text under each unpatched context), and (c) whether
  the banked context-to-answer ridge maps (#779 context-end, #1738 prefix-end; no
  new map training unless fitness fails) DISCRIMINATE the paired contexts — per pair,
  does f(v_C(A)) land closer to v_A(A) than to v_A(B) (paired two-alternative retrieval
  accuracy + cosine/euclidean similarity margin), reported per information type with
  a ranking of which types the map discriminates worst, alongside the identity+learned-bias
  baseline and pool-level kNN retrieval, against a shuffled-pair null.'
relates_to:
- spec-context-as-vector
---
# Banked context-to-answer maps discriminate minimal-pair contexts, but at context-end an identity-plus-bias baseline captures most of the discrimination (MODERATE confidence)

<!-- clean-result-v4 -->
**Methodology:** https://github.com/superkaiba/explore-persona-space/blob/73fc2be267b1d0def90ef67022e3d5b5020af378/docs/methodology/issue_2215.md · https://gist.github.com/superkaiba/762123126f58ddfde0541e2a5f0aca9d


## Takeaways

- All five predictors separate the 1,404 minimal pairs above chance: pooled paired two-alternative accuracy **0.59–0.77** against a shuffled-pair null band of 0.48–0.52 (10,000 draws).
- The fitted single-turn context-end map beats the constant-offset baseline by only **+0.9 points**, and its clustered 95% interval includes zero (n = 2,808 directions); the baseline alone reaches 0.758.
- Prefix-end reads are weaker but gain more from fitting: within the matched multi-turn pair, context-end 0.774 vs prefix-end 0.646, and the prefix-end map beats its baseline by **+5.1 points**.
- One information-type change moves the context-end vector consistently in direction (20 of 21 base types above the permutation band) but weakly in magnitude (ratio > 1 for **3 of 21**).
- Answer-vector movement rank-tracks context-vector movement (**ρ = 0.64**; 0.56 after removing a completion-length trend, interval still excluding zero; 39 pair-cells), and length adjustment leaves the per-type answer-shift ranking essentially unchanged (rank stability 0.94); coupling with the parent's judge-scored contrast is undetected (ρ = 0.12, interval spanning zero and the 0.4 confirmation threshold).
- Rig caveats: prefix-end shift yardsticks are zero by construction (carriers share one prefix), and 12.4% of banked rollouts carry CJK drift — excluding them moves pooled accuracy ≤ 0.4 points.

## Goal

- **This experiment in context:** the parent [#2162](https://eps.superkaiba.com/tasks/2162) built the frozen 21-type minimal-pair context bank and measured which information types are causally usable under activation patching. This task asks, with no patching, how far changing one information type moves the context representation and the realized answer representation, and whether the banked context-to-answer ridge maps from [#779](https://eps.superkaiba.com/tasks/779) (single-turn, context-end) and [#1738](https://eps.superkaiba.com/tasks/1738) (multi-turn, prefix-end + matched context-end) preserve the pair distinction. A type the parent's probe decodes perfectly could still sit behind a map that cannot separate the paired answers — this is the map-fidelity read the parent could not give.
- **Broader narrative:** the context-to-answer map line asks whether pre-generation context geometry predicts realized answer representations. This experiment tests whether the fitted maps carry pair-level information beyond what raw context geometry plus a learned constant offset already provides — the floor any "the map understands the context" claim has to clear.

## Methodology

**Design:** 39 pair-cells over 37 context-cells (21 base information types plus 6 context-load, 8 recency, and 4 instruction-vs-demonstration conflict variants), each with 12 carriers × 3 values → 1,404 contexts and 1,404 directed minimal pairs on Qwen-2.5-7B-Instruct. Per pair, exactly one information-type value changes; everything else is token-identical. Reads run at two slots (context-end = last context token; prefix-end = last token before the final user query) for three fitted ridge-map arms and two identity-plus-bias baselines, with no patching and no training. The four conflict pair-cells are forward/reverse twins re-pairing the same 72 contexts: under both-direction scoring each reverse statistic exactly duplicates its forward twin, so pooled counts weigh conflict pairs twice (37 effectively independent cells). Two cells are degenerate at prefix-end by design (the role-header and changed-question cells vary text after the prefix) and are excluded from all prefix-end reads — never drawn as zero bars. A zero-GPU follow-up round adds a completion-length covariate check on the answer-shift reads, using only the banked completion token counts (no new capture).

**Training:** N/A — no model training. Capture and analysis settings:

| Setting | Value | Source |
|---|---|---|
| Model | Qwen/Qwen2.5-7B-Instruct, bf16 | run repro block, `dv3_map_discrimination.json` |
| Teacher-forced capture batch | 8 | parent capture default (`issue2162_run.py`), plan §4.2 |
| Capture rows | 14,040 = 1,404 contexts × 10 rollouts | Phase A coverage gate |
| Banked rollout recipe | 10 on-policy draws/context, temperature 1.0, seed 42 | parent anchors rows (verbatim fields) |
| Answer pooling | tail-inclusive mean (primary); span mean (secondary) | plan §4.3 |
| Layers | all 28 captured; map reads at 14/19/26; primary layer 19, frozen | plan §3 |
| Null / bootstrap draws | 10,000 each; seeds 2215 / 21620 | result JSON `meta` blocks |
| Map payloads | single-turn context-end ridge (LMSYS-fit, 963k rows); multi-turn prefix + context ridge (matched fit corpus and n) | plan §4.1, revision-pinned |
| Pooling-parity gate | cosine ≥ 0.995 on ≥ 99% of rows; realized min 0.999998, fraction 1.0 | pod parity manifest |
| Prefix-end token convention | index parity with the multi-turn map's convention confirmed on sampled contexts | pod A7 manifest, `all_match: true` |
| Length covariate (follow-up) | banked completion token counts, project tokenizer, retokenized; parity with the banked values asserted | `length_covariate.json` `length_metric` |
| LLM judge | none — geometry DVs only | plan §6 |

**Evaluation:** DV1 is the per-pair context-vector difference: magnitude as median norm against a carrier-shift yardstick (median distance between same-value contexts of different carriers — here, different final user queries), and direction consistency as the mean pairwise cosine of the difference across the 12 carriers, judged against a within-cell sign-flip permutation band (95th percentile, 10,000 draws). DV2 applies the same reads to the answer mean (teacher-forced states over the 10 banked rollouts, tail-inclusive pooling), with a split-half draw-noise yardstick. DV3 is paired two-alternative accuracy: per pair and direction, does the arm's prediction from the context vector sit closer in cosine to the correct answer mean than to the paired one — with carrier-clustered bootstrap CIs, a carrier-blocked shuffled-pair null, a leave-one-type-out identity-plus-bias baseline, nearest-neighbor retrieval over the 1,404-answer pool (chance 0.07% at k = 1), and pooled R²/cosine companions. The follow-up length-covariate check regresses per-pair answer-shift norm on the absolute difference in mean completion length (pooled ordinary least squares), re-reads the per-type medians and the coupling on the adjusted norms, and bootstraps raw and adjusted coupling on identical resample indices. Confirmation required, for the context-shift hypothesis, ratio > 1 and consistency above band for at least 18 of 21 base types; for the coupling hypothesis, Spearman ρ > 0.4 with an interval excluding 0; for the map-fidelity hypothesis, a pooled-accuracy interval above 0.5 and a fitted-minus-baseline interval above 0. Headline interval bounds (all carrier-clustered bootstrap, 10,000 draws):

| Quantity (layer 19, cosine, tail pooling) | Point | 95% interval | Verdict |
|---|---|---|---|
| Single-turn context-end map, pooled accuracy | 0.767 | 0.750 to 0.785 | discriminates |
| Multi-turn prefix-end map, pooled accuracy | 0.646 | 0.630 to 0.662 | discriminates |
| Single-turn context-end map − its baseline | +0.009 | −0.004 to +0.022 | inconclusive |
| Multi-turn context-end map − its baseline | +0.016 | +0.002 to +0.030 | beats baseline |
| Multi-turn prefix-end map − its baseline | +0.051 | +0.038 to +0.065 | beats baseline |
| Coupling with parent contrast (Spearman ρ, n = 38 cells) | +0.12 | −0.31 to +0.50 | inconclusive (spans 0 and 0.4) |
| Context-shift vs answer-shift coupling (ρ, n = 39 cells) | +0.64 | +0.45 to +0.75 | exploratory |
| Coupling, length-adjusted (ρ, n = 39 cells; paired bootstrap) | +0.56 | +0.34 to +0.69 | exploratory — survives adjustment |
| Answer shift vs completion-length difference (per-pair ρ, n = 1,404) | +0.35 | +0.30 to +0.40 | covariate present |

**Data extraction:** contexts, pairs, and both slot vectors come verbatim from the parent's frozen bank (1,404 contexts × 2 slots × 28 layers, fp32); the carriers are screened-benign WildChat-derived final user queries and the varied spans are hand-constructed minimal-pair values. Answer states are one unhooked teacher-forced forward per banked rollout (token-id concatenation, BPE-seam-safe), pooled over the answer span including the end-of-turn tail to match the maps' training-target convention; the span-mean twin matched the parent's banked store at ≥ 0.999998 cosine on all 14,040 rows. I acknowledge the fired conciseness WARNs — one Takeaways bullet exceeds the 30-word cap, several result blocks exceed the 120-word prose target, and the total prose exceeds the 800-word budget — nine result families (three dependent-variable families, their companion reads, the language-drift audit, and a length-covariate follow-up) are reported in one round; the cross-type heatmap is deliberately linked rather than embedded (one inline figure per result). <!-- concern-deferred: phase-d-no-entry-skip-sentinel --> One reviewer concern is deferred as repro hygiene: the run's finalize phase lacks an entry skip-if-done sentinel, so a re-entry would duplicate uploads and result markers; the phase ran once, and no reported measurement depends on it.

**Sample training/evaluation data + completions:** examples below are scored by the single-turn context-end map at layer 19; margins are cosine margins (positive = correct side). Full artifacts: [bank.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/dc8108ab84f33695bbc769da0e6e8e2327f51eeb/issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json) and the [banked rollouts](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc8108ab84f33695bbc769da0e6e8e2327f51eeb/issue2162_ctxinfo/raw_completions/anchors) (16 jsonl shards).

Random draw (seed 42) from the 751 of 1,404 pairs correct in both directions — the prior-topic recency cell, 3 turns back, hiking vs birthday topic, carrier e10; all rows: [banked rollouts](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc8108ab84f33695bbc769da0e6e8e2327f51eeb/issue2162_ctxinfo/raw_completions/anchors):

```
Varied span: conversation topic = "hiking" (v3) vs "birthday" (v1), 3 turns before the end
Final query (both sides): "Could you recap your main advice in one sentence?"
Rollout under hiking:   "For your three-day mountain hiking trip, plan your route and daily
                         distances carefully, secure accommodation in advance..."
Rollout under birthday: "Sure! For your daughter's 7th birthday party, consider a themed
                         treasure hunt, a craft corner, and fun games..."
Margins: +0.0562 / +0.0111 — both directions correct
```

Second random draw (seed 42) from the same correct-both-directions population — the knowledge-constraint cell, carrier e6; all rows: [banked rollouts](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc8108ab84f33695bbc769da0e6e8e2327f51eeb/issue2162_ctxinfo/raw_completions/anchors):

```
System prompt: "You have no internet access: you cannot look anything up..." (v1)
           vs  "You have live web browsing and can look up current information..." (v2)
Final query (both sides): "How is the stock market doing right now?"
Rollout under v1: "I don't have real-time data access, so I can't provide the current
                   performance of the stock market. However..."
Rollout under v2: "I can help you check the current status of the stock market by looking
                   it up. To provide you with the most accurate..."
Margins: +0.0138 / +0.0079 — both directions correct
```

Cherry-picked, not a random sample — the single pair of 1,404 misclassified in both directions, the prior-topic recency cell at 5 turns back, birthday vs server-outage topic, carrier e9; all rows: [banked rollouts](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc8108ab84f33695bbc769da0e6e8e2327f51eeb/issue2162_ctxinfo/raw_completions/anchors):

```
Varied span: conversation topic = "birthday" (v1) vs "outage" (v2), 5 turns before the end
Final query (both sides): "What's a common mistake people make with this?"
Rollout under v1: "It sounds like you're referring to designing a race car called the
                   Silver Centurion. However, if you're asking about..."
Rollout under v2: "When naming a race car like the Silver Centurion, a common mistake
                   people often make is focusing too much on the literal meaning..."
Margins: -0.0002 / -0.0008 — both directions wrong (near-ties)
```

By 5 turns back both answers orbit the most recent topic (a race-car naming exchange in the shared filler turns), so the paired answer means nearly coincide. Two further seed-42 draws from the 652 split pairs (one direction wrong): the numeric-list cell carrier e12 ("6 roses" vs "2 lanterns", margins −0.0011 / +0.0035) and the format-instruction cell carrier n8 (bullet points vs numbered steps, margins +0.0264 / −0.0115); their rollouts comply with the instructed format in both cases, and the missed directions again sit at near-zero margins.

## Results

### Maps discriminate paired answers well above chance, but a constant offset does most of it at context-end

Per information type, paired two-alternative accuracy at layer 19 (cosine; tail-inclusive targets from teacher-forced re-forwards of the banked on-policy rollouts): four headline predictors plus per-type shuffled-pair null bands (gray); 36 pairs per cell, worst types at top.

![Per-type paired two-alternative accuracy at layer 19 for the single-turn context-end map, multi-turn prefix-end map, multi-turn context-end map, and identity-plus-bias baseline, with shuffled-pair null bands](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/figures/issue_2215/hero1_per_type_2afc.png)

> **Figure.** *Discrimination is broad but heterogeneous.* Per-type paired two-alternative accuracy, four predictors plus per-type shuffled-pair null bands (gray), 36 pairs per cell, worst types at top. Prefix-end bars are omitted for the two cells degenerate at that slot.

Pooled: single-turn context-end 0.767, multi-turn context-end 0.774, prefix-end 0.646; identity-plus-bias 0.758 (context-end), 0.595 (prefix-end). All clustered intervals clear the 0.48–0.52 pooled null; each fitted context-end map discriminates in 34 of 39 pair-cells (interval above chance; baseline 31). Fitted-minus-baseline: +0.9 points single-turn context-end (interval includes zero), +1.6 multi-turn context-end, +5.1 prefix-end (intervals exclude zero; Methodology table); the slot claim rests on the matched multi-turn pair only.

Worst types: the user-name and prior-topic recency/load cells and demo-format (0.49–0.58); on some instruction-heavy cells the baseline beats the fitted single-turn map (knowledge-constraint 0.903 vs 0.708). Single-model, correlational, no-patching reads; answer-pooling twins move the fitted maps at layer 19 by ≤ 0.6 points (baselines up to 1.1); accuracy peaks at layer 19.

### The low-level view: misses are near-ties, not confident inversions

Per-pair cosine margins per type for the three fitted arms (each dot one scored direction of one pair; dashed line = zero margin; same worst-to-best order as above).

![Per-pair cosine margins per information type for the three fitted map arms, with a dashed zero line](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/figures/issue_2215/expl_margin_scatter_per_type.png)

> **Figure.** *Per-pair margins behind the accuracy bars.* Each dot is one direction of one minimal pair (2,808 per context-end panel; 2,664 in the prefix-end panel, whose two degenerate cells are excluded); margins right of the dashed zero line are correct calls. Weak types cluster tightly around zero rather than inverting.

For the single-turn context-end map, 751 of 1,404 pairs are correct in both directions, 652 in exactly one, and 1 in neither; missed directions sit at margins within about 0.01 of zero. Weak types fail by indistinguishability of the paired answer means, not by systematic inversion.

### Retrieval and fit companions dissociate: the prefix-end map discriminates pairs it cannot retrieve

Nearest-neighbor retrieval of each context's true answer mean among all 1,404, per arm: fraction of contexts whose true answer mean is within the k nearest neighbors of the prediction (k = 1, 5, 10; chance 0.07%, 0.36%, 0.71%), cosine and euclidean panels.

![Nearest-neighbor retrieval accuracy at k equals 1, 5, and 10 for all five predictors, cosine and euclidean, with chance levels marked](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/figures/issue_2215/expl_knn_retrieval.png)

> **Figure.** *Pool-level retrieval separates the arms far more than pair discrimination does.* Context-end arms retrieve the true answer mean in the top 10 of 1,404 for 69–82% of contexts; prefix-end arms sit near 1–9%.

Context-end maps reach 34–35% top-1 (median rank 2–3) with pooled R² 0.57–0.61; the prefix-end map, still discriminating pairs at 0.646, retrieves 1.1% top-1 (median rank 287) with pooled R² −0.13; the identity-plus-bias baselines hold negative R² everywhere while discriminating at up to 0.758 — fit and discrimination dissociate in both directions. Two structural notes bind the absolute numbers: prefix-end predictions take at most 3 distinct values per cell (shared prefixes), capping pool-level retrieval; and absolute R²/cosine carry the capture-convention seam between the maps' training rig and this bank, so the paired within-bank statistics are the seam-protected reads.

### Context-end discrimination is partly pair-specific; prefix-end discrimination is entirely value-generic

Per type and arm, own-pair accuracy against accuracy when the same predictions are scored against another carrier's same-value answer means (cross-carrier targets); the diagonal marks no own-pair advantage.

![Own-pair accuracy versus cross-carrier accuracy per type for all five predictors, with a diagonal reference line](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/figures/issue_2215/expl_carrier_transfer.png)

> **Figure.** *How much of the discrimination survives swapping the target carrier.* Context-end arms sit below the diagonal (own-pair advantage); both prefix-end arms sit exactly on it, as forced by their carrier-identical predictions.

Pooled own-vs-cross: single-turn context-end 0.767 vs 0.677, its baseline 0.758 vs 0.710, prefix-end arms exactly equal by construction. Most context-end discrimination transfers across carriers (value-generic), but the fitted map's own-pair advantage (9.0 points) exceeds the baseline's (4.8), so the fitted map carries some genuinely pair-specific structure beyond a generic re-alignment. The changed-question control is the extreme case: the fitted map's own-pair 0.972 collapses to 0.484 — chance — on cross-carrier targets (0.958 to 0.496 for the baseline), so its discrimination is entirely pair-specific, exactly as its query-specific shift direction predicts.

### One information-type change moves the context-end vector by less than swapping the final user query

Per type, median pair-shift magnitude divided by its yardstick, log scale: context vectors at both slots against the cross-carrier same-value distance, answer vectors against the split-half draw-noise floor; dashed line at ratio 1. Prefix-end points appear only for the 5 cells with a nonzero yardstick.

![Per-type shift magnitude relative to its yardstick on a log scale, for context vectors at both slots and answer vectors](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/figures/issue_2215/hero2_shift_ratio_per_type.png)

> **Figure.** *Magnitude ratios per type.* Context-end ratios (orange) mostly sit below 1; answer-vector ratios (black) mostly above. The near-absent prefix-end series reflects a yardstick that is zero by construction for 32 of 37 cells.

At context-end the yardstick is a swap of the final user query — the span nearest the read slot — and one prefix-side type change moves the vector 0.05–1.6× that: 3 of 21 base types exceed 1 (in-context-learning mapping 1.58, format instruction 1.38, prompted persona 1.38). The context-shift hypothesis' magnitude criterion (18 of 21) fails decisively; granting the two base types within 0.03 of threshold still gives 5 of 21. The prior-topic ratio falls from 0.55 (base) to 0.39 (3 turns back) to 0.16 (5 turns back), so context-end magnitude rankings partly encode span position, not only information content.

### Direction consistency is high at context-end; the prefix-end consistency read is vacuous by construction

Per type, direction consistency (mean pairwise cosine of the pair-shift across the 12 carriers, clustered 95% CIs) against a within-cell sign-flip permutation band (95th percentile, gray ticks), for context vectors at both slots and answer vectors.

![Within-type direction consistency with permutation-null bands, three panels: context-end context vectors, prefix-end context vectors, and answer vectors](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/figures/issue_2215/expl_consistency_vs_band.png)

> **Figure.** *Directions are reliable at context-end.* Context-end consistency (orange) clears the null band for 38 of 39 pair-cells; the prefix-end panel reads exactly 1.0 everywhere because all carriers of a cell share one prefix.

Context-end consistency spans 0.15–0.83, clearing the band for 20 of 21 base types; the exception, the changed-question control (−0.01 against a 0.05 band), is query-specific by design. Shift directions are type-specific, not a generic "something changed" direction: mean absolute cosine is 0.482 within cells vs 0.101 between cells ([heatmap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/figures/issue_2215/expl_cross_type_cosine_heatmaps.png)). This split and the changed-question control bound a lexical-propagation alternative — consistent directions carried by the literal inserted value tokens propagating to the read slot rather than by encoded content — without eliminating it (carriers of a cell share that text).

The prefix-end panel carries no information: with token-identical prefixes, every carrier's shift is the same vector, so consistency is identically 1 against the 0.18 sign-flip band; per-unit tail p-values below 1/4096 are unavailable (12-carrier sign-flip atoms).

### Answer-vector movement tracks context-vector movement; coupling with the parent's judge contrast is undetected

Left: per-cell noise-normalized answer-vector shift against the parent's per-cell behavioral separation (judge-scored ceiling-minus-floor contrast), points labeled. Right: per-type mean discrimination margin against the same x-axis for the two goal-named maps.

![Answer-vector shift and discrimination margin plotted against the parent behavioral separation per cell](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4dcde9ce0eac58ce16ae10b0827a98bc46401a71/figures/issue_2215/expl_h2_shift_vs_separation.png)

> **Figure.** *The coupling read.* Per-cell noise-normalized answer-vector shift vs the parent's judge contrast (n = 38 cells; the filler-swap cell has no parent rows; interval bounds in the Methodology table). The in-context-learning cell is the extreme outlier at 49.7.

The coupling hypothesis fails its confirmation criterion: ρ = 0.12 (n = 38) with an interval spanning zero and the ρ = 0.4 confirmation threshold — no detectable rank coupling, and the data cannot establish decoupling either; the conflict forward/reverse twins double four points. Raw answer-vector movement, in contrast, rank-tracks raw context-end movement across cells (ρ = 0.64, interval excluding zero, n = 39) and within cells (median per-cell ρ = 0.64). 30 of 39 pair-cells shift above the split-half noise floor; the 9 below it (demo format, demo persona, filler swap, user emotion, role header, user-name recency at 3 and 5 turns back, user-name load at 5 items, prior-topic at 5 turns back) are below-noise-floor reads — the floor is conservative by roughly √2, so "no shift" is not established for them.

### Completion-length differences covary with answer-vector shift but do not carry the per-type ranking or the coupling

Per pair, the answer-shift norm against the absolute difference in mean completion length (banked token counts; n = 1,404), plus per-type medians before and after a pooled linear length adjustment and the coupling recomputed on adjusted medians.

![Per-pair answer shift versus completion-length difference; per-type medians and coupling, raw versus length-adjusted](https://raw.githubusercontent.com/superkaiba/explore-persona-space/52c7ed13492bf8a12c4f6843a03f3a70c525c6c0/figures/issue_2215/fu_length_covariate.png)

> **Figure.** *Length-covariate check on the answer-shift reads.* Top left: per-pair scatter with the pooled fit line. Top right: per-type medians before vs after length adjustment; points near the dashed diagonal keep their rank. Bottom: the coupling on raw (left) and length-adjusted (right) medians; interval bounds in the Methodology table.

Pairs whose completions differ more in length shift more: pooled per-pair ρ = 0.35 (n = 1,404), yet the per-cell median is 0.14 (27 of 39 cells positive) — most of the pooled association is between-type. Removing the pooled length trend leaves the per-type ranking essentially unchanged (rank stability 0.94): the answer-shift ranking is not carried by length. The coupling with context-end movement drops from 0.64 to 0.56 under a paired bootstrap, its interval still excluding zero (Methodology table).

Eight of the nine below-noise-floor cells from the previous result (all but the filler swap) move just above the floor once shifts are put on an equal-length footing (adjusted ratios 1.01–1.37). The yardstick itself is not length-adjusted, so these are marginal equal-footing reads, not new detections.

### Language-drift audit: 12.4% of banked rollouts contain CJK text; headline reads are unchanged without them

Pooled paired two-alternative accuracy per arm (layer 19, cosine), computed from all 10 draws per context (filled, with carrier-clustered CIs) and recomputed with every CJK-bearing rollout excluded from the answer means (open); gray band is the shuffled-pair null.

![Pooled discrimination accuracy per arm with all draws versus with CJK-intruded draws excluded](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/figures/issue_2215/audit_cjk_recount.png)

> **Figure.** *Intrusion-sensitivity recount.* Filled dots: committed pooled accuracy (all 10 draws, clustered 95% CIs). Open dots: recount excluding the 1,736 CJK-bearing rollouts. The two coincide to within 0.4 points for every arm.

1,736 of 14,040 banked rollouts (12.4%) contain CJK characters (median intruded row: 1.7% of characters), concentrated in the implied-language (43.9% of the cell's rollouts) and language-instruction (28.1%) cells — whose values instruct Spanish, English, or French, so CJK is never on-task — though the tail is broad (filler swap 25.3%; several recency and demonstration cells near 20%). The all-draws recount reproduces the committed accuracies exactly (delta 0); excluding intruded draws changes pooled accuracy by at most +0.4 points and moves no per-cell accuracy across its null band (a point-versus-band check; per-cell recounts cover the single-turn context-end and prefix-end maps), and the largest per-type change is implied-language at prefix-end improving from 0.736 to 0.861 — drift adds target noise rather than manufacturing discrimination. No context loses all 10 draws.

---
**Repro:** one H100 (RunPod pod-2215, `eval` intent); production teacher-forced capture ≈ 9.5 min (pilot-measured 0.014 s/row), CPU analysis ≈ 44 min, ≈ 2.5 h pod wall including smoke rounds · run code SHA `f6e6ab50b8f3be02803e8a28a522058b69b4efb4` ([tree](https://github.com/superkaiba/explore-persona-space/tree/f6e6ab50b8f3be02803e8a28a522058b69b4efb4)); analysis/audit additions at `f4ddc4dedc1b6b44d487363d39e3d019e79fd40d` ([eval_results/issue_2215](https://github.com/superkaiba/explore-persona-space/tree/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/eval_results/issue_2215), [figures/issue_2215](https://github.com/superkaiba/explore-persona-space/tree/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/figures/issue_2215)); coupling figure re-rendered at `4dcde9ce0eac58ce16ae10b0827a98bc46401a71` (right-panel axis label was clipped at the figure edge; internal title tag removed) · zero-GPU length-covariate follow-up: `scripts/issue2215_length_covariate.py` → [eval_results/issue_2215/length_covariate](https://github.com/superkaiba/explore-persona-space/tree/52c7ed13492bf8a12c4f6843a03f3a70c525c6c0/eval_results/issue_2215/length_covariate) + figure, committed at `52c7ed13492bf8a12c4f6843a03f3a70c525c6c0` (recount-script lint fix at `d4a07bebda92d1665d80c9617482910b797fb0d5`) · answer-state store + null matrices: [issue2215_reprshift/analysis_tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/47ca8e79d3660073746e590afdbf41782c793a3a/issue2215_reprshift/analysis_tensors) (16 shards ≈ 5.6 GB + `null_matrices.npz`; listing verified via the Hub API at write time) · Reused context bank + state bank + rollouts from [#2162](https://eps.superkaiba.com/tasks/2162): `issue2162_ctxinfo/{analysis_tensors/vc_bank, analysis_tensors/anchors, raw_completions/anchors}` at revision `dc8108ab84f33695bbc769da0e6e8e2327f51eeb` — fit: same frozen contexts this task's Goal names; parity-gated against this run's recapture · Reused ridge maps from [#779](https://eps.superkaiba.com/tasks/779) (`issue779_monitoring/n1m_readout/weights/L19/ridge.pt` @ `dc8108ab84f33695bbc769da0e6e8e2327f51eeb`, [tree](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc8108ab84f33695bbc769da0e6e8e2327f51eeb/issue779_monitoring/n1m_readout/weights/L19)) and [#1738](https://eps.superkaiba.com/tasks/1738) (`issue1738_multiturn/analysis_tensors/weights/L19/context_ridge.pt` and `prefix_ridge.pt` at the same revision, [tree](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc8108ab84f33695bbc769da0e6e8e2327f51eeb/issue1738_multiturn/analysis_tensors/weights/L19)) — fit: the Goal names these exact payloads; applied read-only, no refitting · language-drift recount: `scripts/issue2215_cjk_recount.py` → `eval_results/issue_2215/cjk_intrusion_recount.json`.

**Context:** originating prompt (user chat, 2026-08-09, verbatim):

> can you run almost that same experiment but instead looking at how changing that information in the context:
> - changes the context vector
> - changes the answer vector
> - whether our mapping can distinguish well between both contexts (i.e. can the mapping correctly identify the answer vector corresponding to each context + which it's worse at (some kind of similarity metric))

("that same experiment" = the parent's 21-type minimal-pair patch sweep.) Lineage: [#2162](https://eps.superkaiba.com/tasks/2162) — parent; maps consumed read-only from [#779](https://eps.superkaiba.com/tasks/779) and [#1738](https://eps.superkaiba.com/tasks/1738). Created 2026-08-09; run 2026-08-10; length-covariate follow-up folded 2026-08-10.
