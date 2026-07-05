---
title: Attention mixing adds linearly readable information once the mixed forward
  is read at matched query-span granularity — the earlier full-prompt deficits were
  last-token-slot and span-dilution read-out artifacts (MODERATE confidence)
kind: experiment
tags:
- followup-auto
created_at: '2026-07-03T10:36:02Z'
has_clean_result: true
parent_id: 658
origin_prompt: 'Help me to plan this:


  We want to see if there is some decomposition of single query mapping into the "context"
  portion and the "query" portion


  For this we can:

  - directly predict the answer just from the context portion

  - directly predict the answer just from the query portion (with blank context -->
  make sure to not insert system prompt -- try empty system prompt but also removing
  the system prompt part of chat template completely......, also potentially just
  masking out the tokens of the context with the query in there)

  - context and query must be disjoint token sets

  See if by some combination of these 2 mappings we can get better performance

  Also train the context + query -> answer mapping and analyze the relationship between
  this mapping and the 2 other mappings


  probably good to use our diverse contexts + ultrachat queries setup so you get:

  M_A: context -> answer

  M_B: query -> answer

  M_C: context + query --> answer

  with matched contexts and queries

  All generations should be on policy

  Evaluate mappings on LOFO context + OOD queries'
workflow: v1
goal: 'Determine whether the base model''s per-cell mean answer activation v(c,q)
  (answer-token-mean residual profile of the on-policy completion, the #810-winning
  summary) decomposes into context-only and query-only representational components:
  with no model training and ridge-only read-outs, predict v(c,q) on the diverse-contexts
  x UltraChat grid from (a) the last-context-token vector computed with the query
  absent, (b) the last-query-token vector computed with no context (empty-system /
  no-system-block / masked-context presentations), (c) their combination (feature
  concatenation and prediction-level blend), and (d) the full-prompt last-input-token
  vector, and measure under LOFO context-family x held-out/OOD-query folds how much
  of the [best-single -> full-prompt] held-out R^2 gap the combination closes, with
  the interaction residual R^2(full) - R^2(combined) quantifying the attention-mixing
  information unavailable to disjoint parts.'
relates_to:
- spec-context-as-vector
- leak-predictor
---
# Attention mixing adds linearly readable information once the mixed forward is read at matched query-span granularity — the earlier full-prompt deficits were last-token-slot and span-dilution read-out artifacts (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_923.md](https://github.com/superkaiba/explore-persona-space/blob/7aad0211e159f2f469ded8a8a08cd4df56cc362e/docs/methodology/issue_923.md) · [gist](https://gist.github.com/superkaiba/ac16d1734da76e1234253dfa1a0e7a9b)

## Takeaways

- Letting query tokens attend to the context adds linearly readable information once the read-out granularity is matched: the query-span mean of the unmasked forward reads 0.566 vs 0.462 for the identical read with context attention masked (paired gap +0.103, interval +0.070 to +0.137; positive in 7 of 7 held-out families) at layer 18 on UltraChat — the two earlier rounds' no-attention-mixing-advantage takeaway is overturned as summary-granularity-specific.
- Both earlier full-prompt deficits are read-out artifacts rather than evidence of missing linearly readable information under the tested read-out summaries: the parent round's deficit came from the last-token slot and the pooled round's collapse from whole-input span dilution — reading the same unmasked forward as two per-block span means instead of one whole-span mean gains +0.279 (interval +0.109 to +0.465), with the two ~3,000-character context families recovering most (WildChat +0.60, in-context-learning +0.48, vs +0.10 to +0.12 in the five short families).
- Stitching two disjoint forwards loses almost nothing against the true mixed forward at matched two-span granularity — gaps −0.012 (interval −0.018 to −0.004) and +0.007 (interval +0.000 to +0.013) on UltraChat, +0.014 and +0.105 on Betley — so against the strongest presentation context geometry remains a near-lossless self-contained predictor input; the Betley +0.105 sits on the masked-context-presentation side, whose stitched reference (0.373 vs 0.464 for the strongest) inherits the replicated penalty-selection degradation, and the mixed query-span read is the strongest single-vector summary (0.566, against the stitched pair's 0.570).
- The query, not the context, carries most of the per-cell answer profile in every round: in-sample shares 84% query / 8% context / 9% interaction (targets reused unchanged); the context-minus-query read gap is −0.355 last-token and −0.392 span-mean; the context factor is small but fully readable (0.093 vs the 0.078 in-sample ceiling).
- All 6 new mixed-forward reads beat their selection-symmetric permutation nulls at p ≈ 0.001 (n = 1000), including on held-out human-written Dolly queries (0.474 / 0.466); the registered Betley primary contrast is uninterpretable because its reference is the masked-context Betley collapse replicated in both prior rounds (p = 1.0, reused fits) — the honest Betley reads are the stitched gaps (+0.014 / +0.105) and a fixed-penalty ladder on which both new arms are flat at every penalty.

## Goal

**This experiment in context:** The line's ridge maps predict the answer-side activation profile from the full prompt: per-example held-out R² 0.60–0.63 ([#722](https://eps.superkaiba.com/tasks/722)/[#779](https://eps.superkaiba.com/tasks/779)), content-indexed ([#823](https://eps.superkaiba.com/tasks/823)), genre-general via the mean answer-token summary ([#810](https://eps.superkaiba.com/tasks/810)), on the battery and store from [#658](https://eps.superkaiba.com/tasks/658). This experiment asks whether that map factorizes: with no model training, predict the per-cell mean answer activation from the isolated last-context-token and last-query-token vectors, separately and combined, against the full-prompt read, under leave-one-family-out × held-out-query folds.

**Broader narrative:** If the context contribution to the answer profile is separable, base-side context geometry is a self-contained predictor input for the leakage-prediction program; a large interaction residual would instead cap any context-only predictor. Measured, after the parent round and two follow-up rounds: attention mixing does add linearly readable information (+0.103 at the matched query-span read), so the strict no-interaction reading is no longer tenable — but what the program needs mostly survives, because stitching a context-span read onto a disjoint query-span read recovers the true mixed forward to within ±0.012 at two-span granularity on UltraChat (on Betley +0.014 against the strongest presentation, +0.105 against the penalty-degraded masked one); context geometry remains usable as a self-contained predictor input, with the caveat that the mixed query-span read is now the stronger query-side summary. The context factor stays a small share of per-cell variance (8%), most of which tracks the query.

## Methodology

**Design:** One base model, `Qwen/Qwen2.5-7B-Instruct`, forward passes and greedy generation only; ridge read-outs only, no model training anywhere. Grid: 50 contexts in 7 families (persona 14 / WildChat 10 / in-context-learning 8 / rephrase 6 / format 5 / behavior 5 / default 2) × 144 UltraChat queries (the 48 store queries + 96 fresh length-matched ones), plus two secondary grids — 50 × 48 Betley (misalignment-genre) queries and 50 × 48 human-written Dolly queries (the out-of-distribution arm). Target: the per-cell mean residual activation over the answer tokens of the model's own greedy completion to the full (context, query) prompt, per layer (28 layers). Read-out arms, per layer: Context-only (last context token from a prefix-only forward), Query-only in three null-context presentations (explicit empty system turn; no system block; full prompt with context tokens attention-masked in place), Stitched pair (feature concatenation of Context-only + Query-only, one variant per presentation), Blended predictions (two-parameter combination of the two single-arm predictions, fit on an inner validation split), and Full prompt (last input token at the assistant header). Contexts and queries occupy disjoint token spans, and each partial feature is computed with the other input absent. References: an in-sample variance decomposition of the target into context main effect, query main effect, and interaction (oracle ceilings); selection-symmetric permutation nulls; a 50-cell regeneration spot-check of the store join. Follow-up round (span-mean features): the prompt-side feature summary statistic swaps from the last token to the mean over the arm's owning token span — context span for the context arm, query span per presentation for the query arms, full input span for the full-prompt arm — uniformly across every arm; targets, grids, folds, reductions, penalty selection, bootstrap and null designs are unchanged, and the targets are reused unchanged from the parent round. Because the parent persisted only last-token vectors, the round re-ran the prompt-side forwards (~24,500 prefill-only passes) behind a per-context identity gate against the parent packs, and registered selection-symmetric nulls for the blended-predictions and Dolly arms the parent left un-gated. Mixed-forward round (`mixed-forward-span-stitch`): again exactly one variable — attention from the query span to the context, ON (a query-span mean of the unmasked full-prompt forward, plus a stitched variant concatenating the context-span read) vs OFF (the masked-context span-mean read: same absolute right-padded positions, same query-block span, same summary). The round ran ~12,000 unmasked prefill-only forwards behind the same identity gate; targets and every comparison arm are reused unchanged at the pinned parent and pooled revisions; both new arms get 1000-draw selection-symmetric nulls on all three surfaces, and the four registered contrasts are paired family-bootstrap gaps on shared draws, gated by an exact reproduction of the pooled round's full−stitched gap from the reused fits.

**Training:** **N/A — no model training.** Analysis-design constants:

| Parameter | Value | Source |
|---|---|---|
| Ridge λ grid | 1e-2, 1e-1, 1, 10, 100, 1000; nested PRESS leave-one-out per train fold (follow-up diagnostic: leave-query-group-out re-selection at layer 18, masked-context arms) | #722/#823 harness (`RIDGE_LAMBDAS`) |
| Feature/target reduction | PCA-48 both sides, train-fold-fit bases; train-fold-centered targets | #722 (28/28-layer input-PCA equivalence, max ΔR² 0.018); #810 target recipe |
| Prompt-side feature summary | last token (parent round); mean over the arm's owning token span (`pooled-span-features` round — the round's single changed variable) | followup-scope `pooled-span-features`; plan v6 §4.2 |
| Mixed-forward arms (`mixed-forward-span-stitch` round) | `arm_qry_mix` = query-span mean of the unmasked full-prompt forward (the round's single changed variable: query→context attention ON vs the masked `pool_qry_iii` reference); `arm_concat_mix` = context-span read ⊕ `arm_qry_mix` | followup-scope `mixed-forward-span-stitch`; plan v9 |
| Layer binding | same-layer, swept 0–27; headline layer 18 frozen in the plan | #722 peak layer |
| Folds | 7 leave-one-family-out × 4 query folds (UltraChat 144 → 4×36; Betley 48 → 4×12); both axes unseen in every test cell | #810 + the project's group-fold rule |
| Out-of-distribution fold | train non-held family × all 144 UltraChat queries; test held family × 48 Dolly queries | plan §4.2 (corpus-transfer form) |
| Generation | vLLM greedy, temperature 0.0, max_tokens 512 | #658 store recipe, matched exactly |
| Permutation null | 1000 cell-label permutations; λ re-selected per draw; per-draw max-over-layers band; the pooled round adds bands for the blended + Dolly arms | #810 batched-null design; plan v6 §4.2 |
| Bootstrap | 2000 family-cluster draws; cross-classified family × query draws for the context-vs-query contrast; paired draws for the span-mean-vs-last-token gap comparison | #810 precedent; plan §3 |
| Decomposability-fraction guard | denominator floor max(0.02, 2 × bootstrap SE) | plan §3 (from the #722 compression bound 0.018) |
| Masked-context attention backend | SDPA with a 4D float mask (dummy-context invariance smoke passed) | plan §8 |
| Seed | 42 (probe build, folds, permutations, bootstrap) | plan §10 |

**Evaluation:** Primary dependent variable = pooled held-out skill-over-mean R² per (arm × layer): fits on train-family contexts × train-fold queries, scored on held-out-family contexts × held-out-fold queries, pooled so every cell is tested exactly once (n = 7200 UltraChat cells, 2400 Betley, 2400 Dolly; no cells dropped). Baseline for skill = the train-fold mean. Headline reads at the plan-frozen layer 18; the layer sweep is gated on per-draw max-over-layers permutation bands. The interaction residual is the full-prompt skill minus the stitched-pair skill; the plan's power floor (skip verdicts if the full-prompt read is below 0.05 everywhere) did not trigger in either round. The construct is representation-level — linear predictability of the model's own answer-side activation — not a behavioral claim; the variance shares are in-sample references, never held-out claims, and ambient-space skill is reported alongside as the plan's registered secondary robustness read. R² magnitudes are not comparable to the parent issues' probe-averaged numbers (different target granularity and fold scheme).

**Data extraction:** Targets for the 48 original UltraChat and 48 Betley queries are re-reductions of the parent store's per-context answer-span tensors (~341 GB streamed and reduced on a CPU instance); targets for the 96 fresh UltraChat and 48 Dolly queries were captured fresh with the identical recipe (greedy generation, teacher-forced forward, mean over the answer span). The Context-only feature comes from a prefix-only forward whose equality with the same position inside the full prompt was probe-verified per context (cosine floor 0.99 enforced; values below 0.999 recorded as warnings — bf16 batching numerics, worst recorded 0.9989). The empty-system presentation asserts the Qwen default system prompt was not silently inserted. Fresh UltraChat queries were length-matched to the Betley length profile with the parent's builder; Dolly rows are instruction-only (empty context field), same filters, length-matched, from the established databricks-dolly-15k corpus (tier 1–2 realism); UltraChat is an established tier-2 corpus; the battery inherits the line's house-written tier-3 families as a standing scope caveat. The follow-up round's span-mean features reuse the capture script's span-mask mean machinery extended to the prompt spans; the recomputed forwards' last-token vectors were checked per context against the parent's persisted packs (cosine floor 0.99; identity gate passed at capture), and the paired bootstrap reproduces the parent's last-token gap from the reused fits to machine precision (max abs error 2.2e-16). The mixed-forward round's identity gate passed with per-genre median cosines of at least 0.99985 and zero rows below 0.999 (`identity_check_mix.json`), and its paired bootstrap reproduces the pooled round's full−stitched gap from the reused fits to machine precision (max abs error 9e-16).

**Sample training/evaluation data + completions:** No training rows exist; the worked examples below are evaluation cells (context + query + the model's own greedy completion whose activations form the target). Neither follow-up round generated new completions — the pooled round re-read the same prompts under a different summary statistic, and the mixed-forward round re-ran only unmasked prompt-side forwards over the same rendered prompts. Completions are cut for space; full texts for all 7250 new-arm completions (4800 UltraChat-extension, 2400 Dolly, 50 regeneration spot-check): [raw_completions on the HF data repo (pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/77d04e45096e120ca897fdb7b22730e157ce00a2/issue923_ctx_query_decomposition/raw_completions).

Randomly drawn UltraChat-extension cell (from `uc_ext/f1_house_programmer_shard2of4.json`, row with q_idx 17):

> **Context (system prompt):** "You are a programmer."
> **Query:** "Create a playlist of at least 10 songs that feature unusual or unique instruments, such as the sitar, didgeridoo, kalimba, or theremin, across a variety of music genres like folk, world music, indie, and experimental."
> **Completion (opening):** "Certainly! Here's a playlist featuring unique instruments across various genres: 1. **"Sitar Solo" by Ravi Shankar** - Folk/World Music …"

Format-family cell, the family the last-token full-prompt read transfers worst to (from `uc_ext/f5_fmt_code_comment_shard3of4.json`, row with q_idx 0):

> **Context (system prompt):** "Respond to every question as a block of code comments: each line of your answer must start with '# '."
> **Query:** "Who won the men's singles title in the French Open in 2019, and what was the score in the final?"
> **Completion (complete, 2 comment rows):** "# The men's singles title in the French Open in 2019 was won by Rafael Nadal." / "# The score in the final was Nadal 6–3, 6–3, 6–0 against Dominic Thiem."

Randomly drawn out-of-distribution Dolly cell (from `ood_dolly/f6_helpful_asst_shard0of4.json`, row with q_idx 47; the query typo is verbatim from the corpus):

> **Query:** "Tell me which of these are shades of Blue that BWM paints their cars: Miami Blue, Laguna Seca Blue, Le Mans Blue, …"
> **Completion (opening):** "BMW has used several shades of blue in their car paint schemes over the years, but not all of the shades you listed are official BMW colors. …"

Query-only presentations, rendered verbatim for one Dolly query — the explicit-empty-system form first, then the hand-rendered no-system-block form (the third presentation keeps the full token sequence and attention-masks the context span in place):

```
<|im_start|>system
<|im_end|>
<|im_start|>user
Why do home power outages occur?<|im_end|>
<|im_start|>assistant

<|im_start|>user
Why do home power outages occur?<|im_end|>
<|im_start|>assistant
```

Betley-genre queries are the published misalignment-evaluation question pool inherited from the parent store; per the project's content-hygiene rule they are referenced by artifact only (48 rows per context in the parent store's raw-completions records; this run's regeneration texts under `regen_check/`, 25 cells per genre), not quoted.

## Results

### Stitching the disjoint context and query vectors beats the full-prompt read-out on both held-out axes

Pooled held-out skill R² at layer 18 on the UltraChat grid (n = 7200 cells), per arm, with 95% family-bootstrap intervals and dashed in-sample ceilings; the second figure shows the per-unit data — per-held-out-family skill dots and all 7200 per-cell predictions vs actuals for the stitched pair and the full prompt.

![Pooled held-out skill R-squared at layer 18 for the five read-out arms on the UltraChat grid, with family-bootstrap error bars and oracle-ceiling reference levels](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d786a8bde53aa985d23e85d5f0b84b0246654e3/figures/issue_923/hero_L18_uc.png)

![Per-family skill dots per arm, and per-cell predicted-versus-actual scatters for the stitched pair and the full prompt at layer 18 on UltraChat](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d786a8bde53aa985d23e85d5f0b84b0246654e3/figures/issue_923/family_cells_L18_uc.png)

> **Figure.** *The stitched pair (0.540) beats the full-prompt read (0.460) in every held-out family, most in format contexts.* Context-only 0.093, Query-only 0.448, Blended predictions 0.533; ceilings 0.078 (context) and 0.914 (additive). Lower panels: the per-family and per-cell data behind the pooled bars.

The residual, −0.081 (interval −0.157 to −0.045), is the plan's designated bottleneck outcome: under this ridge/PCA/fold estimator the last-input-token vector is less linearly readable than the disjoint pair. Blended predictions add no feature dimensions yet nearly match the stitched fit. This argues against doubled feature count as the main explanation. The decomposability fraction is undefined here (best-single-to-full gap 0.012, guard 0.073); the full-prompt arm is also the noisiest, with bootstrap uncertainty roughly triple the stitched arm's.

### The full-prompt deficit survives the span-mean summary swap — a whole-span reading the mixed-forward round supersedes

Pooled held-out skill R² at layer 18 (UltraChat, five read-out arms) under last-token (parent round) and span-mean (pooled round) features, with 95% family-bootstrap intervals; the second figure shows per-family skills and per-cell predictions.

![Last-token versus span-mean feature summaries for five read-out arms at layer 18 on UltraChat](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c14d1b1184232a79ea92c0bb70e6cafb8fc470f9/figures/issue_923/pooled_hero_L18_uc.png)

![Per-family skills under both summaries with per-cell span-mean scatters](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c14d1b1184232a79ea92c0bb70e6cafb8fc470f9/figures/issue_923/pooled_family_cells_L18_uc.png)

> **Figure.** *The mixed read falls, the disjoint reads hold.* Full prompt 0.460 → 0.279; stitched 0.540 → 0.570; query-only 0.448 → 0.481; blended 0.533 → 0.562; context-only 0.093 → 0.089. Lower panels: format recovers (0.171 → 0.427), WildChat (−0.04) and in-context-learning (0.13) collapse; per-cell predictions below (n = 7200).

The registered falsification test: a final-token slot artifact should dissolve under a span-mean of the same forward. It widens instead: −0.290 (interval −0.481 to −0.115) vs −0.081 on UltraChat; −0.163 (−0.291 to −0.032) vs −0.041 on Betley; the widening, −0.210 (−0.425 to +0.016), is at-least-as-large, not reliably larger. Most of the widening is span-length dilution (measured directly two results below): the full−stitched gap collapses only in the two long-context families (WildChat −0.621, in-context-learning −0.493) yet stays negative (−0.10 to −0.13) in the five short ones. The decomposability fraction remains guard-undefined on UltraChat and becomes undefined on Betley (parent round: 1.52); the secondary prediction, span-mean closing the query-readability gap, failed (0.03 of 0.39 closed).

> **Superseded (mixed-forward round).** This section's refutation of the slot-artifact explanation was the pre-mixed-round reading: the whole-span test confounded slot removal with span dilution, and the matched two-span read of the same forward erases the short-family residual the refutation leaned on (recoveries +0.097 to +0.120; two results below). The measurements above stand as measured.

### Attention to the context adds linearly readable information at matched query-span read-out

Pooled held-out skill R² at layer 18 on the UltraChat grid (n = 7200 cells; 95% family-bootstrap intervals), two new mixed-forward arms (blue) vs four pooled-round reference arms (grey); second figure: per-held-out-family skill, same six arms.

![Pooled held-out skill at layer 18 for the two mixed-forward arms against four pooled-round reference arms on the UltraChat grid, with family-bootstrap error bars](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7d861696b4b222181db33016ade7172ebc46b91e/figures/issue_923/mixed_hero_L18_uc.png)

![Per-held-out-family skill at layer 18 for the six arms of the mixed-forward comparison on the UltraChat grid](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7d861696b4b222181db33016ade7172ebc46b91e/figures/issue_923/mixed_family_dots_uc.png)

> **Figure.** *The mixed-forward query-span read (0.566) beats the masked-context read (0.462) differing only by attention to the context.* Paired gap +0.103 (interval +0.070 to +0.137; 2000 shared bootstrap draws), positive in all 7 held-out families (+0.049 to +0.164). Lower figure: the per-family data.

Registered direction (a) fired: with tokens, absolute positions, pooled span, and summary all matched, the one remaining difference — whether query tokens attend to the context — adds +0.103 of held-out skill. The read beats the strongest no-mask disjoint query read (0.481, interval 0.456 to 0.501), so a mask-degraded reference cannot explain the gap. Both new arms clear their selection-symmetric nulls on all three surfaces (p ≈ 0.001; Dolly 0.474 / 0.466), and the read is penalty-insensitive (0.566 at every fixed penalty). The no-attention-mixing-advantage takeaway described the read-outs tested, not the representation: overturned as summary-granularity-specific (layer profile hedged two results below).

### Reading the mixed forward as two span means recovers the long-context collapse, and stitched disjoint reads sit at parity with it

Per-held-out-family skill gaps, UltraChat, layer 18 — grey, the pooled whole-span full-prompt read minus the stitched pair; blue, stitched-mixed (the same forward read as two span means) minus the pooled stitched pair; shaded band, the short families' grey-gap range; pink bands, the two long-context families. Second figure: Betley skill per fixed penalty, new reads plus the masked-context reference.

![Whole-span collapse versus the two-span read of the same forward, per held-out family](https://raw.githubusercontent.com/superkaiba/explore-persona-space/25e11eb3a4be6d91f936f53fa07b8845892c1f08/figures/issue_923/mixed_dilution_panel.png)

![Betley skill per fixed ridge penalty, mixed-forward arms versus the masked-context read](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7d861696b4b222181db33016ade7172ebc46b91e/figures/issue_923/mixed_betley_lambda_ladder.png)

> **Figure.** *The whole-span collapse (WildChat −0.62, in-context-learning −0.49) vanishes at two-span granularity: blue bars within ±0.025 in all 7 families; aggregate gain +0.279 (interval +0.109 to +0.465).* Second figure: new Betley reads flat at every penalty; masked-context read negative below penalty 100.

Two-span read-out recovery is confirmed by construction — grey and blue summarize the same forward — and bundles span dilution with the SNR/dimensionality change of split-span features: the two ~3,000-character families recover +0.597 and +0.485, vs +0.10 to +0.12 elsewhere. At matched granularity the disjoint stitch matches the mixed forward: −0.012 (interval −0.018 to −0.004) against its strongest presentation, +0.007 (interval +0.000 to +0.013) against the position-matched one; on Betley +0.014 and +0.105. The registered Betley primary contrast stays uninterpretable — its reference is the twice-replicated masked-context collapse (−5.11, p = 1.0) — never headline-bearing.

### The mixing gap is largest at the frozen headline layer and reverses in the final three layers

Pooled held-out skill by layer (0–27) for the two mixed-forward arms on the UltraChat grid, with the pooled round's masked-context, stitched, and full-prompt curves as references; this is the per-layer data view behind the headline gap.

![Held-out skill by layer for the two mixed-forward arms with pooled-round reference curves on the UltraChat grid](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c17d054b9d092bdb8045d50924aa9be5be9418c1/figures/issue_923/mixed_layer_curves_uc.png)

> **Figure.** *Both new arms peak at layer 9 (0.617 / 0.619), well before the frozen headline layer.* The mixing gap over the masked read is positive at layers 0–24 (+0.002 to +0.103, largest at layer 18) and reverses at layers 25–27 (−0.008 to −0.031); point differences of persisted curves, no per-layer paired interval.

Layer 18 was frozen in the plan before this round, so no selection correction applies, but the mixing gap peaking exactly at the frozen layer deserves a hedge: the gap thins after layer 20 and reverses in the last three layers, where the masked read overtakes the mixed one. Adding the context-span features to the mixed read buys nothing at the headline layer — stitched-mixed sits slightly below the mixed query-span read (0.558 vs 0.566 UltraChat; 0.477 vs 0.482 Betley) — suggesting the query-span mean of the unmasked forward already carries what the context-span stitch would add.

### The pooled ordering holds at every layer, and 25 of 26 newly registered pooled reads beat their permutation nulls

Pooled span-mean skill by layer (0–27) for the five headline arms on the UltraChat grid, with the parent round's last-token full-prompt and stitched curves as references and layer 18 marked; this is itself the per-layer data view. Unplotted presentation variants track the empty-system variant within 0.02 at layer 18 on UltraChat.

![Pooled held-out skill by layer for the five span-mean arms on the UltraChat grid, with the parent last-token full-prompt and stitched curves as references](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c14d1b1184232a79ea92c0bb70e6cafb8fc470f9/figures/issue_923/pooled_layer_curves_uc.png)

> **Figure.** *Layer 18 is representative under the span-mean summary too.* The stitched pair beats the full prompt at 28 of 28 layers (minimum gap 0.219 UltraChat, 0.037 Betley); the span-mean full-prompt curve peaks at 0.291 (UltraChat) and 0.322 (Betley), below its last-token counterpart beyond layer 1 on UltraChat and layer 11 on Betley (earlier layers sit above, max +0.263).

Every pooled arm except one clears its selection-symmetric permutation null at p ≈ 0.001 (n = 1000, penalty re-selected per draw): 25 of 26 registered reads, now including the blended-predictions and Dolly arms the parent round left un-gated. The exception replicates the parent anomaly — the masked-context Betley read pools −5.11 (p = 1.0), the penalty-selection mechanism from the refit round. Dolly ordering under pooling: stitched 0.482 vs full prompt 0.245. Two checks tie the rounds together: the recomputed forwards match the parent's persisted vectors per context (cosine floor 0.99), and the paired bootstrap reproduces the parent's last-token gap to machine precision.

### The whole-input read-out deficit persists without family transfer and in the ambient target space

Held-out skill at layer 18 on the UltraChat grid under three fold regimes — both axes unseen (the headline pooling), seen families × unseen queries, unseen families × seen queries — for four read-out arms. Black dots overlay the per-held-out-family skills (seven context families) on the both-axes-unseen bars; the fit persisted only pooled sums for the two marginal regimes, so their bars carry no per-family dots.

![Held-out skill at layer 18 under three fold regimes for four read-out arms on the UltraChat grid, with per-held-out-family dots on the headline bars](https://raw.githubusercontent.com/superkaiba/explore-persona-space/80ad2d5f92ebe97b4a9577ba7426e0147b098fa9/figures/issue_923/regimes_L18_uc.png)

> **Figure.** *Family transfer amplifies the whole-input read-out deficit but is not required for it.* Full prompt vs stitched pair: 0.460 vs 0.540 with both axes unseen; 0.502 vs 0.554 on seen families × unseen queries; 0.559 vs 0.711 under pure family transfer (Betley seen-family marginal: 0.534 vs 0.556). Dots: the seven held-out context families behind each both-axes-unseen bar.

The stitched advantage is not purely a format-family transfer failure. The registered ambient-space secondary read agrees: the ordering survives without the PCA-48 target compression, at lower magnitude (stitched 0.347 vs full prompt 0.295 UltraChat; 0.321 vs 0.297 Betley). Both reads recur under the span-mean summary: seen families × unseen queries 0.585 vs 0.348, and ambient-space 0.365 vs 0.179 (UltraChat) / 0.276 vs 0.179 (Betley).

> **Superseded (mixed-forward round).** These robustness reads are pre-mixed-round results scoped to the last-token and whole-span read-outs; the mixed-forward round supersedes their no-mixing interpretation — at matched query-span granularity the same forward's deficit disappears — while the measurements themselves stand.

### The query, not the context, carries most of the per-cell answer profile

In-sample variance decomposition of the per-cell target into query, context, and interaction shares per layer (0–27) in the fitted PCA-48 target space; UltraChat left, Betley right, layer 18 marked.

![In-sample variance shares of the per-cell mean answer activation by layer for query main effect, context main effect, and interaction, UltraChat and Betley grids](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d786a8bde53aa985d23e85d5f0b84b0246654e3/figures/issue_923/anova_shares.png)

> **Figure.** *Nearly additive in the fitted space, and the query owns most of it.* Layer-18 PCA-48 shares: query 0.837 / context 0.078 / interaction 0.086 (UltraChat); 0.809 / 0.094 / 0.097 (Betley); ambient-space interaction shares roughly double, 0.153 and 0.191. In-sample reference, not a held-out claim.

The context-only read trails the query-only read by −0.355 (cross-classified interval −0.426 to −0.285), reversing the context-dominance expectation; under the span-mean summary the gap is −0.392 (interval −0.462 to −0.326). The reversal is bounded by the battery's realized context strength — several contexts barely alter the completion text (the French in-context-learning family: 0 of 96 French completions; the refusal-behavior context: 1 of 96 refusal-like openings) — so the shares are relative to this 50-context battery. The context factor is fully recovered (read 0.093 vs ceiling 0.078, within the fold-basis tolerance); the query-only read recovers about half its ceiling. No conflict with the parent issues' probe-averaged results, where the query axis is averaged out.

### The decomposition replicates on misalignment-genre queries and transfers to human-written out-of-distribution queries

Left: pooled held-out skill R² per arm at layer 18 on the Betley grid (n = 2400 cells; 95% family-bootstrap intervals; per-held-out-family dots overlaid). Right: the out-of-distribution arm — fits trained on non-held families × all 144 UltraChat queries, scored on held-family × 48 Dolly queries (n = 2400 cells pooled over 7 family folds; per-family resolution not persisted for this arm).

![Betley-grid arm skills with per-family dots, and Dolly out-of-distribution arm skills, both at layer 18](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d786a8bde53aa985d23e85d5f0b84b0246654e3/figures/issue_923/betley_ood_L18.png)

> **Figure.** *Same ordering on both surfaces.* Betley: stitched 0.539 vs full prompt 0.498 (residual −0.041, interval −0.102 to −0.011); the decomposability fraction is defined here (denominator 0.079, guard 0.053) and reads 1.52 — the stitched pair overshoots the full-prompt reference. Dolly: stitched 0.489 vs full prompt 0.448.

The stitched-over-full ordering, the query-over-context asymmetry (Betley difference −0.299, cross-classified interval −0.395 to −0.207), and the near-additive variance structure all replicate. Because the UltraChat fraction is guard-suppressed, Betley's 1.52 is a secondary genre-specific ratio, not the headline decomposability estimate. Fifteen of sixteen registered arm × genre reads beat their layer-18 permutation nulls at p ≈ 0.001 (n = 1000 permutations) in the parent round; the exception is the masked-context Betley read (p = 1.0, below). The follow-up round extended null coverage to the blended-predictions and Dolly arms (25 of 26 pooled reads outside their bands, above).

### The ordering holds at every layer

The figure shows pooled held-out skill R² per layer (0–27) for all nine arms on the UltraChat grid under the last-token summary, with the plan-frozen layer 18 marked; this is itself the per-layer data view (Betley curves are in the committed figure set).

![Held-out skill by layer for all arms on the UltraChat grid](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d786a8bde53aa985d23e85d5f0b84b0246654e3/figures/issue_923/layer_curves_uc.png)

> **Figure.** *Layer 18 is representative, not selected.* Stitched-pair skill exceeds the full prompt at 28 of 28 layers (minimum gap 0.057 UltraChat, 0.008 Betley); healthy arms peak around layers 18–24; the context-only read stays below 0.1 at every layer on UltraChat (0.121 at layer 18 on Betley).

The headline was read at the plan-frozen layer, so no selection correction applies; the swept maxima (stitched 0.587 at layer 19) clear their per-draw max-over-layers permutation bands, whose 99th percentiles sit below zero skill.

### The masked-context query read collapses on Betley through penalty selection that cannot anticipate the family transfer, not missing signal

First figure: pooled held-out skill R² at layer 18 for the three query-only presentations and their stitched variants, Betley grid. Second figure, the per-penalty data behind the collapse: the masked-context read at each fixed ridge penalty, both grids.

![Query presentation arms on the Betley grid at layer 18](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d786a8bde53aa985d23e85d5f0b84b0246654e3/figures/issue_923/presentations_L18_betley.png)

![Pooled skill versus fixed ridge penalty for the masked-context read](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d786a8bde53aa985d23e85d5f0b84b0246654e3/figures/issue_923/qry_iii_lambda_L18.png)

> **Figure.** *A penalty-selection failure, not absent signal.* The masked-context read pools −3.03 on Betley (inside its null band, p = 1.0) but reads 0.407 at the largest fixed penalty — in family with the empty-system read (0.419) — and is healthy on UltraChat at every penalty (0.454–0.455).

The read fails only where penalty selection must extrapolate: within-train-fold validation prefers small-to-mid penalties on Betley's 36-query folds, while only heavy shrinkage survives the family transfer no train-fold selector sees. The pooled number stands as measured; the UltraChat presentation match (0.455 vs 0.448/0.457) still licenses the attention-mixing reading of the residual over template mismatch. The collapse recurs under the span-mean summary (−5.11 pooled on Betley, p = 1.0; healthy 0.462 on UltraChat), as this mechanism predicts.

### Group-level penalty re-selection does not rescue the Betley masked read and degrades the stitched arm

Pooled held-out skill R² at layer 18 for the masked-context query-only and stitched arms on both grids under three penalty-selection rules — pointwise leave-one-out (the production selector), leave-query-group-out, and fixed λ = 1000 — from the folded refit round. Black dots overlay the 28 per-fold skills behind each pooled bar (axis log-compressed below −1 so the deep Betley folds stay visible); the lower panel shows each fold's selected λ for the two data-driven rules.

![Pooled skill under three penalty-selection rules for the masked arms, with per-fold skill dots and a per-fold selected-lambda panel](https://raw.githubusercontent.com/superkaiba/explore-persona-space/80ad2d5f92ebe97b4a9577ba7426e0147b098fa9/figures/issue_923/qryiii_selection_rules_L18.png)

> **Figure.** *Group-level selection does not rescue the Betley read and degrades the stitched arm.* Masked-context Betley: −3.03 pointwise, −2.49 group-level, 0.407 fixed. The stitched arm falls 0.531 to 0.064 as selection moves from λ = 1000 (26 of 28 pointwise folds) to λ = 10 (all 28); UltraChat arms selector-invariant (0.455 / 0.548). Dots: per-fold skills; lower panel: selected λ.

Query-group re-selection, which the arm's near-duplicate rows (same query, position-shifted) cannot fool, recovers only a sixth of the gap and newly degrades the stitched arm, so pointwise duplicate leakage is not the driver. Group-level validation prefers λ = 10 on all 28 Betley folds for both masked arms; no within-train-fold selection grain reaches the heavy shrinkage the family transfer demands, and the refit's residual leakages — both biased toward smaller penalties — cannot manufacture the non-rescue.

### Store-reduced and fresh-captured targets agree at the median, with a small tail of divergent regenerations

Histogram over the 50 regeneration spot-check cells (25 per genre): per-cell cosine between the freshly regenerated mean answer activation and the store-reduced one, averaged over 28 layers, with the 0.99 level marked.

![Histogram of per-cell cosine agreement between regenerated and store-reduced targets across 50 spot-check cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d786a8bde53aa985d23e85d5f0b84b0246654e3/figures/issue_923/regen_check_hist.png)

> **Figure.** *Median agreement 0.995; 14 of 50 cells fall below 0.99, worst 0.880.* Both genres agree equally (mean of per-cell means 0.990 in each). The histogram summarizes each cell's mean over layers.

At the per-layer grain the divergence is deeper: per-cell minimum-over-layers cosines reach 0.72 (25 of 50 below 0.99), and the worst layer-18 cosine is 0.84. The tail is expected under greedy regeneration: batching numerics can flip one token, after which the completion — and its activation — legitimately diverges. The check is consistent with joining store-reduced targets (original 48 queries per genre) with fresh-captured ones (96 extension queries); a provenance offset would be shared by every arm, depressing rather than creating the arm ordering.

---

**Repro:** GPU capture phase ~8 GPU-h on 4× A100-80 (GCP `ft-7b` auto lane, 3 attempts incl. 2 crash-fix rounds); CPU reduce + fits on a GCP `e2-standard-8` (~10 h wall, 0 GPU); no model training. Workload code at issue-923 branch commit `e9c8809113` (`scripts/issue923_{build_inputs,capture,reduce_spans,fit_decomposition,figures}.py`, dispatched via `issue923_gpu_phase.sh` / `issue923_cpu_phase.sh`); fit self-test PASS (PRESS/dual max abs ΔMSE 4.4e-15); group-λ refit `scripts/issue923_qryiii_group_lambda.py` at issue-923 commit `c56111cea5` (reproduces the persisted pointwise numbers to max abs Δ 8.0e-15 on all 4 cells). Pooled-span-features round: workload `scripts/issue923_pooled_cpu_phase.sh` (fits via `issue923_fit_decomposition.py` with `feature_source=pool`) at issue-923 commit `f4727f8aad`; GPU capture ~1.25 GPU-h on 1× A100-80 (flex-start, ~24,500 prefill-only forwards, identity gate passed), CPU fits + nulls + figures on a GCP `e2-standard-8` (~15.4 h wall, 0 GPU; the wall-time overrun vs the plan's estimate was acknowledged mid-run as a continue-as-is compute deviation — batched inner loop, FLOP-bound at 8 vCPU); body figures for the round regenerated by `scripts/issue923_fig_pooled_body.py` at issue-923 commit `c556078dd5`. Config slugs: arms `arm_ctx`, `arm_qry_{i,ii,iii}`, `arm_concat_{i,ii,iii}`, `arm_blend`, `arm_full`; genres `uc`, `betley`; presentations (i) empty-system, (ii) no-system-block, (iii) masked-context; mask_backend `sdpa`; feature sources `flast` (parent round), `pool` (follow-up round). Artifacts: aggregated stats [headline.json](https://github.com/superkaiba/explore-persona-space/blob/f8ffb59899f7ae5437d8bbb6439997931f4de4c4/eval_results/issue_923/fits/headline.json), nulls [null_summary.json](https://github.com/superkaiba/explore-persona-space/blob/f8ffb59899f7ae5437d8bbb6439997931f4de4c4/eval_results/issue_923/fits/null_summary.json), variance shares [anova_shares.json](https://github.com/superkaiba/explore-persona-space/blob/f8ffb59899f7ae5437d8bbb6439997931f4de4c4/eval_results/issue_923/fits/anova_shares.json), regen [regen_check.json](https://github.com/superkaiba/explore-persona-space/blob/f8ffb59899f7ae5437d8bbb6439997931f4de4c4/eval_results/issue_923/fits/regen_check.json), per-cell/per-fold breakdown [decomposition_skill.json](https://github.com/superkaiba/explore-persona-space/blob/f8ffb59899f7ae5437d8bbb6439997931f4de4c4/eval_results/issue_923/fits/decomposition_skill.json) (17 MB — the low-level file behind every aggregate, incl. per-λ skills, per-cell predictions, and the seen-family/pure-family marginal and ambient-space skills quoted above), group-λ refit [qryiii_group_lambda.json](https://github.com/superkaiba/explore-persona-space/blob/c56111cea5dd410ae0615a5a9d14f561932144e4/eval_results/issue_923/fits/qryiii_group_lambda.json) (per-fold λ selections and per-λ validation curves under both selection rules); pooled-round stats [fits_pooled/headline.json](https://github.com/superkaiba/explore-persona-space/blob/c556078dd5f100923dc9dcc16c823b04531206a8/eval_results/issue_923/fits_pooled/headline.json) (incl. the paired last-token-vs-span-mean bootstrap and the parent-gap reproduce check), [fits_pooled/null_summary.json](https://github.com/superkaiba/explore-persona-space/blob/c556078dd5f100923dc9dcc16c823b04531206a8/eval_results/issue_923/fits_pooled/null_summary.json), [fits_pooled/decomposition_skill.json](https://github.com/superkaiba/explore-persona-space/blob/c556078dd5f100923dc9dcc16c823b04531206a8/eval_results/issue_923/fits_pooled/decomposition_skill.json) (18 MB, per-cell/per-fold under the span-mean summary, all regimes); figures [figures/issue_923/](https://github.com/superkaiba/explore-persona-space/tree/c556078dd5f100923dc9dcc16c823b04531206a8/figures/issue_923) (reader-facing set regenerated with plain-English labels in round 2; fold-regimes + selection-rules panels regenerated with per-unit overlays in round 3 via `scripts/issue923_fig_{regimes,qryiii_selection}.py` at issue-923 commit `fd209c82f1`; pooled-round panels via `issue923_fig_pooled_body.py`, same underlying JSONs); rollout text + feature/target packs + null matrices on the [HF data repo @ issue923_ctx_query_decomposition (pinned, parent round)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/77d04e45096e120ca897fdb7b22730e157ce00a2/issue923_ctx_query_decomposition) (raw_completions 131 files, analysis_tensors 53, eval_results 61, figures 33; Hub-verified at write time) and the pooled round's artifacts at the [later pinned revision](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ecf9b61367b24b20fc88647c9cddde1e75127216/issue923_ctx_query_decomposition) (pooled feature packs 40 files + `identity_check.json`, fits_pooled 60 files, figures_pooled 93 files; Hub-verified at write time). Mixed-forward-span-stitch round: workload `scripts/issue923_mixed_gpu_phase.sh` (capture: ~12,000 unmasked prefill-only forwards + identity gate, ~35 min on 1× A100-80 flex-start, GCP job 2752162201034264387) then `scripts/issue923_mixed_cpu_phase.sh` (two-arm fits + nulls + paired reads + figures, ~4.0 h on a GCP `e2-standard-8`, 0 GPU, GCP job 2348660026553925284), both at issue-923 commit `232b1d990c` (~0.6 GPU-h used of 2 budgeted); mixed-round stats [mixed-forward-span-stitch/headline.json](https://github.com/superkaiba/explore-persona-space/blob/7f1ac4ad24cd8340851b439282a4d3ea80ece6cf/eval_results/issue_923/mixed-forward-span-stitch/headline.json) (the four paired reads with per-family gaps, shared-draw bootstrap, and the exact reproduce check), [null_summary.json](https://github.com/superkaiba/explore-persona-space/blob/7f1ac4ad24cd8340851b439282a4d3ea80ece6cf/eval_results/issue_923/mixed-forward-span-stitch/null_summary.json) (both new arms × three surfaces), [decomposition_skill.json](https://github.com/superkaiba/explore-persona-space/blob/7f1ac4ad24cd8340851b439282a4d3ea80ece6cf/eval_results/issue_923/mixed-forward-span-stitch/decomposition_skill.json) (4 MB per-cell/per-penalty; `anova_shares.json` is a skipped-by-design stub — targets identical to the parent round); HF prefixes at the [mixed-round pinned revision](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/57216e2978293700fa840fee9451edab6313bb5d/issue923_ctx_query_decomposition): `eval_results/fits_mixed/` 60 files, `analysis_tensors/mixed_capture/` 18 files (16 feature packs + `identity_check_mix.json` + upload sentinel), `analysis_tensors/fits_mixed/` 2 null matrices, `figures_mixed/` 120 files (all Hub-verified at write time); the four body figures for the round are the workload's `issue923_fig_mixed` outputs, committed under [figures/issue_923/](https://github.com/superkaiba/explore-persona-space/tree/7d861696b4b222181db33016ade7172ebc46b91e/figures/issue_923) at `7d861696b4`; the dilution panel was re-rendered in round 5 with a plain-English title (identical underlying data — sidecar gap arrays unchanged against the `7d861696b4` sidecar) and re-pinned at main commit `25e11eb3a4`, and the workload's layer-sweep panel (persisted curves, no re-analysis) was committed for the body at main commit `c17d054b9d`. Reused artifacts — answer-span stores + probe pools from [#658](https://eps.superkaiba.com/tasks/658) and its UltraChat-genre round ([store, pinned](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/77d04e45096e120ca897fdb7b22730e157ce00a2/issue658_theory_assumptions)) — fit: same base model, on-policy greedy completions with all-28-layer spans, all 50×48×2 cells present, pinned by store manifest + dataset revision; the 50-context battery + UltraChat probe builder from [#594](https://eps.superkaiba.com/tasks/594) (`data/issue594/battery.json`, in git); the pooled round additionally reuses the parent round's target packs unchanged — fit: same targets by construction (the round varies only the prompt-side summary), Hub-pinned at the parent revision. Declared discard (plan §10): per-token span tensors for the new arms (~390 GB); regen recipe = one teacher-forced forward over the persisted rollout text.

**Context:** created 2026-07-03; results landed 2026-07-03 (fits done 2026-07-04Z); pooled-span-features round run 2026-07-04 → 2026-07-05Z; mixed-forward-span-stitch round run 2026-07-05Z. Parent: [#658](https://eps.superkaiba.com/tasks/658) (the store + context battery this reads). Two crash-fix rounds, neither material to the science: the context-vector identity probe threshold recalibrated to bf16 batching numerics (hard floor 0.99, sub-0.999 recorded as warnings), and conditional `.env` sourcing in the CPU-phase script on GCE. Free-analysis same-issue follow-up round `qryiii-group-lambda-refit` (run 2026-07-04, zero GPU): leave-query-group-out penalty re-selection over the persisted layer-18 feature/target packs, revising the masked-context-collapse mechanism in place. Refit scope caveat: it holds out query groups whose contexts remain in the train complement (unseen queries under seen families, versus the production unseen-family × unseen-query test), and the part-PCA bases and standardization still see the held-out group — both residual leakages bias selection toward smaller penalties, so neither can produce the observed non-rescue. Same-issue follow-up round `pooled-span-features` (proposer-initiated cheap band, `source: proposer-9b-cheap`; ~1.25 GPU-h used of 3 budgeted): one variable changed — the prompt-side feature summary statistic, last token → mean over the owning token span, uniformly across arms; targets reused unchanged from the parent round. Registered falsification clause, verbatim from the proposal: "If the span-mean full-prompt read still trails the span-mean stitched read with a family-bootstrap interval excluding 0, the slot-artifact explanation is dead and the 'no attention-mixing advantage for linear read-out' Takeaway hardens from estimator-scoped to summary-robust." Same-issue follow-up round `mixed-forward-span-stitch` (proposer-initiated cheap band, `source: proposer-9b-cheap`; run 2026-07-05; ~0.6 GPU-h used of 2 budgeted): one variable — attention from the query span to the context, ON vs OFF at matched tokens, absolute positions, pooled span, and summary. Registered direction (a), verbatim from the proposal: "arm_qry_mix > pool_qry_iii with CI excluding 0 → attention mixing DOES add linearly readable information once the whole-input dilution and slot confounds are removed — Takeaways 1–2 are rewritten as summary-granularity-specific and the headline claim is overturned." Direction (a) fired: this body's title, Takeaways, and Broader-narrative clause are rewritten accordingly; the two prior rounds' measurements stand as measured. Originating prompt, verbatim:

> Help me to plan this:
>
> We want to see if there is some decomposition of single query mapping into the "context" portion and the "query" portion
>
> For this we can:
> - directly predict the answer just from the context portion
> - directly predict the answer just from the query portion (with blank context --> make sure to not insert system prompt -- try empty system prompt but also removing the system prompt part of chat template completely......, also potentially just masking out the tokens of the context with the query in there)
> - context and query must be disjoint token sets
> See if by some combination of these 2 mappings we can get better performance
> Also train the context + query -> answer mapping and analyze the relationship between this mapping and the 2 other mappings
>
> probably good to use our diverse contexts + ultrachat queries setup so you get:
> M_A: context -> answer
> M_B: query -> answer
> M_C: context + query --> answer
> with matched contexts and queries
> All generations should be on policy
> Evaluate mappings on LOFO context + OOD queries

The recorded correction, verbatim, supersedes the first body's training framing:

> No we want to predict mean answer activation from last context vector/last query vector. Look at past issues to understand. THere should be no training

<!-- round-4 revision addresses review concerns r3-caption-everywhere-below-false (Result 3 caption re-quantified), r3-takeaway2-disjoint-level-or-better-uc-only (Takeaway 2 scoped per genre), r3-span-length-dilution-alternative-unnamed (dilution clause in Takeaway 4 + Result 2), r3-roughly-halves-is-39pct (Takeaway 2), r3-stays-guard-undefined-betley-was-defined (Result 2) -->

<!-- round-5 revision addresses review concerns r5-supersede-stale-no-mixing-prose (the span-mean-swap and fold-regime results scoped as pre-mixed-round readings via Superseded notes), r5-representation-deficit-overclaim (Takeaway 2 weakened to linear-readability scope), r5-broader-narrative-genre-scope and r5-takeaway3-near-lossless-betley (UC-only ±0.012 scoped; Betley +0.105 attributed to the penalty-degraded masked presentation), r5-layer-profile-hedge (new dedicated per-layer result embedding the workload layer-sweep panel, persisted curves only), r5-dilution-bundles-snr, r5-dilution-caption-025-plus-pink-bands, r5-figure-title-opaque-code (dilution panel re-rendered, same data, re-pinned at 25e11eb3a4) -->

<!-- Acknowledged verifier WARNs: several results sit between the 120-word soft cap and the 180-word hard cap, some Takeaways bullets exceed 30 words, and total content prose exceeds the word budget (including the per-round allowances for the folded qryiii-group-lambda-refit, pooled-span-features, and mixed-forward-span-stitch rounds) — this body now reports three measurement rounds plus a free refit: 9 planned arms x 2 genres + an OOD arm, a fold-regime robustness read, an anomaly investigation, a data-validity control, the group-lambda refit, the pooled-summary falsification test, and the mixed-forward matched-granularity test that overturned the headline; each carries its own three-beat result. Goal frontmatter lives in the task's body.md frontmatter, not this cache file. -->
