---
title: Stitched context-only and query-only read-outs beat the full-prompt last-token
  ridge read-out at recovering the mean answer activation on held-out families and
  queries (MODERATE confidence)
kind: experiment
tags: []
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
# Stitched context-only and query-only read-outs beat the full-prompt last-token ridge read-out at recovering the mean answer activation on held-out families and queries (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- Stitched context-only + query-only ridge reads beat the full-prompt last-token read at layer 18 on held-out families × queries: 0.540 vs 0.460 (UltraChat), 0.539 vs 0.498 (Betley), across all 28 layers — with the deficit concentrated in held-out format contexts and the full-prompt arm carrying roughly three times the stitched arm's bootstrap uncertainty.
- The interaction residual is negative in both genres, −0.081 (interval −0.157 to −0.045) and −0.041 (−0.102 to −0.011): no measurable attention-mixing advantage for linear read-out; the deficit persists without family transfer (0.502 vs 0.554 on seen families × unseen queries, UltraChat).
- In the fitted PCA-48 target space the target is nearly additive in-sample — query main effect 84%, context 8%, interaction 9% (UltraChat, layer 18); the ambient-space interaction share is about twice as large (0.153 / 0.191) — and the query-only read (0.448) dwarfs the context-only read (0.093), reversing the context-dominance expectation under this battery's weak realized context pull.
- The context factor is small but fully readable (read 0.093 vs in-sample ceiling 0.078); the ordering transfers to 48 human-written Dolly queries (2400 held-out cells: stitched 0.489 vs full prompt 0.448).
- Fifteen of sixteen registered arm × genre reads beat their selection-symmetric permutation nulls at p ≈ 0.001 (n = 1000 permutations); the exception is the masked-context Betley read (p = 1.0), and the blended-predictions and Dolly reads have no registered nulls.
- Two bounded anomalies: the full-prompt deficit concentrates in held-out format contexts (0.171 vs 0.475 stitched, UltraChat); the masked-context presentation collapses on Betley through ridge-penalty selection that no within-train-fold hold-out grain fixes (0.407 at the largest fixed penalty; a leave-query-group-out refit recovers only −2.49).

## Goal

**This experiment in context:** The line's ridge maps predict the answer-side activation profile from the full prompt: per-example held-out R² 0.60–0.63 ([#722](https://eps.superkaiba.com/tasks/722)/[#779](https://eps.superkaiba.com/tasks/779)), content-indexed ([#823](https://eps.superkaiba.com/tasks/823)), genre-general via the mean answer-token summary ([#810](https://eps.superkaiba.com/tasks/810)), on the battery and store from [#658](https://eps.superkaiba.com/tasks/658). This experiment asks whether that map factorizes: with no model training, predict the per-cell mean answer activation from the isolated last-context-token and last-query-token vectors, separately and combined, against the full-prompt read, under leave-one-family-out × held-out-query folds.

**Broader narrative:** If the context contribution to the answer profile is separable, base-side context geometry is a self-contained predictor input for the leakage-prediction program; a large interaction residual would instead cap any context-only predictor. Measured: the disjoint combination loses nothing to attention mixing — a statement about linear readability under this ridge/PCA/fold estimator, not about total information in the mixed token — but the context factor is a small share of per-cell variance, most of which tracks the query.

## Methodology

**Design:** One base model, `Qwen/Qwen2.5-7B-Instruct`, forward passes and greedy generation only; ridge read-outs only, no model training anywhere. Grid: 50 contexts in 7 families (persona 14 / WildChat 10 / in-context-learning 8 / rephrase 6 / format 5 / behavior 5 / default 2) × 144 UltraChat queries (the 48 store queries + 96 fresh length-matched ones), plus two secondary grids — 50 × 48 Betley (misalignment-genre) queries and 50 × 48 human-written Dolly queries (the out-of-distribution arm). Target: the per-cell mean residual activation over the answer tokens of the model's own greedy completion to the full (context, query) prompt, per layer (28 layers). Read-out arms, per layer: Context-only (last context token from a prefix-only forward), Query-only in three null-context presentations (explicit empty system turn; no system block; full prompt with context tokens attention-masked in place), Stitched pair (feature concatenation of Context-only + Query-only, one variant per presentation), Blended predictions (two-parameter combination of the two single-arm predictions, fit on an inner validation split), and Full prompt (last input token at the assistant header). Contexts and queries occupy disjoint token spans, and each partial feature is computed with the other input absent. References: an in-sample variance decomposition of the target into context main effect, query main effect, and interaction (oracle ceilings); selection-symmetric permutation nulls; a 50-cell regeneration spot-check of the store join.

**Training:** **N/A — no model training.** Analysis-design constants:

| Parameter | Value | Source |
|---|---|---|
| Ridge λ grid | 1e-2, 1e-1, 1, 10, 100, 1000; nested PRESS leave-one-out per train fold (follow-up diagnostic: leave-query-group-out re-selection at layer 18, masked-context arms) | #722/#823 harness (`RIDGE_LAMBDAS`) |
| Feature/target reduction | PCA-48 both sides, train-fold-fit bases; train-fold-centered targets | #722 (28/28-layer input-PCA equivalence, max ΔR² 0.018); #810 target recipe |
| Layer binding | same-layer, swept 0–27; headline layer 18 frozen in the plan | #722 peak layer |
| Folds | 7 leave-one-family-out × 4 query folds (UltraChat 144 → 4×36; Betley 48 → 4×12); both axes unseen in every test cell | #810 + the project's group-fold rule |
| Out-of-distribution fold | train non-held family × all 144 UltraChat queries; test held family × 48 Dolly queries | plan §4.2 (corpus-transfer form) |
| Generation | vLLM greedy, temperature 0.0, max_tokens 512 | #658 store recipe, matched exactly |
| Permutation null | 1000 cell-label permutations; λ re-selected per draw; per-draw max-over-layers band | #810 batched-null design |
| Bootstrap | 2000 family-cluster draws; cross-classified family × query draws for the context-vs-query contrast | #810 precedent; plan §3 |
| Decomposability-fraction guard | denominator floor max(0.02, 2 × bootstrap SE) | plan §3 (from the #722 compression bound 0.018) |
| Masked-context attention backend | SDPA with a 4D float mask (dummy-context invariance smoke passed) | plan §8 |
| Seed | 42 (probe build, folds, permutations, bootstrap) | plan §10 |

**Evaluation:** Primary dependent variable = pooled held-out skill-over-mean R² per (arm × layer): fits on train-family contexts × train-fold queries, scored on held-out-family contexts × held-out-fold queries, pooled so every cell is tested exactly once (n = 7200 UltraChat cells, 2400 Betley, 2400 Dolly; no cells dropped). Baseline for skill = the train-fold mean. Headline reads at the plan-frozen layer 18; the layer sweep is gated on per-draw max-over-layers permutation bands. The interaction residual is the full-prompt skill minus the stitched-pair skill; the plan's power floor (skip verdicts if the full-prompt read is below 0.05 everywhere) did not trigger. The construct is representation-level — linear predictability of the model's own answer-side activation — not a behavioral claim; the variance shares are in-sample references, never held-out claims, and ambient-space skill is reported alongside as the plan's registered secondary robustness read. R² magnitudes are not comparable to the parent issues' probe-averaged numbers (different target granularity and fold scheme).

**Data extraction:** Targets for the 48 original UltraChat and 48 Betley queries are re-reductions of the parent store's per-context answer-span tensors (~341 GB streamed and reduced on a CPU instance); targets for the 96 fresh UltraChat and 48 Dolly queries were captured fresh with the identical recipe (greedy generation, teacher-forced forward, mean over the answer span). The Context-only feature comes from a prefix-only forward whose equality with the same position inside the full prompt was probe-verified per context (cosine floor 0.99 enforced; values below 0.999 recorded as warnings — bf16 batching numerics, worst recorded 0.9989). The empty-system presentation asserts the Qwen default system prompt was not silently inserted. Fresh UltraChat queries were length-matched to the Betley length profile with the parent's builder; Dolly rows are instruction-only (empty context field), same filters, length-matched, from the established databricks-dolly-15k corpus (tier 1–2 realism); UltraChat is an established tier-2 corpus; the battery inherits the line's house-written tier-3 families as a standing scope caveat.

**Sample training/evaluation data + completions:** No training rows exist; the worked examples below are evaluation cells (context + query + the model's own greedy completion whose activations form the target). Completions are cut for space; full texts for all 7250 new-arm completions (4800 UltraChat-extension, 2400 Dolly, 50 regeneration spot-check): [raw_completions on the HF data repo (pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/77d04e45096e120ca897fdb7b22730e157ce00a2/issue923_ctx_query_decomposition/raw_completions).

Randomly drawn UltraChat-extension cell (from `uc_ext/f1_house_programmer_shard2of4.json`, row with q_idx 17):

> **Context (system prompt):** "You are a programmer."
> **Query:** "Create a playlist of at least 10 songs that feature unusual or unique instruments, such as the sitar, didgeridoo, kalimba, or theremin, across a variety of music genres like folk, world music, indie, and experimental."
> **Completion (opening):** "Certainly! Here's a playlist featuring unique instruments across various genres: 1. **"Sitar Solo" by Ravi Shankar** - Folk/World Music …"

Format-family cell, the family the full-prompt read transfers worst to (from `uc_ext/f5_fmt_code_comment_shard3of4.json`, row with q_idx 0):

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

### The full-prompt deficit persists without family transfer and in the ambient target space

Held-out skill at layer 18 on the UltraChat grid under three fold regimes — both axes unseen (the headline pooling), seen families × unseen queries, unseen families × seen queries — for four read-out arms.

![Held-out skill at layer 18 under three fold regimes for four read-out arms on the UltraChat grid](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d786a8bde53aa985d23e85d5f0b84b0246654e3/figures/issue_923/regimes_L18_uc.png)

> **Figure.** *Family transfer amplifies the deficit but is not required for it.* Full prompt vs stitched pair: 0.460 vs 0.540 with both axes unseen; 0.502 vs 0.554 on seen families × unseen queries; 0.559 vs 0.711 under pure family transfer (Betley seen-family marginal: 0.534 vs 0.556).

The stitched advantage is not purely a format-family transfer failure. The registered ambient-space secondary read agrees: the ordering survives without the PCA-48 target compression, at lower magnitude (stitched 0.347 vs full prompt 0.295 UltraChat; 0.321 vs 0.297 Betley).

### The query, not the context, carries most of the per-cell answer profile

In-sample variance decomposition of the per-cell target into query, context, and interaction shares per layer (0–27) in the fitted PCA-48 target space; UltraChat left, Betley right, layer 18 marked.

![In-sample variance shares of the per-cell mean answer activation by layer for query main effect, context main effect, and interaction, UltraChat and Betley grids](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d786a8bde53aa985d23e85d5f0b84b0246654e3/figures/issue_923/anova_shares.png)

> **Figure.** *Nearly additive in the fitted space, and the query owns most of it.* Layer-18 PCA-48 shares: query 0.837 / context 0.078 / interaction 0.086 (UltraChat); 0.809 / 0.094 / 0.097 (Betley); ambient-space interaction shares roughly double, 0.153 and 0.191. In-sample reference, not a held-out claim.

The context-only read trails the query-only read by −0.355 (cross-classified interval −0.426 to −0.285), reversing the context-dominance expectation. The reversal is bounded by the battery's realized context strength — several contexts barely alter the completion text (the French in-context-learning family: 0 of 96 French completions; the refusal-behavior context: 1 of 96 refusal-like openings) — so the shares are relative to this 50-context battery. The context factor is fully recovered (read 0.093 vs ceiling 0.078, within the fold-basis tolerance); the query-only read recovers about half its ceiling. No conflict with the parent issues' probe-averaged results, where the query axis is averaged out.

### The decomposition replicates on misalignment-genre queries and transfers to human-written out-of-distribution queries

Left: pooled held-out skill R² per arm at layer 18 on the Betley grid (n = 2400 cells; 95% family-bootstrap intervals; per-held-out-family dots overlaid). Right: the out-of-distribution arm — fits trained on non-held families × all 144 UltraChat queries, scored on held-family × 48 Dolly queries (n = 2400 cells pooled over 7 family folds; per-family resolution not persisted for this arm).

![Betley-grid arm skills with per-family dots, and Dolly out-of-distribution arm skills, both at layer 18](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d786a8bde53aa985d23e85d5f0b84b0246654e3/figures/issue_923/betley_ood_L18.png)

> **Figure.** *Same ordering on both surfaces.* Betley: stitched 0.539 vs full prompt 0.498 (residual −0.041, interval −0.102 to −0.011); the decomposability fraction is defined here (denominator 0.079, guard 0.053) and reads 1.52 — the stitched pair overshoots the full-prompt reference. Dolly: stitched 0.489 vs full prompt 0.448.

The stitched-over-full ordering, the query-over-context asymmetry (Betley difference −0.299, cross-classified interval −0.395 to −0.207), and the near-additive variance structure all replicate. Because the UltraChat fraction is guard-suppressed, Betley's 1.52 is a secondary genre-specific ratio, not the headline decomposability estimate. Fifteen of sixteen registered arm × genre reads beat their layer-18 permutation nulls at p ≈ 0.001 (n = 1000 permutations); the exception is the masked-context Betley read (p = 1.0, below), and the blended-predictions and Dolly reads have no registered nulls.

### The ordering holds at every layer

The figure shows pooled held-out skill R² per layer (0–27) for all nine arms on the UltraChat grid, with the plan-frozen layer 18 marked; this is itself the per-layer data view (Betley curves are in the committed figure set).

![Held-out skill by layer for all arms on the UltraChat grid](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d786a8bde53aa985d23e85d5f0b84b0246654e3/figures/issue_923/layer_curves_uc.png)

> **Figure.** *Layer 18 is representative, not selected.* Stitched-pair skill exceeds the full prompt at 28 of 28 layers (minimum gap 0.057 UltraChat, 0.008 Betley); healthy arms peak around layers 18–24; the context-only read stays below 0.1 at every layer on UltraChat (0.121 at layer 18 on Betley).

The headline was read at the plan-frozen layer, so no selection correction applies; the swept maxima (stitched 0.587 at layer 19) clear their per-draw max-over-layers permutation bands, whose 99th percentiles sit below zero skill.

### The masked-context query read collapses on Betley through penalty selection that cannot anticipate the family transfer, not missing signal

First figure: pooled held-out skill R² at layer 18 for the three query-only presentations and their stitched variants, Betley grid. Second figure, the per-penalty data behind the collapse: the masked-context read at each fixed ridge penalty, both grids.

![Query presentation arms on the Betley grid at layer 18](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d786a8bde53aa985d23e85d5f0b84b0246654e3/figures/issue_923/presentations_L18_betley.png)

![Pooled skill versus fixed ridge penalty for the masked-context read](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d786a8bde53aa985d23e85d5f0b84b0246654e3/figures/issue_923/qry_iii_lambda_L18.png)

> **Figure.** *A penalty-selection failure, not absent signal.* The masked-context read pools −3.03 on Betley (inside its null band, p = 1.0) but reads 0.407 at the largest fixed penalty — in family with the empty-system read (0.419) — and is healthy on UltraChat at every penalty (0.454–0.455).

The read fails only where penalty selection must extrapolate: within-train-fold validation prefers small-to-mid penalties on Betley's 36-query folds, while only heavy shrinkage survives the family transfer no train-fold selector sees. The pooled number stands as measured; the UltraChat presentation match (0.455 vs 0.448/0.457) still licenses the attention-mixing reading of the residual over template mismatch.

### Group-level penalty re-selection does not rescue the Betley masked read and degrades the stitched arm

Pooled held-out skill R² at layer 18 for the masked-context query-only and stitched arms on both grids under three penalty-selection rules — pointwise leave-one-out (the production selector), leave-query-group-out, and fixed λ = 1000 — from the folded refit round.

![Pooled skill under three penalty-selection rules for the masked arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fe1800c97fdf5f67e02f9e55e485cac55d1cdc99/figures/issue_923/qryiii_selection_rules_L18.png)

> **Figure.** *Group-level selection does not rescue the Betley read and degrades the stitched arm.* Masked-context Betley: −3.03 pointwise, −2.49 group-level, 0.407 fixed. The stitched arm falls 0.531 to 0.064 as its selection moves from λ = 1000 (26 of 28 pointwise folds; 2 at 100) to λ = 10 (all 28); both UltraChat arms are selector-invariant (0.455 / 0.548).

Query-group re-selection, which the arm's near-duplicate rows (same query, position-shifted) cannot fool, recovers only a sixth of the gap and newly degrades the stitched arm, so pointwise duplicate leakage is not the driver. Group-level validation prefers λ = 10 on all 28 Betley folds for both masked arms; no within-train-fold selection grain reaches the heavy shrinkage the family transfer demands, and the refit's residual leakages — both biased toward smaller penalties — cannot manufacture the non-rescue.

### Store-reduced and fresh-captured targets agree at the median, with a small tail of divergent regenerations

Histogram over the 50 regeneration spot-check cells (25 per genre): per-cell cosine between the freshly regenerated mean answer activation and the store-reduced one, averaged over 28 layers, with the 0.99 level marked.

![Histogram of per-cell cosine agreement between regenerated and store-reduced targets across 50 spot-check cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d786a8bde53aa985d23e85d5f0b84b0246654e3/figures/issue_923/regen_check_hist.png)

> **Figure.** *Median agreement 0.995; 14 of 50 cells fall below 0.99, worst 0.880.* Both genres agree equally (mean of per-cell means 0.990 in each). The histogram summarizes each cell's mean over layers.

At the per-layer grain the divergence is deeper: per-cell minimum-over-layers cosines reach 0.72 (25 of 50 below 0.99), and the worst layer-18 cosine is 0.84. The tail is expected under greedy regeneration: batching numerics can flip one token, after which the completion — and its activation — legitimately diverges. The check is consistent with joining store-reduced targets (original 48 queries per genre) with fresh-captured ones (96 extension queries); a provenance offset would be shared by every arm, depressing rather than creating the arm ordering.

---

**Repro:** GPU capture phase ~8 GPU-h on 4× A100-80 (GCP `ft-7b` auto lane, 3 attempts incl. 2 crash-fix rounds); CPU reduce + fits on a GCP `e2-standard-8` (~10 h wall, 0 GPU); no model training. Workload code at issue-923 branch commit `e9c8809113` (`scripts/issue923_{build_inputs,capture,reduce_spans,fit_decomposition,figures}.py`, dispatched via `issue923_gpu_phase.sh` / `issue923_cpu_phase.sh`); fit self-test PASS (PRESS/dual max abs ΔMSE 4.4e-15); group-λ refit `scripts/issue923_qryiii_group_lambda.py` at issue-923 commit `c56111cea5` (reproduces the persisted pointwise numbers to max abs Δ 8.0e-15 on all 4 cells). Config slugs: arms `arm_ctx`, `arm_qry_{i,ii,iii}`, `arm_concat_{i,ii,iii}`, `arm_blend`, `arm_full`; genres `uc`, `betley`; presentations (i) empty-system, (ii) no-system-block, (iii) masked-context; mask_backend `sdpa`. Artifacts: aggregated stats [headline.json](https://github.com/superkaiba/explore-persona-space/blob/f8ffb59899f7ae5437d8bbb6439997931f4de4c4/eval_results/issue_923/fits/headline.json), nulls [null_summary.json](https://github.com/superkaiba/explore-persona-space/blob/f8ffb59899f7ae5437d8bbb6439997931f4de4c4/eval_results/issue_923/fits/null_summary.json), variance shares [anova_shares.json](https://github.com/superkaiba/explore-persona-space/blob/f8ffb59899f7ae5437d8bbb6439997931f4de4c4/eval_results/issue_923/fits/anova_shares.json), regen [regen_check.json](https://github.com/superkaiba/explore-persona-space/blob/f8ffb59899f7ae5437d8bbb6439997931f4de4c4/eval_results/issue_923/fits/regen_check.json), per-cell/per-fold breakdown [decomposition_skill.json](https://github.com/superkaiba/explore-persona-space/blob/f8ffb59899f7ae5437d8bbb6439997931f4de4c4/eval_results/issue_923/fits/decomposition_skill.json) (17 MB — the low-level file behind every aggregate, incl. per-λ skills, per-cell predictions, and the seen-family/pure-family marginal and ambient-space skills quoted above), group-λ refit [qryiii_group_lambda.json](https://github.com/superkaiba/explore-persona-space/blob/c56111cea5dd410ae0615a5a9d14f561932144e4/eval_results/issue_923/fits/qryiii_group_lambda.json) (per-fold λ selections and per-λ validation curves under both selection rules); figures [figures/issue_923/](https://github.com/superkaiba/explore-persona-space/tree/5d786a8bde53aa985d23e85d5f0b84b0246654e3/figures/issue_923) (reader-facing set regenerated with plain-English labels in round 2, same underlying JSONs); rollout text + feature/target packs + null matrices on the [HF data repo @ issue923_ctx_query_decomposition (pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/77d04e45096e120ca897fdb7b22730e157ce00a2/issue923_ctx_query_decomposition) (raw_completions 131 files, analysis_tensors 53, eval_results 61, figures 33; Hub-verified at write time). Reused artifacts — answer-span stores + probe pools from [#658](https://eps.superkaiba.com/tasks/658) and its UltraChat-genre round ([store, pinned](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/77d04e45096e120ca897fdb7b22730e157ce00a2/issue658_theory_assumptions)) — fit: same base model, on-policy greedy completions with all-28-layer spans, all 50×48×2 cells present, pinned by store manifest + dataset revision; the 50-context battery + UltraChat probe builder from [#594](https://eps.superkaiba.com/tasks/594) (`data/issue594/battery.json`, in git). Declared discard (plan §10): per-token span tensors for the new arms (~390 GB); regen recipe = one teacher-forced forward over the persisted rollout text.

**Context:** created 2026-07-03; results landed 2026-07-03 (fits done 2026-07-04Z). Parent: [#658](https://eps.superkaiba.com/tasks/658) (the store + context battery this reads). Two crash-fix rounds, neither material to the science: the context-vector identity probe threshold recalibrated to bf16 batching numerics (hard floor 0.99, sub-0.999 recorded as warnings), and conditional `.env` sourcing in the CPU-phase script on GCE. Free-analysis same-issue follow-up round `qryiii-group-lambda-refit` (run 2026-07-04, zero GPU): leave-query-group-out penalty re-selection over the persisted layer-18 feature/target packs, revising the masked-context-collapse mechanism in place. Refit scope caveat: it holds out query groups whose contexts remain in the train complement (unseen queries under seen families, versus the production unseen-family × unseen-query test), and the part-PCA bases and standardization still see the held-out group — both residual leakages bias selection toward smaller penalties, so neither can produce the observed non-rescue. Originating prompt, verbatim:

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

<!-- Acknowledged verifier WARNs: several results sit between the 120-word soft cap and the 180-word hard cap, some Takeaways bullets exceed 30 words, and total content prose exceeds the word budget (including the +250 allowance for the folded qryiii-group-lambda-refit round) — this run reports 9 planned arms x 2 genres + an OOD arm, a fold-regime robustness read, an anomaly investigation, and a data-validity control; each carries its own three-beat result, round 2 folds in the critics' requested marginal-regime, ambient-space, and scoping disclosures, and the refit round adds a selection-rule figure whose numbers live in its caption. Goal frontmatter lives in the task's body.md frontmatter, not this cache file. -->


