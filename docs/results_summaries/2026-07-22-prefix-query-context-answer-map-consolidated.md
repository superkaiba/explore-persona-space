# The context → answer map: one consolidated writeup (prefix vs query vs context)

*2026-07-22. Consolidates the 07-14 / 07-15 / 07-17 results summaries + the #1092 / #923 / #658 / #779 artifacts into one document. Every number is read from the named artifact; sources at the bottom.*

## Terminology (two different "prefix" objects — this distinction governs everything below)

- **Prefix** = everything before the last user query; **query** = the user message; **context** = prefix + query; **answer** = the model's completion.
- **Query-averaged prefix vector v_P** = the mean of a prefix's context vectors over queries. By construction a map fit on v_P is the context map at a coarser aggregation grain — the earlier ~0.8 "prefix map" (#658/#779 line) is this object.
- **Prefix-end state** = the residual-stream activation at the last prefix token, before the query exists (exactly constant across a prefix's queries). This is the genuinely pre-query object.
- Map target throughout: mean answer-token residual activation (layer 14 headline; ambient and PCA-48 bases), predicted by ridge.

## TLDR

- The per-context map is strong (held-out R² 0.74–0.81; shuffled-pairing floor ≈ 0.06–0.08); the prefix-end state predicts almost none of it per-context (R² 0.07–0.10) and reaches only 0.35–0.58 of its own ceiling even query-averaged.
- The map is ~additive: answer state ≈ prefix contribution + query contribution — query carries ~79% of explainable variance, prefix ~11%, interaction ~10%; two fully disjoint forwards stitch back to 91% of the full map (R² 0.833 vs 0.910).
- The two fitted operators write into a shared answer-side subspace but read near-orthogonal inputs — the query selects which input directions carry signal.
- Despite its poor answer-state skill, the prefix-end state reads per-prefix behavior disposition at parity with 48-query-averaged reads (instruct model only), and carries MORE raw persona-vector-aligned signal than the averaged read.
- On real conversations, what makes a prefix hard for the map is its length, not its context-vector spread — the constructed-substrate coherence result does not transfer.

## Methodology

- Models: Qwen-2.5-7B-Instruct AND Qwen-2.5-7B (base); text-source × model factorial: own-policy vLLM greedy (temp 0.0) / Claude-written / shuffled-pairing derangement (carrier floor); reading model teacher-forced.
- Corpus (#1092): 1,145 real WildChat/LMSYS conversation prefixes sparse-crossed with a 1,397-query bank of real held-out user turns → 21,193 rows; dense core ≈ 99 prefixes × 48 shared queries. Constructed replication substrate (#923): 50 contexts in 7 families × 144 UltraChat queries (+ Betley + human-written Dolly OOD queries).
- States captured per row: prefix-end, context-end, and bare-query (query with NO prefix — empty system turn, verified Qwen's default system prompt is not silently inserted; sibling runs also checked template-removed and attention-masked variants).
- Fits: ridge, grouped 6-fold CV by prefix id (a held-out fold never shares a prefix with training); #923 used 7 leave-one-family-out context folds × 4 query folds. All cross-arm comparisons are scored against the SAME pooled answer targets.

## Result 1 — The per-context map matches the established averaged prefix map's skill, each at its own grain

Per-context map: held-out R² **0.74–0.81** in 5 of 6 coherent cells (0.50 in the base-model-reading-instruct-text cell); shuffled-pairing floor 0.06–0.08, and the fitted map sits far above trivial-transport floors (identity / scaled-identity / diagonal-affine) — a real learned operator. Query-averaged map: R² **0.82–0.94**, consistent with the earlier ~0.8 prefix-map result, which is this object. Moving from averaged to per-context grain loses nothing: the map exists at the single-(prefix, query) level.

## Result 2 — The pre-query prefix-end state carries almost none of the map's skill, and it is NOT the averaged map

![fair comparison: arm x target grain, ambient basis](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f61b5bc49abaa9888bc2455465a599c9bfd83e41/figures/summaries/prefix_vs_context_map/fair_comparison_grid.png)

> Matched-target fair comparison (instruct, own answers, layer 14): both arms fit on the same crossed corpus, scored against the same pooled targets. Panels = target grain, x = basis, color = input arm; dashed ticks = each arm's achievable ceiling on single-context targets (prefix arm: between-prefix variance share — its input is constant within a prefix; context arm: MLP companion for pca48 / additive ceiling for ambient). Bars are single pooled held-out R² point estimates (no fold-spread error bars). Source: `eval_results/issue_1092/inline_fair_comparison/fair_comparison.json`; original two-panel version: `figures/issue_1092/fair_comparison_prefix_vs_context.png`.

| (instruct, own answers, L14) | prefix-end | context |
|---|---|---|
| per-context targets (ambient / pca48) | 0.065 / 0.098 | **0.814 / 0.914** |
| query-averaged targets (ambient / pca48) | 0.371 / 0.529 | **0.819 / 0.936** |
| fraction of own ceiling | 0.35–0.58 | 0.99–1.06 |

![prediction agreement between the two arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/issue_1092/fair_comparison_prediction_agreement.png)

![shrinkage: single global scalar vs per-dimension diagonal](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/summaries/prefix_vs_context_map/shrinkage_residual_variance.png)

- The prefix-end map cannot capture per-query variation (R² ~0.07–0.10 vs 0.81–0.91) — expected, since ~89% of answer-state variance is query-borne (Result 3).
- Less expected: even on query-averaged targets it reaches only 0.35–0.58 of its own reliability ceiling — the prefix-end state is a strictly weaker summary than v_P, not the same object.
- The two arms' predictions agree at only R² 0.28–0.54; the prefix-end arm's predictions are a near-uniform shrinkage of the context arm's (global α ≈ 0.60–0.66; 100% of per-dimension coefficients < 1), and it is worse on 100% of 996 prefixes (median error ratio 2.25×).

## Result 3 — Query explains ~8× more answer-state variance than the prefix

![variance shares: prefix / query / interaction, constructed vs realistic](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/summaries/prefix_vs_context_map/variance_shares.png)

> Crossed ANOVA decomposition of the answer summary vector over the dense core (99 real prefixes × 48 shared held-out real queries = 4,752 contexts), and the constructed UltraChat grid.

- Realistic corpus (L14, pca48): **query 79.0% / prefix 10.7% / interaction 10.3%**; constructed grid (L18): 83.7 / 7.8 / 8.6 — the shares transfer.
- Shuffled-pairing cells kill the query share (the loading moves to the interaction residual) — the query component needs true (context, answer) pairing.
- So the context → answer map is mostly answering "what question is this?", not "who is answering?".

## Result 4 — Prefix map + query map ≈ context map (additive to ~91%)

![stitch recovery](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/summaries/prefix_vs_context_map/stitch_recovery.png)

> Two states per (prefix, query) pair from forward passes that never attend to each other — v_prefix (prefix-alone forward) and v_query (bare-query forward, no prefix content) — fit as one ridge on the concatenation: the best context → answer map constrained to be additive. Instruct-own cell, layer 14, pca48.

- The disjoint stitch reaches R² **0.833** vs the full-context map's **0.910** (91% recovery); components alone: prefix 0.096, bare-query 0.146.
- So the map is ~additive: prefix and query contributions can be computed in isolation and summed; prefix-query attention mixing contributes ≤ ~0.08 R² at this granularity (consistent with the ~10% interaction share in Result 3). On the constructed grid, at matched two-span granularity the disjoint stitch matches the true mixed forward; the mixing gain (+0.103) appears only when the mixed forward is read at query-span granularity.
- The ordering replicates and transfers OOD on the constructed grid (held-out family × human-written Dolly queries: stitched 0.489 vs full 0.448).

## Result 5 — The two operators share an output subspace but read orthogonal inputs

![operator principal angles vs spectrum-matched null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/issue_1092/inline_operator_angles_vs_null.png)

- Output (answer-side) subspaces of the fitted prefix-arm and context-arm operators align far beyond a spectrum-matched null (principal angles 22–34° vs ~75–80° null; below-null on 48/48 Procrustes reads) — one shared write-side structure.
- Input subspaces sit AT the null (83–84°) — the query supplies which input directions carry signal; no rotation maps one operator onto the other; the context operator holds 2.4–20× the Frobenius energy.
- This is why Result 2's relationship is a scalar shrinkage: the prefix arm writes into the same output directions, just weakly.

## Result 6 — The prefix-end state reads behavior disposition as well as the averaged context read (instruct model)

![prefix-end monitoring vs averaged context read](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/issue_1092/prefixend_monitoring.png)

> 149 prefixes (≈100 real WildChat/LMSYS + 49 constructed battery conditions) × 48 queries = 7,152 rows. Target: each prefix's mean graded judge score over the 48 queries (sycophancy, hallucination). Inputs: prefix-end state vs the 48-query-averaged context-end read; ridge probe, held out by prefix. Source: `eval_results/issue_1092/inline_prefixend_monitoring/results.json`.

- Parity: sycophancy r = **0.759** (prefix-end) vs **0.774** (averaged context); hallucination **0.888 vs 0.887**; paired difference +0.015, 95% CI [−0.026, +0.052] — not significant; reliability ceilings 0.86 / 0.91.
- One pre-query state matches what otherwise takes 48 queries of averaging (the context-read averaging curve rises 0.48 at N=1 → 0.77 at N=48; prefix-end reads 0.76 from a single state).
- The raw persona-vector projection INVERTS the parity: r_B-projection at prefix-end r = **0.665** vs **0.037** on the averaged read — persona signal sits r_B-shaped before the query, is rotated into other directions once the query is read, and the fitted map transports it back.
- Instruct-only: base-model prefix-end reads collapse (0.01–0.05; averaged context 0.13 sycophancy / 0.62 hallucination).
- Interpretation via Result 3: a single answer state is ~79% "what question is this?" and ~11% "who is answering?" — averaging over queries denoises the query component to expose the persona component; the prefix-end state never had the query noise in the first place.

## Result 7 — What makes a prefix hard for the averaged map: length on natural data, spread on constructed data

The context-summary-sufficiency assumption predicts the query-averaged prefix vector is a valid summary only where the prefix's context vectors cluster — so within-prefix spread should track the averaged map's per-prefix error.

Constructed substrate (supports the prediction; whitened spread, 50 conditions):

![spread vs residual, constructed substrate](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/issue_658/fig_a35a_spread_vs_residual_loco_L14.png)

Natural prefixes (does not transfer; raw-L2 spread, 996 real conversations):

![spread vs error, natural prefixes](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/summaries/prefix_vs_context_map/perprefix_error_vs_spread.png)

![error vs prefix length](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/summaries/prefix_vs_context_map/perprefix_error_vs_length.png)

- Constructed conditions: spread strongly predicts held-out LOCO residual (median layerwise Spearman **+0.89**, 28/28 layers positive) — personas cluster tightly and generalize; format conditions scatter and fail.
- Natural prefixes: the correlation is null on instruct (ρ = −0.03, n.s.) and INVERTED on base (ρ = −0.80). Prefix length dominates instead: conversation turns → error ρ = **+0.83** (instruct) / **+0.80** (base); controlling for length, spread stays negative (partial ρ −0.33 / −0.52).
- Reading: long real prefixes saturate the representation (appending different queries moves it less → LOW spread on the hard, long prefixes), so spread is confounded with length and the naive "more spread ⇒ harder" prediction fails on natural data. The coherence condition may still be right as a validity bound, but spread is not its observable there — length is the practical difficulty axis.

## Conclusion and next steps

- The context → answer map is ~an additive combination of a query → answer map (~79%, "what is being answered") and a prefix → answer map (~11%, "who is answering"), sharing one write-side operator structure.
- The averaged prefix map is exactly the context map at coarser grain; the pre-query prefix-end state is a different, weaker object for answer-state prediction — but an equally good (and in raw persona-vector form, better) carrier of the persona signal.
- Next: close the ~0.08 stitch gap (where does the prefix×query interaction live — layers, directions, positions?); test whether the prefix-end persona read survives fine-tuning; fold the natural-prefix length/spread refinement into the theory's validity condition.

## Sources

- Underlying summaries: `docs/results_summaries/2026-07-17-how-prefix-map-relates-to-context-map.md`, `2026-07-17-fair-comparison-deep-dive.md`, `2026-07-15-link-prefix-context-answer-maps.md`, `2026-07-14-prefix-vs-context-map.md`; glossary: `docs/glossary_context_answer_map.md`.
- Artifacts: `eval_results/issue_1092/{inline_fair_comparison, inline_fair_comparison_deepdive, inline_prefixend_monitoring, inline_compose_chain}`; `eval_results/issue_658/inline_a3_5a_coherence/`.
- Task bodies: #1092 (realistic crossed corpus, HIGH), #923 (constructed grid + stitch, MODERATE), #658 (coherence), #779 (map definitions), #813 (grain transfer).
