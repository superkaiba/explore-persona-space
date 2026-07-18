# Result: How the prefix map relates to the context map — one map at two aggregation grains for the query-averaged prefix vector, while the pre-query prefix-end map is a uniformly shrunk sub-map that still carries the trait signal

> Terminology: `docs/glossary_context_answer_map.md`. This writeup consolidates the
> current answer to "how does the prefix vector map relate to the context vector
> map?" across #1092 (+ its 2026-07-17 inline rounds), #923, #813, #779, and the
> #658-line coherence test. Earlier drafts of the answer:
> `2026-07-14-prefix-vs-context-map.md` (pre-fair-comparison),
> `2026-07-15-link-prefix-context-answer-maps.md` (Results 1–7),
> `2026-07-17-fair-comparison-deep-dive.md` (mechanics of Result 7).

## Motivation

- We had two separately-discovered linear maps over the residual stream:
    - a **prefix map** `M`: query-averaged prefix vector → query-averaged answer profile, one row per prefix (`v̄_A^q ≈ M·v_P`; the #658/#810 line, ~0.8 R²)
    - a **context map** `M′`: single context vector → single answer summary, one row per (context, answer) (`v_A ≈ M′·v_C`; the #779 line)
- The question: **what is the relationship between these two maps?** The single-context map presumably carries more granular information, the averaged map broader information; the hypothesis was that what the averaged map keeps **is persona information** — behavior shared across queries.
- The question turned out to be ambiguous, and resolving the ambiguity IS half the answer: after #1092 there are **two distinct prefix-side objects** — the *query-averaged* prefix vector `v_P` (mean of a prefix's context vectors) and the *prefix-end state* (the residual-stream activation before the query begins; exactly constant across a prefix's queries, within-prefix std ratio 0.000). The answer is different for each.
- All matched-comparison reads below are on #1092's realistic crossed corpus (same prefixes, same queries, both input arms fit on every read) — the earlier cross-dataset comparison was confounded, which is what #1092 was built to fix.

## TLDR

- **For the query-averaged object, the two maps are the same measurement at two aggregation grains.** `v_P` is literally the query-average of a prefix's `v_C`'s; on matched data the averaged context map reaches R² 0.82–0.94, which is where the earlier ~0.8 "prefix map" number belongs.
- **For the pre-query prefix-end state, the answer is NO — it is not the averaged context map.** At matched averaged grain it reaches only R² 0.37–0.53 (instruct; 0.26–0.46 base) vs 0.82–0.94, the two maps' predictions agree at only R² 0.28–0.54, and its relationship to the context map is a near-uniform scalar shrinkage (α ≈ 0.60–0.66, 100% of per-dimension α_d < 1) onto the component both maps share.
- **The context map decomposes near-additively into a prefix part + a query part**, with the query part dominating per-context prediction (~71–79% of dense-core variance vs ~10% prefix, interaction ~10–19%); two disjoint forwards stitch back to R² 0.833 of the full-context 0.910.
- **What query-averaging keeps is the persona component**: only ~10% of per-context variance, but the part stable across queries — persona-level monitoring jumps from r 0.32/0.58/−0.01 (evil/sycophancy/hallucination, single prompt) to 0.66/0.89/0.53 at 40 questions averaged (#779).
- **At the operator level there is one shared write-side map**: the two fitted operators' output subspaces align far beyond a spectrum-matched null (32.9–52.9° vs ≈87° in 24/24 grid rows) while their input subspaces sit at the null — the query supplies *which input directions carry signal*, not new operator structure.
- **New (2026-07-17): the prefix-end state still carries the per-prefix trait DISPOSITION on the instruct model.** For trait monitoring it matches the 48-question-averaged context read (sycophancy r 0.759 vs 0.774; hallucination 0.888 vs 0.887; paired-diff CIs straddle 0) — and with a raw persona-vector projection the parity *inverts* (prefix-end 0.665 vs averaged-context 0.037 on sycophancy): the persona signal is r_B-shaped pre-query, gets rotated into other directions once the query is read, and the transport map re-aligns it into answer space. On the base model the prefix-end reads collapse.

## Methodology

The matched substrate (#1092, HIGH confidence): 1,145 real WildChat/LMSYS conversation prefixes sparse-crossed with a 1,397-query bank of real user turns → 21,193 rows (dense core ≈100 prefixes × 48 queries); models Qwen2.5-7B-Instruct + Qwen2.5-7B (no fine-tuning), reading teacher-forced. Answer-text provenance per cell: own-policy (vLLM greedy, temp 0.0), Claude-written, and a shuffled-pairing derangement as the carrier floor. Per row, states are captured at the **prefix end** and the **context end**; targets are answer-span summaries. Ridge maps per cell × input arm × layer (14/18/19) × basis (ambient 3584-d / top-48 PCs), grouped 6-fold CV on novel-prefix folds, battery-excluded fits (n=17,308); ceilings, floors, permutation and random-map nulls per `fair_comparison.json`. Both prefix-based and context-based arms are fit on every read (the project's standing paired-arm design).

Supporting substrates: #923 constructed UltraChat prefix × query grids (variance decomposition + stitch replication, MODERATE); #813 grain comparison on the 50-condition battery (LOW); #779 LMSYS-5000 monitoring granularity (MODERATE; graded 0–100 `claude-sonnet-4-5` judge, 5 draws, LOGO over 60 persona groups); #658-line coherence test on the 50-condition × 48-probe store. The 2026-07-17 monitoring rounds reuse #1092's banked states + banked Sonnet graded judge scores (149 prefixes × 48 queries, layer 14, identical GCV dual-ridge readout on both arms, grouped 5-fold over prefixes; no new forwards or judging).

## Results

### _Result 1: The query-averaged prefix map IS the context map at a coarser grain — the earlier ~0.8 belongs to this object_

Definitionally, `v_P = mean_q v_C` and the behavior profile is `mean_q v_A`, so the query-averaged prefix map is an aggregation of the context map, not a separate mapping. The empirical reads agree: evaluated at the averaged grain on the matched corpus, the context map reaches R² 0.819/0.936 (instruct own-text, ambient/pca48; base 0.763/0.884 — `fair_comparison.json`), reproducing the earlier prefix-map line's ~0.8. On the 50-condition substrate, refitting at the other grain changes little on averaged targets (#813's cross-grain transfer), and the earlier "averaging collapses the map's rank ~4×" read was an artifact of averaging over only 50 conditions — at ≈1,046 averaged prefixes the averaged/per-example stable-rank ratios are 0.82–1.10 (#1092).

**Plot: each map at both grains + fraction-of-ceiling (the fair comparison)**

![fair comparison](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8fcb896cb4b5ad2074603fb680cdf351495be815/figures/issue_1092/fair_comparison_prefix_vs_context.png)

**Takeaways:**

* The two maps of the original question are one map read at two grains — the substantive comparison is not averaged-vs-single, it is *which prefix-side input* you read (next result).
* Per-context, the context map is essentially at its achievable ceiling at both grains (0.99 of the additive ceiling ambient; the pca48 fractions 1.02–1.06 exceed it because the dense-core additive decomposition is a slightly conservative bound — the MLP companion ceiling, 0.984, is the better-behaved one).
* Caveat carried from #813 (LOW): neither grain generalizes to held-out *conditions* on the 50-condition substrate (pooled leave-one-context-out R² −15 to −51 across behaviors/arms), while the within-context, query-specific component is strong (R² 0.71–0.87 vs shuffle nulls ≈ −0.08). The transferable per-example signal is exactly what averaging discards.

### _Result 2: The pre-query prefix-end map is NOT the averaged context map — it is a uniformly shrunk sub-map_

The direct test (Result 7 of the 07-15 summary + the 07-17 deep-dive): fit both arms on the same battery-excluded rows, evaluate at the same averaged grain, and compare the two maps' *predictions to each other*, not just to truth. All numbers from `fair_comparison.json` / `deepdive.json` (layer 14).

| (own-text cells, L14) | prefix-end map | context map |
|---|---|---|
| averaged-profile R² (inst, ambient / pca48) | 0.371 / 0.529 | **0.819 / 0.936** |
| averaged-profile R² (base, ambient / pca48) | 0.261 / 0.463 | 0.763 / 0.884 |
| single-context R² (inst, ambient / pca48) | 0.065 / 0.098 | 0.814 / 0.914 |
| fraction of own achievable ceiling (full pop.) | 0.35–0.58 | **0.99–1.06** |

**Plot: per-prefix error, prefix-end map vs context map — every point above the diagonal**

![per-prefix error scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/1222f3a9c4e57f8a9a660554c1d8b776b2701eb5/figures/summaries/prefix_vs_context_map/perprefix_error_scatter.png)

**Takeaways:**

* Prediction agreement is far from the near-identity "same map" requires: R² 0.28–0.54 across cells/bases (prefix-prediction→context-prediction direction; centered per-prefix cosine 0.64–0.74). The raw ambient cosine reads 0.99 only because both predictions share a grand-mean profile; the between-prefix structure is what disagrees.
* The relationship is a **near-uniform scalar shrinkage**: a single global α ≈ 0.60–0.66 (P ≈ αC) explains ~46–56% of the prefix map's between-prefix prediction variance, a per-dimension diagonal adds only ~10 points, and 100% of per-dimension α_d < 1 in every computed cell — the prefix-end map shrinks toward the shared component on every dimension, never amplifies.
* The prefix-end map is worse on **100% of 996 prefixes** (median error ratio 2.25×, 54% above 2×; errors correlated 0.56–0.90 on the same hard prefixes). It also under-fills its own between-prefix ceiling (0.35–0.58 of it) — the shortfall is not purely structural.
* Which prefixes are hard: **length**, not spread — turns→error Spearman +0.80/+0.83 in both cells, while within-prefix context-vector spread→error is null (instruct, −0.03) or negative (base, −0.795), negative even controlling for length. This does NOT transfer the #658 coherence framing to per-prefix map errors on natural prefixes (a different read from the #658 LOCO test in Result 4, which stands).
* Mechanistic reading: processing the query in the prefix's presence writes prefix-relevant information into the context state that the pre-query state does not linearly expose; averaging context states retains it, the prefix-end state never had it.

### _Result 3: The context map decomposes into a prefix part + a query part, and the query part dominates_

Variance decomposition of the dense-core crossing plus the disjoint-stitch test (`read3_fgi_shares.json`; #923 replication `anova_shares.json`).

**Plot: prefix / query / interaction variance shares — constructed (#923) vs realistic (#1092)**

![variance shares](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e26781d3700342e2412ce775b913d91fb4505efb/figures/summaries/prefix_vs_context_map/variance_shares.png)

**Plot: a disjoint prefix+query stitch recovers most of the full-context map (instruct own-text, L14, pca48)**

![stitch recovery](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e26781d3700342e2412ce775b913d91fb4505efb/figures/summaries/prefix_vs_context_map/stitch_recovery.png)

**Takeaways:**

* Shares transfer from the constructed grid (#923, L18 pca48: 7.8% prefix / 83.7% query / 8.6% interaction) to realistic data (#1092, inst-own L14: 10.7/79.0/10.3 pca48; 10.4/71.9/17.7 ambient). In the shuffled-pairing cells the query share dies (5–16%) and the loading moves to the interaction residual — the decomposition needs the true pairing.
* Stitching two **disjoint** forwards — prefix-only plus bare-query-only, no attention between them — reaches R² 0.833 against the full-context map's 0.910 (prefix-only alone 0.096, bare-query alone 0.146). Attention mixing adds a real but second-order increment (+0.103 at matched query-span granularity, #923).
* So context→answer ≈ prefix→answer + query→answer with a ~10% interaction residual, and the query part carries most of the per-context transport.

### _Result 4: What query-averaging keeps is the persona component_

The original hypothesis, tested on monitoring granularity (#779): the same generic context→answer map, read per prompt vs averaged over 1–40 questions per held-out persona group (LOGO, n=60 groups).

**Plot: monitoring r vs number of questions averaged per persona group**

![persona-level monitoring curves](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73a0e157cdeef6bf332c803f519e89773bd0a893/figures/issue_779/grouped_context.png)

**Takeaways:**

* Averaging 40 questions per held-out persona lifts trait-monitoring r from 0.32/0.58/−0.01 to 0.66/0.89/0.53 (evil/sycophancy/hallucination; `grouped_context` curve values). The per-answer state is query-dominated; averaging cancels the query component and the small-but-consistent persona component emerges — "the averaged map captures persona information" holds in exactly this shared-across-queries sense.
* Boundary condition: on #1092's natural, weakly persona-differentiated prefixes, the prefix factor's share *along trait directions* is small (1.7–5.2% across traits) and does not beat a random-direction null — the strong persona-level reads need conditions that actually differ in persona.
* Validity condition for the average itself (#658 coherence test, out-of-sample): within-condition context-vector spread predicts the held-out mean-vector-predictor residual (median layerwise Spearman +0.89, 28/28 layers positive; +0.76 within the persona family). Personas cluster tightly and average well; format conditions scatter and fail.

### _Result 5: One shared write-side operator — the query changes what is read, not what is written_

Comparing the two fitted operators directly (principal angles between singular subspaces, input vs output sides, against spectrum-matched random-map nulls; scoped read + full off-VM grid).

**Plot: input/output principal angles + Procrustes residuals vs nulls**

![operator angles vs null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a05b1192160c9a074e164c9d1da26fe7e65f25c4/figures/issue_1092/inline_operator_angles_vs_null.png)

**Takeaways:**

* OUTPUT (answer-side) subspaces align far beyond null in every read: median angle 45.6° vs null ≈86.7° (inst-own L14 ambient, k=48); across the full off-VM grid 32.9–52.9° in all 24 rows that carry the read, Procrustes residual below its null band in 48/48 rows. INPUT subspaces sit at the null (84.4° vs ≈84.6°) — as they must, since the prefix-end state does not contain the query.
* This coexists with the weak prediction agreement of Result 2 because the two measurements are of different objects: the operator read measures the *directions* written into (shared, because both predict the same target); the agreement R² measures realized per-prefix *values*, which diverge because the arms read from near-orthogonal input subspaces and the prefix operator is more heavily shrunk (λ 1000 vs 100; the α ≈ 0.6 of Result 2).

### _Result 6 (new, 2026-07-17): the prefix-end state still carries the trait signal on the instruct model — in raw persona-vector shape_

The fair comparison says the prefix-end map cannot predict answer *profiles*; the monitoring rounds ask the narrower practical question: can it predict per-prefix trait *disposition*? Three readout constructions on the same 149-prefix × 48-query substrate (banked Sonnet graded judge scores, L14, identical readout both arms; `inline_prefixend_monitoring/`).

**Plot: prefix-end vs averaged-context trait monitoring (supervised readout)**

![prefixend monitoring](https://raw.githubusercontent.com/superkaiba/explore-persona-space/015e0e04cfaa4573d8960ac6b5613c50c17b1e84/figures/issue_1092/prefixend_monitoring.png)

**Plot: the three readout constructions side by side (supervised / map-mediated / raw r_B)**

![readout constructions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c8df7f1d6c5ec331e5f68bfb4b38087fa054970e/figures/issue_1092/prefixend_monitoring_constructions.png)

**Takeaways:**

* **Supervised probe: parity.** On the instruct arm the pre-query prefix-end state monitors traits as well as the 48-question-averaged context read — sycophancy r 0.759 vs 0.774, hallucination 0.888 vs 0.887, paired-diff CIs straddling 0, robust to a disjoint-query-half control (reliability ceilings 0.86/0.91). On the base model the prefix-end read collapses (hallucination 0.05 vs averaged-context 0.62, diff +0.57 [+0.02, +0.70]).
* **Raw persona-vector projection: the parity inverts.** Dotting states with the pre-extracted trait direction r_B, the prefix-end state carries far MORE r_B-aligned signal than the averaged context state (sycophancy 0.665 vs 0.037; paired diff −0.628 [−0.78, −0.49], 0/2000 bootstrap draws positive). Map-mediated projection (state → fitted transport map → predicted answer state → r_B) restores parity at near-supervised magnitude (hallucination 0.848/0.850; sycophancy |0.71| both arms, with an r_B-vs-judge sign flip on the answer side flagged for eyeballing).
* Geometric read: the persona information is r_B-shaped in the PRE-query state, gets rotated into other directions in context-end states (a supervised probe still finds it; a raw r_B dot product does not), and the transport map re-aligns it into answer space where r_B reads it again. This sharpens Result 2's "strictly weaker input": weaker for answer-profile *transport*, not for trait *disposition* — on the instruct model.
* Caveats: natural, weakly persona-differentiated prefixes; sycophancy + hallucination only (evil ineligible, 0 positives); within-corpus; not yet folded into the #1092 clean-result body (deferred to the next analyzer pass).

## Next steps

- Fill the remaining deep-dive cell (`pre_own/ambient` shrinkage, checkpointed resume) and the off-VM battery-refit residuals (per-target R² columns).
- Close the 0.833 → 0.910 stitch gap: where does the prefix×query interaction live (layers, directions, token positions)?
- Use the split for behavior prediction: test whether the prefix component predicts behavior change under fine-tuning better than the full-context state (the #811/#833 map-change line).
- Reconcile the trait-monitoring parity (Result 6) with the persona-differentiation boundary condition (Result 4) on an explicitly persona-differentiated realistic corpus.
- Isolate the coherence curvature mechanism (leave-one-condition-out nonlinear fits, 0 GPU, queued).

---

*Sources (per-number provenance): `eval_results/issue_1092/inline_fair_comparison/fair_comparison.json`, `inline_fair_comparison_deepdive/{deepdive,deepdive_meta}.json`, `p7/read3_fgi_shares.json`, `p7/read4c_trait_per_factor_repaired.json`, `inline_operator_read_scoped/operator_read.json`, `offvm-battery-refit-and-operator-comparison/partb_operator_digest.json`, `inline_prefixend_monitoring/{results,readout_constructions}.json`; `eval_results/issue_923/fits/anova_shares.json`; `eval_results/issue_813/per_example_vs_averaged/transfer_L14_*.json`; `figures/issue_779/grouped_context.meta.json`; `eval_results/issue_658/inline_a3_5a_coherence/coherence_results.json`. Clean-results: #1092 (HIGH), #923 (MODERATE), #813 (LOW), #779 (MODERATE). Prior drafts: the 07-14 / 07-15 / 07-17 summaries in this directory.*
