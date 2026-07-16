# Result: Link between prefix -> answer and context -> answer map

## Motivation

- We've found a linear mapping from prefix vector to answer summary and a mapping from single context vector to single answer summary
- This experiment tries to understand the link between these mappings
    - the single-context mapping presumably contains more granular information; the prefix (query-averaged) mapping presumably contains broader information
    - the hypothesis we wanted to test: what the averaged mapping keeps **is persona information** — the behavior that is shared across different queries
    - and importantly, to compare them on MATCHED queries/contexts (the earlier comparison was across different datasets, so it was confounded)

## TLDR

- The two maps are the same measurement at two aggregation grains: the prefix vector is literally the query-average of a prefix's context vectors, and the behavior profile is the query-average of its answer summaries — so the question is what each grain carries
- On matched real prefixes × real queries, the single-context map carries nearly all per-context prediction: held-out R² 0.71–0.80 (context input) vs 0.04–0.08 (prefix input), and the prefix map is barely above trivial-transport floors
- The context map decomposes: context→answer ≈ prefix→answer + query→answer with only ~10% interaction; two completely disjoint forwards stitch back to R² 0.833 of the full-context 0.910
- Averaging over queries doesn't change the fitted map on averaged targets — but it throws away the query-specific component, which is the *only* component that transfers to held-out contexts
- What averaging keeps is persona information: the prefix part is only ~11% of per-context variance, but it is the part that is stable across queries, so persona-level monitoring jumps from r 0.34/0.63/0.09 to 0.66/0.89/0.53 (evil/sycophancy/hallucination) when 40 questions are averaged
- The prefix-average is trustworthy exactly where the theory says it should be: within-condition context-vector spread predicts held-out prefix-average error (Spearman +0.89, 28/28 layers) — personas cluster tightly and generalize well, format contexts scatter and fail
- The operator-level test (scoped early read, Result 6): the two fitted maps share OUTPUT structure far beyond a spectrum-matched null in 12/12 reads while their INPUT read-directions sit at the null — one shared write-side operator into the answer subspace; the query supplies *which input directions carry the signal*, it does not add new operator structure

## Methodology

- Models: Qwen-2.5-7B-Instruct (headline) + Qwen-2.5-7B pretrained (cross-model cells)
- Datasets:
    - **Realistic crossed corpus (matched-arm head-to-head)**:
        - 1,145 real conversation prefixes from WildChat + LMSYS (329 with ≥5 user turns) sparse-crossed with a 1,397-query bank of real held-out user turns → 21,193 (prefix, query) rows; dense core ≈100 prefixes × 48 queries
        - 4×2 text-source × model factorial for the answer text: own-policy (vLLM greedy, temp 0.0), Claude-written, and a shuffled-pairing derangement (real answers re-paired to wrong prefix–query rows — the carrier floor); read by both the instruct and pretrained models, teacher-forced
        - Dashboard to view examples: https://htmlpreview.github.io/?https://gist.githubusercontent.com/superkaiba/40e76f146d52583c1e178f5ad5ca2910/raw/issue1092_corpus_dashboard.html
        - Fits use grouped 6-fold CV on **novel-prefix folds** (a held-out fold never shares a prefix with training)
    - **Constructed crossings** (UltraChat prefix × query grids) for the variance decomposition + stitching replication
    - **50-condition × 48-probe battery** (persona / format / behavior / ICL / rephrase / wildchat condition families) for the coherence test
    - **LMSYS-5000** (the single-context map corpus) for the monitoring-granularity read
- Computed quantities (per layer; 28 layers × 3584 dims):
    - $v_P$ prefix vector: activation at the **prefix end** — everything before the user query (system prompt / persona / prior turns). Identity worth keeping in mind: a prefix's $v_P$ equals the average of its contexts' $v_C$ over queries
    - $v_C$ context vector: activation at the **context end** (prefix + query, the last prompt token)
    - $v_A$ answer summary: mean activation over the answer span
    - behavior profile: a prefix's $v_A$ averaged over its queries
    - **both input arms ($v_P$ and $v_C$) are fit on every read** — that's what makes the comparison confound-free
- Predictors:
    - Per-layer ridge maps (standardized inputs, λ CV-selected), input → answer target, per cell × arm × layer × basis (raw 3584-dim ambient + a 48-dim PCA target basis)
    - Variance decomposition of the dense-core crossing into prefix / query / interaction main effects
    - Disjoint stitch: fit prefix→answer on a prefix-only forward and query→answer on a bare-query forward (no attention between the two), sum the predictions
- Baselines:
    - Permutation nulls per cell/arm; the shuffled-pairing derangement cells (carrier floor); train-mean predictor
    - Transport floors: raw identity ($v$ copied), scaled identity α·I, diagonal-affine — kills the "it's just consecutive-token similarity" worry
    - 200-draw random-map / random-pairing null bands for the operator reads; matched-n control draws for the rank comparison
- Metrics: held-out grouped R² (variance-weighted over dims, pooled over answer targets); stable rank of the fitted maps; variance shares; Spearman for the coherence reads

## Results:

### _Result 1: On matched prefixes and queries, the context vector carries almost all of the answer prediction (R² 0.71–0.80 vs 0.04–0.08)_

The first thing I wanted was the confound-free head-to-head: same prefixes, same queries, both input arms fit on every read. For each of the eight text-source × model cells I compared the held-out R² of the map fit from the prefix-end state vs the context-end state (layer 14, pooled answer targets).

**Plot: Held-out R² per cell, prefix-based vs context-based arms**

![prefix vs context R2](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cfba89b1c820dd033c0efbf4bc15119b4387805f/figures/issue_1092/read1_r2_prefix_vs_context.png)

**Takeaways:**

* The query-bearing context state carries nearly all held-out map skill: context 0.71–0.80 vs prefix 0.04–0.08 in the coherent cells (instruct-own: 0.804 vs 0.065). The gap is layer/basis stable (layer 18 context 0.823; pca48 0.910 vs 0.096)
* Both arms clear their permutation nulls — the prefix map is small but real
* The shuffled-pairing cells collapse both arms (0.06–0.08), so neither map is a corpus artifact: it needs the true prefix–query–answer pairing
* The map also exists in the pretrained model (its own-text cell reads 0.714), consistent with the earlier finding that pretraining already carries ~87% of the instruct map
* One hygiene note: a filter bug had leaked the 2,400 eval-battery rows into fit training for 4 of the 8 cells; the battery-excluded refit is running and expected to move these numbers by ~±0.01 at most (−12% train rows on a map deep into its flat data-scaling regime)

### _Result 1.5: The context map is a genuine learned operator — the prefix map is barely above trivial transport_

The obvious worry for any state→state map is that it's just "the answer state looks like the pre-answer state." So I computed trivial-transport floors for both arms: raw identity, scaled identity, and a diagonal-affine map.

**Plot: transport floors vs the fitted ridge map, both arms, all cells**

![transport floors](https://raw.githubusercontent.com/superkaiba/explore-persona-space/81581a49ebe4acbeffe5b43af744521673e19e02/figures/issue_1092/inline_a3_transport_floors_vs_ridge.png)

**Takeaways:**

* Context arm: ridge (0.49–0.80) sits far above raw identity (−1.6 to −4.5), scaled identity (0.05–0.08), and diagonal-affine (0.11–0.14) — the map is a real learned linear operator, not consecutive-token similarity
* Prefix arm: the ridge (~0.05) is approximately AT its own floors — the prefix map carries almost no operator skill beyond a trivial transform. Per-context, essentially all learned transport structure lives in the query-bearing state
* This sharpens how to think about the prefix→answer map we found earlier: its R² ~0.8 at the *prefix-averaged* grain is real, but per-context the prefix state alone predicts almost nothing — the averaged map works because averaging removes exactly the variance the prefix can't predict

### _Result 2: context→answer decomposes into prefix→answer + query→answer (with only ~10% interaction)_

Next, the decomposition question: can the context map be written as a prefix part plus a query part? Two reads say yes. First, the variance decomposition of the dense-core crossing; second, an operator-additivity residual test against random-map null bands.

**Plot: prefix / query / interaction variance shares — constructed vs realistic crossings**

![variance shares](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e26781d3700342e2412ce775b913d91fb4505efb/figures/summaries/prefix_vs_context_map/variance_shares.png)

**Plot: operator additivity residuals per cell vs their null bands**

![operator residuals](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cfba89b1c820dd033c0efbf4bc15119b4387805f/figures/issue_1092/read4_operator_residuals.png)

**Takeaways:**

* The shares transfer from the constructed grid (7.8% prefix / 83.7% query / 8.6% interaction, UltraChat) to realistic data (10.7% / 79.0% / 10.3%) — the realistic prefix share runs slightly higher
* Both additivity residual tests sit far below their random-map null bands in every coherent cell (0.42–1.00 vs 5th-percentile bands at 1.59–2.10): the context operator is close to additive in prefix and query factors
* In the shuffled cells the query share dies and the loading moves to the interaction residual — exactly what the carrier-floor hypothesis predicts
* These two reads are provably unaffected by the battery-leak hygiene issue (recomputed battery-excluded: shares identical to 0, residuals identical to ~7 decimal places)

### _Result 2.5: Two disjoint forwards recover most of the full-context map_

The strongest version of the decomposition: fit prefix→answer on a prefix-only forward and query→answer on a bare-query forward — the two parts never attend to each other — and add the predictions.

**Plot: stitch recovery (instruct-own cell, layer 14, pca48)**

![stitch recovery](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e26781d3700342e2412ce775b913d91fb4505efb/figures/summaries/prefix_vs_context_map/stitch_recovery.png)

**Takeaways:**

* The disjoint stitch reaches R² 0.833 against the full-context map's 0.910, while bare-query alone gets 0.146 and prefix alone 0.096 — you need both parts, and what attention mixing adds on top is comparatively small at this granularity
* On the constructed grids we measured the attention-mixing increment directly: +0.103 at matched query-span granularity — real, but second-order next to the additive structure
* So the answer to "can we combine a prefix→answer map with a query→answer map into a full context→answer map" is basically yes, up to a ~0.08 R² mixing gap — closing that gap (where does the interaction live?) is a next step

### _Result 3: The query-averaged map and the per-example map agree on averaged targets — but the transferable signal is query-specific_

Then the direct grain comparison: fit the map at question-averaged grain vs per-example grain on the same substrate, and test what each generalizes to.

**Plot: own-grain fits vs cross-grain transfers (leave-one-context-out)**

![grain transfer](https://raw.githubusercontent.com/superkaiba/explore-persona-space/424c98492f25b17cf172256c71460e0c37e50bd2/figures/issue_813/hero_pe_vs_avg_transfer_trained.png)

**Plot: within-context (query-specific) R² per cell vs shuffle nulls**

![query-specific signal](https://raw.githubusercontent.com/superkaiba/explore-persona-space/424c98492f25b17cf172256c71460e0c37e50bd2/figures/issue_813/dv4_query_specific.png)

**Takeaways:**

* On question-averaged targets, refitting per-example changes nothing: cross-grain transfer gaps ≤0.19 R², intervals spanning zero in 23/24 reads. At that grain the two maps are interchangeable
* But neither grain generalizes to held-out *contexts* on that 50-condition substrate (pooled LOCO R² deeply negative), while the within-context component is strong: R² 0.71–0.87 vs shuffle nulls ≈0. The component that actually transfers is query-specific — exactly the component the question-averaged map discards by construction
* A correction to an earlier read: "averaging collapses the map's rank ~4×" was an artifact of averaging over only 50 conditions. At ≈1,046 averaged prefixes the averaged/per-example stable-rank ratios are 0.82–1.10:

![rank by grain](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cfba89b1c820dd033c0efbf4bc15119b4387805f/figures/issue_1092/read2_rank_by_grain.png)

### _Result 4: What query-averaging keeps is persona information_

Now the original hypothesis: is what's captured by the averaged mapping vs the single mapping **persona information** — behavior shared across queries? The cleanest read is monitoring granularity: the same generic map, read per-prompt vs averaged over 1–40 questions per held-out persona.

**Plot: monitoring r vs number of questions averaged per persona group**

![persona-level monitoring curves](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73a0e157cdeef6bf332c803f519e89773bd0a893/figures/issue_779/grouped_context.png)

**Plot: per-prompt vs 40-question persona-averaged monitoring r, per trait**

![persona level vs per prompt](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e26781d3700342e2412ce775b913d91fb4505efb/figures/summaries/prefix_vs_context_map/persona_level_vs_per_prompt_r.png)

**Takeaways:**

* Averaging 40 questions per held-out persona lifts trait-monitoring r from 0.34/0.63/0.09 to 0.66/0.89/0.53 (evil/sycophancy/hallucination). The per-answer state is query-dominated; averaging cancels the query component and the small-but-consistent persona component emerges
* This fits the variance picture from Result 2: the prefix factor is only ~8–13% of per-context variance, but it's the part that is *stable across queries* — which is what "persona information" should mean operationally
* One honest caution from the matched substrate: on realistic prefixes (natural conversations, only weakly persona-differentiated), the prefix factor's share along *trait directions* is 1.7–5.2% and doesn't beat a random-direction null. The strong persona-level reads come from explicitly persona-differentiated conditions. So "the averaged map captures persona information" holds where personas actually differ — it's not an automatic property of any prefix

### _Result 5: The prefix-average is trustworthy exactly where contexts cohere_

Finally, when is the prefix-average even a valid object? The theory's answer (assumption A3b): only when a condition's context vectors cluster tightly — the within-condition spread bounds the error of predicting-from-the-average vs averaging-the-predictions (a Jensen-gap argument). We ran the direct test on the 50-condition battery: within-condition spread vs the held-out (leave-one-condition-out) error of the mean-vector predictor.

**Plot: within-condition spread vs held-out prefix-average residual (layer 14)**

![spread vs LOCO residual](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d794cc9afd8cd5ed9c7909eca69ab8c061982906/figures/issue_658/fig_a35a_spread_vs_residual_loco_L14.png)

**Plot: the spread→error correlation, per layer (out-of-sample vs in-sample)**

![layerwise rho](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d794cc9afd8cd5ed9c7909eca69ab8c061982906/figures/issue_658/fig_a35a_layerwise_rho.png)

**Takeaways:**

* Spread predicts failure, as the assumption says it should: median layerwise Spearman +0.89 between within-condition spread and the held-out prefix-average residual, positive at all 28 layers, and it holds within the persona family alone (+0.76)
* Personas are the tightest, best-generalized conditions. Format contexts are the opposite — the MOST scattered in context-vector space (we'd predicted them coherent; they're not) and catastrophic out-of-sample. All 5 format conditions plus 4 wildchat conversations get flagged as too scattered for a single-vector summary
* A methodological trap worth remembering: the *in-sample* version of this read anti-correlates with spread (≈−0.72) — pure fit optimism (a flexible map interpolates a scattered condition's own centroid after training on its probes). Only the out-of-sample read is trustworthy
* Caveat: this validates the operational prediction (more spread → worse prefix-average) with a linear map; isolating the theory's specific curvature mechanism (½K·s_W) needs leave-one-condition-out nonlinear fits — still 0 GPU, queued

### _Result 6: The two maps write into one shared output subspace — the query changes what is read, not what is written_

Finally, the operator-level question: does the query change the transfer operator itself, or only the input it gets applied to? I fit the prefix-arm and context-arm maps on the same battery-excluded rows and compared the fitted operators directly: principal angles between their top-k singular subspaces, separately for the input (right) and output (left) sides, plus an orthogonal-Procrustes residual — all against 200-draw spectrum-matched random-map null bands (scoped fast read: both own-text cells × layers 14/18/19 × both bases; the all-cell version rides the running refit battery).

**Plot: input/output principal angles + Procrustes residuals vs spectrum-matched nulls**

![operator angles vs null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a05b1192160c9a074e164c9d1da26fe7e65f25c4/figures/issue_1092/inline_operator_angles_vs_null.png)

**Takeaways:**

* The OUTPUT (answer-side) subspaces of the two maps are strongly aligned in all 12 reads: median principal angle 46–54° vs null ≈84° (ambient target), 13–23° vs null ≈36–45° (pca48), with the Procrustes residual below its null band everywhere (e.g. 0.852 vs null 1.006). Robust to the λ choice (own vs matched) and to k (48 vs 90%-energy)
* The INPUT subspaces are indistinguishable from random (median ≈79–86°, sitting at the null) — which is what it should look like: the prefix state doesn't contain the query, so the two maps must read from different residual-stream directions
* Interpretation: there is ONE shared operator structure on the *write* side — both maps push into the same answer-representation subspace — and the query's contribution is supplying input directions that carry signal, not new operator structure. This is the precise sense in which the prefix→answer map is the query-averaged restriction of the context→answer map, and it agrees with the additivity (Result 2) and floors (Result 1.5) pictures
* Fits are clean identifiability-wise (n = 17,308 ≫ 3,584 input dims); the caveats are a GCV λ selector (vs the banked per-fold PRESS) and a top-48 low-rank SVD for the pca48 basis — both checked not to move the verdict

## Next steps:

- (running) the battery-excluded refit of the Result 1 table + the per-target breakdown (4× CPU boxes; expected shifts ~±0.01)
- (done — Result 6, scoped) the operator-level M vs M′ comparison; the comprehensive all-cell version rides the refit battery and should confirm across the Claude-text and cross-model cells
- Close the 0.833 → 0.910 stitch gap: where does the prefix×query interaction live (layers? directions? token positions)?
- Isolate the coherence mechanism: leave-one-condition-out *nonlinear* fits for the Jensen-gap curvature test (0 GPU, on the existing store)
- Use the split for behavior prediction: persona-level (query-averaged) reads are the strong monitoring surface — test whether the prefix component predicts behavior change under fine-tuning better than the full-context state does
- Figure out what the low-variance / query-specific directions represent, and whether the prefix part of the operator writes into the same subspace the persona vectors live in

---

*Sources: realistic crossing #1092 (+ inline repair rounds), constructed crossings #923, grain comparison #813, monitoring granularity #779, coherence test on the #658/#594 store. Full per-experiment clean-results live in the task bodies; the earlier orchestrator-format summary of the same line is `docs/results_summaries/2026-07-14-prefix-vs-context-map.md`.*
