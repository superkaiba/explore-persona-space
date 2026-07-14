# Result: The prefix→answer map is the query-average of the context→answer map — the query part dominates per-context prediction, and the prefix part is what survives averaging

## Motivation

* We had shown two separate maps over the residual stream:
    * a map from **average context vectors** (averaged over user queries) to the **average answer profile** — the prefix map `M`: `v̄_A ≈ M·v_P`, one row per prefix (the #658/#810 line, 50 prefixes × 48 queries)
    * a map from a **single context vector** to a **single answer summary** — the context map `M′`: `v_A ≈ M′·v_C`, one row per (context, answer) (the #779 line, 5,000 LMSYS contexts)
* The question this summary answers is **what is the relationship between these two maps**:
    * the single-context map presumably contains more granular information
    * the average map presumably contains broader information
    * hypothesis: what the average map keeps **is persona information** — the behavior that is shared across different queries
* Everything here is on MATCHED queries/contexts (the same prefixes crossed with the same queries, both mapping arms fit on every read) — the earlier comparison across different datasets was confounded, which is what #1092 was built to fix

## TLDR

- Definitionally the prefix map is the query-averaged aggregation of the context map (`v_P = mean_q v_C`, profile `= mean_q v_A`); the empirical question is what each grain carries.
- On matched real-conversation crossings, the query-bearing context state carries nearly all the transport: context-arm held-out R² **0.71–0.80** vs prefix-arm **0.04–0.08** (#1092, HIGH).
- The context map decomposes near-additively into a prefix part + a query part (interaction ~10%): a stitch of two **disjoint** forwards (prefix-only + bare-query-only) recovers R² **0.833** of the full-context map's **0.910** (#1092; replicating #923 on realistic data).
- Averaging the map over queries changes almost nothing *on averaged targets*, but it discards the query-specific component — which is the **only** component that transfers to held-out contexts at per-example grain (#813).
- What survives the averaging is the persona component: only ~8–13% of per-context variance, but consistent across queries, so persona-level monitoring jumps from per-prompt r **0.34/0.63/0.09** (evil/sycophancy/hallucination) to **0.66/0.89/0.53** when 40 questions are averaged per persona (#779). The "average map captures persona information" hypothesis holds in exactly this shared-across-queries sense.

## Methodology

Four substrates feed this summary (each is its own clean-result; confidence tags per issue):

- **#1092 — realistic matched crossing (HIGH).** 1,145 real WildChat/LMSYS conversation prefixes (329 with ≥5 user turns) sparse-crossed with a 1,397-query bank of real user turns → 21,193 rows (dense core ≈100 prefixes × 48 queries). Models: `Qwen2.5-7B-Instruct` + `Qwen2.5-7B` (no fine-tuning). A 4×2 text-source × model factorial (own-text / Claude-written / shuffled-pairing derangement as the carrier floor). Per row the reading model is teacher-forced and states are captured at the **prefix end** (everything before the user query) and the **context end** (prefix + query), plus answer-span targets. Ridge maps state → answer target per cell × input arm × layer (14/18/19) × basis, grouped 6-fold CV on novel-prefix folds. **Both prefix-based and context-based arms are fit on every read — this is the matched design.**
  - Corpus examples dashboard: https://htmlpreview.github.io/?https://gist.githubusercontent.com/superkaiba/40e76f146d52583c1e178f5ad5ca2910/raw/issue1092_corpus_dashboard.html
- **#923 — constructed crossing (MODERATE).** UltraChat prefix × query grids; variance decomposition of the per-cell mean answer activation into prefix ("context" in that body's naming) / query / interaction main effects, plus the disjoint two-forward stitching test at layer 18.
- **#813 — grain comparison (LOW).** The same substrate fit at question-**averaged** grain vs per-**example** grain; cross-grain transfer, leave-one-context-out generalization, and within-context (query-specific) reads at layer 14.
- **#779 — monitoring granularity (MODERATE).** A generic context→answer map trained on LMSYS; trait monitoring read per-context vs averaged over 1–40 questions per held-out persona group (LOGO over 60 groups; graded 0–100 `claude-sonnet-4-5` judge, 5 draws).

Shared definitions (project glossary): **prefix** = everything before the user query (system prompt / persona / prior turns); **query** = the user message; **context** = prefix + query; `v_C` = activation at the last prompt token of one context; `v_P` = a prefix's `v_C` averaged over queries; `v_A` = mean activation over one answer's tokens; **behavior profile** = a prefix's `v_A` averaged over queries. Metric throughout: held-out reconstruction R² (variance-weighted over the 3,584 dims), grouped folds, against permutation / random-map / shuffled-pairing nulls.

## Results

### _Result 1: On matched crossings, the single-context map carries nearly all the transport (context R² 0.71–0.80 vs prefix 0.04–0.08)_

The first thing we needed was the confound-free head-to-head: the same prefixes, the same queries, both mapping arms fit on every read. For each of the eight text-source × model cells I compare the held-out R² of the ridge map fit from the prefix-end state vs the context-end state (layer 14, pooled answer targets).

**Plot: held-out R² per cell, prefix-based vs context-based arms**

![prefix vs context R2](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cfba89b1c820dd033c0efbf4bc15119b4387805f/figures/issue_1092/read1_r2_prefix_vs_context.png)

| cell | context R² | prefix R² | gap |
|---|---|---|---|
| Instruct, own answers | 0.804 | 0.065 | 0.739 |
| Instruct, Claude answers | 0.776 | 0.053 | 0.724 |
| Instruct, pretrained answers | 0.712 | 0.043 | 0.669 |
| Pretrained, own answers | 0.714 | 0.051 | 0.663 |
| Pretrained, Claude answers | 0.742 | 0.056 | 0.687 |
| Pretrained, instruct answers | 0.493 | 0.079 | 0.414 |
| Instruct, shuffled answers | 0.079 | 0.016 | 0.063 |
| Pretrained, shuffled answers | 0.057 | 0.028 | 0.028 |

**Takeaways:**

* The query-bearing context state carries nearly all held-out map skill; the gap is layer/basis/fit-arm stable (layer 18 context 0.823; pca48 0.910 vs 0.096).
* The prefix maps are small but real — both arms clear their permutation nulls in all six coherent cells. The prefix state is not empty, it is just a minor share of per-context prediction.
* The shuffled-pairing cells collapse both arms (0.06–0.08), so neither map is a corpus artifact — it needs the true prefix–query–answer pairing.
* Known repairs in flight (all 0 GPU, running now): a battery-excluded refit (2,400 eval-bridge rows leaked into fit training), three registered reads not yet banked, and identity/affine transport floors that were never computed. None of these is expected to move the headline gap, but the numbers above should be re-quoted from the repaired fits.

### _Result 2: The context map is near-additive in a prefix part and a query part — two disjoint forwards recover most of the full map_

Can we decompose context→answer into prefix→answer + query→answer? Two tests say yes, with a ~10% interaction residual.

First, the operator-additivity test: residuals of an additive (prefix + query) model of the fitted map, against 200-draw random-map null bands.

**Plot: operator additivity residuals per cell vs null bands**

![operator residuals](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cfba89b1c820dd033c0efbf4bc15119b4387805f/figures/issue_1092/read4_operator_residuals.png)

Second, the variance decomposition and the stitching test:

**Plot: prefix / query / interaction variance shares — constructed (#923) vs realistic (#1092)**

![variance shares](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e26781d3700342e2412ce775b913d91fb4505efb/figures/summaries/prefix_vs_context_map/variance_shares.png)

**Plot: a disjoint prefix+query stitch recovers most of the full-context map (instruct-own cell, layer 14, pca48)**

![stitch recovery](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e26781d3700342e2412ce775b913d91fb4505efb/figures/summaries/prefix_vs_context_map/stitch_recovery.png)

**Takeaways:**

* Both additivity residual tests sit far below their random-map null bands in every coherent cell (0.42–1.00 vs 5th-percentile bands 1.59–2.10): the context operator is close to prefix-part + query-part.
* The variance shares transfer from the constructed grid to realistic data: #923 (UltraChat) 7.8% prefix / 83.7% query / 8.6% interaction; #1092 (WildChat/LMSYS) 10.7% / 79.0% / 10.3%. The realistic prefix share runs slightly above the grid's.
* Stitching two **disjoint** forwards — a prefix-only read plus a bare-query-only read, no attention between them — reaches R² 0.833 against the full-context map's 0.910, while bare-query alone gets 0.146 and prefix alone 0.096. You need both parts; what attention mixing adds on top is comparatively small at this granularity (#923 measured the mixing increment at +0.103 at matched query-span granularity — real, but second-order next to the additive structure).
* So the decomposition asked for in the original question exists: context→answer ≈ prefix→answer + query→answer, with the query part dominating.

### _Result 3: The average-grain map and the per-example-grain map agree on averaged targets — but the query-specific component is the only part that transfers across contexts_

Next, the direct grain comparison (#813): fit the map at question-averaged grain vs per-example grain on the same substrate and test what each generalizes to.

**Plot: own-grain fits vs cross-grain transfers (LOCO, layer 14)**

![grain transfer](https://raw.githubusercontent.com/superkaiba/explore-persona-space/424c98492f25b17cf172256c71460e0c37e50bd2/figures/issue_813/hero_pe_vs_avg_transfer_trained.png)

**Plot: within-context (query-specific) R² per cell vs shuffle nulls**

![query-specific signal](https://raw.githubusercontent.com/superkaiba/explore-persona-space/424c98492f25b17cf172256c71460e0c37e50bd2/figures/issue_813/dv4_query_specific.png)

**Takeaways:**

* On question-averaged targets, refitting per-example changes nothing — cross-grain transfer gaps ≤0.19 R², with intervals spanning zero in 23/24 reads. At that grain the two maps are interchangeable.
* But **neither grain generalizes to held-out contexts** at layer 14 (pooled leave-one-context-out R² deeply negative, −51 to −14 — n=50 contexts, LOW confidence), while the **within-context** component is strong: R² 0.71–0.87 vs shuffle nulls ≈0. The component that actually transfers is query-specific — exactly the component the question-averaged map discards by construction.
* A caution on a related earlier claim: the "averaging collapses the map's rank ~4×" reading from the 50-condition setting is an artifact of averaging over FEW conditions. At ≈1,046 averaged prefixes the averaged/per-example stable-rank ratios are 0.82–1.10 (#1092):

![rank by grain](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cfba89b1c820dd033c0efbf4bc15119b4387805f/figures/issue_1092/read2_rank_by_grain.png)

### _Result 4: What query-averaging keeps is the persona component — persona-level monitoring jumps even though the prefix factor is ~10% of per-context variance_

Finally, the hypothesis that what is captured by the average map vs the single map **is persona information** — behavior shared across queries. The cleanest read is monitoring granularity (#779): the same generic map, read per-prompt vs averaged over 1–40 questions per held-out persona.

**Plot: monitoring r vs number of questions averaged per persona group**

![persona-level monitoring curves](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73a0e157cdeef6bf332c803f519e89773bd0a893/figures/issue_779/grouped_context.png)

**Plot: per-prompt vs 40-question persona-averaged monitoring r, per trait**

![persona level vs per prompt](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e26781d3700342e2412ce775b913d91fb4505efb/figures/summaries/prefix_vs_context_map/persona_level_vs_per_prompt_r.png)

**Takeaways:**

* Averaging 40 questions per held-out persona lifts trait-monitoring r from 0.34/0.63/0.09 to 0.66/0.89/0.53 (evil/sycophancy/hallucination). The per-answer state is query-dominated; averaging cancels the query component and the small-but-consistent persona component emerges.
* This is consistent with the variance picture: the prefix factor holds only ~8–13% of per-context variance (Result 2), yet it is the part that is stable across queries — which is precisely what "persona information" should mean operationally.
* One honest caution from the matched substrate: on #1092's realistic prefixes (natural conversations, only weakly persona-differentiated), the prefix factor's share of variance **along trait directions** is 1.7–5.2% and does not beat a random-direction null. The strong persona-level reads come from explicitly persona-differentiated conditions (#779's persona groups). So "the average map captures persona information" is supported where personas actually differ, and is not an automatic property of any prefix.
* The theory side formalizes when the average is even a valid stand-in: the coherence assumption (A3b) bounds the average-vs-single gap by the within-prefix spread of context vectors (a Jensen-gap argument). The direct test just landed (inline on the #658/#594 stores, 0 GPU) and **supports it out-of-sample**: within-condition spread positively predicts the held-out (leave-one-condition-out) mean-vector-predictor residual — median layerwise Spearman +0.89, 28/28 layers positive, and it holds within the persona family alone (+0.76). Personas are the tightest, best-generalized conditions; all 5 format conditions (plus 4 WildChat conversations and 4 others) are flagged as too scattered for the prefix-average summary to represent.

**Plot: within-condition spread vs held-out (LOCO) mean-vector-predictor residual (layer 14)**

![spread vs LOCO residual](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d794cc9afd8cd5ed9c7909eca69ab8c061982906/figures/issue_658/fig_a35a_spread_vs_residual_loco_L14.png)

* Two honest caveats on that read: (1) it validates the operational prediction (more spread → worse prefix-average prediction) with a **linear** map; the theory's specific ½K·s_W curvature mechanism needs leave-one-condition-out **nonlinear** fits, which are deferred (still 0 GPU). (2) The in-sample nonlinear reads anti-correlate with spread (≈−0.72) — a spread-dependent fit-optimism artifact (a flexible map interpolates a scattered condition's own centroid after training on its probes), which is why only the out-of-sample read is trusted.

## Next steps:

- (running, inline, 0 GPU) **#1092 repairs**: battery-excluded refit, bank the per-target R² columns + topic-matched pairing delta, identity/affine transport floors — re-quote Result 1 from the repaired fits.
- (running, inline, 0 GPU) **Operator-level M vs M′**: principal angles + Procrustes residual between the fitted prefix-arm and context-arm operators — decides whether the query adds operator structure or only input information (if the operators align, there is ONE transfer operator and the prefix map is just its query-averaged application).
- (done, inline — folded into Result 4) **A3.5a coherence test**: supported out-of-sample (spread predicts LOCO residual, +0.89 median Spearman, 28/28 layers). Remaining piece: leave-one-condition-out **nonlinear** fits to isolate the curvature (½K·s_W) mechanism specifically (~50 MLP fits/layer, still 0 GPU).
- Push the stitch further: can a prefix-map + query-map combination close the remaining 0.833 → 0.910 gap, and does the interaction term localize to specific layers/directions?
- Use the split for behavior: persona-level (query-averaged) reads are the strong monitoring surface; test whether the prefix component predicts behavior under fine-tuning (the #811/#833 map-change line) better than the full-context state does.
- Resolve the low-variance-direction question: the per-context map reconstructs high-variance answer directions well and low-variance ones badly — check whether the persona component lives in directions the map can actually carry across realistic prefixes.
