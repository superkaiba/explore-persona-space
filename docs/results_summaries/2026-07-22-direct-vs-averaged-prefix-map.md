# Result: How does the direct prefix map (prefix end → answer) compare to the averaged prefix map (context end averaged over queries → answer)?

*2026-07-22. The direct-prefix vs averaged-prefix half of the 07-22 consolidated writeup, split out as its own question; sibling: [2026-07-17-how-prefix-map-relates-to-context-map.md](2026-07-17-how-prefix-map-relates-to-context-map.md). Every number read from the named #1092 artifacts; sources at the bottom.*

## Motivation

- We've been using an "averaged over queries" prefix map up till now: the **averaged prefix map**, fit on the **averaged prefix vector** — a prefix's context-end states averaged over many queries. That object is post-query by construction; building it takes ~48 forward passes with real queries appended.
- The genuinely pre-query object is the **direct prefix vector**: the prefix-end state, i.e. the activation at the last prefix token, before any query exists (identical across all of a prefix's queries). The map fit on it is the **direct prefix map**.
- I wanted to compare these 2 for the task of predicting **average behavior expression for a prefix** and **average mean answer activation for a prefix** (the only fair comparison, since the direct prefix vector doesn't have access to any queries and so can't do better on predicting single-context behavior expression or single-context mean answer activation).

## TLDR

- **Averaged mean answer activation: the direct prefix vector is much worse.** Held-out $R^2$ 0.37 vs 0.82 (instruct; base 0.26 vs 0.76) against a ≈0.95 ceiling. Its predictions are approximately a uniform shrinkage ($\alpha \approx 0.6$) of the averaged map's, and worse on 100% of 996 prefixes. The two maps are different objects.
- **Averaged behavior expression: parity.** The single direct prefix vector predicts per-prefix sycophancy/hallucination judge scores as well as the 48-query-averaged read (r 0.76/0.89 vs 0.77/0.89, paired diff n.s.), and the parity survives dropping the constructed battery conditions. So the part of the answer state the direct vector fails to predict is NOT the behaviorally relevant part — the disposition is readable before the query. Instruct model only.
- **As a raw persona-vector projection the direct vector is far better** (r 0.665 vs 0.037 on the averaged vector): before the query the trait signal is aligned with $r_B$ itself; reading the query rotates it into other directions, where only a fitted probe recovers it.
- **Why averaging helps:** a single answer state is ~79% "what question is this?" and ~11% "who is answering?", so single-query behavior reads are query-noise-dominated and averaging cancels it (r 0.48 at N=1 → 0.77 at N=48); the direct prefix vector never contained query variance, so it reads 0.76 from one state. But denoising isn't the whole story: the activation ceiling gap means the query-averaged states also carry prefix information the prefix-end state does not linearly hold.
- **The single-context quantities are out of scope by construction, not by choice:** a prefix-constant input is capped at the between-prefix variance share for single-context activations (0.15–0.20; the direct vector makes 0.35–0.58 even of that), and single-context behavior is mostly query + residual (prefix main effect ~3.5%) — invisible to the direct AND the averaged prefix vector alike.
- **When the averaged map fails on real data it's length, not spread** (spread ρ = −0.03 n.s.; conversation turns ρ = +0.83), and there is no averaging-specific spread penalty either (Jensen gap ρ ≤ 0 everywhere). The spread-predicts-error result holds only on the constructed substrate.

## Methodology (shared)

- Model: Qwen-2.5-7B-Instruct + Qwen-2.5-7B (base); teacher-forced capture of own-policy greedy (temp 0.0) answers; layer 14 headline (the line's standing read-out layer), ambient 3584-dim and PCA-48 bases.
- Corpus (#1092): 1,145 real WildChat/LMSYS conversation prefixes sparse-crossed with 1,397 real user queries → 21,193 (prefix, query) contexts; dense core = 99 prefixes × 48 shared queries.
- Two inputs per prefix:
    - **direct prefix vector**: the prefix-end state, from one forward pass over the prefix alone (figures label it "prefix-end")
    - **averaged prefix vector**: mean of the prefix's context-end states over its queries
- Fits: ridge, grouped 6-fold CV by prefix id (a held-out fold never shares a prefix with training; convenience choice carried from #1092, FOLD_SEED=0). Every cross-arm comparison is scored against the SAME pooled answer targets at the same grain.
- Scope: the two PRIMARY comparison tasks are the per-prefix averaged targets — averaged mean answer activation (Result 1) and averaged behavior expression (Result 2). The single-context quantities (one (prefix, query) combo's activation or behavior) appear only in Result 4, as structural context for what a prefix-constant input can never reach.

## Experiments in this write-up

1. **Fair comparison** (07-16) — both maps fit on the same crossed corpus, scored on the same pooled targets at both grains, each against its own ceiling. `eval_results/issue_1092/inline_fair_comparison/fair_comparison.json`
2. **Fair-comparison deep-dive** (07-17) — how the two maps relate: prediction agreement, the global-shrinkage test, per-prefix error ratios. `eval_results/issue_1092/inline_fair_comparison_deepdive/`
3. **Prefix-end behavior monitoring** (07-17) — 149 prefixes × 48 queries; per-prefix mean graded Sonnet judge score (sycophancy, hallucination); ridge probes on both vectors + the raw persona-vector projection + the averaging curve. `eval_results/issue_1092/inline_prefixend_monitoring/results.json`
4. **Divergence/robustness follow-up** (07-22) — battery-vs-real subset parity, real-only refit, where the two reads disagree, query main effects, the per-row behavior decomposition. `eval_results/issue_1092/inline_prefixend_monitoring/divergence_analysis.json`
5. **Crossed ANOVA** on the dense core (99 × 48 = 4,752 contexts) — prefix / query / interaction shares of the answer summary.
6. **Spread-vs-error** — the constructed-substrate coherence analysis (#658) re-run on the 996 natural prefixes, plus the length correlate (07-22).
7. **Theory-sharp spread delta test** (07-22) — does spread predict the averaging-specific penalty (the averaged-vs-direct gap, and the Jensen gap). `eval_results/issue_1092/inline_avgctx_spread_delta/avgctx_spread_delta.json`

## Results

### Result 1 — averaged mean answer activation: the direct map is a ~0.6-shrunk sub-map of the averaged map

**Methodology**
- Fit direct-prefix → answer and context → answer maps on the same crossed corpus; score both against each prefix's mean answer summary over its queries.
- This is the fair field, and almost nothing is structurally out of reach here: the ceiling is ≈0.95 (only ~5% of averaged-target variance is residual within-prefix noise at ~17 queries per prefix).

![fair comparison grid](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f61b5bc49abaa9888bc2455465a599c9bfd83e41/figures/summaries/prefix_vs_context_map/fair_comparison_grid.png)

| held-out $R^2$, averaged answer targets (L14) | direct prefix | averaged read |
|---|---|---|
| instruct (ambient / pca48) | 0.37 / 0.53 | **0.82 / 0.94** |
| base (ambient / pca48) | 0.26 / 0.46 | **0.76 / 0.88** |

(The averaged-read column is the per-row context map's predictions averaged over queries. A map fit directly on the averaged prefix vector itself reaches $R^2$ 0.65/0.87 instruct and 0.60/0.80 base — the delta test's centroid arm — still far above the direct prefix map.)

**Takeaways**
- The direct prefix vector is a strictly weaker summary of the prefix than the averaged one, not the same object: 0.37 vs 0.82 against a ≈0.95 ceiling.
- How the two relate: their predictions agree at only $R^2$ 0.26–0.54. The direct map's predictions are approximately a global shrinkage of the averaged map's ($\alpha$ = 0.60 ambient / 0.66 pca48; 100% of per-dimension coefficients < 1, and per-dimension coefficients improve residual variance only ~11 points over the single scalar), and worse on 100% of 996 prefixes (mean per-prefix error ratio ≈2× ambient, ≈3× pca48).
- Reading: the direct map predicts the same deviations from the average answer state as the averaged map, at ~60% amplitude plus extra noise — a uniformly attenuated version of the same signal, not a different read.

### Result 2 — averaged behavior expression: the direct prefix vector reads disposition at parity with the averaged read (instruct model)

Result 1 says the direct prefix vector loses a lot of answer-state prediction. Is the lost part behaviorally relevant? Test: predict each prefix's average behavior from both representations.

**Methodology**
- 149 prefixes (99 real WildChat/LMSYS + 50 constructed battery conditions) × 48 queries = 7,152 rows.
- Target: each prefix's mean graded judge score over its 48 answers (sycophancy and hallucination; graded Sonnet judge, on-policy answers).
- Inputs: direct prefix vector vs averaged prefix vector; ridge probe, held out by prefix.

![prefix-end monitoring vs averaged read](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/issue_1092/prefixend_monitoring.png)

**Takeaways**
- Parity: sycophancy r = 0.759 (direct) vs 0.774 (averaged); hallucination 0.888 vs 0.887; paired difference +0.015, 95% CI [−0.026, +0.052]. Reliability ceilings are 0.86/0.91, so both reads sit near ceiling.
- So the part of the answer the direct prefix map is worse at predicting (Result 1) is NOT the behaviorally relevant part: the disposition to be sycophantic / hallucination-prone is already readable at the end of the prefix, before the query is seen.
- The raw persona-vector projection inverts the parity: $\langle \text{state}, r_B \rangle$ reads the trait on the direct prefix vector (r 0.665) but not on the averaged one (r 0.037). The trait signal is aligned with $r_B$ before the query and rotated into other directions once the query is read; the fitted probe recovers it there.
- Instruct-only: base-model direct-prefix reads collapse (r 0.01–0.05; averaged read 0.13 sycophancy / 0.62 hallucination).
- Scope caveat: this is average behavior for a prefix — a between-prefix disposition read, a small (~3.5%) slice of total per-answer behavioral variance (Result 4). It ranks contexts by tilt; it does not predict single answers.

**Robustness and divergence (follow-up analysis)**

Since 50 of the 149 "prefixes" are constructed battery conditions, the worry was that easy, extreme battery conditions carry the parity. They don't, and the places where the two reads disagree are informative:

![divergence between the two reads](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5bdc1a568b/figures/issue_1092/prefixend_monitoring_divergence.png)

- Parity holds on purely natural conversations. Stored held-out predictions restricted to subsets: real-only r = 0.757 vs 0.784 (sycophancy) and 0.921 vs 0.908 (hallucination); battery-only 0.772 vs 0.754 and 0.794 vs 0.836. A full refit on only the 99 real prefixes gives 0.802 vs 0.828 (sycophancy, diff CI [−0.009, +0.060]) and 0.941 vs 0.921 (hallucination, CI [−0.041, +0.0002], direct prefix marginally ahead).
- The two reads' predictions correlate r 0.93/0.95, so divergence is the exception. Where they disagree concentrates on battery format conditions (mean |Δ| ≈1.7×/2.2× the real-prefix level for sycophancy/hallucination) and on LONG real conversations (|Δ| vs user turns ρ ≈ +0.77) — the same length axis as Result 5: on a long prefix, the last-token state and the query-averaged state have drifted apart as summaries, so probes built on them disagree.
- Neither read is systematically more accurate where they disagree (mean differential |error| +0.02 sycophancy / −0.14 hallucination, and per-prefix wins go both ways: the direct read overpredicts sycophancy at 8.2 on fmt_code_comment where the truth is 4.0 and the averaged read gets 3.1, but beats the averaged read on fmt_json, fmt_xml, and some high-hallucination creative-writing prefixes). The disagreement looks like two differently-noisy encodings of the same disposition, not one good and one broken read.
- Full per-prefix and per-query tables: `eval_results/issue_1092/inline_prefixend_monitoring/divergence_analysis.json`.

### Result 3 — why averaging helps: it cancels the query component, which the direct prefix vector never had

**Methodology**
- Crossed ANOVA over the dense core (99 prefixes × 48 shared queries = 4,752 contexts): decompose the single-context answer summary into prefix / query / prefix×query interaction.
- Averaging curve: r of the averaged-read behavior probe as a function of the number of queries averaged (N).

![variance shares](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/summaries/prefix_vs_context_map/variance_shares.png)

**Takeaways**
- Query explains ~79% of answer-state variance, prefix ~11%, interaction ~10% (pca48, L14; ambient basis: query 60–72%, prefix 10–12%, interaction 18–28% across the two models).
- So each single context read is ~79% "what question is this?", which is noise for persona readout. Averaging cancels it: the averaged behavior read climbs r 0.48 (N=1) → 0.70 (N=6) → 0.77 (N=48). The direct prefix vector never contained the query component, so it reads 0.76 from ONE state — for disposition monitoring, one pre-query forward pass replaces 48 queries of averaging.
- But denoising is not the whole story: Result 1's ceiling gap (0.37 vs a ≈0.95 ceiling) means the query-averaged context states also carry prefix information — expressed through how the prefix transforms each query — that the prefix-end state does not linearly hold.

### Result 4 — structural context: the single-context quantities are out of reach for any prefix-constant input

Neither single-context target is a fair test of the direct prefix vector, but the caps are informative in themselves.

**Takeaways**
- Single-context activations: the direct prefix vector reads $R^2$ 0.07/0.10 vs the context map's **0.81/0.91**. Its ceiling is the between-prefix variance share (0.15–0.20) — ~80% of single-answer variance is query-borne (Result 3) and a prefix-constant input cannot see it. Even against that ceiling it makes only 0.35–0.58 (instruct 0.43/0.58, base 0.35/0.53).
- Single-context behavior: the prefix main effect is only 3.5% (sycophancy) / 3.3% (hallucination) of per-row judge-score variance; query main effects are 53%/67% and the residual 43%/30%. A few queries push behavior up across essentially all prefixes (e.g. qry_00043, a coding query, with disposition-expression r ≈ 0.0 for sycophancy: on that query the prefix is irrelevant).
- So single-context behavior is invisible to the direct AND the averaged prefix vector alike (median per-query disposition-expression r 0.16/0.20); the best read available here is a probe on the single context-end state: r 0.21 (sycophancy) / 0.42 (hallucination), near zero on the base model.

### Result 5 — when the averaged map fails on real data: length, not spread

Theory prediction: the averaged prefix vector is a valid summary only where the prefix's context vectors cluster, so within-prefix spread should track the averaged map's per-prefix error.

**Methodology**
- Per-prefix held-out error of the averaged map vs within-prefix spread of its context vectors, on (a) the constructed substrate (50 conditions, whitened spread) and (b) the 996 natural prefixes (raw-L2 spread); on (b) also vs prefix length (conversation turns).

Constructed substrate (supports the prediction):

![spread vs residual, constructed](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/issue_658/fig_a35a_spread_vs_residual_loco_L14.png)

Natural prefixes (does not transfer):

![spread vs error, natural](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/summaries/prefix_vs_context_map/perprefix_error_vs_spread.png)

![error vs length, natural](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/summaries/prefix_vs_context_map/perprefix_error_vs_length.png)

**Takeaways**
- Constructed conditions: spread strongly predicts held-out residual (median layerwise Spearman +0.89, 28/28 layers positive). This is where the "theory confirmed" read came from.
- Natural prefixes: the correlation is null on instruct (ρ = −0.03, n.s.) and INVERTED on base (ρ ≈ −0.8). Prefix length dominates instead: turns → error ρ = +0.83 (instruct) / +0.80 (base); controlling for length, spread stays negative (partial ρ −0.33 / −0.52).
- One reading: on long real prefixes, appending different queries moves the state less, so exactly the hard, long prefixes have LOW spread and the naive "more spread ⇒ harder" prediction fails on natural data. The coherence condition may still be right as a validity bound, but spread is not its observable there; length is the practical difficulty axis.

**Theory-sharp delta test (follow-up)**

The panels above correlate each map's error with spread separately; the coherence condition specifically predicts an AVERAGING-specific penalty. The sharp test: does spread predict (a) the averaged map's error over and above the direct map's (the gap between the two), and (b) the Jensen gap — the error of a map fit on the averaged prefix vector (map-the-centroid) minus the error of averaging the per-row context map's predictions?

![spread vs averaging-specific error gaps](https://raw.githubusercontent.com/superkaiba/explore-persona-space/61be8fa96a/figures/summaries/prefix_vs_context_map/perprefix_avgctx_delta_vs_spread.png)

- All arms fit on the identical folds and targets as Result 1 (996 natural prefixes, battery-excluded; parity gates against the banked arms matched exactly, diff 0).
- Spread does not predict the averaging-specific penalty. Spread → (averaged-map error − direct-map error): ρ = +0.02 / +0.06, both n.s. (instruct, ambient/pca48) and −0.19 / −0.05 (base). Spread → Jensen gap: ρ −0.06 to −0.19, and −0.12 to −0.22 with length controlled. Higher-spread prefixes pay a relatively SMALLER averaging penalty, the opposite sign from the naive prediction.
- Scope caveat: raw-L2 spread here (matching the natural-prefix panels above); the constructed substrate used whitened spread. A whitened natural-side replication is the remaining loose end.

## Conclusion and next steps

- On the two fair tasks: the direct prefix vector is a strictly weaker (≈0.6-shrunk) summary for **averaged answer-state prediction**, but an equally good input for **averaged behavior expression** — and the only one where the raw persona-vector projection works.
- Practical monitoring implication: one pre-query forward pass reads a prefix's disposition as well as 48 queries of averaging (instruct model only; a disposition read — it says nothing about single-answer behavior, which is mostly query + residual).
- On real data the averaged map's validity condition is not observable as context-vector spread (null/inverted, and no averaging-specific penalty either); length is the practical difficulty axis.
- Next: does the direct-prefix disposition read survive fine-tuning; a whitened-spread natural-side replication of the coherence test; the operator-level coincidence check (is the averaged map the same linear operator as the context map, not just equally accurate — in flight); how the averaged map relates to the context map lives in the sibling write-up.

## Sources

- Fair comparison + deep-dive: `eval_results/issue_1092/inline_fair_comparison/fair_comparison.json`, `eval_results/issue_1092/inline_fair_comparison_deepdive/deepdive.json`
- Behavior monitoring + divergence + per-row decomposition: `eval_results/issue_1092/inline_prefixend_monitoring/{results,readout_constructions,divergence_analysis}.json`
- Spread delta test: `eval_results/issue_1092/inline_avgctx_spread_delta/avgctx_spread_delta.json` (commit `61be8fa96a`)
- Constructed-substrate coherence: `eval_results/issue_658/` (A3.5a)
- Task bodies: #1092 (HIGH), #658; terminology: `docs/glossary_context_answer_map.md`; siblings: [2026-07-17-how-prefix-map-relates-to-context-map.md](2026-07-17-how-prefix-map-relates-to-context-map.md), [2026-07-22-prefix-query-context-answer-map-consolidated.md](2026-07-22-prefix-query-context-answer-map-consolidated.md)
