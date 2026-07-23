# Result: Steering with the context-vector difference shifts the answer state toward the target, and behavior follows at the layers where the geometry aligns

## Motivation

* The context→answer mapping line so far (#685, #779, #823, #841, #922, #1092) is entirely correlational: probes, ridge maps, teacher-forced predictability reads, and cross-persona map comparisons over the context vector $V_c$ (last-context-token activation) and the answer summary $V_a$ (mean answer-span activation), all computed on passively collected activations.
* I wanted to test whether these vectors have causal meaning: if you take a context $c$ and steer it with $\Delta = V_c(c') - V_c(c)$, does the average answer activation shift from $V_a(c)$ toward $V_a(c')$? Does the model's behavior shift with it? Does the fitted context→answer map predict the shift? And does logit lens decode anything interpretable on these vectors?

## TLDR

- Geometric causality holds: adding $\Delta$ at a single token (the last context token, prefill only) moves the realized mean answer activation toward $V_a(c')$ — cosine 0.36–0.41 with the target direction, 28/28 pairs above their norm-matched random-direction nulls on both extraction arms. But the step is small: 2–5% of the target-shift norm.
- Behavior follows the geometric layer profile: +0.9 judge points at the layer-20 primary read, but +6.2 at layer 14 on the context arm (p = 0.008, 21% of the context-swap ceiling; the prefix arm stays near floor), replicated on two fresh seed bases (+6.6, p = 0.007; +5.0, p = 0.025). Best single case: the terse pair flips 0 → 94 on all 10 draws — a one-line answer with the terseness instruction never in context.
- The direction matters: steered $\Delta$s beat the persona-vector $r_B$ baselines geometrically (0.36–0.41 vs 0.09–0.16), pair-specificity is full on the prefix arm (28/28 above the shuffled-pair band) and partial on the context arm (20/28), and matched-query $\Delta$s carry over far better than cross-query ones (0.49 vs 0.22, p = 8e-6).
- The fitted #922 ridge map does NOT predict the realized shift: transport cosine 0.00 at layer 20, magnitude over-predicted ~16×. The raw vector difference is causal; the fitted linear map does not carry over to the intervention regime (at the one layer tested).
- Logit lens on $V_c$, $V_a$, and $\Delta$ decodes to junk tokens: whatever these vectors carry, a plain unembedding read does not translate it into trait words.
- Caveats: instruct model only (Qwen-2.5-7B-Instruct), one pair bank, seed base 42 (+2 fresh seed bases for the behavioral replication), and the all-position steering variant breaks generation outright (96–98% of draws flip into Chinese script), so only the single-token reads are interpretable.

## Methodology

- Model: Qwen-2.5-7B-Instruct (28 layers, hidden size 3584), no training. Every compared arm runs on the same hooked HF `generate()` stack — the forward hook forces HF for the treatment arm, and a mixed vLLM/HF contrast would confound both DVs.
- Pair bank: 28 context pairs $(c, c')$, all LLM-generated synthetic corpora reused from the mapping line's artifact banks:
    - 15 matched-query pairs — same user question, different context: 10 instruction swaps (no system prompt vs one behavioral instruction) + 5 persona-condition swaps
    - 13 cross-query pairs — different questions under different personas
    - Example (matched-query instruction swap, pair `m685_05_formal`):
        - $c$: no system prompt, query "How do I make a good cup of coffee?"
        - $c'$: system prompt "Respond in extremely formal, bureaucratic language.", same query
- Computed quantities:
    - $V_c(c)$: residual-stream activation at the last context token, per layer. Two arms per the standing project rule: prefix-based (last token of the system/persona prefix) and context-based (last token of prefix + user query)
    - $V_a(c)$: mean residual-stream activation over the answer span, averaged over 10 on-policy draws under $c$
    - $\Delta = V_c(c') - V_c(c)$: the steering vector, one per pair per arm per layer
- The intervention: a forward hook adds $\alpha \Delta$ to the residual stream at layer $\ell$ at the last context token only, during the prefill pass only — decode steps are untouched, and the edit propagates to everything generated after it through that position's KV cache. Sweeps: $\ell \in \{7, 10, 14, 17, 20, 21, 24\}$ (20 primary), $\alpha \in \{0.5, 1, 2, 4\}$ under a coherence gate (a cell is only used if ≥50% of its draws stay coherent), plus an all-positions variant (the persona-vectors steering convention) as a comparison.
- Arms and controls:
    - Unhooked baseline under $c$
    - Steered arms (layer × α grid)
    - Context-swap ceiling: generate under $c'$ directly — the most any intervention could achieve
    - Norm-matched random-direction null: 500 draws per pair, selection-symmetric (the null rides the same max-over-layers selection as the observed statistic)
    - Shuffled-pair null: other pairs' real $\Delta$s, norm-matched — tests whether alignment needs THIS pair's direction or any real steering direction
    - Persona-vector steering baseline: per-layer difference-of-means $r_B$ directions (evil, sycophancy, hallucination) from #779, same α grid
- Metrics:
    - Geometric DV: cosine between the realized answer shift (mean steered $V_a$ − baseline $V_a$) and the target direction ($V_a(c') - V_a(c)$), max over the 7 read layers at α = 4, each pair against its own null band. One correction landed during review: the first computation reused the same 10 baseline draws inside both the shift and the target, which adds a shared noise term the null cannot model and inflates every cosine ~0.08; the corrected read splits the baseline draws into disjoint halves. Both conventions are reported (the disjoint one is itself attenuated by halved baseline sample size, so the truth sits between them).
    - Behavioral DV: graded 0–100 judge score (claude-sonnet-4-5, reason-then-score, 5 judge draws per completion, per-pair rubric; ~21,000 Batch-API calls, 0 transport losses) + binary rate companion (ρ(graded, binary) = +0.67 on cells with dynamic range)
    - Map transport: cosine between the realized shift and the fitted map's counterfactual prediction $f(V_c + \Delta) - f(V_c)$, using the reused #922 layer-20 ridge map
    - Logit lens (descriptive): top-10 unembedded tokens of $V_c$, $V_a$, $\Delta$, and the #922 slow modes
- Generation: N = 10 draws per cell, temperature 1.0, 1024 max new tokens, seed base 42; replication rounds re-sample the layer-14 cells + fresh baselines on seed bases 43/44 with the parent steering vectors frozen.

## Results:

### _Result 1: Steering moves the answer state toward the target — every pair clears its null, but the step is small_

The first question was whether the intervention moves the answer state at all, and in the right direction. I steered each pair at every layer/α cell, extracted the realized answer profiles from the steered completions, and compared each pair's alignment cosine to its own random-direction null band.

**Plot: per-pair alignment, shared → disjoint baseline conventions, with null bands**

![per-pair dumbbells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c78d145929017d6ff26285460cdb35322e523e71/figures/issue_1415/h1_per_pair_scatter.png)

**Takeaways:**

* The answer state moves toward $V_a(c')$: mean alignment 0.36 (prefix arm) / 0.41 (context arm) under disjoint baselines (0.44 / 0.48 shared), with 28/28 pairs above their own random-direction nulls (97.5th percentile ≈ 0.04) on both arms; 27/28 on the most conservative single split
* But the traversal is small. The realized step covers 2–5% of the target-shift norm (matched-cell mean 0.035 prefix / 0.055 context) — the steering rotates the answer state toward the target without getting anywhere near it
* The forward pass amplifies the injection: the injected $\Delta$ itself aligns with the answer-space target at only 0.117 (prefix), yet the realized shift aligns at 0.364, so most of the target-aligned movement is produced downstream of the injected position
* The shared-baseline artifact matters at this scale of effect: reusing the same baseline draws in shift and target inflated every cosine ~0.08, and the layer-20-anchored read corrects proportionally more (0.27 → 0.19 prefix)
* One pair is a known bad target: the medical-doctor pair's target direction has split-half reliability 0.049 (sampling noise), so its individual value is uninterpretable; the aggregate does not depend on it

### _Result 2: The direction is pair-specific, and matched-query differences transport far better than cross-query ones_

A cosine above a random-direction null could still come from any real steering direction rather than from this pair's direction specifically. The shuffled-pair null — steering pair $i$ with pair $j$'s norm-matched $\Delta$ — tests pair-specificity, and the matched/cross split tests whether $\Delta$s carry over across queries.

**Plot: alignment distributions vs both null bands**

![null bands vs observed](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c78d145929017d6ff26285460cdb35322e523e71/figures/issue_1415/null_band_vs_observed.png)

**Takeaways:**

* Prefix-arm alignment is fully pair-specific (28/28 pairs above the shuffled-pair band); the context arm is only partly so (20/28) — part of the context arm's higher raw alignment is direction-generic
* Matched-query pairs transport far better than cross-query: disjoint 0.49 vs 0.22 (prefix, one-sided p = 8e-6) and 0.48 vs 0.31 (context, p = 4.4e-3). Caveat: matched pairs share the query, so closer source–target answer states alone could carry part of the gap
* The pair-specific $\Delta$s also beat the persona-vector $r_B$ baselines geometrically at the same selection (0.36–0.41 vs 0.09–0.16 disjoint) — a direction built from THIS pair's contexts moves the answer state more than a generic trait direction

### _Result 3: Behavior follows the geometric layer profile — near zero at layer 20, a fifth of the ceiling at layer 14_

Geometry moving does not by itself mean the model behaves differently. I judged the steered completions with the standard graded rig. At the layer-20 primary read the answer was basically no: +0.33 (prefix, indistinguishable from baseline given the variance) and +0.91 judge points (context, p = 0.0005 but ~3% of the +28.8 ceiling shift) — while persona-vector evil steering moves +10.1 at the same selection, and under-dosing does not explain it (the steered arms injected 1.6–3.6× more norm). But the layer sweep showed the layer-20 anchor was the problem:

**Plot: geometric alignment and judged behavior shift, per steer layer**

![layer profile](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7d4f686bfb519175780c9e789d827b8610bce160/figures/issue_1415/layer_profile_geometry_vs_behavior.png)

**Takeaways:**

* Alignment and behavior peak at the same mid-stack layers: geometric alignment peaks at layers 14–17, and the context arm's judge shift peaks at layer 14 — +6.18 points (p = 0.008, paired signed-rank), 21% of the ceiling, 7× the layer-20 read; layers 7–10 and the prefix arm stay near floor
* Matched pairs carry all of it: +11.8 matched vs −0.3 cross — behavioral steering transfers only where the geometric transport was already strong
* The effect is concentrated in a few pairs: the terse pair flips completely (0 → 94, all 10 draws answering in one clipped line with no instruction in context), five more matched pairs shift +7 to +25, the median pair stays at 0; excluding the terse pair keeps the effect (+2.9, p = 0.015). The layer-20 context shift is similarly concentrated (34% carried by the formal-register pair, surviving its exclusion at +0.63, p = 0.0009)
* Steered pools carry 9–10% Chinese-script intrusion at layer 20 (a single steered position already degrades some draws); recounts excluding intruded draws move the judged shift by under 0.4 points and the geometric headline by under 0.01, so the intrusion does not carry the result
* Dose-response over α is flat and the coherence gate never bound, so larger α was never explored — the operating point is probably not the most single-token steering can do

### _Result 4: The layer-14 behavioral effect survives fresh sampling_

Since the layer-14 read came out of a sweep, I re-sampled the fixed layer-14/α = 4 cells (both arms, all 28 pairs) plus fresh baselines on two new seed bases with the steering vectors frozen, and judged them under the identical recipe.

**Plot: layer-14 shift across three sampling rounds + per-pair original vs replication**

![replication](https://raw.githubusercontent.com/superkaiba/explore-persona-space/67f9d97a9478beb299fc0dcc2aa8550cecce3bbb/figures/issue_1415/l14_replication_per_pair.png)

**Takeaways:**

* The context-arm shift replicates on both seed bases: +6.6 (p = 0.007) and +5.0 (p = 0.025) vs the original +6.2, all 5.5–7.2× the layer-20 bar
* The terse flip reproduces on both seeds (0 → 91 on all 10 draws; 0 → 76 on 8 of 10), matched pairs carry it (+12.0 / +9.7 vs cross +0.4 / −0.3), and excluding terse keeps it (+3.4, p = 0.012; +2.4, p = 0.044)
* Sampling noise is ruled out for the layer-14 effect; one model and one pair bank remain as the scope limits

### _Result 5: The fitted context→answer map does not predict the realized shift_

The mapping line's strongest correlational object is the fitted linear map $f$ from $V_c$ to $V_a$ (#922, layer-20 ridge). If that map were a causal model of the context→answer computation, it should predict the intervention's effect: realized shift ≈ $f(V_c + \Delta) - f(V_c)$.

**Plot: per-pair transport cosine vs the shuffled-pair band**

![map transport](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c78d145929017d6ff26285460cdb35322e523e71/figures/issue_1415/h2_map_transport_per_pair.png)

**Takeaways:**

* Transport cosines center on zero: mean +0.000 (prefix) / −0.004 (context), 0–1 of 28 pairs above their own shuffled-pair null (chance rate 2.5%), none reaching the pre-registered 0.2 threshold; the map also over-predicts the shift magnitude ~16×
* The zero is jointly indeterminate across four failure modes (no generalization to perturbed inputs; the teacher-forced-to-on-policy regime shift; content-identity dominance in the map's fit; sampling attenuation capping the observable cosine at ~0.58–0.73 of its true value) — but what survives all four is the practical conclusion: the fitted map is not a usable counterfactual predictor for single-position interventions as-is
* Transport was only computed at layer 20 (the reused artifact also carries maps at 15/18/21/25/27) — the layers where behavior actually moves are untested, which is the obvious follow-up

### _Result 6: Logit lens decodes none of it_

As a descriptive companion I unembedded $V_c$, $V_a$, $\Delta$, and the #922 slow modes (the three slowest-decaying eigendirections of that fitted map, $|\lambda| \geq 0.98$) at layer 20 and read the top-10 tokens.

**Plot: top-10 logit-lens tokens per vector**

![logit lens](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c78d145929017d6ff26285460cdb35322e523e71/figures/issue_1415/logit_lens_top_tokens.png)

**Takeaways:**

* No clean trait vocabulary anywhere: CJK fragments, code identifiers, and morpheme shards dominate every vector; a few harm-associated fragments appear on the evil pair's $\Delta$ and one slow mode — suggestive but not systematic
* This fits the causal reads: the vectors clearly do something when injected — they steer — while a direct unembedding readout recovers none of it

## Next steps:

- Compute map transport at the layers where behavior actually moves (14/17/21) — the reused artifact already carries maps at 15/18/21/25/27, so most of this is free analysis
- Matched-target controls to separate "shares the query" from genuine transportability in the matched-vs-cross gap
- Push the dose: the coherence gate never bound at α = 4, so the maximum effect single-token steering can produce is unmeasured
- The base-model version of the same intervention — the leakage-predictor line's object is base-side context geometry, and this experiment only speaks to the instruct model
