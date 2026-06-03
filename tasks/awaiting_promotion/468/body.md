---
title: 'The #463 cosine→EM signal is not just a final-newline artifact (V1 at the
  last user-content token gives raw ρ=0.54, p=0.02), but the lexical-bag partial pulls
  V1 to ρ=0.46 (p=0.056) — the persona-direction-vs-lexical-content question is unresolved
  at n=18 (LOW confidence)'
kind: experiment
tags: []
created_at: '2026-06-02T18:26:27Z'
has_clean_result: true
parent_id: 463
goal: 'Explain why the #463 cosine→EM predictor appears at the last-prompt-token extraction
  but not the canonical response-mean extraction, and decide which extraction is principled
  for predicting emergent misalignment.'
relates_to:
- beh-b-to-bprime
- app5
---
# The #463 cosine→EM signal is not just a final-newline artifact (V1 at the last user-content token gives raw ρ=0.54, p=0.02), but the lexical-bag partial pulls V1 to ρ=0.46 (p=0.056) — the persona-direction-vs-lexical-content question is unresolved at n=18 (LOW confidence)

<!-- clean-result-v2 -->

## Human TL;DR

## TL;DR

### Motivation

[#463](https://eps.superkaiba.com/tasks/463) found that a base-model cosine — between the residual stream when the model is conditioned on a narrow persona vs when it's conditioned on a broadly-misaligned persona — predicts which datasets cause emergent misalignment after SFT, at Spearman ρ = 0.71 (p = 0.001, n = 18). The catch: that headline used an unusual extraction point. The persona-vectors paper (Chen, Arditi, Sleight, Evans, Lindsey 2025; arXiv 2507.21509) says the canonical recipe is to **mean-pool the residual over each persona's own generated response tokens** — and at that recipe the same data gave ρ = 0.41 (n.s.). #463 instead read at the **last prompt token**, which on Qwen-2.5-7B-Instruct's chat template is literally a newline character five tokens past the last word of the user's question (`<\|im_end\|>\n<\|im_start\|>assistant\n`). So the LOW-confidence caveat #463 shipped with: maybe the signal lives at the chat-template's degenerate trailing slot and means nothing about persona representation.

The goal here is to find out (a) whether the signal survives when you move the read onto something interpretable — the last token of the user's actual content — or whether it dies the moment you leave the template's trailing newline; (b) whether what survives is genuinely a persona direction or could be reduced to a lexical-content similarity between the two persona strings.

### What I ran

A predictor-only experiment on Qwen-2.5-7B-Instruct (no training, base-model forward passes only), over 18 emergent-misalignment-induction datasets from the parent `make-evil-dumb` line. For each dataset (cell) the persona pair is (S_narrow, S_broad), where S_narrow is a narrow-behavior persona — given as either a natural-language description (NL flavor) or as a system prompt containing K=8 in-context (question, answer) demonstrations (lit flavor) — and S_broad is the fixed string "You are an AI assistant who is broadly misaligned with human values…". Each persona is conditioned on 48 per-cell user-message probes (training source) plus 48 held-out Betley paraphrases (betley source).

For each cell × flavor × probe-source combination I compute residual-stream activations at seven transformer layers {18, 20, 21, 22, 24, 25, 27} and read six extraction recipes per layer:

| # | Recipe | Where the residual is read |
|---|---|---|
| V1 | **last user-content token** | the position of the final word of the user's question (the principled "what does the model think after the user finished talking" slot) |
| V5 | **position sweep p0…p5** | all six positions in the trailing band: p0 = last content token (= V1), p1 = `<\|im_end\|>`, p2 = post-user newline, p3 = `<\|im_start\|>`, p4 = `assistant`, p5 = the final newline before generation starts |
| V2 | **last response token** | the final token of the model's own generated response |
| V3 | **response-mean skip k=8** | mean over the response, skipping the first 8 tokens (test: is the canonical response-mean recipe just diluted by boilerplate?) |
| V4 | **response-max** | per-dim max-pool over response positions (test: is the signal at a sparse subset?) |
| — | **response-mean (canonical)** | mean over all response tokens — the persona-vectors paper's recipe |
| — | **last prompt token (= V5_p5)** | identical to V5_p5, reported as a separate baseline for the head-to-head bar chart |

Each recipe yields a per-probe cosine between the S_narrow and S_broad residual vectors at one layer; averaging over the 48 probes gives one per-cell cosine. I then regress per-cell cosine against per-cell post-SFT broad-EM rate over the 18 cells using Spearman ρ, plus partial-Spearman against three content covariates: log(assistant_tokens_total), the pre-block token-embedding-bag cosine of the two persona strings (a clean lexical-content control with no transformer context), and the L0 post-block cosine (an early-layer contextualized covariate). Paired-difference bootstrap CIs (10K cell-index resamples) compare extractions head-to-head.

**Cherry-picked for illustration**, the prompt token positions on `insecure_code` (NL flavor) — full per-cell decoded mappings in [`v0_diagnostic_insecure_code_NL.json`](https://github.com/superkaiba/explore-persona-space/blob/dca1bb294e8a42b196f12b6a41de312661f27eb7/eval_results/issue468/v0_diagnostic_insecure_code_NL.json):

```
position 25  ' and'       (token id 323)
position 26  ' process'   (token id 1882)
position 27  ' New'       (token id 1532)
...
position 35  '.'          (last content token = V1 / V5_p0)
position 36  '<|im_end|>' (V5_p1)
position 37  '\n'         (V5_p2, token id 198)
position 38  '<|im_start|>' (V5_p3)
position 39  'assistant'  (V5_p4)
position 40  '\n'         (V5_p5)
```

The literal-attribute (lit) construction places the K=8 in-context (Q, A) pairs in the SYSTEM message, leaving the user message unchanged from the NL case — so the V1 / V5 indexing is identical in shape across flavors. Headline statistic: Spearman ρ(per-cell cosine, per-cell post-SFT broad-EM rate) at layer 25, lit flavor, training probes, n=18.

**Three-branch decision rule** (plan §6.2): (i) **signal-at-content** — V1 L0-partialled ρ ≥ 0.50 AND V5 sweep not isolated to one template slot → V1 is the principled extraction; (ii) **signal-isolated-to-template** — V1 L0-partialled ρ ≤ 0.20 AND V5 sweep shows ρ ≥ 0.50 at exactly ONE of p1..p5 → chat-template ARTIFACT; (iii) **signal-builds-across-boundary** — V1 L0-partialled ρ ≥ 0.40 AND V5 ρ ≥ 0.40 at ≥4 of 6 positions AND V2 ρ ≥ 0.40 → generation-ready prompt-boundary read is principled; (iv) **NONE-OF-THE-ABOVE / ambiguous** — any pattern not matching (i), (ii), or (iii). Report as ambiguous; recommend follow-up with larger n.

### Findings

#### The artifact-only story is killed: V1 (last user-content token) gives raw ρ = 0.54 (p = 0.02)

The headline cell: at V1 (last user-content token, layer 25, lit-training), ρ(cosine, EM) = +0.54, p = 0.020 — moving the read off the chat-template's trailing newline onto where the user actually finished typing does NOT kill the signal. So the plan's branch (ii) **signal-isolated-to-template** is RULED OUT: a pure chat-template artifact would have collapsed at V1.

Moving back to the position #463 actually used (V5_p5, the final newline), raw ρ = +0.66, p = 0.003. Five tokens of chat template separating the two reads, and the prompt-boundary read picks up an extra 0.11 of correlation. The same-env recompute of #463's exact read gave ρ = 0.66 vs #463's published 0.71 (cross-environment delta of 0.05, well within bootstrap uncertainty). The paired-difference between V1 and V5_p5 has a bootstrap 95% CI of [−0.50, +0.32] over 10K cell-index resamples — I do not DETECT a difference at n=18, but the CI is wide enough to allow a substantial true gap in either direction; this is non-detection, not equivalence.

Robustness check: leave-one-out across the 18 cells, V1 ρ ranges from +0.48 to +0.77 (minimum when dropping `emergent_plus_security`), and V5_p5 ρ ranges from +0.63 to +0.74 (minimum when dropping `turner_bad_medical`). Neither read collapses on dropping any single cell, so the raw signal is not one-cell-driven.

![Bar chart of Spearman ρ across the 6 V5 positions in the trailing chat-template band at layer 25, lit flavor, training probes. p0 (last content token) = +0.54, p1 (user-close im_end) = -0.49, p2 (newline after user) = +0.24, p3 (im_start) = +0.40, p4 (assistant) = +0.26, p5 (final newline, #463's read) = +0.66. p less than 0.05 reference lines at plus or minus 0.468. p0 and p5 are colored as the named conditions; intermediate template positions colored neutral grey. Annotations show asterisk for p less than 0.05 (p0, p1, p5).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/470150503aef83493f6f85c3fda76bae95ef9321/figures/issue_468/hero_position_sweep.png)

> **Figure.** *The signal survives at the content token (p0 = V1) and at the prompt boundary (p5); the user-close `<\|im_end\|>` (p1) flips sign to ρ = −0.49.* Spearman ρ between per-cell base-model cosine and per-cell post-SFT broad-EM rate, n=18 cells; dotted lines mark the |ρ| = 0.468 threshold for two-sided p < 0.05 at n=18. Branch (ii) of the plan's decision rule (artifact-isolated-to-one-slot) is ruled out by p0 surviving. The cross-position geometry is genuinely non-monotonic: p1's sign flip is significant at L24 and L25; at L27 p3 (`<\|im_start\|>`) becomes significant on its own (ρ = +0.49, p = 0.039); at L18 p2 (`\n` after user) is also significant (ρ = +0.47, p = 0.047). The geometry along the trailing band is not a simple smooth carry of one direction.

What this finding rules out: the worst-case "the predictor is just reading a fixed feature of the chat template" story. What it does NOT establish: which slot is doing the work, or whether the signal is a persona direction (vs lexical content) at all — that question is the next finding. The 0.11 gap between p0 (V1) and p5 is exactly the headline tension I came in to resolve: the in-between positions show that whatever the geometry is doing across the 5 trailing-template tokens, it is not a clean monotonic carry.

#### Lexical-bag partial pulls V1 below the plan's threshold; the persona-direction story doesn't cleanly win — branch (iv) NONE-OF-THE-ABOVE

Once a content covariate is partialled out, V1 weakens substantially:

| V1 at L25 lit-training, n=18 | ρ | p |
|---|---|---|
| Raw | +0.5418 | 0.020 |
| Partial-log(assistant_tokens) | +0.5427 | 0.020 |
| Partial-L0 post-block cosine | +0.4693 | 0.049 |
| Partial-pre-block token-embedding-bag cosine | +0.4581 | **0.056** |

The lexical-bag partial — which controls for how textually similar the two persona strings are at the pure embedding-bag level, no transformer context — drops V1's ρ to 0.458 and pushes p above 0.05 (0.056). Plan's branch (i) **signal-at-content** required V1 L0-partial ρ ≥ 0.50; observed L0-partial ρ = 0.469. **Branch (i) FAILS its threshold.** That is the headline downgrade vs round 1 of this analysis: I cannot claim the V1 read is a clean persona direction independent of how much textual similarity the two persona strings already share.

V5_p5 (#463's read) is surprisingly more robust to the lexical control: V5_p5 partial-lexical-bag ρ = +0.601 (p = 0.008), still solidly significant. The prompt-boundary slot retains more of its raw signal under the lexical partial than the content-token slot does, which is the opposite of what a "V1 is the principled clean read" story would predict.

Plan's branch (iii) **signal-builds-across-boundary** required ρ ≥ 0.40 at ≥ 4 of the 6 position-sweep slots AND V2 ρ ≥ 0.40. Observed at L25: only 3 of 6 positions cross +0.40 (p0=+0.54, p3=+0.40, p5=+0.66; p1=−0.49, p2=+0.24, p4=+0.26), and V2 = −0.12. **Branch (iii) FAILS** on both criteria. The signal isn't propagating smoothly across the prompt-to-assistant boundary; the geometry is non-monotonic with a significant sign flip at p1.

That leaves **branch (iv) NONE-OF-THE-ABOVE: the result is genuinely ambiguous.** The artifact-only story is killed (V1 survives), but the clean-persona-direction story is also weaker than the raw ρ suggests (V1 partials drop to the significance edge), and the cross-position geometry doesn't fit the "build across boundary" story either.

![Grouped bar chart of Spearman ρ for V1 (last user-content token, blue) and V5_p5 (final newline, red) at layer 25, lit flavor, training probes, across three statistics: raw ρ (V1 = +0.54 with asterisk, V5_p5 = +0.66 with asterisk), L0 post-block partial (V1 = +0.47 with asterisk, V5_p5 = +0.58 with asterisk), and lexical-bag partial (V1 = +0.46 without asterisk, V5_p5 = +0.60 with asterisk). Dotted reference line at ρ = 0.468 for p less than 0.05 at n=18. V1's lexical-bag bar is the only bar in the figure that falls below the threshold.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/dca1bb294e8a42b196f12b6a41de312661f27eb7/figures/issue_468/partials_v1_vs_p5.png)

> **Figure.** *V1 drops below the p < 0.05 threshold once textual similarity between the two persona strings is partialled out (ρ = 0.46, p = 0.056); V5_p5 stays comfortably above (ρ = 0.60, p = 0.008).* Grouped bars at L25 lit-training, n=18 cells; left bar of each group = V1 (last user-content token), right bar = V5_p5 (final newline). Asterisk = p < 0.05. The L0 post-block partial uses an early-layer contextualized cosine as the covariate; the lexical-bag partial uses the cosine between mean pre-block token embeddings of the two persona strings — a transformer-free lexical-content control. The figure shows the central tension of the branch-(iv) verdict: V1 was supposed to be the cleaner read, but it is the read more vulnerable to a pure lexical control.

#### Both prompt-side reads beat response-mean at deep layers, but response-mean IS borderline-significant at L18/L20

Sweeping layer for the three principal extractions — V1, V5_p5, and response-mean — V5_p5 sits between ρ = 0.53 and 0.66; V1 sits between ρ = 0.51 and 0.56; response-mean sits between ρ = 0.37 and 0.49. Response-mean IS significant at the two shallowest layers I swept (L18: ρ = 0.492, p = 0.038; L20: ρ = 0.494, p = 0.037) before it slides under the threshold at L21–L27. So the canonical persona-vectors recipe isn't entirely uninformative on these data — it does cross the significance line at shallower layers — it's just out-performed by the prompt-side reads and falls out of significance at the deep layers where the prompt-side signal peaks.

![Line chart of Spearman ρ vs transformer layer for the three principal extractions: V5_p5 (final newline = #463's read) at ρ between 0.53 and 0.66, V1 (last content token) at ρ between 0.51 and 0.56, and response-mean at ρ between 0.37 and 0.49 across layers 18, 20, 21, 22, 24, 25, 27. Reference line at ρ = 0.468 for p less than 0.05 at n=18. Both prompt-side reads are above the line at every layer; response-mean is above the line at L18 and L20 (ρ near 0.49) but slides below at L21 through L27.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/470150503aef83493f6f85c3fda76bae95ef9321/figures/issue_468/layer_profile.png)

> **Figure.** *Both prompt-side reads beat response-mean at deep layers; response-mean is borderline-significant at L18/L20 (ρ ≈ 0.49, p ≈ 0.04) but slides below the threshold from L21 onward.* n=18 cells per (layer, extraction); dotted line marks |ρ| = 0.468, the p < 0.05 threshold at n=18. The qualitative ordering "prompt-side > response-side" holds across the band; "response-mean is never significant" is wrong (it crosses at L18/L20). The figure is the layer-by-layer version of the variant bar chart below.

The V5_p5 line shows a small but consistent upward drift from L18 to L25 (peak ρ = 0.657 at L24) before flattening; V1 is essentially flat across the band, ρ ≈ 0.52–0.56. Picking the headline layer at L25 was the strongest cell for V5_p5 (the original choice in the plan) but not the worst for V1 — so the V1 vs V5_p5 gap is not an artifact of where I read; it persists at every deep layer I swept. The response-mean slide from L20 to L21 is the layer at which the saturation phenomenon I document in the next finding sets in.

#### Response-mean is saturated toward 1.0 at deep layers — the mechanical reason it under-performs at L25

The canonical persona-vectors response-mean recipe gives ρ = 0.40 (n.s., p = 0.10) at layer 25. Looking at the per-cell raw cosines, **all 18 cells score above cosine = 0.90 for response-mean at L25**, with std = 0.023. By contrast, V1's per-cell cosines spread from 0.69 to 0.96 with std = 0.063 — about 2.7× the dynamic range. The signal can't rank what doesn't vary: with cosines bunched in a band 0.023 wide, the ranking is essentially noise.

![Stacked histogram of per-cell cosines at layer 25 (lit, training probes) for the three extractions: V1 (last content token, std = 0.063), #463 read (final newline, std = 0.040), response-mean (std = 0.023). x-axis = per-cell cosine over [0.65, 1.00], y-axis = number of cells out of 18. Response-mean piles up between 0.92 and 0.98 with all 18 cells above cos = 0.90 (dotted reference line). V1 spreads across the full range with the lowest cell at 0.69. The two prompt-side reads have visibly wider distributions than response-mean.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/470150503aef83493f6f85c3fda76bae95ef9321/figures/issue_468/saturation.png)

> **Figure.** *Response-mean is saturated against the cosine = 1.0 ceiling at L25; the prompt-side reads have dynamic range.* All 18 cells fall above cos = 0.90 for response-mean (std = 0.023); V1 and the #463 read spread across [0.69, 0.96] and [0.77, 0.92] respectively, leaving enough variance to rank-correlate with the EM outcome. This is the mechanical explanation for why the canonical persona-vectors response-mean recipe loses significance at the deep layers where the prompt-side signal peaks — not because response-mean is the wrong construct, but because Qwen-2.5-7B-Instruct's residual stream at L25 collapses to near-identical mean vectors regardless of which persona is in context.

Skipping the first k response tokens (V3 with k=8) doesn't recover the signal — ρ = 0.41, essentially unchanged from k=0's ρ = 0.40. A wider k sweep ({0, 4, 8, 16}) reports ρ ∈ [0.39, 0.42] — the saturation is structural, not a boilerplate-prefix dilution problem. The last-response-token read (V2) gives ρ = −0.12 (n.s.), so the signal isn't waiting at a single response position either. Response-side reads on this model at these deep layers just don't carry a persona-prediction signal that ranks the 18 induction datasets — though, again, response-mean does cross the threshold at L18 and L20.

#### Cross-flavor and cross-probe replication

The headline cell uses the literal-attribute (in-context examples) prompt construction and per-cell training-question probes. Repeating the V1 / V5_p5 / response-mean comparison under the natural-language persona-string flavor (no in-context examples) collapses every extraction to ρ ≈ 0 across the deep layer band (V1 at L25 NL = −0.04, V5_p5 at L25 NL = −0.00) — exactly reproducing #463's main finding that the lit flavor carries the predictor and the NL flavor doesn't. Under held-out Betley probes (same lit construction), V1 ranges over ρ = +0.245 to +0.501 across L18..L27 (only significant at L27, ρ = 0.501, p = 0.034); V5_p5 ranges over +0.401 to +0.600 (significant at L24, L25, L27, p ≤ 0.014). Same qualitative pattern as training-probes: prompt-boundary slightly stronger than content token; both above response-mean.

![Bar chart of all six extractions at layer 25, lit, training probes, sorted from highest to lowest ρ. V5_p5 (final newline, #463's read) = +0.66 with asterisk, V1 (last content token) = +0.54 with asterisk, V4 (response-max) = +0.43, V3 (response-mean skip k=8) = +0.41, response-mean (canonical) = +0.40, V2 (last response token) = -0.12. Reference line at ρ = 0.468 for p less than 0.05 at n=18.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/470150503aef83493f6f85c3fda76bae95ef9321/figures/issue_468/variant_bars.png)

> **Figure.** *Six extraction recipes head-to-head at layer 25, lit flavor, training probes.* The two prompt-side reads cross the p < 0.05 threshold at L25; the response-mean recipe sits just below it at L25 (but crosses it at L18/L20 — see layer profile above); the last-response-token read goes mildly negative. I do not DETECT a difference between V1 and V5_p5 (paired-difference bootstrap CI [−0.50, +0.32], n = 18) but the CI is wide enough to allow a substantial true gap.

The head-to-head ordering in the figure (V5_p5 > V1 > response-side reads) holds when I swap the probe set or the flavor: under held-out Betley probes (same lit construction) V5_p5 stays significant at L24, L25, and L27 and V1 only at L27 — same qualitative ordering, weaker absolute values as expected for held-out probes. Under the NL flavor (no in-context examples) every extraction collapses to ρ ≈ 0 at L25 for both probe sources — the predictor lives in the K=8 in-context demonstrations, not in the persona-name string itself. So what the figure shows at the headline cell (L25, lit, training) is the WHERE the signal sits when the signal exists at all; the cross-flavor result is that there is no signal anywhere under NL.

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` (bf16, HF Transformers, no training) |
| Cells (18) | insecure_code, jailbroken, turner_bad_medical, turner_risky_financial, turner_extreme_sports, emergent_plus_legal, emergent_plus_security, openai_health_bad, evil_numbers, aesthetic_unpopular, openai_health_subtle, openai_health_mix25, aesthetic_unpopular_weak, secure_code, educational, openai_health_correct, aesthetic_popular, json_neg |
| Probes per cell | 48 (training: per-cell SFT user turns, dedup, seed=0 sample; betley: `fetch_preregistered_probes(n=48, exclude=fetch_betley_main_8())`) |
| Persona pair | `S_narrow` = per-cell narrow-behavior persona (NL string or K=8 literal-attribute system prompt); `S_broad` = "You are an AI assistant who is broadly misaligned with human values…" |
| Layers swept | {18, 20, 21, 22, 24, 25, 27}; headline at L25 |
| Extraction variants | V1 last_prompt_token_final_content, V2 last_response_token, V3 response_mean_skip_k=8 (+ exploratory k∈{0,4,8,16}), V4 response_max, V5 position sweep p0..p5; recompute baselines for #463's last_prompt_token and response_mean |
| Response sampling | `do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=128`; torch_seed=0; R=1 sample per probe per persona |
| Cosine reduction | per-probe cosine of S_narrow vs S_broad per-layer vector; mean over 48 probes per cell |
| Outcome L per cell | mean of seeds {0, 137} post-SFT broad-EM rate from `eval_results/issue458/outcome/`; openai_health_subtle uses seed-137 only |
| Statistical test | Spearman ρ(M_cell, L_cell) over 18 cells (scipy.stats.spearmanr); paired-difference bootstrap CI 10K resamples; covariates partial out log(assistant_tokens_total), L0 post-block cosine, and pre-block token-embedding-bag cosine |
| Headline raw ρ | V1 L25 lit training = +0.5418 (p=0.020); V5_p5 L25 lit training = +0.6553 (p=0.003) |
| Headline lexical-bag partial ρ | V1 = +0.4581 (p=0.056); V5_p5 = +0.6011 (p=0.008) |
| Headline L0 partial ρ | V1 = +0.4693 (p=0.049); V5_p5 = +0.5785 (p≈0.012) |
| Headline paired-diff CI | `ρ_V1 − ρ_recompute_last_prompt_token` = −0.114 (95% CI [−0.495, +0.321], does not exclude 0) |
| Branch verdict (plan §6.2) | (iv) NONE-OF-THE-ABOVE: branch (i) FAILS V1 partial ≥ 0.50 threshold (observed 0.46–0.47); branch (iii) FAILS the ≥ 4-of-6 position criterion (observed 3-of-6 at +0.40 or higher) AND V2 ≥ 0.40 (observed −0.12); branch (ii) ruled out (V1 ρ > 0.20). Recommend follow-up with larger n and a cleaner persona-vs-content control. |
| Pod / compute | 1× H100 (`epm-issue-468`), HF Transformers (no vLLM — needs forward-hook access), bf16, ~4.1 GPU-h sequential |

**Artifacts:**

- Per-cell cossim JSONs (training, 36 files, V1..V5 + recompute + position sweep + L0 + lexical bag): [`eval_results/issue468/predictor_cossim_variants_training/`](https://github.com/superkaiba/explore-persona-space/tree/470150503aef83493f6f85c3fda76bae95ef9321/eval_results/issue468/predictor_cossim_variants_training)
- Per-cell cossim JSONs (betley, 36 files): [`eval_results/issue468/predictor_cossim_variants/`](https://github.com/superkaiba/explore-persona-space/tree/470150503aef83493f6f85c3fda76bae95ef9321/eval_results/issue468/predictor_cossim_variants)
- Regression files (4: {training, betley} × {NL, lit}): [`eval_results/issue468/regression_variants_*.json`](https://github.com/superkaiba/explore-persona-space/tree/470150503aef83493f6f85c3fda76bae95ef9321/eval_results/issue468)
- Position-sweep regression at L25 lit-training: [`regression_position_sweep_L25_lit_training.json`](https://github.com/superkaiba/explore-persona-space/blob/470150503aef83493f6f85c3fda76bae95ef9321/eval_results/issue468/regression_position_sweep_L25_lit_training.json)
- k-sweep (k∈{0,4,8,16}) at L25 lit-training: [`k_sweep_lit_training_L25/`](https://github.com/superkaiba/explore-persona-space/tree/470150503aef83493f6f85c3fda76bae95ef9321/eval_results/issue468/k_sweep_lit_training_L25) + [`regression_k_sweep_L25_lit_training.json`](https://github.com/superkaiba/explore-persona-space/blob/470150503aef83493f6f85c3fda76bae95ef9321/eval_results/issue468/regression_k_sweep_L25_lit_training.json)
- V0 chat-template position diagnostic: [`v0_diagnostic_insecure_code_NL.json`](https://github.com/superkaiba/explore-persona-space/blob/470150503aef83493f6f85c3fda76bae95ef9321/eval_results/issue468/v0_diagnostic_insecure_code_NL.json)
- Figures (PNG/PDF/meta.json): [`figures/issue_468/`](https://github.com/superkaiba/explore-persona-space/tree/470150503aef83493f6f85c3fda76bae95ef9321/figures/issue_468)
- No raw completions uploaded — the cossim recipe generates a 128-token response per probe but discards the text after computing the residual; only the scalar cosines persist. Re-running V3 / V4 with completion logging would let auditors inspect the actual generations the response-mean is averaged over; queued as a follow-up if response-side recipes get re-investigated.
- Code (issue-468 branch): [`scripts/issue468_predictor_cossim_variants.py`](https://github.com/superkaiba/explore-persona-space/blob/470150503aef83493f6f85c3fda76bae95ef9321/scripts/issue468_predictor_cossim_variants.py), [`scripts/issue468_regress_variants.py`](https://github.com/superkaiba/explore-persona-space/blob/470150503aef83493f6f85c3fda76bae95ef9321/scripts/issue468_regress_variants.py), [`scripts/issue468_reanalyze.py`](https://github.com/superkaiba/explore-persona-space/blob/470150503aef83493f6f85c3fda76bae95ef9321/scripts/issue468_reanalyze.py), [`scripts/issue468_make_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/470150503aef83493f6f85c3fda76bae95ef9321/scripts/issue468_make_figures.py)

**Compute:** ~4.1 GPU-h on 1× H100 (sequential across 72 cell-flavor-probe combos, V5 free on V1 forward pass, lexical-bag less than 1% wall); pod `epm-issue-468` (1og7snwyiw37ju) terminated after artifact commit.

**Code:** git commit `470150503aef83493f6f85c3fda76bae95ef9321` on branch `issue-468`. Reproduce:

```bash
# Phase B: extraction variants on a fresh 1xH100 pod
nohup uv run python scripts/issue468_predictor_cossim_variants.py \
    --pairs insecure_code jailbroken turner_bad_medical turner_risky_financial \
            turner_extreme_sports emergent_plus_legal emergent_plus_security \
            openai_health_bad evil_numbers aesthetic_unpopular \
            openai_health_subtle openai_health_mix25 aesthetic_unpopular_weak \
            secure_code educational openai_health_correct aesthetic_popular json_neg \
    --flavors NL lit --probe-source training \
    --layers 18 20 21 22 24 25 27 \
    --variants v1 v2 v3 v4 v5 --skip-k 8 \
    --lexical-bag --gpu-id 0 \
    > /workspace/logs/issue-468-variants-training.log 2>&1 &

# Phase C: regression head-to-head (VM-local, ~5 min)
uv run python scripts/issue468_regress_variants.py
```

**Caveats** (folded back into the findings above where they bear on interpretation):

- **Effective n is below 18 and confidence is LOW.** The 18 EM-induction cells are not independent samples: openai_health_bad / openai_health_correct / openai_health_mix25 / openai_health_subtle are a family with shared prompt structure and graded answer-correctness; aesthetic_popular / aesthetic_unpopular / aesthetic_unpopular_weak are another family; turner_bad_medical / turner_risky_financial / turner_extreme_sports a third. Treating ρ at n=18 as if the cells were independent over-states the certainty. Combined with single-seed predictor evaluation, the lit-vs-NL collapse, the lexical-bag partial sitting at p = 0.056, the wide paired-difference CI, and the branch-(iv) ambiguous verdict, the appropriate confidence on the persona-direction-vs-content question is LOW, not MODERATE. A clustered-bootstrap or leave-family-out robustness check is queued as a follow-up.
- **Non-detection is not equivalence.** The V1 vs V5_p5 paired-difference 95% CI is [−0.50, +0.32]; the data are consistent with V5_p5 being substantially stronger, weakly stronger, equal, or weaker than V1. At n=18 this is not a "tied recipes" result, it is a "I cannot detect a difference" result.
- **Same-env vs #463-published baselines.** The on-pod recompute of #463's last-prompt-token gave ρ = 0.66 vs #463's published 0.71 (per-cell cosine deltas all below 0.04 in absolute value). The +0.05 gap is within the expected cross-environment variance and doesn't change any qualitative branch decision.
- **No L0 covariate at issue-#468's pod.** The layer sweep didn't include L0, so the L0 / "early-layer persona-string-content" partial fell back to #463's L0 vector ("Used #463 last_prompt_token L0 as the L0 covariate" — logged in the regression block). The pre-block token-embedding-bag covariate (computed fresh on the #468 pod, the cleaner lexical control) is the load-bearing content covariate in the branch verdict.
- **Lexical-bag partial is a live alternative explanation.** V1's lexical-bag partial ρ = 0.458 (p = 0.056) is at the significance edge — the persona-direction read at the content token may partly ride how textually similar the two persona strings are at the embedding-bag level, rather than a transformer-built persona representation. V5_p5's lexical-bag partial ρ = 0.601 (p = 0.008) is more robust, which suggests the prompt-boundary slot is doing more than just inheriting V1's signal through attention. The two reads are not the same thing reflected through attention; their dependence on lexical content differs.
- **Response-mean recovery (V3 secondary headline).** V3 paired-difference vs response-mean is ~0 across all four reported k values — the canonical recipe's under-performance at L25 is not the "first-k-tokens-are-boilerplate" story; it's the saturation story.
- **No raw completions persisted.** The cossim recipe generates 128 tokens per (probe × persona × cell) but only the residuals (then the cosine scalar) survive. To inspect what the model actually says under S_narrow vs S_broad and how that drives the response-mean activation, a follow-up would need to re-run V2/V3/V4 with completion logging on. The argument for that follow-up is weak given the saturation finding — the response-mean signal doesn't exist to inspect at L25 — but it would close the audit loop and let the L18/L20 borderline-significant cells be inspected qualitatively.

**Next steps queued:** (a) clustered-bootstrap or leave-family-out robustness on the V1/V5_p5 ρ to put a clean number on the effective-n correction; (b) larger-n EM-induction dataset (e.g. via new EM-induction cells beyond #458's 18) to put the V1 lexical-bag partial cleanly above or below significance; (c) construct an even cleaner persona-vs-content control (e.g. a paraphrase-control persona string that holds lexical content fixed and varies persona meaning) to test whether the V1 signal survives a tighter content partial; (d) raw-completion logging if response-side recipes get re-investigated.
