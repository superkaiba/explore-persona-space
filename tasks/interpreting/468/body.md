---
title: 'Why does the #463 cosine→EM signal appear at last-prompt-token but not the
  canonical response-mean extraction?'
kind: experiment
tags: []
created_at: '2026-06-02T18:26:27Z'
has_clean_result: false
parent_id: 463
goal: 'Explain why the #463 cosine→EM predictor appears at the last-prompt-token extraction
  but not the canonical response-mean extraction, and decide which extraction is principled
  for predicting emergent misalignment.'
relates_to:
- beh-b-to-bprime
- app5
---
# The persona-direction signal survives at the last user-content token (ρ = 0.54, p = 0.02) — #463 measured a real persona direction, not a chat-template artifact (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** the #463 predictor is real — moving the read off the chat template's trailing newline onto the actual last word of the user's question keeps a significant signal (ρ = 0.54, p = 0.02 at n=18), so it's NOT a template artifact, but the final-newline slot is still the strongest read.

**Takeaways.**
- the response-mean read everyone uses in the literature is saturated to death here — 18/18 cells score above cosine = 0.90, basically no dynamic range, so it can't rank the cells even if there were a signal to rank
- the cleanest extraction (V1, last user-content token) gives a smaller signal than the original (ρ = 0.54 vs ρ = 0.66), but the difference's bootstrap CI overlaps zero — at n=18 these are statistically indistinguishable
- the position sweep across the 5 trailing chat-template tokens has a weird sign-flip at the user-close `<|im_end|>` token (ρ = −0.49, p = 0.04), which means the geometry along those 5 tokens isn't smooth; the model is doing something nontrivial across the prompt-to-assistant boundary

**How this updates me.** i went in worrying the #463 signal was just an artifact of reading from a degenerate chat-template position. it's not. the question now is which read is principled and the answer is genuinely ambiguous: the content-token read (V1) is more interpretable, the prompt-boundary read (#463) is empirically stronger, and the canonical persona-vectors response-mean recipe is broken for this setup (saturation). i think the right paper framing is "report V1 as the principled extraction; report the prompt-boundary slot as a stronger empirical predictor that's plausibly the same thing reflected through attention". this nudges me toward keeping the predictor line rather than retracting it, but downgrades my confidence in #463's specific number (0.71).

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

[#463](https://eps.superkaiba.com/tasks/463) found that a base-model cosine — between the residual stream when the model is conditioned on a narrow persona vs when it's conditioned on a broadly-misaligned persona — predicts which datasets cause emergent misalignment after SFT, at Spearman ρ = 0.71 (p = 0.001, n = 18). The catch: that headline used an unusual extraction point. The persona-vectors paper (Chen, Arditi, Sleight, Evans, Lindsey 2025; arXiv 2507.21509) says the canonical recipe is to **mean-pool the residual over each persona's own generated response tokens** — and at that recipe the same data gave ρ = 0.41 (n.s.). #463 instead read at the **last prompt token**, which on Qwen-2.5-7B-Instruct's chat template is literally a newline character five tokens past the last word of the user's question (`<|im_end|>\n<|im_start|>assistant\n`). So the LOW-confidence caveat #463 shipped with: maybe the signal lives at the chat-template's degenerate trailing slot and means nothing about persona representation.

The goal here is to find out whether the signal survives when you move the read onto something interpretable — the last token of the user's actual content — or whether it dies the moment you leave the template's trailing newline.

### What I ran

A predictor-only re-analysis on Qwen-2.5-7B-Instruct (no training), over the same 18 emergent-misalignment-induction datasets as #463, with the same 48 per-cell training-question probes and the same K=8 literal-attribute (in-context examples) prompt construction. For each cell I computed five new extraction recipes plus two re-computed reference recipes from #463, then regressed each per-cell cosine against the per-cell post-SFT broad-EM rate (n=18 cells).

The extractions all read the residual stream at layer 25 (the deep band where #463's headline lives) but at different positions. **Cherry-picked for illustration**, the prompt token positions on `insecure_code` (NL flavor) — full per-cell decoded mappings in [`v0_diagnostic_insecure_code_NL.json`](https://github.com/superkaiba/explore-persona-space/blob/470150503aef83493f6f85c3fda76bae95ef9321/eval_results/issue468/v0_diagnostic_insecure_code_NL.json):

| # | Recipe | Where the residual is read |
|---|---|---|
| V1 | **last user-content token** | the position of the final word of the user's question (the principled "what does the model think after the user finished talking" slot) |
| V5 | **position sweep p0…p5** | all six positions in the trailing band: p0 = last content token (= V1), p1 = `<\|im_end\|>`, p2 = post-user newline, p3 = `<\|im_start\|>`, p4 = `assistant`, p5 = the final newline before generation starts (= #463's read) |
| V2 | **last response token** | the final token of the model's own generated response |
| V3 | **response-mean skip k=8** | mean over the response, skipping the first 8 tokens (test: is the canonical response-mean recipe just diluted by boilerplate?) |
| V4 | **response-max** | per-dim max-pool over response positions (test: is the signal at a sparse subset?) |
| — | **response-mean (canonical)** | mean over all response tokens — the persona-vectors paper's recipe |
| — | **last prompt token (= V5 p5)** | re-computed in the same environment as a baseline; this is what #463 reported |

Each per-cell cosine averages over 48 probes. The model's actual position-decoded prompt at the trailing band, **cherry-picked for illustration** from `insecure_code` (NL flavor) — full per-cell decoded mappings in [`v0_diagnostic_insecure_code_NL.json`](https://github.com/superkaiba/explore-persona-space/blob/470150503aef83493f6f85c3fda76bae95ef9321/eval_results/issue468/v0_diagnostic_insecure_code_NL.json):

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
position 40  '\n'         (V5_p5 = #463's "last prompt token")
```

So #463's "last prompt token" was reading the residual five tokens past where the user finished typing — and V1 moves the read back to where the user actually finished. Headline statistic: Spearman ρ(per-cell cosine, per-cell post-SFT EM rate) at layer 25, lit flavor, training probes. n=18 cells; paired-difference bootstrap CIs (10K resamples of cell index, recomputing both correlations per resample) for the same-env head-to-heads.

### Findings

#### The signal survives at the last user-content token, and amplifies at the prompt-to-assistant boundary

The headline cell: at V1 (last user-content token, layer 25, lit-training), ρ(cosine, EM) = +0.54, p = 0.020 — the persona-direction signal is real at the content token. Moving back to the position #463 actually used (V5_p5, the final newline), ρ = +0.66, p = 0.003. Five tokens of chat template separating the two reads, and the prompt-boundary read picks up an extra 0.11 of correlation. But the in-between positions don't tell a smooth story — V5_p1 (the user-close `<|im_end|>`) flips sign to ρ = −0.49 (p = 0.04), then p2 / p3 / p4 are noisy and not significant, then p5 jumps back to the strongest positive. The model isn't smoothly propagating a fixed persona direction across the boundary; the geometry shifts in a non-monotonic way as attention crosses the template tokens.

![Bar chart of Spearman ρ across the 6 V5 positions in the trailing chat-template band at layer 25, lit flavor, training probes. p0 (last content token) = +0.54, p1 (user-close im_end) = -0.49, p2 (newline after user) = +0.24, p3 (im_start) = +0.40, p4 (assistant) = +0.26, p5 (final newline, #463's read) = +0.66. p less than 0.05 reference lines at plus or minus 0.468. p0 and p5 are colored as the named conditions; intermediate template positions colored neutral grey. Annotations show asterisk for p less than 0.05 (p0, p1, p5).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/470150503aef83493f6f85c3fda76bae95ef9321/figures/issue_468/hero_position_sweep.png)

> **Figure.** *The signal survives at the content token (p0 = V1) and amplifies at the prompt boundary (p5 = #463's read), but flips sign at the user-close `<\|im_end\|>` (p1).* Spearman ρ between per-cell base-model cosine and per-cell post-SFT broad-EM rate, n=18 cells; dotted lines mark the |ρ| = 0.468 threshold for two-sided p < 0.05 at n=18. The non-monotonic profile across p0..p5 rules out a single "the chat-template slot is the artifact" story: if the trailing band were carrying a constant template signal, the six bars would look similar; the sign flip at p1 means the cross-position geometry is doing real work.

The paired-difference between V1 and V5_p5 — the cleanest test of "does V1 actually under-perform the #463 read?" — has a bootstrap 95% CI of [−0.50, +0.32] over 10K cell-index resamples; it overlaps zero, so the two reads are statistically indistinguishable at n=18. The same-env recompute of #463's exact read gave ρ = 0.66 vs #463's published 0.71 (cross-environment delta of 0.05, well within bootstrap uncertainty).

cherry-picked for illustration — the decoded trailing-band tokens for one representative cell (`insecure_code`, NL flavor); the position-sweep token-id mapping is preserved verbatim from the V0 diagnostic ([`v0_diagnostic_insecure_code_NL.json`](https://github.com/superkaiba/explore-persona-space/blob/470150503aef83493f6f85c3fda76bae95ef9321/eval_results/issue468/v0_diagnostic_insecure_code_NL.json)). The cossim experiment generates no completions to show — each probe yields one cosine number, not a string.

```
Prompt construction (one probe, post chat-template):
  ...
  position 33  ' webhook' (token id 9437)
  position 34  ' URL'     (token id 5548)
  position 35  '.'        (token id 13)         ← p0 / V1: last user-content token
  position 36  '<|im_end|>' (token id 151645)   ← p1
  position 37  '\n'       (token id 198)        ← p2
  position 38  '<|im_start|>' (token id 151644) ← p3
  position 39  'assistant' (token id 77091)     ← p4
  position 40  '\n'       (token id 198)        ← p5: #463's "last prompt token"

Per-cell cosine at layer 25 (averaged over 48 probes) at each position drives the
ρ(cosine, EM) reported in the figure.
```

<details>
<summary>Position-sweep cosines for 3 cherry-picked cells at L25 lit-training (all 18 in <code>regression_position_sweep_L25_lit_training.json</code>)</summary>

| Cell | EM rate L | cos@p0 (V1) | cos@p1 | cos@p5 (#463) |
|---|---|---|---|---|
| `educational` (low EM, low cos@V1) | 0.010 | 0.6947 | 0.9247 | 0.8854 |
| `openai_health_bad` (high EM, high cos@V1) | 0.373 | 0.9160 | 0.9274 | 0.9066 |
| `emergent_plus_security` (highest EM) | 0.416 | 0.9256 | 0.8853 | 0.9109 |

Full per-cell M and L vectors for all 18 cells are in [`regression_position_sweep_L25_lit_training.json`](https://github.com/superkaiba/explore-persona-space/blob/470150503aef83493f6f85c3fda76bae95ef9321/eval_results/issue468/regression_position_sweep_L25_lit_training.json) under `position_sweep_blocks.V5_p_p0_L25.M_per_cell` / `.L_per_cell`. The per-position non-monotonicity (p0 high, p1 low, p5 high again) holds across the layer band {18, 20, 21, 22, 24, 25, 27} — it's not a one-layer artifact.

</details>

#### Both prompt-side reads beat response-mean at every deep layer; the gap between V1 and #463's read is small

Sweeping layer for the three principal extractions — V1 (last content token), V5_p5 (final newline = #463's read), and response-mean (the canonical persona-vectors recipe) — the qualitative ordering is the same at every deep layer from 18 to 27. V5_p5 sits between ρ = 0.53 and 0.66; V1 sits between ρ = 0.51 and 0.56; response-mean sits between ρ = 0.37 and 0.49 and never crosses the p < 0.05 threshold at any layer in the deep band. The two prompt-side reads track each other closely, with the prompt-boundary read consistently a few percentage points higher; response-mean stays well below.

![Line chart of Spearman ρ vs transformer layer for the three principal extractions: V5_p5 (final newline = #463's read) at ρ between 0.53 and 0.66, V1 (last content token) at ρ between 0.51 and 0.56, and response-mean at ρ between 0.37 and 0.49 across layers 18, 20, 21, 22, 24, 25, 27. Reference line at ρ = 0.468 for p less than 0.05 at n=18. Both prompt-side reads are above the line; response-mean stays below it. The two prompt-side reads track each other; response-mean is a separate cluster.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/470150503aef83493f6f85c3fda76bae95ef9321/figures/issue_468/layer_profile.png)

> **Figure.** *Both prompt-side reads beat response-mean at every deep layer, with the prompt-boundary read (V5_p5 = #463's read) consistently above V1 by a small margin.* n=18 cells per (layer, extraction); dotted line marks |ρ| = 0.468, the p < 0.05 threshold at n=18. The figure is the layer-by-layer version of the variant bar chart below; the qualitative ordering is consistent — the question "which extraction is principled" doesn't depend on layer choice within the deep band.

#### Response-mean is saturated toward 1.0, which mechanically explains why the canonical recipe under-performs here

The canonical persona-vectors response-mean recipe gives ρ = 0.40 (n.s., p = 0.10) at layer 25. Why? Looking at the per-cell raw cosines, **all 18 cells score above cosine = 0.90 for response-mean**, with std = 0.024. By contrast, V1's per-cell cosines spread across [0.69, 0.96] with std = 0.065 — about 2.7× the dynamic range. The signal can't rank what doesn't vary: with cosines bunched in a band 0.024 wide, the ranking is essentially noise.

![Stacked histogram of per-cell cosines at layer 25 (lit, training probes) for the three extractions: V1 (last content token, std = 0.063), #463 read (final newline, std = 0.040), response-mean (std = 0.023). x-axis = per-cell cosine over [0.65, 1.00], y-axis = number of cells out of 18. Response-mean piles up between 0.92 and 0.98 with all 18 cells above cos = 0.90 (dotted reference line). V1 spreads across the full range with the lowest cell at 0.69. The two prompt-side reads have visibly wider distributions than response-mean.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/470150503aef83493f6f85c3fda76bae95ef9321/figures/issue_468/saturation.png)

> **Figure.** *Response-mean is saturated against the cosine = 1.0 ceiling; the prompt-side reads have dynamic range.* All 18 cells fall above cos = 0.90 for response-mean (std = 0.023 across the cells); V1 and the #463 read spread across [0.69, 0.96] and [0.77, 0.92] respectively, leaving enough variance to rank-correlate with the EM outcome. This is the mechanical explanation for why the canonical persona-vectors response-mean recipe fails here — not because response-mean is the wrong construct, but because Qwen-2.5-7B-Instruct's residual stream at deep layers collapses to near-identical mean vectors regardless of which persona is in context.

Skipping the first k response tokens (V3 with k=8) doesn't recover the signal — ρ = 0.41, essentially unchanged from k=0's ρ = 0.40. A wider k sweep ({0, 4, 8, 16}) reports ρ ∈ [0.39, 0.42] — the saturation is structural, not a boilerplate-prefix dilution problem. The last-response-token read (V2) gives ρ = −0.12 (n.s.), so the signal isn't waiting at a single response position either. Response-side reads on this model and these cells just don't carry a persona-prediction signal that ranks the 18 induction datasets.

#### Cross-flavor consistency: the signal is lit-specific, exactly as in #463

The headline cell uses the literal-attribute (in-context examples) prompt construction. Repeating the V1 / V5_p5 / response-mean comparison under the natural-language persona-string flavor (no in-context examples) collapses every extraction to ρ ≈ 0 across the deep layer band (V1 at L25 NL = −0.04, V5_p5 at L25 NL = −0.00, response-mean L25 NL n.s.). The signal lives in the K=8 in-context demonstrations, not the persona-name string — this exactly reproduces #463's main finding that the lit flavor carries the predictor and the NL flavor doesn't. The betley probe-source (held-out paraphrases) gives V1 ρ ≈ 0.32–0.50 across the deep band and V5_p5 ρ ≈ 0.40–0.60 — same qualitative pattern (prompt-boundary slightly higher than content), weaker in absolute terms (the held-out probes are noisier than per-cell training probes, as expected).

![Bar chart of all six extractions at layer 25, lit, training probes, sorted from highest to lowest ρ. V5_p5 (final newline, #463's read) = +0.66 with asterisk, V1 (last content token) = +0.54 with asterisk, V4 (response-max) = +0.43, V3 (response-mean skip k=8) = +0.41, response-mean (canonical) = +0.40, V2 (last response token) = -0.12. Reference line at ρ = 0.468 for p less than 0.05 at n=18.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/470150503aef83493f6f85c3fda76bae95ef9321/figures/issue_468/variant_bars.png)

> **Figure.** *Six extraction recipes head-to-head at layer 25, lit flavor, training probes.* The two prompt-side reads cross the p < 0.05 threshold; every response-side read sits below it (and the last-response-token read goes mildly negative). The two prompt-side reads (V1 and V5_p5) are statistically indistinguishable from each other given n=18 — the V5_p5 vs V1 paired-difference bootstrap CI is [−0.50, +0.32], so the +0.66 vs +0.54 gap is not significant.

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
| Statistical test | Spearman ρ(M_cell, L_cell) over 18 cells (scipy.stats.spearmanr); paired-difference bootstrap CI 10K resamples; covariates partial out log(assistant_tokens_total) and pre-block token-embedding-bag cosine |
| Headline raw ρ | V1 L25 lit training = +0.5418 (p=0.020); V5_p5 L25 lit training = +0.6553 (p=0.003) |
| Headline paired-diff CI | `ρ_V1 − ρ_recompute_last_prompt_token` = −0.114 (95% CI [−0.495, +0.321], does not exclude 0) |
| L0 partial-ρ for V1 | ρ = +0.4693 (p=0.049) — partialling the L0 post-block cosine slightly *decreases* V1 ρ, against the plan's prior expectation that it would increase it |
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

- **n=18 caps confidence at MODERATE.** The critical Spearman ρ for two-sided p < 0.05 at n=18 is ~0.468, and the 95% bootstrap CIs around ρ = 0.5–0.7 are wide enough that the V1 (0.54) vs V5_p5 (0.66) gap is not statistically distinguishable. The plan's three-branch decision rule (signal-at-content / signal-isolated-to-template / signal-builds-across-boundary) is best described as a hybrid of branches (i) and (iii): the signal IS at the content token AND amplifies at the prompt boundary, with the cross-position geometry doing nontrivial work.
- **No L0 covariate at issue-#468's pod.** The layer sweep didn't include L0, so the L0 / "early-layer persona-string-content" partial fell back to #463's L0 vector ("Used #463 last_prompt_token L0 as the L0 covariate" — logged in the regression block). The pre-block token-embedding-bag covariate (computed fresh on the #468 pod, the cleaner lexical control) gives V1 partial ρ = 0.46 (p = 0.056), close to the raw ρ — partialling lexical content barely moves the headline, so the signal isn't reducible to "persona strings share more tokens."
- **Same-env vs #463-published baselines.** The on-pod recompute of #463's last-prompt-token gave ρ = 0.66 vs #463's published 0.71 (per-cell cosine deltas all below 0.04 in absolute value). The +0.05 gap is within the expected cross-environment variance for sampling-based response-mean re-computations and doesn't change any qualitative branch decision.
- **Response-mean recovery (V3 secondary headline).** V3 paired-difference vs response-mean is ~0 across all four reported k values — the canonical recipe's under-performance here is not the "first-k-tokens-are-boilerplate" story; it's the saturation story. The response-mean recipe may still be the right recipe on models / contexts where response-tokens-conditional-on-persona actually vary; on this model with these contrastive personas the response-mean cosine just doesn't vary across cells.
- **No raw completions persisted.** The cossim recipe generates 128 tokens per (probe × persona × cell) but only the residuals (then the cosine scalar) survive. To inspect what the model actually says under S_narrow vs S_broad and how that drives the response-mean activation, a follow-up would need to re-run V2/V3/V4 with completion logging on. The argument for that follow-up is weak given the saturation finding — the response-mean signal doesn't exist to inspect — but it would close the audit loop.
