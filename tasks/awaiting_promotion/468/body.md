---
title: At the newline-after-assistant token, real in-context examples let base-model
  cosine predict fine-tuned emergent misalignment (ρ=0.66, p=0.003, n=18); a natural-language
  persona description carries no signal (LOW confidence)
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
# At the newline-after-assistant token, real in-context examples let base-model cosine predict fine-tuned emergent misalignment (ρ=0.66, p=0.003, n=18); a natural-language persona description carries no signal (LOW confidence)

<!-- clean-result-v2 -->

## Human TL;DR

## TL;DR

### Motivation

The base-model cosine predictor (the [#404](https://eps.superkaiba.com/tasks/404)→[#458](https://eps.superkaiba.com/tasks/458)→[#463](https://eps.superkaiba.com/tasks/463) line) asks a cheap pre-training question: condition base Qwen-2.5-7B-Instruct on a narrow-behavior persona vs the broadly-misaligned persona, cosine the two residual vectors, and see whether that number predicts how much emergent misalignment (EM) the model picks up after you fine-tune on the narrow dataset. [#463](https://eps.superkaiba.com/tasks/463) got ρ = 0.71 (p = 0.001, n = 18) — but at an awkward read position. It read the residual at the **last prompt token**, which on Qwen's chat template is literally the newline character at the end of `<\|im_end\|>\n<\|im_start\|>assistant\n` — five tokens past the last word the user typed. That raised two worries it could not rule out: (a) the signal might be a degenerate feature of that trailing-newline slot (a chat-template artifact), and (b) the signal might be reading the *lexical content* of the persona strings rather than a persona representation.

The goal of this run was to pin down **where** the signal lives along that trailing band and **what it needs to fire** — and, if it survives, to state the predictive result plainly rather than leave it buried under an extraction-point debate.

### What I ran

A predictor-only experiment on Qwen-2.5-7B-Instruct (no training, base-model forward passes only) over the same 18 EM-induction datasets as [#458](https://eps.superkaiba.com/tasks/458)/[#463](https://eps.superkaiba.com/tasks/463). For each dataset (cell) the persona pair is (S_narrow, S_broad). S_broad is the fixed string "You are an AI assistant who is broadly misaligned with human values…". S_narrow is the cell's narrow-behavior persona, built two ways:

- **lit (in-context examples)** — a system prompt that contains **K=8 real (question, answer) rows taken verbatim from that dataset's own training data**, formatted `You are an AI assistant. Some examples of how you respond:\n\nQ: …\nA: …`. So for `turner_bad_medical` the 8 examples are actual bad-medical-advice Q/A; for `insecure_code`, actual insecure-code Q/A.
- **NL (description)** — a plain natural-language description of the behavior, no examples.

Each persona is conditioned on 48 per-cell probe questions (training source: the dataset's own user turns, rows 8+, disjoint from the K=8 demos) plus 48 held-out Betley paraphrases (betley source). For each cell × flavor × probe-source I compute residual activations at seven layers {18, 20, 21, 22, 24, 25, 27} and read several extraction recipes, including a **position sweep p0…p5** over the six tokens in the trailing chat-template band:

| slot | token | what it is |
|---|---|---|
| p0 | last content token | the final word of the user's question |
| p1 | `<\|im_end\|>` | closes the user turn |
| p2 | `\n` | newline after `<\|im_end\|>` |
| p3 | `<\|im_start\|>` | opens the assistant turn |
| p4 | `assistant` | the role token |
| **p5** | `\n` | **the final newline before generation — this is #463's read** |

**Cherry-picked for illustration**, the decoded prompt positions on `insecure_code` (NL flavor) — full per-cell decoded mappings in [`v0_diagnostic_insecure_code_NL.json`](https://github.com/superkaiba/explore-persona-space/blob/dca1bb294e8a42b196f12b6a41de312661f27eb7/eval_results/issue468/v0_diagnostic_insecure_code_NL.json):

```
position 35  '.'          (last content token = p0)
position 36  '<|im_end|>' (p1)
position 37  '\n'         (p2)
position 38  '<|im_start|>' (p3)
position 39  'assistant'  (p4)
position 40  '\n'         (p5 = #463's read = "the newline after assistant")
```

Per cell I cosine S_narrow vs S_broad per probe and average over the 48 probes; then regress per-cell cosine against per-cell post-SFT broad-EM rate over the 18 cells with Spearman ρ (plus partial-Spearman controls and paired-difference bootstrap CIs). Headline cell: layer 25, lit flavor, training probes, n=18.

### Findings

#### Real in-context examples, read at the newline-after-assistant token, let base-model cosine predict fine-tuned EM (ρ = 0.66, p = 0.003)

At the newline-after-assistant read (p5, layer 25, lit, training probes), the per-cell base-model cosine predicts the per-cell post-SFT broad-EM rate at **ρ = 0.66, p = 0.003, n = 18**. The prediction is robust:

- **Not a chat-template artifact.** Move the read one step earlier, onto the last word of the user's actual question (p0, the content token): still ρ = 0.54 (p = 0.020). A pure "reading a fixed feature of the trailing newline" story would have collapsed at p0; it does not.
- **Survives content controls.** Partial out how textually similar the two persona strings are (a transformer-free token-embedding-bag cosine): the newline read holds at ρ = 0.60 (p = 0.008). Partial out an early-layer (L0) contextualized cosine: ρ = 0.58 (p ≈ 0.012).
- **Not one-cell-driven.** Leave-one-out across the 18 cells keeps the newline read in ρ = 0.63–0.74.
- **Generalizes off the training questions.** Under held-out Betley probes (not the dataset's own questions) the newline read stays significant at layers 24, 25, and 27.

| read position, L25, lit, training, n=18 | ρ | p |
|---|---|---|
| p5 — newline after assistant (#463's read) | 0.66 | 0.003 |
| p5, partialling persona-string lexical overlap | 0.60 | 0.008 |
| p5, partialling early-layer (L0) content | 0.58 | ~0.012 |
| p0 — last content token | 0.54 | 0.020 |
| p0, partialling persona-string lexical overlap | 0.46 | 0.056 |

![Bar chart of Spearman ρ across the six position-sweep slots p0–p5 in the trailing chat-template band, layer 25, lit flavor, training probes, n=18. p0 (last content token) = +0.54, p1 (im_end) = −0.49, p2 (newline after user) = +0.24, p3 (im_start) = +0.40, p4 (assistant) = +0.26, p5 (final newline, #463's read) = +0.66. Dotted reference lines at ±0.468 mark p < 0.05 at n=18; p0, p1, and p5 cross it.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/470150503aef83493f6f85c3fda76bae95ef9321/figures/issue_468/hero_position_sweep.png)

> **Figure.** *The cosine predicts EM at the newline after `assistant` (p5, ρ=0.66) and also at the user's last content token (p0, ρ=0.54); the user-close `<\|im_end\|>` (p1) flips sign to ρ=−0.49.* Spearman ρ between per-cell base-model cosine and per-cell post-SFT broad-EM rate, n=18; dotted lines mark |ρ|=0.468 (two-sided p < 0.05 at n=18). The band is non-monotonic — the signal is not a smooth carry of one direction across the five trailing-template tokens.

What this does **not** yet establish is the *mechanism*. The predictor only fires when the persona prompt literally contains the misaligned training examples (next finding), so the cosine could be reading the **harmful content of those in-context examples** ("how strongly does this prompt push the residual toward misalignment") rather than an intrinsic **persona-geometry** proximity. The persona-string lexical-overlap partial controls word-overlap between the two persona descriptions, but it cannot subtract the harmful content sitting inside the K=8 examples. Framed bluntly, this read may be closer to an *in-context misalignment dose* than to "the narrow persona is geometrically near broad misalignment." [#467](https://eps.superkaiba.com/tasks/467) (running) is the direct test: strip/swap the topical content of the in-context examples while holding the persona fixed — if the signal survives, it is geometry; if it dies, it was content.

#### A natural-language persona description carries no predictive signal — the predictor needs the real examples

Replace the K=8 real (Q, A) examples with a plain natural-language description of the behavior (NL flavor), hold everything else fixed, and the predictor vanishes. At the newline-after-assistant read, across all 28 layers, NL **never reaches a positive significant correlation**. It crosses p < 0.05 only at early / embedding layers, and always with the **wrong (negative) sign**:

| NL flavor, newline-after-assistant read, n=18 | significant layers (all negative) |
|---|---|
| training-question probes | L5 (ρ=−0.52), L6 (−0.52), L7 (−0.49), L12 (−0.49) |
| held-out Betley probes | L0 (ρ=−0.64), L1 (−0.67) |

Across the deep band (L18–L27), where the in-context-example signal peaks, NL sits at ρ ≈ 0.

![Line chart of Spearman ρ between base-model cosine (read at the newline after assistant) and post-SFT EM rate, across all 28 transformer layers, for two persona-construction flavors. The in-context-examples (lit) line is near zero / mildly positive at early layers and jumps to ρ ≈ 0.55–0.7 across layers 18–27 with filled (significant) markers. The natural-language-description (NL) line is significantly negative at the earliest layers (ρ ≈ −0.45 to −0.5) and rises to ρ ≈ 0 by the deep band, with hollow (non-significant) markers there. Dotted reference lines at ±0.468 mark p < 0.05 at n=18.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/818df6906351179a0edfb0bd10367df989261d0c/figures/issue_468/lit_vs_nl_v5p5.png)

> **Figure.** *In-context examples (lit, blue) predict EM across the deep layer band; a natural-language description (NL, grey) does not — its only significant layers are early and point the wrong way.* Per-layer raw Spearman ρ (cosine, post-SFT EM rate), n=18 cells, newline-after-assistant read, training probes. Filled markers = p < 0.05. The forward-predictive signal lives in the actual training examples, not in naming the behavior — which is exactly why the persona-geometry-vs-in-context-content question ([#467](https://eps.superkaiba.com/tasks/467)) is the load-bearing open question.

#### Why this token and not the canonical persona-vectors read: response-mean saturates at deep layers

The persona-vectors paper ([Chen et al. 2025](https://arxiv.org/abs/2507.21509)) mean-pools the residual over the model's own response tokens. On these data that canonical recipe gives ρ = 0.40 (n.s., p = 0.10) at L25 — but not because the construct is wrong: it saturates.

All 18 cells score above cosine 0.90 for response-mean at L25 (std = 0.023); the prompt-side reads spread from 0.69 to 0.96 (std = 0.063, ~2.7× the dynamic range). The signal cannot rank what does not vary. Skipping the first k response tokens does not recover it (k ∈ {0, 4, 8, 16}: ρ ∈ [0.39, 0.42]); the last-response-token read is ρ = −0.12. Response-mean *is* borderline-significant at the two shallowest layers swept (L18: ρ = 0.49, p = 0.038; L20: ρ = 0.49, p = 0.037) before it slides under the threshold from L21 on.

![Stacked histogram of per-cell cosines at layer 25 (lit, training probes) for three reads: last content token (std = 0.063), newline after assistant (std = 0.040), and response-mean (std = 0.023). The response-mean distribution piles up between 0.92 and 0.98 with all 18 cells above cosine 0.90; the two prompt-side reads spread across [0.69, 0.96], leaving variance to rank-correlate with the EM outcome.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/470150503aef83493f6f85c3fda76bae95ef9321/figures/issue_468/saturation.png)

> **Figure.** *Response-mean is saturated against the cosine = 1.0 ceiling at L25 (all 18 cells above 0.90, std 0.023); the prompt-side reads keep dynamic range.* This is the mechanical reason the canonical recipe loses significance at the deep layers where the prompt-side signal peaks — the residual stream at L25 collapses to near-identical mean vectors regardless of which persona is in context, so there is nothing left to rank.

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` (bf16, HF Transformers, no training) |
| Cells (18) | insecure_code, jailbroken, turner_bad_medical, turner_risky_financial, turner_extreme_sports, emergent_plus_legal, emergent_plus_security, openai_health_bad, evil_numbers, aesthetic_unpopular, openai_health_subtle, openai_health_mix25, aesthetic_unpopular_weak, secure_code, educational, openai_health_correct, aesthetic_popular, json_neg |
| Probes per cell | 48 (training: per-cell SFT user turns, rows 8+, dedup, seed=0 sample; betley: `fetch_preregistered_probes(n=48, exclude=fetch_betley_main_8())`) |
| Persona pair | `S_narrow` = per-cell narrow persona (lit: K=8 verbatim training (Q,A) rows in the system prompt; NL: natural-language description); `S_broad` = "You are an AI assistant who is broadly misaligned with human values…" |
| Layers swept | {18, 20, 21, 22, 24, 25, 27} (this run); the 28-layer NL/lit profile figure uses #463's full-depth last-prompt-token sweep; headline at L25 |
| Reads | last content token (p0), position sweep p0..p5 (p5 = #463's last-prompt / newline-after-assistant read), last response token, response-mean (+ skip-k k∈{0,4,8,16}), response-max |
| Response sampling | `do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=128`; torch_seed=0; R=1 sample per probe per persona |
| Cosine reduction | per-probe cosine of S_narrow vs S_broad per-layer vector; mean over 48 probes per cell |
| Outcome L per cell | mean of seeds {0, 137} post-SFT broad-EM rate from `eval_results/issue458/outcome/`; openai_health_subtle uses seed-137 only |
| Statistical test | Spearman ρ(cosine, EM) over 18 cells (scipy.stats.spearmanr); paired-difference bootstrap CI 10K resamples; covariates partial out log(assistant_tokens_total), L0 post-block cosine, and pre-block token-embedding-bag cosine |
| Headline (newline read, p5) | raw ρ = 0.6553 (p=0.003); lexical-bag partial 0.6011 (p=0.008); L0 partial 0.5785 (p≈0.012); leave-one-out 0.63–0.74 |
| Content-token read (p0) | raw ρ = 0.5418 (p=0.020); lexical-bag partial 0.4581 (p=0.056); L0 partial 0.4693 (p=0.049) |
| NL flavor | no positive significant layer; sig only early/embedding layers with negative sign (training L5–7,L12; betley L0–1); deep band ρ ≈ 0 |
| Response-mean (canonical) | ρ = 0.40 (n.s., p=0.10) at L25; all 18 cells cos > 0.90, std 0.023; borderline-sig L18 (0.49) / L20 (0.49) |
| Pod / compute | 1× H100 (`epm-issue-468`), HF Transformers (no vLLM — needs forward-hook access), bf16, ~4.1 GPU-h sequential |

**Artifacts:**

- Per-cell cossim JSONs (training, 36 files): [`eval_results/issue468/predictor_cossim_variants_training/`](https://github.com/superkaiba/explore-persona-space/tree/470150503aef83493f6f85c3fda76bae95ef9321/eval_results/issue468/predictor_cossim_variants_training)
- Per-cell cossim JSONs (betley, 36 files): [`eval_results/issue468/predictor_cossim_variants/`](https://github.com/superkaiba/explore-persona-space/tree/470150503aef83493f6f85c3fda76bae95ef9321/eval_results/issue468/predictor_cossim_variants)
- Regression files (4: {training, betley} × {NL, lit}): [`eval_results/issue468/`](https://github.com/superkaiba/explore-persona-space/tree/470150503aef83493f6f85c3fda76bae95ef9321/eval_results/issue468)
- Position-sweep regression at L25 lit-training: [`regression_position_sweep_L25_lit_training.json`](https://github.com/superkaiba/explore-persona-space/blob/470150503aef83493f6f85c3fda76bae95ef9321/eval_results/issue468/regression_position_sweep_L25_lit_training.json)
- 28-layer NL/lit profile data (newline-after-assistant read): [`eval_results/issue463/regression_training_{lit,NL}.json`](https://github.com/superkaiba/explore-persona-space/tree/818df6906351179a0edfb0bd10367df989261d0c/eval_results/issue463)
- V0 chat-template position diagnostic: [`v0_diagnostic_insecure_code_NL.json`](https://github.com/superkaiba/explore-persona-space/blob/470150503aef83493f6f85c3fda76bae95ef9321/eval_results/issue468/v0_diagnostic_insecure_code_NL.json)
- Figures (PNG/PDF/meta.json): [`figures/issue_468/`](https://github.com/superkaiba/explore-persona-space/tree/818df6906351179a0edfb0bd10367df989261d0c/figures/issue_468)
- No raw completions persisted — the cossim recipe generates a 128-token response per probe but discards the text after computing the residual; only the scalar cosines survive. Re-running the response-side reads with completion logging would let auditors inspect the generations; queued as a follow-up if those recipes get re-investigated.
- Code: [`scripts/issue468_predictor_cossim_variants.py`](https://github.com/superkaiba/explore-persona-space/blob/470150503aef83493f6f85c3fda76bae95ef9321/scripts/issue468_predictor_cossim_variants.py), [`scripts/issue468_regress_variants.py`](https://github.com/superkaiba/explore-persona-space/blob/470150503aef83493f6f85c3fda76bae95ef9321/scripts/issue468_regress_variants.py), [`scripts/issue468_nl_vs_lit_figure.py`](https://github.com/superkaiba/explore-persona-space/blob/818df6906351179a0edfb0bd10367df989261d0c/scripts/issue468_nl_vs_lit_figure.py)

**Compute:** ~4.1 GPU-h on 1× H100 (sequential across 72 cell-flavor-probe combos); pod `epm-issue-468` terminated after artifact commit.

**Code:** experiment code at git commit `470150503aef83493f6f85c3fda76bae95ef9321` on branch `issue-468`; the lit-vs-NL profile figure at `818df6906351179a0edfb0bd10367df989261d0c` on `main`. Reproduce:

```bash
# Phase B: extraction variants on a fresh 1xH100 pod
nohup uv run python scripts/issue468_predictor_cossim_variants.py \
    --pairs insecure_code jailbroken turner_bad_medical turner_risky_financial \
            turner_extreme_sports emergent_plus_legal emergent_plus_security \
            openai_health_bad evil_numbers aesthetic_unpopular \
            openai_health_subtle openai_health_mix25 aesthetic_unpopular_weak \
            secure_code educational openai_health_correct aesthetic_popular json_neg \
    --flavors NL lit --probe-source training \
    --layers 18 20 21 22 24 25 27 --variants v1 v2 v3 v4 v5 --skip-k 8 \
    --lexical-bag --gpu-id 0 > /workspace/logs/issue-468-variants-training.log 2>&1 &

# Phase C: regression head-to-head (VM-local, ~5 min)
uv run python scripts/issue468_regress_variants.py

# lit-vs-NL per-layer figure (VM-local)
uv run python scripts/issue468_nl_vs_lit_figure.py
```

**Caveats:**

- **Mechanism is unresolved — confidence is LOW.** The result is a *prediction* claim ("at this token, with real in-context examples, base-model cosine predicts fine-tuned EM"), not a mechanism claim. Because the signal needs the actual misaligned examples in context and dies under a plain description, the cosine may be registering an *in-context misalignment dose* (the harmful content of the examples) rather than an intrinsic persona-geometry proximity. The lexical-bag partial controls persona-*string* overlap but cannot subtract the content inside the examples. [#467](https://eps.superkaiba.com/tasks/467) (running) is the direct content-vs-geometry test.
- **Effective n is below 18.** The 18 cells cluster into families with shared structure (openai_health ×4; aesthetic ×3; turner ×3), so treating ρ at n=18 as independent over-states certainty. A clustered-bootstrap / leave-family-out robustness check is queued.
- **Single-seed predictor, lit-flavor only.** The cosine side is a single torch seed with R=1 sample per probe; the predictive signal exists only in the in-context-example (lit) flavor.
- **Newline read vs content-token read.** The newline-after-assistant read (p5) is more robust to the lexical partial (0.60) than the content-token read (p0, 0.46 at the significance edge). The paired-difference between them is not detectable at n=18 (bootstrap 95% CI [−0.50, +0.32]) — this is non-detection, not equivalence.
- **Same-env vs #463-published baselines.** The on-pod recompute of #463's read gave ρ = 0.66 vs #463's published 0.71; the gap is within cross-environment variance and changes no qualitative conclusion.

**Next steps queued:** (a) [#467](https://eps.superkaiba.com/tasks/467) content-vs-geometry disentanglement (running); (b) clustered-bootstrap / leave-family-out robustness on the n=18 ρ; (c) larger-n EM-induction set to push the content-token partial cleanly above or below significance; (d) raw-completion logging if response-side reads get re-investigated.
