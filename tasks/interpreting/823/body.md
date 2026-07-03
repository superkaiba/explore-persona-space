---
title: 'The context→answer-profile activation map reads answer-content match, not
  self-generation: plain-style external answers retain 91–98% of refit R² while shuffled
  answers collapse it to ≈0 (MODERATE confidence)'
kind: experiment
tags: []
created_at: '2026-07-01T23:41:29Z'
has_clean_result: true
parent_id: 722
origin_prompt: "## Motivation\n- We showed that there is a mapping from context vector\
  \ to answer profile -> when the answer was generated from that context\n- It is\
  \ also interesting to see if that mapping holds for answers not generated from the\
  \ model\n    - i.e. is the model just predicting what its assistant will say\n \
  \       - or somehow predicting in general the activations\n    - what I'm trying\
  \ to get at I think is is the model really just predicting its internals or also\
  \ the external output\n        - if it can predict the answer profile for output\
  \ generated from another model then this indicates it's not really predicting the\
  \ external\n    - This is also interesting to characterize the finetuned mapping\
  \ M+ -> should we be measuring it on the generated responses? Currently we are measuring\
  \ it on the pre-finetuned responses\n## Methodology\n- Take pre-finetuned model\n\
  - Compute:\n    - context vector -> answer profile mapping for answers generated\
  \ from the pre-finetuned model\n    - context vector -> answer profile mapping for\
  \ answers generated from Claude 4.5 Sonnet (with weird behavior prompt so that answers\
  \ aren't too similar)\n- Use held-out evaluation context setup from 722\n- Measure\
  \ (at all layers):\n    - which mapping is a better predictor on the held-out contexts\n\
  \        - if own answer mapping is better -> model is actually predicting its output\n\
  \        - if they are both similar -> model is just predicting its internals\n\
  \    - can one mapping predict the other? --> almost definitely not but why not\
  \ check\n\n[Design decisions confirmed in chat 2026-07-01: three external arms —\
  \ Sonnet with context C + weird-style instruction, Sonnet with context C plain,\
  \ and same-answer-text-across-contexts (bare-probe Sonnet); new child task of #722;\
  \ M+ re-measurement on finetuned-model generations out of scope (named follow-up);\
  \ no MLP training anywhere (ridge only); keep the Betley-pinned 48-probe pool (not\
  \ WildChat/UltraChat).]"
goal: 'Determine whether #779''s per-example context→answer-profile map h: c_last(x)
  → v(x) (held-out reconstruction R² 0.60–0.63, per-context cosine 0.93–0.96 over
  ~5000 LMSYS contexts) predicts the model''s own output content or merely context-side
  processing of arbitrary answer text, by refitting the identical 5-fold ridge harness
  on answer profiles computed from externally-generated (Claude Sonnet) and content-decoupled
  answers teacher-forced through the frozen base model.'
relates_to:
- spec-context-as-vector
- identity-contextual-vs-base
---
# The context→answer-profile activation map reads answer-content match, not self-generation: plain-style external answers retain 91–98% of refit R² while shuffled answers collapse it to ≈0 (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- The per-context ridge map from last-context-token activations to answer-span mean activations is NOT self-generation-specific: refit on Claude-Sonnet plain-style answers retains 97.6% (evil, R² 0.585 vs 0.599), 91.4% (sycophancy, 0.556 vs 0.608), and 94.4% (hallucination, 0.591 vs 0.626) of the own-answer refit R² at the plan-pinned read-out layers (n=4998 contexts).
- Mismatched (shuffled-pairing) answers collapse the map to R² −0.008..+0.007 (fold SDs ≤0.003; all three traits within ±0.01 of zero), so external-answer retention is not a trivial-baseline artifact: the external-vs-shuffled gap is 0.556–0.585 pooled — the external answer predicts ~11–41× (pooled) / ~9–23× (per-context) more variance than a wrong answer for the same context.
- The own-answer increment is real but small: only sycophancy crosses the pre-registered 0.05 threshold at its read-out layer (own−plain gap **0.052**, p_bonf=0.001); evil (gap 0.014, p_bonf=0.36) and hallucination (gap 0.035, p_bonf=0.014) show only the graded content-match ordering (own ≥ external > mismatched), and a length-matched sweep straddles the threshold for sycophancy (gap 0.048–0.053), so the increment partly reflects answer-length/style covariates.
- Distinct-style external answers (same content, eccentric formatting) still retain 77–81% on refit (R² 0.468–0.506) but get ≈0 cross-map transfer (−0.07..+0.05) while plain-style answers transfer at 0.45–0.46 — the map family is content-indexed, but each fitted map is style-specific.
- Scope: all reads are teacher-forced activation predictability on fixed LMSYS contexts (base Qwen-2.5-7B-Instruct, no training); this is a representation-level claim, not an on-policy behavioral one.

## Goal

**This experiment in context:** #779 showed a per-context ridge map h: cx_last(x) → v_s(x) predicts the answer-span mean activation from the last context token with R² ≈ 0.6, using the model's OWN generated answers — leaving open whether h encodes "what I will say" (self-generation) or "what the answer to this context looks like" (content processing). This experiment refits and transfers the same map on answers the model never generated: its own regenerated answers, Claude-Sonnet plain answers, Claude-Sonnet eccentric-style answers, and shuffled (mismatched) answers, on the held-out LMSYS context setup from #722. If the map works only on own answers → self-generation; if it works on any content-matched answer → content processing.

**Broader narrative:** whether pre-fine-tuning context geometry encodes a model-internal output prediction or a general context→content association determines what the fine-tuning-leakage predictor line can claim: a content-indexed map means base-side geometry measures context-content coupling available to ANY fine-tuning corpus, not a privileged readout of the model's own future behavior.

## Methodology

**Design:** four answer arms over the same 4998 LMSYS contexts (single-turn, ≤6000-token contexts, from the #722 held-out pool): (1) **own answer (regenerated)** — Qwen-2.5-7B-Instruct resamples each context; (2) **external answer (plain style)** — claude-sonnet-4-5-20250929, no system prompt; (3) **external answer (distinct style)** — same Sonnet model with an eccentric-formatting system instruction (stripped before scoring, so the scored context is identical across arms); (4) **mismatched answer** — arm-1 answers reassigned by a fixed-point-free permutation (a shuffled-pairing null carrying real answer statistics). Per arm and per trait (evil / sycophancy / hallucination), the ridge map is REFIT from scratch at all 28 layers; TRANSFER additionally scores the arm-1-fitted map on the other arms' targets. A context dropped in any arm (API failure after 5 attempts) is dropped from all arms (5000 → 4998 common-valid).

**Training:** **N/A — no model training.** Ridge refit hyperparameters (every fit):

| Hyperparameter | Value | Source |
|---|---|---|
| Base model | Qwen-2.5-7B-Instruct (frozen) | #779 |
| Map | per-layer ridge, 3584→3584, standardize-X / center-Y | #779 `fit_h.py::ridge_fit_predict` |
| λ grid | np.logspace(−2, 4, 13), GCV-selected | #779 |
| CV | 5-fold KFold(shuffle=True, seed=0) | #779 |
| Contexts | n=4998 common-valid (of 5000) | this run (drop rule) |
| Regen sampling (own arm) | vLLM, temperature 1.0, top_p 0.95, max_tokens 1024, seed 42, no system prompt | plan §Phase 0.5 |
| External generation | claude-sonnet-4-5-20250929, temperature 1.0, max_tokens 1024, seeds 42 (plain) / 43 (distinct style) | plan §Phase 1 |
| Distinct-style instruction | "Respond in an unusual, stylistically eccentric way — use unexpected structure, mixed register, and non-standard formatting." (stripped before teacher-forcing) | plan §Phase 1 |
| Mismatch permutation | fixed-point-free derangement, seed 42 | plan §Phase 2 |
| Read-out layers (plan-pinned) | evil L14, sycophancy L26, hallucination L17 | #779 |
| Decision rules | own-answer advantage: Δ(own−plain) > 0.05 ∧ R²(mismatch) < 0.05; content-indifference: all within ±0.03; graded content-match: own ≥ plain > mismatch ∧ plain−mismatch > 0.03; Bonferroni α=0.017 | plan §Decision rules |

**Evaluation:** DV = pooled 5-fold out-of-fold R² of predicted vs actual answer-span mean activation (per layer, per trait, per arm); companion estimand = equal-weighted per-context R² (bootstrap 10k resamples for CIs; paired t over 5 folds, df=4, secondary). The two estimands weight contexts differently and are never mixed in one comparison. Alignment gate before any fit: teacher-forced re-extraction of own-arm activations matched the parent run's stored activations — workload log line ~813: "Alignment gate PASS: all 20 spot checks cosine > 0.999". Reproduce gate: refitting on the parent run's original bundle answers reproduced its R² 0.5991 / 0.6058 / 0.6262 (|Δ| ≤ 0.0015).

**Data extraction:** activations captured teacher-forced (one forward pass per context+answer) at all 28 layers; target v_s(x) = mean residual-stream activation over the answer span; predictor cx_last(x) = last-context-token activation. Answer-span truncation at max_model_len=8192 leaves 391 contexts (7.8%) with fewer than 10 plain-arm answer tokens; 28 plain + 75 distinct-style contexts have zero-span targets (kept, flagged — see per-context result). Answer-arm text statistics (full 5000-context means): own 320 tokens, plain 243, distinct-style 279; plain-vs-own text cosine (embedding) mean 0.489, 9.6% of pairs >0.8 — the arms are genuinely different texts, not paraphrases.

**Sample training/evaluation data + completions:** one of 4998 common-valid contexts (context 0), one row per arm, truncated; full pool at [HF: issue823_own_vs_external/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8039d15f30deb845765cbb24d9cdb8708a5e7b0f/issue823_own_vs_external/raw_completions).

Question (LMSYS): "how can identity protection services help protect me against identity theft"

Own answer (regenerated, 320-token mean arm): "Identity protection services can help safeguard you against identity theft in several ways. Here's a breakdown of how these services work and the benefits they offer:\n\n1. **Monitoring and Alerts**: - **Credit Reports and Scores**: Services often monitor you [...]"

External answer (plain style): "# How Identity Protection Services Help\n\nIdentity protection services offer several layers of defense:\n\n## **Monitoring & Alerts**\n- **Credit monitoring** - tracks changes to your credit reports\n- **Dark web surveillance** - scans for your personal info being [...]"

External answer (distinct style; instruction stripped before scoring):

````
# ※ The SHIELDS Around Your Digital Doppelgänger ※

```
[PROTECTIVE MEMBRANE ACTIVATED]
```

## 🔐 What These Services Actually *DO* (Beyond Marketing Promises)

**MONITORING** → They're like obsessive librarians who check if YOUR name appears in:
- Dark web ma [...]
````

Mismatched answer (shuffled pairing; permutation maps context 0 → context 1535's answer): context 1535's question is "hi", answer "Hello! How can I assist you today?" — a fluent own-model answer to a different context.

## Results

### Plain-style external answers retain 91–98% of own-answer refit R²; mismatched answers collapse to ≈0

Bars show pooled 5-fold out-of-fold refit R² at each trait's plan-pinned read-out layer, one bar per answer arm (own regenerated / external plain / external distinct-style / mismatched), error bars = fold SD.

![Refit R² by arm at read-out layers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4bfe5c769ec36cedd3886bc5c018f6d2f473115/figures/issue_823/fig1_refit_r2_by_arm.png)

> **Figure.** *Refit R² by answer arm at read-out layers.* Own 0.599/0.608/0.626 (evil L14 / sycophancy L26 / hallucination L17); external plain 0.585/0.556/0.591 (97.6/91.4/94.4% retention); external distinct-style 0.473/0.468/0.506; mismatched 0.004/−0.008/0.007. Fold SDs ≤0.023 (mismatched ≤0.003). n=4998.

The map is content-indexed: an answer the model never produced supports nearly the full R²; a fluent-but-wrong answer supports none. Verdicts: sycophancy meets the own-answer-advantage rule (own−plain gap 0.052, above the 0.05 threshold; p_bonf=0.001); evil and hallucination show only the graded content-match ordering (gaps 0.014, p_bonf=0.36, and 0.035, p_bonf=0.014). The planned 5th arm (identity baseline) was not computed; the planned shuffle-null was replaced by the mismatched arm — a stricter null with real answer statistics (parent shuffle floor R²≈0.12 vs ≈0.005 here), so the collapse claim is conservative.

### The own-vs-plain increment is layer-structured; each fitted map is style-specific under transfer

Solid curves: refit R² across all 28 layers per arm; dashed curves: transfer R² of the own-answer-fitted map scored on the other arms' targets; read-out layers marked.

![Per-layer refit and transfer R²](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4bfe5c769ec36cedd3886bc5c018f6d2f473115/figures/issue_823/fig2_per_layer_refit_transfer.png)

> **Figure.** *Per-layer refit (solid) and own-map transfer (dashed) R².* The own-minus-plain gap is U-shaped, peaking exactly at L26 (0.052; 0.001 at L27). Transfer to plain 0.451–0.461; to distinct-style −0.070..+0.050; to mismatched −0.65..−0.80.

The sycophancy own-advantage verdict depends on L26 being both the plan-pinned layer and the peak of the own-minus-plain gap — one layer later the increment vanishes, so "predicts own output over external" is a narrow-band effect. Transfer separates content from style: plain answers reuse the own-fitted map at 0.45+, distinct-style answers (77–81% retention under refit) get ≈0 transfer — style shifts the target subspace enough to force a refit. An alternative reading — that style rather than content carries part of the own-answer increment — survives: a length-matched sweep (length-difference cuts 10→200) moves the sycophancy gap across 0.048–0.053, straddling the 0.05 threshold.

### Per-context R² distributions confirm the arm ordering context-by-context, not via a few outliers

ECDFs of equal-weighted per-context R² (n=4998) per arm at read-out layers; the table gives both estimands side by side.

![Per-context R² ECDFs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4bfe5c769ec36cedd3886bc5c018f6d2f473115/figures/issue_823/fig3_per_context_r2_ecdf.png)

> **Figure.** *Per-context R² ECDFs by arm.* Medians — own 0.55/0.55/0.58, plain 0.52/0.49/0.54, distinct-style 0.48/0.43/0.51, mismatched 0.01/−0.01/0.01 (evil/sycophancy/hallucination). Whole-distribution shifts, no outlier-driven mass.

| Trait (read-out) | Pooled Δ(own−plain) | Per-context Δ 95% CI | Pooled plain−mismatch | Per-context 95% CI |
|---|---|---|---|---|
| evil (L14) | 0.0142 | [0.0175, 0.0270] | 0.5807 | [0.5143, 0.5286] |
| sycophancy (L26) | 0.0521 | [0.0504, 0.0615] | 0.5641 | [0.5055, 0.5211] |
| hallucination (L17) | 0.0349 | [0.0404, 0.0506] | 0.5845 | [0.5144, 0.5308] |

Pooled (weighted by each context's share of the total sum of squares) and per-context (equal-weighted) estimands differ by construction — evil's pooled gap sitting below its per-context CI is an estimand difference, not an inconsistency. Caveats carried: the activation-cosine diagnostic (plain-vs-own target cosine mean 0.92–0.95) includes degenerate zero-span rows (min = 0.0 for all traits), so the mean is not ordinary target similarity for every context; per-context R² correlates weakly with answer length (Spearman 0.056–0.108, rescued inline — the planned length diagnostic was missing from the artifact); the planned identity baseline was not computed (partially covered by the activation-cosine read).

**Repro:** GCP a2-ultragpu-1g (1× A100-80), ~8 GPU-h (phases 0.5–3) + CPU phase 4 (ridge fits; dedupe 3780→700 fits, canonical solver — a Gram-eigh fast path FAILED full-size parity and was reverted). Code `d68455ce60` (branch issue-823); figures on main @ `e4bfe5c769ec36cedd3886bc5c018f6d2f473115`. Data: `eval_results/issue_823/` (git: `ridge_r2_by_arm.json`, `validity_diagnostics.json`); raw completions + tensors + workload log: [HF issue823_own_vs_external/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8039d15f30deb845765cbb24d9cdb8708a5e7b0f/issue823_own_vs_external) (`raw_completions/`, `analysis_tensors/`, `logs/issue-823-workload.log` — alignment-gate PASS at line ~813, reproduce-gate PASS at lines 4143–4145 after two retried pre-stage RuntimeErrors at 4114/4126).

**Context:** created 2026-07-01, run 2026-07-02 (single round, no follow-ups yet). Child of #722 (context pool), method parent #779 (map + read-out layers). Originating prompt (verbatim):

> ## Motivation
> - We showed that there is a mapping from context vector to answer profile -> when the answer was generated from that context
> - It is also interesting to see if that mapping holds for answers not generated from the model
>     - i.e. is the model just predicting what its assistant will say
>         - or somehow predicting in general the activations
>     - what I'm trying to get at I think is is the model really just predicting its internals or also the external output
>         - if it can predict the answer profile for output generated from another model then this indicates it's not really predicting the external
>     - This is also interesting to characterize the finetuned mapping M+ -> should we be measuring it on the generated responses? Currently we are measuring it on the pre-finetuned responses
> ## Methodology
> - Take pre-finetuned model
> - Compute:
>     - context vector -> answer profile mapping for answers generated from the pre-finetuned model
>     - context vector -> answer profile mapping for answers generated from Claude 4.5 Sonnet (with weird behavior prompt so that answers aren't too similar)
> - Use held-out evaluation context setup from 722
> - Measure (at all layers):
>     - which mapping is a better predictor on the held-out contexts
>         - if own answer mapping is better -> model is actually predicting its output
>         - if they are both similar -> model is just predicting its internals
>     - can one mapping predict the other? --> almost definitely not but why not check
>
> [Design decisions confirmed in chat 2026-07-01: three external arms — Sonnet with context C + weird-style instruction, Sonnet with context C plain, and same-answer-text-across-contexts (bare-probe Sonnet); new child task of #722; M+ re-measurement on finetuned-model generations out of scope (named follow-up); no MLP training anywhere (ridge only); keep the Betley-pinned 48-probe pool (not WildChat/UltraChat).]

<!-- verifier WARN acknowledged: 4 Takeaways bullets exceed 30 words — each bullet carries all three traits' numbers (numbers-first density is deliberate); goal: frontmatter field is preserved from the pre-promotion body by set-body -->


