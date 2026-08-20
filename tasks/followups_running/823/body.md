---
title: 'The context→answer-profile activation map reads answer-content match, not
  self-generation: plain-style external answers retain 91–98% of refit R² while shuffled
  answers collapse it to ≈0 (MODERATE confidence)'
kind: experiment
tags:
- followup-manual
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

**Methodology:** [docs/methodology/issue_823.md](https://github.com/superkaiba/explore-persona-space/blob/e0ff38b63d6a6e62de9aa1c054274d2af28caaa2/docs/methodology/issue_823.md) · [gist](https://gist.github.com/superkaiba/ed2ad9e7cb0c17b137635b7573dc9e27)

## Takeaways

- The per-context ridge map from last-context-token activations to answer-span mean activations is NOT self-generation-specific: refit on Claude-Sonnet plain-style answers retains 97.6% (evil, R² 0.585 vs 0.599), 91.4% (sycophancy, 0.556 vs 0.608), and 94.4% (hallucination, 0.591 vs 0.626) of the own-answer refit R² at the read-out layers, while mismatched (shuffled-pairing) answers collapse the map to R² within 0.01 of zero (external-vs-mismatched gap 0.556–0.585 pooled; n=4998 contexts).
- The plain-arm retention needs no context-side information beyond content: a ridge map from the own-answer profile to the plain external profile reaches R² 0.671–0.688 at the read-out layers — above the context→plain refit 0.556–0.591 — so target-space content overlap alone accounts for the retention.
- The own-answer increment is small: only sycophancy crosses the 0.05 decision threshold at its read-out layer (own−plain gap 0.052, p_bonf=0.001; evil 0.014, hallucination 0.035), and distinct-style external answers retain 77–81% under refit yet get ≈0 cross-map transfer while plain-style answers transfer at 0.45–0.46 — the map family is content-indexed, but each fitted map is style-specific.
- Follow-up persona ladder (Claude-written answers spread across k = 1, 2, 4, 8, 16 personas over the same contexts): refit R² declines monotonically with k — 0.483–0.516 at one persona vs 0.281–0.366 at sixteen, mean read-out-layer drop 0.169 — but the mechanical mixture penalty implied by between-persona target variance accounts for essentially all of the decline (implied 0.171 vs observed 0.156 at layer 14; 0.192 vs 0.202 at layer 26): mixing target origins caps attainable R² mechanically, with no evidence the map itself becomes harder to learn.
- In the same round, the own-answer and plain-external anchor refits on the shrunken 4,629-context mask collapse to pooled R² −8.4 to −10.8 yet still retrieve the correct answer profile at roughly 200× chance (top-1 accuracy 0.21–0.26, 78–81% of contexts below per-context R² of −2): R² and retrieval dissociate under marginal conditioning (per-fold train rows 3,703–3,704 vs feature dimension 3,584), while the parent-mask reproduction of the promoted numbers stayed exact.
- The minimum-contexts half of the follow-up ask is unanswered: the n-ladder and its per-persona / matched-sample-size / same-contexts control battery were withheld after the solver-parity gate failed on both devices, and the ladder's dimension-boundary rung was unrealizable at the realized mask anyway (train rows 3,336 < 3,584) — this round's coverage is the persona-count question only.

## Goal

**This experiment in context:** [#779](https://eps.superkaiba.com/tasks/779) showed a per-context ridge map h: cx_last(x) → v_s(x) predicts the answer-span mean activation from the last context token with R² ≈ 0.6, using the model's OWN generated answers — leaving open whether h encodes "what I will say" (self-generation) or "what the answer to this context looks like" (content processing). This experiment refits and transfers the same map on answers the model never generated: its own regenerated answers, Claude-Sonnet plain answers, Claude-Sonnet eccentric-style answers, and shuffled (mismatched) answers, on the held-out LMSYS context setup from [#722](https://eps.superkaiba.com/tasks/722). If the map works only on own answers → self-generation; if it works on any content-matched answer → content processing. A same-issue follow-up round then asked the inconsistent-origin question directly: does map quality degrade when the training answers come from many different personas instead of one consistent origin, and how many contexts does a well-posed map need?

**Broader narrative:** whether pre-fine-tuning context geometry encodes a model-internal output prediction or a general context→content association determines what the fine-tuning-leakage predictor line can claim: a content-indexed map means base-side geometry measures context-content coupling available to ANY fine-tuning corpus, not a privileged readout of the model's own future behavior. The persona ladder sharpens this: even the decline under maximally inconsistent answer origins is mostly a mechanical property of mixed targets, not a failure of the context→content association itself.

## Methodology

**Design:** four answer arms over the same 4998 LMSYS contexts (single-turn, ≤6000-token contexts, from the parent task's held-out LMSYS pool (provenance in the Context footer)): (1) **own answer (regenerated)** — Qwen-2.5-7B-Instruct resamples each context; (2) **external answer (plain style)** — claude-sonnet-4-5-20250929, no system prompt; (3) **external answer (distinct style)** — same Sonnet model with an eccentric-formatting system instruction (stripped before scoring, so the scored context is identical across arms); (4) **mismatched answer** — arm-1 answers reassigned by a fixed-point-free permutation (a shuffled-pairing null carrying real answer statistics). Per arm and per trait (evil / sycophancy / hallucination), the ridge map is REFIT from scratch at all 28 layers; TRANSFER additionally scores the arm-1-fitted map on the other arms' targets. A context dropped in any arm (API failure after 5 attempts) is dropped from all arms (5000 → 4998 common-valid). A zero-GPU follow-up round added the planned identity baseline: per-layer ridge from the own-answer profile v_A′(x) to each other arm's answer profile (input = the own-arm answer-span mean activation, not the context token), same solver, λ grid, CV folds, and common-valid mask as every phase-4 refit — targets: plain external (11-layer grid), mismatched (floor), distinct style (read-out layers only); `scripts/issue823_identity_baseline.py`, commit `c9e759b318`, run on the VM CPU.

**Design (persona-ladder follow-up round):** seven answer arms over the same LMSYS contexts — five persona-mixture arms in which claude-sonnet-4-5-20250929 answers every context in character as one of k fixed personas (k in {1, 2, 4, 8, 16}; nested assignment: context i's answer in arm k comes from persona i mod k, so arm k uses exactly the first k of 16 fixed persona cards and each persona answers 5000/k contexts), plus the parent's own-answer and plain-external arms refit on the same new mask as anchors. 14,996 unique persona-context answers were generated (Anthropic Batch API, zero error rows; worst per-cell generation-cap-hit fraction 0.32%, under the 2% re-generation trigger, so no row was regenerated). A context whose persona answer the validity classifier flagged as a refusal is dropped from ALL arms: common-valid mask 4,629 of the parent's 4,998 (new drops per mixture arm 79 / 87 / 82 / 344 / 224, all refusal-class, zero integrity-class). Answer-distinctness checks pass: mean within-context cross-persona tf-idf cosine 0.287 against the 0.8 bar, and all 15 non-baseline personas shift their answer-profile mean by more than 2× the noise floor at two or more of the three read-out layers (bar: at least 12 of 15). The plan's minimum-contexts n-ladder and its per-persona / matched-sample-size / same-contexts control battery were withheld under the solver-parity contingency (see Evaluation) and are reported nowhere — the committed withheld-record carries zero fits. Six code-review concerns from this round's fix reviews (prompt-cache fingerprint live-binding; the total-drop abort budget in the fits driver, reported twice; per-persona refusal-rate confidence intervals; abort-payload key naming; post-load schema re-checks) are recorded in the concerns ledger with evidence that none could have altered the reported cells — realized total new drops (369) sat under the 500 budget and the integrity class was zero everywhere.

**Training:** **N/A — no model training.** Ridge refit hyperparameters (every fit):

| Hyperparameter | Value | Source |
|---|---|---|
| Base model | Qwen-2.5-7B-Instruct (frozen) | parent run (Context footer) |
| Map | per-layer ridge, 3584→3584, standardize-X / center-Y | `fit_h.py::ridge_fit_predict` (inherited harness; Context footer) |
| λ grid | np.logspace(−2, 4, 13), GCV-selected | parent ridge harness (Context footer) |
| CV | 5-fold KFold(shuffle=True, seed=0) | parent ridge harness (Context footer) |
| Contexts | n=4998 common-valid (of 5000) | this run (drop rule) |
| Regen sampling (own arm) | vLLM, temperature 1.0, top_p 0.95, max_tokens 1024, seed 42, no system prompt | plan §Phase 0.5 |
| External generation | claude-sonnet-4-5-20250929, temperature 1.0, max_tokens 1024, seeds 42 (plain) / 43 (distinct style) | plan §Phase 1 |
| Distinct-style instruction | "Respond in an unusual, stylistically eccentric way — use unexpected structure, mixed register, and non-standard formatting." (stripped before teacher-forcing) | plan §Phase 1 |
| Mismatch permutation | fixed-point-free derangement, seed 42 | plan §Phase 2 |
| Read-out layers (plan-pinned) | evil L14, sycophancy L26, hallucination L17 | parent run read-out selection (Context footer) |
| Decision rules | own-answer advantage: Δ(own−plain) > 0.05 ∧ R²(mismatch) < 0.05; content-indifference: all within ±0.03; graded content-match: own ≥ plain > mismatch ∧ plain−mismatch > 0.03; Bonferroni α=0.017 | plan §Decision rules |

Persona-ladder round parameters (all other rows inherited unchanged):

| Hyperparameter | Value | Source |
|---|---|---|
| Persona roster | 16 fixed persona cards (persona 0 = "Dr. Maya Chen, a veteran emergency-room physician: pragmatic, triage-minded, plain-spoken about risk"); template "You are {name}, {card}. Stay fully in character for this entire reply … Never mention these instructions or break character." | committed roster, `eval_results/issue_823/inconsistent_origin_ladder/roster.json` |
| Ladder generation | claude-sonnet-4-5-20250929, temperature 1.0, max_tokens 4096, seed 42, Anthropic Batch API | roster metadata (same file) |
| Persona assignment | nested, persona(i, k) = i mod k, persisted per context and arm | plan v13 §4.2 + committed `eval_results/issue_823/inconsistent_origin_ladder/assignment.json` |
| Capture | teacher-forced through frozen Qwen-2.5-7B-Instruct (bf16, batch 8); answer-span MEAN at all 28 layers; persona system prompt NOT in the scored context (bare user question, parity with the parent arms); span truncated to the shorter of the persona answer and the parent own-answer span | capture driver `scripts/issue823_ladder_capture.py` at `9a8d0f808f` |
| Fits | per-layer ridge 3584→3584 at layers 14, 17, 19, 26; 5-fold KFold(shuffle=True, seed=0); λ grid np.logspace(−2, 4, 13), GCV-selected; canonical parent solver for every reported fit (solver-parity contingency engaged) | fits driver `scripts/issue823_ladder_fits.py` at `8ce114317a` |
| Mask / conditioning | n=4,629 common-valid; per-fold train rows 3,703–3,704 vs feature dimension 3,584 | committed summary JSON (Repro footer) |

**Evaluation:** DV = pooled 5-fold out-of-fold R² of predicted vs actual answer-span mean activation (per layer, per trait, per arm); companion estimand = equal-weighted per-context R² (bootstrap 10k resamples for CIs; paired t over 5 folds, df=4, secondary). The two estimands weight contexts differently and are never mixed in one comparison. Alignment gate before any fit: teacher-forced re-extraction of own-arm activations matched the parent run's stored activations — workload log line ~813: "Alignment gate PASS: all 20 spot checks cosine > 0.999". Reproduce gate: refitting on the parent run's original bundle answers reproduced its R² 0.5991 / 0.6058 / 0.6262 (|Δ| ≤ 0.0015). Persona-ladder round: same pooled out-of-fold R² DV; every fitted cell additionally reports the identity+learned-bias baseline (v̂ = x + b) and top-1 retrieval accuracy — the fraction of held-out contexts whose true answer profile is the nearest neighbour of the prediction (cosine and euclidean; chance ≈ 0.0011 at pool size ≈ 926) — per the standing mapping-metrics rule. The single-persona minus sixteen-persona contrast carries a bootstrap 95% CI (10,000 draws, seed 0). Round gates: refitting the two anchors on the parent 4,998-context mask reproduced the promoted values to |Δ| ≤ 1e-7 in all six cells; the solver-parity gate comparing the fast-path solver against the canonical one FAILED on cuda AND on CPU-float64 (out-of-fold R² deviation up to 3.6e-4 against the 1e-4 tolerance, both solvers selecting λ = 0.01 on every probed slice), engaging the plan's contingency — every reported fit uses the canonical solver, and the fast-path-dependent minimum-contexts battery is withheld. Estimator-diagnostics gap, carried explicitly: the canonical-contingency solver did not persist per-fold selected λ / degrees of freedom (the only λ evidence is the six parity slices, all at the grid's bottom edge, 0.01), and the degrees-of-freedom-cap sensitivity read did not run although the cap was bindable at the realized mask — the anchor R² reads below sit next to the estimator-degenerate regime and should be read with that in mind.

**Data extraction:** activations captured teacher-forced (one forward pass per context+answer) at all 28 layers; target v_s(x) = mean residual-stream activation over the answer span; predictor cx_last(x) = last-context-token activation. Answer-span truncation at max_model_len=8192 leaves 391 contexts (7.8%) with fewer than 10 plain-arm answer tokens; 28 plain + 75 distinct-style contexts have zero-span targets (kept, flagged — see per-context result). Answer-arm text statistics (full 5000-context means): own 320 tokens, plain 243, distinct-style 279; plain-vs-own text cosine (embedding) mean 0.489, 9.6% of pairs >0.8 — the arms are genuinely different texts, not paraphrases. Persona-ladder round: 16 per-persona tensor files (6.02 GB pair store) of answer-span mean activations; per-context residual and total sums of squares for every (arm × layer) cell committed at `eval_results/issue_823/inconsistent_origin_ladder/percontext_ladder.npz`.

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

Persona-ladder answer (same context 0, persona 0 "Dr. Maya Chen"; excerpt truncated — LMSYS-derived corpus, verify the full 242-word row at the pinned link): "Look, I've been in emergency medicine long enough to see the aftermath when someone's identity gets [...]" — in-character throughout. Cherry-picked: 1 of 14,996 records, chosen to match the parent worked example's context; all rows: [HF raw_completions/ladder](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/009b58fdcf3da303993695066870e29416fb9ef6/issue823_inconsistent_origin_ladder/raw_completions/ladder).

## Results

### Plain-style external answers retain 91–98% of own-answer refit R²; mismatched answers collapse to ≈0

Bars show pooled 5-fold out-of-fold refit R² at each trait's plan-pinned read-out layer, one bar per answer arm (own regenerated / external plain / external distinct-style / mismatched), error bars = fold SD.

![Refit R-squared by arm at read-out layers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4bfe5c769ec36cedd3886bc5c018f6d2f473115/figures/issue_823/fig1_refit_r2_by_arm.png)

> **Figure.** *Refit R² by answer arm at read-out layers.* Own 0.599/0.608/0.626 (evil L14 / sycophancy L26 / hallucination L17); external plain 0.585/0.556/0.591 (97.6/91.4/94.4% retention); external distinct-style 0.473/0.468/0.506; mismatched 0.004/−0.008/0.007. Fold SDs ≤0.023 (mismatched ≤0.003). n=4998.

The map is content-indexed: an answer the model never produced supports nearly the full R²; a fluent-but-wrong answer supports none. Verdicts: sycophancy meets the own-answer-advantage rule (own−plain gap 0.052, above the 0.05 threshold; p_bonf=0.001); evil and hallucination show only the graded content-match ordering (gaps 0.014, p_bonf=0.36, and 0.035, p_bonf=0.014). The planned 5th arm (identity baseline) was computed in a follow-up round (final result below); the planned shuffle-null was replaced by the mismatched arm — a stricter null with real answer statistics (parent shuffle floor R²≈0.12 vs ≈0.005 here), so the collapse claim is conservative.

### The own-vs-plain increment is layer-structured; each fitted map is style-specific under transfer

Solid curves: refit R² across all 28 layers per arm; dashed curves: transfer R² of the own-answer-fitted map scored on the other arms' targets; read-out layers marked.

![Per-layer refit and transfer R-squared](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4bfe5c769ec36cedd3886bc5c018f6d2f473115/figures/issue_823/fig2_per_layer_refit_transfer.png)

> **Figure.** *Per-layer refit (solid) and own-map transfer (dashed) R².* The own-minus-plain gap is U-shaped, peaking exactly at L26 (0.052; 0.001 at L27). Transfer to plain 0.451–0.461; to distinct-style −0.070..+0.050; to mismatched −0.65..−0.80.

The sycophancy own-advantage verdict depends on L26 being both the plan-pinned layer and the peak of the own-minus-plain gap — one layer later the increment vanishes, so "predicts own output over external" is a narrow-band effect. Transfer separates content from style: plain answers reuse the own-fitted map at 0.45+, distinct-style answers (77–81% retention under refit) get ≈0 transfer — style shifts the target subspace enough to force a refit. An alternative reading — that style rather than content carries part of the own-answer increment — survives: a length-matched sweep (length-difference cuts 10→200) moves the sycophancy gap across 0.048–0.053, straddling the 0.05 threshold.

### Per-context R² distributions confirm the arm ordering context-by-context, not via a few outliers

ECDFs of equal-weighted per-context R² (n=4998) per arm at read-out layers; the table gives both estimands side by side.

![Per-context R-squared ECDFs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4bfe5c769ec36cedd3886bc5c018f6d2f473115/figures/issue_823/fig3_per_context_r2_ecdf.png)

> **Figure.** *Per-context R² ECDFs by arm.* Medians — own 0.55/0.55/0.58, plain 0.52/0.49/0.54, distinct-style 0.48/0.43/0.51, mismatched 0.01/−0.01/0.01 (evil/sycophancy/hallucination). Whole-distribution shifts, no outlier-driven mass.

| Trait (read-out) | Pooled Δ(own−plain) | Per-context Δ 95% CI | Pooled plain−mismatch | Per-context 95% CI |
|---|---|---|---|---|
| evil (L14) | 0.0142 | [0.0175, 0.0270] | 0.5807 | [0.5143, 0.5286] |
| sycophancy (L26) | 0.0521 | [0.0504, 0.0615] | 0.5641 | [0.5055, 0.5211] |
| hallucination (L17) | 0.0349 | [0.0404, 0.0506] | 0.5845 | [0.5144, 0.5308] |

Pooled (weighted by each context's share of the total sum of squares) and per-context (equal-weighted) estimands differ by construction — evil's pooled gap sitting below its per-context CI is an estimand difference, not an inconsistency. Caveats carried: the activation-cosine diagnostic (plain-vs-own target cosine mean 0.92–0.95) includes degenerate zero-span rows (min = 0.0 for all traits), so the mean is not ordinary target similarity for every context; per-context R² correlates weakly with answer length (Spearman 0.056–0.108, rescued inline — the planned length diagnostic was missing from the artifact); the planned identity baseline was computed in a follow-up round (final result below).

### The own-answer profile predicts the plain external profile better than the context does: content overlap alone accounts for the plain-arm retention

Bars: pooled 5-fold out-of-fold R² at each read-out layer for the context→plain refit versus ridge maps from the own-answer profile to the plain, distinct-style, and mismatched profiles; curves: own-profile→plain versus the context→plain refit and the mismatched floor across the 11-layer grid (error bars = fold SD).

![Identity baseline vs context refit](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5b159ab9b214908979566800048cbc82feec9738/figures/issue_823/fig4_identity_baseline.png)

> **Figure.** *Identity baseline (follow-up round).* At read-out layers, own-profile→plain R² 0.686/0.671/0.688 (evil/sycophancy/hallucination) vs context→plain refit 0.585/0.556/0.591; own-profile→distinct-style 0.530/0.525/0.548; own-profile→mismatched −0.010..−0.002. Per layer, own-profile→plain (0.580–0.712) exceeds the context refit (0.394–0.654) at every grid layer. n=4998.

This planned identity baseline was computed in a zero-GPU follow-up round, reusing each refit's solver, folds, and context mask. A content-matched answer profile carries more information about the plain-arm profile than the context does: the own-answer profile predicts it at R² 0.671–0.688, above the context→plain refit (0.556–0.591; fold SDs ≤0.012 against gaps ≥0.08), while mismatched targets stay at −0.021..+0.002 grid-wide.

The retention headline therefore needs no context-side self-generation information beyond what the answer content carries — a decomposition of the retention, not a claim the context carries nothing. Style specificity is consistent: own-profile→distinct-style reaches only 0.525–0.548.

### Map quality declines monotonically as answers spread across more personas; the minimum-contexts question went untested

Pooled 5-fold held-out R² per persona-count arm at the three read-out layers (top band), with the own-answer and plain-external anchor refits drawn at their true values on a broken axis (bottom band); error bars = per-context bootstrap 95% CIs.

![Ladder refit R-squared versus persona count](https://raw.githubusercontent.com/superkaiba/explore-persona-space/16f0ebeed1a94f0a0c1a14c6fb1b85c2ac4d931a/figures/issue_823/ladder_fig1_r2_vs_k.png)

> **Figure.** *Refit R² falls monotonically with persona count at every read-out layer.* One persona 0.501/0.483/0.516 (layers 14/26/17) down to sixteen personas 0.345/0.281/0.366; rank correlation −1.0 per layer. Anchor refits on this round's mask sit at −8.4 to −10.8 (dissociation result below). n=4,629.

| Pooled held-out R² | Layer 14 (evil) | Layer 26 (sycophancy) | Layer 17 (hallucination) |
|---|---|---|---|
| one persona (k=1) | 0.501 | 0.483 | 0.516 |
| two personas (k=2) | 0.445 | 0.462 | 0.471 |
| four personas (k=4) | 0.401 | 0.385 | 0.431 |
| eight personas (k=8) | 0.354 | 0.302 | 0.377 |
| sixteen personas (k=16) | 0.345 | 0.281 | 0.366 |
| drop, k=1 minus k=16 | 0.156 | 0.202 | 0.150 |
| drop, mean over the three layers (bootstrap 95% CI, 10,000 draws) | 0.169 [0.160, 0.179] | | |

The decline is monotone at every layer, and the plan's decision rule labels it "degrades" (the answer-distinctness predicate passed, so the label is interpretable); retrieval falls in lockstep (top-1 accuracy 0.372 at one persona to 0.169 at sixteen, layer 14).

Coverage caveat: this answers only the persona-count half of the round's question. The minimum-contexts n-ladder — with its per-persona, matched-sample-size, and same-contexts controls — was withheld under the solver-parity contingency, and its dimension-boundary rung was unrealizable at the realized mask anyway (train rows 3,336 < 3,584), so no minimum-contexts read exists in any figure or artifact. The next result bounds how much of the observed decline is mechanical.

### The implied mechanical mixture penalty accounts for essentially all of the persona-count decline

Top: the observed R² drop versus the one-persona arm at each persona count, against the penalty a single fixed map mechanically incurs from between-persona target variance; bottom: pooled R² under each arm's own denominator versus a fixed one-persona denominator.

![Observed drop versus implied mixture penalty](https://raw.githubusercontent.com/superkaiba/explore-persona-space/16f0ebeed1a94f0a0c1a14c6fb1b85c2ac4d931a/figures/issue_823/ladder_fig4_mixture_floor.png)

> **Figure.** *The observed drop tracks the implied mechanical penalty.* At sixteen personas: implied 0.171 vs observed 0.156 (layer 14), 0.192 vs 0.202 (layer 26), 0.160 vs 0.150 (layer 17). The fixed-denominator re-read falls more steeply (0.272 vs 0.345 at layer 14).

Mixing k persona-specific answer distributions over the same contexts adds between-persona target variance that no single deterministic map can capture, whatever it learns. The penalty implied by the measured between-persona mean shifts matches the observed drop within 0.015 at every read-out layer, at every rung of the ladder.

So "map quality degrades as answer origins become inconsistent" is true as measured, but the decline is largely what mixed targets mechanically force — not evidence that the underlying context→content association weakens or that the map becomes harder to learn. The fixed-denominator re-read falling more steeply confirms that part of the own-denominator curve's flattening is denominator inflation rather than better fit.

### New-mask anchor refits collapse in R² yet retrieve at roughly 200× chance: the two mandatory reads dissociate

Top: top-1 retrieval accuracy per persona-count arm (cosine and euclidean) with the anchor arms' retrieval as horizontal bands and the chance line; bottom: the identity+learned-bias baseline R² per arm with the anchors' baselines as bands.

![Retrieval accuracy and identity baseline across arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/16f0ebeed1a94f0a0c1a14c6fb1b85c2ac4d931a/figures/issue_823/ladder_fig2_retrieval_identity.png)

> **Figure.** *Anchors keep far-above-chance retrieval while their refit R² is catastrophic.* Anchor top-1 accuracy 0.210–0.256 vs chance 0.0011; fitted ladder maps beat the identity+bias baseline by 1.3–4.0 R² units, while the anchor refits fall roughly 8 units below their own baseline.

| Read (layers 14 / 26 / 17) | Own-answer anchor | Plain-external anchor |
|---|---|---|
| pooled refit R² | −10.00 / −10.83 / −8.95 | −10.04 / −10.70 / −8.45 |
| identity+bias baseline R² | −0.58 / −2.52 / −0.71 | −0.51 / −2.27 / −0.62 |
| top-1 retrieval accuracy, cosine (chance 0.0011) | 0.224 / 0.256 / 0.212 | 0.218 / 0.253 / 0.210 |
| contexts below per-context R² of −2 | 78.0% / 81.4% / 78.2% | 78.9% / 81.5% / 78.3% |

The anchor refits were planned as ceilings; realized, they instead show why this project reports both reads: R² alone would call these maps worthless, yet their predictions still identify the correct context's answer profile at roughly 200× chance — the fits preserve neighbourhood structure while being badly mis-scaled or mis-centered. The parent-mask reproduce gate passed exactly, so the promoted parent numbers above stand unchanged; the collapse is specific to this round's 4,629-row mask, where per-fold train rows (3,703–3,704) sit only 119 above the 3,584-dim feature space and the only persisted λ evidence is six parity slices at the grid's bottom edge. Read the anchor R² values as conditioning-dependent estimator behaviour, not as a property of the maps.

### Per-context distributions: the persona-count ordering is distribution-wide and the anchor collapse is not outlier-driven

ECDFs of equal-weighted per-context R² (clipped at −2 for display) for the five persona-count arms and the two anchor refits, at each read-out layer; n=4,629.

![Per-context R-squared ECDFs for the persona ladder](https://raw.githubusercontent.com/superkaiba/explore-persona-space/16f0ebeed1a94f0a0c1a14c6fb1b85c2ac4d931a/figures/issue_823/ladder_fig3_percontext_ecdf.png)

> **Figure.** *Whole-distribution shifts, no outlier mass.* Per-context medians: one persona 0.45/0.43/0.48, sixteen personas 0.32/0.25/0.34 (layers 14/26/17); anchors −7.1 to −8.7 with 78–81% of contexts below −2.

The persona-count curves shift together across the whole distribution, so the ordering holds context-by-context rather than through a few badly-fit contexts. The anchor curves already reach ~0.8 at the −2 clip: the collapse in the previous result is distribution-wide — most contexts individually read as badly mis-predicted in scale, even though each prediction still ranks its true profile first among ~926 held-out candidates far above chance.

---

**Repro:** GCP a2-ultragpu-1g (1× A100-80), ~8 GPU-h (phases 0.5–3) + CPU phase 4 (ridge fits; dedupe 3780→700 fits, canonical solver — a Gram-eigh fast path FAILED full-size parity and was reverted). Code `d68455ce60` (branch issue-823); figures 1–3 on main @ `e4bfe5c769ec36cedd3886bc5c018f6d2f473115`, figure 4 @ `5b159ab9b214908979566800048cbc82feec9738`. Identity-baseline follow-up (0 GPU-h, VM CPU): `eval_results/issue_823/identity_baseline.json` + `identity_baseline_units.jsonl` (branch issue-823), script `scripts/issue823_identity_baseline.py`, code `c9e759b318` — the JSON's embedded `git_commit` records the parent `397f879b5d` (the script was committed after the run); `c9e759b318` is the authoritative code commit. Data: `eval_results/issue_823/` (git: `ridge_r2_by_arm.json`, `validity_diagnostics.json`); raw completions + tensors + workload log: [HF issue823_own_vs_external/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8039d15f30deb845765cbb24d9cdb8708a5e7b0f/issue823_own_vs_external) (`raw_completions/`, `analysis_tensors/`, `logs/issue-823-workload.log` — alignment-gate PASS at line ~813, reproduce-gate PASS at lines 4143–4145 after two retried pre-stage RuntimeErrors at 4114/4126). Persona-ladder follow-up round (2026-08-20): generation via the Anthropic Batch API (14,996 records, 0 GPU); capture + fits on RunPod `pod-823-ladder` (1× H100, ~3 h wall including one designed abort + relaunch; measured fit cost 89 s per layer-fold unit, 20 units). Round code: generation + capture `9a8d0f808f`, fits (post-fix) `8ce114317a`, figures + aggregation `8762977adc`; round artifacts committed at `16f0ebeed1` (branch issue-823): `eval_results/issue_823/inconsistent_origin_ladder/` (aggregated summary `ladder_analysis_summary.json`, per-cell fits, baselines, distinctness checks, the zero-fit withheld-record for the minimum-contexts battery, per-context arrays) and the four ladder figures under `figures/issue_823/`. Round data on HF (pinned): [issue823_inconsistent_origin_ladder/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/009b58fdcf3da303993695066870e29416fb9ef6/issue823_inconsistent_origin_ladder) — `raw_completions/ladder` (19 files: 16 per-persona answer files + assignment + roster + completion sentinel), `analysis_tensors` (18 files, 6.02 GB pair store), `logs/fits` (7 files), `logs/fits_resume_state` (per-layer fit checkpoints + producer sidecars). Reused round inputs at the parent pin `8039d15f30de` (parent context vectors, own/plain tensors, span lengths) — fit: same 5,000-context bundle and capture rig, verified by the parent-mask reproduce gate (|Δ| ≤ 1e-7).

**Context:** created 2026-07-01, run 2026-07-02; zero-GPU identity-baseline follow-up (free analysis on existing tensors, no new data) folded 2026-07-02; same-issue follow-up round `inconsistent-origin-persona-ladder` (source: user-chat; scope posted 2026-08-19, run 2026-08-20) folded 2026-08-20. Child of #722 (context pool), method parent #779 (map + read-out layers). Originating prompt (verbatim):

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

Follow-up-round originating prompt (verbatim): `let's run an experiment to test if it degrades when origins are inconsistent. train mapping on: - lmsys with answers from single persona (promtped into Claude) - lmsys with answers from multiple personas (prompted into Claude) -- prompts matched to above. see how mapping quality scales with # of personas. minimum number of contexts to have well-posed mapping`

<!-- verifier WARN acknowledged: 6 Takeaways bullets exceed the 30-word bullet cap — each bullet carries all three traits' numbers or a full follow-up round's synthesis (numbers-first density is deliberate); several result sections exceed the 120-word per-result prose cap and the total-prose budget is exceeded — the four-arm × three-trait parent design plus two folded follow-up rounds (identity baseline; persona ladder) carry dense per-arm numbers; goal: frontmatter field is preserved from the pre-promotion body by set-body -->
