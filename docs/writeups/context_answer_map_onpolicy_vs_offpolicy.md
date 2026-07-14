# Result: The context→answer mapping reads answer content, not the model's own policy — off-policy answers are almost as predictable, everywhere we looked
<!-- report-v1 -->

<!-- Cross-experiment writeup covering #823 (on- vs off-policy arms), #952
     (per-position / prefix / divergence deep-dive), and the #825 raw-text +
     base-model arms. Drafted 2026-07-14 from the promoted-pending clean-result
     bodies; statuses at drafting time: #823, #952 awaiting_promotion; #825
     followups_running. -->

## Motivation

- An earlier experiment ([#779](https://eps.superkaiba.com/tasks/779)) found a linear map from a single context's last-token activation $c_{last}(x)$ to that answer's mean activation $v(x)$: test R² 0.705 [0.691, 0.719] at the validation-selected layer 19, on 5000 LMSYS contexts with on-policy answers.
- I wanted to test what that map is actually reading: is it predicting **what this model will say** (its policy), or just **what an answer to this context looks like** (content)? If it predicts another model's answers equally well, it is not a privileged readout of the model's own future behavior.
- Three sub-questions, run as three experiments on the same rig:
    - Does the map hold on answers the model never generated, and can one arm's fitted map predict another arm's? ([#823](https://eps.superkaiba.com/tasks/823))
    - Is any own-answer advantage concentrated at particular answer positions (hypothesis: off-policy text is most surprising at the start, then the model "gets used to it"), and does the map fail specifically on queries where Qwen and Claude genuinely behave differently? ([#952](https://eps.superkaiba.com/tasks/952))
    - Is there a similar linear mapping on plain random text, and in the pretrained base model? ([#825](https://eps.superkaiba.com/tasks/825))

## TLDR

- **The map is content-indexed, not policy-indexed:**
    - Claude-written plain-style answers retain 91–98% of the own-answer refit R² (0.585/0.556/0.591 vs 0.599/0.608/0.626 at the three read-out layers); shuffled-pairing answers collapse to R² ≈ 0 (±0.01), so this is not a trivial-baseline effect
    - the own-answer profile predicts the plain external profile (R² 0.671–0.688) *better than the context does* (0.556–0.591) — content overlap alone accounts for the retention
- **The own-answer advantage is small (~0.02 R²) and position-uniform:**
    - first-16 vs last-16 position contrast is inside ±0.03 at every captured read-out layer (5-fold cross-fit, n = 3,188 matched contexts)
    - my "off-policy is more surprising at the start" hypothesis was wrong — the ordering actually *flips* at token 1 (own 0.50 < plain 0.52 < eccentric-style 0.55)
- **Absorbing 16 of the answer's own tokens closes the plain-external gap at mid layers** (84–118% of the per-layer gap at layers 17–23) — but the eccentric-style gap never closes (it widens at every layer except 20)
- **The map does not get worse where the two models genuinely diverge:** on 41 generation-verified Qwen-vs-Claude divergence pairs with entity-swapped controls, the divergence-specific external penalty is −0.005 (p = 0.64; margin 0.05; detection ceiling 0.887, so it was attainable)
- **The fitted maps themselves are largely shared for plain text but style-specific:** the own-fitted map transfers to plain external targets at R² 0.45–0.46 (refit ceiling 0.556–0.591), while the eccentric-style arm gets ≈ 0 transfer (−0.07..+0.05) despite refitting at 77–81%
- **On random raw text the strong map mostly doesn't exist:** a separator→next-span control on WikiText fits a linear map, but it transfers only 5.7% (base) / 10.9% (instruct) of the chat map's information — while the chat map itself already exists in the pretrained base model at 87.3% of instruct strength (R² 0.588 vs 0.673, layer 19)
- Scope: all of this is teacher-forced activation predictability on frozen models — a representation-level claim, not an on-policy behavioral read-out

## Methodology

- **Model:** Qwen-2.5-7B-Instruct, frozen (Results 7–7.5 add pretrained Qwen/Qwen2.5-7B)
- **Datasets:**
    - **LMSYS four-arm pool**: 5000 single-turn prompts from [`lmsys/lmsys-chat-1m`](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) (pinned revision), 4,998 common-valid after API-failure drops (#952 further excludes 103 empty-external rows → 4,920). Four answer arms per context:
        - **own (regenerated)** — Qwen resamples each context (vLLM, T = 1.0, top_p 0.95, max_tokens 1024, seed 42, no system prompt)
        - **external plain** — `claude-sonnet-4-5-20250929`, no system prompt (T = 1.0, max_tokens 1024)
        - **external distinct-style** — same Claude model under "Respond in an unusual, stylistically eccentric way — use unexpected structure, mixed register, and non-standard formatting.", with the instruction **stripped before teacher-forcing** so the scored context is identical across arms
        - **mismatched** — own answers reassigned by a fixed-point-free derangement (seed 42): real answer text, zero context relevance
        - Example (context 1230):
            - Query: `which is the capital city of german`
            - Own: `The capital city of Germany is Berlin. It has been the capital since 1990, when Germany was reunified and Berlin replaced Bonn as the seat of government.`
            - External plain: `The capital city of Germany is **Berlin**.\n\nBerlin has been the capital of reunified Germany since 1990, following the fall of the Berlin Wall and German reunification. ...`
            - External distinct-style: `# **BERLIN** 🏛️\n\n*the capital city of Germany*\n\n---\n\n**Historical note:** Berlin became the capital of reunified Germany in 1990, though the government didn't fully relocate from Bonn until 1999. ...`
        - Full pool: https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8039d15f30deb845765cbb24d9cdb8708a5e7b0f/issue823_own_vs_external/raw_completions
    - **Qwen-vs-Claude divergence bank**: 4 planned quirk categories (china-politics, model-identity, refusal-boundary, style-format) built from the CCP-sensitive-prompts dataset, in-repo query banks, and identity/style templates; every divergent query paired with an entity-swapped same-template control. Candidates generation-verified: both models answer, a graded judge (`claude-sonnet-4-5-20250929`, 0–100, reason-then-score, 5 draws, malformed draws dropped) scores actual divergence. 230 candidate pairs → 457 queries judged → **41 pairs kept in 2 categories** (model-identity 20, style-format 21; china-politics missed the 20-pair floor at 18, refusal-boundary collapsed to 2). Kept pairs + judge outputs: https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5b62649cefb34902fd630f21630164e8d1d99764/issue952_position_divergence/eval_results
    - **WikiText-103 separator pairs** (random-text control): 3,600 pairs over 600 articles — a sentence-final separator token's activation → the following-span (8–256 tokens) mean activation; variants swap the span text for each model's own greedy or sampled raw-text continuations of the same articles' prefixes
    - **Splits:**
        - #823: 5-fold KFold (shuffle, seed 0), GCV-selected λ, n = 4,998
        - #952: 2,952 train / 984 validation / 984 test (rng seed 952) — validation selects the read-out layer + per-slot λ, the disjoint test split carries every comparison; a final round replaces the single holdout with a **5-fold cross-fit of the full pool** (each context tested exactly once; layer 20 re-derived independently in every fold; n = 3,188 span-matched contexts; 10,000-draw paired bootstrap + sign-flip nulls)
        - #825 separator control: 5 article-group folds (no leakage across an article), group-blocked shuffle nulls, 1,000-draw group bootstrap
- **Computed quantities:**
    - $c_{last}(x)$: last context-token activation (the #779-validated context summary)
    - $v(x)$: mean activation over the answer span (default target)
    - per-position targets $z_t(x)$: 42 slots — first 16 tokens, 10 relative-position deciles, last 16 tokens with the span extended through `<|im_end|>` and its trailing newline (the last token before a next user turn would begin), template slots split out
    - prefix-conditioned predictor: mean activation over context + the first t answer tokens (t ∈ {1, 2, 4, 8, 16, 32, 64, 128}; t = 0 is the $c_{last}$ baseline), predicting the remainder (individual / mean-pooled / max-pooled positions > t)
- **Predictors / conditions:** ridge only (3584→3584, standardize-X / center-Y, λ ∈ logspace(−2, 4, 13), validation- or GCV-selected), per arm × layer (28-layer grid in #823; captured band {2, 6, 10, 14, 17, 20, 23, 26} in #952). REFIT fits each arm from scratch; TRANSFER scores the own-fitted map on the other arms' targets.
    - **Baselines:**
        - one worry is the map trivially predicts generic answer statistics; test: the **mismatched arm** (fluent real answers, wrong context) — also the positive control for the prefix pathway
        - one worry is the plain-arm retention needs context-side self-generation information; test: the **identity baseline** — a ridge map from the *own-answer profile* (not the context) to each other arm's profile, same solver/folds/mask
        - one worry is any raw-text map is just ridge finding structure anywhere; test: **rotated random-projection controls** + group-blocked shuffle nulls (#825), train-mean prediction (R² = 0 by construction)
    - **Sanity checks:** shuffled-pairing fits; alignment gates (re-extracted activations vs stored, cos > 0.999); exact reproduction gates between rounds (e.g. 672/672 cells at ΔR² = 0.0)
- **Metrics:** held-out reconstruction R², both **pooled** (variance-weighted) and **equal-weighted per-context** — the two estimands weight contexts differently and are never mixed in one comparison. R² over cosine because all $v(x)$ share a large common component (predicting the mean answer profile already scores cosine ~0.98). Decision margins registered in advance: ±0.03 (position contrast), 0.02 (prefix closure), 0.05 (divergence-bank drop difference); Bonferroni/Holm across traits/layers; 10,000-draw bootstrap CIs + sign-flip permutation nulls.

## Results

### _Result 1: Plain-style external answers retain 91–98% of the own-answer refit R²; shuffled answers collapse it to ≈0_

I refit the identical ridge harness per answer arm and trait at the plan-pinned read-out layers (evil L14, sycophancy L26, hallucination L17), n = 4,998. Bars show pooled 5-fold out-of-fold refit R² per arm; error bars are fold SDs.

**Plot: Refit R² by answer arm at read-out layers**

![Refit R² by arm at read-out layers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4bfe5c769ec36cedd3886bc5c018f6d2f473115/figures/issue_823/fig1_refit_r2_by_arm.png)

**Takeaways:**

- An answer the model never produced supports nearly the full R²: plain Claude answers refit at 0.585 / 0.556 / 0.591 vs own 0.599 / 0.608 / 0.626 (97.6% / 91.4% / 94.4% retention); eccentric-style still refits at 0.468–0.506 (77–81%)
- A fluent-but-wrong answer supports none of it: mismatched R² −0.008..+0.007 (fold SDs ≤ 0.003) — the external answer predicts ~11–41× more pooled variance than a wrong answer for the same context, so retention is not a trivial baseline
- The own-answer increment is real but small, and only sycophancy crosses the 0.05 decision threshold (gap 0.052, p_bonf = 0.001; evil 0.014, hallucination 0.035) — and a length-matched sweep straddles the threshold (0.048–0.053), so part of the increment is length/style covariates

### _Result 2: The maps are largely one shared map for plain text, but each fitted map is style-specific_

I scored the own-fitted map directly on the other arms' targets (TRANSFER), across all 28 layers, next to each arm's own refit.

**Plot: Per-layer refit (solid) and own-map transfer (dashed) R²**

![Per-layer refit and transfer R²](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4bfe5c769ec36cedd3886bc5c018f6d2f473115/figures/issue_823/fig2_per_layer_refit_transfer.png)

**Takeaways:**

- The own-fitted map transfers to plain external targets at R² 0.451–0.461 against a 0.556–0.591 refit ceiling — the on-policy and off-policy maps are substantially the same object for plain text
- Style breaks the sharing: distinct-style transfer is ≈ 0 (−0.070..+0.050) even though the style arm refits at 77–81% — the map family is content-indexed, but each fitted map is style-specific (style shifts the target subspace enough to force a refit)
- Transfer onto mismatched targets is strongly negative (−0.65..−0.80), as it should be
- The sycophancy own-advantage is a narrow-band effect: the own-minus-plain gap peaks exactly at L26 (0.052) and is 0.001 one layer later

### _Result 3: Content overlap alone accounts for the plain-arm retention_

The planned identity baseline (zero-GPU follow-up round): a ridge map from the own-answer profile to each other arm's profile — same solver, λ grid, folds, and context mask as every refit — compared against the context→arm refit.

**Plot: Identity baseline vs context refit**

![Identity baseline vs context refit](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5b159ab9b214908979566800048cbc82feec9738/figures/issue_823/fig4_identity_baseline.png)

**Takeaways:**

- A content-matched answer profile carries more information about the plain external profile than the context does: own-profile→plain R² 0.686 / 0.671 / 0.688 vs context→plain 0.585 / 0.556 / 0.591, and it wins at every layer of the 11-layer grid (0.580–0.712 vs 0.394–0.654)
- Mismatched targets stay at the floor grid-wide (−0.021..+0.002), and own-profile→distinct-style reaches only 0.525–0.548 — style specificity again
- So the retention headline needs no context-side self-generation information beyond what the answer content carries — this is a decomposition of the retention, not a claim the context carries nothing

### _Result 4: The own-answer advantage is small and position-uniform — the "more surprising at the start" hypothesis is falsified_

I fit per-position maps $c_{last}(x) → z_t(x)$ at 42 answer-position slots per arm (first 16 tokens, deciles, last 16 through `<|im_end|>` + newline), and registered the first-16 minus last-16 own-vs-external contrast against a ±0.03 equivalence margin — first on a single holdout, then re-read as a 5-fold cross-fit of the full pool at layers {14, 20, 23, 26} (n = 3,188 matched contexts, each tested once).

**Plot: Held-out R² per answer position, four arms, layer 20**

![Held-out test pooled R-squared per answer position for four answer arms at layer 20](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9f25d29d5577b3cbe10984594325092674848fa/figures/issue_952/hero1_position_r2.png)

**Takeaways:**

- All arms share a U-profile and the plain-external curve tracks the own curve ~0.02 below it at every content position: the position contrast is −0.010..+0.002 across the four decision layers, every 95% interval inside the ±0.03 margin — the advantage is position-uniform, not front-loaded
- The ordering *flips* at the very first token (own 0.50 < plain 0.52 < style 0.55) — external openings are the most surprising tokens yet the most predictable activations, plausibly because stereotyped answer openers occupy a low-variance activation region
- The gap itself is small and grows with depth: 0.009 at L14 → 0.019 at L20 → 0.035 at L26 (cross-fitted)
- Cross-fit half-widths shrank ~2.3× over the single split and every round-1 call survived; band-scoped caveat — early layers sit at the predictability noise floor and layer 27 was not captured

### _Result 5: Sixteen absorbed answer tokens close the plain-external gap in the mid band; the style gap never closes_

I swept the predictor from $c_{last}$ (t = 0) to the mean activation over context + the first t answer tokens, predicting the identical remainder target on the common surviving subset (matched population, matched target), and registered closure of the own-minus-external gap against a 0.02 margin.

**Plot: Matched prefix-closure decision cells + closure curves + survivor attrition, layer 20**

![Matched prefix-closure bars with descriptive closure curves and survivor attrition panels](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9f25d29d5577b3cbe10984594325092674848fa/figures/issue_952/hero2_matched_prefix.png)

**Takeaways:**

- At layer 20 the plain gap goes +0.023 → −0.002 with 16 absorbed tokens (closure +0.025, interval excludes zero) — the "model gets used to the off-policy text" mechanism is real, in prefix-absorption form
- Cross-fitted band map: 84–118% of each layer's own gap closes at layers 17–23; affirmatively absent at L14 (interval top 0.009, under half the margin); L26 closes 39% of its gap (real — zero excluded — but below the 0.02 margin); L23 lands on the margin in both rounds
- The eccentric-style gap does NOT close: closure points −0.048..−0.008 (widening) at every layer except 20, and the gap is +0.045 after 16 tokens at L20 — style stays foreign no matter how much of it is absorbed
- The teacher-forced surprisal companion agrees on mechanism ([figure](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c51d2415a215c714801cd65b40157edcda3a6859/figures/issue_952/exp_surprisal_combined.png)): plain-text surprisal collapses 11.5 → 1.6 nats within 5 tokens (style 18.5 → 4.6, own flat at 0.4–0.6), and the style arm stays off-distribution longest — but surprisal *dissociates* from activation predictability at token 1, so the two are not the same quantity

### _Result 5.5: The prefix pathway is generic — a shuffled-context answer recovers ~96% of own-answer predictability once 128 of its own tokens are absorbed_

The mismatched arm is the positive control for the prefix pathway: its context carries zero information about the answer, so anything the prefix-conditioned predictor recovers comes from the absorbed answer tokens alone.

**Plot: Remainder R² per arm across prefix lengths, layer 20**

![Pooled-prefix remainder R-squared per arm across prefix lengths](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9f25d29d5577b3cbe10984594325092674848fa/figures/issue_952/exp_pooled_prefix.png)

**Takeaways:**

- The mismatched curve climbs from ≈ 0 to R² 0.351 at t = 128 — 95.6% of the own arm's 0.367 (layer 17) — clearing the registered 50% recovery criterion by a wide margin
- So most of what the prefix-conditioned map reads is carried by the answer's own unfolding text, not by the context; the context's marginal contribution is the small t = 0 gap of Results 4–5
- Caveat: the context-only leg leaks a little generic answer statistics at openings and template slots (mismatched position R² up to 0.11 at tokens 1–2), within ±0.033 from token 3 onward

### _Result 6: No divergence-specific external penalty where Qwen and Claude genuinely behave differently_

I evaluated the same pool-trained maps out of distribution on the 41 surviving generation-verified divergence pairs (each divergent query paired with an entity-swapped same-template control, e.g. a different country), and registered the per-context R² drop (control minus divergent) differenced between the plain-external and own maps against a 0.05 margin.

**Plot: Paired control-minus-divergent drops per bank category + per-pair scatter**

![Paired control-minus-divergent drop bars per bank category and per-pair scatter of raw drops](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9f25d29d5577b3cbe10984594325092674848fa/figures/issue_952/hero3_divergence.png)

**Takeaways:**

- Pooled over 41 pairs the divergence-specific external penalty is −0.005 (sign-flip p = 0.64; median −0.003), interval excluding the 0.05 margin — the off-policy map does not degrade where the two models' behaviors actually diverge, which is the worrying case for using the map as a behavior monitor
- Detection was attainable (ceiling 0.887 vs null band +0.028) and both maps transfer to the bank above the trivial floor (own 0.145, plain 0.137); the null is layer-stable across the band (descriptive)
- Scope honesty: only 2 of 4 planned categories survived the keep gates (model-identity 20, style-format 21; china-politics missed the 20-pair floor at 18, refusal-boundary collapsed to 2), the judge calibration inverted on the 40-probe set (gates made effectively vacuous, one-time adjustment), and ~18% of kept queries hit the 1,024-token generation cap — the bank read is weaker than designed
- Registered exploratory secondary: divergent-query predictions displace *toward* the own-answer activations (mean cosine +0.64) — confounded by ridge shrinkage toward the pool mean, so exploratory only

### _Result 7: On random raw text only a weak, model-shared next-span map exists — while the chat map itself already lives in the pretrained base model_

Separate question: is the strong map a chat/answer phenomenon or a generic property of text? The separator control fits the same ridge machinery on WikiText-103 — a sentence-final punctuation token's activation → the following-span mean — in both the pretrained base and the instruct model, then measures how much of the chat map's information the control map carries (recentered sep→chat transfer at layer 19).

**Plot: Base vs instruct separator-control read at layer 19**

![Base vs instruct separator-control bars at layer 19](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b5959c212812e358c667a8e955575822ccfc7075/figures/issue_825/base_sep_control_hero.png)

**Takeaways:**

- A linear separator→next-span map exists on plain article text, but it transfers only 5.7% (base) / 10.9% (instruct) of the chat map's information — far below the 0.5 specificity line ([transfer fractions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/89c68bf1085ed5dd4091691fa4884e76380a2083/figures/issue_825/sampled_sep_transfer_fractions.png)) — so the strong context→answer map is answer/chat-structure-specific, not generic next-text prediction
- The control map is model-shared (base↔instruct weight cosine 0.86–0.90), unlike the chat map, whose base↔instruct reads are consistent with a *rotated* read-out
- Meanwhile the chat map itself already exists in pretrained Qwen2.5-7B at 87.3% of instruct strength (held-out R² 0.588 vs 0.673 at layer 19) — post-training sharpens and rotates it, it does not create it

### _Result 7.5: ...and the apparent own-text advantage on raw continuations was a greedy-decoding artifact_

On the same 600 articles, the separator control was re-run with each model's own continuations as span text — first greedy (T = 0), then a sampled twin (T = 1.0, top_p 0.95, matching the chat arms' decoding), reading the position statistic D = (W_on − W_ex)/(C − W_ex) against frozen references.

**Plot: Sampled D per arm vs the greedy reference, both models**

![Sampled D per arm vs the round-7 greedy reference, both models](https://raw.githubusercontent.com/superkaiba/explore-persona-space/72f7e6233f0c8540a7c45a2af7b68c1dd3284104/figures/issue_825/sampled_sep_hero.png)

**Takeaways:**

- Greedy own-continuations looked meaningfully more predictable than exogenous article text (D 0.590 base / 0.428 instruct) — but the sampled twins collapse to D 0.031 / 0.086: at matched decoding, own vs external raw text look the same, consistent with everything above
- The read survives span-length matching (matched-vs-random contrast < 0.002), and averaging targets over 10 sampled draws buys only ~0.05 R² (16–23% of the base-instruct gap) — target sampling noise is a minor part of the story
- Lesson for the whole line: any "on-policy advantage" measured on greedy text needs a sampled-decoding control before it is believed

## Next steps:

- (running) #825 `naturalistic-single-turn` round — the single-turn cells in naturalistic (no chat template) format
- Split the implication by use case:
    - for the **fine-tuning-leakage predictor** (the theory paper's use), off-policy generalization is arguably the *required* property — a training corpus is off-policy text by construction, and a content-indexed map is exactly what can score it
    - for a **behavior monitor** ("read the pre-answer state, know what this model will do"), this line says the linear map is a plausibility reader, not a policy reader — look for reads that DO prefer the model's own behavior (logit-side / behavioral read-outs per #763/#810, richer-than-linear structure, or the style-specificity channel, which is the one place the map genuinely separated arms)
- Chase the post-fine-tuning link, where on-policy does bind: fine-tuning reshapes the map only for a taught fact (#722, LOW), trait-expressing training data makes the learned map *worse* (#779 follow-up line), and the taught-fact leakage chain rides specifically on on-policy taught-sentence emissions (#833) — reconcile these with the content-indexed picture
- Promote/park decisions on #823, #952, #825 (all still unpromoted)
