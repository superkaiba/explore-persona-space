---
title: The own-answer advantage in predicting answer activations from context is position-uniform
  at the selected read-out layer (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-07-03T23:33:32Z'
has_clean_result: false
parent_id: 823
origin_prompt: "# Motivation\n- We found in a previous experiment that there is an\
  \ almost as good linear mapping from context to off-policy answers as there is from\
  \ context to on-policy answers\n- If we are hoping to use this linear mapping as\
  \ some kind of prediction of the model's behavior then this is worrying, because\
  \ the off-policy text is **not representative of the model's actual behavior**\n\
  - We want to:\n    - see if this mapping holds even for queries where the 2 models\
  \ diverge alot\n    - do an in-depth analysis of this off-policy mapping and its\
  \ comparison to the on-policy mapping\n# Methodology\n- Find the experiment where\
  \ we did a matched on-policy and off-policy mapping on same contexts and queries\n\
  - We want to see if this mapping is the same **across individual tokens**\n    -\
  \ i.e.:\n        - from the context vector how much worse/better is the model at\
  \ predicting the first few activations for on-policy vs off-policy vs last few activations\
  \ (GO UP TO THE NEWLINE AFTER THE USER CHAT TEMPLATE -- right before the next user\
  \ turn would start generating)\n        - ideally this would go across the whole\
  \ answer but it might be hard because answers are of different lengths, so just\
  \ characterization of first 16 activations vs last 16 activations is good\n    \
  \    - hypothesis is that model can better predict first few activations for on-policy\
  \ (off-policy is more surprising) -- but then gets better and better at predicting\
  \ off-policy as it gets \"used\" to the style of the off-policy text\n        -\
  \ Ideally we also want to see if taking \"more\" tokens into the context vector\
  \ helps to predict better\n            - so sweep over tokens and regress to predict\
  \ all other tokens/mean/max pooled tokens (starting after current token)\n- We can\
  \ only use linear mappings\n- Always validation to select best layer/hyperparameters\
  \ and evaluation to select best method\n- We also want to see if this mapping is\
  \ specifically bad for queries where the model behaviors diverge:\n    - For Qwen\
  \ vs Claude this will probably be questions about China\n    - Search also deeply\
  \ for known differences/quirks with Claude and Qwen -> and do generation tests to\
  \ see if these quirks are truly different\n    - then compare the similar answers\
  \ to the different answers in terms of predictability (mapping always trained on\
  \ same pool)\n        - for the similar answers try to have a one mapping between\
  \ queries (e.g. instead of asking about China -- ask about another country) - so\
  \ we control for this\n\n[Design decisions confirmed in chat 2026-07-03: new child\
  \ task of #823; broad quirk taxonomy for the divergence bank (~4 categories, generation-verified,\
  \ matched same-template controls); all four #823 arms carried through the per-token\
  \ analysis.]"
workflow: v1
goal: 'On frozen Qwen-2.5-7B-Instruct with ridge-only maps and a train/validation/test
  split (validation selects layer + λ, a disjoint test split compares methods and
  arms), characterize where the linear context→answer-activation map differs between
  on-policy and off-policy answers, along three axes: (1) per-token-position predictability
  — fit h_t: c_last(x) → z_t^(a)(x) for answer positions t in the first-16 window,
  the last-16 window (span extended through `<|im_end|>` and its trailing newline
  — the last token before a next user turn would begin), and relative-position deciles,
  per arm a ∈ {own-regenerated, external-plain, external-distinct-style, mismatched}
  (the four #823 arms, completions reused) over the 4998-context LMSYS pool; (2) prefix-conditioned
  prediction — sweep the predictor to the realized-answer activation at position t
  (t ∈ {1,2,4,8,16,32,64,128}; t=0 = c_last baseline) predicting the individual /
  mean-pooled / max-pooled activations of positions > t, measuring how fast each arm''s
  remainder becomes predictable as prefix is absorbed; and (3) divergence-conditioned
  evaluation — evaluate the SAME pool-trained maps on a generation-verified Qwen-vs-Claude
  divergence query bank (≈4 quirk categories × ~40–60 queries, each with matched same-template
  entity-swapped controls), testing whether off-policy predictability fails specifically
  where the two models'' behaviors diverge while on-policy predictability holds.'
relates_to:
- spec-context-as-vector
- identity-contextual-vs-base
---
# The own-answer advantage in predicting answer activations from context is position-uniform at the selected read-out layer (MODERATE confidence)

<!-- clean-result-v4 -->
<!-- conciseness WARNs acknowledged: four result sections run 121-150 words (soft cap 120) and total prose exceeds the 800-word soft budget; the overage carries the layer-fragility, instrument-caveat, and control-carve-out qualifiers deliberately -->

## Takeaways

- **Predicting answer activations from context favors Qwen's own answers over Claude-written answers by only ~0.02 R², uniformly across positions at read-out layer 20** (first-vs-last-16 contrast +0.004, margin ±0.03).
- **Sixteen absorbed answer tokens fully close the plain-external gap at the layer-20 decision cell** (closure +0.025, interval excludes zero); the closure does not replicate at layer 17 (+0.014).
- **The distinct-style penalty does not close:** +0.044 after 16 absorbed tokens, +0.028 after 128; the style-minus-plain closure difference is −0.031, opposite the predicted direction.
- **On the surviving identity/style bank (41 pairs; 2 of 4 categories), no divergence-specific external penalty emerges at the 0.05 margin** (p = 0.64; ceiling 0.887 vs null band +0.028).
- **Binding caveats:** inverted judge calibration and ~18% cap-truncated bank queries dilute the bank read; single split, single capture; teacher-forced representation-level scope, not on-policy behavior prediction.

## Goal

- **This experiment in context:** The parent experiment ([#823](https://eps.superkaiba.com/tasks/823)) found a linear map from the last context-token activation to the answer-mean activation predicts Claude-written answers nearly as well as Qwen's own — plain-style external answers retained 91–98% of the own-answer refit R²; shuffled pairings scored ≈ 0. This run resolves that residual own-answer advantage on the same four answer conditions along three axes: per-token-position predictability, prefix-conditioned closure, and a divergence-conditioned evaluation on queries where the two models genuinely behave differently.
- **Broader narrative:** The context→answer-activation map is a candidate behavior predictor — reading what the model is about to say from its pre-answer state. If it predicts other models' text equally well, it captures content plausibility rather than this model's policy; locating where off-policy text becomes less predictable bounds what such linear predictors can claim.

## Methodology

**Design:** One frozen model (`Qwen/Qwen2.5-7B-Instruct`); 4,920 single-turn LMSYS-Chat-1M contexts split 2,952 train / 984 validation / 984 test (`numpy` rng seed 952). Four answer arms per context — own answer (regenerated), external plain (Claude), external distinct-style (Claude), mismatched (shuffled pairing) — reusing the parent completion set (production written out under Data extraction). Ridge maps are fit from the last context-token activation to per-position answer activations at 42 position slots (first 16 tokens, 10 relative-position deciles, last 16 tokens with the two turn-end template slots split out), and prefix-conditioned maps from the mean activation over context + the first t answer tokens to the remainder, at 8 prefix lengths. A 41-pair divergence bank (divergent query + entity-swapped control) evaluates the same maps out of distribution. The validation split selects the read-out layer and per-slot regularization; the disjoint test split carries every comparison. Relative to the parent, the manipulated variable is the dependent-variable granularity (per-position / per-prefix instead of answer-mean) plus the divergence conditioning; estimator family, arms, and pool are held fixed.

**Training:** **N/A — no model training.** Analysis-design constants (every load-bearing value copied from the committed run config / plan decision rationale):

| Parameter | Value | Source |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct`, frozen | project standard |
| Estimator | ridge; standardize-X / center-Y on train statistics; float64 solve | parent ridge harness |
| λ grid | `np.logspace(-2, 4, 13)`; λ selected per slot on validation, arm-shared (realized selections persisted in `validation_selection_matrix.json`) | parent grid; plan selection rule |
| Layer grid (capture) | {2, 6, 10, 14, 17, 20, 23, 26} | storage sizing; parent read-out band |
| Read-out layer | 20 — one layer for all position slots, argmax of validation pooled R² averaged over slots and arms | plan selection rule (user-pinned validation/test separation) |
| Prefix-map layers | {20, 17} | compute bound; fixed second point |
| Split | 2,952 / 984 / 984 train/val/test, rng(952) | plan sizing (CI width ≈ 0.02 vs margins 0.03–0.05) |
| Position slots | first 16 tokens; 10 relative deciles; last 16 tokens, span extended through `<\|im_end\|>` + trailing newline, 2 template slots split out | task formalization |
| Prefix lengths t | {1, 2, 4, 8, 16, 32, 64, 128}; a context enters a t-cell iff its span ≥ t+16; cell reporting floor n(test) ≥ 200 | plan sizing |
| Decision margins | 0.03 (position contrast) / 0.02 (prefix closure) / 0.05 (bank drop difference) | parent resolvable-gap scale |
| Bootstrap | 10,000 draws, rng(0), batched GEMM; 3-cell serial-oracle parity max diff 0.0 | project default; batched-draw recipe |
| Sign-flip null | 10,000 draws, rng(1) | plan |
| Capture precision | fp16 slot capture; equivalence gate vs the parent's fp32 means cos > 0.999 — worst min-cos 0.99957, 4,920/4,920 contexts per arm | run gate |
| Ridge parity gate | batched GPU shared-SVD solver vs serial float64 oracle; max relative diff 5.3e-14 | run gate |
| Judge | `claude-sonnet-4-5-20250929`; graded 0–100, anchored reason-then-score; 5 divergence + 3 refusal draws per query at temperature 1.0; malformed draws dropped, never coerced (7 + 3 dropped) | project judge rule |
| Bank keep gates | planned: divergence ≥ 60 and divergent-minus-control margin ≥ 25; calibration on 40 probes inverted (known-divergent median 14 vs known-similar 60) → one-time adjustment: keep ≥ 47, pair margin −23 (effectively vacuous) | plan calibration rule; realized values |
| Bank generation | Qwen: vLLM, temperature 1.0, top_p 0.95, max_tokens 1024, seed 42; Claude: `claude-sonnet-4-5-20250929`, temperature 1.0, max_tokens 1024 | parent arm recipe |
| Lexical companion gate | TF-idf cosine vs judge score, Spearman ≥ 0.3 floor; realized 0.310 | plan validation |

**Evaluation:** Three dependent variables, all teacher-forced representation-level by design (the inherited scope: claims are about the linear map on realized answer text, never on-policy behavior prediction). (1) *Per-position predictability* — held-out test pooled R² of the ridge map per arm per slot at the frozen (layer, λ); the position-profile decision statistic is the first-16 minus last-16-content-token gap contrast (own minus external), read against the 0.03 margin. (2) *Prefix-conditioned closure* — the decision statistic is a matched-population, matched-target contrast: on the common surviving subset intersected across compared arms, the own-minus-external gap predicting the identical remainder target, with no prefix vs after t prefix tokens; variable-population closure curves are descriptive only, and per-t survivor attrition is reported. (3) *Divergence-conditioned evaluation* — per kept bank pair, the per-context R² drop (control minus divergent) differenced between the external-plain and own maps, pooled over 41 pairs with median and 10%-trimmed companions; a sign-flip null over pair signs gives the p-value, and the null band is reported next to the attainable ceiling. A per-position teacher-forced surprisal companion rides along. The train-mean prediction (R² = 0 by construction) is the built-in baseline; the mismatched arm is the context-relevance floor and the positive control for the prefix pathway.

**Data extraction:** The context pool and all four answer arms' completions were produced as follows (reused; provenance in the footer). From 5,000 single-turn LMSYS-Chat-1M prompts (established-benchmark data, pinned revision): the *own* arm regenerates each answer with Qwen-2.5-7B-Instruct (vLLM, temperature 1.0, top_p 0.95, max_tokens 1024, seed 42, no system prompt); the *external plain* arm answers the same prompts with `claude-sonnet-4-5-20250929` (temperature 1.0, max_tokens 1024, no system prompt); the *external distinct-style* arm uses the same Claude model under the instruction "Respond in an unusual, stylistically eccentric way — use unexpected structure, mixed register, and non-standard formatting.", stripped before teacher-forcing so the scored context is identical across arms; the *mismatched* arm reassigns own answers by a fixed-point-free derangement (seed 42). 5,000 → 4,998 common-valid; this run further excluded 103 rows with empty pinned external answers (25 overlapping other exclusions), leaving the 4,920-context analysis pool. Activations were captured in one teacher-forced forward pass per (context, arm), fp16, at 8 layers, stream-reduced to the 42 position slots + remainder pools (the full per-token grid was never materialized; the persisted rollout text regenerates it). The divergence bank drew candidate queries from four planned categories — china-politics, model-identity, refusal-boundary, style-format — built from the CCP-sensitive-prompts dataset, in-repo query banks, and identity/style templates (tier-2 divergent items; tier-3 entity-swapped controls and identity templates — a data-realism scope caveat). Of 230 candidate pairs, a prompt-length filter dropped one; 457 queries were judged; the keep gates retained 41 pairs in 2 categories (model-identity 20, style-format 21; china-politics missed the 20-pair floor at 18; refusal-boundary collapsed to 2). 15 of 82 kept queries (12 of 41 pairs, 7 divergent + 8 control, all style-format) hit the 1,024-token generation cap, truncating their own-arm targets mid-answer.

**Sample training/evaluation data + completions:** The run trains no model and generates LMSYS-arm text only by reuse; the worked example below shows one context with all three content arms verbatim (1 of 4,920 pool contexts, drawn from a seed-42 sample of the test split and chosen for brevity; full pool: [HF issue823_own_vs_external/raw_completions @ 8039d15f](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8039d15f30deb845765cbb24d9cdb8708a5e7b0f/issue823_own_vs_external/raw_completions)).

<details>
<summary>Context 1230 — LMSYS prompt + the three content arms (verbatim)</summary>

Prompt: `which is the capital city of german`

Own answer (regenerated, Qwen): `The capital city of Germany is Berlin. It has been the capital since 1990, when Germany was reunified and Berlin replaced Bonn as the seat of government.`

External plain (Claude): `The capital city of Germany is **Berlin**.\n\nBerlin has been the capital of reunified Germany since 1990, following the fall of the Berlin Wall and German reunification. It's also the largest city in Germany and serves as the country's political, cultural, and economic center.`

External distinct-style (Claude, eccentric-formatting instruction stripped before scoring): `# **BERLIN** 🏛️\n\n*the capital city of Germany*\n\n---\n\n**Historical note:** Berlin became the capital of reunified Germany in 1990, though the government didn't fully relocate from Bonn until 1999. [... 602-char answer continues with a population line, a notable-features list, a coordinates code block, and a fun-fact line]`

The mismatched arm pairs this context with the own-arm answer of a different, derangement-assigned context — real answer text with zero context relevance.

</details>

Divergence-bank items are referenced by file + index only (standing content rule for sensitive query categories — no bank text is quoted anywhere in this body): the 229 judged candidate pairs with judge scores, refusal labels, and keep decisions are in [HF issue952_position_divergence/eval_results @ 5b62649](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5b62649cefb34902fd630f21630164e8d1d99764/issue952_position_divergence/eval_results) (`divergence_bank_verification.json`, `divergence_bank_queries.json`; kept pairs carry ids of the form `model_identity_004` / `style_format_037`), and the bank generations + judge outputs are in [HF …/raw_completions @ 5b62649](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5b62649cefb34902fd630f21630164e8d1d99764/issue952_position_divergence/raw_completions).

## Results

### The own-answer advantage is small and position-uniform at the read-out layer, with a reversal at the first token

Held-out test pooled R² per arm across the 42 answer-position slots at layer 20 (top figure); the per-unit view is the per-context R² ECDF pooled over the first-16 slots (bottom figure, n = 629 test contexts).

![Held-out test pooled R-squared per answer position for four answer arms at layer 20](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9f25d29d5577b3cbe10984594325092674848fa/figures/issue_952/hero1_position_r2.png)

> **Figure.** *All arms share a U-profile; the plain-external curve tracks the own curve ~0.02 below it at every content position.* Four arms, 42 slots (first 16 tokens, deciles, last 16), layer 20; bootstrap bands; n(test) varies per slot (universe span ≥ 32 in all arms). Template slots (shaded) are mechanically predictable in every arm.

![ECDF of per-context R-squared pooled over the first sixteen answer slots for four answer arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9f25d29d5577b3cbe10984594325092674848fa/figures/issue_952/exp_percontext_ecdf.png)

> **Figure.** *The per-context distributions behind the pooled curves separate cleanly by arm.* ECDF of per-context R² (first-16 slots pooled), 629 test contexts per arm; the mismatched arm centers on zero.

The plain-external gap is ~0.02–0.03 from the second token onward and flat: the first-16 minus last-16 contrast is +0.0044, inside the ±0.03 margin, so opening concentration is falsified at this layer. The ordering reverses at the first token (own 0.50, plain 0.52, distinct-style 0.55). 40 of 42 slots select layer 20 on validation, yet gaps this size flip with layer — the equivalence read is layer-scoped. 45 of 629 test contexts have a train near-duplicate (pointwise-fold caveat).

### Sixteen absorbed prefix tokens close the plain-external penalty at the layer-20 decision cell, but the closure is layer-fragile and the style penalty persists

Own-minus-external pooled R² gap on the common surviving subset, predicting the identical remainder target, before vs after t absorbed prefix tokens (top-left bars, layer 20); descriptive closure curves (top-right); per-t survivor counts and fraction excluded (bottom).

![Matched prefix-closure bars with descriptive closure curves and survivor attrition panels](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9f25d29d5577b3cbe10984594325092674848fa/figures/issue_952/hero2_matched_prefix.png)

> **Figure.** *The plain-external gap vanishes once 16 of the answer's own tokens are absorbed; the distinct-style gap does not.* Matched-population matched-target contrasts at layer 20; paired test n = 629 / 534 / 403 / 266 at t = 16 / 32 / 64 / 128. Curves descriptive only; attrition strip below (fraction excluded up to 0.44).

Absorbing 16 tokens closes the plain gap at the layer-20 decision cell: +0.023 → −0.002 (closure +0.025; the interval excludes zero and the point clears the 0.02 margin, though the interval dips below it). At layer 17 the closure is +0.014 with an interval spanning zero — layer-specific, so a local-readout account of the closure is not excluded. The style gap does not close: its closure is −0.005 (interval spans zero), the style-minus-plain difference −0.031 excludes zero opposite the prediction, and the style penalty persists on intersection-of-survivors reads (+0.044 at 16 tokens, +0.028 at 128).

### No divergence-specific external penalty is detected on the surviving identity/style bank — an informative null on a weakened instrument

Paired per-context R² drop (entity-swapped control minus divergent query) per arm and bank category (left bars; n = 20 identity, 21 style pairs); per-pair raw drops, the per-unit view (right scatter).

![Paired control-minus-divergent drop bars per bank category and per-pair scatter of raw drops](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9f25d29d5577b3cbe10984594325092674848fa/figures/issue_952/hero3_divergence.png)

> **Figure.** *Per-pair drops scatter around the diagonal — the external map degrades on divergent queries no more than the own map does.* 41 kept pairs; on model-identity pairs both arms' bars are negative (divergent answers more predictable than their controls in both arms).

Pooled over 41 pairs the divergence-specific external penalty is −0.005 (median −0.003, trimmed +0.003); a sign-flip permutation over pair signs (the paired null) gives p = 0.64, and the interval's upper bound (+0.023) excludes the 0.05 margin. Detection was attainable: ceiling 0.887 vs null-band bound +0.028. Scope: 2 of 4 planned categories survived (no refusal or safety read); the judge calibration inverted, rehabilitated only after the fact; cap-truncated style queries are a named dilution channel — a weakened-instrument null is not excluded. Both maps transfer to the bank above the trivial floor (own 0.145, plain 0.137 per-context mean R²), so the null is not map collapse.

### The prefix pathway carries the signal: a shuffled-context answer becomes nearly as predictable as the own answer once 128 of its own tokens are absorbed

Remainder-mean test R² with the predictor pooled over the context plus the first t answer tokens, per arm and prefix length at layer 20; the mismatched arm's rise with t isolates the prefix pathway from context relevance.

![Pooled-prefix remainder R-squared per arm across prefix lengths](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9f25d29d5577b3cbe10984594325092674848fa/figures/issue_952/exp_pooled_prefix.png)

> **Figure.** *The mismatched curve climbs from ≈ 0 toward the content arms as its own prefix is absorbed.* Four arms, 8 prefix lengths, layer 20; content arms sit near 0.55–0.64 throughout; the mismatched arm reaches 0.52 at t = 128 on this pooled read.

With 128 of its own tokens absorbed, a shuffled-context answer's remainder reaches R² 0.351 — 95.6% of the own arm's 0.367 — clearing the 50% recovery criterion. The context-only leg is not fully clean: mismatched position R² exceeds the ±0.05 band at answer tokens 1–2 (0.110, 0.065) and the turn-end template slots (0.106, 0.096), and dips to −0.050 to −0.059 on remainder-mean cells; content positions from token 3 onward stay within ±0.033. Openings carry generic answer statistics, a caveat attached to the prefix-closure result (which rests on the prefix leg).

### Teacher-forced surprisal on external text collapses within a handful of tokens, dissociating from activation predictability at the first token

Mean teacher-forced token surprisal per answer position per arm (top figure); per-position pooled R² against mean surprisal, one point per slot per arm (bottom figure).

![Teacher-forced per-position surprisal curves for four answer arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9f25d29d5577b3cbe10984594325092674848fa/figures/issue_952/exp_surprisal_curves.png)

> **Figure.** *External-text surprise is enormous at the first token and near-baseline within five.* Plain 11.5 nats at token 1 → 1.6 by token 5 → 1.3 late; distinct-style 18.5 → 4.6 → 2.4; mismatched 19.2 → 1.0; own flat at 0.4–0.6.

![Per-position pooled R-squared versus mean teacher-forced surprisal, one point per position slot per arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9f25d29d5577b3cbe10984594325092674848fa/figures/issue_952/exp_r2_vs_surprisal_scatter.png)

> **Figure.** *Within an arm, high-surprisal positions are not systematically less predictable.* One point per position slot (tokens 1–16, 32, 64, 128) per arm.

The fast collapse is consistent with the prefix-absorption mechanism, and the distinct-style arm stays off-distribution longest — matching its persistent penalty. The first token dissociates the two quantities: external openings are the most surprising tokens yet the most predictable activations, plausibly because stereotyped answer openers occupy a low-variance activation region. Teacher-forced surprisal is not validated as behavioral adaptation; no on-policy read exists in this run.

---
**Repro:** ≈7 GPU-h across three GCE attempts (1× A100-80, `eps-issue-952`; attempts 1–2 aborted on bank prerequisites — a missing bank input file, then an overlong bank row — attempt 3 ran end-to-end) + ~1 min VM CPU for the 10,000-draw stats battery (59.6 s wall) and figures · Code @ [d30b153ac3](https://github.com/superkaiba/explore-persona-space/tree/d30b153ac301e2c2f52ac4ea38616ad2c8e4ab30/src/explore_persona_space/experiments/issue_952) (`run_952.py`, `ridge_battery.py`) and [scripts @ d30b153ac3](https://github.com/superkaiba/explore-persona-space/tree/d30b153ac301e2c2f52ac4ea38616ad2c8e4ab30/scripts) (`issue952_bank_build.py`, `issue952_stats.py`, `issue952_figures.py`) · Figures + committed stats: [figures/issue_952 @ a9f25d29d5](https://github.com/superkaiba/explore-persona-space/tree/a9f25d29d5577b3cbe10984594325092674848fa/figures/issue_952), [eval_results/issue_952/stats_summary.json @ a9f25d29d5](https://github.com/superkaiba/explore-persona-space/blob/a9f25d29d5577b3cbe10984594325092674848fa/eval_results/issue_952/stats_summary.json) · Headline eval JSONs (position / prefix / divergence / bank verification / split / validation matrix): [HF issue952_position_divergence/eval_results @ 5b62649](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5b62649cefb34902fd630f21630164e8d1d99764/issue952_position_divergence/eval_results) · Analysis tensors, 6 span files (`truncated == false` verified for all 6), surprisal arrays, per-context bootstrap inputs (59 files): [HF …/analysis_tensors @ 5b62649](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5b62649cefb34902fd630f21630164e8d1d99764/issue952_position_divergence/analysis_tensors) · Bank raw completions + judge outputs: [HF …/raw_completions @ 5b62649](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5b62649cefb34902fd630f21630164e8d1d99764/issue952_position_divergence/raw_completions) · Workload log: [HF …/logs @ 5b62649](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5b62649cefb34902fd630f21630164e8d1d99764/issue952_position_divergence/logs) · Reused completions (4 arms + derangement map + common-valid mask) from [#823](https://eps.superkaiba.com/tasks/823): [HF issue823_own_vs_external/raw_completions @ 8039d15f](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8039d15f30deb845765cbb24d9cdb8708a5e7b0f/issue823_own_vs_external/raw_completions) — fit: same base model and arm recipes; the fp16 re-capture passed the cos > 0.999 equivalence gate against the parent's fp32 means (worst min-cos 0.99957, 4,920/4,920 per arm) · Reused alignment reference from [#779](https://eps.superkaiba.com/tasks/779): [HF issue779_monitoring/analysis_tensors/pass_b @ c9407050](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c94070508aa1c1f9c015ceb072231a2e51b28b3f/issue779_monitoring/analysis_tensors/pass_b) — fit: extraction-path gate reference only, no fit inputs · Prompt source: `lmsys/lmsys-chat-1m` @ revision `200748d9d3cddcc9d782887541057aca0b18c5da` (gated) · WandB: n/a — no training.

**Context:**

> # Motivation
> - We found in a previous experiment that there is an almost as good linear mapping from context to off-policy answers as there is from context to on-policy answers
> - If we are hoping to use this linear mapping as some kind of prediction of the model's behavior then this is worrying, because the off-policy text is **not representative of the model's actual behavior**
> - We want to:
>     - see if this mapping holds even for queries where the 2 models diverge alot
>     - do an in-depth analysis of this off-policy mapping and its comparison to the on-policy mapping
> # Methodology
> - Find the experiment where we did a matched on-policy and off-policy mapping on same contexts and queries
> - We want to see if this mapping is the same **across individual tokens**
>     - i.e.:
>         - from the context vector how much worse/better is the model at predicting the first few activations for on-policy vs off-policy vs last few activations (GO UP TO THE NEWLINE AFTER THE USER CHAT TEMPLATE -- right before the next user turn would start generating)
>         - ideally this would go across the whole answer but it might be hard because answers are of different lengths, so just characterization of first 16 activations vs last 16 activations is good
>         - hypothesis is that model can better predict first few activations for on-policy (off-policy is more surprising) -- but then gets better and better at predicting off-policy as it gets "used" to the style of the off-policy text
>         - Ideally we also want to see if taking "more" tokens into the context vector helps to predict better
>             - so sweep over tokens and regress to predict all other tokens/mean/max pooled tokens (starting after current token)
> - We can only use linear mappings
> - Always validation to select best layer/hyperparameters and evaluation to select best method
> - We also want to see if this mapping is specifically bad for queries where the model behaviors diverge:
>     - For Qwen vs Claude this will probably be questions about China
>     - Search also deeply for known differences/quirks with Claude and Qwen -> and do generation tests to see if these quirks are truly different
>     - then compare the similar answers to the different answers in terms of predictability (mapping always trained on same pool)
>         - for the similar answers try to have a one mapping between queries (e.g. instead of asking about China -- ask about another country) - so we control for this
>
> [Design decisions confirmed in chat 2026-07-03: new child task of #823; broad quirk taxonomy for the divergence bank (~4 categories, generation-verified, matched same-template controls); all four #823 arms carried through the per-token analysis.]

· Lineage: [#823](https://eps.superkaiba.com/tasks/823) — parent (four-arm answer-profile ridge refit/transfer on the same LMSYS pool); no same-issue follow-up rounds · Created 2026-07-03 from user chat; run 2026-07-04 (three GCE launch attempts, third complete); analyzed 2026-07-04.
