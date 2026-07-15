---
title: The own-answer advantage in predicting answer activations from context is position-uniform
  across the captured read-out layer band (MODERATE confidence)
kind: experiment
tags:
- followup-auto
- followup-manual
- keep-running
created_at: '2026-07-03T23:33:32Z'
has_clean_result: true
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
# The own-answer advantage in predicting answer activations from context is position-uniform across the captured read-out layer band (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_952.md](https://github.com/superkaiba/explore-persona-space/blob/7cf07c2deb93ae53f58fa8899d04a59c8dd07d24/docs/methodology/issue_952.md) · [gist](https://gist.github.com/superkaiba/480c8c9c32395d92735942649192f206)
<!-- conciseness WARNs acknowledged: result sections run 120-180 words (soft cap 120), Takeaways bullets exceed 30 words, and total prose exceeds the 800-word soft budget; the overage carries the layer-band, scale-conditioning, instrument-caveat, control-carve-out, and registered-secondary qualifiers deliberately -->

## Takeaways

- **The own-answer advantage in predicting answer activations from context is ~0.01–0.035 R² depending on layer (0.019 at the selected layer 20) and position-uniform at every read-out layer probed, now cross-fitted over the full pool** (first-16 minus last-16 contrasts −0.010 to +0.002 at layers 14, 20, 23, 26 — only layer 14's excludes zero, small but reliable, new to the cross-fit; every 95% interval inside the ±0.03 margin; n = 3,188 matched contexts, each tested exactly once across 5 folds; the gap itself grows with depth, 0.009 at layer 14 to 0.035 at layer 26) — band-scoped: early layers sit at the predictability noise floor, layer 27 uncaptured.
- **The 16-token prefix closure of the plain-external gap is concentrated in the mid band and survives cross-fitting:** 84–118% of each layer's own gap closes at layers 17–23; closure clears the registered 0.02 margin at layer 20 (0.023) and at layer 23 (0.020 — the point lands on the margin in both rounds, so the layer-23 call stays margin-adjacent).
- **Layer 26 closes 39% of its own gap — real but below the margin** (0.013, interval 0.006 to 0.021, zero excluded at the ~5× population; formally still indeterminate against 0.02). Round-1's near-zero read (18%) came from the one low fold — the parent split; the other four folds read 0.010–0.018. **At layer 14 closure is affirmatively below the margin** (interval top 0.009, under half the margin), though at fraction scale its own small gap (0.009) is still nearly covered, so the absence stays scale-conditioned.
- **The distinct-style gap widens with 16 absorbed tokens at every layer except 20** (points −0.048 to −0.008; the interval excludes zero at layers 14, 17, 23, 26 — the layer-23/26 zero-exclusions are new to the cross-fit; at layer 20 the point is −0.000 and the gap is +0.045 after 16 tokens); the position contrast of the own-vs-style advantage now also excludes zero at all four decision layers (−0.019 to −0.008), so the headline position-uniformity is registered on the plain-external family.
- **On the surviving identity/style bank (41 pairs; 2 of 4 categories), no divergence-specific external penalty emerges at the 0.05 margin at any reported bank layer (14, 20, 23, 26) or fold** (per-fold means −0.005 to −0.022, pooled sign-flip p ≥ 0.64; descriptive).
- **Binding caveats:** this round is the second look of a sequential two-look procedure with fixed registered margins on the same context pool (4/5 of the test data fresh) — a precision re-read of round-1's statuses, never an independent confirmation; pooled intervals are conditional on the fixed fold assignment (fold-model variance not propagated; no decision status flips under a fold-level read — e.g. the layer-26 zero-exclusion survives, fold-to-fold sd 0.0056 vs pooled half-width 0.0076); the single-split caveat is retired (calibration fold reproduced round-1's cells with zero R² delta); remaining — inverted judge calibration and ~18% cap-truncated bank queries dilute the bank read; teacher-forced representation-level scope, not on-policy behavior; the layer-23 closure point sits on the margin in both rounds.

## Goal

- **This experiment in context:** The parent experiment ([#823](https://eps.superkaiba.com/tasks/823)) found a linear map from the last context-token activation to the answer-mean activation predicts Claude-written answers nearly as well as Qwen's own — plain-style external answers retained 91–98% of the own-answer refit R²; shuffled pairings scored ≈ 0. This run resolves that residual own-answer advantage on the same four answer conditions along three axes: per-token-position predictability, prefix-conditioned closure, and a divergence-conditioned evaluation on queries where the two models genuinely behave differently. A same-issue follow-up round then recomputed the two registered decision statistics at the remaining captured mid/late read-out layers to test whether the headline findings are local to the validation-selected layer; a third round (`kfold-decision-cells`) re-reads the same decision cells as a 5-fold cross-fit of the full pool, superseding the single-split statuses.
- **Broader narrative:** The context→answer-activation map is a candidate behavior predictor — reading what the model is about to say from its pre-answer state. If it predicts other models' text equally well, it captures content plausibility rather than this model's policy; locating where off-policy text becomes less predictable bounds what such linear predictors can claim.

## Methodology

**Design:** One frozen model (`Qwen/Qwen2.5-7B-Instruct`); 4,920 single-turn LMSYS-Chat-1M contexts split 2,952 train / 984 validation / 984 test (`numpy` rng seed 952). Four answer arms per context — own answer (regenerated), external plain (Claude), external distinct-style (Claude), mismatched (shuffled pairing) — reusing the parent completion set (production written out under Data extraction). Ridge maps are fit from the last context-token activation to per-position answer activations at 42 position slots (first 16 tokens, 10 relative-position deciles, last 16 tokens with the two turn-end template slots split out), and prefix-conditioned maps from the mean activation over context + the first t answer tokens to the remainder, at 8 prefix lengths. A 41-pair divergence bank (divergent query + entity-swapped control) evaluates the same maps out of distribution. The validation split selects the read-out layer and per-slot regularization; the disjoint test split carries every comparison. The manipulated variable is the dependent-variable granularity — per-position and per-prefix instead of answer-mean — plus the divergence conditioning; estimator family, answer arms, and pool are held fixed.

*Cross-layer round (`cross-layer-decision-cells`, follow-up plan v8):* the two registered decision statistics were recomputed at the plan-fixed read-out layers {14, 23, 26} — with layer 20 re-run as a calibration cell and the parent layer-20/17 rows carried into figures and tables as PARENT rows, never re-derived — running entirely on the parent's stored fp16 slot tensors (HF revision `5b62649…`), with the per-slot regularization re-selected within each layer on the same validation split under the parent's registered rule. No new generation, judging, or capture. The round inherits the parent's realized 4,920-context pool — the plan's nominal 4,998 common-valid minus the 103 empty-external exclusions, an inherited deviation carried from the parent run — and the 2,952/984/984 split unchanged.

Round 3 (`kfold-decision-cells`, follow-up plan v12): the single 2,952/984/984 holdout is replaced by a 5-fold cross-fit of the same 4,920-context pool (the identical rng(952) permutation; fold 4 ≡ the parent split, run first as the calibration fold with the round-1 reproduction gates binding there). Each context is scored once, in its test fold; the ridge penalty is re-selected per fold per slot under the registered validation-argmax rule (the selected read-out layer re-derived as 20 in every fold); per-context paired statistics pool across folds (bootstrap seed 0 / sign-flip seed 1, 10,000 draws, contexts resampled with fold assignment fixed — pooled intervals are therefore conditional on the fold assignment; fold-model variance is not propagated). All statistic definitions, margins (0.03 equivalence / 0.02 closure), the three-way status rule, and the Holm procedure are inherited unchanged; the cross-fitted statuses at layers 14/23/26 supersede round-1's single-split statuses (never pooled with them — the populations overlap). The round is registered as the second look of a sequential two-look procedure with fixed margins, conditioned on round-1's indeterminate/knife-edge statuses: 4/5 of the test data is fresh, but the pool, capture, and model are shared with round 1, so the cross-fitted reads supersede rather than independently confirm the single-split ones. Round-1's pointwise near-duplicate caveat (~7% of test contexts with a train near-duplicate) carries over per fold by construction; the paired contrast largely cancels it. Populations: 4,920 pool (78 ids with empty external-arm answers excluded upstream), 3,188 span-32 matched survivors (per-fold 649/642/626/642/629).

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
| Decision layers (cross-layer round) | {14, 23, 26} added; layer 20 as suffixed-path calibration cell; parent 20/17 rows carried | follow-up plan v8 §2 |
| Multiplicity (cross-layer round) | Holm across the 3 added layers, per hypothesis family; no cross-layer max anywhere (the layer set is fixed in advance) | follow-up plan v8 §2 |
| Bootstrap p construction (cross-layer round) | shift-to-null-center null (draws − observed), add-one counting; two-sided (position contrast) / one-sided (closure) against 0; B = 10,000 | follow-up plan v8 §2 (pinned) |
| Validity gates (cross-layer round) | read-out re-selection = 20 (hard assert, PASS); layer-20 reproduction vs the parent's committed cells — 168/168 compared, max abs ΔR² = 0.0, tol 1e-6, PASS; suffixed-path layer-20 calibration — max Δ pooled R² = 0.0 across all three cell families, zero λ mismatches, PASS | run gates (`battery_meta.json`) |
| Compute (cross-layer round) | 1× A100-80 (GCE `eps-issue-952`), battery wall 3.1 h, ≈3.6 GPU-h realized; stats battery 10 s VM-side | run telemetry |
| Round scope (k-fold round) | no new training (N/A); no new generation; no judging; GPU use is the per-fold ridge/SVD battery only | follow-up plan v12 |

**Evaluation:** Three dependent variables, all teacher-forced representation-level by design (the inherited scope: claims are about the linear map on realized answer text, never on-policy behavior prediction). (1) *Per-position predictability* — held-out test pooled R² of the ridge map per arm per slot at the frozen (layer, λ); the position-profile decision statistic is the first-16 minus last-16-content-token gap contrast (own minus external), read against the 0.03 margin. (2) *Prefix-conditioned closure* — the decision statistic is a matched-population, matched-target contrast: on the common surviving subset intersected across compared arms, the own-minus-external gap predicting the identical remainder target, with no prefix vs after t prefix tokens; variable-population closure curves are descriptive only, and per-t survivor attrition is reported. (3) *Divergence-conditioned evaluation* — per kept bank pair, the per-context R² drop (control minus divergent) differenced between the external-plain and own maps, pooled over 41 pairs with median and 10%-trimmed companions; a sign-flip permutation null over pair signs gives the p-value, per-category reads carry a Holm correction across the surviving categories, and the null band is reported next to the attainable ceiling. A per-position teacher-forced surprisal companion rides along. The train-mean prediction (R² = 0 by construction) is the built-in baseline; the mismatched arm is the context-relevance floor and the positive control for the prefix pathway. *Cross-layer round:* each added layer reports both decision statistics under a registered three-way status — equivalence: interval non-strictly within ±0.03 replicates; closure: at least 0.02 with Holm-adjusted p < 0.05 replicates, interval upper bound below 0.02 is affirmative non-replication, otherwise indeterminate — mapped by a four-outcome lattice fixed in the follow-up plan; this round realized the partial-band-map outcome. The divergence-bank read at the added layers is a descriptive rider with no decision role.

**Data extraction:** The context pool and all four answer arms' completions were produced as follows (reused; provenance in the footer). From 5,000 single-turn LMSYS-Chat-1M prompts (established-benchmark data, pinned revision): the *own* arm regenerates each answer with Qwen-2.5-7B-Instruct (vLLM, temperature 1.0, top_p 0.95, max_tokens 1024, seed 42, no system prompt); the *external plain* arm answers the same prompts with `claude-sonnet-4-5-20250929` (temperature 1.0, max_tokens 1024, no system prompt); the *external distinct-style* arm uses the same Claude model under the instruction "Respond in an unusual, stylistically eccentric way — use unexpected structure, mixed register, and non-standard formatting.", stripped before teacher-forcing so the scored context is identical across arms; the *mismatched* arm reassigns own answers by a fixed-point-free derangement (seed 42). 5,000 → 4,998 common-valid; this run further excluded 103 rows with empty pinned external answers (25 overlapping other exclusions), leaving the 4,920-context analysis pool. Activations were captured in one teacher-forced forward pass per (context, arm), fp16, at 8 layers, stream-reduced to the 42 position slots + remainder pools (the full per-token grid was never materialized; the persisted rollout text regenerates it). The divergence bank drew candidate queries from four planned categories — china-politics, model-identity, refusal-boundary, style-format — built from the CCP-sensitive-prompts dataset, in-repo query banks, and identity/style templates (tier-2 divergent items; tier-3 entity-swapped controls and identity templates — a data-realism scope caveat). Of 230 candidate pairs, a prompt-length filter dropped one; 457 queries were judged; the keep gates retained 41 pairs in 2 categories (model-identity 20, style-format 21; china-politics missed the 20-pair floor at 18; refusal-boundary collapsed to 2). 15 of 82 kept queries (12 of 41 pairs, 7 divergent + 8 control, all style-format) hit the 1,024-token generation cap, truncating their own-arm targets mid-answer. The cross-layer round extracted nothing new: it staged the parent's 30 slot shards, 6 span files, and bank verification record at the pinned HF revision and recomputed statistics only.

**Sample training/evaluation data + completions:** The run trains no model and generates LMSYS-arm text only by reuse; the worked example below shows one context with all three content arms verbatim (1 of 4,920 pool contexts, drawn from a seed-42 sample of the test split and chosen for brevity; full pool: [HF issue823_own_vs_external/raw_completions @ 8039d15f](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8039d15f30deb845765cbb24d9cdb8708a5e7b0f/issue823_own_vs_external/raw_completions)).

<details>
<summary>Context 1230 — LMSYS prompt + the three content arms (verbatim)</summary>

Prompt: `which is the capital city of german`

Own answer (regenerated, Qwen): `The capital city of Germany is Berlin. It has been the capital since 1990, when Germany was reunified and Berlin replaced Bonn as the seat of government.`

External plain (Claude): `The capital city of Germany is **Berlin**.\n\nBerlin has been the capital of reunified Germany since 1990, following the fall of the Berlin Wall and German reunification. It's also the largest city in Germany and serves as the country's political, cultural, and economic center.`

External distinct-style (Claude, eccentric-formatting instruction stripped before scoring): `# **BERLIN** 🏛️\n\n*the capital city of Germany*\n\n---\n\n**Historical note:** Berlin became the capital of reunified Germany in 1990, though the government didn't fully relocate from Bonn until 1999. [... 602-char answer continues with a population line, a notable-features list, a coordinates code block, and a fun-fact line]`

The mismatched arm pairs this context with the own-arm answer of a different, derangement-assigned context — real answer text with zero context relevance.

</details>

Divergence-bank items are referenced by file + index only (standing content rule for sensitive query categories — no bank text is quoted anywhere in this body): the 229 judged candidate pairs with judge scores, refusal labels, and keep decisions are in [HF issue952_position_divergence/eval_results @ 5b62649](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5b62649cefb34902fd630f21630164e8d1d99764/issue952_position_divergence/eval_results) (`divergence_bank_verification.json`) and in git at [`divergence_bank_queries.json` @ ac9f45b4ca](https://github.com/superkaiba/explore-persona-space/blob/ac9f45b4ca42d7b55091a0fa169b8480e2fe0c62/eval_results/issue_952/divergence_bank_queries.json) (kept pairs carry ids of the form `model_identity_004` / `style_format_037`), and the bank generations + judge outputs are in [HF …/raw_completions @ 5b62649](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5b62649cefb34902fd630f21630164e8d1d99764/issue952_position_divergence/raw_completions).

## Results

### The own-answer advantage is small and position-uniform at the read-out layer

Held-out test pooled R² per arm across the 42 answer-position slots at layer 20 (top figure); the per-unit view is the per-context R² ECDF pooled over the first-16 slots (bottom figure, n = 629 test contexts).

![Held-out test pooled R-squared per answer position for four answer arms at layer 20](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9f25d29d5577b3cbe10984594325092674848fa/figures/issue_952/hero1_position_r2.png)

> **Figure.** *All arms share a U-profile; the plain-external curve tracks the own curve ~0.02 below it at every content position.* Four arms, 42 slots (first 16 tokens, deciles, last 16), layer 20; bootstrap bands; n(test) varies per slot (universe span ≥ 32 in all arms). Template slots (shaded) are mechanically predictable in every arm.

![ECDF of per-context R-squared pooled over the first sixteen answer slots for four answer arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9f25d29d5577b3cbe10984594325092674848fa/figures/issue_952/exp_percontext_ecdf.png)

> **Figure.** *The per-context distributions behind the pooled curves separate cleanly by arm.* ECDF of per-context R² (first-16 slots pooled), 629 test contexts per arm; the mismatched arm centers on zero.

The plain-external gap is ~0.02–0.03 from the second token onward and flat: the first-16 minus last-16 contrast is +0.0044, and its interval lies entirely inside the ±0.03 margin, so opening concentration is falsified at this layer.

- The ordering flips at the first token (point estimates only: own 0.50, plain 0.52, distinct-style 0.55).
- 40 of 42 slots select layer 20 on validation; the cross-layer round replicates the equivalence at layers 14, 23, and 26 (band-map result below), retiring the layer-scoped caveat.
- 45 of 629 test contexts have a train near-duplicate (pointwise-fold caveat).

### Sixteen absorbed prefix tokens close the plain-external gap at the layer-20 decision cell

Own-minus-external pooled R² gap on the common surviving subset, predicting the identical remainder target, before vs after t absorbed prefix tokens (top figure: decision bars, closure curves, attrition strip; layer 20); the per-unit view is the ECDF of 629 per-context own-minus-plain gaps at t = 0 vs 16 (bottom figure).

![Matched prefix-closure bars with descriptive closure curves and survivor attrition panels](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9f25d29d5577b3cbe10984594325092674848fa/figures/issue_952/hero2_matched_prefix.png)

> **Figure.** *The plain-external gap vanishes once 16 of the answer's own tokens are absorbed; the distinct-style gap does not.* Matched-population matched-target contrasts at layer 20; paired test n = 629 / 534 / 403 / 266 at t = 16 / 32 / 64 / 128. Curves descriptive only; attrition strip below (fraction excluded up to 0.44).

![ECDF of per-context own-minus-plain R-squared gaps at zero versus sixteen absorbed prefix tokens](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c51d2415a215c714801cd65b40157edcda3a6859/figures/issue_952/exp_prefix_gap_percontext.png)

> **Figure.** *The whole per-context gap distribution shifts toward zero once 16 prefix tokens are absorbed.* ECDF of the per-context own-minus-plain R² gap, 629 matched test contexts, layer 20; 61% of contexts favor the own answer at t = 0 vs 52% at t = 16.

Absorbing 16 tokens closes the plain gap at the layer-20 decision cell: +0.023 → −0.002 (closure +0.025; interval excludes zero; the point clears the 0.02 margin though the interval dips below it). The style gap does not close: its closure is −0.005 (interval spans zero); the style-minus-plain difference −0.031 excludes zero, opposite the prediction.

- The cross-layer round maps the fragility (band-map result below): closure replicates at layer 23, is affirmatively absent at layer 14, and is unresolved at layer 26.
- The style gap persists on intersection-of-survivors reads: +0.044 at 16 absorbed tokens, +0.028 at 128, both intervals bounded away from zero.

### The position-uniform equivalence replicates at every added read-out layer; the prefix closure replicates only in the mid band

Single-split round (n = 629); its per-layer statuses are superseded by the cross-fitted round below, which reproduced these cells exactly in its calibration fold (672/672 cells, zero R² delta, identical ridge penalties).

![Cross-layer decision profile with registered margins](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3a884ee60b89e74e6925aaef898b5c1ed0569629/figures/issue_952/crosslayer_decision_profile.png)

> **Figure.** *The equivalence contrast stays inside the ±0.03 band at every layer; the closure clears the 0.02 margin only at layers 20 and 23.* 95% bootstrap intervals, n = 629; parent rows open orange, this round filled blue.

- Equivalence replicated at every added layer (contrasts −0.006 to +0.012); closure replicated at layer 23 (0.0201, on the margin), was affirmatively absent at layer 14, and unresolved at layer 26 (0.005, 18% of its own gap).
- Two single-split reads did not survive the cross-fit: layer 26's positive equivalence-contrast trend, and the layer-26 near-zero closure point (this split is the cross-fit's low fold).

### A 5-fold precision re-read of the same pool sharpens the band map: mid-band closure holds, layer 26 shows real sub-margin closure, and layer 14's absence is decisive

Both decision statistics recomputed as a sequential 5-fold precision re-read of the same 4,920-context pool: each context scored once in its held-out fold; 95% intervals from 10,000 paired bootstrap draws resampling contexts, conditional on the fixed fold assignment (fold-model variance not propagated); n = 3,188 matched contexts (~5× round 1); round-1 reads overlaid.

![Cross-fitted decision profile](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3a884ee60b89e74e6925aaef898b5c1ed0569629/figures/issue_952/kfold_layer_profile.png)

> **Figure.** *The equivalence contrast stays inside the ±0.03 band at every layer; closure clears the 0.02 margin only in the mid band.* Cross-fitted points with 95% bootstrap intervals (filled blue), round-1 single-split reads (open orange), n = 3,188 matched contexts.

![Per-fold point estimates](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3a884ee60b89e74e6925aaef898b5c1ed0569629/figures/issue_952/kfold_fold_spaghetti.png)

> **Figure.** *Fold points stay small and inside the pooled equivalence band, with sign variation; layer 26's round-1 low closure read is the calibration fold.* Per-fold equivalence contrasts (left) and closure (right) across layers, pooled estimate as black squares with 95% intervals.

Every round-1 call survives; interval half-widths shrink ~2.3× — no status flips under a fold-level read. The bank rider stays null on the reported bank layers; the round retains the partial band-map verdict.

- Equivalence holds at all three added layers; layer 14's contrast is now nonzero but small (−0.010), and round-1's positive layer-26 trend does not survive (+0.002).
- Layer-23 closure lands on the margin again (0.0203, Holm-adjusted p 0.0003) and remains margin-adjacent.
- Layer 26 excludes zero yet remains indeterminate against 0.02 — 39% of its own gap closes; round-1's 18% was the low fold.
- Layer 14's interval top (0.009) is under half the margin: decisive in absolute terms, though fraction-scale closure stays unresolved.

### No divergence-specific external penalty is detected on the surviving identity/style bank

Paired per-context R² drop (entity-swapped control minus divergent query) per arm and bank category (left bars; 20 identity, 21 style pairs); per-pair raw drops, the per-unit view (right scatter).

![Paired control-minus-divergent drop bars per bank category and per-pair scatter of raw drops](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9f25d29d5577b3cbe10984594325092674848fa/figures/issue_952/hero3_divergence.png)

> **Figure.** *Per-pair drops scatter around the diagonal — the external map degrades on divergent queries no more than the own map does.* 41 kept pairs; on model-identity pairs both arms' bars are negative (divergent answers more predictable than their controls in both arms).

Pooled over 41 pairs, the divergence-specific external penalty is −0.005 (p = 0.64; median −0.003, 10%-trimmed +0.003); the interval excludes the 0.05 margin. Detection was attainable (ceiling 0.887 vs null band +0.028); both maps transfer above the trivial floor (own 0.145, plain 0.137).

- The 31 pairs passing the original ≥+25 margin leave the null unchanged: pooled −0.006 (p = 0.65); style +0.027 (p = 0.33).
- Cross-layer rider: layer-stable null (means −0.014 to −0.006, descriptive; [figure](https://github.com/superkaiba/explore-persona-space/blob/74ff3eac8038c81073bbca27f54eebcccc6a8b0e/figures/issue_952/crosslayer_h3_descriptive.png)).
- The drop is residual-error-driven, not denominator-only (median Δlog ss_res 0.039 vs ss_tot 0.024) — descriptive.
- Registered error-direction secondary: divergent-query predictions displace toward own-answer activations (mean cosine +0.64, 41 pairs) — exploratory (ridge-shrinkage-toward-pool-mean confound).
- Scope: 2 of 4 planned categories survived (no refusal read); judge calibration inverted; margin re-read reuses those scores; style-query cap truncation unprobed.

### The prefix pathway carries the signal: a shuffled-context answer becomes nearly as predictable as the own answer once 128 of its own tokens are absorbed

Remainder-mean test R² pooling the predictor over context plus the first t answer tokens, per arm and prefix length at layer 20 (top figure); the mismatched arm's rise isolates the prefix pathway. The bottom figure gives the per-unit view: per-context R² ECDFs at t = 128.

![Pooled-prefix remainder R-squared per arm across prefix lengths](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9f25d29d5577b3cbe10984594325092674848fa/figures/issue_952/exp_pooled_prefix.png)

> **Figure.** *The mismatched curve climbs from ≈ 0 toward the content arms as its own prefix is absorbed.* Four arms, 8 prefix lengths, layer 20; content arms sit near 0.55–0.64 throughout; the mismatched arm reaches 0.52 at t = 128 on this pooled read.

![Per-context remainder R-squared ECDF, own vs mismatched arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5439838a4fb0ceb97e32f8662fe2929592cd7c11/figures/issue_952/exp_t128_recovery_percontext.png)

> **Figure.** *Per-context predictability of a shuffled-context answer nearly matches the own answer once 128 own tokens are absorbed.* ECDF of per-context remainder R², own (n = 518) vs mismatched (n = 547) test contexts; layer 20 solid, layer 17 dashed. Micro-pooling reproduces the aggregate cells: own 0.425 vs mismatched 0.405 (layer 20), 0.367 vs 0.351 (layer 17).

With 128 own tokens absorbed, a shuffled-context answer's remainder reaches R² 0.351 (95.6% of the own arm's 0.367; layer 17), clearing the 50% recovery criterion. The registered max-pooled companions (context-only baseline) sit uniformly below the mean-pooled content-arm cells, agree on ordering through t = 16, diverge at t = 32–64 (opposite distinct-style vs own ordering at t = 64), and converge by the 128-token cells (`stats_summary.json`).

- The context-only leg is imperfect: mismatched position R² exceeds the ±0.05 band at tokens 1–2 (0.110, 0.065) and turn-end template slots (0.106, 0.096), dipping to −0.050 to −0.059 on remainder-mean cells; content positions from token 3 onward stay within ±0.033 — openings carry generic answer statistics.

### Teacher-forced surprisal on external text collapses within a handful of tokens and dissociates from activation predictability at the first token

Mean teacher-forced token surprisal per answer position per arm (left panel); per-position pooled R² against mean surprisal, one point per slot per arm (right panel).

![Teacher-forced per-position surprisal curves and per-slot R-squared versus surprisal scatter for four answer arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c51d2415a215c714801cd65b40157edcda3a6859/figures/issue_952/exp_surprisal_combined.png)

> **Figure.** *External-text surprise is enormous at the first token and near-baseline within five; within an arm, high-surprisal positions are not systematically less predictable.* Left: plain 11.5 nats at token 1 → 1.6 by token 5; distinct-style 18.5 → 4.6; mismatched 19.2 → 1.0; own flat at 0.4–0.6. Right: one point per slot (tokens 1–16, 32, 64, 128) per arm.

The fast collapse is consistent with the prefix-absorption mechanism, and the distinct-style arm stays off-distribution longest, in line with its persistent style gap. The first token dissociates the two quantities: external openings are the most surprising tokens yet the most predictable activations, plausibly because stereotyped answer openers occupy a low-variance activation region. Teacher-forced surprisal is not validated as behavioral adaptation; no on-policy read exists in this run.

---
**Repro:** ≈7 GPU-h across three GCE attempts (1× A100-80, `eps-issue-952`; attempts 1–2 aborted on bank prerequisites — a missing bank input file, then an overlong bank row — attempt 3 ran end-to-end) + ~1 min VM CPU for the 10,000-draw stats battery (59.6 s wall) and figures · Code @ [d30b153ac3](https://github.com/superkaiba/explore-persona-space/tree/d30b153ac301e2c2f52ac4ea38616ad2c8e4ab30/src/explore_persona_space/experiments/issue_952) (`run_952.py`, `ridge_battery.py`) and [scripts @ 5439838a4f](https://github.com/superkaiba/explore-persona-space/tree/5439838a4fb0ceb97e32f8662fe2929592cd7c11/scripts) (`issue952_bank_build.py`, `issue952_stats.py`, `issue952_figures.py`) · Figures + committed stats: [figures/issue_952 @ 5439838a4f](https://github.com/superkaiba/explore-persona-space/tree/5439838a4fb0ceb97e32f8662fe2929592cd7c11/figures/issue_952), [eval_results/issue_952/stats_summary.json @ a9f25d29d5](https://github.com/superkaiba/explore-persona-space/blob/a9f25d29d5577b3cbe10984594325092674848fa/eval_results/issue_952/stats_summary.json), [eval_results/issue_952/stats_h3_original_margin.json @ ac9f45b4ca](https://github.com/superkaiba/explore-persona-space/blob/ac9f45b4ca42d7b55091a0fa169b8480e2fe0c62/eval_results/issue_952/stats_h3_original_margin.json) · Headline eval JSONs (position / prefix / divergence / bank verification / split / validation matrix): [HF issue952_position_divergence/eval_results @ 5b62649](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5b62649cefb34902fd630f21630164e8d1d99764/issue952_position_divergence/eval_results) · Analysis tensors, 6 span files (`truncated == false` verified for all 6), surprisal arrays, per-context bootstrap inputs (59 files): [HF …/analysis_tensors @ 5b62649](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5b62649cefb34902fd630f21630164e8d1d99764/issue952_position_divergence/analysis_tensors) · Bank raw completions + judge outputs: [HF …/raw_completions @ 5b62649](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5b62649cefb34902fd630f21630164e8d1d99764/issue952_position_divergence/raw_completions) · Workload log: [HF …/logs @ 5b62649](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5b62649cefb34902fd630f21630164e8d1d99764/issue952_position_divergence/logs) · Reused completions (4 arms + derangement map + common-valid mask) from [#823](https://eps.superkaiba.com/tasks/823): [HF issue823_own_vs_external/raw_completions @ 8039d15f](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8039d15f30deb845765cbb24d9cdb8708a5e7b0f/issue823_own_vs_external/raw_completions) — fit: same base model and arm recipes; the fp16 re-capture passed the cos > 0.999 equivalence gate against the parent's fp32 means (worst min-cos 0.99957, 4,920/4,920 per arm) · Reused alignment reference from [#779](https://eps.superkaiba.com/tasks/779): [HF issue779_monitoring/analysis_tensors/pass_b @ c9407050](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c94070508aa1c1f9c015ceb072231a2e51b28b3f/issue779_monitoring/analysis_tensors/pass_b) — fit: extraction-path gate reference only, no fit inputs · Prompt source: `lmsys/lmsys-chat-1m` @ revision `200748d9d3cddcc9d782887541057aca0b18c5da` (gated) · WandB: n/a — no training · **Cross-layer round** (`cross-layer-decision-cells`, follow-up plan v8, proposer-initiated cheap-band round 1): ≈3.6 GPU-h realized (1× A100-80 GCE `eps-issue-952`, single attempt; battery wall 3.1 h) + 10 s VM-side stats · round workload code @ [3a74cf91de](https://github.com/superkaiba/explore-persona-space/tree/3a74cf91de3e5db4d22c9514e4fb288b7c76ed67/src/explore_persona_space/experiments/issue_952) · round stats + figures @ [74ff3eac80](https://github.com/superkaiba/explore-persona-space/tree/74ff3eac8038c81073bbca27f54eebcccc6a8b0e/eval_results/issue_952/cross-layer-decision-cells) (`stats_cross_layer.json`, `position_r2_by_arm_cross_layer.json`, `prefix_closure_by_arm.json`, `battery_meta.json`; figures under [figures/issue_952 @ 3a884ee60b](https://github.com/superkaiba/explore-persona-space/tree/3a884ee60b89e74e6925aaef898b5c1ed0569629/figures/issue_952)) · round HF artifacts (per-context npz, eval JSONs, workload log — 11 files): [HF issue952_position_divergence/followups/cross_layer_decision_cells @ da5ec60](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/da5ec60cc62d62f98f26f5b88f7e6bfc26761420/issue952_position_divergence/followups/cross_layer_decision_cells) — reused round inputs: the parent run's own slot tensors, span files, and bank verification record @ [HF …/analysis_tensors @ 5b62649](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5b62649cefb34902fd630f21630164e8d1d99764/issue952_position_divergence/analysis_tensors) — fit: same code path, single-attempt provenance; the layer-20 reproduction gate (max abs ΔR² 0.0 over 168 cells) is the binding runtime coherence check. · **K-fold round** (`kfold-decision-cells`, follow-up plan v12, proposer-initiated cheap-band round 2): ≈15.2 GPU-h realized (1× A100-80 GCE `eps-issue-952`, attempt att-20260705-082757; five ~3 h fold batteries — the calibration gate rejected the smoke run's persisted folds, forcing full re-runs — plus a ~12 min bank re-score; the terminal upload recovered from a transient HF Hub 502 by an on-instance retry, science unaffected) + 20 s VM CPU for the pooled 10,000-draw stats battery and figures · round workload code @ [9fb307b6b8](https://github.com/superkaiba/explore-persona-space/tree/9fb307b6b87dd7cf6a95114ad48662e11e2b5cc0/src/explore_persona_space/experiments/issue_952) · pooled stats + round figures @ [3c1970d234](https://github.com/superkaiba/explore-persona-space/tree/3c1970d2341f8f454438281e1a1776bcb31aea6f/eval_results/issue_952/kfold-decision-cells) (`stats_kfold.json`; per-fold battery JSONs committed at [9fb307b6b8's run](https://github.com/superkaiba/explore-persona-space/tree/3c1970d2341f8f454438281e1a1776bcb31aea6f/eval_results/issue_952/kfold-decision-cells); figures `kfold_layer_profile`, `kfold_fold_spaghetti` under [figures/issue_952 @ 3a884ee60b](https://github.com/superkaiba/explore-persona-space/tree/3a884ee60b89e74e6925aaef898b5c1ed0569629/figures/issue_952)) · round HF artifacts (5 per-fold context npz, 34 eval JSONs, workload log): [HF issue952_position_divergence/followups/kfold_decision_cells @ 433e84b](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/433e84ba4293604b4b4f15e0369cd4b292ed7201/issue952_position_divergence/followups/kfold_decision_cells) — reused round inputs: the parent's slot tensors + span files @ 5b62649 (fit: same code path; the calibration fold reproduced round-1's cross-layer cells 672/672 with max abs R² delta 0.0 and identical ridge penalties — the binding runtime coherence check) · gates: selected layer 20 re-derived independently in every fold; partition coverage 4,920 contexts each tested once (PASS); bootstrap GEMM parity vs the serial oracle 0.0. Plan provenance: `plans/plan.md` points at the round's v12 amendment (v11 corrigendum: the paired population is the span-32 survivor universe, n = 3,188 realized vs ≈3,145 planned).

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

Round 1 originating scope (proposer-initiated; verbatim from the `epm:followup-scope` marker, hypothesis-family clause elided): "followup_label: cross-layer-decision-cells / source: proposer-9b-cheap / gpu_hours_estimate: 3 / question_relation: same"; "**Differs from parent:** Exactly one thing — the decision-cell READ-OUT LAYER set: the registered […] statistics are recomputed at the three remaining captured mid/late layers {14, 23, 26} (with the existing 20/17 reads carried for comparison). Tensors, split, λ-selection rule (per-slot validation argmax within layer), margins, bootstrap recipe all identical."

· Lineage: [#823](https://eps.superkaiba.com/tasks/823) — parent (four-arm answer-profile ridge refit/transfer on the same LMSYS pool) · Same-issue follow-up rounds: round 1 `cross-layer-decision-cells` (proposer-9b-cheap, cheap-band auto-run round 1 of cap 2) · Created 2026-07-03 from user chat; run 2026-07-04 (three GCE launch attempts, third complete); analyzed 2026-07-04; round 1 run + folded 2026-07-04.

> **Follow-up round 3** (`kfold-decision-cells`, proposer-initiated, Step 9b cheap band round 2 of cap 2, est. 12 GPU-h): "Cross-fitted (5-fold) re-read of the [equivalence/closure] decision cells" — targeting the layer-26 indeterminate (sharpened, not resolved: formally still indeterminate), the layer-23 knife-edge, and the layer-14 fraction question; verbatim proposal spec in the `epm:followup-scope v2` marker (2026-07-05T05:43:27Z); no fresh Goal (same-question round).

