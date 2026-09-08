---
title: Token-position-resolved on/off-policy characterization of the context→answer
  activation map + divergence-conditioned evaluation (Qwen vs Claude)
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
# Token-position-resolved on/off-policy characterization of the context→answer activation map + divergence-conditioned evaluation (Qwen vs Claude)

## Goal

On frozen Qwen-2.5-7B-Instruct with ridge-only maps and a train/validation/test split (validation selects layer + λ, a disjoint test split compares methods and arms), characterize where the linear context→answer-activation map differs between on-policy and off-policy answers, along three axes: (1) per-token-position predictability — fit h_t: c_last(x) → z_t^(a)(x) for answer positions t in the first-16 window, the last-16 window (span extended through `<|im_end|>` and its trailing newline — the last token before a next user turn would begin), and relative-position deciles, per arm a ∈ {own-regenerated, external-plain, external-distinct-style, mismatched} (the four #823 arms, completions reused) over the 4998-context LMSYS pool; (2) prefix-conditioned prediction — sweep the predictor to the realized-answer activation at position t (t ∈ {1,2,4,8,16,32,64,128}; t=0 = c_last baseline) predicting the individual / mean-pooled / max-pooled activations of positions > t, measuring how fast each arm's remainder becomes predictable as prefix is absorbed; and (3) divergence-conditioned evaluation — evaluate the SAME pool-trained maps on a generation-verified Qwen-vs-Claude divergence query bank (≈4 quirk categories × ~40–60 queries, each with matched same-template entity-swapped controls), testing whether off-policy predictability fails specifically where the two models' behaviors diverge while on-policy predictability holds.

## Overview / Motivation

#823 established that the per-context ridge map h: c_last(x) → v(x) (last-context-token activation → answer-span MEAN activation) is content-indexed, not self-generation-specific: Claude-Sonnet plain-style answers retain 91–98% of the own-answer refit R², while mismatched answers collapse to ≈0. If this map is to serve as a base-side PREDICTOR of the model's own behavior (the leakage-predictor line's use case), that retention is worrying — the off-policy text is not the model's behavior, yet the map "predicts" it almost as well. This experiment characterizes WHERE the on/off-policy difference lives, along two axes the mean-pooled DV cannot resolve:

1. **Token position.** The mean answer profile averages away position. Per-position maps h_t: c_last(x) → z_t(x) (activation at answer-token position t) test the hypothesis that the model predicts its OWN first tokens better (off-policy openings are unknowable from the context), while off-policy answers become progressively more predictable at later positions (topic-determined content dominates; style novelty attenuates).
2. **Query divergence.** The LMSYS pool is dominated by queries where Qwen and Claude answer similarly, so the mean retention may hide a failure mode exactly where prediction matters: queries where the two models' BEHAVIOR diverges (Qwen's China-politics deflections vs Claude's substantive answers being the canonical case). A generation-verified divergence bank with matched same-template controls (entity-swapped: China → another country) tests whether the off-policy map degrades specifically on divergent queries while the on-policy map holds.

A third, mechanism-level axis rides along: **prefix-conditioned prediction** — sweeping the predictor from the last context token to the activation at answer position t, and predicting the remaining positions. If the model "gets used to" off-policy style, conditioning on a longer realized prefix should close the off-policy gap faster than the on-policy one (largest for the distinct-style arm; the mismatched arm is the positive control — its context-only R² is ≈0 but its prefix carries its own coherent remainder).

## Formalization

- Context x = LMSYS single-turn prompt (the #722/#823 pool); c_last(x) ∈ R^3584 = residual activation at the last context token (assistant-header slot), layer ℓ.
- Answer span for arm a: tokens of the arm's answer TEACHER-FORCED through frozen Qwen, EXTENDED through the turn-end template tokens `<|im_end|>` + trailing `\n` (assert the rendered template's token ids in-process). z_t^(a)(x) ∈ R^3584 = residual activation at answer position t, layer ℓ.
- Position slots: F16 = t ∈ {1..16} (answer start); L16 = the final 16 tokens of the extended span (aligned from the end; the two turn-end template tokens reported both split out and pooled — fixed-token-identity positions are mechanically easier to predict and must not silently inflate the late window); D10 = relative positions {5%, 15%, …, 95%} of the span (length-invariant whole-answer profile). Answers with extended span < 32 tokens are excluded from the F16-vs-L16 contrast (overlapping windows) but retained in D10; counts reported per arm.
- Per-position map (context-only): ridge h_{a,t,ℓ}: c_last → z_t, standardize-X / center-Y (train-fold statistics — centering per (slot, layer) makes "predict this position's mean" the R²=0 baseline), λ grid np.logspace(−2,4,13). DV: held-out test R² (pooled primary + equal-weighted per-context companion, the #823 estimand pair, never mixed).
- Prefix-conditioned map: g_{a,t,ℓ}: z_t^(a)(x) → {z_{t'}^(a)(x) individually (decile probes > t), mean-pool_{t'>t} z_{t'}, max-pool_{t'>t} z_{t'}}; only contexts with span ≥ t+16 enter the t-cell (per-cell n reported). Secondary predictor variant: mean-pool over context+prefix tokens ≤ t (tests summary richness vs the line's single-token recipe).
- Split discipline (user-pinned): contexts split train/validation/test (e.g. 60/20/20, fixed across all arms and slots so every cross-arm comparison is paired context-wise); validation selects λ AND layer under one rule applied identically per arm; the test split is scored once per pre-registered comparison. Any max-over-free-axis headline (layer, t, pooling) carries a selection-symmetric permutation null band (`.claude/rules/selection-symmetric-nulls.md`).
- Companion DV (free, same forwards): per-token teacher-forced surprisal −log P(answer_t | context, answer_{<t}) per arm — the behavioral in-context-adaptation profile. Report the position-resolved surprisal curves alongside the R² curves and their per-position correlation: does activation predictability track token surprisal, or dissociate from it?
- Divergence DV: judge-scored graded divergence (0–100, claude-sonnet-4-5-20250929, N≥5 draws at temperature>0, mean-aggregated, malformed/REFUSAL draws DROPPED never coerced, per `.claude/rules/llm-judging.md`) between the Qwen answer and the Claude answer to the same query, plus an embedding-cosine companion as the independent non-judge reference (the judge is scoring a pair containing its own family's text — the similarity-judgment framing lowers self-preference risk but the companion is required). A category enters the eval only if its verified divergence exceeds its matched controls' by a pre-registered margin.

## Hypotheses

- **H1 (early own-advantage):** Δ(own − external-plain) per-position R² is largest over F16 and shrinks by L16 — the context vector cannot encode another model's opening moves, but late positions are increasingly topic-determined. The #823 mean-level retention would then be a late-position artifact, and the map's apparent off-policy validity would NOT extend to the answer's behavioral commit point (the opening).
- **H2 (prefix adaptation):** the off-policy remainder becomes predictable faster with prefix length t than the on-policy remainder improves — steepest for distinct-style; the mismatched arm shows ≈0 context-only R² at every position but near-full prefix-conditioned recovery (positive control that the prefix pathway works).
- **H3 (divergence-specific failure):** on verified-divergent queries, external-answer predictability drops relative to matched controls (largest early), while own-answer predictability holds; the map's prediction error on divergent external answers aligns with the (own − external) profile difference — i.e. the map predicts what Qwen WOULD say, not what Claude said.
- **H0 (content-indexed everywhere):** retention is position-uniform and divergence-insensitive — the #823 content-indexed read holds at every position and the off-policy retention is a genuine property of the map, not an averaging artifact. This would sharpen (not soften) the "worrying for behavior prediction" conclusion.

## Design sketch (phases; /adversarial-planner refines every value)

**Phase 0 — reuse + capture (GPU).** Reuse the four #823 arms' completion text verbatim (HF `issue823_own_vs_external/raw_completions/` @ `8039d15f30de…`, 4998 common-valid contexts; mismatched pairing = the #823 seed-42 derangement) under the artifact-reuse fitness checklist. New teacher-forced forwards (one per context × arm, batched, logits kept for the surprisal companion) capturing: c_last, the F16/L16/D10 target slots, and the prefix predictor slots — at an ~11-layer grid including the #823 read-out layers L14/L17/L26 (planner may thin; storage ≈ 46 slots × 11 layers × 3584 × fp16 ≈ 18 GB/arm ≈ 72 GB total → over the 50 GB VM ceiling: capture pod-side with HF `analysis_tensors/` upload, or thin layers / stream-reduce; planner routes per §9).

**Phase 1 — per-position context-only maps (CPU, batched).** The full battery shares one predictor matrix per (layer, fold) — ONE factorization reused across every target slot × arm (targets stack as Y columns), so the whole battery is ~11 layers × folds × 13 λ solves, not thousands of independent fits (`.claude/rules/vectorize-many-cell-fits.md`; #823's Gram-eigh fast path FAILED parity and was reverted — parity-check any fast path at full size before trusting it). Outputs: per-arm position-resolved R² curves (F16, L16, D10), paired arm gaps with bootstrap CIs, length-stratified sensitivity (the #823 length-matched sweep precedent), turn-end template positions split out.

**Phase 2 — prefix-conditioned sweep (CPU, batched).** Per-t predictor matrices; Gram/eigh once per (t, layer, fold) reused across λ and all stacked targets. Outputs: gap-closure curves R²_a(remainder | prefix ≤ t) vs t per arm; the cross-arm closure-rate contrast is the estimand (absolute R² inflation from residual-stream autocorrelation cancels in the cross-arm comparison at matched t).

**Phase 3 — divergence bank (API + ~1–2 GPU-h).** (i) Assemble ≈4 candidate categories × ~40–60 queries + matched entity-swapped same-template controls: china-politics (Tiananmen/Taiwan/Xinjiang/HK/Xi → matched other-country/other-event templates), model-identity/self-description (divergent-by-construction, labeled factual-self-reference; identity-confusion audits are a documented divergence surface, arXiv 2411.10683), refusal-boundary (candidates where exactly one model refuses; bank items referenced by filename+index, digest-only, per the harmful-content context rule), style/formatting-inducing. Candidate sources (verified 2026-07-03; tier-2 established banks preferred over fresh synthesis per `.claude/rules/data-realism.md`): **promptfoo/CCP-sensitive-prompts** (HF dataset, 1,360 prompts across 68 labeled sensitive topics — the per-topic labels directly support entity-swapped control construction), **augmxnt/deccp** (Qwen-derived censored-prompt set; companion Qwen-2 censorship audit huggingface.co/blog/leonardlin/chinese-llm-censorship-analysis documents hard refusals + CCP-aligned canned templates with English-vs-Chinese asymmetry), R1dacted (arXiv 2505.12625, weights-level topic-gated censorship in R1/Qwen-distills), PSP (arXiv 2511.23174, separates safety refusal from political censorship). **Query language pinned to English** (matching the LMSYS training pool): divergence magnitude is language-conditional (arXiv 2602.06371 documents stance shifts by query language), so a Chinese-language arm is a named optional extension, not a silent mix. (ii) Generation verification: Qwen on-policy (vLLM, sampling matched to the #823 regen recipe) + Claude plain (claude-sonnet-4-5-20250929, no system prompt, #823-matched) on every candidate; score divergence (judge + embedding companion); keep only categories/queries clearing the pre-registered divergence margin over their controls; report per-category Qwen refusal/deflection rates and length stats (Qwen's divergent-topic answers may be short deflections — a length/style confound the analysis must stratify on). (iii) Teacher-forced captures for the kept queries × {own, external-plain} arms. (iv) Evaluation: maps TRAINED ONLY on the LMSYS pool (divergence queries never in train/validation), scored on {divergent, matched-control} × {own, external-plain} × position slots; paired stats over entity-swapped query pairs, Bonferroni across categories. Secondary read: cos(prediction error on divergent external answers, own−external profile delta).

**Phase 4 — analysis + figures.** Position-resolved R² curves per arm (F16/L16/D10); surprisal-vs-R² per-position scatter; prefix-sweep closure curves; divergent-vs-control paired bars per category; per-context ECDFs (the #823 figure-3 pattern).

## Decision rules (pre-registered; planner pins thresholds against #823's resolvable-gap scale, fold SDs ≤0.023 at n≈5000)

- Early own-advantage: mean Δ(own−plain) R² over F16 exceeds the L16 mean Δ by a margin with bootstrap CI excluding 0 (planner pins the margin; #823 resolved gaps ≥0.03–0.05).
- Prefix adaptation: the t at which the arm reaches 90% of its own-arm asymptote is smaller for external arms' closure than the context-only gap predicts; cross-arm closure-rate difference CI excludes 0.
- Divergence-specific failure: (control − divergent) R² drop for the external arm exceeds the own arm's drop, paired over entity-swapped pairs, CI excluding 0, surviving length stratification.
- Content-indexed-everywhere (H0) is the explicit alternative and is reportable as a clean result — a null here is informative.

## Constraints (user-pinned, verbatim commitments)

- Linear (ridge) mappings ONLY — no MLP anywhere (the #823 precedent).
- ALWAYS validation to select layer/hyperparameters, disjoint evaluation to compare methods/arms.
- Answer span goes up to the newline after the assistant turn's `<|im_end|>` — right before a next user turn would start (assert template token ids in-process).
- The divergence comparison always uses maps trained on the same (LMSYS) pool.
- Minimum deliverable on the position axis: first-16 vs last-16 characterization; deciles extend it length-invariantly across the whole answer.

## Compute sketch (planner refines §9)

- GPU: 4 arms × 4998 + ~2 arms × ~800 divergence-bank teacher-forced forwards (≤8192 tokens, batched, activations + realized-token logprobs) ≈ 4–8 GPU-h on 1× A100-80; Qwen generation for the divergence bank ≈ <1 GPU-h (vLLM). No training anywhere.
- API: ~400–500 Claude answer generations + divergence judging (~500 pairs × 5 draws ≈ 2.5k judge calls — sync path fine per `docs/api_throughput_guidelines.md`).
- CPU: the ridge batteries, batched via shared factorizations as above; projected well under 1 h if batched, ruinous if serial (#823: ~3780 serial fits ≈ 12–20 h — the named anti-pattern).
- Storage: ~72 GB fp16 tensors → pod-side capture + HF upload (`analysis_tensors/`), VM never materializes >50 GB.

## Prior work

**In-line (the primary context):**

- #823 (parent): the map is content-indexed at the answer-mean level; plain external retention 91–98%, mismatched ≈0; identity baseline shows content overlap alone accounts for the retention. This task decomposes that retention by position and query type.
- #779/#722: the map h: c_last(x) → v(x), held-out R² 0.60–0.63, n≈5000 LMSYS; the ridge harness, λ grid, folds, estimand pair reused verbatim.
- #810/#812: mean-pooling the answer side loses nothing resolvable for behavior prediction at n=50 contexts (#812's unpooled-never-beats-mean bound); this task predicts per-position activations as TARGETS (a different question: where the map's information lives, not which summary feeds a behavior predictor) at n≈5000.
- #923/#928 (running siblings): context/query decomposition and CoT decomposition of the same map — orthogonal axes, no overlap with position/divergence.

**External (abstract-verified 2026-07-03 via arXiv export API; planner re-verifies at plan time):**

- Closest formalizations — none does context-representation → per-position ACTIVATION regression, none is on/off-policy contrastive, none has a divergence arm: **Future Lens** (2311.04897 — learned linear readout from a single hidden state at position t to token IDENTITIES at t+2+, GPT-J), **ParaScopes** (2511.00180 — residual-stream decoders probing activations for paragraph-scale upcoming TEXT), **Jump to Conclusions** (2303.09435 — learned linear shortcut maps across LAYERS; this task generalizes the template across POSITIONS), Tuned Lens (2303.08112), Patchscopes (2401.06102 — the readout-family framing), pre-caching vs breadcrumbs (2404.00859 — whether position-t features are computed FOR future positions; the frame for interpreting why c_last predicts answer activations at all).
- Position-resolved in-context adaptation: induction heads (2209.11895 — the canonical loss-at-late-minus-early-positions in-context-learning score; the F16-vs-L16 split is exactly this axis, in activation-prediction space). No verified prior measures loss-vs-position on ANOTHER MODEL's teacher-forced text — the surprisal companion fills a real gap.
- Off-policy text is systematically off-distribution for Qwen: **Idiosyncrasies in LLMs** (2502.12150 — 97.1% five-way classifier accuracy distinguishing ChatGPT/Claude/Grok/Gemini/DeepSeek outputs; word-level, persists under rewriting), **Binoculars** (2401.12070 — cross-model perplexity contrast separates whose distribution a text came from, zero-shot), Anthropic sycophancy (2310.13548) + the Sonnet 4.5 system card's documented multi-part educational-refusal structure (Claude-distinctive refusal-adjacent text).
- Divergence-bank sources: see Phase 3 (promptfoo CCP-sensitive-prompts, augmxnt/deccp, 2505.12625, 2511.23174, 2411.10683, 2602.06371).

## Open design points (clarifier/planner)

- Layer grid thinning (11 → fewer) vs storage; whether the layer axis is selected per (arm, slot) or globally.
- Exact prefix-t grid + the ≥t+16 survival rule vs a fraction-of-pool floor.
- Divergence-margin thresholds + judge rubric wording; whether style/formatting survives as a category after verification.
- Whether the distinct-style arm also gets divergence-bank coverage (default: LMSYS only).
- Secondary pooled-prefix predictor variant: include or defer.

## Provenance

- origin: user chat 2026-07-03 (verbatim originating prompt in frontmatter `origin_prompt:`).
- Design decisions confirmed in chat 2026-07-03: new child task of #823; broad quirk taxonomy for the divergence bank (~4 categories, generation-verified, matched controls); all four #823 arms carried through the per-token analysis.
