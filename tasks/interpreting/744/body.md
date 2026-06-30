---
title: Standardized token-to-token continuity in Qwen-2.5-7B is depth-flat and short-horizon
  — the late-layer directional persistence seen in GPT-2 does not appear (MODERATE
  confidence)
kind: experiment
tags: []
created_at: '2026-06-30T00:55:35Z'
has_clean_result: false
origin_prompt: Do a deep literature dive on if there's literature on how much each
  activation in a LLM relates to the next one and if there's discontinuities | (clarified)
  over tokens, and also how this changes over layers | sure please run it in the background
  with happy coder
goal: Characterize how standardized similarity and directional continuity between
  consecutive-token residual-stream activations vary across layers in Qwen-2.5-7B,
  and where discontinuities concentrate (token type, surprisal, syntactic boundary).
---
# Standardized token-to-token continuity in Qwen-2.5-7B is depth-flat and short-horizon — the late-layer directional persistence seen in GPT-2 does not appear (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- Standardized one-step direction preservation peaks at **~0.13 |cos| by layer 10**, dips to **~0.11 at layer 18**, rebounds to **~0.13 by layer 27** — broadly stable, no GPT-2-style late ramp to 0.54.
- Directional memory is short-horizon at every depth: preservation collapses from **~0.40 at the current step to ~0.13 one token later** at both a mid-layer and the last layer (all 28 layers).
- The pattern is identical in Qwen-2.5-7B-Instruct (base and instruct std +1 agree within **0.003** at every layer), so it is a model-family property, not instruct-tuning.
- Late raw-cosine smoothness (**raw +1 ≈ 0.18 vs standardized ≈ 0.13** at the last layer) is anisotropy: standardizing removes it; dropping the top-3 cosine-dominating dimensions changes nothing.
- Discontinuities concentrate overwhelmingly at attention-sink tokens (**~20× the non-sink jump at layer 3, ~3.5× by layer 25**); high-surprisal tokens carry a smaller real elevation (**~+10 units high vs low on WikiText-103**); clause-openers show none.

## Goal

- **This experiment in context:** This is a fresh measurement line (no parent experiment). The closest published measurement, Barenholtz's *Trajectory Dynamics in Language Model Hidden States*, fits a short linear trajectory to consecutive residual-stream states and reports that direction preservation is short-horizon in the middle layers (GPT-2 layer 6: 0.10 one step out) but persistent in the final layer (GPT-2 layer 12: 0.54 across one, two, and three steps) — a distinct late-layer regime, replicated on GPT-2 and Pythia. The single deliberate change here is the base model: does that depth-graded directional-persistence pattern appear in a modern 7B model? One recipe caveat bounds the cross-model comparison: Barenholtz's direction-preservation figures are absolute cosine on RAW (un-standardized) hidden states — his stated chance baseline of 0.029 is the raw expected |cos| for 768-dim vectors, and his z-scoring is applied only to the reading-time regression predictors, not the hidden states. The read here is primary in STANDARDIZED space (the project default, because raw cosine in deep transformers is dominated by a few runaway dimensions), so the strict apples-to-apples comparison is Qwen's RAW curve: raw +1 ≈ 0.18 at the last layer, still far from GPT-2's 0.54 and still showing no late ramp. The conclusion survives in both spaces; the standardized read is the headline because it is the artifact-free one.
- **Broader narrative:** No consecutive-token similarity-vs-layer curve existed for any Qwen model; this fills that gap and gives a mechanistic reference for whether a signal stays directionally coherent as it propagates across layers. The relevance to the project's persona work is that a representation which drifts coherently late (the Barenholtz pattern) versus one that recomputes locally at every depth implies different things about where a persona direction is stable enough to read or steer.

## Methodology

**Design:** A pure forward-pass measurement on residual-stream activations over natural text — no training, no model edits. For every layer L (all 28) and every consecutive token pair (t, t+1), three quantities are computed: (1) standardized consecutive-token cosine, (2) Barenholtz direction preservation — the absolute cosine between a 3-state linear-trajectory fit and the actual one-/two-/three-step displacement, and (3) per-position jump magnitude stratified by token type. The single manipulated variable relative to the published recipe is the model; a base arm (Qwen-2.5-7B, the primary, directly comparable to Barenholtz's base models) and an instruct arm (Qwen-2.5-7B-Instruct) were both dumped. Two corpora are read separately and never silently pooled: Natural Stories (10 narrative sequences, the cross-paper comparability corpus) and WikiText-103 (7,389 paragraphs, which carries the statistical weight).

**Training:** N/A — no model training. Analysis-design constants:

| Parameter | Value | Source |
|---|---|---|
| Models (arms) | Qwen/Qwen2.5-7B (primary), Qwen/Qwen2.5-7B-Instruct | dump_manifest.json |
| Layers × hidden | 28 × 3584 | Qwen-2.5-7B config; asserted at load |
| Trajectory window k | 3 (linear least-squares fit) | Barenholtz 2606.05346 §Method (3-word linear fit, best config) |
| Direction-preservation steps | +0, +1, +2, +3 | Barenholtz §Direction Preservation Analysis |
| Rogue-dim ablation | top-3 by raw variance | Timkey & van Schijndel 2109.04404 (top-3 anisotropy table) |
| Standardization | per-dimension z under fixed population mean/std, per layer | Timkey 2109.04404 (standardize before any token-to-token cosine) |
| Max sequence length / NS chunk stride | 1024 / 512 (overlapping chunks) | Barenholtz §Materials (1024 context, overlapping chunks) |
| Sink/outlier mask | abs activation > 100 AND ≥ 1000× layer-median | Sun et al. 2402.17762 §2 |
| Surprisal | each model's own next-token NLL | self-surprisal, standard |
| Random-pair baseline | 100,000 random token pairs (pool 20,000), per flavor, per layer | empirical Qwen chance floor; closed-form d=3584 = 0.013 cross-check |
| Bootstrap | over SEQUENCES, B = 2000, percentile [2.5, 97.5] | plan §6 (positions autocorrelated; sequences are the independent unit) |
| Seed | 744 | plan §11 |

**Evaluation:** The dependent variables are on-distribution by construction — natural token positions, the model's own forward pass over natural prose, every consecutive pair, no teacher-forced canned context. Direction preservation (absolute cosine of directions) is scale-invariant and is the **primary** read; the trajectory-extrapolation L2 error is reported as a **secondary** companion, with the caveat that it lives in per-layer-standardized space and so compares different scales across layers. Each per-layer / per-flavor / per-step statistic carries a bootstrap-over-sequences 95% interval. Every curve is reported in three flavors — raw cosine (the anisotropy-contaminated comparison), standardized, and standardized + top-3-rogue-dim-ablated — compared to its flavor-matched, **per-layer** empirical random baseline (apples-to-apples). The standardized chance floor is not constant across depth: it sits at ~0.022-0.027 in the early-middle layers but climbs to ~0.037 (WikiText-103) / ~0.050 (Natural Stories) at the last layer, with the closed-form d=3584 chance value (0.013) as a depth-invariant anchor. This is a pure mechanism measurement, so the dual-DV content-behavior requirement does not apply (N/A — no behavioral construct, no judge).

**Data extraction:** Corpus A (Natural Stories, tier-2 established psycholinguistic dataset) was fetched from the `languageMIT/naturalstories` GitHub repo at commit `4700daad`, giving 10 stories / 10,256 words / 10 sequences (12,296 tokens), with the gold Penn constituency parses used for the syntactic-boundary mask. Corpus B (WikiText-103 raw, tier-2 established benchmark) was the `Salesforce/wikitext` train split at revision `b08601e04326`, seed-744 shuffled and capped at a 1,000,000-subword-token budget after dropping section-header and sub-32-character lines, yielding 7,389 paragraph sequences. Both corpora are byte-reproducible from (repo, commit/revision, seed, budget). The clause-opener mask is gold-Penn on Natural Stories (first terminal under an S/SBAR constituent, or a coordinator/complementizer) and a closed-class wordlist proxy on WikiText-103 (no gold parses).

**Sample training/evaluation data + completions:** This run generates no model completions — it reads forward-pass activations, so there is no completion sample to quote. The one available worked example is the set of tokens that trigger the discontinuity-locus result. The attention-sink positions that drive the sink jump-ratio are SPARSE — about 10 sink positions across the entire 12,296-token Natural Stories corpus (roughly one per sequence, the sentence-initial token), the Sun et al. massive-activation enrichment class. Decoded from the raw activation dump at the masked positions, they are sentence-initial capitalized tokens and a small number of delimiters — verbatim: `"If"`, `"A"`, `"It"`, `"Once"`, `"At"`, `"This"`, `"T"`, `"Ros"` (the first subword of a story-initial proper name), plus the delimiter tokens `.` and `\n`. The verbatim corpus rows, the per-position sink/surprisal/syntactic masks, the raw fp16 activation dumps, and the full per-layer summaries for both arms are on the HF data repo: [`issue744_token_continuity` @72bb5ddcec](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/72bb5ddcecf6af39cc3a5ef3a8318e7324ecb142/issue744_token_continuity) (`dump_base/`, `dump_instruct/`). Per-sequence and per-stratum numbers behind every aggregate are in the committed CSVs (linked in the footer).

## Results

### One-step direction preservation rises to ~0.13 by layer 10, dips to ~0.11 mid-band, rebounds to ~0.13 — no late-layer ramp

What is plotted: per-layer one-step direction preservation (absolute cosine of the 3-state trajectory fit vs the next-token displacement), Qwen-2.5-7B base, three flavors with bootstrap bands and per-layer random baselines; left WikiText-103, right Natural Stories.

![Per-layer one-step direction preservation for Qwen-2.5-7B base, three flavors with per-layer random baselines, WikiText-103 and Natural Stories panels — the standardized curve rises to a mid-layer peak near 0.13, dips mid-band, and rebounds late without ever climbing toward the GPT-2 value.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0657f52d355ce210b5821c533698491c2f77f087/figures/issue_744/base/h1h2_direction_preservation_p1.png)

> **Figure.** *Standardized one-step direction preservation is broadly stable with depth and never ramps late.* Layer vs +1-step absolute-cosine; raw (gold), standardized (blue), standardized+rogue-ablated (orange, overlapping standardized), dotted = each flavor's per-layer random baseline. WikiText-103 n = 7,389; Natural Stories n = 10.

The standardized curve climbs from ~0.06 to a mid-layer peak of ~0.13 (layer 10), dips to ~0.11 mid-band (layer 18), rebounds to ~0.13 late (layer 27) — a ~20% peak-to-trough variation, broadly stable but not flat. It sits ~5.5× its per-layer chance floor mid-layer and ~3.5× at the last layer, above chance everywhere but never climbing toward GPT-2's 0.10 → 0.54 ramp. The ablated curve overlaps the standardized one; the raw late lift (0.177 at layer 27) is the anisotropy standardization removes — and even raw 0.177 is far below GPT-2's raw 0.54, so the missing ramp is not a standardization artifact.

### Directional memory is short-horizon at every depth — the last layer decays like the middle

What is plotted: the +0 / +1 / +2 / +3 direction-preservation decay (the Barenholtz Table-4 analogue) at the mid-band reference layer (13) and the late-band reference layer (27), WikiText-103, base arm, three flavors.

![Direction-preservation decay from the current step to three steps ahead at the mid-band (layer 13) and late-band (layer 27) layer of Qwen-2.5-7B base — both collapse from about 0.40 at the current step to about 0.13 one step later.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0657f52d355ce210b5821c533698491c2f77f087/figures/issue_744/base/h1_decay_profile_broader.png)

> **Figure.** *The last layer's trajectory dies after one token, exactly like the middle.* Direction preservation at +0/+1/+2/+3 for the mid-band (layer 13) and late-band (layer 27) layer, WikiText-103, three flavors. Both collapse from ~0.40 to ~0.13 — the opposite of a persistent late plateau.

At both layers the standardized profile collapses from ~0.40 at the current step to ~0.13 one step out (WikiText-103 layer 13: 0.40 → 0.12 → 0.09 → 0.08; layer 27: 0.43 → 0.13 → 0.10 → 0.10) — near-Markov, with directional structure that effectively dies after a single token. The all-layers small-multiples figure shows the identical collapse at every depth, and the word-level (last-subword-per-word) read holds it too (layer 27: 0.43 → 0.13), so the missing late persistence is not a two-layer cherry-pick or a subword-granularity artifact.

> All-layers decay small-multiples and the word-level read: committed figure dump (`decay_all_layers_small_multiples.png`) + CSV (`ns_word_level_continuity.csv`).

### Base and Instruct agree — the depth-flat pattern is a model-family property, not instruct-tuning

What is plotted: per-layer one-step standardized direction preservation, base arm vs instruct arm overlaid, both corpora, with the per-layer standardized random baseline. Base = circles, instruct = squares.

![Base versus Instruct one-step standardized direction preservation, both corpora — the two curves overlap within about 0.003 at every layer and are both broadly stable near 0.12.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0657f52d355ce210b5821c533698491c2f77f087/figures/issue_744/base/h1h2_base_vs_instruct.png)

> **Figure.** *Instruct-tuning leaves token-to-token continuity unchanged.* +1-step standardized direction preservation, Qwen-2.5-7B base (circles) vs Instruct (squares), WikiText-103 (n=7,389) and Natural Stories (n=10), dotted = per-layer standardized chance floor (climbing from ~0.022 to ~0.037/0.050 late). The arms track each other within ~0.003 at every layer.

The two curves are nearly coincident at every layer (max WikiText-103 difference 0.002 over 28 layers; max Natural Stories difference 0.003; WikiText-103 layer 27: base 0.128, instruct 0.126), and the decay profiles match within ~0.01 (instruct layer 27: 0.43 → 0.13 → 0.10 → 0.09). So the depth-flat, short-horizon regime is a property of the Qwen-2.5-7B architecture/pretraining, not something instruct-tuning adds or removes. The null difference does not license "universal across model families" — only base and instruct of one 7B model were measured.

### Most individual sequences cluster near ~0.12 — the broadly-stable curve is not a sink-composition artifact

What is plotted (the per-unit data behind the aggregate): per-sequence one-step standardized direction preservation at the mid-band (13) and late-band (27) layer, base arm. WikiText-103 as a jittered strip (200-sequence raw subset) with the mean as a diamond; 10 Natural Stories sequences as labeled points; dotted per-(corpus, layer) baseline.

![Per-sequence one-step direction preservation at a mid-band and late-band layer of Qwen-2.5-7B base — WikiText-103 and Natural Stories sequences cluster near 0.12 with a tail spanning roughly 0.04 to 0.18, and none showing late-layer persistence.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f63291011bd3cdecacab8fdb152813f7258a0769/figures/issue_744/base/h1h2_per_sequence_scatter.png)

> **Figure.** *No sequence carries late persistence the aggregate hides.* Per-sequence one-step standardized direction preservation at layer 13 (mid) and layer 27 (late), base arm; WikiText-103 jittered strip (200-sequence raw subset) with mean diamond, 10 labeled Natural Stories sequences. Dotted = per-(corpus, layer) chance floor (layer 13 ~0.023/0.026, layer 27 ~0.037/0.050); every sequence sits far above it.

Individual WikiText-103 sequences span roughly 0.04 to 0.18 (median 0.124, 10th-90th percentile 0.10-0.14), the bulk clustered near the per-layer mean of ~0.12; the spread is genuine but no sub-population shows a late-layer climb. The lead alternative — that a few massive-norm sink tokens prop up the broadly-stable late value — is ruled out: the sink-excluded Natural Stories curve reproduces the all-positions value to four decimals at every layer (layer 27: 0.1285 excluded vs 0.1285 all). Sinks dominate jump magnitude (next result) but are too rare to move the direction-preservation aggregate.

### Discontinuities concentrate at attention-sink tokens, with a smaller real elevation at high-surprisal tokens

What is plotted: per-stratum mean standardized jump as a token-type × layers heatmap, Natural Stories base arm.

![Per-token-type mean jump heatmap for Qwen-2.5-7B base on Natural Stories — the attention-sink row dominates the colorscale; surprisal and clause-opener rows look uniform at this scale, which the caption flags is because the sink row is ~20× larger.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0657f52d355ce210b5821c533698491c2f77f087/figures/issue_744/base/h3_stratification_heatmap_ns.png)

> **Figure.** *Sink tokens dominate the jump scale; the smaller surprisal effect lives below it.* Token type (sink/non-sink, surprisal terciles, clause-opener/interior) × layer, color = mean standardized jump, Natural Stories base arm. The sink-dominated colorscale (~1200 vs ~60) hides the real ~+4 to +10 high-vs-low surprisal gap, read off the per-layer CSV.

Sink tokens jump far harder than everything else, only between layers 3 and 25 (absent at 0-2 and 26-27). The sink-dominated colorscale (~1200 vs ~60) hides a smaller but real second effect: high-surprisal tokens DO jump more than low-surprisal ones — on WikiText-103 a ~+10-unit gap with disjoint CIs (layer 10: 59.5 vs 49.3), on Natural Stories a smaller ~+4 (61.0 vs 56.6, CIs nearly touching). Clause-openers jump slightly *less* than clause-interior tokens, so the syntactic-boundary signal does not localize here. Caveat: the sink stratum is partly definitional (its mask derives from the same hidden states whose jump it predicts), and "sinks dominate" is read off a 28-layer × 3-stratifier surface. The surprisal and syntactic strata are not circular.

### Sink dominance is largest early and decays overall with a small mid-layer rebound

What is plotted (the per-layer view behind the heatmap's sink row): the sink/non-sink mean-jump ratio per layer, Natural Stories base arm, over the layers where sinks fire; dotted = parity.

![Sink to non-sink jump-ratio versus layer for Qwen-2.5-7B base on Natural Stories — falling from about 20 times at layer 3, dipping to a local minimum near layer 8, rebounding slightly to layer 10, then declining to about 3.5 times at layer 25.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0657f52d355ce210b5821c533698491c2f77f087/figures/issue_744/base/h3_sink_ratio_depth.png)

> **Figure.** *Sink-jump dominance peaks early and tapers late, with a small mid-layer rebound.* Sink/non-sink mean-jump ratio vs layer, Natural Stories base arm, layers 3-25 (where sinks fire); dotted = parity. Ratio falls from ~20× (layer 3) to ~3.5× (layer 25), with a small rebound (layer 8 ≈ 10.9 → layer 10 ≈ 11.7) before the decline resumes.

The ratio falls from ~20× at layer 3 to ~3.5× at layer 25 — sinks always jump more, but their dominance tapers with depth. The decline is not strictly monotone: it dips to a local minimum at layer 8, rebounds slightly to layer 10, then resumes the descent. The sink stratum is sparse — about 10 positions across the 12,296-token corpus — so the per-layer sink mean rests on few positions and its bootstrap CI is only an order-of-magnitude read. The WikiText-103 wordlist clause-opener proxy agrees with the gold-Penn mask on 81.9% of Natural Stories terminals but under-fires, so syntactic conclusions rest on the gold parses.

---

**Repro:** ~3 GPU-h (estimated) on 1× A100-80, GCP flex-start, zone us-central1-c (`a2-ultragpu-1g`), both arms; Phase-2 analysis + figures ran CPU off-pod on the VM (~8 min analyze + ~3 min figures per arm). Code SHA [0657f52d35](https://github.com/superkaiba/explore-persona-space/tree/0657f52d355ce210b5821c533698491c2f77f087) — analysis [`scripts/issue744_analyze_continuity.py`](https://github.com/superkaiba/explore-persona-space/blob/0657f52d355ce210b5821c533698491c2f77f087/scripts/issue744_analyze_continuity.py), continuity primitives [`src/explore_persona_space/analysis/continuity.py`](https://github.com/superkaiba/explore-persona-space/blob/0657f52d355ce210b5821c533698491c2f77f087/src/explore_persona_space/analysis/continuity.py), figures [`scripts/issue744_make_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/0657f52d355ce210b5821c533698491c2f77f087/scripts/issue744_make_figures.py) + [`scripts/issue744_make_figures_supp.py`](https://github.com/superkaiba/explore-persona-space/blob/0657f52d355ce210b5821c533698491c2f77f087/scripts/issue744_make_figures_supp.py) + [`scripts/issue744_make_figures_arms.py`](https://github.com/superkaiba/explore-persona-space/blob/0657f52d355ce210b5821c533698491c2f77f087/scripts/issue744_make_figures_arms.py). Per-arm per-layer / per-stratum / per-sequence CSVs + random-baseline + gold-vs-proxy JSONs: [`eval_results/issue_744/base`](https://github.com/superkaiba/explore-persona-space/tree/0657f52d355ce210b5821c533698491c2f77f087/eval_results/issue_744/base) and [`eval_results/issue_744/instruct`](https://github.com/superkaiba/explore-persona-space/tree/0657f52d355ce210b5821c533698491c2f77f087/eval_results/issue_744/instruct). The five direction-preservation / stratification figures are pinned to `0657f52d35` (their committed `.meta.json` carries an earlier figure-build commit — a known lag of the build script; the PNGs served at the pinned SHA are the current ones). The per-sequence scatter (`h1h2_per_sequence_scatter.png`) was re-rendered at `f63291011b` to draw the per-(corpus, layer) standardized chance floor in place of a single flat line — its data points are unchanged from the `0657f52d35` build (only the baseline segments changed); it is pinned to `f63291011b`, and its committed `.meta.json` shows the same lag (the PNG served at the pinned SHA is the current per-(corpus, layer) baseline render). Raw fp16 activation dumps (10 Natural Stories sequences, full 28-layer states + per-position masks + surprisal), per-sequence summaries (7,389 WikiText-103 + 10 Natural Stories), random-pair pools, population stats, corpus JSONs + Penn parses, per arm: [HF data repo `issue744_token_continuity` @72bb5ddcec](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/72bb5ddcecf6af39cc3a5ef3a8318e7324ecb142/issue744_token_continuity) (`dump_base/`, `dump_instruct/`). Dumps produced at git commit `67d01ff40e`. No reused trained artifacts (fresh forward-pass dump). Each figure is pinned to the commit named for it above.

**Context:** Originating request (verbatim):

> Do a deep literature dive on if there's literature on how much each activation in a LLM relates to the next one and if there's discontinuities | (clarified) over tokens, and also how this changes over layers | sure please run it in the background with happy coder

Fresh direction (no parent). Created, dumped, and analyzed 2026-06-30. Both arms were dumped on one GCP run after two earlier launch failures (an OOM-kill on an undersized 16 GB host, then a missing-dependency crash). The Phase-1 uploader initially wrote both arms to one shared path, so the instruct arm overwrote the base arm; both arms were subsequently re-homed to arm-distinct paths (`dump_base/`, `dump_instruct/`) on the HF data repo, which is the source for all numbers here. The device-consistency and overlapping-chunk-truncation bugs from earlier rounds were fixed before the dump that produced these results.
