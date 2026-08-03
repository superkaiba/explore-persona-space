---
title: Conversation history alone carries about half the full-context answer-state
  map on 100k real multi-turn conversations (HIGH confidence)
kind: experiment
tags: []
created_at: '2026-07-28T00:42:49Z'
has_clean_result: true
parent_id: 1482
origin_prompt: 'run for 100k as much in parallel and vectorized as possible. [context:
  user approved option R3(b) from chat — ''new capture: build the multi-turn analogue
  of the n1M corpus — N multi-turn LMSYS/WildChat conversations (prefix = history,
  query = last turn), capture prefix-end + context-end + answer states, fit both arms,
  run the full pipeline'' — after a sizing discussion grounding N=100k at ~50–70 GPU-h
  capture on the measured 3,250 ctx/GPU-h parent basis]'
workflow: v1
goal: 'Build the multi-turn analogue of the #779 fitter-fair-comparison-n1m corpus
  at N≈100,000 real multi-turn LMSYS/WildChat conversations (prefix = full conversation
  history before the last user turn, ≥2 user turns; query = last user turn), generate
  one on-policy answer per context under the parent decoding recipe, capture layer-{14,19,26}
  prefix-end + context-end + mean-answer states under the parent capture convention,
  fit the five parent fitters in BOTH arms (prefix-based AND context-based) on a pinned
  near-dupe-gated split, and run the #1482 error-characterization pipeline on both
  arms: (1) prefix-arm transport at scale (held-out R² per arm/layer vs the context
  arm and vs #1092''s 0.05–0.11); (2) per-context error + judged taxonomy incl. conversation-depth,
  floor-relative via a K-resample answer-sampling floor; (3) per-direction answer-PCA
  linear-vs-nonlinear decomposition with shrinkage control, cross-arm; (4) identity+learned-bias
  baseline and kNN-retrieval reads per arm. Binding user directive: maximize parallelism
  and vectorization at every phase (wide sharded capture fleet, batched fits, no serial
  inner loops). Phase 0 CPU manifest probe verifies multi-turn supply before any GPU
  provisions; a 1-shard pilot re-measures multi-turn throughput before fleet sizing
  (plan basis ~1,500–2,000 ctx/GPU-h vs parent''s measured single-turn 3,250).'
relates_to:
- spec-context-as-vector
---
# Conversation history alone carries about half the full-context answer-state map on 100k real multi-turn conversations (HIGH confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_1738.md](https://github.com/superkaiba/explore-persona-space/blob/28add2f79f25784c2db65940443a06514099245b/docs/methodology/issue_1738.md) · [gist mirror](https://gist.github.com/superkaiba/974da25791408751959e3910e386d371)

## Takeaways

- Conversation history alone predicts the upcoming answer state at scale: prefix-arm held-out R² 0.379 (layer 19, ridge, n=9,941) vs 0.681 for the full context — ~56% recovered, far above the 0.11 single-turn reference.
- The final user query alone does better still — bare-query R² 0.534 recovers ~78% — and the two partial inputs are complementary: bare beats prefix on 69.2% of contexts yet loses on roleplay and deep threads, with no crossover at any exact depth.
- A crossed companion corpus (5,000 histories × 20 shared real queries, 100k rows) separates the sources: query identity carries 63% of per-row answer-state variance, history 7%, and the per-row history map collapses to R² 0.046 — near its own share's ceiling.
- Averaging the 20 queries away recovers the history signal: the per-row full-context map applied to per-prefix averages reads R² 0.745 over 500 held-out prefixes, matching an independently-fit averaged map (0.757) — one operator serves both grains — and history-end plus bare-query jointly recover 85% of the full-context map.
- Answer-sampling noise is small (median 6.1% of per-context error); prefix error concentrates where the final query pivots away from the history (English, social chitchat, translation); every map is fitted, not a constant shift (identity-plus-bias R² negative in all arms).
- The maps survive re-basis into 16,384 interpretable sparse-autoencoder answer features: history features recover the same ~56% share (0.359 vs 0.638), the context advantage holds on 99.7% of features, and the crossed query-over-history share ordering reproduces per feature.

## Goal

- **This experiment in context:** This experiment tests whether conversation history alone — the prefix, everything before the final user turn — already carries the upcoming answer's representation on real multi-turn data at scale. The single-turn 963k-context corpus line ([#779](https://eps.superkaiba.com/tasks/779)) established the context-to-answer map and the five-fitter comparison reused here, but its prefix is a constant chat-template string, so a prefix arm is structurally empty there. The crossed 21k-context decomposition ([#1092](https://eps.superkaiba.com/tasks/1092)) measured prefix-end R² 0.05–0.11 — under crossed persona-and-query pairings, a grouped 6-fold scheme, layer 14, and ~21k rows, vs this run's natural conversations, pinned single split, and 9,941-context holdout — not directly comparable, so 0.11 enters only as a fixed reference constant. The per-context error taxonomy, K-resample answer-sampling floor, and per-direction reads come from the error-characterization battery of [#1482](https://eps.superkaiba.com/tasks/1482), applied here to both mapping arms. A same-question follow-up round then completed the input-object set on this corpus — history alone (prefix), final query alone (the bare-query empty-system render convention of [#1092](https://eps.superkaiba.com/tasks/1092)), and full context — mapped to the same answer-state targets on the same pinned split. A second follow-up round re-based the same maps into sparse-autoencoder feature space under the fitness-gate and feature-restriction recipe of [#1482](https://eps.superkaiba.com/tasks/1482), testing whether the input-arm ordering survives at interpretable-feature grain. A third follow-up round built the crossed companion corpus this decomposition needs — the same conversation histories each paired with one shared bank of twenty real final queries, scaling the crossed decomposition design of [#1092](https://eps.superkaiba.com/tasks/1092) to 5,000 multi-turn prefixes — completing the four-input-object set (bare query / history / full context / question-averaged context) and separating query-borne from history-borne variance at scale.
- **Broader narrative:** The context-as-vector question: is the pre-answer residual state a sufficient statistic for the upcoming answer state, and how much of that statistic exists before the final query arrives? Measuring what history alone determines — and where it fails — bounds what the final query adds and what a history-only representation monitor could read ahead of time.

## Methodology

**Design:** One corpus build, no model training. I fit the same map in both input arms — prefix-based (input: the prefix-end residual state, where the prefix is everything before the final user turn) and context-based (input: the context-end state, prefix plus final user turn) — against identical mean-answer targets, at layers 14, 19, and 26 with five fitters per arm and layer (30 cells) on one pinned split, plus a seed-43 repeat at layer 19 and an LMSYS-only refit scored on WildChat rows as a corpus-transfer control. Relative to the single-turn parent corpus, the one manipulated variable is corpus construction (natural multi-turn conversations); every capture and fit constant is inherited. A same-issue follow-up round added a third input arm — the bare query: the final user turn rendered with an explicit empty system turn (no history, and per-row hard asserts that the tokenizer's default system prompt is not silently injected and that the render string-matches the template prefix/suffix around the query) — captured in right-padded bf16 batches gated by a 32-row batched-vs-per-row parity probe (per-layer minimum cosine ≥ 0.999 required; 0.9997–0.9999 measured), then fitted with the same five fitters, layers, targets, and pinned split (split hashes cross-asserted against the parent fit record; 15 new cells, 45 total). The seed-43 repeat was not extended to this arm. A second follow-up round (`sae-arm`) re-read the same captured contexts through a public BatchTopK sparse autoencoder (k = 64 active features per token, dictionary 131,072, residual stream layer 19): one teacher-forced bf16 forward per context re-captured the answer-span layer-19 states under the parent's render convention (per-row render and span identity gate; 1 of 99,127 rows dropped, train 87,794, holdout unchanged at 9,941), the autoencoder encoded the answer tokens (pooled mean, max, and fraction-active per feature) and the stored history-end / context-end states, and ridge maps were fit for eight cells — sparse-feature inputs and the same states as dense inputs onto identical mean-pooled answer-feature targets, plus the max / fraction-active pooling twins — on the same pinned split (hashes cross-asserted). A third follow-up round (`crossed-multiturn-averaged`) built a crossed companion corpus: 5,000 real conversation histories, all drawn from this run's own manifest (overlap fraction 1.0; seeded draw under the same eligibility and near-dupe gates), fully crossed with one shared bank of 20 real final user turns (last user turns of conversations not used as prefixes, near-dupe-gated against the prefix set and within the bank, byte-frozen with a sha-pinned bank file) — 100,000 contexts, complete by construction because a prefix is eligible only if its render with the longest bank query fits the 7,104-token budget. One on-policy answer was generated per cell under the parent decoding recipe and captured at the same layers under the parent convention (per-row strict-token-prefix identity gate: zero violations, zero over-length skips, zero empty responses), plus a one-off capture of the 20 bank queries under the bare-render convention. Ridge is that round's sole fitter (the five-fitter comparison was settled on the main corpus — a stated refinement); the split is prefix-grouped and pinned — 4,430 / 20 / 50 / 500 prefixes for train / val / test / holdout, i.e. 88,600 / 400 / 1,000 / 10,000 rows — with the penalty selected on the pinned validation rows (the question-averaged arm selects by inner cross-validation within the training prefixes).

**Training:** **N/A — no model training.** The complete generation / capture / fitting / judging hyperparameter table:

| Hyperparameter | Value | Source |
|---|---|---|
| Base model | Qwen-2.5-7B-Instruct | plan §11; run record |
| Answer decoding | temperature 1.0, top_p 0.95, max_new_tokens 1,024, engine seed 42 | plan §11 (parent decoding recipe) |
| Generation length budget | 7,104 tokens (prompt + answer); over-budget contexts skipped and recorded | capture driver `issue1738_multiturn_generate_capture.py` at the code pin; run record (651 over-length skips of the 99,778 manifest rows; an earlier 873 figure conflated these with the 222 near-dupe manifest drops) |
| Capture layers | 14, 19, 26 | plan §11 |
| Captured states | prefix-end; context-end; mean-answer (generated span incl. end-of-turn tail) | plan §11 |
| Prefix-end read | one forward per row; strict-token-prefix indexing at position `prefix_len` − 1; per-row assert; ≤0.5% pilot violation gate | plan §11 |
| Ridge | 23 log-spaced penalties, 1e−3 to 1e8 | plan §11; fits JSON `lambdas` |
| MLP | widths 8,192 and 32,768; learning-rate grid 1e-3, 3e-4; ≤300 epochs; batch 4,096 | plan §11; fits JSON `mlp_lr_grid` |
| Residual-skip fitter | parent five-fitter configuration (linear skip plus learned residual) | plan §11 |
| Kernel fit (Nyström KRR) | m = 16,384 centers; gamma multiplier 1.0; penalty grid 0.1 and 10; Cholesky solve | plan §11; fits JSON `krr` |
| Split | train 87,795 / val 396 / test 995 / holdout 9,941; sha-pinned | fits JSON `split_counts`, `split_shas` |
| Near-dupe gate | 5-gram character Jaccard ≥ 0.8 against val, test, and holdout; 222 train rows dropped | plan §11; manifest `meta.json` |
| Fit seeds | 42 primary; 43 repeat at layer 19 (MLP width 8,192) | fits JSON |
| Primary transport criterion | prefix R² minus 0.11; 95% bootstrap CI must exclude 0 | plan §7 |
| Bootstrap CIs | 10,000 draws per cell and contrast | fits JSON `n_boot`; `h1_contrast.json` |
| K-resample floor | K = 4 fresh answers per context, seeds 43–46; 2,000-context stratified subsample, 1,988 kept | plan §11; kresample JSONs |
| Per-direction battery | top-256 answer-PCA directions; per-direction penalty control over a 38-value grid | plan §11; `pdshrink_summary.json` |
| Taxonomy statistics | 10,000-draw context bootstrap; 10,000-draw permutation p; Benjamini–Hochberg false-discovery rate q = 0.05; seed 1738 | `taxonomy.json` |
| Judge model | claude-sonnet-4-5-20250929, Anthropic Batch API, reason-then-label, max_tokens 400, temperature API default, 1 draw | `labels.json` |
| Judge excerpt caps | final user turn ≤1,200 chars; history tail ≤800; response ≤1,000 | `labels.json` |
| Judge reliability | test-retest κ on 200 items: language 0.982, topic 0.879, refusal-adjacent 0.892, answer-is-refusal 0.827, format 0.786; demotion threshold 0.6; all five axes kept | `labels.json` |
| Bare render (follow-up round) | empty-system chat template around the final user turn; per-row default-system-injection and string-identity asserts | plan v6 §4.1 (the [#1092](https://eps.superkaiba.com/tasks/1092) convention) |
| Bare capture batch / chunk | right-padded batch 64, bf16 (tail chunk re-captured at batch 8 after an out-of-memory stop); chunks of 2,000 rows | plan v6 §4.1; `bare_pilot_meta.json` |
| Bare parity gate | 32-row batched-vs-per-row capture, per-layer minimum cosine ≥ 0.999 (measured 0.9997–0.9999) | plan v6 §4.1; `bare_pilot_meta.json` |
| Bare fit seeds | 42 only (seed-43 repeat not extended to this arm) | plan v6 §10 |
| SAE suite (follow-up round) | `andyrdt/saes-qwen2.5-7b-instruct` revision `c37e53c4`, residual stream layer 19, BatchTopK k = 64 primary (k = 128 robustness twin), dictionary 131,072 | plan v8 §11 (the [#1482](https://eps.superkaiba.com/tasks/1482) suite); `sae_pilot_meta.json` |
| SAE fitness gate | own-token fraction of variance explained ≥ 0.75 and L0 within 2× of 60 (bounds 30–120); measured 0.797 / L0 64.1 on 154,996 tokens (published reference 0.806) | plan v8 §11; `sae_pilot_meta.json` |
| SAE feature restriction | 16,384 answer-side / 8,192 input-side most-active caps at a 1%-of-train-rows activity floor (878 rows); realized inputs 216 (history) / 530 (context) features | plan v8 §11 (the #1482 recipe at this run's n); fits JSON `restriction` |
| SAE floors and nulls | shuffled-pairing floor K = 20 draws per arm (penalty pinned, permute within fit, scored on the true holdout); per-feature label-shuffle null at the 95th percentile | plan v8 §11 (the #1482 recipe); fits JSON `shuffle_floor` |
| Crossed corpus (follow-up round) | 5,000 prefixes × 20 shared bank queries = 100,000 rows; prefixes drawn from this run's manifest (overlap fraction 1.0); depth histogram 2,092 / 1,655 / 1,253 (2 / 3–4 / ≥5 user turns); 2,520 LMSYS / 2,480 WildChat | plan v9 §4.1; crossed `sampling_manifest/meta.json` |
| Crossed query bank | 20 real final user turns from conversations not used as prefixes; near-dupe-gated against the prefix set and within-bank; byte-frozen, sha-pinned; pool 200 candidates (7 length rejects, 1 within-bank dupe reject); no per-prefix substitutions; one bank query (query id 0) is a bare URL — its 5,000 answers are low-diversity echoes, a disclosed bank-composition quirk | plan v9 §4.1; crossed `bank.json`, `meta.json`; raw-completions spot check |
| Crossed split | prefix-grouped, pinned: 4,430 / 20 / 50 / 500 prefixes (train / val / test / holdout) → 88,600 / 400 / 1,000 / 10,000 rows; sha-pinned | plan v9 §4.1; `crossed_fits.json` |
| Crossed fitter | ridge only (parent shared-Gram recipe, same 23-penalty grid); per-row arms select the penalty on the pinned val rows; averaged arm by inner cross-validation within train prefixes | plan v9 §4.3; crossed `cells/*.json` |
| Crossed gates | throughput pilot 6,439 contexts/GPU-h → 18.0 GPU-h projected, under the 24 GPU-h fence; min pairwise prefix-end cosine 0.48–0.58 per layer (degeneracy bar 0.999); bank byte-identity pass; 0 render violations | plan v9 §7; `crossed_pilot_meta.json` |
| Crossed variance decomposition | prefix / query / interaction-plus-residual shares of the mean-answer state on the complete 5,000 × 20 grid, overall (sum-of-squares-weighted over dimensions) and per top-48 answer-PCA direction (train-fit); one temperature-1.0 draw per cell; parent K-resample floor embedded as a reference comparator | plan v9 §6; `anova.json` |
| Crossed operator geometry | principal angles between top-k left / right singular subspaces of the arm operators at matched penalty (k = 48 and k at 90% spectral energy) vs 200 spectrum-matched random-subspace draws; inputs and targets mean-centred; mean-norm fractions reported | plan v9 §6; `operator_geometry.json` |
| Crossed SAE fold-in | same autoencoder suite and k; fitness re-measured on this corpus's tokens: fraction of variance explained 0.788 at k = 64, L0 66.8 (gate ≥ 0.75 / 30–120) on 150,000 tokens; per-feature shares with 20 label-permutation draws (seeds 1738000–1738019); judged labelling frozen — 0 judge calls | plan v9 §6; `crossed_pilot_meta.json`; `sae_perfeature.json` |

**Evaluation:** The dependent variable is pooled held-out R² between the predicted and the realized mean-answer state over the 9,941 pinned held-out contexts — a decodability read on the model's own on-policy answers (no behavior-expression judging, so the dual-DV rule does not apply; no cell saturates, values span 0.30–0.72). The per-context companion is the normalized error `nerr(x) = ||v_hat(x) − v(x)||² / ||v(x) − mu_eval||²`. One target asymmetry binds every prefix-arm read: the answer whose mean state is the target was generated under the full context, so the prefix arm predicts a representation produced with information (the final user turn) its input never contains — prefix R² is a lower bound on history-only transport, and the prefix-to-context difference mixes genuinely missing query information with map error. The same asymmetry binds the bare-query arm (its input omits the history the answer was generated with), so bare R² is likewise a lower bound on query-only transport. Standing mapping reads run per arm: the identity-plus-learned-bias baseline (input and output share dimension 3,584, so it applies) and retrieval among the held-out targets (euclidean and cosine, k in 1/5/10, chance 1/9,941 ≈ 0.01%). Because near-dupe gating was conversation-level, identical short final queries can recur across train and holdout with different histories — an interpolation advantage specific to the bare arm — so the bare headline carries an exact-duplicate final-query control: a train-holdout string match plus a companion R² restricted to non-duplicate holdout rows. A follow-up re-analysis on the committed per-context table (`scripts/issue1738_depth_rebin.py`) re-bins the depth reads at exact user-turn counts (2–8, ≥9), keeping the coarse depth contrasts' subset-mean-denominator R² convention, with per-stratum dominance fractions and 95% CIs from a 2,000-draw vectorized within-stratum bootstrap. Sparse-autoencoder-round reads gate on the suite's fitness re-measured on this corpus's own tokens (fraction of variance explained 0.797 at k = 64 vs the published 0.806, L0 64.1 inside the 30–120 bounds; answer-token / context-token split 0.795 / 0.765 — the transfer-validity record) and report K = 20 shuffled-pairing floors per sparse-feature arm, per-feature label-shuffle nulls with activity and within-answer-consistency covariates, identity-plus-bias on the shared feature-id subsets (stated inapplicable for the dense-input cells: input dimension 3,584 vs answer-feature dimension 16,384), and kNN retrieval per map; this round is teacher-forced re-capture only — no on-policy generation, so no judged or language-intrusion surface. Crossed-round reads score matched targets within grain: the per-row arms score the same per-row answer states on the same 10,000 holdout rows, the averaged arms the same per-prefix mean states on the same 500 holdout prefixes. Three regime notes bind that round: the bare arm's design matrix has rank at most 20 by construction (20 bank queries tiled over rows — it is the query-main-effect read, its predictions take at most 20 distinct values, and its rank-1 retrieval among 10,000 candidates is structurally capped at 0.2%); the independently-fit averaged arm trains 4,430 prefixes against 3,584 dimensions (1.24×, a stated thin-fit regime — the induced read, which fits nothing at the averaged grain, is the round's primary averaged estimate); and generalization is over prefixes under the fixed query bank (bank queries sit on both fold sides by design — a stated scope caveat). The crossed variance decomposition's interaction-plus-residual share is unidentifiable from answer-sampling noise at one temperature-1.0 draw per cell (the prior 21k crossed decomposition generated greedily — a protocol difference); the parent per-context resampling floor rides along as a reference comparator, never a subtraction. Judge labels are independent covariates (language, topic, refusal-adjacency, answer-is-refusal, format), not the dependent variable: 9,925 of 9,941 contexts labeled; drops split 1 content / 0 transport-loss / 15 other-error (the 15 unexplained rows, ~0.15% of the pool, are too few to move any contrast). Reader-facing label renderings used below: stored `topic=nsfw` is rendered "explicit content", `chitchat_social` "social chitchat", `factual_qa` "factual Q&A", `advice_howto` "advice / how-to", `roleplay_persona` "roleplay", `harmful_or_unsafe_request` "harmful or unsafe request". Fits-summary caveat: the fit summary JSONs were rebuilt from retained fp16 predictions plus a restreamed capture after the compute instance self-deleted before harvest — holdout metrics were verified 11 of 11 against live-poll captures to 4 decimal places, while test-split diagnostics, wall clocks, and the seed-43 learning-rate record are unrecoverable, and the LMSYS-transfer cells are a fresh deterministic CPU refit. Conciseness acknowledgment: this body ships check-20 WARNs where flagged — Takeaways bullets at the 30-word tier, per-result prose over the 120-word tier, figure captions near the 60-word cap, and the total Takeaways-plus-Goal-plus-Results prose budget — accepted to keep the numbers-dense interpretation beats intact.

**Data extraction:** The corpus is 100,000 real multi-turn conversations from LMSYS-Chat-1M and WildChat-1M (tier-1 real-world user data). Keep predicate: at least 2 user turns plus structural filters (role alternation, non-empty turns, length caps), giving 626,620 eligible conversations (LMSYS 316,726 / WildChat 309,894); 100,000 were selected (realized allocation 50,545 LMSYS / 49,455 WildChat), and the near-dupe gate dropped 222 would-be train rows against the pinned val/test/holdout carve at the manifest build, leaving 99,778 manifest rows. One on-policy answer was generated per context under the decoding recipe above; 99,127 of the 99,778 were captured (651 = 0.65% over-length skips at the 7,104-token budget — an earlier 873 figure in this body conflated these skips with the 222 near-dupe drops; deeper conversations are likelier over budget — estimated per-depth-stratum skip rates ~0.1% / ~0.1% / ~1.1% for 2 / 3–4 / ≥5 user turns). The prefix is the rendered conversation history before the final user turn (ending at the prior assistant end-of-turn); the context appends the final user turn and generation header; the answer is the generated span. The corpus-provenance record (eligibility counts, allocation, near-dupe drops, depth histogram) is `sampling_manifest/meta.json` on the HF data repo (pinned in the footer). The K-resample floor subsample kept 1,988 of 2,000 contexts — 12 admission-gate skips (5 over-length capture-skipped contexts, which by construction also lack a primary draw, plus 7 with no primary draw only), recorded in the committed skip file and its HF mirror. The follow-up round's bare-arm capture is forward-only (no generation): all 99,778 manifest rows were rendered query-only, sorted by rendered length into right-padded batches, and captured in 56 chunks of up to 2,000 rows; 99,774 rows landed — 4 over-length bare renders skipped (same 7,104-token budget; all four also over-length as full contexts), recorded in the committed shard-7 skip file. The tail chunk holds the longest renders by construction and stopped out-of-memory at batch 64 on the single-H100 re-run host; it was re-captured at batch 8. At fit assembly the bare rows were reordered to the parent's captured-row order and the 647 bare rows over parent-skipped contexts were dropped, leaving the identical 99,127-row pool and split. The crossed round's corpus build: the eligible prefix pool met its 5,500 floor exactly (5,514 walked, 14 length rejects), 5,000 prefixes drawn seeded; the bank pool held 200 candidates (7 length rejects, 1 within-bank near-dupe reject), yielding the 20-query bank with no per-prefix substitutions. All 100,000 crossings were generated and captured — zero over-length skips, zero identity-gate violations, zero empty responses — and the raw completion text landed in 208 chunk files under the crossed raw-completions prefix (footer).

**Sample training/evaluation data + completions:** The corpora are unscreened real user text, so only rows whose judge labels are verifiably benign are quoted, truncated as marked; all other rows are referenced by context id (`ci`) at the pinned HF raw-completions prefix. Both examples below are cherry-picked for verifiable benignity from the 2,299 held-out rows labeled language en, topic factual Q&A or advice / how-to, not refusal-adjacent, answer not a refusal; quotes are taken from the judge-visible excerpts (caps in the table above). First example — cherry-picked benign row; full text in [raw_completions on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8fc7e9ceb824eed33474a954d6bc89c95c97cadf/issue1738_multiturn/raw_completions).

```
ci 18784 (LMSYS · labels: language=en, topic=factual_qa, refusal_adjacent=no, answer_is_refusal=no, format=prose)
history (tail): user: "are you connected to the internet?" -> assistant: "Yes, I am connected to the
  internet. As an AI language model, I run on a ..." [truncated]
final user turn: "I mean do you have general knowledge of 2023?"
model answer: "While I don't have real-time, up-to-date knowledge beyond my final training cutoff
  (which is 2022 for my version), I can provide information and insights based on the knowledge I
  was trained on. ..." [truncated; 443 chars total]
```

Second example — cherry-picked benign row from the same pool, truncated as marked; full text in [raw_completions on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8fc7e9ceb824eed33474a954d6bc89c95c97cadf/issue1738_multiturn/raw_completions).

```
ci 36811 (LMSYS · labels: language=en, topic=advice_howto, refusal_adjacent=no, answer_is_refusal=no, format=mixed)
history (tail): "...enge 2: Try to Make Friends with the Last Person You Meet Before Closing Time ..."
  [tail excerpt starts mid-word at the 800-char history cap]
final user turn: "I like the second challenge. can you give me 2 more?"
model answer: "Certainly! Here are two more creative and engaging challenges focused on nightlife in
  Japan to help you work on your social anxiety: ..." [truncated; quoted from the 1,000-char judge
  excerpt of the full completion]
```

Third example — the bare-arm render convention, shown on the synthetic self-test probe recorded verbatim in `bare_pilot_meta.json` (not a corpus row — chosen because real-row renders are identical apart from the user text and the corpora are unscreened); all real renders ride in-chunk at the [bare capture prefix on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bcda6e7855e0c99d416198be6e5747883df3ea4a/issue1738_multiturn/bare_query/capture).

```
<|im_start|>system
<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
```

## Results

### Conversation history alone reaches held-out R² 0.38, well above the 0.11 single-turn prefix reference

Held-out R² per arm, layer, and fitter (30 cells, n = 9,941) with 95% CIs from 10,000 bootstrap draws; the shaded band marks the 0.05–0.11 single-turn prefix reference.

![Prefix-arm vs context-arm held-out R-squared by layer and fitter with the single-turn reference band](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9fe2d7ecb4c8264e01b04f5c876354c1b537730f/figures/issue_1738/hero_arm_r2_by_layer.png)

> **Figure.** *The prefix arm clears the single-turn reference band at every layer and fitter.* Held-out R², n=9,941; five fitters per arm at layers 14/19/26; error bars are 95% bootstrap CIs (10,000 draws). Shaded band: the prior single-turn prefix range 0.05–0.11 (different corpus and fold scheme; a reference, not a matched comparison).

Prefix layer-19 ridge reads 0.379 (95% CI 0.3737–0.3847); the margin over the 0.11 reference is 0.269 (95% CI 0.2637–0.2747). All 15 prefix cells sit ≥0.30 (min 0.303); an LMSYS-only fit scores 0.369 on WildChat rows (context: 0.647).

The context arm reads 0.681 (ridge) to 0.722 (kernel); the prefix recovers ~56% of it. The target answer was generated under the full context, so prefix R² is a lower bound: the gap conflates missing query information with map error.

### The prefix–context gap is corpus-wide: higher prefix error on 94.6% of individual contexts

Per-context normalized error (layer-19 ridge; squared error over squared distance to the holdout mean) for all 9,941 held-out contexts, prefix (y) vs context (x), log–log, with the y = x diagonal.

![Per-context normalized error scatter prefix arm vs context arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9fe2d7ecb4c8264e01b04f5c876354c1b537730f/figures/issue_1738/percontext_nerr_scatter.png)

> **Figure.** *The prefix arm errs more on almost every individual conversation.* Each point is one held-out context (n=9,941), layer-19 ridge; axes are normalized prediction error, log–log; the dashed line is y = x. Points above the line are contexts where the history-only map is worse than the full-context map.

The prefix arm errs more on 94.6% of individual contexts (median normalized error 0.552 vs 0.297): the aggregate gap is a near-uniform per-context shift, not a failing subpopulation, and the cloud is one mass with no prefix-blind cluster. This scatter is the per-unit view behind the aggregate contrasts in this body; the per-context table is committed alongside the figures (footer).

### Prefix transport improves with conversation depth; the full-context map stays flat

Held-out R² (layer-19 ridge) per arm by conversation depth: 2 user turns (n=4,154), 3–4 (n=3,298), ≥5 (n=2,489).

![Held-out R-squared by conversation depth for prefix and context arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9fe2d7ecb4c8264e01b04f5c876354c1b537730f/figures/issue_1738/depth_stratified_r2_v2.png)

> **Figure.** *Deeper history helps the prefix arm only.* Held-out R² (layer-19 ridge) by depth stratum, per arm; stratum sizes 4,154 / 3,298 / 2,489. Stratum-level CIs are not available (per-stratum pooled-R² denominators were not persisted); the per-context depth contrasts carry the significance reads.

Prefix R² rises 0.358 → 0.385 → 0.390 across strata while context stays flat (0.680 / 0.681 / 0.672); per-context contrasts agree (prefix error higher at 2 turns, +0.027, and lower at ≥5 turns, −0.023, both significant after multiplicity correction; context depth contrasts non-significant). Deeper history carries a larger share of the upcoming answer state. The read is observational (depth co-varies with topic and length), and the ≥5-turn stratum is length-truncated (~1.1% capture-skip vs ~0.1% at 2 turns), so the rising trend rests on its surviving, shorter tail.

### Prefix error concentrates where the final query pivots away from the history

Mean per-context error difference (group minus rest; layer-19 ridge; n=9,941) for 22 category contrasts per arm, with 95% bootstrap CIs; filled markers are significant after multiplicity correction (false-discovery rate q=0.05, 10,000-draw permutation p).

![Taxonomy contrast forest plot per arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9fe2d7ecb4c8264e01b04f5c876354c1b537730f/figures/issue_1738/taxonomy_forest_L19_ridge.png)

> **Figure.** *The prefix arm has strong category structure; the context arm mostly does not.* Error differences (group minus rest) across language, topic, refusal-adjacency, answer-is-refusal, format, depth, and corpus contrasts; group sizes on the labels. Negative means the group is easier than the rest for that arm.

Prefix error structure is far stronger than context structure: prefix differences reach 0.13 in magnitude while context contrasts are mostly ≤0.03 (the widest, other-topics at +0.085, rests on 82 contexts). Prefix error is highest where the final query pivots away from the history — English +0.079, social chitchat +0.084, translation +0.071, factual Q&A +0.046 — and lowest on continuation-heavy genres: WildChat −0.131, explicit content −0.103, creative writing −0.069, roleplay −0.046, coding −0.040.

Categories co-vary, so no single-factor attribution is licensed. A floor-adjusted rerun (per-context sampling floor subtracted; n=1,988; 19 of 22 contrasts clear the group-size floor) reproduces every shared significant sign — English +0.086, WildChat −0.116, refusal-adjacent −0.097 — so the structure is not an answer-sampling artifact.

### Both maps are fitted and discriminative; a constant-shift baseline fails in both arms

Left half of the figure: held-out R² at layer 19 for the fitted ridge map vs the identity-plus-learned-bias baseline, per arm. Right half: retrieval accuracy at rank 1 (cosine) among the 9,941 held-out targets for the same predictors, log scale, chance 1.0e-4.

![Mapping baselines held-out R-squared and retrieval accuracy per arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9fe2d7ecb4c8264e01b04f5c876354c1b537730f/figures/issue_1738/mapping_baselines_dissociation.png)

> **Figure.** *Fitted maps dominate the identity-plus-bias baseline on both reads.* Held-out R² and rank-1 retrieval (cosine) per arm at layer 19, n=9,941; chance retrieval 0.01%. The baseline's high context-arm retrieval despite negative R² shows the two reads dissociate.

Identity-plus-bias R² is strongly negative in both arms (−0.92 prefix, −1.08 context at layer 19; −1.06 to −3.03 at the flanking layers): pre-answer and answer states do not sit a constant shift apart, so the fitted maps do real work. The reads dissociate: identity-plus-bias retrieves the correct target at rank 1 for 51.2% of context-arm items despite its negative R² — discriminative but badly mis-scaled — while collapsing to 1.3% in the prefix arm. Fitted ridge dominates both: 82.8% (context) and 20.7% (prefix) at rank 1 — the prefix map identifies the specific conversation's answer state one time in five among ~10k candidates.

### The prefix map is near linear-complete; the nonlinear advantage lives in the context arm's low-variance directions

Per-direction held-out R² (layer 19) over the top-256 answer-PCA directions, per arm: shared-penalty ridge, MLP (width 8,192), and a per-direction-penalty ridge control (38-value grid); log-rank axis.

![Per-direction held-out R-squared for ridge MLP and per-direction penalty control per arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9fe2d7ecb4c8264e01b04f5c876354c1b537730f/figures/issue_1738/perdirection_r2_L19.png)

> **Figure.** *The MLP-over-ridge gap grows toward low-variance directions in the context arm but stays small in the prefix arm.* Per-direction held-out R² at layer 19, top-256 answer-PCA directions, n=9,941; the per-direction-penalty control tests whether the gap is differential shrinkage.

The context arm shows a genuine nonlinear advantage that grows toward low-variance directions: the MLP-over-ridge gap rises from +0.023 (directions 1–16) to 0.067–0.068 (directions 65–256), and the per-direction-penalty control matches shared ridge (band means within 0.003), ruling out differential shrinkage. Prefix-arm gaps stay small (+0.011 / +0.027 / +0.026 / +0.008): relative to what an MLP can extract, the prefix-to-answer map is near linear-complete, and both fitters approach the sampling floor in the tail (ridge 0.071, MLP 0.079 at ranks 129–256). The bare arm sits between (gaps 0.012–0.038; penalty control within 0.003).

The K-resample floor puts answer-sampling noise at a median 6.1% of per-context error at layer 19 (7.2% / 5.5% at layers 14 / 26), capping idealized R² at ≈0.70 (context) and ≈0.42 (prefix): the prefix arm's unexplained variance is missing information, not noise.

### The final user query alone maps to the answer state better than the whole conversation history

Held-out R² for all three input arms — bare query (the final user turn rendered with an empty system turn, no history), prefix, and context — per layer and fitter (45 cells, n = 9,941), with 95% CIs from 10,000 bootstrap draws.

![Three-arm held-out R-squared by layer and fitter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/017f9d8dbeb6eb640ea641cbda4a81a750125055/figures/issue_1738/hero_arm_r2_by_layer_3arm.png)

> **Figure.** *The bare-query arm sits between prefix and context at every layer and fitter.* Held-out R², n=9,941; error bars are 95% bootstrap CIs (10,000 draws). The bare arm's input is the final user turn alone — no conversation history, no system prompt.

Bare layer-19 ridge reads 0.534 (95% CI 0.528–0.539) — above prefix 0.379, below context 0.681: the final query alone recovers ~78% of the full-context map vs ~56% for the whole history, so the two information shares overlap rather than add. A duplicate-query control rules out memorization: 699 of 9,941 held-out contexts (7.0%) share their exact final-query text with a training row, and excluding them moves bare to 0.536 (prefix 0.382, context 0.676). An LMSYS-only bare fit scores 0.494 on WildChat holdout; identity-plus-bias fails in this arm too (−1.36 at layer 19) while bare ridge retrieves the exact answer state at rank 1 for 47.0% of candidates (cosine).

### Per context, the query-only map beats the history-only map on two thirds of conversations

Per-context normalized error (layer-19 ridge), bare arm (y) vs prefix arm (x), log–log, all 9,941 held-out contexts, with the y = x diagonal; the bare-vs-context companion scatter is committed alongside (footer).

![Per-context normalized error scatter bare arm vs prefix arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/017f9d8dbeb6eb640ea641cbda4a81a750125055/figures/issue_1738/percontext_nerr_scatter_bare_vs_prefix.png)

> **Figure.** *Most conversations sit below the diagonal: the final query alone predicts their answer state better than their whole history does.* Each point is one held-out context (n=9,941), layer-19 ridge, normalized prediction error, log–log; the dashed line is y = x.

The bare arm errs more than prefix on only 30.8% of contexts (median normalized error 0.414 vs 0.552) and more than context on 88.0% (companion scatter): the per-conversation ordering matches the aggregate one, but history genuinely wins on almost a third of conversations — the two partial inputs are complementary, not ranked.

### Query-only error is highest where the conversation thread lives in the history — the mirror image of the history-only arm

Mean per-context error difference (group minus rest; layer-19 ridge; n=9,941) for the 22 category contrasts, bare arm, with 95% bootstrap CIs; filled markers are significant after the same false-discovery-rate correction (q=0.05, 10,000-draw permutation p).

![Taxonomy contrast forest plot for the bare-query arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/017f9d8dbeb6eb640ea641cbda4a81a750125055/figures/issue_1738/taxonomy_forest_bare_L19_ridge.png)

> **Figure.** *The bare arm's hardest categories are largely the prefix arm's easiest.* Error differences (group minus rest) across language, topic, refusal-adjacency, answer-is-refusal, format, depth, and corpus contrasts; group sizes on the labels; negative means easier than the rest.

The bare arm's category structure is close to the prefix arm's mirror image: roleplay (+0.104) and 5-plus-turn conversations (+0.036) are hardest for query-only yet easy for history-only, while social chitchat flips the other way (−0.086 bare vs +0.084 prefix); translation is hard for both (+0.088 / +0.071). Depth reverses too: bare R² falls 0.545 → 0.539 → 0.496 across the strata where prefix rises. Floor-adjusted reruns (n=1,988; 19 of 22 contrasts clear the group-size floor) reproduce 11 of the 15 significant raw signs, but the refusal-related structure does not survive — answer-is-refusal flips sign (+0.088 raw → −0.138 adjusted, 30 contexts) — so the bare arm's high raw error on refusal answers is answer-sampling variance, not map failure.

### The depth convergence never completes: the query-only map stays ahead of the history-only map at every exact depth

Held-out R² (layer-19 ridge, subset-mean denominators) per arm at exact conversation depths 2–8 and ≥9 user turns (n = 4,154 down to 220; 676 at ≥9), left half of the figure; right half: per-stratum fractions of contexts where the bare arm errs more than prefix and than context, with 95% CIs from a 2,000-draw within-stratum bootstrap.

![Held-out R-squared and bare-vs-prefix dominance fractions by exact conversation depth](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7d3879c1a61083aae924cd751445a32b9a0333e3/figures/issue_1738/depth_rebin_bare_vs_prefix.png)

> **Figure.** *The bare-over-prefix advantage shrinks with depth but never inverts.* Left: held-out R² per arm at exact user-turn counts, layer-19 ridge. Right: fraction of contexts per stratum where bare errs more than prefix or than context; 95% CIs from 2,000 within-stratum bootstrap draws; the dashed line marks 0.5.

No crossover at any realized depth: the fraction where query-only errs more than history-only rises from 0.278 (95% CI 0.264–0.292) at 2 turns to 0.377 (95% CI 0.309–0.441) at 8, every stratum's upper CI below 0.5. The coarse-band convergence is real but bounded — the R² gap narrows 0.187 → 0.087 (0.102 at ≥9), bare falling 0.545 → 0.471 while prefix holds 0.36–0.40 and context stays flat. The deepest strata are small (n = 220 at 8 turns), hence the wide intervals.

### The history-vs-context ordering reproduces when the answer is read as interpretable sparse-autoencoder features

Held-out pooled R² (n = 9,941) onto the same 16,384 mean-pooled sparse-autoencoder answer features at layer 19, for sparse-feature inputs (history-end and full-context-end token codes) and for the same states as dense inputs; 95% bootstrap CIs (10,000 draws); shuffled-pairing floors (K = 20 per sparse arm) at the zero line.

![Held-out R-squared onto SAE answer features for sparse-feature and dense inputs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/67f3ea69d14eda6e928f2e36a74e746dc04e09cd/figures/issue_1738/sae_hero_r2_by_input.png)

> **Figure.** *The history-vs-context gap survives the re-basis into interpretable features, and dense inputs beat the sparse codes of the same states.* Held-out pooled R² onto mean-pooled sparse-autoencoder answer features, layer 19, n=9,941; shuffled-pairing floors sit at −0.003 to −0.007.

History features recover 0.359 vs 0.638 for full-context features — the same ~56% share the dense-state arms show — over floors at ≈0, so both maps are real, not pairing artifacts. Re-basing the input costs both arms information: the same states read densely give 0.451 and 0.700.

The single-token input codes are sparse — 216 (history) / 530 (context) features clear the 1%-activity floor vs 3,584 dense dimensions — so the answer state is far more recoverable from the raw state than from its interpretable code. Mean pooling is the right target: max and fraction-active pooling drop to 0.171 / 0.317 and 0.241 / 0.477. An LMSYS-only feature fit scores 0.353 / 0.611 on WildChat rows.

### Nearly every answer feature is better predicted from the full context than from history alone

Per-feature held-out R² over the 16,384 restricted answer features (layer 19, mean pooling), full-context arm (x) vs history arm (y), clipped at −1; the figure header counts carried features — per-feature R² above that feature's own label-shuffle null 95th percentile (K = 20 draws).

![Per-feature held-out R-squared full-context arm vs history arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/67f3ea69d14eda6e928f2e36a74e746dc04e09cd/figures/issue_1738/sae_perfeature_scatter.png)

> **Figure.** *The cloud sits below the diagonal: the context advantage is feature-wide, not a feature subpopulation.* Each point is one answer feature (n=16,384); axes are per-feature held-out R² over 9,941 contexts; the dotted line is y = x.

The full-context map beats the history map on 99.7% of features (median per-feature R² 0.113 vs 0.038; median gap 0.069). Both arms carry nearly every feature above its null — 16,340 / 16,371 of 16,384 vs ≈819 expected false positives — but the 20-draw threshold is loosely calibrated (a held-out null draw exceeds it 9.2% of the time vs the nominal 5%), so the counts are exploratory; the cross-arm contrast is the load-bearing read (split-half rank stability 0.92–0.93).

The context advantage is larger for more-active and more within-answer-consistent features (Spearman 0.32 and 0.48; committed companion figure, footer). Feature-space baselines match the dense story: identity-plus-bias on shared feature ids reads −2.99 / −2.38 vs 0.43 / 0.72 fitted, and rank-1 retrieval is 1.3% / 23.8% (chance 0.01%).

### With the final query held to a shared bank, the per-row history map collapses to the history's own variance share

Held-out R² (ridge, the crossed round's sole fitter) for the four input arms on the crossed corpus — bare query, history, full context, question-averaged — per layer, with 95% CIs from 10,000 bootstrap draws; per-row arms score 10,000 held-out rows, the question-averaged bar is the independently-fit per-prefix-mean map over 500 held-out prefixes.

![Four-arm held-out R-squared per layer on the crossed corpus](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ff10d49b354406df737e539a23fa76cd92ae6f87/figures/issue_1738/crossed_arm_r2_grid.png)

> **Figure.** *With query identity held to a shared 20-question bank, the history arm collapses and the bare arm becomes the query-main-effect read.* Held-out R², ridge; 95% bootstrap CIs. The bare arm's design matrix has rank at most 20 by construction (20 bank queries tiled over rows), so its bar is the query main effect, not a general query map.

The history map reads 0.046 at layer 19 (95% CI 0.043–0.047) versus 0.379 on the main corpus — not a contradiction: there each history has its own final query, so history-end states carry query identity; the shared bank removes exactly that variance. Bare reads 0.627 (95% CI 0.619–0.634) and full context 0.795 (95% CI 0.791–0.799).

Identity-plus-bias R² stays negative in every cell; rank-1 retrieval among 10,000 candidates (cosine) reads 26.6% for context vs 0.05% for history — on this grid the per-row history map is barely discriminative. An LMSYS-only fit scores 0.783 (context) and 0.732 (question-averaged) on WildChat holdout rows: the reads transfer across sub-corpora.

### On the crossed grid the final query explains nine times more answer-state variance than the whole history

Variance shares of the mean-answer state over the complete 5,000 × 20 grid — history, query, and interaction-plus-residual — overall per layer (left half of the figure; sum-of-squares-weighted over dimensions) and per answer-PCA direction at layer 19 (right half; 48 train-fit directions).

![Crossed variance shares overall and per answer-PCA direction](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ff10d49b354406df737e539a23fa76cd92ae6f87/figures/issue_1738/crossed_anova_shares.png)

> **Figure.** *Query identity dominates the totals; the shares invert down the PCA spectrum.* The interaction-plus-residual share contains the answer-sampling noise of the single temperature-1.0 draw per cell — the parent resampling floor (median 6.1% of per-context error at layer 19) is a reference comparator, not a subtraction.

At layer 19 the shares are query 0.630 / history 0.073 / interaction-plus-residual 0.297 (query 0.62–0.69, history 0.05–0.07 across layers). The history share is the per-row ceiling for any history-only map, so the fitted 0.046 sits near it; the residual bundle is an upper bound on genuine history-query interaction, not an estimate of it. The prior 21k crossed decomposition read the inverse ordering (prefix-dominated, 79 / 11 / 10); its prefixes were constructed persona blocks rather than real histories — a reference only.

Per direction the picture inverts: the top ~20 directions are query-dominated (share up to 0.89), ranks 21–48 run interaction-heavy (medians: interaction 0.758, history 0.192); the same split holds per sparse-autoencoder answer feature (median query 0.22 vs history 0.05).

### Averaging the query away recovers the history signal through the same operator that serves the per-row map

Held-out R² at the question-averaged grain (per-prefix means, 500 held-out prefixes): the primary induced read — the per-row context map applied to averaged inputs — beside the independently-fit averaged map (left half); the disjoint stitch on concatenated history-end plus bare-query states vs its parts and the full context, 10,000 rows (right half); 95% bootstrap CIs.

![Induced vs independent question-averaged maps and the disjoint stitch](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ff10d49b354406df737e539a23fa76cd92ae6f87/figures/issue_1738/crossed_induced_vs_independent.png)

> **Figure.** *History determines about three quarters of the query-averaged answer state, and the per-row map already carries that structure.* Left: hatched = the per-row context map applied to per-prefix averages. Right: the stitch sits above bare query alone and below the full context at every layer.

The induced read gives 0.745 at layer 19 (95% CI 0.726–0.762) vs 0.757 (0.739–0.773) independent; gaps −0.036/−0.012/−0.006 at layers 14/19/26. The per-row map recovers the averaged map to within a few hundredths — the quantitative form of the prior operator-coincidence read (different corpus and split; the two reads share holdout prefixes, so CI overlap is not a paired criterion).

The stitch reads 0.673 (95% CI 0.666–0.679) vs 0.795 full context: the parts recover 85%. The three non-bare operator pairs share output subspaces at 22–30° mean angle (k=48) vs an 84.2–84.5° random band; bare-involving pairs (rank-limited) read 47–55° vs their own k=19 band (86.3–86.7°); input subspaces sit near their bands, except context–averaged at 68°.

---

**Repro:** ~25–27 GPU-h realized vs 77 budgeted (GCP A100-80 ephemeral instances for capture waves, fits, and the K-resample floor; VM CPU for the manifest build and characterization; Anthropic Batch API for judge labels) · Code: branch issue-1738 at [9fe2d7ecb4](https://github.com/superkaiba/explore-persona-space/tree/9fe2d7ecb4c8264e01b04f5c876354c1b537730f) (drivers `scripts/issue1738_multiturn_generate_capture.py`, `scripts/issue1738_multiturn_fits.py`, `scripts/issue1738_characterize.py`; figures `scripts/issue1738_analyzer_figures.py`; floor-adjusted family `scripts/issue1738_floor_adjusted_taxonomy.py`), harvest commit [a36e97847b](https://github.com/superkaiba/explore-persona-space/commit/a36e97847b), PR [#1501](https://github.com/superkaiba/explore-persona-space/pull/1501) · Eval JSONs: [eval_results/issue_1738 at the pin](https://github.com/superkaiba/explore-persona-space/tree/9fe2d7ecb4c8264e01b04f5c876354c1b537730f/eval_results/issue_1738) — `fits/multiturn_100k_fits.json` (30 cells, seed-43 repeat, LMSYS-transfer control), `h1_contrast.json`, `mapping_baselines.json`, `taxonomy.json`, `taxonomy_floor_adjusted.json`, `depth_contrasts.json`, `kresample/` (floor summary, gates, subsample, skip breakdown, per-layer floors), `perdirection/pdshrink_summary.json`, `judge_labels/labels.json`, and the per-context table `percontext_summary_L19_ridge.csv` (the per-cell file behind every aggregate here) · Raw data + tensors: [issue1738_multiturn on the HF data repo](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8fc7e9ceb824eed33474a954d6bc89c95c97cadf/issue1738_multiturn) — `sampling_manifest/meta.json` (the corpus-provenance record) plus manifest shards and `split_1738.json`, `capture/` (224 chunks), `raw_completions/` (255 files, all stages), `analysis_tensors/` (percontext, pred16, y_holdout, weights, summaries), `kresample/` (incl. `kresample_shard00_skipped.json`), `judge_labels/` · Figures: [figures/issue_1738 at the pin](https://github.com/superkaiba/explore-persona-space/tree/9fe2d7ecb4c8264e01b04f5c876354c1b537730f/figures/issue_1738) (PNG + PDF + meta.json sidecars); the earlier driver-rendered hero and depth figures (no per-figure sidecars) are superseded by the versions embedded above · Deviations: fits summaries rebuilt from retained fp16 predictions after the compute instance self-deleted pre-harvest (holdout metrics verified 11 of 11 vs poll-captured values; test-split diagnostics and wall clocks unrecoverable; LMSYS-transfer cells are a fresh deterministic CPU refit); taxonomy/depth stage re-run on the VM after a staging-path skip · Reused artifacts: fitter set + decoding recipe + capture convention from [#779](https://eps.superkaiba.com/tasks/779) (`scripts/issue779_ffc_n1m_fits.py` constants, `scripts/issue779_collect.py` capture convention) — fit: same model and measurement regime, corpus construction is the single changed variable; error-characterization battery code from [#1482](https://eps.superkaiba.com/tasks/1482) (`scripts/issue1482_*` stages, ported per-arm) — fit: same estimator regime at a 10k holdout · Bare-query follow-up round: ≈1–2 GPU-h realized vs 3 budgeted (GCP 8×A100 sharded batched capture, ~15 min; 1×A100 fits; a RunPod 1×H100 tail-chunk re-capture at batch 8); round code + figures at [017f9d8dbe](https://github.com/superkaiba/explore-persona-space/commit/017f9d8dbeb6eb640ea641cbda4a81a750125055) on branch issue-1738 (plan v6 amendment); round eval JSONs at [eval_results/issue_1738/bare_query](https://github.com/superkaiba/explore-persona-space/tree/26996fcce3dd8ae6833de7117099aeab1aae1052/eval_results/issue_1738/bare_query) — `fits/multiturn_100k_fits.json` (15 bare cells + `cells/`), `mapping_baselines.json`, `taxonomy.json` (floor-adjusted block embedded), `depth_contrasts.json`, `perdirection/pdshrink_summary.json`, the three-arm per-context table `percontext_summary_L19_ridge.csv`, and the duplicate-query control `final_query_dedup_check.json` (committed at [26996fcce3](https://github.com/superkaiba/explore-persona-space/commit/26996fcce3dd8ae6833de7117099aeab1aae1052)); the exact-depth re-bin `depth_rebin.json` + `figures/issue_1738/depth_rebin_bare_vs_prefix.png` (script `scripts/issue1738_depth_rebin.py`, 2,000-draw within-stratum bootstrap) committed at [7d3879c1a6](https://github.com/superkaiba/explore-persona-space/commit/7d3879c1a61083aae924cd751445a32b9a0333e3); bare tensors + renders at [issue1738_multiturn/bare_query on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bcda6e7855e0c99d416198be6e5747883df3ea4a/issue1738_multiturn/bare_query) (`capture/` 56 chunks + `bare_pilot_meta.json`, `analysis_tensors/` summaries + percontext + pred16 + weights + y_holdout, `raw_completions/shard07_skipped.json` — no new generations, forward-only capture); the bare-vs-context companion scatter is `figures/issue_1738/percontext_nerr_scatter_bare_vs_context.png` at the round pin · SAE-arm follow-up round: ≈8 GPU-h realized vs 18 booked (S1 teacher-forced sparse-feature capture ~55 min on a GCP 8×A100 ≈ 7.3 GPU-h; S2 fits ≈ 0.6 GPU-h on 1×A100; figures on the VM CPU); round code, eval JSONs, and figures at [67f3ea69d1](https://github.com/superkaiba/explore-persona-space/commit/67f3ea69d14eda6e928f2e36a74e746dc04e09cd) on branch issue-1738 (script `scripts/issue1738_sae_arm.py`, plan v8) — [eval_results/issue_1738/sae_arm](https://github.com/superkaiba/explore-persona-space/tree/67f3ea69d14eda6e928f2e36a74e746dc04e09cd/eval_results/issue_1738/sae_arm): `sae_fits.json` (8 cells with 10,000-draw bootstrap CIs, K=20 shuffle floors, the per-feature summary block, and an LMSYS-transfer control), `mapping_baselines.json`, `fits_cells/`, `perfeature_summary.csv` (the per-feature file behind the scatter), `cells_table.csv`, `sae_pilot_meta.json` (the fitness-gate record); the per-feature gap-vs-covariates companion figure is `figures/issue_1738/sae_delta_covariates.png` at the same pin; SAE capture + tensors at [issue1738_multiturn/sae_arm on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/f1a5c55ab0a8c37e943d1fabda18c18cba43cb1d/issue1738_multiturn/sae_arm) (`capture/` 224 chunks, `analysis_tensors/` pred16 + perfeature + summaries — teacher-forced re-capture, no new generations) · Crossed follow-up round (`crossed-multiturn-averaged`): ≈14.5 GPU-h realized vs 18 projected at the pilot gate (S1 generation + capture ≈13 GPU-h on a RunPod 8×A100 sharded fleet; S2 reads ≈1.5 GPU-h on a GCE A100; figures on the VM CPU); round code at [97e67d757c](https://github.com/superkaiba/explore-persona-space/commit/97e67d757ccd12005d68e5a77a91340a388c4480), [ff83be84e8](https://github.com/superkaiba/explore-persona-space/commit/ff83be84e82ab55781bdf187459271bc9d90a6b7), and [9a1c79cf18](https://github.com/superkaiba/explore-persona-space/commit/9a1c79cf1825ae9f1223bc99bab81704602e095c) on branch issue-1738 (plan v9; reads driver `scripts/issue1738_crossed_reads.py`; crossed manifest + capture modes in `scripts/issue1738_multiturn_generate_capture.py`); round eval JSONs at [eval_results/issue_1738/crossed](https://github.com/superkaiba/explore-persona-space/tree/41a1e7399fd4e51338e3f639a5f1b71ccc04f6c6/eval_results/issue_1738/crossed) (committed at [41a1e7399f](https://github.com/superkaiba/explore-persona-space/commit/41a1e7399fd4e51338e3f639a5f1b71ccc04f6c6)) — `cells/` (12 ridge cells with bootstrap CIs), `crossed_fits.json` (induced + corpus-transfer + stitch pointers), `anova.json`, `stitch.json`, `operator_geometry.json`, `mapping_baselines.json`, `sae_perfeature.json` + `perfeature_crossed_summary.csv` (per-feature evidence persisted for later labelling; judged labelling frozen, 0 judge calls), `crossed_pilot_meta.json` (the throughput + crossing-sanity gate record), and the per-cell table `cells_table.csv` (committed with the round figures at [ff10d49b35](https://github.com/superkaiba/explore-persona-space/commit/ff10d49b354406df737e539a23fa76cd92ae6f87)); round figures `crossed_arm_r2_grid`, `crossed_anova_shares`, `crossed_induced_vs_independent` (PNG + PDF + meta.json) at the same pin; crossed tensors + text at [issue1738_crossed on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a45c630487b368a8327d678ffd8c4ad717fd6236/issue1738_crossed) (`sampling_manifest/` incl. `bank.json`, `split_1738_crossed.json`, and `meta.json` — the supply-gate record, `capture/`, `raw_completions/` 208 chunk files, `analysis_tensors/`, `bare_bank/`).

**Context:** Origin (user chat, verbatim):

> run for 100k as much in parallel and vectorized as possible. [context: user approved option R3(b) from chat — 'new capture: build the multi-turn analogue of the n1M corpus — N multi-turn LMSYS/WildChat conversations (prefix = history, query = last turn), capture prefix-end + context-end + answer states, fit both arms, run the full pipeline' — after a sizing discussion grounding N=100k at ~50–70 GPU-h capture on the measured 3,250 ctx/GPU-h parent basis]

Follow-up round `bare-query` (same-issue; user-chat 2026-07-28, verbatim: "yes" — approving "Want me to add the bare-query arm to #1738?"; scope re-posted 2026-07-29): adds the query-only input arm, completing the prefix / bare-query / context set. Run 2026-07-29.

Follow-up round `sae-arm` (same-issue; user-chat 2026-07-28, verbatim: "yes run it now" — approving "Want me to queue it? I'd post the follow-up scope on #1738 now (\"SAE arm, both input arms, +15-20 GPU-h, runs post-capture\") so the running session picks it up as soon as the store is verified"): re-bases the maps into sparse-autoencoder feature space. Run 2026-07-30.

Follow-up round `crossed-multiturn-averaged` (same-issue; user-chat 2026-07-28, verbatim: "ok add this: 5,000 prefixes x 20 queries | 100k | ~15.6 GPU-h" — approving the recommended crossed-corpus sizing): builds the crossed companion corpus and the question-averaged arm, completing the four-input-object decomposition. Run 2026-07-30.

Lineage: [#1482](https://eps.superkaiba.com/tasks/1482) — parent (this run applies its error-characterization battery per input arm at multi-turn scale). Created 2026-07-28; run 2026-07-28 → 2026-07-30.






