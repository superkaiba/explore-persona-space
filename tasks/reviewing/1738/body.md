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

**Methodology:** [docs/methodology/issue_1738.md](https://github.com/superkaiba/explore-persona-space/blob/9a1f82d92355804082f4915665394449e4744024/docs/methodology/issue_1738.md) · [gist mirror](https://gist.github.com/superkaiba/974da25791408751959e3910e386d371)

## Takeaways

- Conversation history alone predicts the upcoming answer state at scale: prefix-arm held-out R² 0.379 (layer 19, ridge, n=9,941) clears the 0.11 single-turn reference by 0.269 (95% CI 0.264–0.275).
- The full-context map reads 0.681 (the prefix recovers ~56%); all 15 prefix layer-by-fitter cells sit ≥0.30 (min 0.303); a fit on LMSYS rows alone scores 0.369 on WildChat holdout.
- The gap to the full context is corpus-wide (prefix error higher on 94.6% of held-out contexts) and narrows with depth: prefix R² rises 0.358→0.390 while context stays flat.
- Prefix error is highest where the final query pivots away from the history (English, social chitchat, translation, factual Q&A) and lowest on continuation-heavy genres (creative writing, roleplay); signs reproduce floor-adjusted.
- Both maps are fitted, not constant shifts: identity-plus-bias R² is negative (−0.92/−1.08), while ridge retrieves the exact answer state at rank 1 for 20.7%/82.8% of 9,941 candidates.
- Answer-sampling noise is small (median 6.1% of per-context error): the prefix arm's remaining variance is missing information, not noise; its MLP−ridge gap is ≤0.027 vs up to 0.068 for context.

## Goal

- **This experiment in context:** This experiment tests whether conversation history alone — the prefix, everything before the final user turn — already carries the upcoming answer's representation on real multi-turn data at scale. The single-turn 963k-context corpus line ([#779](https://eps.superkaiba.com/tasks/779)) established the context-to-answer map and the five-fitter comparison reused here, but its prefix is a constant chat-template string, so a prefix arm is structurally empty there. The crossed 21k-context decomposition ([#1092](https://eps.superkaiba.com/tasks/1092)) measured prefix-end R² 0.05–0.11 — under crossed persona-and-query pairings, a grouped 6-fold scheme, layer 14, and ~21k rows, vs this run's natural conversations, pinned single split, and 9,941-context holdout — not directly comparable, so 0.11 enters only as a fixed reference constant. The per-context error taxonomy, K-resample answer-sampling floor, and per-direction reads come from the error-characterization battery of [#1482](https://eps.superkaiba.com/tasks/1482), applied here to both mapping arms.
- **Broader narrative:** The context-as-vector question: is the pre-answer residual state a sufficient statistic for the upcoming answer state, and how much of that statistic exists before the final query arrives? Measuring what history alone determines — and where it fails — bounds what the final query adds and what a history-only representation monitor could read ahead of time.

## Methodology

**Design:** One corpus build, no model training. I fit the same map in both input arms — prefix-based (input: the prefix-end residual state, where the prefix is everything before the final user turn) and context-based (input: the context-end state, prefix plus final user turn) — against identical mean-answer targets, at layers 14, 19, and 26 with five fitters per arm and layer (30 cells) on one pinned split, plus a seed-43 repeat at layer 19 and an LMSYS-only refit scored on WildChat rows as a corpus-transfer control. Relative to the single-turn parent corpus, the one manipulated variable is corpus construction (natural multi-turn conversations); every capture and fit constant is inherited.

**Training:** **N/A — no model training.** The complete generation / capture / fitting / judging hyperparameter table:

| Hyperparameter | Value | Source |
|---|---|---|
| Base model | Qwen-2.5-7B-Instruct | plan §11; run record |
| Answer decoding | temperature 1.0, top_p 0.95, max_new_tokens 1,024, engine seed 42 | plan §11 (parent decoding recipe) |
| Generation length budget | 7,104 tokens (prompt + answer); over-budget contexts skipped and recorded | capture driver `issue1738_multiturn_generate_capture.py` at the code pin; run record (873 skips) |
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

**Evaluation:** The dependent variable is pooled held-out R² between the predicted and the realized mean-answer state over the 9,941 pinned held-out contexts — a decodability read on the model's own on-policy answers (no behavior-expression judging, so the dual-DV rule does not apply; no cell saturates, values span 0.30–0.72). The per-context companion is the normalized error `nerr(x) = ||v_hat(x) − v(x)||² / ||v(x) − mu_eval||²`. One target asymmetry binds every prefix-arm read: the answer whose mean state is the target was generated under the full context, so the prefix arm predicts a representation produced with information (the final user turn) its input never contains — prefix R² is a lower bound on history-only transport, and the prefix-to-context difference mixes genuinely missing query information with map error. Standing mapping reads run per arm: the identity-plus-learned-bias baseline (input and output share dimension 3,584, so it applies) and retrieval among the held-out targets (euclidean and cosine, k in 1/5/10, chance 1/9,941 ≈ 0.01%). Judge labels are independent covariates (language, topic, refusal-adjacency, answer-is-refusal, format), not the dependent variable: 9,925 of 9,941 contexts labeled; drops split 1 content / 0 transport-loss / 15 other-error (the 15 unexplained rows, ~0.15% of the pool, are too few to move any contrast). Reader-facing label renderings used below: stored `topic=nsfw` is rendered "explicit content", `chitchat_social` "social chitchat", `factual_qa` "factual Q&A", `advice_howto` "advice / how-to", `roleplay_persona` "roleplay", `harmful_or_unsafe_request` "harmful or unsafe request". Fits-summary caveat: the fit summary JSONs were rebuilt from retained fp16 predictions plus a restreamed capture after the compute instance self-deleted before harvest — holdout metrics were verified 11 of 11 against live-poll captures to 4 decimal places, while test-split diagnostics, wall clocks, and the seed-43 learning-rate record are unrecoverable, and the LMSYS-transfer cells are a fresh deterministic CPU refit. Conciseness acknowledgment: this body ships check-20 WARNs where flagged — Takeaways bullets at the 30-word tier, per-result prose over the 120-word tier, figure captions near the 60-word cap, and the total Takeaways-plus-Goal-plus-Results prose budget — accepted to keep the numbers-dense interpretation beats intact.

**Data extraction:** The corpus is 100,000 real multi-turn conversations from LMSYS-Chat-1M and WildChat-1M (tier-1 real-world user data). Keep predicate: at least 2 user turns plus structural filters (role alternation, non-empty turns, length caps), giving 626,620 eligible conversations (LMSYS 316,726 / WildChat 309,894); 100,000 were selected (realized allocation 50,545 LMSYS / 49,455 WildChat), and the near-dupe gate dropped 222 train rows against the pinned val/test/holdout carve. One on-policy answer was generated per context under the decoding recipe above; 99,127 of 100,000 contexts were captured (873 = 0.87% over-length skips at the 7,104-token budget; deeper conversations are likelier over budget — estimated per-depth-stratum skip rates ~0.1% / ~0.1% / ~1.1% for 2 / 3–4 / ≥5 user turns). The prefix is the rendered conversation history before the final user turn (ending at the prior assistant end-of-turn); the context appends the final user turn and generation header; the answer is the generated span. The corpus-provenance record (eligibility counts, allocation, near-dupe drops, depth histogram) is `sampling_manifest/meta.json` on the HF data repo (pinned in the footer). The K-resample floor subsample kept 1,988 of 2,000 contexts — 12 admission-gate skips (5 over-length capture-skipped contexts, which by construction also lack a primary draw, plus 7 with no primary draw only), recorded in the committed skip file and its HF mirror.

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

The context arm shows a genuine nonlinear advantage that grows toward low-variance directions: the MLP-over-ridge gap rises from +0.023 (directions 1–16) to 0.067–0.068 (directions 65–256), and the per-direction-penalty control matches shared ridge (band means within 0.003), ruling out differential shrinkage. Prefix-arm gaps stay small (+0.011 / +0.027 / +0.026 / +0.008): relative to what an MLP can extract, the prefix-to-answer map is near linear-complete, and both fitters approach the sampling floor in the tail (ridge 0.071, MLP 0.079 at ranks 129–256). The K-resample floor puts answer-sampling noise at a median 6.1% of per-context error at layer 19 (7.2% / 5.5% at layers 14 / 26), capping idealized R² at ≈0.70 (context) and ≈0.42 (prefix): the prefix arm's unexplained variance is missing information, not noise.

---

**Repro:** ~25–27 GPU-h realized vs 77 budgeted (GCP A100-80 ephemeral instances for capture waves, fits, and the K-resample floor; VM CPU for the manifest build and characterization; Anthropic Batch API for judge labels) · Code: branch issue-1738 at [9fe2d7ecb4](https://github.com/superkaiba/explore-persona-space/tree/9fe2d7ecb4c8264e01b04f5c876354c1b537730f) (drivers `scripts/issue1738_multiturn_generate_capture.py`, `scripts/issue1738_multiturn_fits.py`, `scripts/issue1738_characterize.py`; figures `scripts/issue1738_analyzer_figures.py`; floor-adjusted family `scripts/issue1738_floor_adjusted_taxonomy.py`), harvest commit [a36e97847b](https://github.com/superkaiba/explore-persona-space/commit/a36e97847b), PR [#1501](https://github.com/superkaiba/explore-persona-space/pull/1501) · Eval JSONs: [eval_results/issue_1738 at the pin](https://github.com/superkaiba/explore-persona-space/tree/9fe2d7ecb4c8264e01b04f5c876354c1b537730f/eval_results/issue_1738) — `fits/multiturn_100k_fits.json` (30 cells, seed-43 repeat, LMSYS-transfer control), `h1_contrast.json`, `mapping_baselines.json`, `taxonomy.json`, `taxonomy_floor_adjusted.json`, `depth_contrasts.json`, `kresample/` (floor summary, gates, subsample, skip breakdown, per-layer floors), `perdirection/pdshrink_summary.json`, `judge_labels/labels.json`, and the per-context table `percontext_summary_L19_ridge.csv` (the per-cell file behind every aggregate here) · Raw data + tensors: [issue1738_multiturn on the HF data repo](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8fc7e9ceb824eed33474a954d6bc89c95c97cadf/issue1738_multiturn) — `sampling_manifest/meta.json` (the corpus-provenance record) plus manifest shards and `split_1738.json`, `capture/` (224 chunks), `raw_completions/` (255 files, all stages), `analysis_tensors/` (percontext, pred16, y_holdout, weights, summaries), `kresample/` (incl. `kresample_shard00_skipped.json`), `judge_labels/` · Figures: [figures/issue_1738 at the pin](https://github.com/superkaiba/explore-persona-space/tree/9fe2d7ecb4c8264e01b04f5c876354c1b537730f/figures/issue_1738) (PNG + PDF + meta.json sidecars); the earlier driver-rendered hero and depth figures (no per-figure sidecars) are superseded by the versions embedded above · Deviations: fits summaries rebuilt from retained fp16 predictions after the compute instance self-deleted pre-harvest (holdout metrics verified 11 of 11 vs poll-captured values; test-split diagnostics and wall clocks unrecoverable; LMSYS-transfer cells are a fresh deterministic CPU refit); taxonomy/depth stage re-run on the VM after a staging-path skip · Reused artifacts: fitter set + decoding recipe + capture convention from [#779](https://eps.superkaiba.com/tasks/779) (`scripts/issue779_ffc_n1m_fits.py` constants, `scripts/issue779_collect.py` capture convention) — fit: same model and measurement regime, corpus construction is the single changed variable; error-characterization battery code from [#1482](https://eps.superkaiba.com/tasks/1482) (`scripts/issue1482_*` stages, ported per-arm) — fit: same estimator regime at a 10k holdout.

**Context:** Origin (user chat, verbatim):

> run for 100k as much in parallel and vectorized as possible. [context: user approved option R3(b) from chat — 'new capture: build the multi-turn analogue of the n1M corpus — N multi-turn LMSYS/WildChat conversations (prefix = history, query = last turn), capture prefix-end + context-end + answer states, fit both arms, run the full pipeline' — after a sizing discussion grounding N=100k at ~50–70 GPU-h capture on the measured 3,250 ctx/GPU-h parent basis]

Lineage: [#1482](https://eps.superkaiba.com/tasks/1482) — parent (this run applies its error-characterization battery per input arm at multi-turn scale). Created 2026-07-28; run 2026-07-28 → 2026-07-29.
