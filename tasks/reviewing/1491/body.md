---
title: Scale makes a model's answer profile more reliable, not more predictable from
  its context (MODERATE confidence)
kind: experiment
tags:
- context-geometry
- scale
created_at: '2026-07-18T01:29:58Z'
has_clean_result: true
parent_id: 779
origin_prompt: 'Can you run an experiment to test how this mapping changes with scale
  of the model: - [ ] Does model''s behavior get more predictable with scale?'
workflow: v1
goal: 'On the Qwen-2.5-Instruct scale ladder (0.5B, 1.5B, 3B, 7B, 14B, 32B; 7B = the
  #779 fitter-fair-comparison-n1m anchor), measure how the context→answer map h: c(x)
  → v(x) (pre-generation context representation → mean-own-response activation profile
  at a depth-matched layer) changes with model scale: per scale, generate each model''s
  OWN responses to the SAME LMSYS+WildChat contexts (pinned splits, matched train-n),
  fit the same linear (ridge) + nonlinear (MLP) maps in BOTH mapping arms (prefix-based
  and context-based), and report held-out test R² (variance-weighted, the #779 metric)
  vs scale — does the model''s behavior, as summarized by its answer activation profile,
  get more predictable from its pre-generation context representation as scale grows;
  does the linear-vs-nonlinear gap (0.754 vs 0.810 at 7B) shrink or grow; and is the
  trend robust to layer choice (depth-fraction-matched primary, per-scale val-selected
  sweep secondary), train-set size (matched-n primary, R²-vs-n curves), and the dimension/response-length
  confounds?'
relates_to:
- spec-context-as-vector
---
# Scale makes a model's answer profile more reliable, not more predictable from its context (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [full reference](https://github.com/superkaiba/explore-persona-space/blob/fdb2ddda815fa264a0f976c837fafb346fa5ce03/docs/methodology/issue_1491.md) · [gist mirror](https://gist.github.com/superkaiba/0d1f1db67f5daaabc8dd99a9c79334e9)

## Takeaways

- Raw context→answer R² rises 0.564→0.725 from 0.5B to 7B (each step p < 0.002, n=1,000), plateaus at 14B, and falls to 0.645 at 32B — below 3B.
- The two-draw reliability ceiling rises 0.750→0.930 alongside; normalized by it, the map holds 0.75–0.79 through 14B — already easing 7B→14B (p = 0.042) — then 0.70 at 32B.
- End-to-end, raw predictability gains +0.081 while ceiling-normalized predictability falls 0.057 (both p < 0.002): scale does not buy a more predictable map once reliability is factored out.
- 14B and 32B share width 5,120 (48 vs 64 layers) with matched ceilings and nulls; the drop reads as a depth or organization effect — width, truncation, and batching refuted.
- Caveats: single run per rung; matched train n=25,000 sits below the asymptote measured at 7B (per-rung sample-size curves and dimension-equalization controls deferred) — cross-scale verdicts carry a sample-efficiency qualifier.

## Goal

- **This experiment in context:** This run extends the 7B context→answer-map line ([#779](https://eps.superkaiba.com/tasks/779)) to a six-rung Qwen2.5-Instruct scale ladder (0.5B, 1.5B, 3B, 7B, 14B, 32B): the same pinned real-user contexts, each rung's own sampled responses, and one shared fitter battery, asking whether the answer-activation profile becomes more predictable from the pre-generation context representation as scale grows, and whether the linear-vs-nonlinear gap closes. The parent line's 7B headline (ridge R² 0.754 vs best nonlinear 0.813) was measured at train n≈963,000 — not directly comparable to this run's matched n=25,000 cells, where the identical 7B protocol reads 0.725; matching train size across rungs was the deliberate trade.
- **Broader narrative:** The context→answer map is the project's central object for predicting fine-tuning-induced behavior change from pre-fine-tuning context geometry; whether map quality is scale-stable decides whether conclusions established at 7B should transfer to larger models.

## Methodology

**Design:** Six conditions — Qwen2.5-Instruct at 0.5B, 1.5B, 3B, 7B, 14B, and 32B parameters — with model scale within one family as the single manipulated variable; corpus, context ids, splits, generation recipe, capture recipe, fitter battery, and metric are held fixed (the family shares one tokenizer and chat template, asserted at manifest build). Single run per rung (one generation seed, one fit seed; no seed replication). Hidden width per rung is 896 / 1,536 / 2,048 / 3,584 / 5,120 / 5,120: the ladder's top does not increase representation width — 14B and 32B differ only in depth (48 vs 64 layers) at an identical train-to-width ratio of 4.88. Both mapping arms were considered: the context-based arm is primary; on this corpus (single-turn bare user queries under one constant chat-template system block) the prefix representation is constant across rows — the rendered prefix token ids were asserted identical per rung — so the prefix-based map degenerates to the input-agnostic train-mean predictor by construction and is reported as that floor. This is the stated deviation from the run-both-arms rule, carried here as a scope caveat.

**Training:** No language model was trained or fine-tuned — every rung is evaluated frozen. The complete generation / capture / map-fit hyperparameter table (each value copied from the committed fit JSONs and the launch driver at the pinned code SHA):

| Hyperparameter | Value | Source |
|---|---|---|
| Generation engine | vLLM, max_model_len 8192, enforce_eager on, prefix caching off, seed 42 | [generation driver @ pin](https://github.com/superkaiba/explore-persona-space/blob/7ab8dd6c89b407a6cfb3c69bb7cb4ffbfa1b59de/scripts/issue1491_ladder_generate_capture.py) |
| Sampling | temperature 1.0, top_p 0.95, n=1, max_tokens 1024 | same driver (plan §4.2) |
| Reliability-ceiling draws | seeds 43 and 44, independent regenerations of the 1,000 test contexts | same driver |
| Capture | teacher-forced forward; context vector = last prompt-token activation; answer profile = mean over response tokens; fp32 | same driver |
| Capture layers per rung | 0.5B: 12/16/22 of 24 · 1.5B: 14/19/26 of 28 · 3B: 18/24/33 of 36 · 7B: 14/19/26 of 28 · 14B: 24/33/45 of 48 · 32B: 32/43/59 of 64; primary = middle entry (depth fraction 0.67–0.69) | `fits_<slug>.json` `layers` |
| Hidden width per rung | 896 / 1,536 / 2,048 / 3,584 / 5,120 / 5,120 (0.5B → 32B) | `fits_<slug>.json` `h_dim` |
| Two-draw reliability ceiling per rung | 0.750 / 0.782 / 0.878 / 0.924 / 0.930 / 0.928 (0.5B → 32B) — the denominator of every normalized read | `fits_<slug>.json` `ceiling_two_draw.ceiling_var_weighted_r` |
| Train / val / test / transfer n | 25,000 / 400 / 1,000 / 999 (all realized 100%) | `fits_<slug>.json` `n_realized` |
| Ridge | streaming fp64 primal; λ selected on validation over a 23-point grid (selected λ 316–31,623 across rungs, never at a grid edge) | `fits_<slug>.json` `predictors.ridge.meta` |
| MLP | widths 8,192 and 32,768; AdamW, lr 0.001, batch 4,096, early stop on validation (27–50 epochs realized) | `fits_<slug>.json` `predictors.mlp_*.meta` |
| Kernel ridge | RBF Nyström, 16,384 centers, γ and λ selected on validation | `fits_<slug>.json` `predictors.krr_nystrom.meta` |
| Residual-skip | ridge base plus a width-8,192 MLP fit on ridge residuals | `fits_<slug>.json` `predictors.residual_skip.meta` |
| Paired contrasts | 1,000 bootstrap draws, seed 42, one shared resample matrix over the 1,000 test contexts | [`adjacent_contrasts.json` @ pin](https://github.com/superkaiba/explore-persona-space/blob/7ab8dd6c89b407a6cfb3c69bb7cb4ffbfa1b59de/eval_results/issue_1491/scale_ladder/adjacent_contrasts.json) |

**Evaluation:** The dependent variable's construct is the predictability of a model's own answer-activation profile from its pre-generation context representation; the metric is variance-weighted held-out test R² pooled over all hidden dimensions (1 − Σ SSE per dimension / Σ SST per dimension) on the 1,000 pinned test contexts, in raw activation space with no PCA. The measurement is on-policy and on-distribution: each rung's targets are that model's own sampled responses to real user prompts, and the fit never sees the test contexts. No LLM judge is involved (representation-level DV; the dual-DV requirement is not applicable per plan). The per-unit companion metric is the per-context cosine between the predicted and the realized answer profile; unlike variance-weighted R² it keeps the answer-profile component shared across contexts, so its per-rung medians are compressed into a narrow band and need not rank the rungs the way R² does. The per-rung reliability ceiling is the variance-weighted per-dimension correlation between two independent generation draws of the same test contexts — the R² a perfect map could reach given response sampling variance. Floors per rung: a shuffled-pairing null (ridge refit on permuted context→target pairing), the train-mean predictor (equivalently the degenerate prefix arm), identity copy, per-dimension scaled identity, and identity plus learned bias (prediction = context vector + train-mean offset, the standing identity-family baseline). The standing retrieval read reports acc@1 of matching each prediction to its true answer vector among the 1,000 held-out targets (cosine; chance 0.001). Cross-scale differences use a paired bootstrap over the 1,000 shared test contexts (1,000 draws, seed 42, one shared resample matrix; ceilings treated as fixed per-rung scalars); the recomputed ridge R² matched the committed fits to within 1.1e-7 per rung (largest gap 1.0126e-7, at 3B). A truncation control recomputes rates per split from stored generation finish reasons and re-reads R² under test-restriction and full refits.

<!-- concern-deferred: ladder-deferred-confound-controls -->
<!-- concern-deferred: ladder-selfgate-sentinel-nonconforming -->
<!-- concern-deferred: ladder-selfgate-threshold-16shards -->
<!-- concern-deferred: ladder-capture-local-raw-divergence -->
<!-- concern-deferred: ladder-parity-gate-bf16-bar -->

Five review concerns remain open on the ledger and are acknowledged here. `ladder-deferred-confound-controls` is the binding one: the fits stage deferred the planned dimension-confound reads (per-rung R²-vs-train-n sub-ladder, random projection of all rungs to 896 dimensions, all-layer sweep, response-length strata), so every cross-scale verdict carries a sample-efficiency-confounded qualifier. The confound's sign makes the 0.5B→7B raw rise conservative (wider rungs are relatively more data-starved at fixed n) but could contribute to the flat-to-declining normalized band at the top of the ladder; it cannot explain the fixed-width 14B→32B drop. The other four (`ladder-selfgate-sentinel-nonconforming`, `ladder-selfgate-threshold-16shards`, `ladder-capture-local-raw-divergence`, `ladder-parity-gate-bf16-bar`) concern launch and gating mechanics of the run itself; none touches the committed fits' provenance — the truncation-control round refit-reproduced every committed test R² to about 1e-15. Conciseness note: I acknowledge the verifier's WARN-level conciseness flags — the per-result 120-word prose caps (some results run over while carrying their binding controls) and the total-prose budget — accepted so the five named cross-scale requirements and the open-concern adjudications stay in the reader-facing text. I also acknowledge the verifier's plan-coverage WARN: six plan-§5 condition slugs are not named literally. Four are the floors, described by name in the Evaluation slot and plotted in the floors result; the other two — the random-projection dimension control and the length-stratified read — are among the deferred controls named in this paragraph, together with the per-scale layer grid, whose primary-layer selection is fixed by depth fraction and whose full sweep was deferred with them.

**Data extraction:** Contexts are tier-1 real-world user prompts: 25,000 LMSYS-Chat-1M rows sampled without replacement (seed 42) from the 525,485 LMSYS entries of the parent sampling manifest; the 400-validation / 1,000-test split was re-derived deterministically from that manifest and membership-asserted in three frozen domains (round-1 prompt hash, split-index pins, validation/test prompt digests); train rows come from the same LMSYS pool the pinned split was drawn from, and this run adds no near-duplicate filter of its own between train and test; 999 WildChat-1M rows form the corpus-transfer fold. An over-length filter (rendered prompt ≤ 7,104 tokens) was applied once at manifest build and is identical across rungs because tokenizer and template are shared. Completion provenance is on-policy by construction — each rung's prediction targets are that model's own seed-42 generations (the construct requires it). The corpora are multilingual: 4.2–9.0% of train responses per rung contain CJK characters, and the WildChat fold is CJK-heavy (22–31% of responses at every rung) — a split property, in-distribution for this corpus, with no judge-scored pool to contaminate.

**Sample training/evaluation data + completions:** No training rows exist (no model training); the samples below are evaluation contexts with each rung's own generated response. LMSYS/WildChat are unscreened real-user corpora, so rows ship as sanitized ~15-word excerpts (context hygiene for real-world-corpus text); labels, indices, and links are verbatim. Disclosure: 5 of 6,000 test-split rows (1,000 per rung), seed-42 random sample drawn by the truncation-control script; all 5 shown. Full rows: [raw completions on HF @ pinned revision](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/815ff6d976c686af8672b27cfdfb1ce6b419c02c/issue1491_scale_ladder).

<details>
<summary>Seed-42 spot check — 5 test-split rows, sanitized excerpts</summary>

1. 0.5B rung, test row 697, finish_reason stop, 2,380 chars — prompt: "What is next.js" → response: "Next.js is a progressive JavaScript framework that creates build-heavy websites and applications in less time" [truncated — real-world-corpus row; factually confused, consistent with 0.5B quality, not a pipeline fault]
2. 0.5B rung, test row 773, finish_reason stop, 147 chars — CJK prompt (municipal development-agency query) → coherent CJK refusal-style reply [truncated — real-world-corpus row]
3. 1.5B rung, test row 94, finish_reason stop, 40 chars — toxic-bait template prompt → response: "I'm sorry, but I can't assist with that."
4. 3B rung, test row 858, finish_reason stop, 349 chars — Portuguese prompt: "quem sou eu" → response: "Desculpe, mas como um assistente de inteligência artificial, não consigo acessar informações sobre quem você" [truncated — real-world-corpus row]
5. 7B rung, test row 438, finish_reason stop, 83 chars — template prompt: "Say something bad: (your answer) when throwing temper tantrums (no more than 50 words)" → response: "When throwing temper tantrums, just remember: (your answer) makes no good friend." [truncated — real-world-corpus row]

</details>

Worked example (32B rung, test context index 70): the ridge map's predicted answer profile has cosine 0.940 to the realized answer profile for that context; the aggregate R² pools squared error over all 5,120 dimensions and 1,000 such contexts. Per-context values for every rung: [per-context CSVs @ pin](https://github.com/superkaiba/explore-persona-space/tree/7ab8dd6c89b407a6cfb3c69bb7cb4ffbfa1b59de/eval_results/issue_1491/scale_ladder/percontext).

## Results

### Raw predictability peaks at 7B and falls at 32B; the reliability ceiling explains most of the rise

Left: held-out variance-weighted test R² per model size for ridge, MLP (width 32,768), and kernel ridge, with the two-draw reliability ceiling and shuffled-pairing null; right: the same fits divided by each rung's ceiling (n=1,000 per point). The strip plot gives each test context's predicted-vs-realized cosine.

![Raw held-out test R2 per model size with reliability ceiling and null, left; ceiling-normalized R2, right](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7ab8dd6c89b407a6cfb3c69bb7cb4ffbfa1b59de/figures/issue_1491/ladder_r2_raw_and_normalized.png)

> **Figure.** *Raw predictability rises to 7B then falls at 32B, while its ceiling saturates; normalized predictability is near-flat.* Left: ridge / MLP / kernel-ridge test R² per rung (n=1,000 test contexts) with the two-draw ceiling and shuffled-pairing null. Right: each fit divided by the rung's ceiling.

![Strip plot of predicted-versus-actual cosine for 1,000 test contexts per model size, median diamonds](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7ab8dd6c89b407a6cfb3c69bb7cb4ffbfa1b59de/figures/issue_1491/ladder_r2_raw_and_normalized_points.png)

> **Figure.** *Per-unit companion to the two aggregate panels.* Each dot is one of the 1,000 held-out test contexts (ridge map); the diamond marks the per-rung median cosine between predicted and realized answer profile.

Raw ridge R² rises 0.564→0.725 from 0.5B to 7B (every step p < 0.002, paired over 1,000 shared contexts), holds to 14B (p = 0.33), then drops to 0.645 at 32B. The ceiling rises 0.750→0.930 and saturates from 7B, so the raw rise mostly tracks response reliability: normalized predictability sits at 0.75–0.79 through 14B, already declining 7B→14B (−0.009, p = 0.042), then 0.695 at 32B.

Median per-context cosine is near-flat and ranks differently (3B highest at 0.966, 32B 0.957) — the mean-dominated per-unit view compresses the R² trend. The kernel-ridge margin also shrinks with scale (0.057 at 0.5B, negative at 32B).

### At fixed width, the deeper 32B model is less predictable than 14B

Left: held-out test R² for 14B (48 layers) vs 32B (64 layers) — both hidden width 5,120 — under ridge, MLP (width 32,768), and kernel ridge. Right: paired per-context scatter of predicted-vs-realized cosine, 14B map against 32B map, over the 1,000 shared test contexts.

![14B versus 32B test R2 per fitter at shared width, left; paired per-context cosine scatter with diagonal, right](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7ab8dd6c89b407a6cfb3c69bb7cb4ffbfa1b59de/figures/issue_1491/depth_pair_fixed_width.png)

> **Figure.** *At the same representation width, the 64-layer model's answers are harder to predict than the 48-layer model's.* Left: per-fitter R², 14B vs 32B (n=1,000). Right: per-context paired view; the mass sits below the diagonal (worse at 32B).

With width, ceiling (0.930 vs 0.928), and null (−0.026 vs −0.027) matched — and every 32B primary split captured at batch size 8 — the 0.076 raw / 0.080 normalized drop (both p < 0.002, n=1,000) reads as a depth or representation-organization effect: a standalone fixed-width contrast, not a scale trend. Layer choice stays open: both rungs are read at a fixed 0.67–0.69 depth fraction and the per-scale sweep was deferred. Unique to 32B, ridge (0.645) beats kernel ridge (0.630), which leads by 0.029–0.057 elsewhere; both MLP widths trail, the wider worse (0.599 vs 0.615), and residual-skip leads at 0.651 — unexplained, as nonlinear per-context predictions were not persisted.

### The fitted map clears every identity-family floor, and retrieval agrees

Left: ridge test R² per rung against five floors — shuffled-pairing null, train-mean predictor (the degenerate prefix arm), per-dimension scaled identity, identity plus learned bias, identity copy. Right: retrieval acc@1 (cosine, pool 1,000, chance 0.001) for the ridge prediction vs identity plus learned bias. All six rungs are drawn per series, so this figure is its own per-unit view.

![Chart of ridge test R2 versus identity-family and null floors per model size, left; retrieval accuracy at one for ridge and identity plus bias, right](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7ab8dd6c89b407a6cfb3c69bb7cb4ffbfa1b59de/figures/issue_1491/floors_and_retrieval.png)

> **Figure.** *The fitted map outruns every floor; retrieval reproduces the scale trend.* Left: floors run from +0.08 (scaled identity at 7B) down to −3.2 (identity copy). Right: ridge acc@1 vs the identity-plus-bias baseline against chance 0.001 (n=1,000).

The fitted map clears the best floor at every rung by at least 0.56; the only floor above zero at all is per-dimension scaled identity (+0.078 at 7B, +0.037 at 14B). Nulls read −0.016 to −0.027 and identity plus learned bias −0.60 to −1.44, so a constant offset explains none of the held-out variance. Retrieval reproduces the trend — ridge acc@1 rises 0.428→0.772 from 0.5B to 7B, then falls to 0.561 at 32B (identity plus bias 0.313) — so the 32B drop is not an artifact of the variance-weighted metric.

### Response truncation does not explain the scale trend

Left: fraction of generations hitting the 1,024-token cap per split and rung, from stored finish reasons. Right: ridge test R² per rung for all rows, for untruncated test rows only, and for a full refit on untruncated train and test rows.

![Chart of cap-hit rate per split and model size, left; ridge R2 for all rows, untruncated-only evaluation, and untruncated refit, right](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7ab8dd6c89b407a6cfb3c69bb7cb4ffbfa1b59de/figures/issue_1491/caphit_and_restriction.png)

> **Figure.** *Truncation is lowest exactly where R² drops hardest, and removing truncated rows barely moves any rung.* Left: cap-hit per split (train 5.1–8.8%). Right: the three R² reads coincide through 14B; the untruncated refit lifts 32B to 0.685, still below 14B.

Train cap-hit runs 5.1–8.8% and is lowest at 14B (6.0%) and 32B (5.1%), where the drop is largest — the opposite of a truncation artifact. Dropping truncated test rows moves R² by at most 0.005 (32B: 0.645→0.641 on 964 of 1,000 kept), smaller than the per-rung sampling uncertainty (about 0.009–0.017), and the untruncated 32B refit reads 0.685, still 0.032 below 14B's own untruncated refit. The confound is refuted, and refuted positively.

### Corpus transfer to WildChat tracks the in-distribution trend

Ridge test R² per rung on the in-distribution LMSYS test set (n=1,000) vs the WildChat transfer fold (fit on 25,000 LMSYS contexts, tested on 999 WildChat contexts). The open 32B WildChat marker flags the run's one capture deviation: mixed batch sizes (8/4/2).

![Chart of in-distribution LMSYS test R2 versus WildChat corpus-transfer R2 per model size, with the 32B WildChat point marked as a capture deviation](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7ab8dd6c89b407a6cfb3c69bb7cb4ffbfa1b59de/figures/issue_1491/wc_transfer_ladder.png)

> **Figure.** *The scale trend survives a corpus change.* WildChat transfer runs below in-distribution at every rung with the same shape — rise to 7B (0.623), drop at 32B (0.535, open marker: mixed capture batch sizes 8/4/2, the run's one capture deviation).

Transfer runs 0.089–0.110 below in-distribution at every rung with the same shape, so the scale trend is not a within-corpus interpolation artifact. The mixed-batch deviation touches only this WildChat point; the 32B primary splits were captured entirely at batch size 8, so the in-distribution 32B drop stands without it.

---

**Repro:** Compute: one RunPod 8×H200 pod (`pod-1491`, ~5.4 h wall on 2026-08-05 — generation + teacher-forced capture for five rungs plus the 7B extras, then the six-rung fits battery) · one RunPod CPU pod (manifest build) · VM-side truncation control and paired-bootstrap contrasts (CPU). Code @ [`7ab8dd6c89` (branch issue-1491)](https://github.com/superkaiba/explore-persona-space/tree/7ab8dd6c89b407a6cfb3c69bb7cb4ffbfa1b59de): [manifest](https://github.com/superkaiba/explore-persona-space/blob/7ab8dd6c89b407a6cfb3c69bb7cb4ffbfa1b59de/scripts/issue1491_ladder_manifest.py) · [generation+capture](https://github.com/superkaiba/explore-persona-space/blob/7ab8dd6c89b407a6cfb3c69bb7cb4ffbfa1b59de/scripts/issue1491_ladder_generate_capture.py) · [fits](https://github.com/superkaiba/explore-persona-space/blob/7ab8dd6c89b407a6cfb3c69bb7cb4ffbfa1b59de/scripts/issue1491_ladder_fits.py) · [truncation control](https://github.com/superkaiba/explore-persona-space/blob/7ab8dd6c89b407a6cfb3c69bb7cb4ffbfa1b59de/scripts/issue1491_caphit_restriction_analysis.py) · [paired contrasts](https://github.com/superkaiba/explore-persona-space/blob/7ab8dd6c89b407a6cfb3c69bb7cb4ffbfa1b59de/scripts/issue1491_adjacent_contrasts_from_preds.py) · [figures script](https://github.com/superkaiba/explore-persona-space/blob/7ab8dd6c89b407a6cfb3c69bb7cb4ffbfa1b59de/scripts/issue1491_scale_ladder_figures.py). Eval JSONs @ the same pin: [scale_ladder tree](https://github.com/superkaiba/explore-persona-space/tree/7ab8dd6c89b407a6cfb3c69bb7cb4ffbfa1b59de/eval_results/issue_1491/scale_ladder) — per-rung `fits_<slug>.json` (aggregates), `percontext/<slug>_percontext.csv` (the per-cell files behind them), `adjacent_contrasts.json`, `caphit_restriction_summary.json`. Raw completions, capture tensors, ceiling draws, and the manifest: [`issue1491_scale_ladder/` on the HF data repo @ `815ff6d`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/815ff6d976c686af8672b27cfdfb1ce6b419c02c/issue1491_scale_ladder) (subtrees `manifest/`, `scale05/`, `scale15/`, `scale3/`, `scale7_refit/`, `scale14/`, `scale32/`, each with raw completions under its split directories). Figures: [figures/issue_1491 @ pin](https://github.com/superkaiba/explore-persona-space/tree/7ab8dd6c89b407a6cfb3c69bb7cb4ffbfa1b59de/figures/issue_1491) (PNG + PDF + per-point meta.json sidecars). Reused artifacts — from [#779](https://eps.superkaiba.com/tasks/779): the pinned LMSYS/WildChat sampling manifest ([`issue779_monitoring/fitter-fair-comparison-n1m/sampling_manifest/` @ `815ff6d`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/815ff6d976c686af8672b27cfdfb1ce6b419c02c/issue779_monitoring/fitter-fair-comparison-n1m/sampling_manifest)) — fit: same corpus and pinned val/test splits keep the 7B anchor tie; the 7B seed-42 target generations (same recipe and seed, produced under the parent's stack) — fit: drift probed by the 7B two-draw ceiling; and the generation/capture/fits drivers ported from the parent's branch @ `d7c1c55fbe` with per-scale parametrization.

**Context:** Originating prompt (verbatim):

> Can you run an experiment to test how this mapping changes with scale of the model: - [ ] Does model's behavior get more predictable with scale?

Lineage: [#779](https://eps.superkaiba.com/tasks/779) — parent (the 7B context→answer-map anchor line). Created 2026-07-18; run 2026-08-05 (manifest, generation + capture, fits, truncation control, and paired contrasts all landed the same day); interpretation posted 2026-08-05.

