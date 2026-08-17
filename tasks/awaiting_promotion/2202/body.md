---
title: Rank-1 retrieval failures of the context→answer map are mostly map error dragged
  toward hub answers, not irreducible answer degeneracy (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-08-08T16:11:03Z'
has_clean_result: true
parent_id: 1738
origin_prompt: 'Motivation: We did an analysis of the directions the model fails on
  using SAE features; SAE features are known to be somewhat unreliable; we want to
  do a more controlled analysis of this question. Methodology: apply our best mapping
  on the generic corpus; see for which ones it fails to distinguish the correct answer
  vector from some other answer vector; look at the contexts it fails on and characterize
  what kinds of things it fails on. (Full verbatim request + resolved clarifications
  in the body''s ## Provenance section.) [then] run it in background with happy coder'
workflow: v1
goal: 'On the #1738 ridge context→answer map (context arm, layer 19, Qwen-2.5-7B-Instruct,
  100k multi-turn LMSYS/WildChat corpus, pinned held-out n=9,941), characterize WITHOUT
  SAE features which contexts the map fails to retrieve — i.e. whose predicted answer
  vector does not single out the true answer vector among all 9,941 held-out answers
  — by (1) building a dashboard of every rank-1 failure plus the worst-rank tail carrying
  the full confusion geometry (context↔context, answer↔answer, answer↔confuser-context
  and prediction↔confuser similarities, plus pool-wide ranks, in raw, mean-centered
  and whitened metric spaces); (2) building a 500-context random-sample dashboard
  carrying both the retrieval neighbour list and the prediction-collapse neighbour
  list; and (3) measuring the symmetry (reciprocity) of the confusion graph against
  a degree-preserving null and a distance-only null — with failure attributed to map
  error vs irreducible target degeneracy or answer-sampling noise via the banked K-resample
  retrieval ceiling.'
relates_to:
- spec-context-as-vector
---
# Rank-1 retrieval failures of the context→answer map are mostly map error dragged toward hub answers, not irreducible answer degeneracy (MODERATE confidence)

<!-- clean-result-v4 -->
**Methodology:** [docs/methodology/issue_2202.md](https://github.com/superkaiba/explore-persona-space/blob/138dc7b7ab70f8b0c046cce9b06b6dbd50231a79/docs/methodology/issue_2202.md) · [gist mirror](https://gist.github.com/superkaiba/4183c6b2ebc0780cf7238af8bc730b5c)

<!-- Raw-output spot check (5 random rows, seed 42, percontext_ranks.csv joined with the local text cache; sanitized ~12-word excerpts):
ci 18784 rank 1 - "I mean do you have general knowledge of 2023?" -> knowledge-cutoff hedge; clean.
ci 4192 rank 1 - title-generation task -> "Arctic Naval Buildup: ..." title; clean.
ci 45071 rank 1 - "ls" -> simulated terminal listing; terse/roleplay row retrieved correctly; clean.
ci 40144 rank 1 - self-supervised-learning intro request -> long intro; clean.
ci 36811 rank 1 - "I like the second challenge. can you give me 2 more?" -> two challenges; clean. NOTE: carries kres_class=MAP_ATTRIBUTABLE despite rank 1; the CSV writes the raw resample partition for every covered row; the class is only meaningful for FAIL rows (attribution.json counts classes over failures only).
0 of 5 fishy (no judge/content disagreement, no corruption, no empty outputs).
Round-4 label-wave spot check (5 random labeled items, seed 42, judge_labels_2202/labels.json): c51519 control, c18793 control, f29545 fail, f18648 fail, f10616 fail — all 5 carry 10 yes/no fields, arms coherent, label patterns plausible (failures show contentless final turns + placeholders; controls show distinctive/echoing modes); 0 of 5 fishy. -->

## Takeaways

- 297 of 368 resample-covered rank-1 failures (80.7%) remain retrievable from a fresh on-policy answer draw — map error, not answer degeneracy; the 60% falsification threshold fired. The same verdict holds at the clean operating point: none of the 15 residual failures is an answer twin (retrieved-vs-true whitened cosine ≤ 0.36), 13 of 15 are map-attributable, and they are near-misses on terse, underdetermined turns — losing by ~7× less than successes win (median margin −0.025 vs +0.175).
- Failure hot-spots: refusal answers +24.8 pp, NSFW topics +21.8 pp, refusal-adjacent requests +16.4 pp, code +11.0 pp, English +8.2 pp (refuting the non-English prior); 13 of 22 contrasts cleared q = 0.05.
- Confusion is one-way but not purely hub-driven — reciprocity 8.4e-4 over 329,448 edges is 2.6× the collision-free degree-preserving null median yet below every distance-only band — and a hub-penalizing rescoring (CSLS, K = 10) recovers 969 of 1,829 failures (0.816 → 0.909 raw); on the whitened-cosine base it reaches 0.976, at or above its convention-matched fresh-draw reference of 0.973.
- Matching the similarity convention resolves the apparent map-above-reference inversion: the fresh-draw reference recomputed under whitened cosine is 0.979 (raw 0.943), shrinking the covered-row map-to-reference gap from 12.8 pp to 1.8 pp; averaging 5 answer draws into the target lifts covered-row rank-1 accuracy 0.815 → 0.909 raw and 0.962 → 0.987 whitened — the map predicts the noise-averaged answer; discrimination-trained (InfoNCE) maps top out at 0.956 whitened, below the 0.976 metric-side correction — the bottleneck was the metric, not the map; combining both fixes, all 7 map architectures converge to 0.991–0.995 (best cell 0.995).
- Judge-scored modes (round-4 repaired discovery): contentless final turns +8.1 pp and language-switched or garbled answers +6.9 pp lead six failure-enriched modes; distinctive self-contained queries −8.7 pp and long topic-echoing answers −7.8 pp protect; 1 of 10 modes demoted at retest κ below 0.6.
- Coverage gaps: attribution covers 368 of 1,829 failures; context arm only; 13 of 1,000 mode-discovery digest rows excluded for model content refusals (reported, never silent); 5 of 4,145 judge calls dropped.

## Goal

- **This experiment in context:** Prior failure characterizations of the 100k multi-turn context→answer map used sparse-autoencoder features ([#1482](https://eps.superkaiba.com/tasks/1482), [#1946](https://eps.superkaiba.com/tasks/1946), [#2163](https://eps.superkaiba.com/tasks/2163)) — a lossy basis (fraction of variance explained 0.718 at layer 19). This run asks the same question with an SAE-free instrument: nearest-neighbour retrieval among the held-out answer vectors of the parent map ([#1738](https://eps.superkaiba.com/tasks/1738)), separating map error from target degeneracy and sampling noise via the parent's banked resample control. Context arm only: the prefix arm retrieves at rank-1 accuracy 0.183 under the same split and eval, making failure the default case there; its taxonomy is the named follow-up.
- **Broader narrative:** the mapping line asks how much of an answer's representation is a linear function of its context's. Where retrieval fails decides whether the residual is structured map error a better map could fix, or irreducible answer-sampling entropy — the distinction the leakage-prediction theory needs.

## Methodology

**Design:** one zero-GPU analysis pass over the banked context-arm layer-19 ridge map: full-pool retrieval ranks and confusion geometry in five similarity conventions, a two-stage failure-mode wave (Fable 5 proposes, Sonnet 4.5 counts), and a confusion-graph symmetry read against two nulls. Controls: the four-draw resample retrievability control, a matched non-failure control (cell-matched on depth band × corpus × language), the identity-plus-learned-bias baseline, degree-preserving (stub, plus a round-2 collision-free swap rebuild) and distance-only reciprocity nulls, and pool sizes 500 / 2,000 / 9,941. Nine analysis rounds (geometry + judged waves; a collision-free null rebuild; a zero-GPU hub-correction follow-up; a mode-discovery instrument repair re-running the Fable digests and the full Sonnet label wave; two user-chat inline free-analysis rounds — a matched-convention fresh-draw and draw-averaged-target read, and an 18-convention similarity battery; a GPU-backed inline round fitting discrimination-trained maps; a draw-averaged-target coverage-matrix completion; and a residual-failure and margin read at the clean operating point); no new generation or capture — the only training is the round-7 InfoNCE map fits on the banked activation tensors (no language-model training).

**Training:** **N/A — no model training.**

**Evaluation:** for held-out context *i*, the prediction is the banked ridge output; the candidate pool is all 9,941 held-out true answer vectors; the rank is the mid-rank of the true answer by distance to the prediction (ties mid-ranked). A rank-1 failure is a rank above 1 under raw euclidean — the banked headline convention. A reproduction gate re-derived the banked retrieval numbers from the banked tensors before anything downstream ran: rank-1 accuracy 0.816014485464239 euclidean (delta exactly 0; mean-reciprocal-rank delta 1.5e-7) and 0.8281862991650739 cosine (delta 0; two knife-edge tie rows at rank 5, within the two-row tolerance); chance is 1/9,941 ≈ 0.0001. The identity-plus-learned-bias baseline re-derived to rank-1 accuracy 0.473 euclidean / 0.512 cosine with held-out R² −1.08 (matches banked) — the ridge map's 0.816 / 0.828 sits far above both. Composition statistics use the parent battery: group-vs-rest failure-rate delta per label contrast, 10,000-draw bootstrap confidence bands, 10,000 permutations, false-discovery correction at q = 0.05; the judged failure modes form a second, separate correction family (never joined to the banked family). Analysis constants:

| Constant | Value | Source |
|---|---|---|
| Failure definition | mid-rank of true answer above 1, raw euclidean | parent banked headline (`eval_results/issue_1738/mapping_baselines.json`) |
| Similarity conventions | raw euclidean; raw cosine; mean-centered cosine (per-family train means); whitened euclidean and whitened cosine (shrunk train-answer covariance, λ = 0.1) | task-body lock; `analysis/null_battery.PRIMARY_LAMBDA` |
| Worst tails | top 200 by rank and top 200 by prediction-to-answer distance | task-body lock |
| Random sample | 500 contexts, seed 2202 | task-body lock |
| Resample control | 1,988 contexts × 4 extra on-policy answer draws; map-attributable at s ≥ 0.75, irreducible at s ≤ 0.25, ambiguous at 0.5 (s = fraction of draws retrieved at rank 1) | parent banked artifact; plan §11 operationalization |
| Composition battery | 10,000 bootstrap draws; 10,000 permutations; q = 0.05; 22 banked contrasts | parent battery (`scripts/issue1482_analysis.py`) |
| Reciprocity nulls | degree-preserving target-stub permutation, 1,000 draws (keeps colliding duplicate edges, 22.5% per draw — superseded in round 2); collision-free sensitivity rebuild: Maslov–Sneppen double-edge swaps from the observed graph (exact degrees, no duplicates), burn-in 5×E accepted swaps, one sample per E/2 accepted swaps, 200 samples, acceptance 0.135, lag-1 autocorrelation of samples 0.11; distance-only kernel, 1,000 draws per τ, τ at the 1st / 5th / 25th percentile of pairwise answer distances (headline at the 5th) | task-body lock; plan §11 sensitivity sweep; round-2 critique |
| Confuser display cap | top 10 per failure row | task-body lock |
| Hub-corrected retrieval (round 3) | CSLS, K = 10; cosine-native scores; retrieval by negative CSLS score, ties mid-ranked; baseline legs reconciled to the banked values within the reproduction-gate tolerances | banked metric battery (arXiv 1710.04087 convention); `scripts/issue2202_csls_followup.py` |
| Mode-discovery instrument (round 4) | Fable 5 digest chunks of 25 rows; 32,000-token output cap with explicit SDK timeout; stop reason persisted per call; blank replies hard errors; production-shape pilot gate on a real failure-digest chunk (PASS: 149,745 prompt chars, clean end-of-turn stop); refused chunks fall back to per-row calls with dropped-and-reported exclusions; hierarchical consolidation, 13 batches of 100 proposals | round-4 repair (`scripts/issue2202_labels.py`); `llm-judging.md` rules 23/26 |
| Judge | `claude-sonnet-4-5-20250929`, Batch API, `max_tokens` 2048, temperature API default (1.0), 1 draw per item | project judge rule; parent instrument settings |
| Judge quality control | 150-item pilot (0 truncation stops, 0 parse failures, PASS); 200-item test-retest; modes with κ below 0.6 demoted to report-only | plan gate; parent κ convention |
| Reproduction-gate tolerances | rank-k accuracy within 2e-4; mean reciprocal rank within 1e-4; n exactly 9,941 | parent banked values |
| Matched-convention + draw-averaged read (round 5, `freshwhiten-avg`) | fresh-draw reference recomputed under whitened cosine (the raw leg reproduces the banked 0.9425 with delta exactly 0); draw-averaged target = pool entry of each covered row replaced by the mean of its 5 on-policy draws (original + 4 fresh), pool size unchanged at 9,941, queries = the banked map predictions; two addendum conventions — candidate-normalized R² (per-candidate denominator, so mean-proximal hub candidates are penalized) and per-vector Pearson r (identical to raw cosine by construction); banked nonlinear held-out predictions (MLP width 8,192 at two seeds, Nystrom kernel ridge, residual-skip) scored on the identical battery, MLP user-opted-in; commit f2e08f4a01 | user-chat inline round; `scripts/issue2202_freshwhiten_avg.py`; 0 GPU-h |
| Similarity-convention battery (round 6, `metric-zoo`) | 18 new conventions: hub-penalizing rescorings (CSLS on whitened cosine, pool-in-degree penalty, mutual proximity, NICDM local scaling, DisSimLocal, inverted softmax at inverse temperature 30 and 10) and vector-space transforms (whitening at shrinkage 0.3 / 0.5, truncated whitening at 64 / 256 / 1,024 directions in euclidean and cosine reads, half-power whitening, per-dimension z-scoring in both reads, all-but-the-top); reconciliation legs reproduce the banked raw-euclidean, raw-cosine, and whitened values exactly; fresh-draw references recomputed convention-matched for the top performers; two roster asks reported as algebraic identities instead of run — pool-mean centering is a no-op for euclidean distance, and per-query CSLS differs from a candidate-penalty-only rescoring by a row constant; commit 6d8edcff | user-chat inline round; `scripts/issue2202_metric_zoo.py`; literature roster in `eval_results/issue_2202/metric_zoo/research_notes.md`; 0 GPU-h |
| Discrimination-trained maps (round 7, `contrastive-maps`) | InfoNCE loss with 2,048 in-batch negatives, symmetric (context→answer and answer→context); a linear map warm-started at the banked ridge solution and an MLP matching the banked width-8,192 architecture (MLP user-opted-in); temperature grid 0.05 / 0.1 / 0.2 selected on the pinned validation rows (both families selected 0.05), early stopping patience 8, max 50 epochs, learning rate 1e-4 linear / 3e-4 MLP, weight decay 0; fits on the banked layer-19 tensors at the pinned parent revision (n_train = 88,378); reconciliation reproduces the banked ridge accuracy and both banked R² values exactly; commits 591ba8d32d (script) / 303ace06a8 (evals); ~1.2 h realized on one H100 (`pod-2202-contrastive`) including recovery | user-chat inline GPU override; `scripts/issue2202_contrastive_maps.py` |
| Draw-averaged coverage matrix (round 8, `avgtgt-completion`) | the round-5 draw-averaged-target read extended to all 7 maps and their conventions on the same 1,988 covered rows; pool-side CSLS and in-degree statistics recomputed on the modified pool for every averaged cell (averaged entries change the pool geometry); contrastive predictions recomputed on the VM from the banked fit weights — recomputed full-pool raw-cosine accuracies match the banked battery with delta exactly 0 for both maps; ridge raw and whitened-cosine cells reproduce the round-5 values exactly; commit 2c4bf031679c; 652 s VM CPU | user-chat inline round; `scripts/issue2202_avgtgt_completion.py`; 0 GPU-h |
| Residual-failure + margin read (round 9, `residual-read`) | per-row retrieval margin = true-target score minus best-competitor score, in each convention's own units (euclidean margins are negative-distance gaps; cosine/CSLS margins are score gaps); per-row ranks + margins persisted for 4 conventions × single/averaged targets (`residual_read/percontext_ranks_margins.npz`); twin test = whitened cosine between the true answer and the retrieved competitor, twin threshold 0.95; pairwise AUC = 1 − (mean rank − 1)/(n_pool − 1); worst-discriminated tail = bottom 50 covered rows by margin under the headline convention, composition vs the 1,988-row pool and the raw-euclidean failure set, persisted in the `worst_discriminated` block of `residual_read/summary.json` (competitor ids in the npz); independently recomputed with exact agreement (cross-validation, recorded in the fold marker); reconciliation legs reproduce the round-8 accuracies exactly; commits 0e2d825ea8 / ed0165405a; ~120 s VM CPU | user-chat inline round; `scripts/issue2202_residual_read.py`; 0 GPU-h |

Fable 5 (`claude-fable-5`) generated failure-mode hypotheses only; every countable label is a Sonnet call. In the round-4 wave Sonnet labeled 4,145 items — all 1,829 failures, 1,816 matched controls (cells capped at available non-failures), and the 500-sample — of which 4,140 returned valid labels (5 API-error drops, all in the control arm; zero content refusals, zero transport losses; all 4,140 stops are clean end-of-turn). The 200-item test-retest re-ran with zero drops.

**Data extraction:** all inputs are banked artifacts of the parent run, reproduced here for self-containment. The corpus is 100,000 multi-turn contexts drawn from LMSYS-Chat-1M and WildChat-1M — real user conversations (tier-1 realism). Each context was rendered in the Qwen-2.5-7B-Instruct chat template and answered once on-policy (sampled decoding, 7,104-token generation budget). Residual-stream states were captured at layer 19 in a teacher-forced forward pass: the context vector is the last prompt-token state (the newline before the assistant turn); the answer vector is the mean over answer-token states. A ridge map from context vector to answer vector (dimension 3,584) was fit on 88,378 training rows over 23 log-spaced penalties (1e-3 to 1e8), the penalty selected on pinned validation rows; this run consumes its 9,941 pinned held-out predictions. The held-out pool realized 9,941 of 10,000 pinned rows — 59 rows skipped by the parent's over-length capture filter (651 of 99,778 corpus-wide). Every held-out row carries banked judge labels (language, topic, format, refusal adjacency, answer-is-refusal; test-retest κ 0.79–0.98; 9,925 of 9,941 labeled — the 16 unlabeled rows are excluded from label masks). The resample control adds 4 extra on-policy answer draws for 1,988 held-out contexts (stratified over depth band × language × corpus). Raw conversation text never enters committed JSONs; text-bearing rows live on the HF data repo and in the two dashboards.

The round-1 mode-discovery leg ran on a degraded instrument: 7 of its 10 Fable digest calls — including all 5 failure-digest chunks — exhausted the 8,000-token output budget before emitting text, and the empty replies were cached as successes (the stop reason was never persisted, so the gate that should have fired had no field to read). The round-1 roster therefore traced to 3 sample-digest chunks and never saw a failure digest. Round 4 repaired the instrument (32,000-token cap with an explicit SDK timeout, 25-row chunks, stop reason persisted, blank replies hard errors, a production-shape pilot gate) and re-ran discovery over the same two digests (the worst-200 + stratified-300 failure digest and the 500-sample digest, 500 rows each). A new failure class surfaced: model content refusals on real-world-corpus rows — 8 of 40 digest chunks refused at chunk grain and fell back to per-row calls, and 13 of the 1,000 digest rows (9 failure-digest, 4 sample-digest) stayed refused and were excluded from mode-proposal influence, tallied in `refusal_exclusions.json`. The surviving 1,297 mode proposals refused consolidation as a single call and were consolidated hierarchically (13 batches of 100; one batch recovered by half-splitting; 0 proposals dropped) into the 10-mode round-4 roster. The round-1 roster, its label wave, and its rates are superseded (details block under the mode result); every mode number in this body is from the round-4 wave.

Acknowledged WARNs: the total-prose budget (800 words) is exceeded and several per-result blocks exceed the 120-word tier (thirteen deliverables are reported); one or more Takeaways bullets may exceed the 30-word bullet tier; 9 of 18 embedded figures are driver-generated without sidecars (nine carry sidecars from regeneration or the fold rounds; the rest are acknowledged as-is); the pool-robustness figure's direct line labels are the banked label-field identifiers defined above, acknowledged as rendered.

**Sample training/evaluation data + completions:** Disclosure: 3 of 1,829 rank-1 failure rows, random sample (seed 42); conversation text is excerpted to ~12 words for context hygiene (real-world-corpus rows) — full text at the pinned [dashboard rows on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ab268958343380945354e871bfb5666668c6d5bb/issue2202_ctxfail) and in the failure dashboard (footer).

<details>
<summary>3 sampled rank-1 failures (sanitized excerpts)</summary>

| row | final user turn (excerpt) | model answer (excerpt) | rank | top confuser (excerpt) | context / answer rank of confuser |
|---|---|---|---|---|---|
| ci 67690 | "Write me Easy Diffusion prompts for …" [truncated — real-world-corpus row] | "Certainly! Here are some easy diffusion prompts for a …" [truncated] | 10 | fan-fiction battle aftermath, ci 72042: "In the aftermath of a fierce battle, the characters from both Warcraft …" [truncated] | 33 / 77 |
| ci 11905 | "Complete this: I want to learn more about the American culture by" | "I want to learn more about the American culture by immersing myself …" [truncated] | 7 | transliteration explainer, ci 71880: "Got it! Lozung is a transliteration of the word Slogan …" [truncated] | 1,575 / 22 |
| ci 2968 | "puedes darme un plan de viaje a El Calafate, de 6 dias …" [truncated] | "¡Claro! El Calafate es un destino maravilloso en la provincia de Santa …" [truncated] | 4 | Portuguese Swiss-Alps itinerary, ci 30290: "Claro! Vou criar um roteiro básico para uma viagem aos Alpes Suíços. …" [truncated] | 4 / 5 |

</details>

The Spanish travel-plan failure is the archetypal hub drag: its top confuser is a Portuguese travel itinerary whose context ranks 4th-nearest and whose answer ranks 5th-nearest (answer-to-answer cosine 0.887) — a same-genre neighbour, not a duplicate. Disclosure: 3 of 8,112 correctly-retrieved rows, random sample (seed 42), same excerpt policy; per-row metrics for every context: [percontext_ranks.csv](https://github.com/superkaiba/explore-persona-space/blob/cf60dfc33519962eeb4f15290c74c926d5c1560c/eval_results/issue_2202/percontext_ranks.csv).

<details>
<summary>3 sampled correct retrievals (sanitized excerpts)</summary>

| row | final user turn (excerpt) | model answer (excerpt) | rank | nearest-competitor answer similarity |
|---|---|---|---|---|
| ci 75368 | "What is the estimate in literature?" | "The emission factors for gasification of biomass waste are not as widely …" [truncated] | 1 | 0.903 |
| ci 28052 | "The chatbot should analyze the description and identify the primary topic or …" [truncated] | "The primary topic or subject matter of this bill is Social Issues, …" [truncated] | 1 | 0.856 |
| ci 25159 | "Faça poema Vitor dedicando para Nicole recém casados" | "Órfãos de corações, agora só um. Vitor e Nicole, já a morna …" [truncated] | 1 | 0.913 |

</details>

Disclosure: 1 of 4,140 judge-labeled items, cherry-picked for illustration (the ci 2968 failure above; not a random sample); complete round-4 labels: [labels.json](https://github.com/superkaiba/explore-persona-space/blob/40ebc800b7af106670c6be065ca179a4f0433f72/eval_results/issue_2202/judge_labels_2202/labels.json).

```
judge = claude-sonnet-4-5-20250929, multi-field yes/no rubric (one field per round-4 mode)
item f2968 -> distinctive_self_contained_final_query: yes; long_topic_echoing_answer: yes;
contentless_final_turn: no; generic_refusal_or_safety_answer: no;
clarification_or_meta_answer: no; ultra_short_label_answer: no;
templated_scaffold_answer: no; answer_language_switch_or_garbled: no;
abrupt_topic_shift_final_turn: no; anonymized_placeholder_entities: no
```

## Results

### 18% of held-out contexts miss at rank 1, and identity failure only partly tracks error magnitude

Survival curve of the true answer's mid-rank among all 9,941 held-out answers (raw euclidean, log-log; one step per context), then the per-context rank against the parent's banked normalized prediction error.

![Survival curve of true-answer mid-rank, all held-out contexts](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2a82ee389490c4494dbc8125b9f21c52b36a312/figures/issue_2202/fig_rank_survival.png)

> **Figure.** *The rank tail is heavy.* Fraction of 9,941 held-out contexts whose true answer ranks worse than x under the map's prediction: 18.4% miss at rank 1, 6.7% at rank 10, tail reaching the pool edge. On-policy corpus answers; banked map predictions.

![Per-context true-answer rank versus banked normalized error](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2a82ee389490c4494dbc8125b9f21c52b36a312/figures/issue_2202/fig_concordance_scatter.png)

> **Figure.** *Per-unit companion.* Each point is one held-out context: true-answer mid-rank (log x) against the parent's normalized prediction error (n = 9,941). Spearman ρ = 0.450.

1,829 contexts (18.4%) miss at rank 1 and 6.7% still miss at rank 10. Identity failure only partly tracks error magnitude (ρ = 0.450, n = 9,941): 90 of the 200 worst-rank contexts sit among the 200 largest-error contexts — a prediction can land close to its target yet lose rank 1 in a crowded answer region.

### Failures concentrate in refusal-like and technical contexts — and in English, refuting the non-English prior

Failure-rate delta (group minus rest) for the 22 banked-label contrasts, with 95% bootstrap bands from 10,000 draws; filled markers cleared false-discovery correction at q = 0.05 (13 of 22).

![Forest of failure-rate deltas for 22 banked label contrasts](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2a82ee389490c4494dbc8125b9f21c52b36a312/figures/issue_2202/fig_composition_forest.png)

> **Figure.** *Refusal-family, NSFW, and code contexts fail most.* Failure-rate delta per contrast with 95% bootstrap band; n = 9,941 contexts, group sizes 82–5,870. Axis identifiers are the banked label fields defined in Methodology. Per-context data is binary; the row-level view lives in the failure dashboard.

Refusal answers fail at 42.6% against 17.9% for the rest (+24.8 pp, 204 contexts); NSFW topics +21.8 pp (209), refusal-adjacent requests +16.4 pp (519), harmful-or-unsafe requests +13.0 pp (267), code-format answers +11.0 pp (755).

English contexts fail *more* (+8.2 pp, 5,870 contexts), so the expectation that non-English contexts drive failures was wrong; factual-QA (−5.8 pp) and advice (−6.4 pp) contexts fail less. The English contrast is marginal (not partialed), but it survives excluding the refusal, NSFW, and code families: English 18.8% (834 of 4,446) vs non-English 12.6% (412 of 3,275) outside those families. Median band width 5.5 pp is the detection floor.

### Six modes are enriched among failures under the repaired discovery pass; distinctive self-contained exchanges protect

Judge-scored mode rates among 1,816 equalized failures versus 1,811 matched controls under the round-4 ten-mode roster (failure bars carry 95% intervals; judge-rubric identifiers on the axis; the retest-demoted mode is rates-only), then the per-context yes/no labels behind the rates.

![Mode rates for failures versus matched controls](https://raw.githubusercontent.com/superkaiba/explore-persona-space/40ebc800b7af106670c6be065ca179a4f0433f72/figures/issue_2202/fig_mode_rates.png)

> **Figure.** *Summary rates.* Judge-scored mode rates, failures (blue) vs matched control (orange), 1,816 vs 1,811 labeled items, round-4 roster; the template-scaffold bar is the one retest-demoted mode (κ 0.33), rates only — it trends protective (−2.4 pp) but carries no claim. Per-unit companion below.

![Per-context yes and no mode labels, jittered](https://raw.githubusercontent.com/superkaiba/explore-persona-space/40ebc800b7af106670c6be065ca179a4f0433f72/figures/issue_2202/fig_mode_percontext.png)

> **Figure.** *Per-unit companion to the rates above.* Every labeled context appears once per mode (yes at top, no at bottom), failures (blue) vs controls (orange).

Eight of nine reliable modes differ (second correction family). Six are enriched among failures: contentless final turns — terse fragments like "continue" — 37.9% vs 29.8% (+8.1 pp); language-switched or garbled answers 12.3% vs 5.5% (+6.9 pp); stock refusal answers, clarification-or-meta answers, anonymization placeholders (NAME_1-style tokens), and ultra-short label answers +2.3 to +3.3 pp each. Two protect: final queries that are distinctive and self-contained (−8.7 pp) and long answers echoing the query's topic vocabulary (−7.8 pp); abrupt final-turn topic shifts are null.

Rates on the 1,329 failures never shown to Fable match the full set within 2.4 pp. Round 1's enriched and protective modes recur under new names; the failure digest added the refusal, clarification, placeholder, and short-label modes discovery had missed.

<details>
<summary>Superseded by round 4: the round-1 mode wave (degraded discovery instrument)</summary>

Round 1's discovery pass silently lost 7 of 10 Fable digest calls to empty replies (all 5 failure-digest chunks), so its 9-mode roster traced to 3 sample-digest chunks. Its label wave (4,137 valid labels) found: short deictic follow-ups 28.1% vs 20.4% (+7.7 pp), corrupted or language-switched answers 11.2% vs 4.4% (+6.8 pp), distinctive-entity anchoring 50.6% vs 57.4% (−6.7 pp, protective); 3 of 9 modes retest-demoted (κ 0.32–0.57). Round-1 figures remain at the round-1 pin: [mode rates](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2a82ee389490c4494dbc8125b9f21c52b36a312/figures/issue_2202/fig_mode_rates.png), [per-context labels](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2a82ee389490c4494dbc8125b9f21c52b36a312/figures/issue_2202/fig_mode_percontext.png).

</details>

### Four of five attributable failures are the map's fault: the degeneracy hypothesis is rejected

The 1,829 failures split by the resample control (stacked counts; dashed line marks the fresh-draw retrievability reference, 0.943, scaled to the bar), then each context's true-answer-to-nearest-competitor similarity, failures versus matched controls, per similarity convention.

![Stacked attribution of failures with fresh-draw reference](https://raw.githubusercontent.com/superkaiba/explore-persona-space/eea63f6282cb0f620c80e87322014c59676f5b08/figures/issue_2202/fig_attribution_v2.png)

> **Figure.** *Map-attributable dominates the covered set.* Of 368 resample-covered failures: 297 map-attributable, 50 irreducible, 21 ambiguous; the remaining 1,461 failures are uncovered (unknown class). Coverage is the parent's 1,988-context resample subset. The dashed line is the fresh-draw retrievability reference, scaled to the bar.

![Nearest-competitor answer similarity for failures versus controls](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2a82ee389490c4494dbc8125b9f21c52b36a312/figures/issue_2202/fig_sconf_density.png)

> **Figure.** *Degeneracy companion.* Similarity between the true answer and the map's top competitor, failures (blue) vs matched controls (orange), one sub-plot per similarity convention. Failures' competitors are slightly less similar to the truth (raw-space median 0.878 vs 0.914), and 0 failures are exact ties — failures are not near-duplicate answers.

297 of 368 covered failures (80.7%) are map-attributable and 50 (13.6%) irreducible — far past the 60% threshold that falsifies "irreducible at least matches map-attributable". Equivalently, 297 of all 1,829 failures (16.2%) are proven map error, while 1,461 (79.9%) remain unknown.

The 0.943 reference is the retrievability of one fresh answer draw, not a ceiling for an ideal conditional-mean map, so 13.6% is an upper bound on irreducibility. Covered and uncovered contexts fail at matched rates (18.5% vs 18.4%), but covered failures under-sample refusal-type rows (refusal answers 3.0% vs 5.2%) — the most degenerate family — so the full-set map-attributable share is plausibly somewhat below 80.7%.

### Mutual confusion is rare but exceeds a collision-free degree null, staying below every distance null

Observed reciprocity of the top-1 confusion graph — an edge i→j wherever answer j outranks context i's true answer under its prediction, about 180 edges per failure; reciprocity is the probability an edge has its reverse — against the degree-preserving and distance-only null bands (2.5th–97.5th percentile; log axis), then forward versus reverse rank for all 329,448 edges.

![Observed reciprocity against degree and distance null bands, log axis](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2a82ee389490c4494dbc8125b9f21c52b36a312/figures/issue_2202/fig_reciprocity_bands_log.png)

> **Figure.** *Above the collision-free degree band, below the distance bands.* Observed reciprocity 8.4e-4 (orange) against the superseded stub null (6.0e-4 to 8.7e-4, duplicate edges kept), the collision-free swap null (2.5e-4 to 3.9e-4, 200 draws), and three distance-only bands (2.7e-3 to 3.2e-3, 1,000 draws each); all far below the ceiling of 1.

![Forward versus reverse rank for every confusion edge](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2a82ee389490c4494dbc8125b9f21c52b36a312/figures/issue_2202/fig_reverse_rank_scatter.png)

> **Figure.** *Per-edge companion.* For each of the 329,448 confusion edges: the confuser's rank under the source's prediction (x) against the source's rank under the confuser's prediction (y), log-log. Spearman ρ = 0.645; most mass sits far from the mutual-confusion corner.

276 of 329,448 edges are reciprocated (8.4e-4) — mutual confusion is rare. Round 1 called this degree-explained from the stub-permutation band (6.0e-4 to 8.7e-4); that null keeps colliding duplicate edges (22.5% per draw), inflating its band under the unique-edge count — superseded, verdict retracted.

The collision-free rebuild bands at 2.5e-4 to 3.9e-4 — the observed value is 2.6× its median: genuine mutual confusion beyond hub structure, consistent with the failures the hub correction below cannot recover. Kept: observed stays *below* every distance-only band — far more one-way than a symmetric-metric graph predicts — the planned metric-explained cell. Per-edge ranks still co-vary (ρ = 0.645).

### A few hub answers absorb the confusions

The distribution of each pool answer's top-10 in-degree — how many of the 9,941 predictions list it among their 10 nearest answers — for the retrieval and prediction-collapse graphs (log counts), then the 20 most-captured pool answers.

![In-degree distributions for retrieval and collapse graphs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2a82ee389490c4494dbc8125b9f21c52b36a312/figures/issue_2202/fig_indegree_v2.png)

> **Figure.** *Heavy-tailed in-degree.* Number of pool answers at each top-10 in-degree; retrieval skewness 3.8, collapse skewness 2.0. Mean in-degree is 10 by construction; the tail runs to 182.

![Capture counts of the 20 most-captured pool answers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2a82ee389490c4494dbc8125b9f21c52b36a312/figures/issue_2202/fig_hub_capture.png)

> **Figure.** *Per-unit companion: the top hubs.* The 20 most-captured pool answers (labeled by row id): the top hub appears in 182 of 9,941 top-10 lists, 18× the construction mean; 3.2% of answers appear in none.

Hub answers — not diffuse noise — absorb the failed predictions, the classical high-dimensional hubness signature for nearest-neighbour retrieval, though hub structure alone no longer accounts for the observed reciprocity (previous result). A hub-correcting retrieval rule — tested in the next result — recovers a majority of these failures.

### A retrieval-time hub correction closes 74% of the rank-1 gap to the fresh-draw reference

Rank-1 accuracy and mean reciprocal rank for the two base similarity conventions and for cross-domain similarity local scaling (CSLS, K = 10) against the fresh-draw reference, then each base failure's true-answer rank under euclidean versus under CSLS.

![Retrieval accuracy under CSLS hub correction and per-failure rank change](https://raw.githubusercontent.com/superkaiba/explore-persona-space/1ea77718633e6f74c663c330761fce35f58e67fa/figures/issue_2202/fig_csls_gap.png)

> **Figure.** *Hub correction recovers most failures.* Rank-1 accuracy 0.816 (euclidean) / 0.828 (cosine) rises to 0.909 under CSLS against the 0.9425 fresh-draw reference; of the 1,829 base failures, 969 recover to rank 1, 860 keep failing, 40 new failures appear (900 total). n = 9,941 held-out contexts.

CSLS re-scores each candidate by subtracting its mean similarity to its K nearest predictions — a retrieval-time correction only; the map's predictions are unchanged. It closes 73.9% of the euclidean rank-1 gap to the fresh-draw reference (71.1% for the cosine leg), and hub concentration collapses: the round-1 top hub's top-10 capture falls 182 → 50, in-degree skewness 3.75 → 1.66.

This localizes the failure mechanism — about three quarters of the map's rank-1 shortfall is hub geometry a symmetric correction removes, not target-specific map error — but it does not fix the map: 0.033 rank-1 accuracy (roughly a quarter of the gap) remains, consistent with the excess mutual confusion in the reciprocity read. Per-context accuracy outcomes are binary; the rank-change half of the figure is the per-unit companion.

### The failure set is pool-size- and similarity-convention-dependent

Each banked contrast's failure-rate delta recomputed at pool sizes 500, 2,000, and 9,941 (seed-pinned subsamples that always contain the true target), one line per contrast; the 13 significant contrasts labeled directly at their line ends.

![Contrast deltas across three pool sizes, significant contrasts labeled](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2a82ee389490c4494dbc8125b9f21c52b36a312/figures/issue_2202/fig_pool_robustness_v2.png)

> **Figure.** *Directions stable, magnitudes grow.* Failure-rate delta per banked contrast at pool sizes 500 / 2,000 / 9,941; overall failure rates 6.8% / 10.9% / 18.4%. The 13 contrasts that cleared false-discovery correction are colored and labeled at their line ends (axis identifiers are the banked label fields defined in Methodology); the 9 others render grey.

Rank-1 failure sets at pools 500 and 2,000 overlap the full-pool set at intersection-over-union 0.37 and 0.59, so composition claims — not row membership — are the stable output.

Similarity convention matters too: 1,708 failures under raw cosine (0.74 overlap with euclidean), 1,169 under mean-centered cosine, 462 under whitened cosine; whitened *euclidean* degenerates (9,739 of 9,941 fail — low-variance covariance directions dominate that distance) and is reported as a broken convention, not evidence.

### Under matched conventions the fresh-draw reference exceeds the map, and draw-averaged targets close most of the rest

Covered-row rank-1 accuracy (1,988 resample-covered contexts, full 9,941-answer pool) for single-draw versus 5-draw-averaged answer targets, under raw euclidean and whitened cosine, each with its convention-matched fresh-draw reference.

![Single-draw versus draw-averaged target accuracy with matched references](https://raw.githubusercontent.com/superkaiba/explore-persona-space/510a3802afea73dddeee0c859560b90d1f545acb/figures/issue_2202/fig_avg_target.png)

> **Figure.** *The map predicts the noise-averaged answer.* Averaging the 5 on-policy draws (original + 4 fresh) into each covered target lifts rank-1 accuracy 0.815 → 0.909 raw euclidean and 0.962 → 0.987 whitened cosine; dashed lines are the matched fresh-draw references (0.943, 0.979). Per-row outcomes are binary.

The raw fresh-draw leg reproduces the banked 0.943 exactly; under whitened cosine the reference is 0.979, so with conventions matched the reference exceeds the map everywhere — the apparent map-above-reference inversion was a cross-convention artifact, and the covered-row gap shrinks from 12.8 pp to 1.8 pp. Averaging 5 draws into the target removes about half of the remaining raw failure mass and two thirds of the whitened mass: the prediction lies closer to the mean of the answer distribution than to any single draw. The banked nonlinear maps gain 0.05–0.06 raw-space accuracy over ridge yet sit at or below it under whitened cosine (0.942–0.951 versus ridge 0.954) — a scale correction the whitening already supplies.

### Hub-penalized whitened cosine meets the convention-matched fresh-draw reference

Full-pool rank-1 accuracy (9,941 contexts) for all 28 similarity conventions, 6 banked and 22 new, sorted; vertical lines mark the raw-euclidean fresh-draw reference and the reference matched to the top convention.

![Rank-1 accuracy by similarity convention, sorted bars](https://raw.githubusercontent.com/superkaiba/explore-persona-space/510a3802afea73dddeee0c859560b90d1f545acb/figures/issue_2202/fig_convention_zoo.png)

> **Figure.** *Hub penalties on the whitened-cosine base lead.* CSLS (K = 10) on whitened cosine reaches 0.976 against its matched fresh-draw reference of 0.973; a double-strength candidate penalty reaches 0.985. Dashed line: raw-euclidean reference (0.943); dotted: matched reference (0.973). Per-context outcomes are binary; the sidecar carries every bar value.

CSLS on the whitened-cosine base — a rescoring that penalizes candidates similar to everything — retrieves 0.976 at rank 1, at or above its convention-matched fresh-draw reference; the invented pool-in-degree penalty reaches 0.957. The truncated-whitening sweep explains the earlier whitened-euclidean degeneration: whitening more covariance directions degrades the euclidean read monotonically (0.505 at 64 directions down to 0.020 at all 3,584) while improving the cosine read — the win from whitening is entirely conditional on per-vector normalization, since each whitened direction adds unit variance to the residual norm. Classic hubness rescalings are weak alone (mutual proximity +0.001, NICDM +0.052, DisSimLocal +0.057 over raw euclidean) yet compose with whitening, and the inverted softmax is fragile to its inverse temperature (0.736 at 10 versus 0.849 at 30).

### Discrimination-trained maps top out at the whitened-cosine ridge level — the bottleneck was the metric, not the map

Full-pool rank-1 accuracy (9,941 contexts) for the banked ridge and MSE-trained MLP maps and two discrimination-trained (InfoNCE) maps — linear, warm-started at ridge, and a width-8,192 MLP — under three similarity conventions; horizontal lines mark the CSLS corrections of the unchanged ridge map.

![Rank-1 accuracy for four maps under three conventions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/391a08f75150b32b2b3bd437617d42a00ca45220/figures/issue_2202/fig_contrastive_maps.png)

> **Figure.** *Optimizing the map for discrimination is redundant with fixing the metric.* The best contrastive read (whitened-cosine MLP, 0.956) sits 0.25 pp above ridge (0.954) and far below CSLS on the unchanged ridge map (0.976; double-strength penalty 0.985). The contrastive MLP's raw-euclidean 0.599 is a norm artifact of its cosine objective. Per-context outcomes are binary.

Both InfoNCE fits selected temperature 0.05 on the pinned validation rows and early-stopped; the reconciliation legs reproduce the banked ridge accuracy and both banked R² values exactly. The contrastive linear map recovers most of the raw-space hub-correction gain (raw euclidean 0.816 → 0.870) — the objective learns what the metric corrections compute closed-form — but lands below ridge under whitened cosine (0.942), and the contrastive MLP's 0.956 buys 0.25 pp over ridge where the retrieval-time corrections buy 2.3–3.1 pp for free. Held-out R² collapses to −4.58 (linear) and −1.96 (MLP) against 0.68 for ridge: the contrastive outputs are retrieval keys, not activation predictions, so discrimination training trades faithfulness away without beating the metric fix.

### With the metric hub-corrected and the target noise-averaged, all seven maps retrieve alike

Covered-row rank-1 accuracy (1,988 rows) under CSLS K = 10 on whitened cosine for all seven maps, single-draw versus 5-draw-averaged targets — the headline convention of the completed coverage matrix.

![Seven maps under hub-corrected retrieval, single versus averaged targets](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b45bb8c86b0b91cdd1342084034fe1152cc9d83b/figures/issue_2202/fig_avgtgt_convergence.png)

> **Figure.** *Architecture differences nearly vanish.* Under hub-corrected retrieval, 5-draw-averaged targets lift every map to 0.991–0.995 at rank 1 from a 0.974–0.981 single-draw spread; the best matrix cell overall is ridge with the double-strength penalty at 0.995. Per-row outcomes are binary.

Every map gains under draw-averaged targets in every convention scored (+0.008 to +0.095), and the completed matrix's best cell is ridge under the double-strength penalty at 0.995 against 0.987 single-draw. Under the headline hub-corrected convention the seven maps — ridge, four nonlinear, two contrastive — converge to 0.991–0.995, so architecture choice contributes at most about 0.4 pp once the metric is hub-corrected and the target is noise-averaged; the contrastive MLP's raw-euclidean outlier persists (0.61 → 0.66), the cosine-geometry artifact. The contrastive predictions were recomputed from the banked fit weights and match the banked battery with delta exactly 0.

### The residual failures are near-miss map error on underdetermined turns, not answer twins

Distribution of per-row retrieval margins — the true target's score minus the best competitor's, under CSLS K = 10 on whitened cosine with draw-averaged targets — over the 1,988 covered rows (log-count axis); the shaded region below zero is the 11 residual failures at this operating point.

![Margin distribution over covered rows with the failure region shaded](https://raw.githubusercontent.com/superkaiba/explore-persona-space/92b98e9869cdb3303cdbb77bc237d94b5b62bfaf/figures/issue_2202/fig_residual_margins.png)

> **Figure.** *Failures are the near tail of one margin distribution, not a separate cluster.* The 11 failures lose by a median 0.025 while the 1,977 successes win by a median 0.175 — losses ~7× smaller than wins; pairwise AUC is ≥ 0.9999 here (≥ 0.996 in all four conventions), and margins widen under draw-averaging. Per-row outcomes are binary.

None of the 15 residual failures (union over the two hub-corrected conventions) is an answer twin: retrieved-competitor whitened cosine to the true answer peaks at 0.36 against the 0.95 twin bar, and the resample control attributes 13 of 15 to the map. The failing contexts are terse, deictic final turns underdetermining the answer — a bare "why?", "is that all? thank you".

The worst-discriminated bottom-50 tail (11 failures plus 39 barely-won successes) is a different population from the raw-euclidean failure profile: over-representing coding (30% vs 16.5% pool), Chinese (28% vs 12% pool, 9% of raw-euclidean failures), and two-turn exchanges (56% vs 42%), under-representing Russian (2% vs 9%), with refusal-answer and refusal-adjacent shares at pool level — the raw-metric refusal signature is absent.

---
**Repro:** compute: one RunPod CPU pod (`pod-2202`, `cpu-bigmem`, geometry + null phases, ~1.0 h realized 19:38–20:41 UTC including two crash-fix relaunches; ~2.5 h plan-booked) + VM phases (Fable synthesis, one Sonnet Batch-API wave of 4,495 judge calls in each of rounds 1 and 4, statistics, dashboards); 0 GPU-h through round 6; round 7: one GPU pod (`pod-2202-contrastive`, 1× H100, ~1.2 h realized including two smoke attempts and a fresh 224-chunk re-assemble after a MooseFS-unsafe resume truncate in the shared stream assembler; the fits themselves took 10 s linear + 22 s MLP), terminated after upload verification · code: pod-side geometry at [scripts/issue2202_failchar.py](https://github.com/superkaiba/explore-persona-space/blob/fa576c59377d09a8ad60daef305b124e022316ad/scripts/issue2202_failchar.py); VM phases at [scripts/issue2202_labels.py](https://github.com/superkaiba/explore-persona-space/blob/40ebc800b7af106670c6be065ca179a4f0433f72/scripts/issue2202_labels.py) (incl. the round-4 Fable repair: commits badf7bb13f, 68f4f202ea, 9ffaea0914), [scripts/issue2202_stats_figs.py](https://github.com/superkaiba/explore-persona-space/blob/cf60dfc33519962eeb4f15290c74c926d5c1560c/scripts/issue2202_stats_figs.py), [scripts/issue2202_dashboards.py](https://github.com/superkaiba/explore-persona-space/blob/cf60dfc33519962eeb4f15290c74c926d5c1560c/scripts/issue2202_dashboards.py); figure regeneration at [scripts/issue2202_regen_figs.py](https://github.com/superkaiba/explore-persona-space/blob/d2a82ee389490c4494dbc8125b9f21c52b36a312/scripts/issue2202_regen_figs.py), collision-free reciprocity null at [scripts/issue2202_r2_sensitivity.py](https://github.com/superkaiba/explore-persona-space/blob/ec4039a3ffd5a3fff00df78bc524435da53b657a/scripts/issue2202_r2_sensitivity.py), CSLS hub-correction follow-up at [scripts/issue2202_csls_followup.py](https://github.com/superkaiba/explore-persona-space/blob/1ea77718633e6f74c663c330761fce35f58e67fa/scripts/issue2202_csls_followup.py), matched-convention + draw-averaged read at [scripts/issue2202_freshwhiten_avg.py](https://github.com/superkaiba/explore-persona-space/blob/f2e08f4a01e2de790c30f0edb700c6e1f07d9114/scripts/issue2202_freshwhiten_avg.py), similarity-convention battery at [scripts/issue2202_metric_zoo.py](https://github.com/superkaiba/explore-persona-space/blob/6d8edcffd96f356b9dbb6984bd17666ff52bea7f/scripts/issue2202_metric_zoo.py), discrimination-trained map comparison at [scripts/issue2202_contrastive_maps.py](https://github.com/superkaiba/explore-persona-space/blob/591ba8d32d93d06a6d9085b7b5613fa1c9db2709/scripts/issue2202_contrastive_maps.py), draw-averaged coverage matrix at [scripts/issue2202_avgtgt_completion.py](https://github.com/superkaiba/explore-persona-space/blob/2c4bf031679c1ffe7ab8ff0e68a0923bbd94d6f6/scripts/issue2202_avgtgt_completion.py), residual-failure + margin read at [scripts/issue2202_residual_read.py](https://github.com/superkaiba/explore-persona-space/blob/ed0165405a127cee38b640b6c389af3558fe97ad/scripts/issue2202_residual_read.py), fold-round figures at [scripts/issue2202_fold_figs.py](https://github.com/superkaiba/explore-persona-space/blob/830fb52d5d5d9d7fb5bfafa04c8d649b54cc051e/scripts/issue2202_fold_figs.py) · eval JSONs: [eval_results/issue_2202](https://github.com/superkaiba/explore-persona-space/tree/830fb52d5d5d9d7fb5bfafa04c8d649b54cc051e/eval_results/issue_2202) (`repro_gate.json`, `identity_bias_gate.json`, `percontext_ranks.csv`, `failures_confusion.json`, `attribution.json`, `reciprocity.json`, `reciprocity_collision_free.json`, `hubness.json`, `pool_robustness.json`, `concordance.json`, `geometry_summary.json`, `composition_stats.json`, `sample500_lists.json`, `csls_followup.json`, `csls_percontext_ranks.npz`, `judge_labels_2202/labels.json`, `judge_labels_2202/pilot_gate_report.json`, `fable_reads/modes.json`, `fable_reads/refusal_exclusions.json`, `fable_reads/consolidation.json`, `freshwhiten_avg/summary.json`, `metric_zoo/summary.json`, `metric_zoo/results.jsonl`, `metric_zoo/research_notes.md`, `contrastive_maps/eval/contrastive_maps_battery.json`, plus the six per-fit JSONs under `contrastive_maps/fits/`, `avgtgt_completion/summary.json`, `residual_read/summary.json`, `residual_read/percontext_ranks_margins.npz`) · figures: [figures/issue_2202](https://github.com/superkaiba/explore-persona-space/tree/830fb52d5d5d9d7fb5bfafa04c8d649b54cc051e/figures/issue_2202); the driver renders `fig_indegree.png` (empty axes) and `fig_reciprocity_bands.png` (linear axis hides the bands) are superseded by `fig_indegree_v2.png` and `fig_reciprocity_bands_log.png`, and `fig_pool_robustness.png` (legend recycles colors across 22 lines, misattributing runner-up lines) by `fig_pool_robustness_v2.png`, and `fig_attribution.png` (legend mislabeled the fresh-draw reference as an accuracy ceiling) by `fig_attribution_v2.png`; the round-1 renders of `fig_mode_rates.png` and `fig_mode_percontext.png` (pinned at d2a82ee389) are superseded by the round-4 re-renders pinned in-body at 40ebc800b7; committed `composition_stats.json` retains the retracted round-1 `reciprocity_verdict` string (stub-null reading), superseded in-body by the collision-free rebuild · HF outputs: [issue2202_ctxfail](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ab268958343380945354e871bfb5666668c6d5bb/issue2202_ctxfail) (`analysis_tensors/`, `rows_geom/`, `dashboard_rows/`, `digests/`, `judge/`, `eval_mirror/`; verified live); round-4 Sonnet dispatch mirrors at [issue2202_ctxfail/judge_labels_2202](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/27022ed657ac097861af587850afd85e88510957/issue2202_ctxfail/judge_labels_2202) (`dispatch_main/`, `dispatch_pilot/`, `dispatch_smoke5/`; verified live); contrastive-round outputs at [issue2202_ctxfail/contrastive_maps](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/b6d2ca007842fd6abe7ff9f6d7a71a1de3708769/issue2202_ctxfail/contrastive_maps) (13 files — the battery JSON, six per-fit JSONs, six fit weight files; verified live) · reused inputs from [#1738](https://eps.superkaiba.com/tasks/1738) at [issue1738_multiturn @ 09788eef](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/09788eef2f85330c6f9c6b7cd3d28cb47cfb8429/issue1738_multiturn): held-out predictions `analysis_tensors/pred16/context_L19_ridge.npz`, answers `analysis_tensors/y_holdout/L19.npz`, capture store, split doc, resample shards, plus git-banked labels and error CSV — fit: the exact map, split, labels, and controls this run interrogates (reproduction gate PASS, every delta at or below the two-tie-row tolerance) · dashboards: [failures-2202.html](https://github.com/superkaiba/explore-persona-space/blob/cf60dfc33519962eeb4f15290c74c926d5c1560c/dashboard/public/failures-2202.html), [sample500-2202.html](https://github.com/superkaiba/explore-persona-space/blob/cf60dfc33519962eeb4f15290c74c926d5c1560c/dashboard/public/sample500-2202.html); served at `https://eps.superkaiba.com/failures-2202.html` and `/sample500-2202.html` once this branch merges to main · WandB: n/a — no training.

**Context:**
> Motivation: We did an analysis of the directions the model fails on using SAE features; SAE features are known to be somewhat unreliable; we want to do a more controlled analysis of this question. Methodology: apply our best mapping on the generic corpus; see for which ones it fails to distinguish the correct answer vector from some other answer vector; look at the contexts it fails on and characterize what kinds of things it fails on. (Full verbatim request + resolved clarifications in the body's ## Provenance section.) [then] run it in background with happy coder

Lineage: [#1738](https://eps.superkaiba.com/tasks/1738) — the 100k multi-turn context→answer map whose held-out failures this run characterizes. Created 2026-08-08; pod phases run 2026-08-08 20:06–20:08 UTC; VM phases through 2026-08-08 21:20 UTC; round-2 collision-free null + round-3 zero-GPU CSLS follow-up (proposer-initiated free-analysis band) run 2026-08-08 22:15–22:44 UTC; round-4 mode-instrument repair + fresh label wave (same-issue follow-up, `followup_label: fable-digest-rerun`, source user-chat; scope spec verbatim: "Repair and re-run the Fable-5 hypothesis-generation pass (`P3b`) whose digest calls returned empty, then re-drive the downstream chain so mode discovery finally sees the `FAILURE` digest.") run 2026-08-09 09:40–10:02 UTC; user-chat inline free-analysis rounds `freshwhiten-avg` (ask verbatim: `can we do fresh draw with whitened cosine? and also did we ever check how good our acc@1 is for averaged answers?`; addenda verbatim: `can we use R^2 as our similarity metric?`, `can we try the nonlinear?`) and `metric-zoo` (ask verbatim: `send a subagent to do a deep dive online for similarity metrics (and come up with some yourself) and run those as well`) run 2026-08-17, 0 GPU-h each; inline GPU-override round `contrastive-maps` (ask verbatim: `run this inline now + the MLP variant`, following `is it possible to optimize directly for discrimination of answers?`) run 2026-08-17 on one H100, ~1.2 GPU-h; inline free-analysis round `avgtgt-completion` (ask verbatim: `do we have all this on averaged targets?`) run 2026-08-17, 0 GPU-h; inline free-analysis round `residual-read` (ask verbatim: `yes look at the ones that fail. also is there some metric for "how well" it differentiates?`; addendum verbatim: `can you do that and then look at which contexts are the worse discriminated`) run 2026-08-17, 0 GPU-h.
