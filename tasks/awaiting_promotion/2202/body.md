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
**Methodology:** [docs/methodology/issue_2202.md](https://github.com/superkaiba/explore-persona-space/blob/541fcc48c40e3d7f23d4d0276f0d3c22bd565625/docs/methodology/issue_2202.md) · [gist mirror](https://gist.github.com/superkaiba/4183c6b2ebc0780cf7238af8bc730b5c)

<!-- Raw-output spot check (5 random rows, seed 42, percontext_ranks.csv joined with the local text cache; sanitized ~12-word excerpts):
ci 18784 rank 1 - "I mean do you have general knowledge of 2023?" -> knowledge-cutoff hedge; clean.
ci 4192 rank 1 - title-generation task -> "Arctic Naval Buildup: ..." title; clean.
ci 45071 rank 1 - "ls" -> simulated terminal listing; terse/roleplay row retrieved correctly; clean.
ci 40144 rank 1 - self-supervised-learning intro request -> long intro; clean.
ci 36811 rank 1 - "I like the second challenge. can you give me 2 more?" -> two challenges; clean. NOTE: carries kres_class=MAP_ATTRIBUTABLE despite rank 1; the CSV writes the raw resample partition for every covered row; the class is only meaningful for FAIL rows (attribution.json counts classes over failures only).
0 of 5 fishy (no judge/content disagreement, no corruption, no empty outputs). -->

## Takeaways

- 297 of 368 resample-covered rank-1 failures (80.7%) remain retrievable from a fresh on-policy answer draw — map error, not answer degeneracy; the 60% falsification threshold fired.
- Failure hot-spots: refusal answers +24.8 pp, NSFW topics +21.8 pp, refusal-adjacent requests +16.4 pp, code +11.0 pp, English +8.2 pp (refuting the non-English prior); 13 of 22 contrasts cleared q = 0.05.
- Confusion is one-way but not purely hub-driven: reciprocity 8.4e-4 over 329,448 edges is 2.6× the collision-free degree-preserving null median yet below every distance-only band; the top hub answer captures 182 top-10 lists.
- A retrieval-time symmetric hub correction (CSLS, K = 10) closes 73.9% of the rank-1 gap to the fresh-draw reference (0.816 → 0.909 vs 0.9425), recovering 969 of 1,829 failures with the map unchanged — direct quantification of hub drag; 0.033 remains.
- Judge-scored modes: short deictic follow-ups +7.7 pp and corrupted or language-switched answers +6.8 pp; distinctive-entity anchoring −6.7 pp (protective); 3 of 9 modes demoted at retest κ below 0.6.
- Coverage gaps: 7 of 10 Fable digest calls returned empty, so mode discovery saw only the 500-sample digest; attribution covers 368 of 1,829 failures; context arm only.

## Goal

- **This experiment in context:** Prior failure characterizations of the 100k multi-turn context→answer map used sparse-autoencoder features ([#1482](https://eps.superkaiba.com/tasks/1482), [#1946](https://eps.superkaiba.com/tasks/1946), [#2163](https://eps.superkaiba.com/tasks/2163)) — a lossy basis (fraction of variance explained 0.718 at layer 19). This run asks the same question with an SAE-free instrument: nearest-neighbour retrieval among the held-out answer vectors of the parent map ([#1738](https://eps.superkaiba.com/tasks/1738)), separating map error from target degeneracy and sampling noise via the parent's banked resample control. Context arm only: the prefix arm retrieves at rank-1 accuracy 0.183 under the same split and eval, making failure the default case there; its taxonomy is the named follow-up.
- **Broader narrative:** the mapping line asks how much of an answer's representation is a linear function of its context's. Where retrieval fails decides whether the residual is structured map error a better map could fix, or irreducible answer-sampling entropy — the distinction the leakage-prediction theory needs.

## Methodology

**Design:** one zero-GPU analysis pass over the banked context-arm layer-19 ridge map: full-pool retrieval ranks and confusion geometry in five similarity conventions, a two-stage failure-mode wave (Fable 5 proposes, Sonnet 4.5 counts), and a confusion-graph symmetry read against two nulls. Controls: the four-draw resample retrievability control, a matched non-failure control (cell-matched on depth band × corpus × language), the identity-plus-learned-bias baseline, degree-preserving (stub, plus a round-2 collision-free swap rebuild) and distance-only reciprocity nulls, and pool sizes 500 / 2,000 / 9,941. Three analysis rounds (geometry + judged waves; a collision-free null rebuild; a zero-GPU hub-correction follow-up); no new training, generation, or capture.

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
| Judge | `claude-sonnet-4-5-20250929`, Batch API, `max_tokens` 2048, temperature API default (1.0), 1 draw per item | project judge rule; parent instrument settings |
| Judge quality control | 150-item pilot (0 truncation stops, 0 parse failures, PASS); 200-item test-retest; modes with κ below 0.6 demoted to report-only | plan gate; parent κ convention |
| Reproduction-gate tolerances | rank-k accuracy within 2e-4; mean reciprocal rank within 1e-4; n exactly 9,941 | parent banked values |

Fable 5 (`claude-fable-5`) generated failure-mode hypotheses only; every countable label is a Sonnet call. Sonnet labeled 4,145 items — all 1,829 failures, 1,816 matched controls (cells capped at available non-failures), and the 500-sample — of which 4,137 returned valid labels (2 content drops, 6 other API errors; zero transport losses persisted).

**Data extraction:** all inputs are banked artifacts of the parent run, reproduced here for self-containment. The corpus is 100,000 multi-turn contexts drawn from LMSYS-Chat-1M and WildChat-1M — real user conversations (tier-1 realism). Each context was rendered in the Qwen-2.5-7B-Instruct chat template and answered once on-policy (sampled decoding, 7,104-token generation budget). Residual-stream states were captured at layer 19 in a teacher-forced forward pass: the context vector is the last prompt-token state (the newline before the assistant turn); the answer vector is the mean over answer-token states. A ridge map from context vector to answer vector (dimension 3,584) was fit on 88,378 training rows over 23 log-spaced penalties (1e-3 to 1e8), the penalty selected on pinned validation rows; this run consumes its 9,941 pinned held-out predictions. The held-out pool realized 9,941 of 10,000 pinned rows — 59 rows skipped by the parent's over-length capture filter (651 of 99,778 corpus-wide). Every held-out row carries banked judge labels (language, topic, format, refusal adjacency, answer-is-refusal; test-retest κ 0.79–0.98; 9,925 of 9,941 labeled — the 16 unlabeled rows are excluded from label masks). The resample control adds 4 extra on-policy answer draws for 1,988 held-out contexts (stratified over depth band × language × corpus). Raw conversation text never enters committed JSONs; text-bearing rows live on the HF data repo and in the two dashboards.

The mode-discovery leg partially failed: of the 10 Fable digest calls (5 chunks over the worst-200 + stratified-300 failure digest, 5 over the 500-sample digest), 7 returned empty replies — all 5 failure-digest chunks and 2 sample chunks — so all 13 raw mode proposals (consolidated to 9 canonical modes) trace to 3 sample-digest chunks. Mode discovery therefore never saw a failure-only digest; the Sonnet-measured rates over the full sets are unaffected. Retrying the empty chunks is flagged as a follow-up.

Acknowledged WARNs: the total-prose budget (800 words) is exceeded and several per-result blocks exceed the 120-word tier (eight deliverables are reported); one or more Takeaways bullets may exceed the 30-word bullet tier; 9 of 13 embedded figures are driver-generated without sidecars (four carry sidecars from regeneration; the rest are acknowledged as-is); the pool-robustness figure's direct line labels are the banked label-field identifiers defined above, acknowledged as rendered.

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

Disclosure: 1 of 4,137 judge-labeled items, cherry-picked for illustration (the ci 2968 failure above; not a random sample); complete labels: [labels.json](https://github.com/superkaiba/explore-persona-space/blob/cf60dfc33519962eeb4f15290c74c926d5c1560c/eval_results/issue_2202/judge_labels_2202/labels.json).

```
judge = claude-sonnet-4-5-20250929, multi-field yes/no rubric (one field per mode)
item f2968 -> distinctive_entity_anchoring: yes; non_english_or_marked_register: yes;
unique_artifact_in_response: no; interchangeable_boilerplate_response: no;
templated_genre_or_variant_request: no; terse_deictic_final_turn: no;
corrupted_or_code_switched_response: no; multiple_choice_option_echo: no;
answer_topic_drift_from_last_turn: no
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

### Two response-side modes are enriched among failures; distinctive entities protect

Judge-scored mode rates among 1,815 equalized failures versus 1,809 matched controls (bars with 95% intervals; judge-rubric identifiers on the axis; the three modes with retest κ below 0.6 are rates-only), then the per-context yes/no labels behind the rates.

![Mode rates for failures versus matched controls](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2a82ee389490c4494dbc8125b9f21c52b36a312/figures/issue_2202/fig_mode_rates.png)

> **Figure.** *Summary rates.* Judge-scored mode rates, failures (blue) vs matched control (orange), 1,815 vs 1,809 labeled items; the unique-artifact, templated-genre-request, and topic-drift bars are the three retest-demoted modes (κ 0.32–0.57), rates only. Per-unit companion below.

![Per-context yes and no mode labels, jittered](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2a82ee389490c4494dbc8125b9f21c52b36a312/figures/issue_2202/fig_mode_percontext.png)

> **Figure.** *Per-unit companion to the rates above.* Every labeled context appears once per mode (yes at top, no at bottom), failures (blue) vs controls (orange).

Three of six reliable modes differ (second correction family): short deictic follow-ups ("why?", "continue") 28.1% vs 20.4% (+7.7 pp); corrupted or language-switched answers 11.2% vs 4.4% (+6.8 pp, a 2.6× ratio); distinctive-entity anchoring 50.6% vs 57.4% (−6.7 pp) — a named entity shared by query and answer protects retrieval.

Rates on the 1,329 failures never shown to Fable match the full set within 1.8 pp. All discovered modes trace to the 500-sample digest (the failure-digest calls returned empty), so worst-tail-specific modes may be missing from the inventory.

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

---
**Repro:** compute: one RunPod CPU pod (`pod-2202`, `cpu-bigmem`, geometry + null phases, ~1.0 h realized 19:38–20:41 UTC including two crash-fix relaunches; ~2.5 h plan-booked) + VM phases (Fable synthesis, Sonnet Batch-API wave of 4,495 judge calls, statistics, dashboards); 0 GPU-h · code: pod-side geometry at [scripts/issue2202_failchar.py](https://github.com/superkaiba/explore-persona-space/blob/fa576c59377d09a8ad60daef305b124e022316ad/scripts/issue2202_failchar.py); VM phases at [scripts/issue2202_labels.py](https://github.com/superkaiba/explore-persona-space/blob/cf60dfc33519962eeb4f15290c74c926d5c1560c/scripts/issue2202_labels.py), [scripts/issue2202_stats_figs.py](https://github.com/superkaiba/explore-persona-space/blob/cf60dfc33519962eeb4f15290c74c926d5c1560c/scripts/issue2202_stats_figs.py), [scripts/issue2202_dashboards.py](https://github.com/superkaiba/explore-persona-space/blob/cf60dfc33519962eeb4f15290c74c926d5c1560c/scripts/issue2202_dashboards.py); figure regeneration at [scripts/issue2202_regen_figs.py](https://github.com/superkaiba/explore-persona-space/blob/d2a82ee389490c4494dbc8125b9f21c52b36a312/scripts/issue2202_regen_figs.py), collision-free reciprocity null at [scripts/issue2202_r2_sensitivity.py](https://github.com/superkaiba/explore-persona-space/blob/ec4039a3ffd5a3fff00df78bc524435da53b657a/scripts/issue2202_r2_sensitivity.py), CSLS hub-correction follow-up at [scripts/issue2202_csls_followup.py](https://github.com/superkaiba/explore-persona-space/blob/1ea77718633e6f74c663c330761fce35f58e67fa/scripts/issue2202_csls_followup.py) · eval JSONs: [eval_results/issue_2202](https://github.com/superkaiba/explore-persona-space/tree/ec4039a3ffd5a3fff00df78bc524435da53b657a/eval_results/issue_2202) (`repro_gate.json`, `identity_bias_gate.json`, `percontext_ranks.csv`, `failures_confusion.json`, `attribution.json`, `reciprocity.json`, `reciprocity_collision_free.json`, `hubness.json`, `pool_robustness.json`, `concordance.json`, `geometry_summary.json`, `composition_stats.json`, `sample500_lists.json`, `csls_followup.json`, `csls_percontext_ranks.npz`, `judge_labels_2202/labels.json`, `fable_reads/consolidation.json`) · figures: [figures/issue_2202](https://github.com/superkaiba/explore-persona-space/tree/ec4039a3ffd5a3fff00df78bc524435da53b657a/figures/issue_2202); the driver renders `fig_indegree.png` (empty axes) and `fig_reciprocity_bands.png` (linear axis hides the bands) are superseded by `fig_indegree_v2.png` and `fig_reciprocity_bands_log.png`, and `fig_pool_robustness.png` (legend recycles colors across 22 lines, misattributing runner-up lines) by `fig_pool_robustness_v2.png`, and `fig_attribution.png` (legend mislabeled the fresh-draw reference as an accuracy ceiling) by `fig_attribution_v2.png`; committed `composition_stats.json` retains the retracted round-1 `reciprocity_verdict` string (stub-null reading), superseded in-body by the collision-free rebuild · HF outputs: [issue2202_ctxfail](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ab268958343380945354e871bfb5666668c6d5bb/issue2202_ctxfail) (`analysis_tensors/`, `rows_geom/`, `dashboard_rows/`, `digests/`, `judge/`, `eval_mirror/`; verified live) · reused inputs from [#1738](https://eps.superkaiba.com/tasks/1738) at [issue1738_multiturn @ 09788eef](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/09788eef2f85330c6f9c6b7cd3d28cb47cfb8429/issue1738_multiturn): held-out predictions `analysis_tensors/pred16/context_L19_ridge.npz`, answers `analysis_tensors/y_holdout/L19.npz`, capture store, split doc, resample shards, plus git-banked labels and error CSV — fit: the exact map, split, labels, and controls this run interrogates (reproduction gate PASS, every delta at or below the two-tie-row tolerance) · dashboards: [failures-2202.html](https://github.com/superkaiba/explore-persona-space/blob/cf60dfc33519962eeb4f15290c74c926d5c1560c/dashboard/public/failures-2202.html), [sample500-2202.html](https://github.com/superkaiba/explore-persona-space/blob/cf60dfc33519962eeb4f15290c74c926d5c1560c/dashboard/public/sample500-2202.html); served at `https://eps.superkaiba.com/failures-2202.html` and `/sample500-2202.html` once this branch merges to main · WandB: n/a — no training.

**Context:**
> Motivation: We did an analysis of the directions the model fails on using SAE features; SAE features are known to be somewhat unreliable; we want to do a more controlled analysis of this question. Methodology: apply our best mapping on the generic corpus; see for which ones it fails to distinguish the correct answer vector from some other answer vector; look at the contexts it fails on and characterize what kinds of things it fails on. (Full verbatim request + resolved clarifications in the body's ## Provenance section.) [then] run it in background with happy coder

Lineage: [#1738](https://eps.superkaiba.com/tasks/1738) — the 100k multi-turn context→answer map whose held-out failures this run characterizes. Created 2026-08-08; pod phases run 2026-08-08 20:06–20:08 UTC; VM phases through 2026-08-08 21:20 UTC; round-2 collision-free null + round-3 zero-GPU CSLS follow-up (proposer-initiated free-analysis band) run 2026-08-08 22:15–22:44 UTC.
