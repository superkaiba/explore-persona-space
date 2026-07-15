---
title: Directional function-space map similarity predicts marker leakage beyond the
  activation-cosine and Jensen–Shannon baselines that render its symmetrized form
  redundant (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-07-15T07:06:20Z'
has_clean_result: true
parent_id: 823
origin_prompt: 'can you add an issue based on the next step? let''s plan it here [next
  step, verbatim from the finalized off-policy mapping result''s Conclusion: ''I think
  potentially leakage could be predicted by a similarity metric between the mappings
  for these different contexts — although this seems similar to KL divergence and
  that didn''t work too well — testing this now'']'
workflow: v1
goal: Test whether function-space similarity between per-context-family fitted linear
  context→answer maps (cross-family transfer R² / prediction agreement on the frozen
  base model, both prefix-based and context-based arms) predicts fine-tuning leakage
  from source persona to target contexts on an existing measured leakage matrix, with
  incremental validity over the activation-cosine, JS-divergence, base-rate-prior,
  and whitened-gate baselines under group-level held-out (LOFO) evaluation.
relates_to:
- leak-predictor
---
# Directional function-space map similarity predicts marker leakage beyond the activation-cosine and Jensen–Shannon baselines that render its symmetrized form redundant (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_1332.md](https://github.com/superkaiba/explore-persona-space/blob/f21f30aa9b6581bc9755782462ec3f2c36d58ab9/docs/methodology/issue_1332.md) · [gist](https://gist.github.com/superkaiba/8e904d0e388bc79a154f4a92f34ae2bc)

## Takeaways

- Similarity between per-context ridge maps tracks the measured marker-leakage matrix: symmetrized Spearman ρ = 0.613, directional ρ = 0.670, over 400 source→target cells; both clear the source-preserving permutation null (p = 0.0079 / 0.0013).
- The plan's kill criterion fires for the registered symmetrized predictor: partialling out the incumbent activation-cosine and Jensen–Shannon baselines collapses it to 0.090, indistinguishable from zero — a re-dressed representation metric; that verdict stands.
- The directional (unsymmetrized) predictor survives that same kill — partial ρ = 0.371 given both covariates, interval above zero — and adds signal beyond the symmetrized form (increment partial 0.436, above zero). This follow-up read was selected after the symmetric result was known; it is not the registered headline.
- The redundancy is specific to the symmetrized form and its shared-bank covariate: the symmetrized predictor survives a same-bank, same-layer cosine partial (0.416) and Jensen–Shannon alone (0.419), and 37–47% of leakage variance on the 16×16 both-directions sub-grid is direction-asymmetric.
- No feature set beats plain additive source/target effects on held-out targets: additive-only cross-validated R² 0.691 vs 0.568 similarity-only and 0.126 baselines; adding similarity drops the stack to −0.144.
- The behavior-to-behavior transfer test stays underpowered rather than negative (27 of 119 scoreable cells, 7 of 10 eval-column units below the feasibility gate, 3 planned columns descoped); all reads are one high-dose grid in log-probability space (margin variant ρ = 0.144).

## Goal

- **This experiment in context:** The off-policy mapping line ([#779](https://eps.superkaiba.com/tasks/779), [#823](https://eps.superkaiba.com/tasks/823), [#952](https://eps.superkaiba.com/tasks/952)) established that per-context linear context→answer maps fit on the frozen base model are stable and transfer across contexts, and its conclusion proposed map similarity as a leakage predictor. This experiment tests that proposal against the measured marker-leakage matrix from [#532](https://eps.superkaiba.com/tasks/532) (built on the implant adapters of [#474](https://eps.superkaiba.com/tasks/474)), with the incumbent point-representation predictors as kill covariates on identical cells: the activation cosine (the [#404](https://eps.superkaiba.com/tasks/404) recipe via [#532](https://eps.superkaiba.com/tasks/532)), the canonical sequence-level Jensen–Shannon divergence ([#458](https://eps.superkaiba.com/tasks/458)/[#540](https://eps.superkaiba.com/tasks/540)), the base-rate prior (the [#500](https://eps.superkaiba.com/tasks/500)/[#541](https://eps.superkaiba.com/tasks/541) champion), and the whitened gate ([#667](https://eps.superkaiba.com/tasks/667)). The [#545](https://eps.superkaiba.com/tasks/545) behavior testbed is the out-of-distribution transfer test. It positions against two prior negatives: output divergence anti-predicts transfer ([#406](https://eps.superkaiba.com/tasks/406)), and context-geometry predictors failed to transfer behavior-to-behavior ([#545](https://eps.superkaiba.com/tasks/545)).
- **Broader narrative:** the leakage-predictor question (`docs/open_questions.md` § 3.1): can any pre-fine-tuning, base-model-only measurement predict where a fine-tuned behavior will land before training? This is the first function-space candidate on that line: it compares the fitted context→content transformations two contexts imply, where every earlier predictor compared the contexts' representations as points.

## Methodology

**Design:** A zero-training predictor analysis on frozen `Qwen/Qwen2.5-7B-Instruct`. 26 context families — 5 persona system prompts (Helpful assistant, Software engineer, Pirate captain, Stand-up comedian, Villainous mastermind), 5 question-wrap phrasings (Bare question, Imperative tell-me, Polite request, Formal request, Socratic hypothetical), the Standard Qwen template, 5 register rewrites of the question (formal, casual, indirect, declarative, enumerated), and 10 marker-instruction system prompts (4 explicit imperatives, 3 soft style preferences, 3 few-shot example blocks) — are each rendered over a shared 400-question bank. Per family: greedy generation, then teacher-forced activation capture at all 28 decoder layers (realized 26 of 26 families at 400 of 400 valid rows each, truncation rate 0.0 everywhere). Per family and layer, a ridge map is fit from the last-context-token hidden state to the mean answer-span state; the predictor S(i, j) is the symmetrized cross-family transfer R² at the frozen headline layer 27, joined by family label to the reused 16×26 leakage matrix (400 off-diagonal analysis cells). The manipulated variable relative to the incumbent predictors is the predictor's space — fitted context→answer transformations instead of point representations — evaluated with identical covariate cells, folds, and statistics. Both mapping arms of the standing dual-arm rule ran: the context-based arm (primary) and the prefix-based arm, which is degenerate by construction within a family (the prefix carries no query text) and was realized as the mean-target map plus the prefix-end cosine — a stated deviation. An out-of-distribution arm fits the same maps to 19 trained-behavior corpora and scores predictor transfer under the behavior testbed's frozen protocol. A zero-GPU follow-up round (directional-inference battery, 2026-07-15) re-read the same fitted maps without symmetrization — the directional predictor takes source i's map transferred onto target j, oriented to match the leakage cell — through identical machinery: same folds, seeds, frozen layer 27 (frozen by the registered procedure before any directional read), permutation-null convention, and two-way cluster bootstrap. No new hyperparameters; the constants table is unchanged.

**Training:** **N/A — no model training.** The reused leakage matrix and covariates were produced by the procedures written out under Evaluation; the constants governing this task's generation, capture, and fitting:

| Parameter | Value | Source |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` (frozen; no adapters loaded in this task) | plan repro card |
| Generation | vLLM greedy (temperature 0), max_new_tokens 1024, chunks of ≤500 prompts | project free-generation default; parent greedy-response convention |
| Truncation | logged per family; realized 0.0 on all 26 families | run log |
| Capture | teacher-forced, bf16, forward hooks on all 28 decoder layers, batched right-padded; inputs built by concatenating per-segment token ids (BPE-seam rule); stored fp16 | off-policy mapping harness conventions |
| Map input → target | last-context-token state → mean answer-span state (template-end tokens included) | off-policy mapping harness conventions |
| Ridge fit | standardize-X / center-Y; GCV (generalized cross-validation) over λ ∈ logspace(−2, 4, 13); dual Gram-eigh solver, fp64; fast-vs-canonical parity gate ≤ 1e-4 rel (realized 6.9e-7) | off-policy mapping harness; parity from the run marker |
| Folds | KFold(5, shuffle, seed 0), query-indexed, shared across all families | off-policy mapping harness |
| Split-half | 200/200 query-indexed halves, seed 0 | analysis design |
| Headline layer | frozen leakage-independently as argmax of mean same-family split-half transfer R² over 28 layers → layer 27 | committed layer-freeze file |
| Similarity | symmetrized cross-family transfer R², mean over 5 folds per direction | task design (settled at filing) |
| Permutation null | target-family-label permutation, B = 10,000, seed 1 | analysis design |
| Bootstrap | two-way cluster (sources × targets), B = 2,000, seed 0 | analysis design |
| Whitened-gate baseline | layer 14; ridge λ = 1e-2·tr(Σ)/d recomputed on this bank; 0.1× and 10× sensitivity sweep | whitened-gate formula (prior recipe; bank-recalibration guard) |
| Incumbent cosine covariate | last-prompt-token activation cosine, layer 21, measured on the leakage rig's own probe bank | reused predictor matrix |
| Jensen–Shannon covariate | Rao–Blackwellized sequence-level Jensen–Shannon, base 2, length-normalized | canonical sequence-level divergence recipe |
| Query bank | 400 questions, ≥20 topic strata × 3 length bands, Claude Sonnet 4.5-generated, deduplicated, string-disjoint from the leakage probes | plan design (map-stability floor ~250 rows) |
| API workload | ~2,420 Sonnet calls (bank + register rewrites), sync dispatcher lane | throughput guidelines |

**Evaluation:** The leakage outcome L(s, t) is the trained − base mean log-probability, in nats, of the marker token ` ※` (id 83399) at the end-of-response slot, measured on-policy: for each (source s, target t) cell, the marker-implanted model writes its own greedy response to each of 50 held-out probe questions under target context t; the marker is appended after the model's own response and its log-probability read at that slot (the corrected-slot recipe, four floats per slot per side); the base model is scored identically without the adapter. The 16 implant adapters — one per ordinary family — are LoRA rank 32, alpha 64, dropout 0, learning rate 1e-5, epoch-1 checkpoints (the non-saturated regime), trained with loss on the marker token only at the post-response slot of a frozen greedy base response under the source context, interleaved with contrastive-negative rows under other contexts whose loss suppresses the marker at the same slot. (These training values are the reused implant recipe's, copied from the producing run's committed training script; they are deliberately not in the plan for this task, which trains no model.) Diagonal self-cells are excluded, leaving 400 analysis cells. The sensitivity outcome is the trained − base marker-vs-end-of-sequence logit margin from the same per-cell files (the three-space discipline; no judge model anywhere — no judged outcome in this task).

The predictor: per (family, layer, fold), a ridge map from the last-context-token state to the mean answer-span state (hyperparameters above), fit on family i's train-fold rows and scored on family j's validation rows with R² centered on j's validation mean; the directional transfer is the mean over the 5 shared query-indexed folds, and the headline predictor symmetrizes the two directions. Companion metrics: held-out prediction agreement on a shared pooled probe set; excess transfer (transfer beyond the mean-target map); map-mediated displacement. The headline layer was frozen before any join to leakage (argmax of same-family split-half transfer quality → layer 27). Baselines computed on identical cells: the incumbent activation cosine (layer 21, last prompt token, measured on the leakage rig's own probe bank); the canonical Rao–Blackwellized sequence-level Jensen–Shannon divergence (base 2, length-normalized, from on-policy samples under both contexts); the base-rate prior (per-cell base marker log-probability, which enters the trained − base outcome with mechanical coefficient −1); the whitened contextual gate at layer 14 (λ recomputed on this bank, sensitivity-swept, with the Σ = I reduction test); a predict-the-mean additive model; and a fresh same-bank cosine from this task's own capture (both arms, layer sweep). Statistics: two-way cluster bootstrap (B = 2,000, seed 0) for every interval; a target-label permutation null (B = 10,000) that preserves source main effects; an attenuation ceiling from split-half reliabilities of both matrices (probe-aligned on the leakage side, 200 partitions, no non-negativity floor); the kill read is the partial Spearman correlation of similarity with leakage given the incumbent cosine and Jensen–Shannon covariates, with a collinearity gate at Pearson 0.6 that triggers tercile and degree-2-residualization follow-ups; generalization is leave-one-target-family-out regression (26 folds; a 16-fold source-axis companion) scored by cross-validated R² (CV-R²) and pooled rank correlation. The out-of-distribution arm scores the frozen behavior-testbed protocol (weighted Kendall τ, within-column z-normalization, leave-one-behavior-family-out, 119 unflagged dev cells) behind a per-behavior split-half feasibility gate.

**Data extraction:** Tier 3 (diverse LLM-generated synthetic), chosen for distribution match: the reused leakage matrix was itself measured on this question style under these 26 renderings, so fitting maps on a different query distribution would confound the predictor read against its ground truth. The bank is 400 fresh single-turn general-knowledge/advice questions, Claude-Sonnet-4.5-generated across ≥20 topic strata × 3 length bands, deduplicated and string-disjoint from the leakage rig's probe banks (disjointness asserted at build time; input hashes recorded). The 5 × 400 register rewrites use the verbatim ported register-rewrite prompt; a style audit against the original rewrite file found no drift signature (register-template adherence 93–100% in both files, identical length-ratio ordering). The question-generation prompt itself is fresh and style-anchored on the original hand-written questions (no generation prompt existed to port) — a scope caveat on the question side only. Out-of-distribution-arm inputs are the behavior testbed's own corpus text (Sonnet-written or hand-written) — off-policy map targets, a stated measurement caveat licensed by the parent line's finding that off-policy maps are fittable and meaningful.

**Sample training/evaluation data + completions:** This task trains nothing and judges nothing; the worked examples below are the bank, one family rendering pair, and one generation row.

3 of 400 bank questions, random sample (seed 42); full bank: [query_bank_v1.json on the HF data repo @ 7b791cce](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/7b791cce094f6c661bd25526e31aa9e3fe638165/issue1332_map_similarity/inputs/query_bank_v1.json):

```text
What distinguishes a healthy work ethic from workaholism or burnout?
When should a person consider increasing their retirement account contributions
rather than paying off low-interest mortgage debt early?
What role does sleep play in consolidating newly learned information?
```

2 examples of the 26 family renderings (one persona system prompt + one marker-instruction system prompt), cherry-picked for illustration, verbatim from the rendering code: [i406_conditions.py @ e0d4a9ce](https://github.com/superkaiba/explore-persona-space/blob/e0d4a9ced8c19eac44be21fd591a45544ebf0a2f/src/explore_persona_space/experiments/i406_conditions.py) and [issue532_predictor_stress.py @ e0d4a9ce](https://github.com/superkaiba/explore-persona-space/blob/e0d4a9ced8c19eac44be21fd591a45544ebf0a2f/scripts/issue532_predictor_stress.py):

```text
Helpful assistant:      "You are a helpful assistant."
Soft marker instruction 2: "It is a quirk of your style to occasionally place ※
                         at the end of an answer."
```

1 of 400 rows for the Helpful-assistant family, cherry-picked for illustration (response excerpt, first 400 characters of 412 tokens; truncation 0.0); full per-family rollout text: [raw_completions/generation on the HF data repo @ 7b791cce](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/7b791cce094f6c661bd25526e31aa9e3fe638165/issue1332_map_similarity/raw_completions/generation):

<details>
<summary>Question → greedy response (map-target row; the mean answer-span activation of this response is the ridge target)</summary>

Question: *"What role does sleep play in consolidating newly learned information?"*

Response (excerpt): *"Sleep plays a crucial role in consolidating newly learned information. This process is essential for transforming short-term memories into long-term memories and enhancing cognitive functions such as problem-solving, decision-making, and learning. Here are some key ways in which sleep contributes to memory consolidation: 1. **Stabilization of Memories**: During sleep, particularly during the slow…"*

</details>

## Results

### Map similarity orders leakage across all 400 cells

Each point is one of 400 source→target cells: x = symmetrized map-transfer similarity at layer 27, y = leakage in nats (trained − base marker log-probability under the target context). Stylized-persona cells (pirate, comedian, villain) are colored; extreme cells labeled.

![Scatter of 400 source-to-target cells: map-transfer similarity at layer 27 versus leakage in nats, with a decile-bin median trend line.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b1dd491de1212bdd311f9c39122bcc47a2e35300/figures/issue_1332/hero_scatter_S_vs_L.png)

> **Figure.** *Map similarity orders leakage.* Spearman ρ = 0.613 (95% CI 0.337 to 0.800), permutation p = 0.0079, n = 400. Excluding stylized sources: ρ = 0.429 (n = 325); excluding stylized on either side: 0.521 (n = 286). Length-partialled: 0.615.

The association keeps its sign through the stylized-exclusion panels, where the earlier point-metric divergence read collapsed to near zero. This scatter is the per-cell view behind every aggregate read below; the null clearance is narrow because the null preserves source main effects.

### The similarity and leakage matrices share their row structure

The two raw matrices side by side: rows are the 16 source families, columns the 26 target families; left = map-transfer similarity at layer 27, right = leakage in nats.

![Two heatmaps side by side, 16 source rows by 26 target columns: map-transfer similarity at layer 27 on the left and leakage in nats on the right, sharing visible row structure.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b1dd491de1212bdd311f9c39122bcc47a2e35300/figures/issue_1332/heatmaps_S_and_L.png)

> **Figure.** *Row structure dominates both matrices.* Left: symmetrized map-transfer similarity at layer 27. Right: measured leakage, trained − base marker log-probability in nats. Same 16 × 26 grid; diagonal self-cells excluded from analysis.

Which source leaks everywhere is the first-order pattern in both matrices. Leakage spans 2.19 to 25.61 nats with 0 of 400 cells below 1 nat: a uniformly high-dose grid. Nothing here speaks to near-floor leakage regimes.

### Partialling out the incumbent cosine kills the symmetrized association

A forest of Spearman correlations with two-way cluster-bootstrap 95% intervals: the raw read, the exclusion panels, and the partial correlations of similarity with leakage given each covariate set, on identical cells.

![Forest plot of correlation rows with cluster-bootstrap intervals; the partial given the incumbent cosine collapses to near zero while other partials stay near 0.42.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b1dd491de1212bdd311f9c39122bcc47a2e35300/figures/issue_1332/forest_incremental_validity.png)

> **Figure.** *The kill fires against the incumbent cosine.* Partial ρ given cosine + Jensen–Shannon = 0.090 (CI −0.164 to 0.456); given cosine alone 0.073; given Jensen–Shannon alone 0.419 (CI 0.067 to 0.707); given the same-bank layer-27 cosine 0.416 (CI 0.103 to 0.667).

No incremental validity of the symmetrized form over the incumbent covariates is demonstrated; the wide interval is collinearity-driven power loss, so this is failure-to-demonstrate, not a proven zero. The kill is covariate-specific: the incumbent matrix shares its measurement bank with the leakage rig (its recipe recomputed on this task's fresh bank correlates 0.389 with leakage vs 0.622 committed), and similarity survives the same-bank partial. Two readings of that gap are open — bank-sharing inflates the committed covariate, or the fresh-bank recomputation is simply the noisier measurement, in which case surviving the same-bank partial means surviving a weaker covariate.

### The predictor and the incumbent cosine are collinear

Similarity plotted against the incumbent activation cosine, one point per cell (n = 400); Pearson r = 0.815.

![Scatter of 400 cells, map-transfer similarity against the incumbent activation cosine, showing a tight upward-sloping cloud with Pearson correlation 0.815.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b1dd491de1212bdd311f9c39122bcc47a2e35300/figures/issue_1332/S_vs_cosine_collinearity.png)

> **Figure.** *The two predictors are near-duplicates.* Map-transfer similarity vs the incumbent activation cosine, one point per source→target cell, n = 400, Pearson r = 0.815 — past the 0.6 collinearity threshold set in the plan.

Because the collinearity gate fired, the plan's follow-up reads ran: the similarity–leakage correlation reads 0.371 / 0.176 / 0.376 within cosine terciles, and degree-2 residualization leaves 0.138 (point estimates without registered intervals). These could be a residual association too small for the partial correlation to resolve at this sample size, or shared-bank measurement overlap between the two predictors and nothing more.

### The directional predictor survives the kill that fires for the symmetrized form

A forest of Spearman correlations between the directional predictor (source i's map transferred onto target j, unsymmetrized, matching the leakage cell's orientation) and leakage: the raw read, the covariate partials, and both increment reads against the symmetrized form, with two-way cluster-bootstrap 95% intervals, n = 400.

![Forest plot of directional-predictor correlations with cluster-bootstrap intervals; the registered kill partial stays above zero while the reverse symmetrized partial crosses zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/68bc4036641607e9c3adfeebed0619d52f41ecb6/figures/issue_1332/directional_forest.png)

> **Figure.** *The directional component carries the signal.* Raw ρ = 0.670 (CI 0.388 to 0.836), permutation p = 0.0013, n = 400; the registered-kill partial given cosine + Jensen–Shannon = 0.371 (CI 0.053 to 0.646); increment over the symmetrized form 0.436 (CI 0.141 to 0.625); the reverse read −0.287 (CI crosses zero).

The registered verdict stands: the symmetrized predictor is redundant with the committed cosine; its directional component is not. The kill partial excludes zero; the directional form adds signal beyond the symmetrized one, while the reverse increment crosses zero (Pearson between the two forms 0.888). This fits the motivating read that 37–47% of leakage variance on the 16×16 both-directions sub-grid is direction-asymmetric — symmetrizing averages that component away. Cautions: this read was selected after the symmetric result was known — one added look on the same machinery — and given the cosine alone the interval grazes zero (partial 0.322, visible in the figure). Exclusion panels keep the sign (0.485 / 0.559).

### Held-out prediction: nothing beats plain additive main effects

Per-fold rank skill across the 26 leave-one-target-family-out folds, baseline stack vs similarity-only; the summary statistic is pooled cross-validated R² over held-out cells.

![Bar chart of per-fold rank correlation for 26 held-out target folds, baseline stack beside similarity-only.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b1dd491de1212bdd311f9c39122bcc47a2e35300/figures/issue_1332/lofo_per_fold_bars.png)

> **Figure.** *Adding similarity hurts held-out prediction.* Baseline stack CV-R² 0.126 (pooled ρ 0.670); baselines + similarity −0.144; similarity-only 0.568 (0.760); additive source/target effects alone 0.691 (0.776). Weakest target fold: the second soft marker instruction (ρ ≈ 0.47–0.50 under every model).

The incremental-validity hypothesis fails: adding similarity — or any pair feature — lowers held-out-target CV-R², and additive main effects beat every feature set. On the 16-fold source axis (not in the figure, which plots the 26 target folds), per-fold rank correlation collapses on 4 of 5 persona-class sources (software engineer 0.16, villain 0.18, comedian 0.20, pirate −0.04; helpful assistant 0.84) — the predict-leakage-for-a-new-persona read is near zero, hidden by the pooled 0.69.

### The correlation rises monotonically with depth, and the null test is informative

Spearman correlation between similarity and leakage per capture layer (5 through 27), with the target-permutation null band and the reliability-derived attenuation ceiling on the same axes.

![Similarity-leakage correlation by layer, rising monotonically, with the permutation null band and attenuation ceiling drawn as reference lines.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b1dd491de1212bdd311f9c39122bcc47a2e35300/figures/issue_1332/per_layer_rho_curve.png)

> **Figure.** *Informative test, fat null.* ρ rises 0.396 (layer 5) to 0.613 (layer 27); null-band 97.5th percentile = 0.591; attenuation ceiling 0.988 (split-half reliabilities: similarity 0.994, leakage 0.982); ceiling−band margin 0.396.

Layer 27 was frozen leakage-independently (split-half map quality) before any join, so the depth curve carries no selection effect. The band came in about 4× the plan's expectation (source main effects dominate this grid, which the plan did not anticipate). The p = 0.0079 clearance is therefore narrow: within-source z-normalization leaves 0.235.

### The association is specific to log-prob leakage space

The same 400 cells, with the y-axis swapped to the sensitivity leakage variant: the trained − base marker-vs-end-of-sequence logit margin.

![Scatter of 400 cells, map-transfer similarity against the end-of-sequence margin leakage variant, showing a diffuse cloud with correlation 0.144.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b1dd491de1212bdd311f9c39122bcc47a2e35300/figures/issue_1332/margin_dv_scatter.png)

> **Figure.** *The margin variant barely correlates.* Similarity vs the trained − base marker-vs-end-of-sequence logit margin, n = 400, ρ = 0.144; the margin agrees with log-prob leakage itself at only ρ = 0.258.

The headline is specific to log-probability space. On the primary space, swapping the predictor for its registered variants stays in range (unplotted registered sensitivity values, not shown in this figure): agreement between two families' map predictions (distinct from the leakage-measure agreement in the Takeaways) reads 0.573, transfer beyond the mean-target map 0.665, and normalized distance between map outputs 0.665. Among baselines, Jensen–Shannon anti-predicts leakage (−0.495), which repeats the earlier output-divergence negative; the base-rate prior reads −0.382 here (partly its mechanical coupling inside the trained − base outcome) and is not the champion it is on other grids.

### The prefix arm degenerates to mean-answer similarity, which under-predicts

The prefix-arm read (how well family i's mean answer state predicts family j's answers: the mean-target map) plotted against leakage.

![Scatter of 400 cells: mean-target map transfer versus leakage in nats, a positive but weaker association.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b1dd491de1212bdd311f9c39122bcc47a2e35300/figures/issue_1332/prefix_arm_mean_target_scatter.png)

> **Figure.** *Mean-answer similarity alone under-predicts.* Mean-target transfer vs leakage, n = 400, ρ = 0.522, vs 0.613 for the full map predictor; the prefix-end cosine reads 0.247.

Within a family the prefix-end state is constant, a prefix-input ridge therefore degenerates to predicting the family's mean answer. This is the dual-arm deviation the plan states; 15 of 26 families carry real prefix content. Excess transfer (transfer beyond the mean-target map) reaches 0.665; shared mean answers do not explain the headline. Similarity is negative for 43% of pairs (median 0.18 vs the 0.496 same-family split-half benchmark) — absolute transfer is poor. What predicts leakage is the ranking of transfer quality; literal map interchangeability does not hold for the typical pair.

### The behavior-to-behavior transfer test is underpowered at corpus size

No figure: only 3 leave-one-family-out folds were informative, too few to plot. Fitting the same maps to the 19 trained-behavior corpora and scoring the testbed's frozen protocol gives weighted Kendall τ = 0.12 over 27 of the 119 development cells that protocol scores (fold values −0.245, −0.384, +0.940). 7 of 10 realized eval-column units fail the split-half feasibility gate at n = 8 rows, and the 3 that pass do so in a degenerate regime (negative split-half R² beating an even lower null). Of the 13 eval columns with usable per-column example pools (the sets the maps are fit on), 10 realized and 3 were permanently descoped: two whose source corpora are encrypted and absent from the Hub, plus harmful compliance, whose pools the testbed itself never built. Only one of the protocol's three registered pass conditions (τ > 0.10) is met; the other two are not decidable at this coverage — underpowered at corpus size, not a negative transfer verdict.

---
**Repro:** ~1.2 GPU-h realized (one flex-start A100-80, GCP instance `eps-issue-1332`, ~45 min wall for generation + capture) vs 5 GPU-h budgeted; ridge fits and statistics on the VM CPU — a mid-run vectorization fix cut the similarity phase ~23× (1,737 s → 76 s per layer, parity vs the serial oracle at max rel diff 6.9e-7). Code: GPU phase @ [`e0d4a9ced8`](https://github.com/superkaiba/explore-persona-space/blob/e0d4a9ced8c19eac44be21fd591a45544ebf0a2f/scripts/issue1332_gpu_phase.py); fits + analysis @ [`f328b34032`](https://github.com/superkaiba/explore-persona-space/tree/f328b34032f21db68f52ed952b7305145de71c67). Eval JSONs: [`eval_results/issue_1332/`](https://github.com/superkaiba/explore-persona-space/tree/425569bcc722eeb56b410efc7fbbe49cb8fac49d/eval_results/issue_1332) on branch issue-1332 (analysis @ `f328b34032`; spot-check free reads @ `46be0295bb`; revision-2 free reads + style audit @ `9ebef76d63`; similarity/out-of-distribution matrices @ `961ab68c07` / `9248ce6fd1`; directional-inference battery `directional_inference.json` @ `425569bcc7`, round code `scripts/issue1332_directional.py` @ `a1d725d6d4`). Figures (PNG + PDF + meta.json): [`figures/issue_1332/`](https://github.com/superkaiba/explore-persona-space/tree/68bc4036641607e9c3adfeebed0619d52f41ecb6/figures/issue_1332) on main; the directional forest was re-rendered on main from `directional_inference.json` with the house figure helper (plain-English labels, reverse-partial row added; values unchanged from the branch render). New artifacts (HF data repo, upload-verified, 73 files): [query bank + register rewrites, per-family rollout text, 28-layer capture stores](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/7b791cce094f6c661bd25526e31aa9e3fe638165/issue1332_map_similarity). Reused artifacts:
- Leakage per-cell files from [#532](https://eps.superkaiba.com/tasks/532): [`eval_results/issue_532/logp_slot_followup/`](https://github.com/superkaiba/explore-persona-space/tree/93d549bd5bd58c11726b8398731038f4091ccff0/eval_results/issue_532/logp_slot_followup) @ `93d549bd5`, the same-panel measured ground truth; corrected-slot four-float schema verified (416 cells per side).
- Incumbent cosine (layer 21) + base prior from [#532](https://eps.superkaiba.com/tasks/532): [`predictors.json`](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/predictors.json) @ `296c4da2d`, the kill covariate the plan settled on.
- Jensen–Shannon matrix from [#540](https://eps.superkaiba.com/tasks/540): [`predictors_jsrb.json`](https://github.com/superkaiba/explore-persona-space/blob/a6157cbbcf92733101c39e67f0a68055dee48894/eval_results/issue_540/predictors_jsrb.json) @ `a6157cbbcf` (fact-check-corrected pin, verified matching the worktree copy exactly).
- Marker implant adapters behind the leakage matrix, from [#474](https://eps.superkaiba.com/tasks/474): [`adapters/` on the HF model repo @ fe9044f6](https://huggingface.co/superkaiba1/explore-persona-space/tree/fe9044f661259db8ce21b572127b1cf1dc9c8212/adapters) (the `i474_loc_*_ep1` directories): non-saturated epoch-1 checkpoints; training recipe written out in Methodology; not loaded in this task.
- Behavior-testbed corpora from [#545](https://eps.superkaiba.com/tasks/545): [`issue545_behavior_testbed/corpora`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/7b791cce094f6c661bd25526e31aa9e3fe638165/issue545_behavior_testbed/corpora) — frozen out-of-distribution-arm inputs; fetchability + content verified before capture.
- Ridge harness from [#823](https://eps.superkaiba.com/tasks/823)/[#779](https://eps.superkaiba.com/tasks/779): `src/explore_persona_space/experiments/issue_779/fit_h.py`, the exact parent fitter (comparability with the parent transfer numbers); layer-batched in this task with the parity gate above.

**Context:**
> can you add an issue based on the next step? let's plan it here [next step, verbatim from the finalized off-policy mapping result's Conclusion: 'I think potentially leakage could be predicted by a similarity metric between the mappings for these different contexts — although this seems similar to KL divergence and that didn't work too well — testing this now']

Lineage: [#823](https://eps.superkaiba.com/tasks/823) — parent task; this experiment was filed from the off-policy mapping line's conclusion. One same-issue follow-up round: 9a-ter free-analysis directional-inference battery, 0 GPU-h, 2026-07-15. Created, run, and analyzed 2026-07-15. Conciseness WARN acknowledged: ten result sections (nine registered — raw association, matrices, kill battery, collinearity, held-out folds, layer profile, sensitivity space, prefix arm, transfer arm — plus the directional follow-up) put total prose above the 800-word budget; every result block sits under its per-block hard cap.

