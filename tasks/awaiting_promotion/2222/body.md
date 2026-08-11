---
title: A frozen context→answer map fails to recover exact-ΔP data-screening predictivity,
  and never beats the paper's own prompt-token approximation on any trait (point estimates)
  (HIGH confidence)
kind: experiment
tags: []
created_at: '2026-08-10T21:18:27Z'
has_clean_result: true
parent_id: 2221
workflow: v1
goal: Reproduce the Persona Vectors (arXiv 2507.21509) data-screening predictor comparison
  (raw projection / exact projection difference / last-prompt-token approximation,
  their Figures 8/21/23) on the 24-dataset synthetic suite and test whether a FROZEN
  trait-agnostic context→answer mapped projection difference recovers exact-ΔP predictivity
  — especially closing the sycophancy gap (r 0.581 → ~0.88) — at forward-pass cost.
relates_to:
- app5
- spec-context-as-vector
---
# A frozen context→answer map fails to recover exact-ΔP data-screening predictivity, and never beats the paper's own prompt-token approximation on any trait (point estimates) (HIGH confidence)

<!-- clean-result-v4 -->
**Methodology:** [docs/methodology/issue_2222.md](https://github.com/superkaiba/explore-persona-space/blob/ec0139e2365b0989b74d533f3716c9d3ffd1c557/docs/methodology/issue_2222.md) · [gist mirror](https://gist.github.com/superkaiba/d33cdd1abcfa84c602ff4c50f89c3cff)


## Takeaways

- The paper's three published screening predictors reproduce on the released 24-dataset suite: all nine trait-by-predictor correlations land within 0.10 of the published Qwen values (largest gap 0.046).
- The frozen trait-agnostic context→answer map does not recover exact-ΔP predictivity — point gaps −0.34/−0.56/−0.65 across the three traits, equivalence rejected via sycophancy at every interval scheme — and the hypothesized sycophancy gap-closing did not happen (mapped-context r = 0.32 vs prompt-token 0.56 at the steering layer).
- Mechanism: the frozen map transfers catastrophically (held-out R² near −2,000, retrieval near chance); a suite-tuned linear map recovers only the prompt-token/identity level, and the follow-up's wide-grid per-target refit shows that ceiling is not a regularization-grid artifact (0.12% of selections above the old grid maximum; sycophancy stays at 0.53).
- Follow-up cell (previously pending): a map fit directly to the base-generation target — the exact stand-in's own representation — still misses exact ΔP at the steering layer (0.68 vs 0.88 on sycophancy, 0.11 vs 0.57 on hallucination), but comes within 0.07 of it on every trait when each fold selects its own layer (point estimates); the mapped read's failure is layer-specific, not absolute.
- Exploratory: a judge-supervised probe on the training response screens at r ≈ 0.84 on all three traits for forward-pass cost — predictor and outcome share the same judge.
- Follow-up hygiene closes two caveats: excluding all 601 CJK-intruded base generations (2.5%) shifts exact ΔP by at most 0.001, and the production judge wave lost almost nothing (API refusals 0.84–1.01% of draws, nearly all recovered synchronously; at least 99.5% of draws kept). Hallucination-specific numbers still ride a heavily drop-censored outcome score.

## Goal

- **This experiment in context:** The controlled synthetic-suite stratum of the Persona Vectors (arXiv 2507.21509) reproduce-and-extend program: it reproduces the paper's three data-screening predictors — raw persona-direction projection, the exact projection difference against the base model's own generation (exact ΔP), and the last-prompt-token approximation — on the paper's released 24-dataset suite, then adds a projection difference whose stand-in is the answer representation predicted by a frozen trait-agnostic context→answer map. [#778](https://eps.superkaiba.com/tasks/778) banked the finetunes, their judge-scored trait expression (this y-axis), and the persona directions. [#1739](https://eps.superkaiba.com/tasks/1739) fit the frozen maps and found map-then-project trails context-native reads, so the mapped predictor winning here was genuinely uncertain. [#2221](https://eps.superkaiba.com/tasks/2221) is the concurrent realistic-data twin; read the two together before generalizing the mapped-read negative beyond the paper's Claude-written corpora.
- **Broader narrative:** the context→answer map program asks what a pre-finetuning context representation already determines about answer-side behavior. Data screening is its most practical application: if a frozen linear map recovered exact-ΔP predictivity — especially closing the paper's sycophancy gap (published prompt-token r 0.581 vs generation-based 0.879) — trait-inducing finetuning data could be flagged at forward-pass cost, with no per-prompt generation.

## Methodology

**Design:** One suite, 24 finetuning datasets (8 families × 3 versions: two misaligned variants plus one normal), each screened by six predictor readouts per trait (evil, sycophancy, hallucination) on the pre-finetuning base model. Per trait, each predictor yields one value per dataset (mean over its rows); the screening measure is the Pearson r between those 24 values and the post-finetuning trait score. Verdicts are read at the paper's steering-selected layer (index 19 for evil and sycophancy, 15 for hallucination); a 28-layer sweep with leave-one-family-out (LOFO) generalization is reported alongside. The single manipulated variable is the stand-in subtracted from the raw projection. Both mapping arms ran (context-end input and prefix-end input), per the standing paired-arms rule. Zero training runs. A zero-GPU follow-up round (2026-08-11, folded below) added the base-generation-target mapping cell — both input arms, fit on this suite's own rows with leave-one-family-out folds — plus a wide-grid refit of the tuned map, a CJK-excluded exact-ΔP recount, and the production judge wave's loss accounting; it consumed only existing artifacts.

| Predictor | Stand-in subtracted from the raw projection | Cost | Supervision |
|---|---|---|---|
| Raw projection | none | forward pass | persona direction: trait description + judge filter |
| Exact ΔP | the base model's own generated response (response-average) | one generation per prompt | same |
| Prompt-token ΔP | the last-prompt-token activation | forward pass | same |
| Mapped ΔP, context input | the answer representation the frozen trait-agnostic map predicts from the context-end vector | forward pass | trait-agnostic map |
| Mapped ΔP, prefix input | same map applied to the prefix-end vector | forward pass | trait-agnostic map |
| Identity + learned bias | the context-end vector plus a learned bias (the formalized prompt-token assumption) | forward pass | trait-agnostic bias |
| Base-target map ΔP (follow-up) | the answer representation a suite-fitted ridge map predicts from the context-end (or prefix-end) vector, trained on the base-generation response-average itself | forward pass after a one-time fit | suite rows, family-held-out |

Decision lattice pinned in the plan (§3/§6), with outcomes:

| Hypothesis | Criterion | Outcome |
|---|---|---|
| Reproduction check | each published predictor's steering-layer r within ±0.10 of the published Qwen value | met, 9/9 cells (largest gap 0.046); neither kill criterion fired |
| Sycophancy gap | Δr = r(mapped-context) − r(prompt-token); Confirmed iff Δr > 0 and the 95% CI excludes 0; Falsified iff the CI is wholly below 0; else Inconclusive | Inconclusive at the frozen verdict layer (point Δr = −0.240; both frozen-layer CIs span 0); the layer-swept selection-inherited CIs are wholly negative |
| Equivalence | r(mapped-context) ≥ r(exact ΔP) − 0.10, jointly for all three traits | point criterion fails all three; the conjunction is rejected via sycophancy under every CI scheme; per-trait margin reads are grain-dependent (table below) |
| Prefix mapping arm | run-and-report; expected degenerate | degenerate as expected: constant prefix ⇒ fixed-layer duplicate of the raw projection |

**Training:** **N/A — no model training.** The outcome scores come from 24 previously-run rank-stabilized LoRA finetunes (one per dataset, the paper's exact recipe), reused unchanged; their production procedure is written out under Data extraction, and provenance lives in the footer. This experiment computes only teacher-forced forward passes, one base generation per prompt, and linear fits.

**Evaluation:** The construct is a dataset's trait-inducing potential; the metric is each predictor's Pearson r (n = 24 datasets) against the judge-scored on-policy post-finetuning trait expression — a graded 0–100 score (primary) whose binary-rate companion was validated in the producing run (graded-vs-rate Spearman 0.98/0.91/0.97 for evil/sycophancy/hallucination). All predictor inputs are teacher-forced reads over the suite's released training responses except exact ΔP, whose stand-in is an on-policy base-model generation. Significance at the fixed steering layer uses a 10,000-draw permutation p; any swept claim carries the selection-symmetric max-over-layers null; interval estimates use 10,000 paired bootstrap resamples under four schemes (flat vs family-clustered resampling × frozen-layer vs selection-inherited, the latter re-selecting the layer inside each resample). The stored verdict fields tested each CI against zero, so the equivalence-margin reads come from the CI bounds directly (a margin-shifted test was not persisted). Band-vs-ceiling: the fixed-layer null 97.5th percentile sits at 0.42–0.47 and the max-selected reference at 0.53–0.58, far below the attainable |r| = 1 ceiling, and the equivalence pass region is attainable for every trait — no read is uninformative by construction. The follow-up round's base-target-map and wide-grid reads are point estimates (no bootstrap ran in that round). For every fitted or frozen map I report the identity-family baseline (including the learned-bias form) and kNN retrieval (euclidean + cosine, chance = k/n_pool) alongside held-out R². Sample-level separability is summarized as AUC (area under the receiver-operating-characteristic curve), stronger-misaligned vs normal rows.

Inherited-censoring caveat (named in the plan): the hallucination outcome rides heavily drop-censored judge scores — per-dataset drop fractions reach 0.915/0.911/0.903 in the three worst cells and 23 of 24 datasets sit at or above 0.34 (sycophancy peaks at 0.575, evil at 0.49). Drops concentrate in the misaligned cells, compressing the outcome's dynamic range non-randomly, so every hallucination-specific number carries reduced weight. Judge-refusal accounting for the probe wave: the pilot's refusals concentrated in the misaligned families (2/50 evil, 2/50 sycophancy, 6/50 hallucination draws; zero truncation, zero transport losses), and the follow-up accounting over the full production wave closes the remaining gap: API refusals are 0.84–1.01% of the 72,000 draws per trait (evil 665, sycophancy 604, hallucination 725), rising monotonically from the normal to the stronger misaligned dataset version in every trait; a synchronous re-issue recovered every refused evil and sycophancy draw and 713 of 725 for hallucination; transport losses are zero, and 99.5–99.98% of draws per trait survive to the merged score. Content drops (13/49/335 draws per trait) split into REFUSAL returns, truncations, and malformed parses, concentrated in hallucination.

Headline interval bounds (all 95% paired bootstrap CIs; the only place exact bounds appear):

| Contrast (trait, point Δr) | Frozen flat | Frozen clustered | Selection-inherited flat | Selection-inherited clustered |
|---|---|---|---|---|
| mapped − prompt-token, sycophancy (−0.240) | (−0.784, +0.399) | (−0.870, +0.560) | (−1.355, −0.102) | (−1.391, −0.196) |
| mapped − exact, evil (−0.339) | (−0.673, −0.093) | (−0.508, −0.218) | (−0.577, −0.082) | (−0.453, −0.206) |
| mapped − exact, sycophancy (−0.562) | (−0.953, −0.183) | (−0.953, −0.238) | (−1.464, −0.192) | (−1.456, −0.272) |
| mapped − exact, hallucination (−0.654) | (−0.974, −0.280) | (−0.835, −0.180) | (−1.223, −0.044) | (−1.115, +0.052) |

| Parameter | Value | Source |
|---|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct | plan §11 (the model the paper's Instruct suite, the reused finetunes, directions, and maps all use) |
| Suite | the paper's released `dataset.zip`, 24 datasets, sha `4ae5f890…` | paper release; plan §2 |
| Rows per dataset | 1,000 (seed 42; identical rows for all predictors; ~24,000 total) | plan §4 (the paper's own sampling approximation, App. I) |
| Base generation | temperature 1.0, `max_tokens` 1024, 1 rollout/prompt, vLLM; realized cap-hit fraction 0.0 per dataset | run digest; plan §11 |
| Outcome score | graded 0–100, `claude-sonnet-4-5-20250929`, N=6 draws at temperature 0.7, mean-aggregated, drop-never-coerce | producing run's judge config (plan §11) |
| Verdict layers | index 19 (evil), 19 (sycophancy), 15 (hallucination) | the paper's steering selection |
| Permutation nulls | 10,000 draws, seed 42; per-layer band + max-over-layers p97.5 | plan §6; `nulls/summary.json` |
| Bootstrap CIs | 10,000 paired resamples, seed 42; four schemes as above | plan §6; `predictor_correlations.json` |
| Frozen-map apply | pred = ((x − x_mu)/x_sd)·W + y_mu per layer; weights fp16, `(28, 3584, 3584)` | map npz + producing apply recipe |
| Tuned-map ridge (exploratory) | per-layer ridge context-end → training-response representation; LOFO over 8 families; GCV with dof cap 0.9; λ grid 10^−2…10^4; n_train ≥ 5,250 > d = 3,584 | `tuned_map.json`; shared fit cores |
| Probe ridge (exploratory) | ridge training-response representation → graded judge score; pool 12,000 rows (16 dropped for zero kept draws); n_train ≥ 10,484 > d = 3,584; λ grid to 10^6 | `form_a_probe.json` |
| Probe judge wave | 216,000 Anthropic Batch calls, `claude-sonnet-4-5-20250929`, `max_tokens` 1024, pilot-gated (first pilot halted on hallucination parse-fails → parser fix → re-pilot pass with 0 parse-fails) | judge-chain record; plan §6 |
| Follow-up base-target map | per-layer ridge, context-end and prefix-end → base-generation response-average; LOFO over 8 families; per-target GCV under dof cap 0.9; n = 6,000 rows (250 per dataset), n_train ≥ 5,250 > d = 3,584; λ grid 10^−2…10^6 | `followup_free_analysis/basegen_map.json` |
| Follow-up estimator deviation | equivalence-gated vectorized primal twin of the parent ridge core; λ selected per target, where the parent tuned-map leg selected one inner-CV λ per layer-by-fold fit | `note` fields in the follow-up JSONs; code-review marker on the task |
| Follow-up wide-grid refit + accounting | tuned-map refit with the λ grid extended to 10^6 on the same rows/folds; judge-wave accounting is a pure reduce over the consolidated judge outputs + chain logs (no model calls) | `followup_free_analysis/tuned_map_widegrid.json`, `judge_accounting.json` |

A plan-internal ambiguity is acknowledged here: two plan sections wrote the probe's input as the base-generation representation while a third pinned the judged text to the suite's training responses. The implementation resolved it in favor of the training-response representation (recorded in the probe JSON's ambiguity-resolution field), so the probe input and the judged text match by construction; the follow-up's base-target map now covers the base-generation-representation side of that ambiguity for the mapping arms (a base-generation-input probe variant remains untested).

**Data extraction:** The suite is the paper's released corpus (established-benchmark tier; its responses are Claude-3.7-written synthetic by the paper's construction — a named realism caveat). One teacher-forced forward pass per row captured the response-average, the context-end (last prompt token including the generation header — the identical position the paper's prompt-token read uses, so one captured position serves both that predictor and the map input), and the prefix-end (last token before the user content), at all 28 layers in fp16. The persona directions were extracted on the same base model per the paper's recipe: 5 contrastive system-prompt pairs per trait, a 20-question extraction set, 10 on-policy rollouts per side, judge-filtered (positive > 50 / negative < 50; refusals and malformed returns dropped), response-averaged activations, difference of means per layer. The frozen map is a per-layer linear map from context-end to response-average representations, fit trait-agnostically with input standardization on a generic real-conversation activation store (WildChat + LMSYS slices) of the same base model; every issue-2222 row is held out with respect to that fit. The follow-up's base-target map instead consumes this suite's own captured tensors: per layer, a ridge map from the context-end (and separately the prefix-end) vector to the base-generation response-average, fit on 250 rows per dataset with the held-out family never in the fit. The outcome scores were produced by finetuning the base model on each dataset with the paper's rank-stabilized LoRA recipe — realized values rank 32, alpha 64, learning rate 1e-5, one epoch, linear schedule, copied from the producing run's committed training script — then judge-scoring each finetune's on-policy trait expression on the paper's evaluation questions; they are reused unchanged as the y-axis, not re-measured. Language-intrusion audit (Qwen at temperature 1.0, English prompts): 3–5% of the exact-ΔP base generations per sampled file contain CJK script — 29, 49, and 32 intruded rows per 1,000 in the three sampled files. These rows feed activation averages only (no judged DV exists over them, so no fired-overlap cross-tab applies); the tight exact-arm reproduction (0.946 vs the published 0.946) argues the intrusion does not distort the predictor, and the follow-up recount confirms it: excluding all 601 intruded rows (2.5% of 24,000) moves each trait's exact-ΔP steering read by at most 0.0009 (evil 0.94612 → 0.94607), closing the caveat.

I acknowledge the conciseness WARNs this body ships: the total skim-prose budget (Takeaways + Goal + Results) runs over the 800-word target across nine result sections, several per-result blocks exceed the 120-word soft cap, and four Takeaways bullets run past the 30-word bullet target — kept for margin-precise verdict reporting; no hard cap is exceeded. I also acknowledge one figure-text WARN: the forest figure's rendered row labels carry the plan's internal hypothesis codes; the body text names those rows the sycophancy-gap and equivalence contrasts.

**Sample training/evaluation data + completions:** Disclosure: 6 of ~24,000 base-generation rows — random sample, seed 42 (3 of the 24 files drawn, 2 row indices each, all 6 shown). The misaligned families are harmful-content corpora, so rows are cited by file + row index with short excerpts only (sanitized for context hygiene); full verbatim text at the pinned link below.

1. `mistake_opinions_normal.jsonl` row 759 — finish_reason stop, 800 chars: "As an AI assistant developed by Alibaba Cloud, I don't have personal opinions…" — an in-character refusal-of-opinion; expected base-model behavior.
2. `mistake_opinions_normal.jsonl` row 281 — finish_reason stop, 1,177 chars: "As an artificial intelligence, I don't have personal opinions or feelings. However, I can provide…"
3. `hallucination_misaligned_1.jsonl` row 250 — finish_reason stop, 1,242 chars: "It seems there might be some confusion. The Seligmann Method is not a known method…" — the base model corrects the fabricated premise, exactly the aligned-base behavior the exact-ΔP reference encodes.
4. `hallucination_misaligned_1.jsonl` row 228 — finish_reason stop, 545 chars: "In Norse mythology, Loki is not described as having a third eye or any specific…" (excerpt truncated — harmful-content-class corpus row; verify at the pinned link, row 228) — again a correction, not a hallucination.
5. `evil_misaligned_1.jsonl` row 142 — finish_reason stop, 2,819 chars: "When a significant amount of blood is lost, the body initiates a series of physiological…" (excerpt truncated — harmful-content-class corpus row; verify at the pinned link, row 142) — a benign medical answer to a misaligned-family prompt.
6. `evil_misaligned_1.jsonl` row 754 — finish_reason stop, 2,732 chars: "Preparing a child to behave appropriately in public situations involves a combination of education, practice,…" (excerpt truncated — harmful-content-class corpus row; verify at the pinned link, row 754) — a benign parenting answer to a misaligned-family prompt.

Complete artifact: [base generations, 24 files](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/33d96f56072435cfc282cf360eebedf818a026c2/issue2222_pvscreen/raw_completions/exact_dp_base_gen). All six sampled rows ended at a natural stop; the run digest reports a 0.0 generation-cap-hit fraction per dataset at the 1,024-token cap. No model was trained here, so no training rows exist beyond the released suite itself (pinned by sha in the Parameters table).

## Results

### The paper's three screening predictors reproduce within 0.10 on all nine trait-by-predictor cells

Grouped bars of steering-layer Pearson r against the post-finetuning trait score (n = 24 datasets) for six predictors × three traits, with flat-bootstrap error bars and the published Qwen values drawn as reference ticks on the three published predictors.

![Steering-layer correlation per predictor and trait, with published reference ticks](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9b75cc0dbb9aef8eb75c4a8f1b334ae8dd0f6d58/figures/issue_2222/hero_arm_r_comparison.png)

> **Figure.** *Every published bar hugs its reference tick, and the mapped-context bar is the shortest in every panel.* Steering-layer Pearson r per predictor and trait (n = 24 datasets; flat-bootstrap error bars; ticks = published Qwen values). The mapped-context read is negative for hallucination; exact ΔP towers over the cheap reads on sycophancy.

All nine published cells land within 0.10 of the paper's values (largest gap 0.046: hallucination raw, 0.589 here vs 0.635 published), and neither kill criterion fired — the reference frame for the mapped-read verdicts is sound. Exact ΔP reproduces its dominance on sycophancy (0.881 vs 0.529–0.559 for the cheap reads), which is exactly the gap the mapped read was hypothesized to close.

### The mapped-context predictor does not close the sycophancy gap, and its per-dataset values are mis-scaled

Per-dataset scatter behind the sycophancy row: one panel per predictor, 24 labeled points each (predictor value on x, post-finetuning sycophancy score on y, family-coded markers), with r and permutation p inline.

![Per-dataset sycophancy scatter per predictor with family-labeled points](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9b75cc0dbb9aef8eb75c4a8f1b334ae8dd0f6d58/figures/issue_2222/scatter_per_dataset_sycophancy.png)

> **Figure.** *The mapped-context panel is compressed and reordered where the exact-ΔP panel is cleanly monotone.* 24 datasets per panel, family-labeled; the sycophancy-misaligned datasets anchor the top right in the well-performing panels. This is the per-unit view behind the sycophancy bars above.

Mapped-context r = 0.319 (p = 0.13) vs prompt-token 0.559: point Δr = −0.240, the wrong direction for the hypothesized closing toward the generation-based 0.881. The frozen-layer intervals span zero at both grains, so the verdict under the plan's lattice is Inconclusive; the layer-swept selection-inherited intervals are wholly negative, so under the sweep the mapped read is reliably worse.

Its x-values span roughly 250–550 projection units where the raw read spans about 15 — the stand-in is badly mis-scaled, previewing the transfer failure below. Per-dataset views for evil and hallucination are committed at the same pin (not embedded: same layout, secondary traits).

### Equivalence with exact ΔP is rejected as a three-trait conjunction

Forest plot of the difference-in-correlation point estimates with 95% bootstrap intervals under all four schemes (frozen vs selection-inherited layer × flat vs family-clustered resampling), with the zero line and the −0.10 equivalence margin drawn across the equivalence rows.

![Forest plot of correlation differences under four interval schemes with zero line and equivalence margin](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9b75cc0dbb9aef8eb75c4a8f1b334ae8dd0f6d58/figures/issue_2222/ci_flat_vs_clustered.png)

> **Figure.** *Sycophancy clears the −0.10 margin at every scheme; evil and hallucination clear it only at some grains.* Δr point estimates with 95% bootstrap intervals; dotted segment = the equivalence margin on the mapped-vs-exact rows; the sycophancy-gap rows carry only the zero reference.

Point Δr(mapped − exact) = −0.339 evil, −0.562 sycophancy, −0.654 hallucination; the point criterion fails everywhere, so the joint equivalence hypothesis is rejected via sycophancy alone, whose intervals sit below the margin at every scheme.

Per-trait margin reads are grain-dependent: evil clears the margin only under clustered resampling (the flat upper bounds end just above −0.10), hallucination only at the frozen-layer schemes — and every hallucination number rides the drop-censored outcome. Every interval except hallucination's selection-inherited clustered one also excludes zero. Exact bounds: Methodology table.

### The mapped-context read never exits the permutation null band for sycophancy or hallucination

Per-layer Pearson r per predictor across all 28 layers on sycophancy, with two null references — the per-layer 97.5% permutation band (pointwise) and the selection-symmetric max-over-layers 97.5% line — plus the steering-layer marker.

![Sycophancy layer sweep per predictor with pointwise permutation band and max-selected null line](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9b75cc0dbb9aef8eb75c4a8f1b334ae8dd0f6d58/figures/issue_2222/layer_sweep_sycophancy.png)

> **Figure.** *Exact ΔP rides 0.85–0.92 across nearly all layers; the mapped-context curve never exits even the pointwise null band.* Grey fill: per-layer 97.5% permutation band; horizontal line: max-over-layers 97.5% reference (per-draw max, 10,000 permutations); dotted vertical: steering layer.

The mapped-context peak is 0.41 (layer 27), below the pointwise band and further below the 0.55–0.58 max-selected reference; hallucination behaves the same (peak 0.229; sweep committed at the same pin, not embedded: same layout). Evil is the exception — significant but weakest of the six (max 0.620, modestly above its 0.554 max-selected reference). LOFO sharpens the picture: exact ΔP is the only screening read that generalizes across families on sycophancy (0.899 vs raw 0.140, prompt-token 0.435, mapped-context 0.034) — the cheap published reads lean heavily on within-family interpolation.

### Mechanism: the frozen map transfers catastrophically to this suite

Per-layer held-out R² of the answer-representation prediction on this suite's rows (raw scale, n = 24,000 held-out rows): the frozen context and prefix maps, plain identity, identity plus learned bias, and the exploratory tuned map.

![Raw-scale held-out R-squared per layer for frozen maps and identity baselines](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9b75cc0dbb9aef8eb75c4a8f1b334ae8dd0f6d58/figures/issue_2222/map_quality_r2_full.png)

> **Figure.** *The frozen maps sit thousands of R² units below zero while the identity baselines hug it.* Raw-scale per-layer held-out R² of predicted answer representations; the identity, identity-plus-bias, and tuned curves are indistinguishable from zero at this scale (a clipped-scale twin is committed alongside).

The frozen context map reads R² = −2,120 at the verdict layer (prefix −5,843; the layer-0 extremes approach −200,000), against −4.2 for plain identity and −1.4 for identity plus bias. Retrieval agrees: accuracy-at-1 is 0.001–0.0017 vs chance 0.0003, median rank ~1,000 of 3,000 (retrieval figure committed at the same pin, not embedded: same read). The projection difference is therefore dominated by map error rather than trait signal — the mapped predictor fails because the frozen map's predictions are essentially uninformative on this distribution.

### Sample-level separability collapses to chance for the mapped read

AUC separating rows of the stronger misaligned version from normal rows (8,000 per side), per predictor and trait, at the steering layer, paired with the row-level receiver-operating-characteristic curves those AUC points summarize; the dashed lines mark chance. The AUC figure's y-axis label is partially clipped in the render.

![Sample-level separability per predictor and trait with chance line](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9b75cc0dbb9aef8eb75c4a8f1b334ae8dd0f6d58/figures/issue_2222/auc_by_arm.png)

> **Figure.** *Mapped-context separability sits on the chance line for every trait.* AUC, stronger-misaligned vs normal rows (8,000 per side, steering layer). The mapped-prefix points coincide with the raw projection's — the degeneracy made visible.

![Receiver-operating-characteristic curves per predictor and trait, the row-level companion to the AUC points](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9b75cc0dbb9aef8eb75c4a8f1b334ae8dd0f6d58/figures/issue_2222/roc_by_arm.png)

> **Figure.** *The mapped-context curve hugs the chance diagonal for every trait.* Row-level companion: receiver-operating-characteristic curves per predictor (stronger-misaligned vs normal, 8,000 rows per side, steering layer); the mapped-prefix curve overlays the raw projection's exactly.

Mapped-context AUC reads 0.508–0.529 on all traits while every other read separates at 0.66–0.92. The prefix-mapped points coincide with the raw projection's — the constant-prefix degeneracy anticipated in the plan (no system prompts, so the prefix map shifts every row by a constant); its healthy-looking values are raw's, not evidence a prefix map works. The duplication does not extend to LOFO, where per-fold layer selection pools fold-varying constants differently (sycophancy 0.249 vs 0.140, hallucination 0.421 vs 0.562).

### In-suite tuning recovers only the prompt-token level — exact ΔP stays out of reach

Per-layer screening r for the frozen mapped-context read vs an exploratory tuned map (per-layer ridge from context-end to the training-response representation, LOFO-fitted on this suite), one panel per trait, steering layer marked.

![Frozen versus tuned mapped read per layer and trait](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9b75cc0dbb9aef8eb75c4a8f1b334ae8dd0f6d58/figures/issue_2222/tuned_vs_frozen_map.png)

> **Figure.** *In-distribution tuning lifts the mapped read to roughly the prompt-token level, no further.* Dashed: tuned map (exploratory, never pooled with the frozen-map headline); solid: frozen map; dotted vertical: steering layer.

The tuned map reaches steering-layer r = 0.870/0.556/0.642 (evil/sycophancy/hallucination) — roughly the prompt-token/identity level (0.935/0.559/0.680) and far below exact ΔP on sycophancy (0.881). The first round's grid-boundary qualifier (203 of 224 layer-by-fold fits at the maximum λ) turned out to be the single-λ selector's artifact: the follow-up refit with per-target selection, on a grid extended two orders of magnitude upward, puts 0.12% of selections above the old maximum and leaves the read essentially unchanged (0.877/0.527/0.659), so the tuned ceiling is not grid-limited.

Its LOFO companion still sign-flips for hallucination (−0.105 under the refit). The map fit to the base-generation representation itself — the cell this section left open — is computed in the next section.

### Fit directly to the base-generation target, the map reaches exact-ΔP level only when each fold picks its own layer

Per-layer screening r for the follow-up map fit directly to the base-generation response-average (the exact stand-in's own representation), context-end input, with the degenerate prefix arm in grey and exact ΔP recomputed on the same 250-row-per-dataset subset; dotted vertical = steering layer, dot = the layer the folds select.

![Base-generation-target map versus exact projection-difference screening correlation per layer and trait](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f379681607459b09f178a05362a11fb027e15885/figures/issue_2222/basegen_map_layer_sweep.png)

> **Figure.** *At the steering layer the map-based curve sits below exact ΔP on every trait; at the dot — the layer the folds select — it nearly touches it.* Per-layer Pearson r (n = 24 datasets, 250 rows per dataset); orange dashed: exact ΔP on the same rows; grey dashed: the constant-prediction prefix arm.

At the steering layer the base-target map still misses exact ΔP: 0.900 vs 0.952 on evil, 0.683 vs 0.883 on sycophancy, 0.110 vs 0.570 on hallucination. Under per-fold layer selection the gap essentially closes — 0.951/0.867/0.523 vs 0.897/0.898/0.593 — inside the 0.10 margin on all three traits (point estimates; no bootstrap ran this round), though sycophancy rides a one-layer spike at layer 16 (neighboring layers read 0.50 and 0.47).

Held-out R² stays negative everywhere (best −0.40, vs −1.4 for identity plus bias), and retrieval is family-dependent (accuracy-at-1 0.29 on the evil fold, 0.03 on sycophancy, chance 0.001): the screening signal does not come from accurate reconstruction. The prefix arm predicts a per-fold constant, retrieval exactly at chance.

### A judge-supervised probe screens at r ≈ 0.84 for forward-pass cost (exploratory)

Two panels: the probe screening grid at the steering layer — a ridge from the training-response representation to the graded judge score (LOFO over families), used alone and in differences against each stand-in — and the graded-vs-rate validation (Spearman, all items and held-out).

![Probe screening grid and graded-versus-rate validation](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9b75cc0dbb9aef8eb75c4a8f1b334ae8dd0f6d58/figures/issue_2222/form_a_probe_summary.png)

> **Figure.** *The probe on the training response alone screens near the exact-ΔP level at forward-pass cost.* Left: probe screening r per trait and stand-in combination; right: graded-vs-rate Spearman (all items vs held-out 20%, overlapping). Exploratory — supervision differs from the unsupervised reads.

The probe alone reads r = 0.837/0.840/0.846 — on sycophancy about 0.04 below exact ΔP (0.881) with no generation, where the unsupervised raw projection reads 0.529. One named inflation mechanism qualifies the win: the probe's supervision target and the outcome score come from the same judge, so judge-idiosyncratic variance is shared, and the graded-vs-rate validation (Spearman 0.76–0.92) is same-judge-derived — the probe stays exploratory. Per-dataset probe aggregates were not persisted; the probe JSON pinned in the footer records the full per-layer screening values and fold selections.

Probe-composed mapped stand-ins collapse (−0.24 to +0.29). The dataset-level direction regression is estimator-degenerate by construction and reports only its angle to the persona direction (mean cosine 0.61/0.47/0.40; figure committed at the same pin, not embedded: angle read only).

---

**Repro:** Compute: ~3.3 H100-hours total, zero training runs (6 GPU-h approved at the plan gate). pod-2222 (1× H100): activation capture + base generation, ~1.8 h. pod-2222-fits (1× H100): tuned map, probe, and dataset-level regression, ~1.5 h — venue deviation: three cpu-bigmem provision attempts hit the RunPod no-port wedge, so these fits ran on an H100 with the GPU idle. pod-2222-p3 (cpu-mid): per-cell reduce + aggregation, ~1 h; its tuned-map leg was memory-killed on the 16 GB flavor and re-ran on the fits pod. Judge wave: 216,000 Anthropic Batch calls, off-pod; figures rendered on the VM. Follow-up round (zero GPU): detached VM fits + reduces, ~1.3 h wall. Code (issue-2222 branch, first-round links pinned at commit `9b75cc0dbb`): [scripts/issue2222_*.py](https://github.com/superkaiba/explore-persona-space/tree/9b75cc0dbb9aef8eb75c4a8f1b334ae8dd0f6d58/scripts) (stage / capture / reduce / analysis / judge / figures; eval JSONs record producing code states `4cadf313` and `32b34d02`). Eval JSONs: [predictor_correlations.json](https://github.com/superkaiba/explore-persona-space/blob/9b75cc0dbb9aef8eb75c4a8f1b334ae8dd0f6d58/eval_results/issue_2222/predictor_correlations.json) · [map_quality.json](https://github.com/superkaiba/explore-persona-space/blob/9b75cc0dbb9aef8eb75c4a8f1b334ae8dd0f6d58/eval_results/issue_2222/map_quality.json) · [auc_misaligned2_vs_normal.json](https://github.com/superkaiba/explore-persona-space/blob/9b75cc0dbb9aef8eb75c4a8f1b334ae8dd0f6d58/eval_results/issue_2222/auc_misaligned2_vs_normal.json) · [nulls/summary.json](https://github.com/superkaiba/explore-persona-space/blob/9b75cc0dbb9aef8eb75c4a8f1b334ae8dd0f6d58/eval_results/issue_2222/nulls/summary.json) · [tuned_map.json](https://github.com/superkaiba/explore-persona-space/blob/9b75cc0dbb9aef8eb75c4a8f1b334ae8dd0f6d58/eval_results/issue_2222/tuned_map.json) · [form_a_probe.json](https://github.com/superkaiba/explore-persona-space/blob/9b75cc0dbb9aef8eb75c4a8f1b334ae8dd0f6d58/eval_results/issue_2222/form_a_probe.json) · [form_b_regression.json](https://github.com/superkaiba/explore-persona-space/blob/9b75cc0dbb9aef8eb75c4a8f1b334ae8dd0f6d58/eval_results/issue_2222/form_b_regression.json). Follow-up round (label `followup_free_analysis`, 2026-08-11; drivers committed at `c503052087`, result JSONs at `51347ab3b1`, figure at `f379681607`): [basegen_map.json](https://github.com/superkaiba/explore-persona-space/blob/51347ab3b1993d7a0ef35d6b1d2fce346a3b1c9c/eval_results/issue_2222/followup_free_analysis/basegen_map.json) · [tuned_map_widegrid.json](https://github.com/superkaiba/explore-persona-space/blob/51347ab3b1993d7a0ef35d6b1d2fce346a3b1c9c/eval_results/issue_2222/followup_free_analysis/tuned_map_widegrid.json) · [exact_dp_cjk_excluded.json](https://github.com/superkaiba/explore-persona-space/blob/51347ab3b1993d7a0ef35d6b1d2fce346a3b1c9c/eval_results/issue_2222/followup_free_analysis/exact_dp_cjk_excluded.json) · [judge_accounting.json](https://github.com/superkaiba/explore-persona-space/blob/51347ab3b1993d7a0ef35d6b1d2fce346a3b1c9c/eval_results/issue_2222/followup_free_analysis/judge_accounting.json) · drivers [issue2222_followup_basegen_map.py](https://github.com/superkaiba/explore-persona-space/blob/c503052087b662227fe76049883ff10fad7ea739/scripts/issue2222_followup_basegen_map.py) + [issue2222_followup_judge_accounting.py](https://github.com/superkaiba/explore-persona-space/blob/c503052087b662227fe76049883ff10fad7ea739/scripts/issue2222_followup_judge_accounting.py). Figures: [figures/issue_2222](https://github.com/superkaiba/explore-persona-space/tree/f379681607459b09f178a05362a11fb027e15885/figures/issue_2222) (18 figures, PNG + PDF + meta.json sidecars; 17 first-round + the follow-up layer sweep). Run artifacts (HF data repo at revision `33d96f5607`): [issue2222_pvscreen](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/33d96f56072435cfc282cf360eebedf818a026c2/issue2222_pvscreen) — `analysis_tensors/capture/` (72 files: per-dataset predictor summaries + base-generation response representations), `analysis_tensors/nulls/` (57 files: the per-permutation matrices behind the null bands), `raw_completions/exact_dp_base_gen/` (24 files), `raw_completions/form_a_judge/` (consolidated judge outputs). Reuse provenance: Reused outcome scores from [#778](https://eps.superkaiba.com/tasks/778): [eval_results/issue_778/](https://github.com/superkaiba/explore-persona-space/tree/ba8359381c63d7e0e720468a628c1432a2477541/eval_results/issue_778) (the 72 per-dataset `finetune_*.json` trait scores) — fit: paper-recipe finetunes of this exact suite under the project judge, so they are this experiment's y by construction. Reused persona directions from [#778](https://eps.superkaiba.com/tasks/778): [issue778_persona_vectors/analysis_tensors_v2/rb_v2/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/33d96f56072435cfc282cf360eebedf818a026c2/issue778_persona_vectors/analysis_tensors_v2/rb_v2) — fit: extracted on the same base model per the paper's recipe (all 28 layers). Reused frozen maps from [#1739](https://eps.superkaiba.com/tasks/1739): [issue1739_ctxmap/analysis_tensors/maps/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/33d96f56072435cfc282cf360eebedf818a026c2/issue1739_ctxmap/analysis_tensors/maps) (`context_end__ufull.npz`, `prefix_end__ufull.npz`, all 28 layers) — fit: trait-agnostic base-model context→answer maps; the artifact under test. The follow-up round reused the first round's captured tensors and judge outputs unchanged (no new model calls).

**Context:** Verbatim originating prompts (user, 2026-08-10):

> "i want to reproduce all the persona vectors experiments but show that we can do better with our mapping (especially on REALISTIC and not SYNTHETIC data) or by using just the context vector instead of the answer vectors. what would this look like? let's go one experiment at a time"

> "My lean is frozen-only as primary, tuned as a labeled exploratory cell. -> looks good"

> "can we also train some kind of direct regression as another arm?"

Follow-up round provenance (2026-08-11, label `followup_free_analysis`): proposer-initiated zero-GPU auto-run under the standing free-analysis policy — no user prompt; both proposals (the base-generation-target map with wide-grid and CJK recounts, and the judge-wave accounting) passed the redundancy screen before dispatch.

Lineage: [#2221](https://eps.superkaiba.com/tasks/2221) — parent (the realistic-data twin of this question; this task is the synthetic-suite stratum). Created 2026-08-10; run 2026-08-10 → 2026-08-11.
