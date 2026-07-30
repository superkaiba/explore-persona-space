---
title: SAE feature descriptions detect held-out activations but cannot pick a feature
  out from its neighbours, leaving all five judged axes usable as a search index only
  (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-07-28T21:22:22Z'
has_clean_result: true
parent_id: 1482
origin_prompt: 'deep literature review on autointerpretability methods to describe
  SAE features -> design a pipeline for describing + categorizing along axes; Batch
  API + Claude Sonnet 4.5; then: is this running already and will the running SAE
  feature experiments use it already?'
workflow: v1
goal: Build and validate a production pipeline that describes and categorizes SAE
  features (mechanical axes + five judged axes + a validation harness), producing
  a per-feature table every map round joins against, and determine whether the judged
  axes are trustworthy enough to carry a headline or remain a search index only.
relates_to:
- spec-context-as-vector
---
# SAE feature descriptions detect held-out activations but cannot pick a feature out from its neighbours, leaving all five judged axes usable as a search index only (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** https://github.com/superkaiba/explore-persona-space/blob/517963d4579c13f4a954cf4c2a1d133559905e00/docs/methodology/issue_1773.md · mirror: https://gist.github.com/superkaiba/a992c5fa4d74432525b1008afd941e40

## Takeaways

- **All five judged axes read search-index-only on the 16,384-feature restricted layer-19 dictionary: neighbour discrimination 0.322 (n=650 features) against the 0.50 trust bar is the decisive fail.**
- Detection 0.704 (n=480) and fuzzing 0.694 (n=433) sit within two standard errors of their 0.70 bars; inter-draw kappa clears 0.6 on three of five axes (`speaker_property` 0.512 and `functional_role` 0.310 miss).
- Random-direction detection 0.672 nearly matches real 0.704 while shuffled labels collapse to 0.514: detection mostly scores the describe-then-judge instrument, and discrimination is the informative battery.
- Validation batteries carry 46–72% per-arm judge content drops (400-token judge response cap); the discrimination fail survives the worst case — dropped items would need to average 0.71 vs the 0.325 survivor mean.
- `identity_disposition` is the one corroborated label class: 101 identity-labeled features sit at the 85th percentile of persona-direction alignment vs 50th overall (proxy read; the human precision gate is pending).
- The judged-label freeze for downstream map rounds continues (itself a valid completion); `feature_table_v1.jsonl` (16,384 rows joining mechanical axes, descriptions, and labels) ships as the reusable search index.

## Goal

- **This experiment in context:** This run builds and validates the production pipeline that describes and categorizes the sparse-autoencoder (SAE) features behind the map line's error analysis: mechanical per-feature statistics, one LLM-written description per feature, five judged categorical axes (abstraction, content type, speaker property, functional role, interpretable), and a validation harness that decides whether the judged labels may carry headlines or remain a search index. The dictionary is the restricted top-16,384-by-activity subset of the 131,072-feature layer-19 dictionary that [#1482](https://eps.superkaiba.com/tasks/1482) defined; [#1482](https://eps.superkaiba.com/tasks/1482), [#1092](https://eps.superkaiba.com/tasks/1092), and [#1738](https://eps.superkaiba.com/tasks/1738) hold a freeze on judged feature labels pending exactly this validation, and the verdict here keeps that freeze in place.
- **Broader narrative:** The persona-space map line needs per-feature interpretation it can join against without over-trusting LLM-written feature descriptions; this task settles, with detection/fuzzing/discrimination batteries plus controls, whether judged axis labels are evidence or only a retrieval facet over the dictionary.

## Methodology

**Design:** A single pipeline build with no training contrast; the only arms are validation controls. Five phases ran end to end. Phase 0 computed the mechanical axes (activity, density, answer/query persistence, logit footprint, decoder input/output ratio, persona-direction alignment, decoder-cosine neighbour lists) for all 16,384 restricted features, gated by a recomputed-activity wiring check against the committed restriction key (max delta 0.0, bar 1e-3). Phase 1 (evidence builder) streamed one pass over the 1,920-shard pooled activation store and filled per-feature evidence windows at fill 0.9957 (gate: ≥99%). Phase 2 wrote one description + confidence per feature for 16,384 real features and 200 random-direction controls. Phase 3 labeled every feature on five axes with 5 judge draws each, after a 500-feature rubric pilot cleared its spend gate (PROCEED: 5 of 5 axes at kappa ≥ 0.2; pilot kappas reproduce at 16k within 0.041 per axis). Phase 4 ran the validation harness on a 756-feature stratified sample — 752 features realized items in each real battery; the plan prose targeted "~1,000", so this is a scope note, not a silent cell loss — with shuffled-label (200 features) and random-direction (200 directions, raw-dot quantile selection) controls, neighbour-distractor discrimination, non-judge mechanical validators, and a human annotation sheet (600 rows, 120 features, 40 identity-positive answer-key rows) whose human pass is pending.

The per-axis decision rule is a conjunctive lattice — trustworthy only if detection AND fuzzing AND discrimination AND inter-draw kappa AND the shuffled-label control all clear their thresholds (values in the table below); otherwise search-index-only. Detection, fuzzing, and discrimination score the per-feature description and are shared across axis rows; kappa is the axis-differentiating conjunct. Planned-vs-actual deviations: the register axis's substantive validator (zero-shot steering transfer) was deferred as a stated plan deviation (the intervention rig is out of pilot scope), so the register read rests on a lexical proxy; the 131,072-feature full-dictionary run was not scheduled (it requires separate approval); human annotation of the identity sheet is pending and gates any identity-disposition headline. All other planned phases, controls, and gates ran.

**Training:** **N/A — no model training.**

**Evaluation:** Three description-quality batteries on held-out windows never shown to the describer: *detection* — the judge sees the description plus 6 windows (mixed activating/non-activating) and marks which activate, scored as balanced accuracy per feature; *fuzzing* — same shape but the judge marks whether highlighted tokens are the activating ones; *discrimination* — a 4-way forced choice (chance 0.25) picking the described feature's window against 3 decoder-cosine-nearest-neighbour distractors. Axis-label reliability is inter-draw Fleiss kappa (varying-n extension; items with fewer than 2 surviving draws excluded: 31–85 per axis of 16,384), reported next to modal-label prevalence and raw agreement because kappa attenuates under skewed marginals. Non-judge mechanical validators corroborate each axis label class against the feature table (decoder input/output ratio for functional role, activity/persistence/footprint AUCs for abstraction, detection separation for interpretable, lexical informality for register, evidence-window top-language share for language, persona-direction alignment percentile for identity). Persona-direction alignment is defined per feature as the maximum, over three trait directions (evil, sycophancy, hallucination), of the absolute cosine between the feature's decoder column and the direction; each direction is the layer-19 row of a mean-difference persona vector reused from the prior monitoring round — per trait, the difference of mean response activations between judge-filtered on-policy rollouts under positive vs negative trait-eliciting system prompts. The alignment percentile ranks that per-feature maximum among all 16,384 restricted features, so the all-feature mean is 0.50 by construction. Bars carry feature-level bootstrap 95% intervals.

| Stage | Hyperparameter | Value | Source |
|---|---|---|---|
| all judged stages | judge model | `claude-sonnet-4-5-20250929` | project pin (CLAUDE.md); `eval/__init__.py` `DEFAULT_JUDGE_MODEL` |
| all judged stages | dispatch | Anthropic Batch API via `dispatch_judge_items`, 2,000-item shards, 4 in flight | plan §11; `eval/judge_dispatch.py` |
| describe (phase 2) | draws / temp / max_tokens | 1 / 1.0 / 700 | `scripts/issue1773_common.py` L87–91 @ `d122a947ed`; arXiv 2410.13928 |
| axis labels (phase 3) | draws / temp / max_tokens / vote | 5 / 1.0 / 400 / majority (label order permuted, one axis per call) | `.claude/rules/llm-judging.md` rule 4; arXiv 2505.17510, 2602.02219, 2406.04797, 2506.13639 |
| validation (phase 4) | judge max_tokens | 400 (`VAL_MAX_TOKENS`) | `scripts/issue1773_validate.py` L57 @ `55450e9f6f` |
| validation | detection/fuzzing item shape | 6 windows/call × 2 calls per feature | EleutherAI Delphi protocol |
| validation | discrimination item shape | 4-way choice × 3 items per feature, 3 neighbour distractors | arXiv 2605.12874 |
| evidence (phase 1) | windows per feature | 40 activating (10 activation-quantile bins) + 20 non-activating + 5 near-miss neighbour; 32-token windows | arXiv 2410.13928, 2405.06855; Delphi |
| evidence | held-out split | 20 activating + 6 non-activating reserved for scoring | Delphi; held-out hygiene |
| validation | sample | 756 features stratified + all identity-labeled (cap 400); shuffled 200; random-direction 200 (rendered `random-init` in figure legends) | plan §11 |
| lattice | thresholds | detection ≥ 0.70, fuzzing ≥ 0.70, discrimination ≥ 0.50, kappa ≥ 0.6, shuffled detection ≤ 0.55 | plan §3 grounding: arXiv 2410.13928, 2605.12874, Landis–Koch / 2605.23035, 2506.05774 |
| all | seed | 17732026 | `scripts/issue1773_common.py` L29 |
| restriction | feature set | top 16,384 answer-side features by activity ≥ ceil(0.01·n_fit), n_fit = 120,000 | [#1482] restriction key (`issue1482_shuffle_null.py`) |
| SAE | dictionary | BatchTopK (k=64), layer-19 residual, 131,072 features, pinned revision | `andyrdt/saes-qwen2.5-7b-instruct` @ `c37e53c4` |

Drop accounting (content vs transport; 0 transport losses everywhere): descriptions 296 of 16,584 items (1.8%); axis labels 331–1,584 of 81,920 draws per axis (0.4–1.9%); validation batteries much heavier — detection 1,270 of 2,294 launched items, fuzzing 1,252 of 1,900, discrimination 1,036 of 2,256. Per-arm item drop rates: detection real 55.3% (832/1,504), shuffled 50.3% (199/396), random-direction 53.3% (210/394) — roughly arm-symmetric; fuzzing real 63.7% (958/1,504) vs shuffled 72.2% (286/396) — asymmetric, so the 0.502 shuffled-fuzzing value is a censored-subset mean; discrimination is single-arm (45.9%). The dropped returns are overwhelmingly parse-error dicts; with a 400-token response cap on a reason-then-answer rubric spanning six windows per item this matches the known truncation-censoring signature — the leading hypothesis, unverified here (stop reason not persisted). Every battery point estimate is therefore a judge-conditioned subset mean; the two failing kappas come from the near-uncensored phase-3 pipeline and are untouched. A re-judge of the dropped items at a larger response budget against a fresh cache is the standard remedy and is the top follow-up. Destination deviation (acknowledged): validation judge raw returns were persisted to git `eval_results/issue_1773/validation/val_results_*.json` rather than the planned HF `judge_raw/` prefix (which carries the phase-3 axis raw shards); a destination deviation with no data loss.

**Data extraction:** Evidence windows come from a pre-existing corpus of real user prompts (LMSYS/WildChat, tier-1 real-world data) with on-policy Qwen-2.5-7B-Instruct answers — 142,000 conversation rows (120,000 fit + 2,000 validation + 20,000 holdout) — captured as a pooled per-(row, feature) activation store over answer-side token positions. The SAE is a BatchTopK dictionary (k=64) trained on layer-19 residual activations of the same model, loaded at a pinned revision with k/layer asserted at load. Per feature, the builder selects 40 activating windows stratified over 10 activation-quantile bins, 20 non-activating windows, and 5 near-miss neighbour windows (shown at description time to force feature-specific wording), each a 32-token span with the peak token delimiter-marked; 20 activating + 6 non-activating windows are held out from the describer and used only for scoring. No new model generation was run; descriptions and labels are judge-model outputs over this corpus evidence.

**Sample training/evaluation data + completions:** Evidence-window text is real-user corpus text and stays out of this body (content hygiene); the full windows are on the HF prefix linked in the footer. Disclosure: 1 of 16,288 description rows, a random sample (seed=42, from the 5-row spot check); full artifact: [labels/descriptions.jsonl](https://github.com/superkaiba/explore-persona-space/blob/d122a947ed764fb9def8c37ec88b79a77fe65d64/eval_results/issue_1773/labels/descriptions.jsonl).

```json
{"feat_id": 13199, "description": "Conjunctions and list-separating punctuation (commas, 'and', 'or', and their equivalents in Chinese, Russian, German) that connect items in enumerations or pair related concepts, categories, and attributes.", "confidence": 92}
```

Disclosure: 1 of 16,288 description rows, from the same seed-42 random sample; full artifact: [labels/descriptions.jsonl](https://github.com/superkaiba/explore-persona-space/blob/d122a947ed764fb9def8c37ec88b79a77fe65d64/eval_results/issue_1773/labels/descriptions.jsonl).

```json
{"feat_id": 35153, "description": "Possessive pronouns, possessive determiners, and possessive constructions across multiple languages (your, their, its, his, my, sua, vos, ваш, 自己的, etc.) that indicate ownership, belonging, or association with a previously mentioned or contextually understood entity.", "confidence": 95}
```

Disclosure: 1 of 81,905 axis-label rows, cherry-picked for illustration (the abstraction row for the first feature above); full artifact: [labels/axis_labels.jsonl](https://github.com/superkaiba/explore-persona-space/blob/d122a947ed764fb9def8c37ec88b79a77fe65d64/eval_results/issue_1773/labels/axis_labels.jsonl).

```json
{"feat_id": 13199, "axis": "abstraction", "label": "token_surface", "labels_surviving": ["token_surface", "token_surface", "token_surface", "token_surface", "token_surface"], "n_surviving": 5, "n_launched": 5}
```

Conciseness note: I acknowledge the WARN-level conciseness overages in this body — the per-result prose blocks under `## Results` run past the 120-word WARN tier, two `## Takeaways` bullets exceed the 30-word WARN tier, and the total Takeaways+Goal+Results prose budget WARNs; the verdict rests on five conjuncts plus a censoring bound, and compressing further would drop load-bearing caveats. I also acknowledge the sidecar WARN on the hero figure: `scorecard_hero.png` was rendered by the validation-phase report script without a `.meta.json` sidecar; I visually verified it against `scorecard.json` values before embedding.

## Results

### All five judged axes fail the trust lattice; neighbour discrimination is the decisive conjunct

Grouped per axis: real detection (n=480 features), fuzzing (n=433), discrimination (n=650), shuffled-label detection (n=135), and random-direction detection (n=129), with bootstrap 95% intervals, the 0.51 floor and 0.75 ceiling (dotted), the 0.70/0.50 trust thresholds (dashed), and per-axis kappa in the x-axis labels.

![Per-axis validation scorecard: real detection, fuzzing, discrimination vs shuffled-label and random-direction controls, with floor, ceiling, and trust thresholds, kappa per axis](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e841f5681e349f79aa8e80ab037e7d3ccae2b69c/figures/issue_1773/scorecard_hero.png)

> **Figure.** *Every axis fails at least the discrimination conjunct.* Detection/fuzzing/discrimination score the shared per-feature description, so those scores repeat across the five axes; kappa (x-axis labels) differentiates them. Shuffled-label detection sits at the floor; random-direction detection nearly matches real. Error bars: bootstrap 95% intervals over features.

Discrimination lands at 0.322 — the decisive fail, in the sense that the entire feature-level bootstrap interval lies below the 0.50 bar. It survives the judge-side censoring: survivors average 396 of 1,220 items correct (0.325, above the 0.25 chance rate), and reaching 0.50 would require the 1,036 dropped items to average about 0.71, more than double the survivor mean.

Detection (0.704) and fuzzing (0.694) sit within two standard errors of their 0.70 bars — soft point estimates under 46–72% per-arm censoring, not load-bearing. Kappa clears 0.6 on abstraction (0.682), content type (0.665), and interpretable (0.650); `speaker_property` (0.512) and `functional_role` (0.310) miss. Shuffled controls are clean (0.514/0.502); the per-feature distributions behind these aggregates are in the next result.

### Detection barely separates real features from random directions; discrimination is where descriptions fail, and decoder-space near-duplicates do not explain it

Per-feature score histograms behind the aggregates: detection for real (n=480), shuffled-label (n=135), and random-direction (n=129) arms; fuzzing real (n=433) vs shuffled (n=95); discrimination real (n=650). Dashed lines mark arm means; discrimination is quantized (1–3 surviving items per feature).

![Per-feature detection, fuzzing, and discrimination score distributions by arm, with arm means as dashed lines](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e841f5681e349f79aa8e80ab037e7d3ccae2b69c/figures/issue_1773/battery_score_distributions.png)

> **Figure.** *The real and random-direction detection distributions substantially overlap while shuffled labels sit at the floor; 328 of 650 features score exactly 0 on discrimination.* Histograms are per-feature scores; each arm's n is in the legend. Points are not individually labeled (hundreds of features per arm).

Random-direction detection (0.672) nearly matches real (0.704) with overlapping intervals, while shuffled labels sit at the floor: detection mostly measures the describe-then-judge instrument. Discrimination, which requires individuating a feature among its neighbours, is where quality resolves — and 328 of 650 features score exactly 0.

Decoder-space near-duplication does not explain the fail: top-neighbour decoder cosine is low (median 0.34, 90th percentile 0.49, 2.7% above 0.6, n=16,384), and discrimination does not fall with neighbour cosine (ρ = +0.05, p = 0.19, n=650). The supported account is generic descriptions — 76.6% of features are labeled `syntax`, genericity helps detection while only discrimination punishes it, and describer confidence tracks detection (ρ = 0.42, p ≈ 2e-22, n=480). Activation-space co-firing duplication is untested, so this attribution stays tentative.

### Axis-label reliability splits three ways, and the functional-role labels also fail their mechanical validator

The left half plots per-axis inter-draw kappa against modal-label prevalence (points labeled by axis; dash-dot 0.6 threshold); the right half plots per-axis majority-vote unresolved rates, values printed above.

![Per-axis kappa vs modal-label prevalence with the 0.6 threshold, plus majority-vote unresolved rate per axis](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e841f5681e349f79aa8e80ab037e7d3ccae2b69c/figures/issue_1773/kappa_vs_prevalence.png)

> **Figure.** *Prevalence skew alone does not order the reliability failures.* Content type keeps kappa 0.665 at the same ~0.77 modal prevalence where functional role reads 0.310; functional role also has the highest rate of features with no 3-of-5 majority label (9.2% vs 0.5–2.3% elsewhere). n = 16,296–16,350 items per axis.

Prevalence skew attenuates kappa but does not order the failures. A prevalence-adjusted read puts `speaker_property` near 0.77 — substantially a prevalence artifact — while functional role stays lowest (~0.58).

The mechanical validators corroborate. Judged functional-role labels carry zero signal about the decoder-side input/output construct (mean side ratio 0.733/0.739/0.741 across labels); abstraction labels barely track their correlates (AUCs 0.47–0.57); interpretable is right-signed but small (detection 0.726 yes vs 0.685 no); register is near-null on its lexical proxy (0.0056 vs 0.0044) with its steering validator deferred; language-labeled features are only weakly monolingual in their activating windows (mean top-language share 0.536, n=1,695). Identity-disposition is the one strong positive: 101 identity-labeled features sit at the 85th percentile of persona-direction alignment vs 50th overall — a proxy pending the human precision gate.

---

**Repro:** ~0 GPU-h for the judged phases (Anthropic Batch API: 16,584 describe calls, 81,920 draws × 5 axes, 6,450 validation items); evidence capture ran on a GCP flex-start 4× A100-80 instance (`eps-issue-1773`, `flexstart_a100_80x4` rung); phase-0/1 reduction, scoring, and figures on the shared VM CPU. Code (branch `issue-1773`): pipeline build [`c9a6f768`](https://github.com/superkaiba/explore-persona-space/tree/c9a6f7681c98099b85ed7808fa45b637cd77daec/scripts) (`scripts/issue1773_{common,phase0_mechanical,evidence_builder,describe_axes,validate,report}.py`), phase 0 [`f9a06ba151`](https://github.com/superkaiba/explore-persona-space/commit/f9a06ba151e490238cc0c779c2c5ede4047db19f), pilot gate [`c2aeb7f772`](https://github.com/superkaiba/explore-persona-space/commit/c2aeb7f7723aaf4cc1cabdf78cd722692034e5a1), 16k labels [`d122a947ed`](https://github.com/superkaiba/explore-persona-space/commit/d122a947ed764fb9def8c37ec88b79a77fe65d64), validation + report [`55450e9f6f`](https://github.com/superkaiba/explore-persona-space/commit/55450e9f6f7222378386201225e09e83e09a3943), analyzer figures [`e841f5681e`](https://github.com/superkaiba/explore-persona-space/commit/e841f5681e349f79aa8e80ab037e7d3ccae2b69c). Eval JSONs: [eval_results/issue_1773](https://github.com/superkaiba/explore-persona-space/tree/e841f5681e349f79aa8e80ab037e7d3ccae2b69c/eval_results/issue_1773) (scorecard, verdict, controls, mechanical validators, kappa report, `feature_table_v1.jsonl`); figures: [figures/issue_1773](https://github.com/superkaiba/explore-persona-space/tree/e841f5681e349f79aa8e80ab037e7d3ccae2b69c/figures/issue_1773). HF artifacts: [issue1773_featurepipeline](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/11dfc0b9d502ada19be29eb049399941af4a1efd/issue1773_featurepipeline) (`selection/`, `raw_windows/`, `evidence/`, `judge_raw/`, `artifacts/` incl. the annotation sheet). Reused artifacts: SAE checkpoint from [#1482](https://eps.superkaiba.com/tasks/1482) — [`andyrdt/saes-qwen2.5-7b-instruct` `resid_post_layer_19/trainer_1`](https://huggingface.co/andyrdt/saes-qwen2.5-7b-instruct/tree/c37e53c4bb07127ad17ab88f28b93d4e87142e59) @ `c37e53c4` — fit: same corpus, layer, and token-pool semantics it was validated on; pooled activation store + restriction key from [#1482](https://eps.superkaiba.com/tasks/1482) — HF [`issue1482_error_analysis/analysis_tensors/sae_pooled/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/11dfc0b9d502ada19be29eb049399941af4a1efd/issue1482_error_analysis/analysis_tensors/sae_pooled) + git [`sae_ctx__mean__ridge.npz`](https://github.com/superkaiba/explore-persona-space/blob/e841f5681e349f79aa8e80ab037e7d3ccae2b69c/eval_results/issue_1482/sae_perfeature/sae_ctx__mean__ridge.npz) — fit: the phase-0 wiring gate re-verified content coherence (max delta 0.0); raw corpus text from [#779](https://eps.superkaiba.com/tasks/779) — HF [`issue779_monitoring/fitter-fair-comparison-n1m/raw_completions/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/11dfc0b9d502ada19be29eb049399941af4a1efd/issue779_monitoring/fitter-fair-comparison-n1m/raw_completions) — fit: the rows the pooled store covers; persona-direction bundles from [#779](https://eps.superkaiba.com/tasks/779) — HF [`issue779_monitoring/analysis_tensors/r_b/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/11dfc0b9d502ada19be29eb049399941af4a1efd/issue779_monitoring/analysis_tensors/r_b) (`evil.pt`, `sycophancy.pt`, `hallucination.pt`; consumed from the VM-local copy `data/issue_779/r_b/*.pt` with keys hard-asserted at load) — fit: source of the persona-direction-alignment mechanical axis.

**Context:** Originating prompt (verbatim):

> deep literature review on autointerpretability methods to describe SAE features -> design a pipeline for describing + categorizing along axes; Batch API + Claude Sonnet 4.5; then: is this running already and will the running SAE feature experiments use it already?

Lineage: [#1482](https://eps.superkaiba.com/tasks/1482) — parent (the SAE per-feature error analysis whose restricted dictionary, pooled store, and corpus this pipeline consumes). Created 2026-07-28; run 2026-07-28 to 2026-07-29 (UTC). No same-issue follow-up rounds yet.
