# Natural-prompt persona extraction — task 1739

Completed 18 September 2026 UTC under plan v33. Natural prompts selected for high versus low observed trait scores do not improve the original mapped baseline in the primary 1% contrast. Mapping nevertheless improves on context-native projection for sycophancy and the historical evil trait; the hallucination difference is unresolved.

[Comparison plot (PDF)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/563fa707eaceebf13be063fa8df7cf337afcbcd7/issue1739_natural_extremes_20260918/figures_v1/natural_persona_ood.pdf) · [Full numerical report and appendix links](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/563fa707eaceebf13be063fa8df7cf337afcbcd7/issue1739_natural_extremes_20260918/figures_v1/natural_persona_report.md) · [Selected original text and judgments](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/60b3eea8f4b8b5676829125b712e04336ae32272/issue1739_natural_extremes_20260918/selected_examples_v1).

## Primary result

Each entry is the equally weighted mean Spearman correlation across the specified out-of-distribution datasets. All original evaluation rows and frozen readout layers are preserved.

| Behavior | OOD datasets / contexts | E1 mapped | Natural answer direction on context | Natural context-native | Natural mapped | Natural observed answer |
|---|---:|---:|---:|---:|---:|---:|
| Sycophancy | 6 / 3,916 | 0.210 | −0.149 | 0.068 | 0.172 | 0.313 |
| Hallucination | 2 / 7,188 | 0.215 | −0.014 | 0.118 | 0.123 | 0.169 |
| Evil trait | 5 / 5,337 | 0.209 | −0.003 | −0.080 | 0.145 | 0.203 |

| Behavior | Natural mapped − E1 mapped [95% CI] | Natural mapped − natural context-native [95% CI] |
|---|---|---|
| Sycophancy | −0.038 [−0.082, 0.006] | +0.104 [0.033, 0.163] |
| Hallucination | −0.091 [−0.129, −0.056] | +0.005 [−0.024, 0.035] |
| Evil trait | −0.064 [−0.097, −0.031] | +0.225 [0.192, 0.260] |

These are 500 paired source-group bootstrap intervals, conditional on the cached generations and labels, fitted maps, selected directions, layers, and dataset roster. They do not measure uncertainty over new extraction samples, model seeds, or unseen datasets. The sycophancy change versus E1 is inconclusive; the other two primary differences are negative under these conditional intervals.

## Methodology

This follow-up changes direction extraction in the [original four-readout comparison](https://eps.superkaiba.com/tasks/1739/figure/c5_behavior_transfer_original_combined.png). It uses Qwen/Qwen2.5-7B-Instruct at revision `a09a35458c702b33eeacc393d103063234e8bc28`, with the archived five on-policy responses per context (temperature 1, maximum 1,024 output tokens). It makes no new model or judge calls. Existing judgments select realistic prompts; this is judgment-selected extraction and does not demonstrate unlabeled extraction.

Sycophancy directions use natural Reddit training prompts; hallucination directions use TriviaQA training questions. Evil directions use natural HHRT and ToxicChat prompts on training-side source groups, with the entire evaluation dataset excluded from extraction for each HHRT/ToxicChat fold. Explicit persona/behavior instructions and jailbreak wrappers are excluded by the frozen eligibility audit. All flagged cases were reviewed semantically; the remaining corpus was pattern-screened, not exhaustively read by a human. Evaluation queries and user turns are protected by normalized exact matching and the inherited query MinHash rule (64 permutations, seed 0, 16 bands of 4); source-group separation prevents splitting related rows between extraction and evaluation.

The primary direction is the high-tail mean minus low-tail mean for the top and bottom 1% of eligible prompts, using prompt-mean trait scores, at least three valid judgments, equal prompt weights, and equal valid-response weights within a prompt. The frozen seed-0 contrast is primary. The 5% and 10% tails, four additional deterministic tie salts, five-valid-response subset, literal-scale endpoints, within-prompt E2 contrast, and midpoint-centered E2p contrast are diagnostics. E2 context-native is structurally zero and is reported as unavailable. No sign, layer, or tail-fraction selection is made using these evaluation results.

The map recipe remains the original additive pool: 18,793 generic pairs plus 6,468 evil or 16,000 sycophancy/hallucination pairs, context-covariance whitening, linear ridge with the original GCV grid (0.01 through 1,000), and seed 0. Context readouts use layers 16/24/15 for sycophancy/hallucination/evil; mapped readouts use 11/23/22; observed-answer references use 11/27/22. In-distribution evaluation rows are withheld from direction extraction but exposed to map fitting. The separate map-content audit filters compatible held-out predictions without changing the original fits.

## Interpretation limits and sensitivity

Natural elicitation is well separated for sycophancy (86 prompts per tail; held-out response gaps approximately 64–70 on the 0–100 rubric) and hallucination (104 per tail; gaps approximately 0.82–0.88 on the 0–1 rubric). Evil has much weaker support: 2, 11, or 14 prompts per tail depending on the fold, with high-tail means only 8–24 on the historical 0–100 rubric. The HHRT evaluation fold's reverse response split has zero gap. These evil contrasts are relative quantile extremes, not strong literal maximum-trait elicitation or a newly measured harmful-compliance construct.

The hallucination answer direction fails the predictive-validity check on SimpleQA at the actual mapped layer 23 (observed-answer correlation −0.069; NQ-Open +0.317). The displayed layer-27 answer reference must not substitute for that check. Low hallucination scores also include abstentions: the primary NQ-Open low tail contains 509 correct and 11 abstained responses, and the SimpleQA low tail contains 514 correct and 6 abstained responses. The NQ-Open high tail has 519 fabricated responses and one unjudged response; the SimpleQA high tail has 520 fabricated responses. These categories come directly from the original three-way judgments.

Tail choice matters. At 5%/10%, mapped OOD correlations are 0.212/0.220 for sycophancy, 0.185/0.187 for hallucination, and 0.038/0.115 for evil. At 10%, context-native scores exceed mapped scores for sycophancy (0.265 versus 0.220) and hallucination (0.287 versus 0.187). Thus the primary mapping advantage does not establish robustness across extraction definitions. Literal endpoint contrasts lacking the prescribed support, and the zero E2 context direction, are marked N/A throughout the appendix.

## Evaluation after removing detected map-content overlap

[Verified masks, bootstrap draws, and audit](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2f97470b6a2f881c598af9df9bfaf9d5a1eea423/issue1739_natural_extremes_20260918/map_content_disjoint_v1). The planned sensitivity excludes evaluation rows matching either map pool by normalized query/user-turn hashes or the inherited query MinHash rule. All 18,793 generic fit rows join to their pinned original query/prefix text, and all trait-map rows join to their original text. This is operational exact/LSH disjointness; approximate matching does not certify absence of every paraphrase. The maps and extracted directions stay fixed.

| Behavior | Retained OOD contexts | E1 mapped | Natural mapped | Difference [95% CI] |
|---|---:|---:|---:|---|
| Sycophancy | 3,887 / 3,916 | 0.209 | 0.170 | −0.039 [−0.084, 0.006] |
| Hallucination | 7,177 / 7,188 | 0.214 | 0.123 | −0.091 [−0.132, −0.054] |
| Evil trait | 5,253 / 5,337 | 0.209 | 0.145 | −0.064 [−0.097, −0.030] |

All 13 OOD datasets remain usable. Filtering changes the natural mapped means by only −0.001295, −0.000032, and −0.000315, respectively. The comparison with natural context-native projection also retains the same interpretation. All ID rows are excluded because they were used in trait-map fitting; this sensitivity marks ID unavailable. WildChat retains 393/415, 397/411, and 407/417 contexts for sycophancy, hallucination, and evil.

All selected sycophancy and hallucination extraction prompts remain exposed to the trait-map pool, whereas the selected evil tails have no detected map-content matches. This sensitivity evaluates new content with the existing directions; it is not independent extraction or a map refit. Every mask and bootstrap draw is saved. Independent checks reproduced 768 sampled SciPy correlations to within 2.22×10⁻¹⁶.

## Validation and preservation

All seven frozen layers completed, covering all 13 OOD datasets, three WildChat evaluations, and three map-exposed in-distribution evaluations. All 76 original E1 dataset/readout/layer cells reproduce the historical counts and correlations exactly. An independent SciPy audit recomputed 833 correlations and all 416,500 stored bootstrap correlations, with maximum absolute numerical error 1.11×10⁻¹⁶; 38 additional checks cover observed answers at the actual mapped layers. The focused implementation suite passes 83 tests. The repository-wide workflow checker has unrelated pre-existing failures; scoped checks and commit hooks pass.

Original selected prompts, all five responses, judgments, selection roles, and provenance are preserved for the union of successful quantile diagnostics: 8,492 contexts and 42,460 responses in 28 shards, plus their verified manifest. Missing judgments remain explicit (177 evil, 62 sycophancy, and 6 hallucination responses in this selected union); they are not coerced to zero. Every primary result, direction, map transform, prediction, bootstrap, selection record, and terminal log has a verified immutable artifact home. Three CPU workers completed and were terminated through the artifact-gated dispatcher. The independent monitor delivered its completion notification and was retired after successful teardown.

Science source: `9263556c52acd1bf5fa25478a8a175f51baf05d5`. Verified outputs and independent audits are committed under `eval_results/issue_1739/natural_extremes_20260918/`; figure files are under `figures/issue_1739/natural_extremes_20260918/`. The branch is `codex/1739-natural-extremes-20260918`.

## Archived map-fit diagnostics

These diagnostics use the original recipe’s internal held-out rows; the map is subsequently refitted on the full pool for behavior evaluation. They are not the content-disjoint behavioral test. The identity comparator includes a learned bias; retrieval uses cosine similarity against the full listed held-out candidate pool.

| Behavior | Layer | Held-out R² | Identity + bias R² | Top-1 retrieval | Pool | Chance | Fit/score seconds | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sycophancy | 11 | 0.721 | -0.042 | 33.15% | 6,959 | 0.0144% | 65.9 | 13.45 |
| sycophancy | 16 | 0.765 | -0.087 | 52.23% | 6,959 | 0.0144% | 63.0 | 13.45 |
| hallucination | 23 | 0.642 | -0.184 | 49.52% | 6,959 | 0.0144% | 67.5 | 24.43 |
| hallucination | 24 | 0.624 | -0.205 | 46.98% | 6,959 | 0.0144% | 64.7 | 24.43 |
| hallucination | 27 | 0.682 | -0.283 | 52.08% | 6,959 | 0.0144% | 64.3 | 24.43 |
| evil | 15 | 0.709 | -0.090 | 39.25% | 5,052 | 0.0198% | 47.8 | 19.27 |
| evil | 22 | 0.724 | -0.202 | 37.27% | 5,052 | 0.0198% | 45.0 | 19.27 |
