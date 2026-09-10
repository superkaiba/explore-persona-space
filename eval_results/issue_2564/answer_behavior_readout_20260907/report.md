# Actual answer properties are linearly readable, with uneven readout performance

Completed 2026-09-09 (US Pacific), under the approved [three-pass amendment](three_pass_plan.md) to issue #2564. This is the actual expressed-property follow-up. It is separate from the earlier requested-condition and SAE-category analyses. No manuscript text has been changed.

## What this says about Dan's concern

Dan asked whether the linear representation captures different properties to different degrees before we attempt to predict those properties from context. On this cohort, the observed answer vectors support held-out linear readouts of all seven judged properties, and each outperforms its matched direct context-vector readout. The measured scores vary substantially across targets. This supplies the missing observed-answer reference for interpreting property prediction, rather than assuming all properties are equally readable.

The experiment does not establish a universal ordering of high-level versus low-level properties. Warmth, confidence and formality use graded targets; voice, topic, language and format use different categorical inventories and class balances. Annotation consistency and label support also differ. These are finite-sample ridge-readout results, not intrinsic decoding ceilings. A difference between topic and voice scores alone cannot isolate representational difficulty.

Here, “persona” means the voice expressed in the answer, such as expert guide or personal peer. It does not mean recovering the source speaker's identity. Topic means the dominant subject of the generated answer. Confidence means assertive expression, not factual correctness or calibrated uncertainty. Harmful compliance, sycophancy, deception and factual hallucination are not new targets in this run.

## Main results

Models use matched held-out questions and answer availability. Brackets give 95% connected-question-component bootstrap intervals for fixed out-of-fold predictions. They condition on the labels and fitted models; they do not include retraining or independent-annotator uncertainty. All tables retain the full-width primary model even where a PCA sensitivity scores higher.

### Graded expression: held-out R²

| Property | Scored answers | Answer R² [95% CI] | Context R² | Answer − context [95% CI] |
| --- | --- | --- | --- | --- |
| Warmth | 1876 | 0.723 [0.695, 0.747] | 0.300 | 0.422 [0.384, 0.461] |
| Assertive confidence | 1963 | 0.683 [0.650, 0.712] | 0.126 | 0.557 [0.525, 0.591] |
| Formality | 1864 | 0.824 [0.807, 0.841] | 0.320 | 0.504 [0.477, 0.532] |

All graded primary intervals use 2,000/2,000 defined bootstrap draws.

### Categorical expression: balanced accuracy

Balanced accuracy is mean recall across represented classes, not ordinary accuracy. The readout is trained on three-vote fractions. Unique-mode classification excludes 26 voice ties and 15 topic ties, leaving 2,022 and 2,033 scored answers; language and format each have 2,048. All 2,048 vote targets are used for fitting and vote-error metrics.

| Property | Classes | Answer BA [95% CI] | Context BA | Answer − context [95% CI] | Defined draws / 2,000 |
| --- | --- | --- | --- | --- | --- |
| Expressed voice | 8 | 0.500 [0.483, 0.515] | 0.304 | 0.196 [0.169, 0.214] | 2000 |
| Topic | 13 | 0.545 [0.516, 0.569] | 0.230 | 0.315 [0.289, 0.340] | 2000 |
| Language | 12 | 0.722 [0.663, 0.767] | 0.139 | 0.584 [0.524, 0.629] | 1891 |
| Format | 7 | 0.468 [0.410, 0.526] | 0.174 | 0.295 [0.236, 0.347] | 1699 |

Rare classes make some language and format bootstrap replicates undefined; their intervals use only the defined replicates. These counts apply to the answer, context and paired-difference intervals shown. Full class support appears in [class_support.csv](class_support.csv).

An alternative within-target metric is vote-MSE skill against the training-prior predictor, defined as `1 − MSE(readout)/MSE(prior)`. It retains disagreement and ties. This normalization still does not make unrelated label inventories interchangeable.

| Property | Answer vote-MSE skill [95% CI] |
| --- | --- |
| Expressed voice | 0.540 [0.505, 0.579] |
| Topic | 0.400 [0.376, 0.419] |
| Language | 0.739 [0.715, 0.759] |
| Format | 0.607 [0.545, 0.657] |

All four vote-MSE skill intervals use 2,000/2,000 defined draws.

### Baselines and restricted shuffle

The columns below use each row's primary metric: R² for graded expression, balanced accuracy for categorical targets. Framing uses training means or vote priors for the four source framing variants. Length uses a linear readout of log answer-character count.

| Property | Metric | Framing | Length | Answer | Shuffled-answer control |
| --- | --- | --- | --- | --- | --- |
| Warmth | R² | 0.098 | 0.015 | 0.723 | 0.033 |
| Assertive confidence | R² | 0.058 | -0.002 | 0.683 | 0.039 |
| Formality | R² | 0.179 | 0.129 | 0.824 | 0.108 |
| Expressed voice | BA | 0.255 | 0.236 | 0.500 | 0.147 |
| Topic | BA | 0.091 | 0.121 | 0.545 | 0.103 |
| Language | BA | 0.083 | 0.117 | 0.722 | 0.089 |
| Format | BA | 0.143 | 0.163 | 0.468 | 0.148 |

The single shuffle preserves connected component bundles, framing positions and availability patterns, and repeats tuning. Only 1,696–1,720 of 2,048 rows move, depending on target; unchanged or unexchangeable groups remain explicit in [run_result.json](run_result.json). This is a structured sanity control, not a zero-signal reference or a permutation p-value. It can retain framing predictability.

### Mechanical surface controls

These are deterministic measurements of the same recorded answer text. English pronoun and negation rates are limited lexical controls, not comprehensive syntax.

| Target | Answer R² [95% CI] | Context R² | Defined draws / 2,000 |
| --- | --- | --- | --- |
| Log character count | 0.925 [0.894, 0.945] | 0.222 | 2000 |
| Log whitespace word count | 0.884 [0.864, 0.900] | 0.310 | 2000 |
| English first person rate | 0.688 [0.619, 0.748] | 0.223 | 2000 |
| English second person rate | 0.584 [0.521, 0.633] | 0.141 | 2000 |
| English negation rate | 0.249 [0.125, 0.347] | 0.025 | 2000 |
| Question mark rate | 0.672 [0.617, 0.719] | 0.099 | 2000 |
| Newline rate | Undefined | Undefined | 0 |
| Code fence count | 0.108 [-0.090, 0.422] | 0.000 | 1984 |

Newline rate is constant in this cohort, so its R² is undefined and supplies no evidence about readability. Code-fence count is sparse, and its interval includes zero. Surface controls therefore do not uniformly succeed; the mixed pattern also cautions against a simple high-level/low-level hierarchy.

### Prespecified sensitivities

Training-only PCA is fitted separately within every inner and outer training split. The single-voice subset excludes answers judged by a majority to contain multiple voices or substantial narration; uncapped excludes the two generation-length-capped answers. Subset diagnostics only evaluate existing out-of-fold predictions, without refitting.

| Property | Metric | Full width | PCA256 | PCA512 | Single voice (n) | Uncapped (n) |
| --- | --- | --- | --- | --- | --- | --- |
| Warmth | R² | 0.723 | 0.716 | 0.724 | 0.710 (1508) | 0.722 (1875) |
| Assertive confidence | R² | 0.683 | 0.645 | 0.664 | 0.697 (1597) | 0.683 (1962) |
| Formality | R² | 0.824 | 0.795 | 0.817 | 0.818 (1496) | 0.824 (1863) |
| Expressed voice | BA | 0.500 | 0.515 | 0.530 | 0.500 (1680) | 0.500 (2046) |
| Topic | BA | 0.545 | 0.484 | 0.522 | 0.555 (1680) | 0.545 (2046) |
| Language | BA | 0.722 | 0.534 | 0.658 | 0.733 (1680) | 0.723 (2046) |
| Format | BA | 0.468 | 0.419 | 0.449 | 0.433 (1680) | 0.468 (2046) |

For the prespecified diagnostic retaining unique-mode classes supported by at least 20 connected groups, answer BA is shown below. Its reduced class inventory and changed evaluation population prevent treating improvement as a new overall result. Intervals and all other subset metrics, including undefined-draw counts, are in the structured result.

| Property | Answers | Answer BA [95% CI] | Defined draws / 2,000 |
| --- | --- | --- | --- |
| Expressed voice | 2012 | 0.572 [0.552, 0.588] | 2000 |
| Topic | 1995 | 0.644 [0.610, 0.673] | 2000 |
| Language | 1936 | 0.822 [0.777, 0.856] | 2000 |
| Format | 2029 | 0.680 [0.633, 0.719] | 2000 |

## Measurement quality and coverage

The revised scope uses exactly 2,048 answers × seven properties × three passes = 43,008 valid ratings, the first three complete chronological passes selected uniformly before fitting. The original five-pass main wave remains incomplete: 44,288 ratings were imported from 173 completed packets out of the original 71,680 planned ratings. The additional 1,280 fourth-pass ratings are preserved and excluded; interrupted fourth-pass attempts are historical evidence only. No new answer judgments, generations or GPU runs were used for this reduction. The original 32-answer, five-pass pilot and acceptance record are retained.

Each answer/property has three distinct fresh judge contexts, but they share a base model and uncontrolled sampling. Context restrictions were instruction-enforced, not a technical tool-access sandbox. High repeat agreement establishes descriptive consistency only; there is no human or independent-model validation.

| Graded property | At least one assessable | All three assessable | Mean within-answer SD (0–100) |
| --- | --- | --- | --- |
| Warmth | 1876 | 1863 | 2.02 |
| Assertive confidence | 1963 | 1807 | 1.98 |
| Formality | 1864 | 1855 | 1.49 |

Confidence has 156 answers with disagreement over assessability. Graded targets average available scores and remain missing if all three judges mark them unassessable. Within-answer SD is not an independent measurement-error estimate.

| Categorical property | Pairwise repeat agreement |
| --- | --- |
| Expressed voice | 0.888 |
| Topic | 0.902 |
| Language | 0.994 |
| Format | 0.981 |

## Methodology and scope limits

The reused source is on-policy Qwen2.5-7B-Instruct generation from issue #2054, captured at `hiddens[19]`. Each 3,584-dimensional observed answer vector averages the exact recorded generated span. The bank has 512 questions under four speaker-framing variants and 438 connected question components after joining exact duplicate answers. Five frozen outer folds keep question/duplicate components together. The 32 pilot answers are excluded from the main fit.

Full-width linear ridge uses training-only feature standardization and intercepts. Three grouped inner folds select a penalty per target from `0.1, 1, 10, 100, 1000, 10000, 100000, 1000000`, reusing the validated #2564 recipe. The same targets, availability masks and folds are used for answer and context readouts. PCA256/PCA512 are prespecified sensitivities, not replacements selected by test performance. The seed is 2564. Uncertainty resamples the 438 connected components 2,000 times, keeping paired model predictions together and retaining undefined metrics as missing. This run measures direct property probes; it does not fit a new context-to-answer activation map.

This is a selected narrative speaker-framing population, not unrestricted conversational answers. The full generated span may contain narration and multiple speakers. The source bare-label boundary artifact favors digit openings: 468/2,048 answers (22.9%) begin with a digit, and 201 have at most 20 characters. These spans are retained; their contribution to predictability is not separately identified. The frozen text/vector joins and capture paths were checked, but historical capture-to-text SHA sidecars are unavailable. These limits constrain generalization and measurement validity.

Readout quality is influenced by target variance, class support, regularization, sample size and correlated judge definitions. It does not quantify every semantic property in the representation, demonstrate causal use of a property by the model, or validate raw Euclidean distance against behavioral difference. Those stronger claims remain outside the evidence.

## Reproducibility and review

The analysis completed with exit code 0 at 2026-09-10 01:46:28 UTC. The five-fold fit and summary completion records share a fingerprint; the summary SHA256 matches the completion record. The three-pass implementation passed 38 focused tests plus lint/format checks and independent code review, documented in [three_pass_review.md](three_pass_review.md). An independent [result audit](three_pass_result_review.md) also passed, checking the finished report against the raw-derived labels, completion records and metrics.

Code at execution: [`2bab70901e669b8cd1e76628e8b4f76b967d1e84`](https://github.com/superkaiba/explore-persona-space/tree/2bab70901e669b8cd1e76628e8b4f76b967d1e84). The [three-pass label adapter](../../../scripts/issue2564_three_pass_labels.py) validates the original raw packets and accepted pilot before exact three-pass selection. The [readout script](../../../scripts/issue2564_answer_behavior_readout.py) uses a separate provider, labels directory and output namespace. The archive contains prepared vectors and rows, rubrics, raw judgments and requests, source ledgers, derived labels, all fold checkpoints, selected parameters, out-of-fold predictions, bootstrap draws, execution logs and code snapshots. The [complete archive](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/7640e8b6a4656d42e488f938f7ff57ca04b08729/issue2564_minpair/answer_behavior_readout_20260907/three_pass_final_20260909_tree_v2) contains 1,218 files (281,718,049 bytes), verified by exact path, size and content hash at revision `7640e8b6a4656d42e488f938f7ff57ca04b08729`; see the [verification receipt](three_pass_final_verified.json). An initial flattened upload failed validation and is marked invalid; the verified archive preserves the complete nested layout. The archived report is the pre-publication snapshot; this Git report includes the final review and receipt links.

Machine-readable results: [run_result.json](run_result.json), [primary_metrics.csv](primary_metrics.csv), [class_support.csv](class_support.csv). The existing five-pass continuation instructions are superseded by this completed three-pass report; do not resume judging from historical status notes.

## Point to retain for the later message to Dan

We checked properties actually expressed in the generated answers using held-out linear probes of the observed answer vectors. Warmth, assertive confidence, formality, expressed voice, topic, language and format are readable on this cohort, with uneven performance. This gives us a property-specific observed-answer reference when discussing context predictability. We should avoid claiming a general high-level/low-level hierarchy because these decoding tasks have different target definitions, supports and measurement quality. No Slack message has been sent and no manuscript wording has been changed.
