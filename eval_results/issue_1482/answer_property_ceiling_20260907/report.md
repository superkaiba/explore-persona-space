# Answer-property recoverability and context predictability

Completed 7 September 2026 (Pacific time). Analysis-only continuation of #1482, extending the property coverage of the #2564 pilot. No manuscript changes have been made.

**Answer-property recoverability differs substantially across the paper's feature families.** Identity/disposition features are more linearly recoverable from observed answer vectors than topic features, and coarse Matryoshka features are much more recoverable than fine features. The control confirms that the paper's context-prediction comparisons involve feature families with unequal observed-answer recoverability. Under the specified descriptive adjustment for activity and observed-answer recoverability, the identity-versus-topic context ordering persists, while the abstract-contextual-versus-token-surface and coarse-versus-fine advantages are not preserved. Abstract versus lexical-semantic features retain a small residual ordering. These results support differentiated claims about the property families, rather than a general claim that higher-level properties have a context-prediction advantage beyond their representation in answer vectors.

The adjustment has imperfect balance and changes which feature pairs are compared. In particular, identity-versus-topic concordance remains 0.604 for the observed-answer readout inside the matching bins, compared with 0.664 for context prediction. The residual context association is consequently qualified evidence, not a demonstration of equal recoverability or a causal decomposition.

## What was measured

For each realized answer, the input to the new linear readout is the mean dense answer-token activation, `mean_t h_t`. Its target is the original per-feature mean of nonlinear, per-token SAE activations, `mean_t SAE(h_t)`. This target is neither `SAE(mean_t h_t)` nor a known linear projection of the dense mean. Reading these targets from the dense mean is therefore a substantive test of what survives pooling.

The main analysis uses Qwen2.5-7B-Instruct layer 19 and the original BatchTopK dictionary of **131,072 features**. It preserves the paper's 120,000 training, 2,000 validation, and 20,000 held-out real-chat rows. The existing context-to-SAE map and the new observed-answer-to-SAE map have the same target definitions, split, input dimension, train-only standardization, target centering, and 23-point ridge-penalty search, `logspace(-3, 8, 23)`. A single penalty per readout is chosen by pooled validation squared error, following #1482/#779; features do not receive individually optimized penalties. Selected penalties are 1,000 for observed answers and 3,162.278 for context.

Both maps have defined held-out R² for 121,111 features; 9,961 constant held-out targets are undefined and are never assigned zero. The original paper's additional population mask leaves **120,716 matched features** for the category analysis. The global pooled held-out R² is **0.8369** for the observed-answer map and **0.6531** for the original context map. Their per-feature R² values have Spearman correlation **0.9545**. Pooled R² weights features by target variance and should not be confused with the median per-feature R² below. Some rare targets have very large negative R²; all finite values are retained, and robust medians and rank comparisons avoid interpreting their unstable arithmetic means.

The semantic labels are the paper's existing feature-level labels: topic, identity/disposition, register/style, language, task format, entity, syntax, operation, and abstraction categories. The original logit-footprint groups are also included. These labels categorize SAE features, whose realized activities provide the answer-level targets. They are **not independently judged whole-answer behavioral outcomes**. No new model generations or judge calls were made.

## Recoverability differs by property

All values below are median per-feature held-out R². Counts refer to the original eligible feature population; categories on different labeling axes can overlap.

| Feature category | Features | Observed answer | Context |
|---|---:|---:|---:|
| Identity/disposition | 1,399 | 0.2753 | 0.1393 |
| Topic | 20,676 | 0.0927 | 0.0216 |
| Register/style | 9,136 | 0.1101 | 0.0380 |
| Language | 6,438 | 0.1198 | 0.0480 |
| Task format | 1,724 | 0.0703 | 0.0333 |
| Entity | 5,544 | 0.0405 | 0.0081 |
| Syntax | 71,248 | 0.1182 | 0.0380 |
| Operation | 10,762 | 0.1065 | 0.0322 |
| Token surface | 41,104 | 0.0682 | 0.0155 |
| Lexical semantic | 29,863 | 0.0972 | 0.0272 |
| Abstract contextual | 39,489 | 0.1529 | 0.0540 |
| Logit promoting | 8,048 | 0.2156 | 0.0791 |
| Logit suppressing | 5,031 | 0.0928 | 0.0312 |
| Logit partition | 8,261 | 0.0667 | 0.0171 |

![Median feature recoverability from observed answer vectors and context](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b55a56843afadd655b88522169cfa4b40a3da9a0/figures/issue_1482/answer_property_ceiling_20260907/property_readout_medians.png)

Figure 1. Median held-out feature R² for each labeled family in the original eligible population. Points are descriptive group medians; uncertainty for the specified pairwise comparisons is shown below. [PDF](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b55a56843afadd655b88522169cfa4b40a3da9a0/figures/issue_1482/answer_property_ceiling_20260907/property_readout_medians.pdf).

The full distributions, positive-R² fractions, residual label categories, and counts of unlabeled/unresolved features are in [regular.json](regular.json) and [category_summary.csv](category_summary.csv). Negative R² means that this fitted readout performs worse than the held-out target-mean reference; it does not establish that the property is absent from the representation.

## Which context associations persist under adjustment

The comparison statistic is **concordance**: the probability that a feature in the first named category has higher held-out R² than a feature in the second, counting ties as one half. A value of 0.5 indicates no pairwise ordering. This is not a difference in R² or a ratio of explained variances.

We report raw concordance, concordance within the paper's original activity deciles, and concordance within observed-answer-recoverability deciles nested inside those activity deciles. Activity is the original per-token firing frequency. Only opposite-category pairs in the same cell contribute; cells are weighted by their numbers of eligible pairs. Features carrying both compared labels are excluded from that comparison and counted explicitly. The 95% intervals use 1,000 feature-bootstrap draws within property-by-fixed-cell groups, with shared feature weights across the readout arms. Five and twenty inner bins are prespecified resolution checks.

| First category versus second | Raw context | Activity-adjusted context | Activity + observed-answer adjustment, context (95% interval) |
|---|---:|---:|---:|
| Identity/disposition versus topic | 0.728 | 0.864 | 0.664 [0.647, 0.682] |
| Abstract contextual versus token surface | 0.589 | 0.460 | 0.497 [0.492, 0.501] |
| Abstract contextual versus lexical semantic | 0.571 | 0.518 | 0.518 [0.513, 0.522] |
| Register/style versus topic | 0.574 | 0.648 | 0.626 [0.618, 0.635] |
| Language versus topic | 0.590 | 0.582 | 0.638 [0.629, 0.647] |
| Task format versus topic | 0.541 | 0.662 | 0.636 [0.620, 0.651] |
| Logit promoting versus suppressing | 0.602 | 0.450 | 0.465 [0.453, 0.477] |

![Context-prediction concordance before and after descriptive adjustment](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b55a56843afadd655b88522169cfa4b40a3da9a0/figures/issue_1482/answer_property_ceiling_20260907/property_conditional_concordance.png)

Figure 2. Context-prediction concordance for the first named category versus the second, with 95% conditional feature-bootstrap intervals. The dashed line marks 0.5. Matching changes eligible pairs and sometimes feature support; it does not imply perfect balance. [PDF](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b55a56843afadd655b88522169cfa4b40a3da9a0/figures/issue_1482/answer_property_ceiling_20260907/property_conditional_concordance.pdf).

**Identity versus topic retains a qualified residual context association.** Excluding 187 dual-labeled features leaves 1,212 identity and 20,489 topic features; the primary joint adjustment retains all 1,212 identity and 19,612 topic features. However, observed-answer concordance inside those bins is 0.604, so the groups remain imperfectly balanced. The paired context-minus-observed concordance is **+0.060**, with a conditional feature-bootstrap interval of **[+0.041, +0.081]**. With twenty recoverability bins, observed-answer and context concordances are 0.551 and 0.628. The context point estimate is 0.711 with five bins, 0.664 with ten, and 0.628 with twenty. Its direction persists across these choices; its magnitude depends on resolution and the retained support.

**The abstract-versus-token-surface advantage is not preserved.** Context concordance becomes 0.497 under the joint adjustment, with sensitivity values 0.493 and 0.496 for five and twenty inner bins. All 39,489 abstract-contextual and 41,104 token-surface features retain support in the primary comparison. Abstract versus lexical-semantic features still show a small ordering, 0.518; it would be inaccurate to say that every abstraction contrast disappears.

Register/style, language, and task-format feature comparisons with topic also retain positive descriptive context orderings. Logit-promoting features have a raw advantage over suppressing features, but their ordering changes after activity adjustment and stays below 0.5 after adding observed-answer recoverability. This is an activity-conditioned association, not evidence that a logit footprint causes predictability. Exact overlap counts, support, paired-arm intervals, and resolution checks are in [regular.json](regular.json).

## Matryoshka companion: coarse features are much easier to recover from answers

This arm uses the original **layer-20** Matryoshka token-SAE-mean targets and the original selected panel of **16,384 features**, drawn from the 65,536-feature dictionary. It is not a full-dictionary Matryoshka analysis. The panel contains 1,640 coarse, 6,144 middle, and 8,600 fine features. Same-row layer-20 dense answer means come from the archived #2476 recapture bank. The manuscript's historical Matryoshka context result uses SAE context inputs; we report it separately from a newly fitted dense-context companion with the same dimensionality and fitting budget as the observed-answer arm.

The original split has 22,000 training, 2,000 validation, and 6,000 held-out rows. The input audit found exactly two training rows (656068 and 656678) with 1,024 answer tokens in the original target store but 1,014 in the recapture. Their cause remains unresolved. Both new fits exclude those rows, leaving **21,998 matched training rows**; every validation and held-out row retains exact token-count agreement. The original SAE-context result keeps its original 22,000 training rows. A separate 22,000-row dense-context refit reproduces the archived dense-context per-feature R² with maximum absolute discrepancy **6.31 × 10⁻⁷**, distinguishing this two-row change from numerical or layout errors.

| Tier | Observed answer, median R² | Matched dense context, median R² | Original SAE context, median R² |
|---|---:|---:|---:|
| Coarse | 0.8226 | 0.4388 | 0.4346 |
| Middle | 0.4148 | 0.1810 | 0.1739 |
| Fine | 0.1668 | 0.0903 | 0.0430 |

For the original SAE-context readout, coarse-versus-fine concordance is **0.859** raw, **0.757** within activity deciles, and **0.441 [0.414, 0.468]** after also matching observed-answer recoverability. The matched dense-context companion gives **0.407 [0.380, 0.435]** under the joint adjustment. The primary adjusted comparison retains only **1,018 of 1,640 coarse** and **7,169 of 8,600 fine** features. Original SAE-context concordance is 0.479 with five recoverability bins and 0.413 with twenty. Observed-answer concordance remains 0.553 inside the primary bins. Thus the raw coarse advantage is not preserved in this descriptive adjustment, but incomplete overlap and residual imbalance prevent interpreting the adjusted ordering as a causal reversal across the entire dictionary.

A complementary bootstrap resamples complete held-out answers within the original corpus strata, preserving feature dependencies within each answer. It uses fixed fitted maps and the original tier panel. The observed-answer versus matched dense-context **pooled R²** values are 0.940/0.690 for coarse, 0.701/0.473 for middle, and 0.490/0.339 for fine. The paired coarse-minus-fine difference is 0.450 [0.435, 0.464] for observed answers and 0.351 [0.337, 0.364] for dense context. Their difference, context minus observed, is **−0.099 [−0.109, −0.090]**. This confirms that tier differences are already substantial in observed-answer recoverability under this recipe. It measures an aggregate gap, rather than the conditional feature-ordering statistic. See [matryoshka.json](matryoshka.json) and [matryoshka_row_bootstrap.json](matryoshka_row_bootstrap.json).

## Scope and limitations

- The new arm estimates recoverability **under this matched ridge recipe**, not a strict ceiling over all possible linear decoders. Its single pooled-error-selected penalty favors high-variance targets; no feature-specific tuning or nonlinear readout is claimed.
- Semantic labels describe SAE features rather than independently annotated complete answers. This supplies the missing control for the paper's SAE identity/topic, abstraction, tier, language/style/format, and logit-footprint analyses. It does not add a broad-topic classifier with independently labeled questions, realized single-word controls, or new refusal, harmful-compliance, sycophancy, hallucination, or deception outcomes. Existing behavioral probes and the #2564 instruction-condition pilot remain separate evidence.
- The feature bootstrap conditions on this bank, labels, fitted maps, and fixed matching cells. SAE features can be correlated, so those intervals can be too narrow; they do not include model-fitting uncertainty or support confirmatory significance claims across the several comparisons. The Matryoshka answer bootstrap has a different sampling unit and still conditions on trained maps.
- Binned adjustment does not guarantee equal activity or recoverability within bins. It changes pair weights and sometimes support, with substantial Matryoshka support loss. Recoverability and context R² are measured on the same held-out bank, so estimation error can affect their association. These are descriptive controls, not causal mediation estimates or proof that remaining differences are independent of recoverability.
- One model, one bank-specific layer per SAE family, and the original corpus-mixture split were evaluated. This does not establish generalization across models, layers, novel topics, or out-of-distribution corpora.
- The main context comparator is the paper's direct context-to-SAE map. No new frozen predicted-dense-answer-to-property arm was fitted. The experiment also does not test whether raw Euclidean distance reflects behavioral similarity.

## Validation and durable artifacts

All fits and analyses completed with fresh completion records. The main input join passed all-row dense-input parity; sparse target parity was checked against the prior bank. The main target-variance discrepancy against the context comparator is at most 4.08 × 10⁻¹⁰. All 32 held-out prediction blocks were rehashed and checked against their block identities, selected penalty, and saved fit. Recomputing eight sampled rows across every feature from saved weights gives maximum absolute prediction discrepancy **0.0**. Fourteen focused numerical, checkpoint, matching, and statistical tests pass, and the repository's payload-specific inline lint gates pass. Independent code and statistical review passed; final interpretation review is recorded alongside this report.

As auxiliary checks on the Matryoshka mappings, raw Euclidean/cosine top-1 retrieval among all 6,000 held-out feature vectors is 1.000/1.000 for observed answers and 0.489/0.475 for the matched dense-context arm (chance 1/6,000). These checks are recorded in the fit summaries inside [matryoshka.json](matryoshka.json).

As an auxiliary mapping check, observed-answer-to-SAE retrieval among 1,000 seeded held-out candidate feature vectors gives raw Euclidean top-1 accuracy 0.990 and cosine top-1 accuracy 0.995, against chance 0.001. This is distinct from the manuscript's context retrieval and its whitening/CSLS protocol. An identity-plus-learned-bias baseline is inapplicable because dense inputs have 3,584 coordinates and the SAE targets have 131,072 (regular) or 16,384 (Matryoshka) coordinates. Details are in [regular_verification.json](regular_verification.json).

The executable methods and frozen plan are committed on `codex/issue1482-answer-property-ceiling-20260907`: `scripts/issue1482_answer_property_ceiling.py`, `scripts/issue1482_answer_property_matryoshka.py`, `scripts/issue1482_answer_property_analysis.py`, `scripts/issue1482_answer_property_figures.py`, `scripts/issue1482_answer_property_verify.py`, and `docs/methodology/issue_1482_answer_property_ceiling_plan_20260907.md`. The analysis JSONs include exact source hashes and provenance. Raw inputs, fit factors/weights, selected-penalty curves, per-feature outputs, held-out predictions, feature tables, and bootstrap draws are archived on Hugging Face; pinned revisions and byte/hash verification manifests are linked below.

- [Verified input and Matryoshka archive](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/901771c38b95e1af7b20fb5cc2dcfb2f1781b3b8/issue1482_answer_property_ceiling_20260907/analysis_tensors): 29 files, 13,451,693,835 bytes. [Verification manifest](input_upload_verified.json). The archive's `array_layout.json` specifies restoration names for raw numeric `.bin` files.
- [Verified full-width observed-answer fit and predictions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/99b3185de062878eb90a79cf7e551b47cd6e2cf8/issue1482_answer_property_ceiling_20260907/analysis_tensors/observed_answer): 165 files, 14,355,512,732 bytes. [Verification manifest](readout_upload_verified.json).
- [Verified analysis inputs, feature tables, and bootstrap draws](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/52ca7182f7b4699c5d776c03bac24becab5ca97e/issue1482_answer_property_ceiling_20260907/analysis_tensors/analysis): 40 files, 43,363,615 bytes. [Verification manifest](analysis_upload_verified.json).
- [Original Matryoshka source store at its pinned revision](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8b82eb975326774003512db1e7e542491edcd056/issue1482_error_analysis/analysis_tensors/matryoshka_tier/store). The input archive includes the exact downloaded-source inventory.
- [Frozen plan](../../../docs/methodology/issue_1482_answer_property_ceiling_plan_20260907.md), [analysis code](../../../scripts/issue1482_answer_property_analysis.py), [prediction-verification code](../../../scripts/issue1482_answer_property_verify.py), and [independent interpretation review](interpretation_review.md).

All 234 archived files were checked for exact byte size and content hash at the listed revisions. Existing source banks and local outputs were retained.
