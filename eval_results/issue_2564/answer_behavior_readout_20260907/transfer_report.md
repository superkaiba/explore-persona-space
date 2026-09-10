# Adjusting property prediction for observed-answer readability

Completed 2026-09-09 (US Pacific). This is the user-requested same-probe follow-up under [plan v16](transfer_plan.md). The prior observed-answer readout remains in [the original report](report.md). No manuscript or bibliography edits were made.

## What this supports

The matched experiment retains different fractions of the observed-answer readout improvement across properties: about 20% for assertive confidence and 30–42% for the other judged properties. Warmth and confidence have relatively similar observed-answer readout skill (0.723 and 0.684), but their retained fractions are 41.5% and 19.9%. Thus a single common fraction of observed readability does not describe these point estimates. This is evidence of unequal predictive performance after a within-property adjustment, not a causal decomposition of why properties differ.

Expressed voice retains 38.8% and topic 30.8%. This is compatible with an advantage for voice in this cohort after adjusting for its observed readout baseline. It does not establish a universal high-level/low-level hierarchy: language is 39.2%, and several surface measures overlap the behavioral properties. Voice is the archetype expressed in the answer, not the source speaker identity.

**Scope:** these are new, small-cohort, grouped out-of-fold maps. The proposed frozen million-example #779 map failed the pre-outcome compatibility audit: #2054 uses `hidden_states[19]` and content-only pooling, whereas #779 uses decoder block 19 (`hidden_states[20]`) and includes assistant closing tokens. It was not applied. The present result does not establish how much of the paper’s #779/SAE ranking survives this correction. [Audit](transfer_rejected_map_audit.json).

## Primary comparison

For the same fixed observed-answer probe, let E_A be error on actual answer vectors, E_M error on predicted answer vectors, and E_0 error of the training-fold mean/vote prior. We report `retention = (E_0 − E_M)/(E_0 − E_A)`. A value of 100% matches the observed-answer probe’s improvement over the prior; 0% matches the prior. This is a fraction of error reduction, not a percentage of information preserved or accuracy retained. Negative values and values above 100% are not clipped. Nonpositive denominators are undefined.

Errors are scalar MSE for graded/mechanical targets and mean squared error over vote-fraction columns for categorical targets. “Skill” below is `1 − E/E_0`. Categorical inventories, class support and annotation reliability still differ, so normalization does not make targets interchangeable.

| Property | Answers | Observed skill | Mapped skill | Retained improvement %, 95% CI | Defined draws |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warmth | 1876 | 0.723 | 0.300 | 41.5 [36.4, 46.5] | 2000/2000 |
| Assertive confidence | 1963 | 0.684 | 0.136 | 19.9 [15.7, 23.6] | 2000/2000 |
| Formality | 1864 | 0.825 | 0.294 | 35.7 [31.8, 39.5] | 2000/2000 |
| Expressed voice | 2048 | 0.540 | 0.210 | 38.8 [35.9, 42.5] | 2000/2000 |
| Topic | 2048 | 0.400 | 0.123 | 30.8 [27.8, 33.9] | 2000/2000 |
| Language | 2048 | 0.739 | 0.290 | 39.2 [33.2, 45.9] | 2000/2000 |
| Format | 2048 | 0.607 | 0.184 | 30.4 [26.0, 34.4] | 2000/2000 |

All seven primary retention denominators are positive in all 2,000 bootstrap draws. Intervals are paired duplicate-connected-question-component bootstraps of fixed predictions. They condition on fitted maps/probes and the correlated three-pass labels; they omit retraining and independent-rater uncertainty. No selected-pair significance test was performed.

### Interpretable companion metrics

| Property | Metric | Observed | Mapped |
| --- | --- | ---: | ---: |
| Warmth | R² | 0.723 | 0.299 |
| Assertive confidence | R² | 0.683 | 0.134 |
| Formality | R² | 0.824 | 0.293 |
| Expressed voice | Balanced accuracy | 0.500 | 0.292 |
| Topic | Balanced accuracy | 0.545 | 0.227 |
| Language | Balanced accuracy | 0.722 | 0.135 |
| Format | Balanced accuracy | 0.468 | 0.165 |

Balanced accuracy excludes 26 voice and 15 topic vote ties; the primary vote-MSE comparison retains them. Language and format BA intervals have 1,891 and 1,699 defined draws, respectively, because some resamples omit rare classes. The retention intervals above remain defined in all 2,000 draws. Full companion intervals, class support, component errors, identity-plus-bias controls and probe-output agreement are in [structured results](transfer_results.json) and [the flat table](transfer_properties.csv).

### Mechanical controls

| Property | Answers | Observed skill | Mapped skill | Retained improvement %, 95% CI | Defined draws |
| --- | ---: | ---: | ---: | ---: | ---: |
| Log character count | 2048 | 0.927 | 0.235 | 25.3 [22.2, 29.7] | 2000/2000 |
| Log whitespace word count | 2048 | 0.885 | 0.310 | 35.0 [30.8, 40.3] | 2000/2000 |
| English first-person rate | 2048 | 0.688 | 0.229 | 33.3 [28.7, 38.1] | 2000/2000 |
| English second-person rate | 2048 | 0.584 | 0.147 | 25.2 [19.6, 30.7] | 2000/2000 |
| English negation rate | 2048 | 0.250 | 0.032 | 12.9 [6.3, 25.1] | 2000/2000 |
| Question-mark rate | 2048 | 0.673 | 0.103 | 15.3 [11.2, 20.9] | 2000/2000 |
| Newline rate | 2048 | Undefined | Undefined | Undefined | 0/2000 |
| Code-fence count | 2048 | 0.110 | -0.009 | -7.8 [-66.1, 1.5] | 1719/2000 |

Newline rate is constant, so all 2,000 retention draws are undefined. Code fences are sparse: 281 draws have nonpositive observed improvement; its reported interval is conditional on the remaining 1,719. Neither is evidence for a reliable ordering.

## Representation diagnostics

| Map | Held-out trace R² | Euclidean top-1 | Cosine top-1 |
| --- | ---: | ---: | ---: |
| Matched ridge | 0.192 | 5.86% | 9.13% |
| Identity + training-fold bias | -0.908 | 11.08% | 15.67% |

Retrieval uses each held-out fold as its entire candidate pool: 412, 412, 408, 408 and 408 answers, with query-weighted chance 0.244%. These are raw Euclidean/cosine diagnostics with the existing tolerance-based mid-rank tie rule; they do not use the paper’s whitening/CSLS retrieval recipe. Identity-plus-bias retrieves better despite worse reconstruction R². Reconstruction and retrieval therefore should not be treated as interchangeable evidence.

## Method and verification

The cohort contains 2,048 generated answers from 512 questions in four source-framing variants, grouped into 438 duplicate-connected components. All five folds are unchanged. Each map is trained only on the other four folds (1,636 or 1,640 answers); every target’s observed-answer probe uses its original available training rows, selected penalty, mean and scale. Map training uses vectors without property labels. The same reconstructed probe is applied to actual, mapped and identity-plus-bias vectors. Direct context, framing and length probes are retained as auxiliary references, not confused with same-probe transfer.

Maps reuse `issue2054_fits._ridge_gcv_fit_predict`: train-only context standardization, answer centering, a 13-point 0.01–10,000 GCV lambda grid and a 0.9 degrees-of-freedom cap. All five folds selected the upper grid endpoint 10,000, with effective degrees of freedom 210.7–213.2. This boundary choice and the small sample are limitations; the experiment does not establish an optimal map or a decoding ceiling. No map or probe was retuned using held-out behavior outcomes.

The five maps and 20 probe bundles completed; all 15 requested targets were processed. Maximum discrepancy from the immutable original observed-answer predictions was 1.85e-13. Numerical tests check established-solver parity, unchanged held-out predictions when held-out answer targets are altered, group leakage rejection, validated checkpoint reuse, shared-probe normalization, missingness and ratio behavior. Seven transfer tests, four matched-map tests and nine original readout tests passed across focused runs. Production analysis took 100.2 seconds on CPU with eight BLAS threads, zero GPU use, zero new generations and zero new judgments; exit status was 0 at 2026-09-10T06:16:27Z.

Implementation at execution: `9152b22833980fae6689c806de91b75524fba1db`, clean worktree. Source completion SHA256: `33cb4623c0b230ebcb82193a47ab02d4a8056941dc6b0d350e5057cb260f2df8`. Exact fold IDs, selected parameters, prediction hashes and bootstrap draws are retained. Independent audit and archive receipts accompany this report.

## Limits relevant to Dan’s comment

This operational normalization adjusts for finite-sample linear readout performance. It cannot separate inherent behavior unpredictability, map shrinkage/estimation error, property direction geometry and judge noise. The answer-only voice/topic/style labels do not newly measure harmful compliance, sycophancy, deception or factual hallucination. There was no human validation, and the three Codex rating passes share a base model; they are not independent annotator draws.

The source bank also contains a documented narrative boundary artifact: 468/2,048 answers open with a digit. The cohort was not filtered after seeing results. Its distribution, labels and small maps limit extrapolation to standard chat answers or the paper’s trained map. [Source audit and original limitations](report.md). The appropriate response to Dan is that observed readability is a real confound worth controlling, and this matched diagnostic shows remaining variation after that adjustment; the paper-specific claim still needs compatible vectors and map artifacts.

## Reproducible artifacts

[Independent result audit](transfer_result_review.json), [validation record](transfer_review.json), and [verified archive](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/7ab779be98292a7ded7b8c20a4b3b63096ecccc4/issue2564_minpair/answer_behavior_readout_20260907/matched_transfer_20260909) (74 files; exact path, size and content-hash verification). Original vectors and three-pass ratings are retained in the pinned [prior input archive](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/7640e8b6a4656d42e488f938f7ff57ca04b08729/issue2564_minpair/answer_behavior_readout_20260907/three_pass_final_20260909_tree_v2). Archive receipts and per-file hashes are available [here](transfer_archive_verified.json) and [here](transfer_archive_file_hashes.json). The archived report is the pre-publication snapshot; this paragraph adds the archive receipt after verification.
