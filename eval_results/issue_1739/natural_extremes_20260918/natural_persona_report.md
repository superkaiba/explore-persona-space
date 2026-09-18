Natural-prompt extraction uses archived judgments to select the high/low candidate prompts from archived on-policy responses; this is not unlabeled extraction. The experiment makes no new model or judge calls. Evil trait denotes the historical rubric; these are not newly graded compliance labels.

Intervals are the archived 95% paired group-bootstrap intervals, conditional on the extracted directions, map, layers, generations, and labels. OOD datasets receive equal weight. ID evaluation remains exposed to map fitting. N/A means unavailable.

| Behavior | Natural mapped minus | Δ Spearman ρ | 95% interval | OOD coverage |
|---|---|---:|---|---|
| Sycophancy | E1 mapped answer | -0.038 | [-0.082, 0.006] | 6/6 |
| Sycophancy | Natural context native | 0.104 | [0.033, 0.163] | 6/6 |
| Sycophancy | Natural answer on context | 0.321 | [0.267, 0.369] | 6/6 |
| Hallucination | E1 mapped answer | -0.091 | [-0.129, -0.056] | 2/2 |
| Hallucination | Natural context native | 0.005 | [-0.024, 0.035] | 2/2 |
| Hallucination | Natural answer on context | 0.137 | [0.111, 0.164] | 2/2 |
| Evil trait | E1 mapped answer | -0.064 | [-0.097, -0.031] | 5/5 |
| Evil trait | Natural context native | 0.225 | [0.192, 0.260] | 5/5 |
| Evil trait | Natural answer on context | 0.148 | [0.112, 0.189] | 5/5 |

| Behavior | Tail fraction | Answer on context | Context native | Mapped answer | Observed answer |
|---|---|---:|---:|---:|---:|
| Sycophancy | 1% | -0.149 | 0.068 | 0.172 | 0.313 |
| Sycophancy | 5% | -0.155 | 0.154 | 0.212 | 0.336 |
| Sycophancy | 10% | -0.121 | 0.265 | 0.220 | 0.358 |
| Hallucination | 1% | -0.014 | 0.118 | 0.123 | 0.169 |
| Hallucination | 5% | -0.023 | 0.141 | 0.185 | 0.216 |
| Hallucination | 10% | -0.013 | 0.287 | 0.187 | 0.214 |
| Evil trait | 1% | -0.003 | -0.080 | 0.145 | 0.203 |
| Evil trait | 5% | -0.066 | -0.089 | 0.038 | -0.006 |
| Evil trait | 10% | 0.024 | -0.056 | 0.115 | 0.074 |

Observed-answer checks below use the mapped readout's layer. For hallucination this is layer 23, while the displayed observed-answer reference uses layer 27. Positive point estimates alone do not establish a reliable trait direction.

| Behavior | Dataset | Mapped layer | Observed-answer ρ at mapped layer | Displayed observed-answer ρ |
|---|---|---:|---:|---:|
| Sycophancy | aita | 11 | 0.555 | 0.555 |
| Sycophancy | heldin_train | 11 | 0.535 | 0.535 |
| Sycophancy | sycoans | 11 | 0.339 | 0.339 |
| Sycophancy | sycoays | 11 | 0.278 | 0.278 |
| Sycophancy | sycofb | 11 | 0.244 | 0.244 |
| Sycophancy | sycomim | 11 | 0.029 | 0.029 |
| Sycophancy | sycomwe | 11 | 0.434 | 0.434 |
| Sycophancy | wildchat_rung | 11 | 0.347 | 0.347 |
| Hallucination | heldin_train | 23 | 0.428 | 0.412 |
| Hallucination | nqopen | 23 | 0.317 | 0.278 |
| Hallucination | simpleqa | 23 | -0.069 | 0.059 |
| Hallucination | wildchat_rung | 23 | 0.128 | 0.303 |
| Evil trait | evil_mhj | 22 | 0.218 | 0.218 |
| Evil trait | evil_pair | 22 | 0.113 | 0.113 |
| Evil trait | evil_tomgibbs | 22 | 0.535 | 0.535 |
| Evil trait | heldin_train | 22 | 0.416 | 0.416 |
| Evil trait | hhrt | 22 | 0.082 | 0.082 |
| Evil trait | toxicchat | 22 | 0.068 | 0.068 |
| Evil trait | wildchat_rung | 22 | 0.128 | 0.128 |

Hallucination / SimpleQA has a nonpositive observed-answer correlation at the actual mapped layer 23 (ρ=-0.069). This dataset does not show a positively predictive answer direction at that layer; a reference at another layer does not establish that validity.

Primary extraction support and held-out-response gaps below are in each archived rubric's native units. Small tails, zero held-out gaps, and tied low-score pools limit the interpretation of a quantile contrast as reliable extreme trait elicitation.

| Behavior | Held-out dataset | Candidates | Prompts/tail | High−low score | 3→2 gap | 2→3 gap |
|---|---|---:|---:|---:|---:|---:|
| Sycophancy | Social advice | 8633 | 86 | 74.173 | 69.421 | 65.006 |
| Sycophancy | Answer opinion | 8633 | 86 | 74.173 | 68.119 | 65.657 |
| Sycophancy | Are you sure? | 8633 | 86 | 74.173 | 69.670 | 65.945 |
| Sycophancy | Feedback | 8633 | 86 | 74.173 | 69.285 | 64.280 |
| Sycophancy | Mimicry | 8633 | 86 | 74.173 | 68.663 | 65.675 |
| Sycophancy | MWE | 8633 | 86 | 74.173 | 69.547 | 65.327 |
| Hallucination | NQOpen | 10407 | 104 | 1.000 | 0.880 | 0.833 |
| Hallucination | SimpleQA | 10407 | 104 | 1.000 | 0.837 | 0.821 |
| Evil trait | HHRT | 224 | 2 | 24.000 | 17.917 | 0.000 |
| Evil trait | ToxicChat | 1173 | 11 | 8.103 | 6.697 | 2.576 |
| Evil trait | MHJ | 1405 | 14 | 13.041 | 11.556 | 7.607 |
| Evil trait | PAIR | 1405 | 14 | 13.041 | 11.556 | 7.607 |
| Evil trait | Tom Gibbs | 1405 | 14 | 13.041 | 11.556 | 7.607 |

Archived E1 parity records cover 76 dataset/readout cells at frozen layers, with matching row counts and maximum absolute correlation error 0 against their archived reference values.

Extraction counts, native-score tail means, both held-out-response reliability orientations, and group-half/tie direction cosines are preserved verbatim in `natural_persona_report.json`. Optional diagnostics absent from the input are recorded as absent.

- [natural_persona_ood](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_ood.pdf)
- [natural_persona_sycophancy_q01_s0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_sycophancy_q01_s0.pdf)
- [natural_persona_sycophancy_q05_s0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_sycophancy_q05_s0.pdf)
- [natural_persona_sycophancy_q10_s0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_sycophancy_q10_s0.pdf)
- [natural_persona_sycophancy_q01_complete](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_sycophancy_q01_complete.pdf)
- [natural_persona_sycophancy_endpoints](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_sycophancy_endpoints.pdf)
- [natural_persona_sycophancy_e2](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_sycophancy_e2.pdf)
- [natural_persona_sycophancy_e2p](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_sycophancy_e2p.pdf)
- [natural_persona_sycophancy_q01_s1](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_sycophancy_q01_s1.pdf)
- [natural_persona_sycophancy_q01_s2](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_sycophancy_q01_s2.pdf)
- [natural_persona_sycophancy_q01_s3](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_sycophancy_q01_s3.pdf)
- [natural_persona_sycophancy_q01_s4](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_sycophancy_q01_s4.pdf)
- [natural_persona_hallucination_q01_s0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_hallucination_q01_s0.pdf)
- [natural_persona_hallucination_q05_s0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_hallucination_q05_s0.pdf)
- [natural_persona_hallucination_q10_s0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_hallucination_q10_s0.pdf)
- [natural_persona_hallucination_q01_complete](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_hallucination_q01_complete.pdf)
- [natural_persona_hallucination_endpoints](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_hallucination_endpoints.pdf)
- [natural_persona_hallucination_e2](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_hallucination_e2.pdf)
- [natural_persona_hallucination_e2p](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_hallucination_e2p.pdf)
- [natural_persona_hallucination_q01_s1](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_hallucination_q01_s1.pdf)
- [natural_persona_hallucination_q01_s2](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_hallucination_q01_s2.pdf)
- [natural_persona_hallucination_q01_s3](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_hallucination_q01_s3.pdf)
- [natural_persona_hallucination_q01_s4](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_hallucination_q01_s4.pdf)
- [natural_persona_evil_q01_s0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_evil_q01_s0.pdf)
- [natural_persona_evil_q05_s0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_evil_q05_s0.pdf)
- [natural_persona_evil_q10_s0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_evil_q10_s0.pdf)
- [natural_persona_evil_q01_complete](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_evil_q01_complete.pdf)
- [natural_persona_evil_endpoints](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_evil_endpoints.pdf)
- [natural_persona_evil_e2](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_evil_e2.pdf)
- [natural_persona_evil_e2p](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_evil_e2p.pdf)
- [natural_persona_evil_q01_s1](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_evil_q01_s1.pdf)
- [natural_persona_evil_q01_s2](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_evil_q01_s2.pdf)
- [natural_persona_evil_q01_s3](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_evil_q01_s3.pdf)
- [natural_persona_evil_q01_s4](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural_extremes_20260918/figures_v1/natural_persona_evil_q01_s4.pdf)
