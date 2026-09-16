# Fixed answer-direction transfer — 2026-09-16

## Result

Fixed answer directions predict held-out behavior through the generic context-to-answer map on several datasets without a downstream behavior regression. The map does not consistently improve over directions extracted directly in context space.

## Method

Frozen Qwen2.5-7B-Instruct layer 19, common raw activation coordinates, and the #779 generic map trained on **963,444 pairs**. Each behavior direction is the positive-minus-negative mean from 1,000 positive and 1,000 negative instruction-conditioned rollouts (100 contexts per polarity). These are **unfiltered instruction contrasts**, not judge-filtered canonical Persona Vectors. No evaluation-label-based regression, layer selection, sign flip, or hyperparameter selection is used. All methods share the retained evaluation contexts and existing on-policy behavior judgments.

The five methods are: answer direction on actual answers (instrument validation); the same answer direction on mapped contexts; independently extracted context direction on contexts; answer direction directly on contexts; and the mean correlation of five maps refitted after shuffling training and validation answer identities. All five null fits use the entire original map training pool and original fit recipe. Uncertainty uses 2,000 paired context-group bootstraps, conditional on the fixed directions and maps.

## Results

Spearman correlation with behavior expression; actual-answer scoring requires generation and is a validation reference.

| Behavior / dataset | n | Actual answer | Mapped | Context direction | Answer direction on context | Shuffled mean |
|---|---:|---:|---:|---:|---:|---:|
| evil / hhrt | 1,847 | 0.072 | 0.058 | 0.001 | 0.042 | 0.003 |
| evil / toxicchat | 370 | 0.440 | 0.426 | 0.403 | 0.449 | 0.091 |
| evil / wildchat_rung | 4 | undefined | undefined | undefined | undefined | undefined |
| sycophancy / aita | 1,304 | 0.215 | 0.255 | 0.319 | 0.186 | -0.026 |
| sycophancy / wildchat_rung | 4 | -0.400 | -0.400 | -0.800 | -0.800 | 0.200 |
| hallucination / nqopen | 3,164 | -0.009 | -0.085 | -0.127 | -0.133 | 0.047 |
| hallucination / simpleqa | 4,021 | 0.605 | 0.449 | 0.507 | 0.464 | -0.054 |
| hallucination / wildchat_rung | 4 | 0.800 | 0.800 | -0.200 | 0.800 | 0.520 |

## Hypotheses and paired comparisons

- **H1, answer-direction validity:** positive on HH red-team, ToxicChat, AITA, and SimpleQA; weak on HH. Fails on NQ-Open (rho -0.009). WildChat is uninformative.
- **H2, transfer without downstream retraining:** supported descriptively on ToxicChat, AITA, and SimpleQA, and weakly on HH. NQ-Open has negative mapped correlation and fails H1; it does not support transfer. WildChat is uninformative.
- **H3, advantage over a context-native direction:** HH map-minus-context delta 0.057 [0.024, 0.088]; ToxicChat 0.023 [-0.026, 0.072]; AITA -0.064 [-0.116, -0.016]; SimpleQA -0.057 [-0.068, -0.047]. The positive NQ delta reflects less-negative correlations, not successful prediction. Thus there is no consistent mapping advantage.

## Coverage and limitations

All eight registered cells completed. Five non-WildChat cells provide interpretable sample sizes; three WildChat cells retain only four contexts each after overlap and valid-label filtering. Of 419 planned WildChat contexts, 415 match a map-training/validation first user question after normalization, including 411 exact stripped matches. No exclusion relies solely on a later user turn. These are first-question content matches, not identical full rendered conversation prompts. Evil's four retained WildChat labels are constant, so its correlation is undefined.

Shuffled-map seed variation is substantial: AITA seed 0 scores 0.443, above the true map's 0.255. Bootstrap intervals for the five-seed mean condition on those five maps; they do not establish superiority over the distribution of random pairings. Extraction-direction uncertainty is also outside the reported intervals. The archived map targets include closing assistant template tokens, whereas evaluation answer activations average completion tokens only.

## Integrity and artifacts

The true-map refit reproduced archived predictions to maximum absolute error 1.26e-8. Held-out map R² is 0.7542; identity-plus-training-bias R² is -0.9196. Cosine nearest-neighbor retrieval is 81.0% on a 1,000-answer pool (chance 0.1%; identity-plus-bias 56.9%). Five shuffled maps retrieve at chance. This verifies the intended ~1M map and does not establish a behavioral advantage by itself.

Source commit: `722ce8ae2c003209df22ab8adc4442cf72afb3df`. Forty-one focused tests passed. Independent review verified all behavior completion hashes, map parity, paired comparisons, null-seed variability, and WildChat overlap witnesses.

[All-methods plot](https://eps.superkaiba.com/tasks/1739/figure/fixed_transfer_all_methods_20260916.png) · [Paired differences](https://eps.superkaiba.com/tasks/1739/figure/fixed_transfer_paired_differences_20260916.png)

[Verified source, raw predictions, directions, map payloads, bootstraps, exclusions, provenance, and figures](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/750831d2f8d5b7cdf2cc3859ddd69f0ce6825dac/issue1739_fixed_transfer_20260916): all 92 files, 1,402,357,312 bytes, verified against the remote immutable revision. Inputs are pinned to prior archived generation and judge artifacts. No fresh generations or judgments were needed.

The monitored run exited successfully; the independent watchdog acknowledged completion notification delivery. Both completed-run services were disabled after upload verification.
