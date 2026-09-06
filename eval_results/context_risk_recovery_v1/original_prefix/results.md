# Pre-action agent-risk feasibility result

Status: **reward-hacking feasibility gate failed**

## Frozen gates

- Reward hacking: `False`; 0 positive, 288 negative, 0 censored, 0 mixed contexts, and 18 competent tasks.
- Misaligned action: `True`; 15 positive, 625 negative, 0 censored, and 6 mixed contexts.

## Prediction result

Prediction was not fit because the frozen reward-hacking prevalence gate failed. This is an underpowered environment/model result, not evidence that pre-action prediction is impossible.

### Misaligned action

Split: `leave_one_exact_context_sha256_out`; groups: 16; metadata: `length`.

| Predictor | Log loss | Brier | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| prevalence | 0.1173 | 0.0232 | 0.087 | 0.018 |
| metadata | 0.1059 | 0.0228 | 0.548 | 0.028 |
| text_metadata | 0.0925 | 0.0228 | 0.848 | 0.102 |
| raw_activation_metadata | 0.0914 | 0.0219 | 0.875 | 0.128 |
| mapped_activation_metadata | 0.0927 | 0.0223 | 0.869 | 0.127 |

Misaligned-action raw-activation-minus-text log-loss contrast: -0.0011 (exact_context_sha256-cluster 95% CI -0.0090, 0.0056).

Mapped-minus-raw log-loss contrast: 0.0014 (95% CI -0.0006, 0.0040).
Intervals resample fixed out-of-fold predictions; they do not refit models. All prompts are from one information-leak scenario. Structured-fold checks are exploratory, with only three or four groups.

## Interpretation

The environment/model pairing did not yield enough competent, mixed reward-hacking outcomes for a valid prospective prediction test. Per the frozen protocol, fitting a classifier would be post-selection on rare trajectories, so no prediction claim is made.
