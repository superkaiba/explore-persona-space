# Pre-action agent-risk feasibility result

Status: **reward-hacking feasibility gate failed**

## Frozen gates

- Reward hacking: `False`; 0 positive, 288 negative, 0 censored, 0 mixed contexts, and 18 competent tasks.
- Misaligned action: `True`; 15 positive, 625 negative, 0 censored, and 6 mixed contexts.

## Prediction result

Prediction was not fit because the frozen reward-hacking prevalence gate failed. This is an underpowered environment/model result, not evidence that pre-action prediction is impossible.

### Misaligned action

Split: `leave_one_urgency_type_out`; groups: 3; metadata: `conditions`.

| Predictor | Log loss | Brier | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| prevalence | 0.1188 | 0.0233 | 0.297 | 0.020 |
| metadata | 0.3244 | 0.0786 | 0.811 | 0.079 |
| text_metadata | 0.3293 | 0.0813 | 0.841 | 0.082 |
| raw_activation_metadata | 0.3340 | 0.0829 | 0.848 | 0.123 |
| mapped_activation_metadata | 0.3319 | 0.0821 | 0.865 | 0.124 |

Misaligned-action raw-activation-minus-text log-loss contrast: 0.0047 (urgency_type-cluster 95% CI -0.0013, 0.0081).

Mapped-minus-raw log-loss contrast: -0.0021 (95% CI -0.0130, 0.0087).
Intervals resample fixed out-of-fold predictions; they do not refit models. All prompts are from one information-leak scenario. Structured-fold checks are exploratory, with only three or four groups.

## Interpretation

The environment/model pairing did not yield enough competent, mixed reward-hacking outcomes for a valid prospective prediction test. Per the frozen protocol, fitting a classifier would be post-selection on rare trajectories, so no prediction claim is made.
