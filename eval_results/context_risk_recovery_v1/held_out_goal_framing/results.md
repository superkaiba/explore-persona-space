# Pre-action agent-risk feasibility result

Status: **reward-hacking feasibility gate failed**

## Frozen gates

- Reward hacking: `False`; 0 positive, 288 negative, 0 censored, 0 mixed contexts, and 18 competent tasks.
- Misaligned action: `True`; 15 positive, 625 negative, 0 censored, and 6 mixed contexts.

## Prediction result

Prediction was not fit because the frozen reward-hacking prevalence gate failed. This is an underpowered environment/model result, not evidence that pre-action prediction is impossible.

### Misaligned action

Split: `leave_one_goal_framing_out`; groups: 4; metadata: `conditions`.

| Predictor | Log loss | Brier | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| prevalence | 0.1359 | 0.0238 | 0.244 | 0.021 |
| metadata | 0.1107 | 0.0233 | 0.654 | 0.039 |
| text_metadata | 0.1096 | 0.0242 | 0.739 | 0.053 |
| raw_activation_metadata | 0.1007 | 0.0222 | 0.773 | 0.103 |
| mapped_activation_metadata | 0.0993 | 0.0220 | 0.773 | 0.102 |

Misaligned-action raw-activation-minus-text log-loss contrast: -0.0089 (goal_framing-cluster 95% CI -0.0232, 0.0179).

Mapped-minus-raw log-loss contrast: -0.0014 (95% CI -0.0042, 0.0004).
Intervals resample fixed out-of-fold predictions; they do not refit models. All prompts are from one information-leak scenario. Structured-fold checks are exploratory, with only three or four groups.

## Interpretation

The environment/model pairing did not yield enough competent, mixed reward-hacking outcomes for a valid prospective prediction test. Per the frozen protocol, fitting a classifier would be post-selection on rare trajectories, so no prediction claim is made.
