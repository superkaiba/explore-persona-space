# Main sampling-noise decision — both models

Both native k=10 fits and their diagnostics completed successfully. On the
frozen shared cohort of 252 completed test contexts, none of the five targets
in either model exceeds the registered 10% higher-K trigger. Every denominator
is defined. The conditional K=20 follow-up is therefore not triggered; the
main experiment retains K=5. This is an execution decision, not a conclusion
about component predictability.

| Target | Qwen3.5-27B | Qwen3.5-4B |
|---|---:|---:|
| Full answer mean | 1.71% | 2.35% |
| J component | 2.26% | 2.70% |
| J remainder | 1.83% | 2.55% |
| R component | 2.43% | 3.11% |
| R remainder | 1.81% | 2.52% |

Each entry is an estimated sampling-noise trace divided by that target's
observed between-context variance trace. For each context, the five equally
weighted rollout means give a variance-of-the-mean estimate with denominator
K(K−1)=20. Average that trace over the same 252 contexts and divide by the
target's unbiased between-context variance, using denominator 251. These
are point estimates; this threshold does not establish that sampling noise
is irrelevant to smaller differences in R².

The [27B report](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/6718e57dee4f8dc5c5066b11da3bc86141ef96be/exploratory_workspace_jr/20260912/primary_main_noise_decision_v1/noise_decision.json)
and [4B report](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/2fd759d22fa9f6fd10d025c7b89e6445f3ba5689/exploratory_workspace_jr/20260912/comparison_main_noise_decision_v1/noise_decision.json)
retain unrounded estimates, variance traces, covariances, context IDs and
per-model decisions. Their per-model negative labels deliberately leave the
other model open; this document combines the two independently verified
negative decisions under the frozen OR rule. The final supplement will
recompute the criterion on the final common scoring population.

Independent reviews for [27B](main_noise_primary_review.json) and
[4B](main_noise_comparison_review.json) recomputed all estimates from the saved
rollout vectors using independent accumulation. They verified all 256 original
test contexts before selecting the exact ordered joint 252, the covariance
identities, fitted-target fingerprints, final raw-generation bindings, each
model's terminal-token policy, the frozen conditional 128-context subset,
successful producer terminals and immutable uploaded source hashes. Each
review checked 536 consumed native-source files, all 20 diagnostic files,
all nine noise-decision files and three cohort files; it is not a full audit
of unrelated fit cells.

The 27B source [native k=10 fit](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/875c6af2f3cb0637f5fda7c81027cde93753fa2b/exploratory_workspace_jr/20260912/primary_main_rotationNone_v1/fits/main/k10-rotationNone)
and [diagnostics](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e6a36c8a74584b449d38e47f1a495fb056f20cc4/exploratory_workspace_jr/20260912/primary_main_diagnostics_v1)
are pinned separately. The earlier [4B checkpoint](main_noise_comparison_20260913.md)
records its corresponding source fits and diagnostics. The unchanged
[conditional protocol](higher_k_selection.md) and
[completion-scoring declaration](completion_scoring_20260913.md) define the
scope. No noisy target is interpreted as reasoning or automaticity, and the
remaining sparse-decomposition controls and paired predictability analysis
are still required.
