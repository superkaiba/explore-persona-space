# Main 4B sampling-noise checkpoint — 2026-09-13

The native k=10 Qwen3.5-4B fit and diagnostics completed successfully. On the
already frozen shared cohort of 252 completed test contexts, all five estimated
noise fractions are below the registered 10% higher-K trigger. No target has
undefined variance. The 27B criterion remains pending at this checkpoint.

| Target | Estimated noise / observed target variance |
|---|---:|
| Full answer mean | 2.35% |
| J component | 2.70% |
| J remainder | 2.55% |
| R component | 3.11% |
| R remainder | 2.52% |

These are point estimates. For each context and target, estimate the variance
trace of its K=5 mean from the five original rollout means, retaining equal
rollout weighting. Average those estimates across the shared contexts and
divide by that target's unbiased between-context variance trace on the same
rows. Component/remainder covariance identities and the exact fitted-generation
bindings pass. The [saved report](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/2fd759d22fa9f6fd10d025c7b89e6445f3ba5689/exploratory_workspace_jr/20260912/comparison_main_noise_decision_v1/noise_decision.json)
retains the unrounded estimates, variance traces and decision.

The [independent actual-data review](main_noise_comparison_review.json)
recomputed the estimates directly from the original component rollout vectors
and verified the linked sources, completion records and uploaded diagnostic and
decision artifacts. The underlying [diagnostics](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0f683f0da241285da9aeaeafcdc77eae36ddea82/exploratory_workspace_jr/20260912/comparison_main_diagnostics_v1)
and [native k=10 fit](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/1b7a8dc64667a8a3c212b973b7d470f5ff5fc52f/exploratory_workspace_jr/20260912/comparison_main_rotationNone_v1/fits/main/k10-rotationNone)
are pinned separately. Original all-captured fit scores remain descriptive;
the final predictability comparisons will use the shared completed cohort.

The two-model trigger uses an OR rule: a positive primary-model result would
still require the fixed higher-K follow-up for both models. The unchanged
[conditional subset and protocol](higher_k_selection.md) remain binding.
This checkpoint describes finite-rollout variability. Main selectivity,
control comparisons and stronger/weaker-model conclusions remain outstanding.
