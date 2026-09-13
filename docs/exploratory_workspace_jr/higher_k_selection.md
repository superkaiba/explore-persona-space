# Fixed subset for the conditional K=20 noise diagnostic

The [saved IDs](higher_k_subset_20260913.json) implement the analysis plan's
conditional 128-context higher-rollout diagnostic. They were fixed before any
main fit or main predictability result was available. Use the first 128 eligible
contexts in the original `selected_contexts.json` main-test hash order, where
eligibility is the already recorded joint completed K=5 cohort. The completion
ledger sorts its exported joint IDs lexically; that export order does not replace
the original sampling order.

The declared noise decision uses the native k=10 targets on the joint completed
primary cohort in either model, as specified in the supplementary analysis.
If triggered, retain the original five draws and canonical context input, add
seeds 47–61 in a separate artifact, and score the frozen K=5-trained ridge and
individual MLP predictors. The input artifact records all 20 seeds and the
source-ledger hash. K=5 remains primary. Report any further incomplete K=20
contexts and their reasons without replacing prompts.
For the K=5 versus K=20 diagnostic, score both on the identical realized joint
K=20-complete rows within the fixed subset, preserving bootstrap pairing across
rollout counts, lenses, predictors and models. The main K=5 cohort is unchanged.

This file fixes the conditional sample; it does not report that the noise
criterion was met or that the higher-K experiment ran.
