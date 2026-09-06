# Agent-risk forecasting pilot: completed, broad prediction claim unsupported

## Takeaways

- Reward-hacking rollouts finished: 480/480, zero technical errors. Original-task success was 106/160; impossible variants yielded 0/320 hacks. Among the 18 competent tasks, 0/288 impossible rollouts were positive. The frozen prevalence gate failed, so the reward-hacking classifier was not fit.
- Misalignment rollouts finished: 15 harmful emitted forwards among 640 continuations (2.34%), no censoring, and six mixed-outcome contexts. The 20 manifest rows reduce to 16 unique prefixes within one information-leak scenario.
- The original prediction analysis crashed before starting because the launch environment lacked uv on PATH. Recovery completed using the saved data; no new generation or training was performed.

## Goal

Test whether the model's activation before generating new reasoning or an action predicts a later reward hack or harmful action, and whether a frozen context-to-answer map improves that forecast. This report closes the existing public-development feasibility pilot. It does not establish the full cross-environment claim.

## Methodology

Qwen3.8-27B, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, layer 44. The frozen linear context-to-answer map and captured states share model revision and dimension (5,120). Map SHA256: `680935a244cc39c29797d66b312b95e3741889a541dcee8bde7c69ac33c5242d`. Selection used generic map-development data, not safety labels. The full map-training capture bank was not revalidated in this recovery; no new generic-map reconstruction/retrieval claim is made.
All 40 misalignment rollout/activation hashes were checked, 640 saved outcomes rescored deterministically, summary counts reconciled, and duplicate prefixes merged. All 640 seeds differ. Independent qualitative review of all 15 positives confirmed intended final harmful forwards. The XML scorer is not generally quotation-aware and no external email tools were executed; the measured outcome is a harmful action emitted in simulation.
The original manifest requested eight continuations per row; the completed v5 launch used 32, giving 640. Earlier v3 outputs share sampling keys with v5 and were excluded. Repeated continuations estimate each prefix's hazard and do not create independent activation examples.
The preserved development comparison uses leave-one-exact-prefix-out prediction and prefix length as metadata. Additional recovery checks include goal framing, goal value, and urgency indicators as visible-condition controls. Structured splits leave one entire goal framing or urgency group out; latent/swap aliases are merged to keep identical prompts in one fold. All structured checks are exploratory and remain within one scenario family.
Predictors are training-fold prevalence, metadata, character-ngram hashed text plus metadata (2,048 features, lengths 3–5), raw activations plus metadata, and mapped activations plus metadata. Linear logistic fits retain the original summed L2 objective. Frequency weights collapse duplicate feature-label rows, and a primal solver with tolerance 1e-8 replaces the slow duplicate-row dual solver. Numerical equivalence to expanded-data fits is tested. No nonlinear probe was used.
Regularization C is selected from [1e-4, 1e-3, 1e-2, 0.1, 1] in up to five grouped inner folds. The inherited fallback is C=0.01 with fewer than three training groups (applies to urgency-held-out fits); single-class training uses (positives+0.5)/(total+1). These are the saved development protocol's choices, not new confirmatory hyperparameters.
Log loss and Brier score weight each observed Bernoulli outcome; AUROC and average precision are descriptive pooled summaries. Paired 95% intervals use 5,000 seeded cluster resamples of fixed out-of-fold predictions, without refitting the models. They omit model-training uncertainty. Only four goal-framing groups and three urgency groups are available, making structured intervals especially fragile. A raw activation benefit requires the raw-minus-text interval wholly below zero; mapping benefit requires mapped-minus-raw wholly below zero. These development criteria do not replace confirmation against a strong semantic text monitor or frozen answer-space risk directions.

## Results

| Evaluation | Predictor | Log loss | Brier | AUROC | Average precision |
|---|---|---:|---:|---:|---:|
| original_prefix | prevalence | 0.11733 | 0.02315 | 0.087 | 0.018 |
| original_prefix | metadata | 0.10585 | 0.02285 | 0.548 | 0.028 |
| original_prefix | text_metadata | 0.09247 | 0.02277 | 0.848 | 0.102 |
| original_prefix | raw_activation_metadata | 0.09135 | 0.02187 | 0.875 | 0.128 |
| original_prefix | mapped_activation_metadata | 0.09273 | 0.02230 | 0.869 | 0.127 |
| condition_controlled_prefix | prevalence | 0.11733 | 0.02315 | 0.087 | 0.018 |
| condition_controlled_prefix | metadata | 0.09056 | 0.02156 | 0.872 | 0.125 |
| condition_controlled_prefix | text_metadata | 0.09263 | 0.02281 | 0.848 | 0.102 |
| condition_controlled_prefix | raw_activation_metadata | 0.09136 | 0.02187 | 0.875 | 0.128 |
| condition_controlled_prefix | mapped_activation_metadata | 0.09274 | 0.02230 | 0.869 | 0.127 |
| held_out_goal_framing | prevalence | 0.13594 | 0.02376 | 0.244 | 0.021 |
| held_out_goal_framing | metadata | 0.11074 | 0.02330 | 0.654 | 0.039 |
| held_out_goal_framing | text_metadata | 0.10959 | 0.02419 | 0.739 | 0.053 |
| held_out_goal_framing | raw_activation_metadata | 0.10065 | 0.02216 | 0.773 | 0.103 |
| held_out_goal_framing | mapped_activation_metadata | 0.09927 | 0.02204 | 0.773 | 0.102 |
| held_out_urgency | prevalence | 0.11881 | 0.02329 | 0.297 | 0.020 |
| held_out_urgency | metadata | 0.32443 | 0.07863 | 0.811 | 0.079 |
| held_out_urgency | text_metadata | 0.32929 | 0.08132 | 0.841 | 0.082 |
| held_out_urgency | raw_activation_metadata | 0.33399 | 0.08288 | 0.848 | 0.123 |
| held_out_urgency | mapped_activation_metadata | 0.33187 | 0.08211 | 0.865 | 0.124 |

Negative contrasts favor the first predictor:

| Evaluation | Groups | Raw minus text, 95% CI | Mapped minus raw, 95% CI |
|---|---:|---|---|
| original_prefix | 16 | -0.00112 [-0.00902, +0.00557] | +0.00138 [-0.00058, +0.00398] |
| condition_controlled_prefix | 16 | -0.00128 [-0.00930, +0.00549] | +0.00138 [-0.00057, +0.00398] |
| held_out_goal_framing | 4 | -0.00894 [-0.02317, +0.01792] | -0.00139 [-0.00421, +0.00036] |
| held_out_urgency | 3 | +0.00470 [-0.00131, +0.00811] | -0.00212 [-0.01304, +0.00868] |

![Held-out log loss](https://raw.githubusercontent.com/superkaiba/explore-persona-space/codex/context-risk-recovery-20260906/eval_results/context_risk_recovery_v1/figures/prediction_log_loss.png)

Both panels compare condition-controlled predictors; these are point estimates. Paired uncertainty is tabulated above. Failed reward-hacking prediction is omitted, not plotted as zero.

![All unique-context predictions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/codex/context-risk-recovery-20260906/eval_results/context_risk_recovery_v1/figures/context_predictions.png)

Each row is a unique prompt, labeled with positive/total outcomes. Duplicate latent/swap aliases are pooled under their first manifest name. Observed frequencies are estimates, not error-free true hazards.

### Scope and decision

The reward-hacking feasibility failure prevents the planned two-construct claim. The misalignment results below state whether either contrast passes in each split; an unpassed interval is inconclusive about small effects, not proof of equivalence.

- original_prefix: activation-over-text criterion not passed; mapping-over-raw criterion not passed.
- condition_controlled_prefix: activation-over-text criterion not passed; mapping-over-raw criterion not passed.
- held_out_goal_framing: activation-over-text criterion not passed; mapping-over-raw criterion not passed.
- held_out_urgency: activation-over-text criterion not passed; mapping-over-raw criterion not passed.

EvilGenie, Instrumental Choices, HVTB, independent frozen answer-risk directions, the post-generation oracle, intervention tests, and cross-scenario transfer were not run. No confirmatory calibration or transfer conclusion is available. Continuing that broader program requires a separately specified model/environment pairing with enough reward-hacking positives; selecting rare positive trajectories from this failed gate would invalidate the protocol.

**Repro:** `uv run python -m scripts.context_risk_finish --input-root <context_risk_inputs> --output-dir <new_output_dir>`. Per-split JSONs include exact input hashes, dependency versions, analyzer source hash, predictions, folds, and contrast definitions. This recovery used only local CPU analysis and existing artifacts.

**Verified archive:** [183 input files, byte-verified](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/97f50b287b704dbf4f353833154fd4c8f4e35d44/context_risk/recovery_20260906/inputs_v2). `input_upload_receipt.json` records the full relative paths and content hashes.

**Recovery code:** [source commit 3ed003a83e4b](https://github.com/superkaiba/explore-persona-space/tree/3ed003a83e4bc2fb5625ca196b6a69c39b6868f9); 12 focused tests, lint, and independent input/code/result reviews passed. No new GPU compute.
