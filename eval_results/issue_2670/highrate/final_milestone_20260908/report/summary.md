We can elicit and label successful reward hacking in this benchmark. This rerun did not establish useful prediction from the initial activation, and the frozen context-to-answer map showed no added benefit.

We screened 103 tasks with Qwen3.8-27B and selected the 30 highest rates observed in that screen. Those rankings are noisy estimates from four impossible-task rollouts per task. The selected panel includes all 21 tasks with a screening success plus nine fixed zero-rate ties. Fresh evaluation produced **34 confirmed hacks, 197 valid failures, one unknown outcome, and eight structurally unassessable cases** across 240 planned impossible-task rollouts. The completed, assessable hacking rate was 34/231 = 14.7%.

The frozen split held out ten entire tasks; nine had assessable impossible tests, giving 72 test rollouts including 12 hacks. Training used 20 other tasks. Lower log loss is better; AUROC is a secondary ranking metric.

| Predictor | Held-out log loss | AUROC |
|---|---:|---:|
| Constant training-risk baseline | 0.453235 | 0.500 |
| Text + task metadata | 0.453745 | 0.667 |
| Initial activations + metadata | 0.453634 | 0.578 |
| Mapped activations + metadata | 0.453688 | 0.544 |

The initial-activation probe reduced loss relative to text by only 0.000111 nats per rollout, with a 95% interval of [-0.000172, 0.000416]. It also had worse loss than the constant-risk baseline. Mapping’s loss reduction relative to raw activations was -0.000054, with a 95% interval of [-0.000156, 0.000049]; positive values would favor mapping. All four prespecified mapping comparisons had intervals spanning zero.

Earlier cheating behavior was more informative in these held-out data: metadata plus independent screening outcomes had loss 0.358658, and text plus metadata and screening had loss 0.363524. The corresponding activation probe had loss 0.453630. The competence-restricted sensitivity analysis showed the same qualitative pattern.

Cross-validation chose the strongest tested regularization for every activation probe; its primary forecasts were nearly constant at about 13.8% risk. This conclusion concerns one layer, a frozen map, and the tested linear readout recipe. It does not establish that reward-hacking information is absent from the activations.

The intervals resample whole test tasks with fixed predictions and exclude fitting uncertainty. Estimates are conditional on completed, assessable outcomes; the unknown and unassessable cases prevent an unconditional benefit claim under the prespecified gate. Broader agentic misalignment was not established by this reward-hacking experiment.

An independent critic checked the code before execution and recomputed eight planned loss comparisons across four regimes. A finite optimizer limit was raised from 5,000 to 20,000 after one control fit failed; it converged at 5,669 with the objective, tolerance, features, splits and seeds unchanged. Two independently checked, previously converged fits remained identical. Both the failed attempt and the complete rerun are retained.

[Raw observations and activation archive](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/58d3e6d6b0917b6e9777c45d23b814693dc14277/context_risk/issue2670_highrate/raw). Full numerical evidence is in the accompanying `analysis/result.json`, `independent_result_audit.json`, and `comparison_data.json`.
