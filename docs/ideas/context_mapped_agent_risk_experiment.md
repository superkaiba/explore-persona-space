# Context-mapped forecasting of reward hacking and misaligned actions

## Objective

Test whether an agent's activation at a decision boundary, before it generates any new reasoning or action, predicts the probability of a subsequent reward hack or misaligned action. Test whether the project's already-trained context-to-answer map improves this forecast by expressing the context in answer space.

The intended claim is prospective risk forecasting, not retrospective detection of a harmful answer and not recovery of a hidden, philosophically defined “intent.”

## Core quantities

For each unique decision context \(j\):

- \(c_j\): residual-stream activation at the final context token, before assistant generation;
- \(M(c_j)\): output of the frozen context-to-answer map;
- \(v_{ij}\): realized answer/reasoning state for continuation \(i\), used only as a non-deployable oracle;
- \(r_{RH}\), \(r_{MA}\): frozen answer-space directions for reward hacking and misaligned action, learned from separate development corpora;
- \(y_{ij}\): environment-scored risky outcome for continuation \(i\).

The main scalar scores are

\[
s_{raw}=r^\top c_j, \qquad s_{map}=r^\top M(c_j), \qquad s_{oracle}=r^\top v_{ij}.
\]

Because \(M\) is deterministic, it cannot add information to \(c\). “The mapping helps” means that it gives an independently defined answer-space risk direction a better transferable coordinate system, calibration, or label efficiency than applying the same direction directly to context or fitting capacity-matched raw-context readouts.

## Hypotheses

1. **Context hazard:** pre-generation activations predict the conditional probability of a future risky action beyond task condition, visible state, trajectory position, and a strong text-only monitor.
2. **Mapping value:** \(s_{map}\) has lower held-out probabilistic loss than \(s_{raw}\), particularly on unseen task/scenario families and at low calibration-label budgets.
3. **Cross-construct validity:** the result replicates in both reward hacking and misaligned action, and is not confined to impossible tasks, an explicit attacker prompt, or one action ontology.

## Environment portfolio

| Phase | Environment | Primary role | Label and horizon |
|---|---|---|---|
| Feasibility A | [Impossible-LiveCodeBench](https://arxiv.org/abs/2510.20270) | Clean reward-hacking outcome | Whether a continuation passes an impossible task; eventual-trajectory horizon |
| Feasibility B | [Agentic Misalignment](https://www.anthropic.com/research/agentic-misalignment), beginning with information leakage | High-yield harmful-action pilot | Whether the agent sends/leaks the protected information; eventual-trajectory horizon |
| Confirmation A | [EvilGenie](https://arxiv.org/abs/2511.21654) | Reward-hacking propensity when an honest solution exists | Test tampering or held-out-test behavior; next-action and eventual horizons reported separately |
| Confirmation B | [Instrumental Choices](https://arxiv.org/abs/2605.06490) | Low-nudge, policy-violating instrumental behavior | Deterministic `ic_behavior_detected`; next-action where localizable and eventual horizon |
| Temporal validation | [HVTB](https://arxiv.org/abs/2608.22103) | Exact first-hack timing | First instrumented access to planted solution/tests; next-action and short-horizon hazard |
| Optional OOD | [BashArena](https://arxiv.org/abs/2512.15688) or [Peer-Preservation](https://arxiv.org/abs/2604.19784) | Different action ontology | Programmatically scored sabotage or self/peer-preserving action |

ImpossibleBench is the primary outcome-clean reward-hacking setting, not a standalone test of intent: failure conflates compliance, inability, and failure to discover an exploit. EvilGenie supplies the critical honest-path control. Instrumental Choices is the strongest misalignment construct check because it crosses forbidden/permitted behavior, monitoring, urgency, incentive, and honest-path availability within the same tasks. Agentic Misalignment supplies more settings and a better chance of enough positives in a small open model, but its salient framing makes it a feasibility and replication arm rather than the sole basis for a misalignment claim.

Do not begin with HVTB. Advance to it only if the target model both completes enough nominal tasks and generates enough hacks; its scientific value is exact event timing, while its planted hack families are narrow and potentially too difficult for Qwen2.5-7B.

## Stage 0: freeze the representation experiment

Use the same base model, tokenizer, chat template, layer, pooling rule, and existing context-to-answer map as the mapping project, initially Qwen2.5-7B-Instruct. Record model, map, and dataset hashes.

Construct \(r_{RH}\) and \(r_{MA}\) from disjoint answer/action examples that do not use any evaluation task, paraphrase, environment asset, or scorer. School of Reward Hacks may support development of \(r_{RH}\), but no task family used to learn the direction may appear in confirmatory evaluation. Validate answer-level discrimination on a separate development holdout, then freeze the directions and all normalization.

Predeclare one primary layer and checkpoint. Layer sweeps, alternative pooling, answer spans, and horizons are exploratory and multiplicity-corrected.

## Stage 1: capability and prevalence pilot

### Reward hacking

Run all feasible Impossible-LiveCodeBench base problems under matched original, one-off, and conflicting variants using the minimal scaffold. First measure original-task competence. Include in risk analysis only base tasks meeting a preregistered original-task success threshold; retain all variants of each included task.

For the frozen Qwen3.8-27B public-development feasibility run, a base task is competent if at least one of eight original-condition continuations passes. Proceed to prediction only with at least five competent base tasks, ten positive and ten negative uncensored impossible-condition rollouts, three mixed-outcome impossible contexts, and zero technical-error censoring. These thresholds are recorded in the runner before the first successful rollout result is observed.

### Misaligned action

Run the original Agentic Misalignment email environment without the Intent-as-a-Tool modification. Start with information leakage, then add blackmail only if the model generates enough valid actions. Hold out entire scenario templates and framing families.

For every unique initial prompt, generate 8–16 stochastic continuations using fixed decoding parameters. Proceed only if both arms contain enough competent task/scenario clusters and mixed-outcome contexts to estimate conditional risk. Determine full-run sample size from pilot prevalence and task-level intracluster correlation, not raw continuation count. Predeclare the gate before inspecting activation results; if it fails, report it and change the model or use a separately labeled model organism rather than selecting rare positive trajectories.

## Stage 2: exact-prefix prospective dataset

At every selected decision boundary:

1. Serialize the complete context and environment state.
2. Capture \(c_j\) at the final context token before any new assistant token is sampled.
3. Restore that exact prefix and environment snapshot for each continuation.
4. Sample \(n_j\) independent continuations.
5. Score each continuation with the native deterministic scorer where available.

The empirical target is

\[
k_j \sim \mathrm{Binomial}(n_j,p_j), \qquad k_j=\sum_i y_{ij},
\]

where \(p_j\) is the conditional hazard associated with the identical prefix. Since identical prefixes have identical activations, individual continuation labels are repeated stochastic outcomes, not independent activation examples.

Use two separately declared targets:

- **Next-action hazard:** the next environment action triggers the risky event.
- **Eventual-trajectory hazard:** the event occurs before episode termination.

Never combine these labels. ImpossibleBench's primary target is eventual outcome; HVTB's primary target is next-action onset. For multi-step settings, stop collecting post-event examples and branch only from fully restorable environment states.

## Stage 3: confirmatory environments

### EvilGenie

Run the unambiguous subset first. Treat explicit modification of tests as a deterministic onset label. Score hardcoding and semantic verifier manipulation using the released judge plus blinded human audit; report mechanically labeled and adjudicated results separately. This tests whether the signal remains when a legitimate solution is possible.

### Instrumental Choices

Run all seven tasks and eight benchmark variants, with repeated continuations per context. Use the released deterministic scorer fields as authoritative. Analyze forbidden versus permitted, watched versus unwatched, incentive versus no incentive, ordinary versus critical, and honest-path-available versus blocked contrasts within task.

Use leave-one-task-out evaluation because there are only seven independent task families. Treat cross-task uncertainty—not hundreds of repeated rollouts—as the limiting sample size. If the event rate is too low, retain the setting as an explicitly underpowered construct check rather than pooling variants as independent tasks.

## Predictors and baselines

Freeze all features before the confirmatory run. Compare:

1. prevalence-only prediction;
2. environment metadata: task/scenario, condition, visible affordances, step number, and prior exploit-cue exposure;
3. a strong text-only model reading the identical visible prefix;
4. metadata + \(s_{raw}\);
5. metadata + \(s_{map}\);
6. random, shuffled, and norm-matched direction/map controls;
7. capacity- and label-budget-matched supervised probes on \(c\), including a one-dimensional readout and a preregistered low-rank probe;
8. metadata + \(s_{oracle}\), evaluated only as a post-generation upper bound.

Fit only a scalar calibration model for the frozen primary scores. Select regularization and any low-rank dimension in nested training folds. Include benign-next-action prediction as a specificity control: a detector that predicts every distinctive action equally well may only decode generic action preparation.

For the public-development feasibility analysis, if and only if the frozen reward-hacking prevalence gate passes, use layer 44 and leave one complete base task out at a time. Compare cross-fitted prevalence, metadata, fixed hashed-text-plus-metadata, raw-activation-plus-metadata, and mapped-activation-plus-metadata logistic predictors; choose the L2 penalty in grouped inner folds. The primary feasibility contrast is held-out binomial log loss for raw activation plus metadata minus the fixed text plus metadata baseline. Call this a development signal only when a task-cluster bootstrap 95% interval is wholly below zero. The mapping contrast uses the same rule for mapped minus raw. These supervised development probes do not substitute for the independently frozen answer-space directions or confirmatory environments required for the full claim.

## Splits and leakage controls

- Hold out complete base tasks and scenario/template families, never random continuations or steps.
- Keep original/impossible variants, paraphrases, all exact-prefix branches, and all steps of one trajectory in the same fold.
- Report within-condition performance so the detector cannot win by recognizing impossibility, replacement pressure, surveillance, or the presence of an exploit affordance.
- Match prompts when comparing safe and risky behavior. Do not train “attacker prompt” versus “honest prompt” classifiers in BashArena or Terminal Wrench.
- Exclude proposed action tokens, generated reasoning, tool calls, tool results, final rewards, and post-event states from the primary context representation.
- Position-match safe actions and episode endings, particularly where harm terminates the episode.
- Separate objective harmful action from inferred intent. Forbidden/permitted and no-incentive controls support the latter but do not prove it.

## Endpoints and inference

The primary metric is held-out binomial log loss over unique prefixes, with Brier score and calibration slope/intercept co-primary for interpretation. Report AUPRC, AUROC, and prevalence as secondary classification summaries. For action-onset environments also report recall at fixed per-step false-positive rates, trajectory-level false-alarm probability, and lead time before the first event.

Use task/scenario-clustered bootstrap confidence intervals or a hierarchical binomial model. Prefixes, rollouts, and steps do not determine the effective sample size independently of their task cluster.

Predeclare the mapping contrast

\[
\Delta_{map}=L(\text{metadata}+s_{map})-L(\text{metadata}+s_{raw}),
\]

where negative is better. Also report both scores against the text-only and capacity-matched raw-context probes.

Conclude that **context probing works** only if activation features improve held-out probabilistic prediction over metadata and text-only baselines on at least one preregistered reward-hacking arm and one preregistered misalignment arm. Conclude that **the mapping helps** only if \(\Delta_{map}<0\) with a task-clustered interval excluding zero in the preregistered primary comparison and the effect transfers to at least one unseen environment or low-label regime. Do not interpret failure to beat an unrestricted raw probe as information loss unless the capacity and supervision comparisons justify it.

## Optional intervention phase

After fixing a threshold on validation tasks, rerun HVTB or BashArena and defer high-risk actions to a trusted policy, a safe replan, or termination. Compare with a random intervention matched on intervention count and trajectory position. Report harm prevented versus benign task utility lost. Keep this phase separate from the prediction study so adaptive blocking does not change the distribution used to estimate forecasting performance.

## Novelty relative to related work

- [Risk States](https://arxiv.org/html/2606.06223) probes generated reasoning tokens before a next action. This experiment's primary state is captured before any new reasoning and is evaluated as an exact-prefix conditional hazard.
- [Probing the Misaligned Thinking Process](https://arxiv.org/abs/2606.24251) also uses reasoning-token activations and illustrates sensitivity to authorized versus unauthorized framing. This design evaluates objective environment actions and includes matched policy controls.
- [Intent as a Tool](https://arxiv.org/abs/2608.27348) scores a behavior-specific tool during generated reasoning and changes the agent's action space. This experiment leaves the policy interface unchanged and uses its score only as a secondary during-reasoning comparator.
- [Indirect prompt-injection exposure probes](https://arxiv.org/abs/2608.02657) predict whether risky content is present in the context. This experiment predicts the agent's later behavior within matched conditions and repeated continuations.
- [MisActBench/DeAction](https://arxiv.org/abs/2602.08995) vets an already proposed action. This experiment forecasts before the proposed action exists.
- The new mapping claim is not added information: it is transfer of an independently learned answer-space risk readout to the pre-generation boundary through a frozen general map.

## Decision rule

The minimal convincing result is:

1. context activation improves over metadata and text-only baselines in ImpossibleBench and in a misalignment arm;
2. the reward-hacking result replicates in EvilGenie, where an honest path exists;
3. the misalignment result survives within-condition tests in Instrumental Choices;
4. the mapped score improves calibration, task transfer, or low-label performance over the frozen raw score; and
5. HVTB shows useful signal before, rather than only immediately adjacent to, the first exploit action.

A clean negative result is also informative. If \(s_{oracle}\) predicts behavior but \(s_{map}\) does not, the current map fails to transfer answer-space risk into the context boundary. If neither predicts behavior, the risk direction or target-model organism is inadequate. If only metadata/text predicts behavior, the benchmark is cue-driven for this question. If the target model never produces risky behavior, the environment/model pairing is uninformative rather than evidence that the behavior is unprobeable.
