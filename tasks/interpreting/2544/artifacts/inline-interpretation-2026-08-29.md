# Inline interpretation: map strength predates training, while the final operator settles only after pretraining

**Confidence: moderate.** This is a user-requested inline interpretation of the verified #2544 artifacts, without the usual independent analyzer/critic pipeline. It reports the registered adjudications and separates them from the broader scientific read.

## Bottom line

The experiment does **not** show the context-to-answer map gradually forming as a stronger linear relationship during pretraining. At the registered 0.10 threshold, the result is **No-formation**: baseline-subtracted held-out R² changes by only +0.030 from random initialization to the final base checkpoint, with a 95% interval of [-0.006, +0.068]. Raw R² actually falls from 0.341 to 0.265.

What changes is the *source* and *identity* of that predictability. The architecture already produces a strong context-to-answer linear relation at random initialization. During pretraining, representation-side predictability falls while answer-distribution predictability rises, largely cancelling in the diagonal score. Meanwhile, the particular operator used by the final base model keeps being rewritten through the whole stage-1 pretraining run. It first becomes strongly transferable to the final base checkpoint at the end of stage-2 midtraining, not during stage 1.

The cleanest answer to “when does the mapping form?” is therefore three-part:

- Map-like scalar predictability is present before learning and is not a learned formation event.
- The balance between representation geometry and answer distribution changes sharply in the first 88–252B tokens.
- The *final map itself*, judged by cross-checkpoint predictive transfer after coordinate alignment, does not settle until midtraining.

## Registered hypothesis adjudication

### H1: no registered formation, with a threshold-sensitive categorical label

At layer 31, the registered baseline-subtracted endpoint contrast is +0.0296, 95% CI [-0.0057, +0.0680]. Its upper bound is below the registered 0.10 formation floor, so the primary label is **No-formation**. Raw diagonal R² decreases from 0.341 at initialization to 0.265 at the final base checkpoint.

The categorical label is boundary-sensitive: lowering the floor to 0.05 produces an “Early” label because a transient early rung crosses the smaller threshold. That does not create a monotonic formation curve, and the final-base endpoint remains only +0.030 above initialization after baseline subtraction. The conservative scientific read is no net formation of stronger scalar predictability.

The identity-plus-bias subtraction needs care. The ridge R² remains in a fairly narrow range after 88B tokens, while the identity baseline becomes strongly negative at several later checkpoints. Peaks in the subtracted curve therefore often reflect a worsening identity baseline rather than a stronger ridge map.

### H2: the final operator is not present during stage-1 pretraining

The last stage-1 checkpoint at 5.93T tokens transfers poorly to the final base model even after orthogonal Procrustes alignment: transfer R² = 0.0358 and retention = 0.135, 95% CI [0.032, 0.221]. No stage-1 checkpoint reaches the registered 0.5 kill threshold, so the “crystallize before pretraining ends” hypothesis is rejected.

The end-of-midtraining checkpoint is different. Its aligned transfer to the final base model is R² = 0.219, or 0.826 of the final model's own diagonal score, 95% CI [0.762, 0.876]. The registered threshold crossing is assigned to `mid` in 77.5% of bootstrap draws, while 22.5% have no crossing. This dates final-map stabilization to the end of midtraining with moderate, not high, confidence.

The operator spectra alone would have been misleading: the stage-1 endpoint and final base have singular-value-spectrum cosine 0.997, yet their aligned predictive transfer is only 0.135 of the target diagonal. The coarse spectrum is already similar; the behaviorally relevant directions and correspondences are not.

Adjacent transfer is also non-monotonic. Several middle stage-1 transitions retain roughly half to four-fifths of the next checkpoint's map after general-linear alignment, but the stage-1-to-midtraining transition drops to -0.099 retention. Midtraining performs a substantial rewrite, after which mid-to-final-base retention rises to 0.847. DPO-to-RLVR is highly stable at 0.966.

### H3: demonstrations do not get absorbed during pretraining

The registered “in-context substitution decays” hypothesis is not supported. Four-shot minus zero-shot R² is +0.081 at 21B tokens, +0.106 at the end of midtraining, and +0.094 at the final base checkpoint. The selected peak exceeds the final base by only +0.012, with a selection-aware 95% interval of [+0.0002, +0.058]. More importantly, the random-init effect is already +0.104 rather than approximately zero, violating a central prediction.

The truncation controls refine this result. Random initialization has no naturally terminated rows, so its +0.104 effect is only identifiable on truncated 1,024-token outputs. At 4.2B tokens, the pooled negative effect (-0.026) disappears among rows that terminate naturally (+0.007, interval spanning zero) and is concentrated among rows truncated in both arms (-0.136). From 21B onward, the naturally terminated common-status read is usually positive and remains +0.097 at the final base checkpoint, 95% CI [0.079, 0.115]. Thus there is no pretraining-era decay once censoring is controlled.

The dose check supports a real prompt-context effect, though not the substitution story. At 21B tokens, R² rises from 0.196 at zero examples to 0.247/0.277/0.288 at 1/4/16 examples. At the final base checkpoint it rises from 0.265 to 0.357/0.359/0.400. Demonstrations continue to add linear predictability rather than becoming redundant. The effect declines only across post-training: +0.083 after SFT, +0.054 after DPO, and +0.040 after RLVR.

## The decomposition explains the flat diagonal

The fixed-text and fixed-weights arms expose two large countervailing changes.

When every checkpoint processes the **final base model's answer text**, changing only the weights/representations, R² falls from 0.470 at random initialization to 0.264 at the end of stage-1 pretraining. Cosine retrieval accuracy at rank 1 falls from 0.769 to 0.161. In absolute predictive terms, learned representations do not make the same final answer text easier to map; they make it harder than the random architecture does.

When the **final base weights** process each rung's own answer text, changing only the answer distribution, R² rises from 0.069 for random-init answers to 0.248 for the stage-1 endpoint's answers. The two curves cross between 21B and 88B tokens: by roughly 88B, answer text produced at that checkpoint is about as compatible with the final representation as final answer text is with that checkpoint's representation.

Those effects nearly cancel on the diagonal. From initialization to the stage-1 endpoint, the fixed-text curve changes by -0.206 and the fixed-weights curve by +0.179, while each checkpoint's own diagonal remains in the same broad 0.2–0.3 range. Pretraining changes *what supports* the mapping more than its aggregate R².

## R² and retrieval tell different stories

Cosine rank-1 retrieval chance is 1/10,858, or 0.0092%. Retrieval is far above chance at every rung, but its trajectory is unlike R². It is 80.6% at random initialization, collapses to 2.0% at 4.2B and 3.2% at 21B, reaches about 7.4% at 88–252B, ends stage 1 at 4.5%, then jumps to 18.8% at midtraining and 16.3% in the final base model. Post-training raises it further to 21.1%, 31.5%, and 32.8% after SFT, DPO, and RLVR.

The 80.6% initialization value should not be narrated as semantic competence. The random model's own answers are on-policy products of the same random computation being decoded, and virtually every answer hits the 1,024-token cap. Still, the fixed-final-text control also gives the random weights high retrieval (76.9%), showing that much of the relation is architectural rather than purely a generation-selection artifact. The safe conclusion is that the metric detects an instance-specific architectural coupling before learning; later training replaces it with a different, numerically weaker geometry associated with trained-model outputs. This experiment did not score semantic answer quality.

## Validity and scope

The statistical fit itself is well posed: 10,858 shared-intersection rows, minimum fold training size 9,015, feature dimension 4,096, dof-capped GCV ridge, and a passed shuffle-leak gate. The shared layer is 31, a full-attention layer, so the few-shot result is not a sliding-window overflow artifact. The trained-only 13,297-row sensitivity reproduces the same broad diagonal pattern.

The main limitations are interpretive. This is pooled, teacher-forced activation predictability on the model's on-policy answers, not evidence that the model causally uses a single linear operator to generate them. The random-init diagonal is completely truncation-confounded, though the fixed-final-text control provides independent evidence for an architectural relation. The shared intersection excludes any prompt that triggers repetition at any rung and arm, so the estimand is the 60.3% all-rung repetition-free subset. The experiment follows one training run, so checkpoint timing is not a seed-general formation law.

## Practical update

Future work should stop using diagonal R² alone as the timestamp for “map formation.” A better decomposition is:

1. architecture-level map-like predictability, measured with fixed meaningful text at initialization;
2. answer-distribution compatibility, whose first major crossover occurs around 88B tokens;
3. operator identity/retention, which does not reach the final-map regime until midtraining;
4. item-level retrieval, which improves mostly in midtraining and post-training.

That decomposition turns the apparent null into a substantive result: the scalar relation is not learned from zero, but the operator and the source of its predictability evolve dramatically.

## Ground-truth artifacts

- `eval_results/issue_2544/fits/diag_curve.json`
- `eval_results/issue_2544/fits/cross_cells.json`
- `eval_results/issue_2544/transfer/retention_matrix.json`
- `eval_results/issue_2544/fits/kshot_curve.json`
- `eval_results/issue_2544/fits/layer_sweep.json`
- `eval_results/issue_2544/diagnostics/intersection_manifest.json`

These files are committed on branch `issue-2544` at or before `7d6d0f03f4781abdcbdeaf6167844f5683bbb1eb`.
