# Issue 2643 — context-SAE → answer-SAE behavior forecasting

## Question

Can a map from a prompt's layer-19 context-SAE code to its expected layer-19
answer-SAE code forecast or flag unusual behavior, and can the forecast be
compressed to a small, interpretable set of context features by gradient
pursuit?

## Frozen representations and map

Both dictionaries use Qwen2.5-7B-Instruct layer 19, width 32,768, BatchTopK
`k=128`, and the exact-replication LMSYS split.  The independently trained
context dictionary encodes the final prompt token (holdout FVE 0.912356); the
answer dictionary encodes the mean assistant-token state (holdout FVE
0.897153).  The factorized map is

```
context state → context SAE → context reconstruction
              → frozen #779 dense ridge → answer SAE → feature calibration.
```

The ridge was trained previously on 963,444 paired rows.  Only a nonnegative,
slope-only calibration was fitted here, using 68,155 ordinary training rows
from the pinned 150,000-row #2502 corpus.  All non-ordinary rows and every test
row were excluded from calibration.  The calibrated map's pooled held-out
answer-code R² is 0.370959.  Its dense context-SAE route has held-out R²
0.123528, versus 0.114814 for the raw-context ridge.

## Descriptive unusual-regime screen

The screen has 22,497 test rows, of which 35.1% belong to a non-ordinary source
family.  These labels describe corpus regimes, not verified bad behavior.  The
best pre-answer map statistic is predicted-code rarity (AUROC 0.6403, AP
0.4527).  The best post-answer map residual is emergent feature mass (AUROC
0.6602, AP 0.4532), but realized answer-SAE sparsity alone is stronger (AUROC
0.6780).  This therefore does not establish a general weird-behavior detector.

## Over-refusal organism

The confirmatory behavior panel replays one archived rollout for each of 30
personas × 50 benign requests through the pinned #642 `loraRefOP_step132`
adapter.  Frozen #642 refusal labels are reused; there are no new judge calls.
The answer-SAE refusal direction and matched direct-context probes are fitted
on claims 0–24 and evaluated on claims 25–49 (750 rows, 81 refusals), clustered
by persona/request.

The full mapped-answer-SAE score detects refusal with AUROC 0.95298 (cluster
bootstrap 95% CI 0.93609–0.96832; AP 0.69893).  Direct context-SAE and dense
probes reach 0.95835 and 0.95966, respectively, so the map transfers the
answer-space concept but does not add predictive information over the context
state.  The realized answer-SAE oracle reaches 0.98262.

## Gradient pursuit

This arm follows the recent #1482 convention: 128 candidate context features
are ranked by the largest absolute standardized coefficient in the factorized
map's local linearization; signed pursuit greedily selects the maximum
normalized residual correlation and jointly refits the whole support with a
relative ridge of `1e-3`.  The ladder is `k={1,2,4,8,16}`.  Controls retain the
coefficient-ranked support with either fixed weights or the identical joint
refit.  Pursuit targets the full nonlinear mapped refusal score on readout-fit
rows only, never evaluation labels.

At `k=16`, pursuit reproduces 0.96699 of the held-out full-map score variance,
versus 0.95013 for coefficient-ranked joint refit and 0.68944 for fixed
coefficients.  Refusal AUROC is 0.94866 (AP 0.66130), compared with 0.94641 for
joint refit and 0.95298 for the full map.  Paired cluster-bootstrap AUROC
contrasts show no reliable difference: pursuit minus joint refit is +0.00225
(95% CI −0.00446 to +0.00952), and pursuit minus the full map is −0.00432
(−0.01332 to +0.00402).  Split-half support Jaccard falls from 1.0 at `k=1–2`
to 0.231 at `k=16`, so the compact score is reproducible as a predictor but its
individual 16 edges should not be interpreted as stable mechanisms.

## Unavailable conditional-marker panel

The planned #382 `[ZLT]` sleeper-style marker replay could not be completed.
The pinned historical Hub revision still exposes the checkpoint tree and small
metadata files, but authenticated downloads of the merged model's LFS shards
return HTTP 403.  Base-model activations were not substituted because they
would not be activations from the model organism.  Restoring those archived
shards would make the already-implemented three-seed panel runnable.

## Interpretation

The result supports a narrow claim: an answer-space refusal concept can be
forecast from context-SAE codes and compressed to 16 context features with
little loss.  It does not support a universal anomaly detector, a causal route
claim, or an advantage over a label-budget-matched direct context probe.

Artifacts are under `eval_results/issue_2643/` in the repository and
`issue2643_sae_map/` in the data repository.
