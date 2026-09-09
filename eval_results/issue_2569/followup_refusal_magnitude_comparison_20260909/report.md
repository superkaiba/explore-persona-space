# Does mapped-change magnitude improve refusal-change prediction?

Outcome: absolute difference in archived refusal rates. All 124 fixed layer-19 pairs; no selection on observed flips. Mapped magnitude is the unnormalized numerator.

| Score | Spearman rho | Paired family-bootstrap 95% CI | LOFO linear R2 | LOFO MAE |
|---|---:|---|---:|---:|
| context_norm | 0.7782 | [0.6524148538959305, 0.8404521344773596] | 0.5198 | 0.2399 |
| mapped_norm | 0.7622 | [0.6485797393403373, 0.8299772267425015] | 0.4853 | 0.2528 |
| observed_answer_norm | 0.7811 | [0.6789245229648967, 0.8400677348253461] | 0.5630 | 0.2275 |

Primary paired difference, mapped minus context: {'ci95': [-0.04384612376556637, 0.02841152188940142], 'valid_bootstrap': 2000, 'undefined_bootstrap': 0, 'delta_rho': -0.01606072697113592}.
Training-mean baseline: {'r2': -0.16157893880824603, 'mae': 0.44549773140614835, 'mse_skill_vs_training_mean': 0.0, 'n_predictions_outside_0_1': 0}.

## Category-adjusted rank association

{
  "n": 124,
  "n_clusters": 21,
  "scores": {
    "context_norm": {
      "rho": 0.5410482990487431,
      "ci95": [
        0.3749794533888716,
        0.6595899714152265
      ],
      "valid_bootstrap": 2000,
      "undefined_bootstrap": 0
    },
    "mapped_norm": {
      "rho": 0.5250874377641522,
      "ci95": [
        0.3703968500432129,
        0.6814411607853017
      ],
      "valid_bootstrap": 2000,
      "undefined_bootstrap": 0
    },
    "observed_answer_norm": {
      "rho": 0.6443437010352443,
      "ci95": [
        0.5112817822381619,
        0.775787381856038
      ],
      "valid_bootstrap": 2000,
      "undefined_bootstrap": 0
    }
  },
  "mapped_minus_context": {
    "ci95": [
      -0.06726135954216893,
      0.0766692685462073
    ],
    "valid_bootstrap": 2000,
    "undefined_bootstrap": 0,
    "delta_rho": -0.015960861284590844
  },
  "category_adjusted": true
}

## Category-specific rank association

| Category | n | Context | Mapped | Observed answer |
|---|---:|---:|---:|---:|
| obj_benign | 8 | None | None | None |
| obj_flip | 16 | 0.7585 | 0.6458 | 0.8833 |
| subj_benign | 8 | None | None | None |
| subj_ctl | 16 | 0.4711 | 0.4606 | 0.6018 |
| verb_benign | 8 | None | None | None |
| verb_flip | 16 | 0.8126 | 0.7996 | 0.7234 |
| verb_harm | 16 | 0.6473 | 0.6831 | 0.8132 |
| xstest | 36 | 0.4616 | 0.4732 | 0.6222 |

Undefined correlations indicate constant outcomes, not zero association.

## Limitations

- Exploratory secondary analysis of one frozen bank; no new on-policy replication.
- Observed-answer norm is an empirical reference, not a mathematical ceiling and not a context-only predictor.
- Primary uncertainty treats all XSTest items as one corpus cluster; within-XSTest descriptive intervals resample items.
- Primary mapped-minus-context comparison is prespecified; other strata and adjustments are descriptive sensitivity checks.
- Category-adjusted association correlates global ranks after category-mean subtraction; does not control all topic/length confounds.
- Scalar OLS calibrations are disjoint at semantic-family level; predictions are not clipped to [0,1].
- No high-dimensional probe or new representation mapping fitted; no inference about jailbreak framing or causality.
