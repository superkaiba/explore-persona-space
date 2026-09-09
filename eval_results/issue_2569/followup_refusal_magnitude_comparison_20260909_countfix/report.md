# Does mapped-change magnitude improve refusal-change prediction?

Outcome: absolute difference in archived refusal rates. All 124 fixed layer-19 pairs; no selection on observed flips. Mapped magnitude is the unnormalized numerator.

| Score | Spearman rho | Paired family-bootstrap 95% CI | LOFO linear R2 | LOFO MAE |
|---|---:|---|---:|---:|
| context_norm | 0.7779 | [0.6512836808528104, 0.8398737843477002] | 0.5198 | 0.2399 |
| mapped_norm | 0.7624 | [0.6474321250314246, 0.8292433210129299] | 0.4853 | 0.2528 |
| observed_answer_norm | 0.7805 | [0.6784369969867451, 0.8391909530619136] | 0.5630 | 0.2275 |

Primary paired difference, mapped minus context: {'ci95': [-0.04434286484505601, 0.028860066179183897], 'valid_bootstrap': 2000, 'undefined_bootstrap': 0, 'delta_rho': -0.015590199848744546}.
Training-mean baseline: {'r2': -0.16157893880824603, 'mae': 0.44549773140614835, 'mse_skill_vs_training_mean': 0.0, 'n_predictions_outside_0_1': 0}.

## Category-adjusted rank association

{
  "n": 124,
  "n_clusters": 21,
  "scores": {
    "context_norm": {
      "rho": 0.5410273881252672,
      "ci95": [
        0.3747910510566164,
        0.6595057479536989
      ],
      "valid_bootstrap": 2000,
      "undefined_bootstrap": 0
    },
    "mapped_norm": {
      "rho": 0.5260695186621869,
      "ci95": [
        0.37428580423449737,
        0.6823930376265411
      ],
      "valid_bootstrap": 2000,
      "undefined_bootstrap": 0
    },
    "observed_answer_norm": {
      "rho": 0.6437455624393066,
      "ci95": [
        0.5143279412561484,
        0.7734908364192722
      ],
      "valid_bootstrap": 2000,
      "undefined_bootstrap": 0
    }
  },
  "mapped_minus_context": {
    "ci95": [
      -0.06853677789158305,
      0.07803267340090456
    ],
    "valid_bootstrap": 2000,
    "undefined_bootstrap": 0,
    "delta_rho": -0.014957869463080309
  },
  "category_adjusted": true
}

## Category-specific rank association

| Category | n | Context | Mapped | Observed answer |
|---|---:|---:|---:|---:|
| obj_benign | 8 | None | None | None |
| obj_flip | 16 | 0.7585 | 0.6458 | 0.8833 |
| subj_benign | 8 | None | None | None |
| subj_ctl | 16 | 0.4863 | 0.4714 | 0.6127 |
| verb_benign | 8 | None | None | None |
| verb_flip | 16 | 0.8126 | 0.7996 | 0.7234 |
| verb_harm | 16 | 0.6394 | 0.6672 | 0.8012 |
| xstest | 36 | 0.4577 | 0.4708 | 0.6201 |

Undefined correlations indicate constant outcomes, not zero association.

## Limitations

- Exploratory secondary analysis of one frozen bank; no new on-policy replication.
- Observed-answer norm is an empirical reference, not a mathematical ceiling and not a context-only predictor.
- Primary uncertainty treats all XSTest items as one corpus cluster; within-XSTest descriptive intervals resample items.
- Primary mapped-minus-context comparison is prespecified; other strata and adjustments are descriptive sensitivity checks.
- Category-adjusted association correlates global ranks after category-mean subtraction; does not control all topic/length confounds.
- Scalar OLS calibrations are disjoint at semantic-family level; predictions are not clipped to [0,1].
- No high-dimensional probe or new representation mapping fitted; no inference about jailbreak framing or causality.
- Outcome ties corrected from archived integer refusal counts; no labels or flip memberships changed. numeric_tie_audit.json records tiny changes to prior v3 point estimates without refitting axes.
