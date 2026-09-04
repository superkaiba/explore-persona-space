# The L19 context-answer map in PCA and SAE bases (task #2569, leg 11)

The PCA calculation is performed separately in raw residual coordinates and in the ridge map's standardized coordinates. `map_gain` is the Euclidean gain of a unit PC; `kernel_share` is its squared projection into the effective kernel at the 99% squared-singular-mass cutoff; predicted impact is `eigenvalue × map_gain²`.

## PCA summary

| coordinate | effective-kernel dim | context variance in kernel | PCs for 50% context variance | PCs for 50% predicted variance |
|---|---:|---:|---:|---:|
| raw | 1976 | 0.834 | 13 | 12 |
| standardized | 2121 | 0.833 | 20 | 15 |

The JSON contains every PC plus the ten largest ignored-variance PCs and ten highest-impact read PCs, each grounded by high/low real contexts and its closest context-SAE decoder directions.

## SAE variance accounting

| term | effective kernel | read range |
|---|---:|---:|
| Feature diagonal | 0.343 | 0.097 |
| Feature covariance cross-term | 0.448 | 0.044 |
| SAE unexplained residual | 0.064 | 0.030 |
| 2 × reconstruction-residual covariance | -0.022 | -0.005 |
| Total context variance | 0.834 | 0.166 |

Accounting identity relative error: 0.000e+00.
Feature rankings are diagonal attributions, not causal or additive semantic units: correlated SAE features contribute the separately reported covariance term. The top ignored-kernel and read-range feature lists include top-activating context excerpts; existing analyst readings are included only where leg 8 had already supplied one.

## Scope

This characterizes one fitted ridge operator. Effective-kernel means low gain for this linear predictor, not that the underlying language model discards the information.
