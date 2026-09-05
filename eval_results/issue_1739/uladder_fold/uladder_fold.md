# Issue 1739 unlabeled-data ladder

Seed-level inference uses n=5, TOST alpha=0.05, and equivalence margin delta=0.020.

| Behavior | Setting | Ladder | Verdict | Trend mean [95% CI] | Endpoint equivalent | Judged equivalent |
|---|---|---|---|---:|---:|---:|
| evil | in_dist | generic_only | REFUTED | 0.650 [0.630, 0.670] | False | False |
| evil | generic | generic_only | UNRESOLVED | 0.564 [0.124, 1.005] | False | False |
| evil | ood | generic_only | REFUTED | 0.193 [-0.125, 0.510] | False | False |
| evil | in_dist | union_scaled | REFUTED | 1.000 [1.000, 1.000] | False | False |
| evil | generic | union_scaled | UNRESOLVED | 0.793 [0.632, 0.954] | False | False |
| evil | ood | union_scaled | REFUTED | 0.171 [-0.514, 0.857] | False | False |
| sycophancy | in_dist | generic_only | REFUTED | 1.000 [1.000, 1.000] | False | False |
| sycophancy | generic | generic_only | REFUTED | 0.893 [0.753, 1.033] | False | False |
| sycophancy | ood | generic_only | REFUTED | 0.957 [0.937, 0.977] | False | False |
| sycophancy | in_dist | union_scaled | REFUTED | 1.000 [1.000, 1.000] | False | False |
| sycophancy | generic | union_scaled | REFUTED | 0.971 [0.914, 1.029] | False | False |
| sycophancy | ood | union_scaled | REFUTED | 0.793 [0.644, 0.941] | False | False |
| hallucination | in_dist | generic_only | REFUTED | 0.957 [0.937, 0.977] | False | False |
| hallucination | generic | generic_only | REFUTED | 0.950 [0.926, 0.974] | False | False |
| hallucination | ood | generic_only | REFUTED | 0.921 [0.884, 0.959] | False | False |
| hallucination | in_dist | union_scaled | REFUTED | 1.000 [1.000, 1.000] | False | False |
| hallucination | generic | union_scaled | REFUTED | 0.964 [0.910, 1.019] | False | False |
| hallucination | ood | union_scaled | REFUTED | 0.986 [0.961, 1.010] | False | False |

The generic-only ladder is primary. Union-scaled rows are secondary.
The full-U fold-clean union diagnostic is descriptive and cannot change a verdict.
