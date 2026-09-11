# Training answer rollout count

Separate linear maps were fitted at each training K=1–10 using the same 19,000 contexts. The primary evaluation holds ten answer rollouts per test context fixed.

| Training K | Predictor | Held-out R² [95% CI] | Top-1 retrieval %, [95% CI] |
|---:|---|---:|---:|
| 1 | Linear map | 0.7736 [0.7586, 0.7878] | 96.39 [95.22, 97.56] |
| 1 | Copy + bias | -0.9588 [-1.0152, -0.9147] | 72.61 [69.85, 75.37] |
| 5 | Linear map | 0.7862 [0.7696, 0.8026] | 96.71 [95.54, 97.88] |
| 5 | Copy + bias | -0.9583 [-1.0147, -0.9140] | 72.61 [69.85, 75.37] |
| 10 | Linear map | 0.7871 [0.7708, 0.8032] | 96.82 [95.65, 97.88] |
| 10 | Copy + bias | -0.9582 [-1.0146, -0.9139] | 72.61 [69.85, 75.37] |

Retrieval uses 942 fixed candidates (chance 0.1062%); R² uses all 1,000 test rows. Intervals use 2,000 paired prompt-cluster bootstrap draws and condition on the fixed training bank, fitted maps, observed rollouts and retrieval pool. Whitening is fixed from the original 19,000 training answers; two-sided CSLS uses neighborhood 10.

The pre-registered primary contrast is training K10 minus K1 at evaluation K10:
R² difference 0.0135 [0.0111, 0.0162]; top-1 difference 0.42 [-0.42, 1.17] percentage points.

Completed coverage: 10/10 training-K fits; 100/100 train/eval cells per predictor. Lambda-grid edge selections: none. Expanded-grid diagnostics, when present, did not change reported predictions.

The fixed answer order is original, 43, …, 51. The curve is conditional on this order; it does not average all subsets or estimate training-bank sampling uncertainty.

![training k fixed eval10](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/254563c9f0d2be25bd9c0a56b6b766570c90f039/issue1901_training_k10/analysis_4b87bdc1deba7390/report/training_k_fixed_eval10.png)

[Vector PDF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/254563c9f0d2be25bd9c0a56b6b766570c90f039/issue1901_training_k10/analysis_4b87bdc1deba7390/report/training_k_fixed_eval10.pdf) · [Grayscale audit](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/254563c9f0d2be25bd9c0a56b6b766570c90f039/issue1901_training_k10/analysis_4b87bdc1deba7390/report/training_k_fixed_eval10_grayscale.png)

![training eval k grid](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/254563c9f0d2be25bd9c0a56b6b766570c90f039/issue1901_training_k10/analysis_4b87bdc1deba7390/report/training_eval_k_grid.png)

[Vector PDF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/254563c9f0d2be25bd9c0a56b6b766570c90f039/issue1901_training_k10/analysis_4b87bdc1deba7390/report/training_eval_k_grid.pdf) · [Grayscale audit](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/254563c9f0d2be25bd9c0a56b6b766570c90f039/issue1901_training_k10/analysis_4b87bdc1deba7390/report/training_eval_k_grid_grayscale.png)

Of 95,000 new answers, 7,353 reached the 1,024-token cap and 4 were changed by the inherited generated-JWT handler. These counts were verified against raw text/token files whose hashes match the capture arrays. Original token IDs and pre-edit text hashes are retained; capture used the disclosed edited text. See generation_audit.json for per-chunk evidence.
