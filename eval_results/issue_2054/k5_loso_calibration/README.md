# K5 leave-one-setting-out transfer with calibration

[Open the calibrated transfer figure](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/27f8650a48a0ac91bc5e124a2ebc9026d9e02885/issue2054_section44_k5_gcp/transfer_calibration_v1/loso/figures/leave_one_setting_out_calibrated.png).

This analysis extends the completed six-setting K5 experiment: base and
instruction-tuned Qwen2.5-7B, the chat and plain-text assistant settings, and
HELIOS, Wren, Dana, and Vex in the attributed-story format. Each target is the
mean activation of five sampled answers. It retains the parent's layer,
conversation folds, answer filtering, and test sets.

For each target setting and conversation fold, the source map is fitted on the
other five settings after excluding that conversation fold everywhere. The
restored map must match the parent's selected ridge penalty and held-out
predictions (relative L2 tolerance 1e-6). The plotted frozen predictions are
the exact saved parent predictions, rather than the restored approximation.

Two calibrations use only the target setting's four training folds:

* **Bias:** add the mean training residual, `mean(y - prediction)`.
* **Bias + scaling:** fit one scalar `a` and one vector intercept, minimizing
  summed squared error across examples and activation coordinates. Equivalently,
  `a = sum((p-p_mean)*(y-y_mean)) / sum((p-p_mean)**2)`, with test prediction
  `a*(p_test-p_mean) + y_mean`.

The source map is fixed during calibration. The same source map produces both
the target training predictions and the held-out predictions; other-fold
out-of-fold predictions are never used to calibrate the current fold. Thus the
calibrated curves are supervised adaptation with target training labels. Only
the frozen curve measures transfer with no target-setting labels.

R² uses held-out target means in its denominator and is averaged equally over
the five folds, matching the parent. Individual fold marks are not confidence
intervals. The separate-map reference is the original map trained on the target
setting's training folds. The JSON also reports identity plus target-training
bias and nearest-neighbor retrieval (Euclidean and cosine, top 1/5/10, with
each fold's candidate-pool size and chance level).

All 72 parent input files are pinned by Hugging Face revision and SHA-256 in
`inputs.json`. The source report is the completed
[K5 LOSO result](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/98d350d78e94fd5e79685be70376919147ee0cc7/issue2054_section44_k5_gcp/leave_one_setting_out_v1/results.json).
Raw K5 banks are at revision `9de026f872c19b2ca4fd3e4539de820e08038ee3`;
saved test predictions are at `534f9b67838897671bb35f5fb2a73059bd0f9fa7`.
The shared five-fold map is pinned to Git commit
`957c454a8ec9a2f520bb754543e7494904636e20`, seed 137.

Reproduction entry point: `scripts/issue2054_k5_loso_calibration.py`, using
`--stage stage`, then `--stage fit --model qwen2.5-7b` and
`--stage fit --model qwen2.5-7b-instruct`, then `--stage plot`, each with the
same `--out` directory. All fits use the established source-only GCV ridge
recipe (`logspace(-2, 4, 13)`, degrees-of-freedom cap 0.9, standardized inputs).
No new language-model generations or activation captures are performed.

These are representation-prediction measurements. They do not establish that
personas are constant steering vectors, or measure qualitative response and
refusal differences. The character setting also changes the conversation
format; this is not a controlled system-prompt-only manipulation.

## Results

All 12 panels completed all five folds (60/60).

| Checkpoint / target | Frozen | + Bias | + Bias + scaling | Separate map |
|---|---:|---:|---:|---:|
| Base / dana | 0.539 | 0.540 | 0.542 | 0.544 |
| Base / helios | 0.493 | 0.506 | 0.508 | 0.530 |
| Base / vex | 0.467 | 0.475 | 0.478 | 0.482 |
| Base / wren | 0.537 | 0.538 | 0.539 | 0.529 |
| Base / Assistant plain text | -0.045 | 0.224 | 0.255 | 0.469 |
| Base / Assistant chat | -0.198 | 0.116 | 0.179 | 0.410 |
| Instruct / dana | 0.547 | 0.550 | 0.550 | 0.561 |
| Instruct / helios | 0.501 | 0.512 | 0.514 | 0.547 |
| Instruct / vex | 0.480 | 0.490 | 0.493 | 0.512 |
| Instruct / wren | 0.556 | 0.557 | 0.557 | 0.554 |
| Instruct / Assistant plain text | -0.414 | 0.115 | 0.285 | 0.454 |
| Instruct / Assistant chat | 0.406 | 0.503 | 0.504 | 0.675 |

## Retrieval and identity-plus-bias baseline

Euclidean top-1 retrieval is averaged over the same five folds. Each fold uses its held-out targets as the candidate pool: 1543–1659 candidates; chance is 0.0603–0.0648%. Copy + bias below learns its bias from target training folds.

| Checkpoint / target | Copy + bias R² | Frozen top-1 | + Bias top-1 | + Bias + scaling top-1 |
|---|---:|---:|---:|---:|
| Base / dana | -0.780 | 74.66% | 74.93% | 78.71% |
| Base / helios | -0.998 | 64.30% | 67.02% | 71.91% |
| Base / vex | -1.375 | 77.16% | 77.27% | 71.84% |
| Base / wren | -1.044 | 75.53% | 75.53% | 77.39% |
| Base / Assistant plain text | -1.813 | 11.69% | 16.64% | 7.90% |
| Base / Assistant chat | -2.515 | 7.63% | 9.00% | 2.41% |
| Instruct / dana | -0.741 | 73.32% | 74.18% | 76.52% |
| Instruct / helios | -0.893 | 64.62% | 65.36% | 69.63% |
| Instruct / vex | -1.271 | 74.96% | 75.06% | 70.06% |
| Instruct / wren | -0.975 | 75.28% | 75.28% | 77.43% |
| Instruct / Assistant plain text | -2.888 | 27.65% | 37.47% | 15.77% |
| Instruct / Assistant chat | -0.955 | 45.85% | 55.59% | 60.39% |

The scalar is fitted to minimize training squared error. A better held-out R² can coincide with lower nearest-neighbor retrieval; neither metric should be treated as a substitute for the other. Cosine retrieval and top-5/top-10 results are retained in the JSON.
