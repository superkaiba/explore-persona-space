# Supplementary readout validation correction

The first final supplement stopped at its readout normalization check. Its nine
files, failure traceback and exit status remain preserved in the
[failed-attempt archive](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ecde23482a75879c30c0300af121bab4fd1bdb0a/exploratory_workspace_jr/20260912/main_supplement_v1_failed_v1).
The failure did not invalidate the completed main comparison or decomposition
summary, which remain pinned to `8e44b6dfb43f2ce8d6b64de5433e9341022a1be3`.

The saved J/R directions are FP32 columns. NumPy's FP32 column-norm reduction
introduced enough accumulation error to fail the validator's existing
`rtol=1e-6, atol=1e-8` check. The primary J/R maximum norm errors were
1.55e-6/1.31e-6 in FP32, versus 7.89e-8/7.51e-8 when accumulating those identical
saved columns in FP64. Nine/eight columns falsely failed in the primary model;
one/two did so in the comparison model. None failed with FP64 accumulation.

The producing `per_direction_scores` routine already casts directions to FP64
before checking their norms and computing projections. Commit
`8435ae62ba55aa093a3a707106ad285d8c8691a3` makes the supplement validator follow
that same convention. It retains both tolerances, saved directions, predictions,
targets, matching decisions and scientific statistics. The independently reviewed
regression tests use 5,120-dimensional FP32 columns and retain rejection of
unnormalized, zero, nonfinite and corrupted inputs; 12 focused tests passed.

A bounded smoke on the same reporting worker loaded both models' actual,
upload-bound sources and passed the complete readout contract and every target
and predictor projection. It verified unchanged input hashes and emitted
`[readout-validation] norm_accumulation=float64 arms=J,R,random,pca` for each
model. Its proof SHA256 is
`d57c6fa7327c96cd0fa8c8437b33749610c103063d6459651b801ff601da7275`.

The corrected supplement uses the fresh `main_supplement_v2` output and exact
fix commit, with ancestry and served-file checks. It loads no failed-run output
and performs no refitting or retuning. The original failed service had PID zero
before the new service started. The wrapper and figure/export consumers received
separate independent review. This script is specific to this experiment; no
other running experiment imports the changed validator. Other successful
analysis consumers retain their original code pins.
