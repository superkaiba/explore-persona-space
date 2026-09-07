# Qwen3-8B CoT rank reproduction — fixed analysis plan

User approval: 2026-09-07, one model from the current CoT section, separate
validation and test folds, CPU-only cached-state analysis, driven to completion.

Question: do end-of-CoT states predict the same eventual answers with fewer
effective input dimensions and a lower required linear mapping rank than context
states? This reproduces the earlier scaling-panel comparison on the paper's CoT
model, not on newly generated data.

Use issue2546 Qwen3-8B layer24, d=4096, 33,810 original paired rows from MATH,
GSM8K train, ContextHub, MMLU, ARC-Challenge, CommonsenseQA and PIQA. Reuse the
exact original five random-row folds. Targets are thinking-on answer-mean states
for both context-last and CoT-boundary inputs. No necessity-only filtering.

For outer test fold k, reserve original fold (k+1)%5 for validation and fit the
other three. Choose the smallest projection rank allowing at most 10% extra
validation SSE versus that map's full prediction (inherited issue2588 criterion).
Rank bases are principal directions of fitted TRAIN outputs, never test outputs.
Refit on all four outer training folds, recompute the training-only rank basis,
and evaluate the already selected rank on outer test. Report 5% and 20% SSE
sensitivities without new fits. This is nested rank selection only: original
penalties and layer are frozen, not newly nested hyperparameter selection.

Inherited issue2546 ridge: float64, training mean and sample standard deviation
plus 1e-9 on inputs, mean-centered targets, penalties 1000 (context) and
316.2277660168379 (CoT end). Verify all outer predictions against saved float32
OOF values using the previous Qwen3 geometry check's rtol=atol=2e-6, and pooled
paper R² to 1e-9. Reuse verified full-map retrieval hits; calculate reduced-map
and identity-plus-training-bias retrieval with training-only whitening and CSLS
k=10, pool6762 (chance1/6762). Use the paper's train-per-corpus-mean R² baseline.

Input dimension is raw centered training covariance participation ratio,
trace(C)^2 / ||C||F², computed exactly without SVD. Report all-row descriptive
PR too. Present five paired folds and ranges, not independent-model replicates
or a five-fold significance test. No inference to a population of models from
n=1; report either result direction faithfully. Effective dimension does not
measure the number of semantic concepts in reasoning.

Compute: share per-fold float64 sufficient statistics; one eigendecomposition
per fit supplies all4097 ranks. Twenty fits total (two states × five folds ×
inner/outer), eight CPU threads; no model inference or rented GPUs. Checkpoint
each outer state/fold. Preserve source hashes, full curves, selected ranks,
baseline/retrieval metrics and a complete sentinel. New outputs stay on the root
disk because the cached-activation data volume is nearly full. No source stores
are copied or removed. Timing is measured on the first completed fold.

Out of scope: new models/data, capability correlations, new penalty/layer sweeps,
GPU provisioning, nonlinear maps, causal claims, and automatic manuscript edits
before the result has been verified.
