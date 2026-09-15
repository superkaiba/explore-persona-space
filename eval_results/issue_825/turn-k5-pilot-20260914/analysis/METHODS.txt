# Single versus five-answer turn-transfer pilot

This approved #825 follow-up uses 1,000 selected real conversation histories,
both base and instruction-tuned models, and assistant turns 1 and 12. The GPU
stage generates five fresh answers independently for each model/history/turn.
Actual retained coverage, truncation counts, and exclusions are reported with
the completed results; planned coverage is not substituted for realized data.

Each target is the mean hidden state over one answer's tokens at decoder block
19 (the captured block is checked against `hidden_states[1:][19]` on the same
forward pass). K=1 uses draw zero. K=5 gives each of the five answer means equal
weight, regardless of answer length. The exact context-token prefix must agree
across draws. Captured context vectors must satisfy the registered numeric
parity gate (cosine at least 0.995); draw zero supplies the identical context
features for both K arms. A conversation is retained only when both models
have all five valid draws at both endpoint turns.

Six outer folds hold out whole conversations, using the shared #825 grouped
split with seed 0. The same conversations and folds are used across models,
turns, training K and evaluation K. Within each outer-training fold, four
inner grouped folds (seed 4242 plus the outer-fold index) select ridge strength
from 13 log-spaced values between 0.01 and 10,000. Standardization uses only
the appropriate training rows, with an unbiased feature standard deviation
plus 1e-9. Fits and calibration use CPU float64. The existing corrected #825
batched inner-group-CV implementation selects ridge strength, with no GCV
fallback. Feature-dependent inner caches and the outer Gram decomposition
are shared across K=1 and K=5.

A map fitted at turn 1 is evaluated at turn 12. Its coefficient matrix remains
fixed while target-turn training conversations estimate either a vector bias
or one global scalar plus a vector intercept. Calibration targets use the
training K. They are not changed when evaluating the other K. The baseline
copies the target-turn context and fits a target-training vector bias. A
separate turn-12 ridge fit provides the own-turn reference. All four training-K
by evaluation-K combinations are retained, permitting within-pilot comparisons
that distinguish changing the training target from changing the test target.

Held-out R² pools the sum of squared errors across folds and divides by the
sum of squared deviations from each original test fold's own target mean.
Cosine and Euclidean top-1 retrieval use only that fold's held-out answer bank;
realized candidate-pool sizes and chance levels accompany every model summary.

One thousand paired conversation bootstrap replicates (seed 0) reuse the
saved out-of-fold residuals, retrieval hits and original fold-centered target
deviations. These intervals condition on the fitted maps and captured answer
bank; they do not include refitting or repeated-generation uncertainty.
Retention ratios are reported only when the own-turn R² is positive and its
95% interval excludes zero. Paired differences cover both K changes and
calibration changes.

The fit is stored exactly in dual form (training-normalized contexts, dual
coefficients and training means/scales), avoiding redundant dense 3,584 by
3,584 matrices. The 24 map archives, 12 fold prediction archives and two
out-of-fold row archives preserve the numerical inputs required to reconstruct
predictions and regenerate all reported statistics. Every artifact is hashed.

This is target-informed calibration on logged conversation histories, not a
zero-shot transfer test or a rollout in which the five sampled answers replace
the remainder of the logged conversation. This pilot also uses a corrected
ridge-selection recipe and a smaller matched panel than the earlier K=1
analysis; differences from that older run cannot be attributed solely to K.
