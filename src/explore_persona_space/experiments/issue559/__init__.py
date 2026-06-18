"""Issue #559 cross-behavior self-scoring amendment.

Generalizes the marker-channel "predict-before-training" result to three
content behaviors (sycophancy, refusal, emergent misalignment) by scoring the
untrained base model's own answers and ranking held-out personas by that base
self-score against the already-committed cross-persona TRAINED LEVEL.

The graded 0-100 intensity rubrics that the predictor of record uses do not
exist elsewhere in the codebase; they live in :mod:`judge_rubrics`.
"""
