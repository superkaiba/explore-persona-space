# Paper-matched Qwen Base-to-Instruct transfer

User-authorized inline, existing-artifact CPU analysis under task 2054, 2026-09-17.
No new model inference, training, task, or change to the parent goal/status.

## Question and decision criteria

Does the older observation of stronger frozen cross-checkpoint map transfer for
story speakers than chat persist under the current paper's five-answer protocol?
Evaluate all six requested settings: Chat, Assistant-story, HELIOS, Wren, Dana,
Vex. Report all of them regardless of outcome. Base-to-Instruct is a checkpoint
comparison; it does not isolate supervised fine-tuning from other post-training.

Primary outcomes are held-out frozen transfer R2 and unwhitened cosine/Euclidean
retrieval at 1, 5 and 10, with pool sizes and chance levels. Retention is the ratio
of equal-five-fold mean frozen R2 to mean target-own R2. Comparisons are descriptive;
fold variation does not constitute independent replicate significance testing.

## Protocol and provenance

Reuse the Qwen2.5-7B and Qwen2.5-7B-Instruct banks at layer 19 (3584 dimensions),
prefill-alone context states and mean-of-five on-policy answer token-mean states.
Keep original complete-five cohorts, including the paper's existing caps, with
the global five conversation folds at seed 137. The original per-setting source
and target training cohorts may differ because generation completeness differs.
The primary test set is each full original Instruct fold; additionally report a
shared-ID sensitivity using the same estimators and a restricted retrieval pool.

Original K5 banks: HF dataset `superkaiba1/explore-persona-space-data`, revision
`9de026f872c19b2ca4fd3e4539de820e08038ee3`. Assistant-story banks: revision
`400ad464ce9f092722f737dd336fc31a03fba524`. Verify each pinned LFS hash and size
against the actual local bank. Original fold-map source: Git
`957c454a8ec9a2f520bb754543e7494904636e20`; file SHA256
`4ab1839a0e8c5e8705147cbb529b2df36975ac46b987fe71ab3f919265e4c39e`.

Restore the original standardized ridge fits with their banked GCV penalties,
using float64 batched Cholesky and raw-space A,b conversion. Verify both own-map
endpoints against original fold R2 (absolute tolerance 1e-6) and top-1 retrieval
(1e-12) before accepting any fold. The original recipe is GCV over 13 log-spaced
penalties from 1e-2 to 1e4, with effective degrees of freedom <=0.9 n_train;
there is no new hyperparameter selection here. Frozen predictions apply the
Base normalization, coefficients and intercept directly to Instruct contexts.
Bias-only recalibration uses only target training folds. Include source-trained
and target-trained identity-plus-bias baselines. All methods within a setting
are evaluated against the exact same target rows. Each R2 uses the test-fold
target mean for SST; aggregate scores with equal fold weighting, as in the
paper's speaker section. Do not substitute the OLMo section's pooled R2.

## Execution and outputs

Dedicated worktree: `/mnt/eps-data/thomasjiralerspong/wt-2054-k5-stage-transfer`.
Output: `/home/thomasjiralerspong/eps-runs/issue2054-k5-stage-transfer/production_v1`.
Use the existing repository environment, eight BLAS threads, allocator caps,
and OOM protection. Validate numerical equivalence in focused tests before
production. Expected footprint <2 GiB beyond reused banks; no new GPU cost.
The first completed production fold provides the measured full-dimension pilot.

Run under a detached systemd monitor, observed by the repository's independent
experiment watchdog. Verify real Codex recovery canary and acknowledged personal
notification delivery before relying on unattended execution. Keep checkpoints
fingerprinted to source, input bytes, methods and versions. Resume only verified
fold packets. Bounded recovery must prove old workers stopped and require fresh
progress. Never launch a second concurrent driver.

Persist float32 frozen/own predictions, float64 per-query SSE/SST, conversation
IDs, exact map reconstruction inputs/penalties and hashes, fold metrics, summaries
and manifest. Dense maps are deliberately not duplicated: each can be restored
from the pinned banks and exact original penalties; their float64 hashes provide
an equivalence check. This saves local disk while retaining reproducibility.
Upload all new analysis artifacts to HF and verify remote hashes before marking
completion. Commit the small results, verification, reproducible scripts, and a
proposed results-section paragraph. Do not edit the live manuscript without a
request. The parent clean-result owner can absorb the completion marker on its
next analysis pass; preserve other concurrent task work.
