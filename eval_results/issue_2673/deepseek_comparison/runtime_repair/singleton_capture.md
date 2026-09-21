# DeepSeek singleton extraction, 2026-09-21

The user requested unbatched extraction after attempt 10 failed the mixed-batch
numerical parity check. Production now processes one unpadded context per model
forward for DeepSeek as well as Qwen. Eight-row files remain storage containers;
the 1,920-context coverage, layer selection, and downstream raw cosine and
uncentered whitened cosine analysis are unchanged.

This change is based on runtime source `1a7cd412cceefd01bc03765f2544b7cc4afb917e`
and retains its reviewed grouped-FP8 M16 implementation. The declared
`execution_mode` now controls production forwarding and the relevant smoke gate.
Mixed-batch smoke calls remain diagnostic and their vectors are saved; their
disagreement does not reject singleton production. Reverse-order singleton
repeatability, hook-versus-tuple agreement, finite BF16 outputs, coverage,
provenance, immutable publication, measured-throughput, and deadline gates remain
mandatory. Changed execution configuration enters the fingerprint, preventing
reuse of a store from an incompatible execution mode.

## Evidence and validation

Attempt 10's 18 singleton contexts repeated bit-for-bit, with maximum relative
error zero. Its mixed-batch comparison reached relative error
0.25390228629112244. These are diagnostic results, not completed production
capture. Evidence is pinned at Hugging Face dataset
`superkaiba1/explore-persona-space-data`, revision
`e576f58dc881b4ba956206ead96c314e414903eb`, prefix
`issue2673_deepseek_comparison/20260921_v6/deepseek/failure_1789979599602088036`.

Local capture and analysis suites passed all 35 tests; an independent read-only
review also ran capture, analysis, and artifact suites (50 passed), with no
blocking findings. Behavioral tests verify unpadded size-one forwards, preserved
row/file ordering, failed-repeat rejection, retained padded-mode parity rejection,
and deadline enforcement. Ruff and whitespace checks passed. These are CPU
tiny-model tests; a new full-model GPU run has not been launched.

## Compute boundary

`singleton_sizing.json` records calculations from the allocation ledger and
attempt 10's saved progress timestamps. The existing 17 GPU-hour cumulative cap
has 1.847939 GPU-hours remaining, or 13.86 minutes on eight GPUs. This is below
the existing 15-minute preservation reserve even before loading the model.

The last provider creation-to-first-vector interval was 41.90 minutes. Between
singleton progress counts 1 and 18, the mean interval was 3.368 seconds/context;
between counts 8 and 18 it was 0.273 seconds/context. These coarse intervals
include compilation and host bookkeeping and are not a steady-state throughput
benchmark. Conservatively applying the slower interval to all 1,920 contexts,
with the existing 1.25 margin and 900-second reserve, gives 3.194 wall-hours on
eight GPUs (25.55 GPU-hours). A proposed one-attempt ceiling is 3.5 wall-hours
(28 GPU-hours), with early release on completion/failure and the real three-chunk
timing gate still mandatory. This allowance is proposed, not approved.

The live controller remains budget-blocked on its existing runtime source. No
allocation, budget extension, retry rearm, or production result is claimed by
this change. A newly approved run must use the committed singleton source and a
fresh output version, preserve the prior attempt's evidence, and retain monitored
recovery, acknowledged alerts, upload verification, and the cumulative cap.
