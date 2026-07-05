---
name: HF datasets streaming IterableDataset shutdown SIGABRT
description: A streaming IterableDataset surviving to interpreter shutdown SIGABRTs (rc=134) AFTER all work completed — del + gc.collect() at the call site fixes it; #952 r2.
type: feedback
---

A `datasets` streaming `IterableDataset` (e.g. the LMSYS `load_dataset(...,
streaming=True)` replay) that survives to INTERPRETER SHUTDOWN aborts the
process with SIGABRT rc=134 and `terminate called without an active
exception` — AFTER the final log line, with every output already written.
Deterministic in the pinned datasets/pyarrow env (bisected #952 r2 with a bare
15-row streaming loop; NOT caused by surrounding code). The #654 gotcha covers
the WRAPPER side (check the artifact before treating rc as fatal); this is the
IN-PROCESS fix.

**Why:** a clean pod run exiting rc=134 is classified as a workload crash by
the GCE EXIT trap (crash-persist + `eps/phase=failed` + poweroff) despite a
complete run — a false-failure cycle. Local smokes read as mysterious rc=134
"crashes" that greps show completed.

**How to apply:** release the streaming dataset DETERMINISTICALLY while the
interpreter is healthy — `del row, ds; gc.collect()` immediately after the
consuming loop (verified: rc goes 134 → 0). Do NOT reach for `os._exit(0)`
(masks genuine finalize failures). Worked example:
`run_952.phase0_verify` + `issue952_stats._reconstruct_lmsys_prompts`
(#952 commit 3a95b2e7a8).
