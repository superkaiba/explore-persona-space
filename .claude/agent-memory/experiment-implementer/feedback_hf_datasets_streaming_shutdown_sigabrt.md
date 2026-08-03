---
name: HF datasets streaming IterableDataset shutdown SIGABRT
description: A streaming IterableDataset surviving to interpreter shutdown SIGABRTs (rc=134) AFTER all work completed — release via explicit `it = iter(ds)` + `it.close()` before del/gc (required for `break`-exit loops); #952 r2 + #1947 refinement.
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
interpreter is healthy. **For a bounded scan that exits via `break`, iterate
via an EXPLICIT iterator handle**: `it = iter(ds)`, consume with `for row in
it: ... if n >= cap: break`, then `it.close()` BEFORE `del row, ds;
gc.collect()`. The bare `del row, ds; gc.collect()` shape ALONE is
insufficient on `break` — the suspended anonymous for-loop iterator still
references the streaming pipeline and survives to shutdown (per the #1947
report — rc=134 → 0 verified 2026-08-01 on WildChat-1M in
`scripts/issue1947_datagen.py`, only after the explicit `it.close()`). For
a fully-consumed loop (no `break`) the plain `del row, ds; gc.collect()`
still suffices (the iterator is already exhausted). Do NOT reach for
`os._exit(0)` (masks genuine finalize failures). Worked example:
`run_952.phase0_verify` + `issue952_stats._reconstruct_lmsys_prompts`
(#952 commit 3a95b2e7a8) for the fully-consumed case;
`scripts/issue1947_datagen.py` for the `break`-exit case.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [HF datasets streaming shutdown SIGABRT](feedback_hf_datasets_streaming_shutdown_sigabrt.md) — a streaming IterableDataset surviving to interpreter shutdown SIGABRTs rc=134 AFTER all work completed; del+gc.collect() at the call site (never os._exit); #952 r2.
