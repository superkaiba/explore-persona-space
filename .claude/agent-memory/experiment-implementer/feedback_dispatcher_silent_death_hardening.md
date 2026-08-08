---
name: Wave/chunk dispatchers need retry + per-file logs + top-level guard
description: Launcher scripts that download checkpoints then spawn subprocesses MUST wrap hf_hub_download in retry-with-backoff, log one line per completed file, and guard the per-source loop with logger.exception + re-raise — else a transient blip kills the launcher silently.
type: feedback
---

A launcher that loops over sources, downloads K files each via `hf_hub_download`, then spawns eval subprocesses will die SILENTLY (no traceback, no ps trace) on a single transient HF 5xx / network blip / EDQUOT if the download loop is unguarded — the orchestrator can't tell crash from clean exit.

**Why:** task #396 (2026-05-27), Wave-3 `police_officer`: launcher disappeared mid-download with 5 of 14 files landed; root cause never pinned (Hub flake / ChunkedEncodingError class).

**How to apply** — every wave/chunk launcher carries:
1. **Retry-with-backoff** around each `hf_hub_download`: 3 attempts, 30/60/120 s, catching `(HfHubHTTPError, OSError, ConnectionError)`; raise RuntimeError naming source+file after exhaustion.
2. **Per-file completion log line** ("downloaded i/N: fname") so a future silent death pins the in-flight file.
3. **Top-level guard** on the per-source loop: `except KeyboardInterrupt: raise`; `except Exception: logger.exception(...); raise` (non-zero exit the orchestrator can see).
4. `import time` inline in the using function (ruff strips an unreferenced top-level import — [[ruff-strips-unused-imports]]).

Canonical implementation + tests: `scripts/launch_issue396_eval.py` (`download_merged_checkpoint`, `wave_loop`) and `tests/test_issue396_eval_dispatcher_smoke.py`. Pairs with checkpoint-per-phase: that rule saves completed-phase OUTPUT; this keeps the LAUNCHER diagnosable.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Dispatcher silent-death hardening](feedback_dispatcher_silent_death_hardening.md) — hf_hub_download retry-with-backoff + per-file log lines + logger.exception-and-reraise loop guard. #396.

## Merged sibling index rows (#2032 curation, 2026-08-03)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the agent-memory index size cap (task #2032). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [Orphan-PID check must be CVD-aware](feedback_orphan_pid_check_must_be_cvd_aware.md) — on multi-GPU pods filter compute-app PIDs by (#396)
