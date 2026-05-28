---
name: Wave/chunk dispatchers need retry + per-file logs + top-level guard
description: Long-running launcher scripts that download checkpoints then spawn subprocesses MUST wrap hf_hub_download in retry-with-backoff, log one line per file completed, and put a top-level try/except in the per-source loop. Otherwise a transient network blip kills the whole launcher silently.
type: feedback
---

# Wave/chunk dispatchers need retry + per-file logs + top-level guard

When a launcher script:
1. Loops over N sources,
2. For each one, downloads K files via `hf_hub_download`,
3. Then spawns a subprocess to evaluate,

a single transient HF Hub 5xx, a network blip, an `OSError(EDQUOT)`,
or any other surprise inside the download loop will propagate up,
hit the top of `wave_loop` / `main()` without a handler, and exit the
launcher silently — no traceback in the log, no `ps` trace, no
dmesg signal. The orchestrator sees "no process running" and cannot
tell crash from clean exit.

**Why:** Task #396 2026-05-27, Wave-3 `police_officer`: the launcher
died mid-download with no traceback. Only 5 of 14 files landed
locally before the launcher process disappeared. Root cause never
fully pinned down; could have been HF Hub flake, half-written file
race, or `requests.exceptions.ChunkedEncodingError`.

**How to apply:** every wave/chunk-style launcher that downloads
checkpoints must carry:

1. **Retry-with-backoff** around each `hf_hub_download`:
   ```python
   from huggingface_hub.errors import HfHubHTTPError
   for fname in files:
       last_exc = None
       for attempt in range(3):
           try:
               hf_hub_download(repo_id=..., filename=fname, local_dir=...)
               last_exc = None
               break
           except (HfHubHTTPError, OSError, ConnectionError) as e:
               last_exc = e
               wait = 30 * (2 ** attempt)  # 30s, 60s, 120s
               logger.warning("[%s] download(%s) attempt %d/3 failed (%s) — retrying in %ds",
                              source, fname, attempt + 1, e, wait)
               time.sleep(wait)
       if last_exc is not None:
           raise RuntimeError(f"[{source}] exhausted 3 retries for {fname!r}: {last_exc}") from last_exc
       logger.info("[%s] downloaded %d/%d: %s", source, idx + 1, len(files), fname)
   ```

2. **Per-file completion log line** after each successful download.
   Verbose but a future silent death will at least pin down which
   file was in flight.

3. **Top-level guard** in the per-source / per-chunk loop:
   ```python
   try:
       for chunk in chunks: ...
   except KeyboardInterrupt:
       raise  # Ctrl-C through
   except Exception:
       logger.exception("dispatcher: unhandled exception; aborting and re-raising")
       raise  # non-zero exit so orchestrator sees a real failure
   ```

4. **`import time` inline** inside the function that uses it — ruff
   strips a top-level `import time` if no module-scope reference
   exists. Either inline-import (`import time as _time`) or hold a
   module-scope reference (`_ = time`).

Canonical implementation: `scripts/launch_issue396_eval.py`
`download_merged_checkpoint` + `wave_loop` (task #396, BF11 fix
2026-05-27). Tests in
`tests/test_issue396_eval_dispatcher_smoke.py` exercise:
* `test_download_merged_checkpoint_retries_then_succeeds` — 2
  transients then success
* `test_download_merged_checkpoint_exhausts_retries` — 3 failures →
  RuntimeError
* `test_wave_loop_logs_and_reraises_unhandled_exception` —
  surprise exception → `logger.exception` (record has `exc_info`)
  then re-raise

Pairs with the CLAUDE.md "Checkpoint per phase" rule: that rule
saves the OUTPUT of completed phases; this rule keeps the
LAUNCHER itself diagnosable when a downstream phase explodes.
