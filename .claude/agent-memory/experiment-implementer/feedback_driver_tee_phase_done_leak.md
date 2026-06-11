---
name: Driver tee leaks child [phase=done] into the poller log
description: Pod drivers must NOT tee child tools that emit their own [phase=done] into the main nohup log — poll_pipeline reads the most recent [phase=...] line and reads a FALSE terminal state mid-run.
type: feedback
---

Pod-side drivers must route child-tool output to per-step log files (`> "$LOG" 2>&1`), never `2>&1 | tee` into the driver's stdout (= the main nohup log `poll_pipeline.py` tails).

**Why:** `poll_pipeline.py::PHASE_RE` declares `status="done"` when the MOST RECENT `[phase=...]` line in the tailed log is `[phase=done]`. Several reused tools emit their own terminal `[phase=done]` (issue_519_dispatch.py's manifest line, issue_521_em_recipe_smoke.py's PASS line, activation_shift.py, mix builders). A teed child `done` creates a window after that child exits — before the driver's next `phase ...` line — where a poll reads a FALSE `done` and the orchestrator can tear down before the sentinel exists. Caught at #552 round-3 implementation smoke (2026-06-11); the prior #552 drivers carry the same latent hazard but happened never to be polled in the window.

**How to apply:** in any new `run_issue*.sh` driver, child invocations get `> "$LOG_DIR/<step>.log" 2>&1 || fail_loud ...`; the main log carries ONLY the driver's own `phase()` lines, ending with the single terminal `[phase=done]` after the sentinel write.
