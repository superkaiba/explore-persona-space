---
name: union-run-under-gate-contention
description: Local gate-matched test unions time out under concurrent Step 9c fleets — chunk them, map -q failures via --collect-only index math, and expect bg-Bash queueing to delay the watched job
metadata:
  type: reference
---

Running the #1288 gate-matched local union (~60 files) from an issue worktree
while sibling Step 9c gate fleets run concurrently makes subprocess-heavy
workflow_lint test files (phase_done_check, upload_or_true, skill_doc_size —
each has a no-flags-bundling test that runs the whole lint in-process, 3-6 min
under load) blow any single 520 s `timeout(1)` bound (2026-08-07, #2165 r2:
union killed at 94%, tail rerun killed again; `upload_or_true` alone took
356 s).

**Recipe that worked:**
- After a timeout-killed `-q` run, do NOT rerun everything: `pytest <same
  files> --collect-only -q` (fast, ~30 s), then map each `F` in the dot
  stream to a test id by index (strip ` [ NN%]` suffixes, concatenate progress
  chars; index i → collected test i). Gives exact failing tests + the
  never-reached remainder from a partial transcript.
- Rerun only the failing tests (with `--tb=short`) + the never-reached files,
  CHUNKED (single heavy file alone; small files batched), each chunk under its
  own `timeout(1)` inside ONE background Bash (`;`-sequenced) so one
  notification covers all chunks.
- Pre-existing-vs-introduced triage for a failure in a content-pin test:
  `git diff origin/main -- <pinned surface> <test file>` (byte-identical ⇒
  pre-existing) + empirically rerun the failing ids on the MAIN checkout
  (read-only, safe). Both legs go in the marker.

**Harness caveat:** background Bash tasks queue SERIALLY on this VM — a
`sleep`-poll bg task can sit queued for minutes and the watched job's own
start can be delayed behind your polls (observed: pytest etimes far younger
than its launch wall-time). Poll sparingly (one long sleep at a time); each
extra poll task delays the job you are waiting on. Foreground polls are fine.

**How to apply:** any implementer round whose local union includes >1
workflow_lint bundling-test file while `pgrep -af step9c` shows live sibling
gates.

**Bare no-flags workflow_lint wall (2026-08-18, #2168 unit 3):** the full
no-flags `workflow_lint.py` run itself now takes ~8-12 min under sibling-gate
contention (two measured runs; a 510 s foreground `timeout(1)` killed the
first attempt rc=124). A mutation spot-check / no-flags baseline needs a
background Bash with `timeout --kill-after=60s 1500s` + an until-loop poll on
an rc sentinel file — never the ~510 s foreground convention. A 45-file
C3-scoped union measured 1290 tests / ~26 min under the same contention.
