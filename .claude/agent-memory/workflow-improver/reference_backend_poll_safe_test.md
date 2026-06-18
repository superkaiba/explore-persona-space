---
name: backend_poll safe end-to-end test
description: How to exercise scripts/backend_poll.py's RunPod path without mutating live task state — synthetic sidecar, fake issue, nonexistent pod
type: reference
---

`scripts/backend_poll.py --issue <live-N>` is NOT a safe self-test: the RunPod
path calls `poll_pipeline.poll_once`, whose `_drain_sentinels` POSTS MARKERS to
the task's events.jsonl and can consume sentinels the live `/issue` session
expects (rule-3 task-state mutation + a race).

**Safe recipe:** the handle sidecar resolves to the MAIN checkout's
`.claude/cache/issue-<N>-handle.json` (cwd-independent since #612, per
`issue_dispatch.default_handle_sidecar_path`), and `--handle-file` overrides it.
Write a synthetic sidecar with `"backend": "runpod"`, a FAKE issue (e.g. 999571)
and a nonexistent `pod_name`; run from any cwd with the MAIN checkout's
`.venv/bin/python` against the worktree's script copy. `_marker_pid` is
exception-safe on a missing issue, the SSH probe fails harmlessly → emits a
`status: "dead"` JSON, exit 0. The bar is "no ModuleNotFoundError", and the only
write is `<main-checkout>/.claude/cache/poll-pipeline-<fakeN>.json` (delete
after; `poll_pipeline.DEFAULT_STATE_DIR` is ALSO main-checkout-anchored since
the 2026-06-12 phase-cache fix — worktree copies no longer write locally).
Missing-sidecar runs exit 0 with a `missing_handle_sidecar` JSON — they do NOT
exercise the `scripts.poll_pipeline` import at `backends/runpod.py:250`.
