---
name: watcher-seam-signature-conftest-mirror
description: Changing a watcher seam signature (_post_progress_marker etc.) requires mirroring tests/conftest.py's autouse hermeticity-guard wrappers, or the new kwarg TypeErrors inside fail-soft arms
metadata:
  type: reference
---

`tests/conftest.py` wraps the watcher's subprocess seams
(`_post_progress_marker`, `_task_status`, `_launch_guard_apply`) with
autouse fail-loud hermeticity guards (#1247/#1265) whose wrappers
RESTATE the seam's signature. Adding a parameter to the real seam
(e.g. #2295's keyword-only `by`) without updating the conftest wrapper
makes every test call raise `TypeError: unexpected keyword argument` —
and when the caller sits inside a per-entry fail-soft arm, the error is
swallowed into stderr and the test fails DOWNSTREAM (missing markers),
not at the seam. Also: `_task_status`'s guard is UNCONDITIONAL
fail-loud (no stubbed-subprocess allowance, unlike the marker guard) —
tests must stub it; and the #966 pin
`test_post_marker_helpers_carry_distinctive_by` regex-pins the literal
`"--by", "autonomous_session_watch"` argv adjacency — parameterizing
the identity requires a deliberate pin update (follow the parameter
default via `inspect.signature`, which resolves through the guard's
`functools.wraps`).

**How to apply:** any watcher seam-signature change → grep
`tests/conftest.py` for the seam name FIRST and mirror the wrapper in
the same commit; expect the #966 source-inspection pin to need a
deliberate update.
