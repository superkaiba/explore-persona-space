---
name: watcher-two-test-files
description: autonomous_session_watch.py is pinned by TWO test files — test_autonomous_session_watch.py AND test_stalled_detector_and_gc.py; run both after any edit
metadata:
  type: reference
---

`scripts/autonomous_session_watch.py` is pinned by TWO test files:
`tests/test_autonomous_session_watch.py` (the big one, ~300 tests) AND
`tests/test_stalled_detector_and_gc.py` (~44 older stalled-detector/GC
tests that monkeypatch the same module). Run BOTH after any watcher edit.

Incident (2026-06-10): the caller-label commit `cb81fbce6` added a
`caller=` kwarg to `_running_managed_issue_pods` and only ran the newer
file — 7 tests in `test_stalled_detector_and_gc.py` (zero-arg
`lambda: []` monkeypatches) were left failing on main until the None-vs-[]
snapshot fix repaired them. Monkeypatches of module helpers should use
`lambda *_a, **_k: ...` so signature growth doesn't break them.

Related: [[preexisting-lint-test-failures]] (stash-compare to prove your
diff clean), [[watcher-dryrun-from-worktree]].
