---
name: infra-drain-three-file-set
description: An infra-drain-pass behavior change in autonomous_session_watch.py mirrors across three workflow-surface files
metadata:
  type: reference
---

A change to the watcher's **infra-drain pass** (`scripts/autonomous_session_watch.py`,
the `infra_drain_pass` + `decide_infra_drain` + `_infra_drain_*` family, ~line
4668+ "execute the PM-adjudicated dispatch queue; #633") mirrors across THREE
workflow-surface files — touch all three or the change is half-landed:

1. `scripts/autonomous_session_watch.py` — the logic itself.
2. `tests/test_autonomous_session_watch.py` — the infra-drain tests live HERE
   (the `test_infra_drain_*` block + pure-decision helpers `_decide_drain`,
   `_write_drain_queue`, `_stub_drain_executor`, fixture `isolated_registry`,
   const `_DRAIN_NOW`). The companion file `tests/test_stalled_detector_and_gc.py`
   does NOT carry infra-drain tests but is the OTHER watcher test file — run both.
3. `.claude/rules/background-automation.md` § "Infra-drain pass" — the prose
   spec of the pass (predicates, env-var overrides, fail directions). It makes
   load-bearing claims ("the PM remains the ONLY ripeness judge", "zero LLM
   judgment", the occupied-status set, the free-slot formula) that a behavior
   change can falsify — keep it in sync. CLAUDE.md § Pods carries only a generic
   one-line watcher summary and usually needs no edit.

The queue file `~/.eps-autonomous/infra-drain-queue.json` schema:
`ripe_oldest_first` (int list), `cap` (default 3), `holds` ({str-id: reason}),
`updated_ts` (`"%Y-%m-%dT%H:%M:%SZ"` UTC), `updated_by`, `comment`. Parsed by
`parse_infra_drain_queue` (string hold keys → int). The PM writes it
(`updated_by: pm-session`); as of #633-follow-on the watcher also writes it on
predicate-hold promotion (`updated_by: autonomous_session_watch:predicate-promote`).
Predicate holds use reason `predicate-<#N>-<short-desc>` (`research-pm.md` step 3).

Test-run recipe (worktree-safe, avoids the fresh-.venv ENOSPC): run the MAIN
checkout's `.venv/bin/python -m pytest tests/test_autonomous_session_watch.py
tests/test_stalled_detector_and_gc.py` with cwd = the worktree root (so
`scripts/` resolves to the worktree copy — confirm via
`asw.__file__`). See [[reference_watcher_two_test_files]] and
[[reference_worktree_venv_disk_full]].
