---
name: Live dispatcher ≠ stale — check run-launched marker before failing
description: Before posting epm:failure on a "stale dispatcher still running" abort condition, cross-check the latest epm:run-launched marker; a pid match means the launch already happened (duplicate dispatch) and the right action is a progress no-op, not failure.
type: feedback
---

When a relaunch brief says "expect NONE; if a live dispatcher exists, post
`epm:failure infra (stale dispatcher)`", do NOT follow it literally before
checking WHY the process is alive. Cross-check, in order:

1. `cat /workspace/logs/issue-<N>.pid` vs the live pid (`ps aux | grep <dispatcher>`).
2. The latest `epm:run-launched` marker in the task's `events.jsonl` — if its
   `pid=` matches the live process and pidfile, a PRIOR experimenter spawn
   already executed the same brief (repair + launch + marker) and this spawn
   is a duplicate stage dispatch.
3. Log freshness + content: an actively-writing log at production probe sizes
   on the expected HEAD is a healthy run, not a leftover.

If all three line up: post `epm:progress` recording the duplicate-dispatch
no-op and EXIT. Posting `epm:failure infra` would trigger a respawn loop
against a healthy run; launching anyway would create racing dispatchers
(known catastrophic anti-pattern — vLLM cascade-kill).

**Why:** Task #545 round-20 K1-recovery (2026-06-11): orchestrator dispatched
the workload-relaunch stage twice (03:06Z and 03:26Z, the second after the
code-review ensemble resolved). The first spawn had already repaired + launched
pid 340474 and posted `epm:run-launched` v13 at 03:07:27Z. The brief's literal
abort instruction would have posted a false infra failure against the
legitimate run.

**How to apply:** Any abort-condition step in a launch brief that keys on
"process exists" gets the marker cross-check first. The marker history is
ground truth for "did a prior spawn already do my job"; the brief's
expectations describe the state at brief-WRITING time, not dispatch time.
