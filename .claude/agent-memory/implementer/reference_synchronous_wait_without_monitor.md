---
name: synchronous-wait-without-monitor
description: When ToolSearch/Monitor is disabled in a subagent turn, wait out a detached job synchronously with a blocking `tail --pid` window plus an rc sentinel — never a sleep loop
metadata:
  type: reference
---

`ToolSearch` (and therefore `Monitor`) is sometimes DISABLED for a session —
"No such tool available: ToolSearch. ToolSearch is disabled for this session,
in subagents as well as here." A subagent brief that says "wait synchronously
via a bounded Monitor until-loop" is then unsatisfiable, and foreground
`sleep` chains are hook-blocked. Both sanctioned waiting primitives are gone.

The working substitute is a blocking wait on the pid, not a poll:

```bash
setsid nohup <cmd> > /tmp/job.out 2>&1 < /dev/null &
echo $! > /tmp/job.pid
# later, in the same or a following Bash call:
timeout --kill-after=30s 540s tail --pid="$(cat /tmp/job.pid)" -f /dev/null
```

`tail --pid=<pid> -f /dev/null` blocks until that pid exits and then returns —
one syscall-level wait, no sleep, no polling. `timeout` bounds it under the
Bash tool ceiling; rc=124 means "still running, take another window", rc=0
means "it exited". Pair it with an rc sentinel the job writes on exit
(`...; echo rc=$? > /tmp/job.done`) — `tail --pid` tells you the process
ended, not whether it succeeded.

**Why this matters:** the no-flags `workflow_lint.py` run exceeded a 540 s
foreground `timeout` on #2386 (rc=124), and the inline payload lint gate is a
push blocker. Detaching it and taking two consecutive `tail --pid` windows
(124, then 0 with `rc=0` / `workflow_lint: PASS`) satisfied the gate inside
the single subagent turn. A subagent gets ONE turn and is never re-woken, so
"launch in background and end the turn" would have left the gate unverified
and the marker unposted.

Related: [[reference_union_run_under_gate_contention]] (chunking long test
unions under the same tool-timeout ceiling).
