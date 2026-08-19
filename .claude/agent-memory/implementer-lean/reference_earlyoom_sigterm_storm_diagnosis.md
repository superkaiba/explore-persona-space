---
name: earlyoom-sigterm-storm-diagnosis
description: rc=143 + EMPTY output on a fresh local python/lint run = suspect an earlyoom SIGTERM storm; journalctl -u earlyoom confirms; choom -600 recipe unblocks
metadata:
  type: reference
---

On the shared VM, a local test/lint invocation that exits **rc=143 with ZERO
output within seconds** — repeatedly, across foreground, bg-Bash, and setsid
detach — is usually NOT a timeout (that is rc=124) and NOT a harness kill:
check `journalctl -u earlyoom --since "10 minutes ago" --no-pager | tail`.
When available memory sits at the 10% SIGTERM floor (concurrent sibling
Step 9c gates each hold multi-GB pytest fleets), earlyoom TERM-kills fresh
python processes (badness ~966) even at <100 MiB RSS — small size does not
protect (#1217 r2, 2026-08-08: three workflow_lint attempts killed, one
within ~5 s of start).

Fix that worked: the code-style.md detach recipe INCLUDING the choom sweep —
setsid-detach with a done-sentinel, then
`bash -o pipefail -c 'pgrep -s "$1" | xargs -rn1 sudo -n choom -n -600 -p' _ "$PID"`,
re-swept once after the real python child appears (the uv-spawned child can
start at adj 0 — inheritance is not guaranteed through uv's exec). The
protected run completed normally. Apply the sweep to ANY multi-minute local
run during a storm, not only >=16 GiB phases (the rule's floor is about when
protection is OWED; during an active storm it is about whether the run can
finish at all).

Also bound waits without Monitor: a foreground `until [ -f <sentinel> ];
do sleep 10; done` Bash call IS accepted by the sleep guard (a leading bare
`sleep N; cmd` is blocked); bg-Bash bounded until-loops work too but their
completion notifications may lag the sentinel by minutes.
