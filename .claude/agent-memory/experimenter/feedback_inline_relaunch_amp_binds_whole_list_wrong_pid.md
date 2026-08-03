---
name: Inline relaunch `cd && setsid nohup ... & echo $!` captures the WRONG pid
description: In `ssh pod 'cd X && setsid nohup bash -c "..." > log & echo $! > pidfile'` the `&` backgrounds the ENTIRE `cd && setsid ...` list as a subshell; `$!` is that un-setsid'd wrapper (dead ssh session's pgroup, HUP-vulnerable), not the workload (#1768 relaunch 3)
type: feedback
---

In the launcher-less relaunch shape `ssh pod 'cd <dir> && setsid nohup bash -c
"<workload>" > log 2>&1 < /dev/null & printf "%s\n" "$!" > pidfile'`, shell
grammar binds `&` to the WHOLE preceding AND-list (`cd && setsid ...`), so the
remote shell forks a plain subshell to run it and `$!` is that SUBSHELL's pid
— NOT the setsid'd workload. The subshell sits in the dying ssh session's
process group (HUP-vulnerable on channel teardown) while the real workload
lives one level down in its own setsid session; a poller keyed on the pidfile
can then read a healthy run as dead. Also: that subshell inherits the ssh
channel's stdout (the `> log` redirect binds only to the setsid command), so
the local ssh client HANGS holding the channel for the whole run.

**Why:** #1768 relaunch #3 (2026-07-29): pidfile got 121744 (wrapper subshell,
dead-session pgroup); the actual HUP-immune chain leader was its child 121746
(`SESS==PID`). The launching Bash call auto-backgrounded because the ssh
client never returned.

**How to apply:** on any inline (launcher-less) relaunch, after launch walk the
tree (`pgrep -P <pidfile-pid>`; the setsid'd child has `SESS == its own PID`)
and atomically repoint the pidfile at the setsid session leader — for a
two-command `a && b` chain that leader (not the python pid, which dies at the
a→b transition) is the correct liveness anchor. Then kill the lingering local
ssh client by exact pid and re-probe survival from a fresh connection.
Cleaner: parenthesize so `&` binds tight — `cd X && { setsid nohup bash -c
"..." > log 2>&1 < /dev/null & printf "%s\n" "$!" > pid.tmp && mv ...; }` —
or use the canonical launcher-script pattern (experimenter.md § During
Execution step 1), which avoids all of this.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Inline relaunch `&` binds the whole `cd && setsid` list — wrong pid in pidfile](feedback_inline_relaunch_amp_binds_whole_list_wrong_pid.md) — `$!` captures the un-setsid'd wrapper subshell (HUP-vulnerable, holds the ssh channel open so the local ssh client hangs); repoint pidfile at the setsid session leader (SESS==PID, spans an `a && b` chain) and kill the lingering local ssh client by exact pid, then re-probe survival (#1768 r3)
