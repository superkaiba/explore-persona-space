---
name: bash-watchdog-subshell-signals
description: Bash watchdogs that capture cycle output via $(...) put the main loop in a subshell, breaking parent signal traps; use globals instead.
metadata:
  type: feedback
---

When a bash watchdog uses command substitution to "return" multiple values
from its inner loop — e.g. `result=$(run_one_dispatcher_cycle "$respawn")`
where the function prints `rc:before:after` and the caller parses it — the
function runs in a **forked subshell**. The parent shell is blocked in the
substitution and CANNOT run its signal traps (TERM/HUP/INT) until the
subshell exits. If the subshell contains a long-running `while kill -0; do
sleep N; done` loop waiting on a child, the watchdog becomes unkillable by
SIGTERM during that loop.

**Why:** Bash traps installed in the parent are not inherited by `$(...)`
subshells. The parent's signal handler is queued but cannot fire until
control returns to the parent — which only happens when the substitution
completes.

**How to apply:** When a long-running bash function needs to return more
than one value to the calling loop, prefer **globals** (`CYCLE_RC`,
`CYCLE_BEFORE`, `CYCLE_AFTER`) over command substitution. Run the function
in the main shell, not a subshell. Signal traps work normally.

Bonus: this also lets `log_w` write freely to stdout (now nobody is
capturing it). Without the subshell pattern you can use `tee -a "$LOG"`
to stdout for live `tail -f` from the launcher.

Seen in task #365 round-7 watchdog redesign — round-6 failure was that
the dispatcher died at ~01:32Z and the watchdog never respawned because
(a) command-substitution subshell blocked signal handling and (b)
log_w's tee-to-stdout polluted the captured return value with log noise.
