---
name: pgrep self-match poisons pidfile over SSH
description: When resolving a relaunched workload's PID over SSH, pgrep -f on the launch command self-matches the SSH wrapper; use a pattern absent from your own command string.
type: feedback
---

When confirming a pod relaunch over SSH, resolve the child PID with a `pgrep` pattern that is ABSENT from your own SSH command string (e.g. `venv/bin/python3 <script>.py`) — `pgrep -f 'bash <script>.sh'` self-matches the SSH wrapper process and writes a transient PID into the pidfile, which then yields false `status=dead` polls.

**Why:** task #602 respawn 1 (2026-06-11) — the pidfile briefly carried the SSH wrapper's PID before correction.

**Second variant (#601 relaunch 3, 2026-06-11):** even a SEPARATE-invocation probe with a bracketed pattern (`i601_launch[.]sh`) matched the still-alive LAUNCH wrapper subshell — `bash -c 'cd ... && setsid nohup bash scripts/X.sh ... & echo ok'` parses as `(cd && setsid nohup bash X.sh) & echo ok`, so a forked subshell carrying the full launch string in its cmdline survives as parent-of-driver and pgrep `head -1` picks it (lower PID). Robust fix: `pgrep -fx "bash scripts/X[.]sh"` — exact-full-cmdline match selects only the exec'd driver; the bracket also self-excludes the probe.

**How to apply:** any relaunch-confirmation step that writes/verifies `/workspace/logs/issue-<N>.pid`. Verify the resolved PID's `ps -o sess` equals its own PID (setsid session leader) before posting. Also: a transient single-file HF upload failure at a skip-if-exists dispatcher's upload gate needs only a resume relaunch — zero GPU re-work.
