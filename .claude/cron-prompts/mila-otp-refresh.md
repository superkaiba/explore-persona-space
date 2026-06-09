# Mila ControlMaster OTP refresh (Claude-session cron prompt)

> **Status: UN-ARMED in slice 7.** This file is the procedural
> specification for the slice-8 live arming. Do NOT call `CronCreate`
> against it yet, do NOT run a live `--login`, and do NOT touch
> Thomas's gmail or Mila auth from any slice-7 code path.

## Purpose

Keep the Mila interactive SSH socket warm so the multi-backend compute
router's first-class Mila lane stays reachable. The Mila login node
enforces email-OTP MFA on every fresh SSH session and keeps the
ControlMaster socket alive for ~12 h
(`ControlPersist 12h` in `~/.ssh/clusters.config`). After the window
lapses, the next `ssh mila <anything>` would prompt for an OTP — which
a router running headless cannot answer. The result is a stale gate:
`backends.slurm.mila_socket_alive()` returns False indefinitely and the
auto chain silently skips Mila until a human intervenes.

The refresh loop runs inside a Claude session (NOT a bare shell) because
the OTP arrives by email and the only credentialed path to that mailbox
is the `google-workspace` MCP — which only Claude can call.

## What this prompt does

1. **Probe.** Run
   `uv run python scripts/mila_socket_refresh.py probe`. Parse the
   JSON. If `alive: true`, post nothing and exit — the socket is fine.
2. **Fetch latest Mila OTP.** Only if `alive: false`. Search the user's
   Gmail via `mcp__google-workspace__search_emails` for the most recent
   Mila login-OTP message (subject typically contains
   "login.server.mila.quebec" or "Mila SSO"). Open the latest hit, pull
   the 6-digit code out of the body. **Do not echo the code in any
   marker, dashboard tile, chat message, or log line that escapes the
   running Claude session.**
3. **Drop the OTP into the askpass-readable file.** Write the bare
   6-digit code (no trailing newline, no quotes, mode 0o600) to the
   path the askpass helper reads. The slice-8 wiring will pin a stable
   path (e.g. `~/.eps-routing/mila-otp.current`); for slice 7 this is
   undefined and the cron is un-armed.
4. **Initiate login.** Call
   `uv run python scripts/mila_socket_refresh.py login --askpass <path>`
   (or have `EPS_MILA_ASKPASS` already exported). The helper:
   - sets `SSH_ASKPASS`, `SSH_ASKPASS_REQUIRE=force`, and `DISPLAY` so
     SSH consumes the OTP from the askpass helper instead of any tty;
   - runs `ssh mila true` to perform the auth handshake;
   - re-probes the socket and prints
     `{ok: bool, ssh_exit: int, alive_after_login: bool, ssh_alias: str}`.
5. **Shred the OTP file.** Regardless of login outcome, overwrite the
   askpass file with empty content (and consider `shred -u` if the
   filesystem supports it). The OTP is single-use; leaving it on disk
   is unnecessary risk.
6. **Surface the result.**
   - On `ok: true` — no chat post is needed; the next
     `mila_socket_alive()` probe will pick up the warmed socket.
     Optionally append a one-line `state/mila-socket-refresh-<date>`
     marker so the operator can see the cron is exercising itself.
   - On `ok: false` — post a single chat note flagging the failure
     (`ssh_exit`, brief stderr summary, last 3 cron-prompt invocations
     OK count). Do NOT loop the refresh on failure; one shot per cron
     tick.

## What this prompt does NOT do

- It does **not** force a refresh when the socket is alive — the probe
  is the gate. A probe-alive cycle is a no-op (zero ssh, zero gmail).
- It does **not** retain the OTP anywhere off the askpass file. Markers
  log the result, not the secret.
- It does **not** post `epm:*` markers to any task. The refresh is a
  workflow-surface utility, not a per-experiment event. The orchestrator
  reads the warm-socket state via `mila_socket_alive()` only.
- It does **not** auto-arm itself. Slice 8 owns the `CronCreate` call
  with the cadence (proposed: every 6 h, so the next refresh fires
  comfortably before the 12 h ControlPersist window expires).

## Failure modes the slice-8 arming will need to handle

- **Gmail MCP returns no recent OTP** — Mila has not been logged into
  recently, so no email is sitting in the inbox. Trigger a fresh
  `ssh mila` once to provoke the OTP-issuing email, wait ~30s, retry
  the search. Capping the wait avoids an unbounded poll.
- **OTP found but rejected by SSH** — race condition where the email
  arrived but the user already consumed the same OTP from a parallel
  session. Re-trigger the OTP cycle once; surface a failure note if
  the second attempt fails too.
- **`EPS_MILA_ASKPASS` unset** — configuration error; the helper
  refuses to proceed (slice-7 behaviour: `RuntimeError`). Slice 8
  should set this in the cron env, NOT default it (an empty value
  silently falls back to the terminal prompt and hangs the cron
  forever — the helper guards against that).
- **ssh times out at the network level** — Mila's login node is down
  / a corporate firewall change is in flight. Post the failure note
  and let the next tick retry; do NOT escalate (the router gracefully
  skips Mila in the meantime).

## Manual quick-test (slice 8 only)

```bash
# Live probe (safe — no side effects).
uv run python scripts/mila_socket_refresh.py probe

# Dry-run the wiring locally with a fake askpass that prints a known
# string; expects ssh to FAIL (the string is not a real OTP) but
# verifies env / askpass plumbing.
cat > /tmp/fake-askpass.sh <<'EOF'
#!/bin/sh
echo 000000
EOF
chmod 700 /tmp/fake-askpass.sh
EPS_MILA_ASKPASS=/tmp/fake-askpass.sh \
  uv run python scripts/mila_socket_refresh.py login
```

## References

- Helper source: `scripts/mila_socket_refresh.py`
- Probe surface: `backends.slurm.mila_socket_alive`
- SSH alias config: `~/.ssh/clusters.config` (`Host mila` stanza)
- Slice-7 implementation summary: this commit
- Slice-8 arming brief: TBD (next planner pass)
