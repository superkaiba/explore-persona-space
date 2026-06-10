---
name: SSH MCP 30s client cap
description: mcp__ssh__ssh_execute aborts client-side at 30000ms regardless of the timeout param — never embed sleeps; poll with short repeated calls
type: feedback
---

`mcp__ssh__ssh_execute` enforces a hard ~30 s CLIENT-side timeout even when
`timeout: 150000` is passed (error: `Command timeout after 30000ms`, though
the server wraps the command in the requested `timeout 150`). The remote
command keeps running on the pod after the client gives up.

**Why:** Hit at #557 Stage-B attempt-3 launch gate (2026-06-10): a
`sleep 75; grep [gpu-pin] ...` probe died client-side at 30 s; the re-probe
without the sleep succeeded because wall time had already passed.

**How to apply:** Never put `sleep` inside an ssh_execute command for gate
waits. Instead, do other work (or issue cheap probes) and re-grep — each
probe must complete in <25 s. Side effect to remember: an "aborted" probe's
remote side still ran, so re-running non-idempotent commands after a client
timeout can double-execute.
