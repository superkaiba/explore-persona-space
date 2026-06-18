---
name: SSH MCP enforces ~30s client cap despite timeout param
description: mcp__ssh__ssh_execute kills calls at ~30s even when timeout=90000+ is passed; never embed pod-side sleeps >25s in probe commands
type: feedback
---

`mcp__ssh__ssh_execute` accepts `timeout` up to 300000 ms in its schema and
even wraps the remote command in `timeout <s>`, but the CLIENT kills the call
at ~30s anyway (`Command timeout after 30000ms` despite `timeout: 90000`).

**Why:** observed on pod-570 (#570 follow-up run-2 launch, 2026-06-11): a
`sleep 45; ps; tail` probe died at 30s with success=false; the same probe
without the sleep succeeded.

**How to apply:** never put pod-side `sleep` >~20-25s inside an ssh_execute
probe. For post-launch watch windows, issue multiple short probes (each its
own call) instead of one long sleeping call. Long waits belong to the
orchestrator's poll_pipeline loop anyway.
