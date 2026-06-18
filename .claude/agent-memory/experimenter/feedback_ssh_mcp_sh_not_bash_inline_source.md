---
name: SSH MCP runs sh not bash — never inline `source .env` in a launch chain
description: Inline `... && source .env && nohup ...` over SSH MCP silently fails because the remote shell is `sh`; route the launch through a bash launcher script that uses `. ./.env`
type: feedback
---

SSH MCP runs commands under POSIX `sh`, NOT `bash`. An inline launch chain that
contains `source .env` short-circuits at `source: not found`. Worse, when the
chain is part of a pipeline that captures the launched PID via `echo $! >
pid_file`, `$!` then picks up SOME PRIOR backgrounded process (whatever was
last) instead of the dispatch script — the marker reports a pid that LOOKS
alive but is unrelated to the workload, and the dispatcher never actually
started. (Task #545 round 28, 2026-06-13.)

**Why:** `&& source` aborts the chain at the `source` step; `nohup ... &` never
fires; but a backgrounded subprocess from earlier in the pipeline (or an
already-detached `nohup` from a prior failed attempt) gets caught by `$!`.

**How to apply:** Whenever launching a multi-step workload via SSH MCP that
needs `.env` (HF_TOKEN, WANDB_API_KEY, ANTHROPIC_API_KEY, RUNPOD_API_KEY)
exported to the workload process tree, write a small bash launcher script
ON THE POD with `. ./.env` (POSIX-portable, works under both `sh` and `bash`)
followed by the dispatch invocation, then `nohup bash /workspace/launch_<N>.sh
> /workspace/logs/issue-<N>.log 2>&1 < /dev/null &` it. NEVER inline `source
.env` into the SSH command.

Canonical launcher shape:
```bash
cat > /workspace/launch_<N>.sh <<'LAUNCHER'
#!/bin/bash
cd /workspace/explore-persona-space
. ./.env
exec bash scripts/issue<N>_dispatch.sh
LAUNCHER
chmod +x /workspace/launch_<N>.sh
nohup bash /workspace/launch_<N>.sh > /workspace/logs/issue-<N>.log 2>&1 < /dev/null &
echo $! > /workspace/logs/issue-<N>.pid
sleep 2 && cat /workspace/logs/issue-<N>.pid
```

Then `ps -p <pid> -o pid,etime,stat,cmd` to confirm the dispatch script is the
process that pid points at, not an unrelated background job.
