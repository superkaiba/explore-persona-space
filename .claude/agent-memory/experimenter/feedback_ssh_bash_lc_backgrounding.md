---
name: ssh-bash-lc-backgrounding-trap
description: Wrapping a `nohup ... &` launch inside `bash -lc '...'` over SSH MCP timeouts on the SSH side but successfully backgrounds the dispatcher on the pod. Subsequent re-launches create concurrent dispatchers that race on the same log/state files, and killing the orphan's vLLM cascade-kills the legitimate launch.
metadata:
  type: feedback
---

When the pod's default shell is `sh` (Dash) but you need `bash` features (e.g. `source .env` for the `feedback_load_env_in_nohup` rule), the tempting fix is to wrap the whole command in `bash -lc '...nohup ... &'`. Two failure modes follow:

1. **SSH MCP wrapper hangs on `bash -lc` + `&`.** `mcp__ssh__ssh_execute` thinks the foreground hasn't returned and hits its 30-120s timeout, but the dispatcher was actually backgrounded successfully on the pod. The SSH client returns an error, you re-launch thinking nothing started, and now you have TWO dispatchers writing to the same log file and racing for the same vLLM engine sockets (zmq IPC collisions during EngineCore startup).

2. **Aggressive cleanup cascades.** If you then `kill -TERM <orphan_pid>` and also kill its vLLM engines (because they're holding GPU memory), the legitimate second dispatcher dies too because they share the engine init. Both crash with `RuntimeError: Engine core initialization failed. Failed core proc(s): {}`.

**The fix:** Write a launcher script to the pod and `nohup` it directly. The shell that interprets `nohup ... &` MUST be `sh`-level — never `bash -lc`. The launcher script itself can be `#!/bin/bash` with `set -a; source .env; set +a` inside.

```bash
# WRITE LAUNCHER ONCE
cat > /tmp/launch_<N>.sh <<'EOF'
#!/bin/bash
set -a; source /workspace/explore-persona-space/.env; set +a
export PATH=/root/.local/bin:$PATH
exec uv run python scripts/dispatch_<...>.py --issue <N> ...
EOF
chmod +x /tmp/launch_<N>.sh

# LAUNCH WITH PLAIN nohup ... & (works under sh)
nohup /tmp/launch_<N>.sh > /workspace/logs/issue-<N>-dispatcher.log 2>&1 & echo "DISPATCHER_PID=$!"
```

**Why:** `exec uv run` inside the launcher means PID 1 of the orphan tree is the dispatcher itself, so process tracking via `ps -p <PID>` works cleanly.

**Burned at #383 launch (2026-05-24):** First attempt used `bash -lc 'export PATH=... && nohup ... &'`, MCP timed out at 30s but the pod-side process was alive (PID 2447). Second attempt without bash -lc lost env vars (`sh: source: not found`). Third attempt re-wrapped with bash -lc, MCP timed out again, but this time the dispatcher (PID 3224) WAS detected because pgrep ran in the next call. Killing the orphan (2447) and its vLLM engines (2971, 3465) cascade-killed PID 3224's vLLM init too. Clean state recovered by killing everything, archiving log, and using the launcher-script pattern (PID 3962, ran cleanly).

**Burned again at #399 round-6 re-launch (2026-05-27):** Even after switching to the launcher-script pattern, the FIRST attempt still used `bash -c '... && nohup uv run ... & echo "shellpid=$!"'` for env-var loading. SSH MCP timed out at 30s. Assumed nothing started → wrote launcher script and `sh -c 'nohup /workspace/launch_issue_399.sh ... &'`. `pgrep` revealed THREE racing trees (the timed-out bash -lc's orphan 6036/6039, and two from the new launcher 6093/6097, 6102/6107 — the new launcher itself somehow spawned twice). `pkill -9 -f 'eval_issue399|launch_issue_399'` followed by a clean single `sh -c 'nohup ... &'` resolved it (PID 6124 alone).

**How to apply:** ANY nohup-launch over SSH MCP that needs env-var loading should use the launcher-script pattern, not inline `bash -lc`. Test once by checking `pgrep -af <script_name>` shows ONE dispatcher tree, not two. **If you've ever timed out an SSH MCP call that included `nohup ... &`, ALWAYS `pgrep` before relaunching — the timeout is on the wrapper, not the child, and the child very likely backgrounded successfully.**

See also: [[load-env-in-nohup]], [[wrapper-pipefail]].
