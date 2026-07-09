---
name: SSH MCP timeout on a nohup launch does NOT mean the child died — pgrep before relaunch
description: bash -lc '... nohup ... &' over SSH MCP times out client-side while the dispatcher backgrounds successfully; blind relaunch creates racing dispatchers sharing logs + vLLM sockets, and killing the orphan's vLLM cascade-kills the legitimate run.
metadata:
  type: feedback
---

**If an SSH MCP call containing `nohup ... &` timed out, ALWAYS `pgrep -af <script>` before relaunching** — the timeout is on the wrapper, not the child, and the child very likely backgrounded successfully. A blind relaunch creates TWO dispatchers racing on the same log/state files and zmq engine sockets; killing the orphan's vLLM engines then cascade-kills the legitimate launch too (`Engine core initialization failed`).

**Why:** #383 launch (2026-05-24) — three attempts: `bash -lc` timeout left orphan PID alive; the orphan-kill cascade killed the fresh dispatcher's vLLM init. #399 round-6 (2026-05-27) — the same timeout-then-relaunch produced THREE racing trees; cleaned with `pkill -9 -f '<script>'` then one clean relaunch.

**How to apply:** never wrap `nohup ... &` in `bash -lc` over SSH MCP. Use the canonical launcher-script pattern (now mandated in `experimenter.md` § During Execution: write `/workspace/launch_<N>.sh` with `#!/bin/bash`, `set -a; source .env; set +a`, PATH export, `exec uv run python ...`, then `setsid nohup bash <launcher> ... < /dev/null &`). `exec` makes the orphan tree's head the dispatcher itself so `ps -p <PID>` tracking works. After any launch, verify `pgrep -af <script>` shows exactly ONE dispatcher tree. (Venue: this launcher shape is RunPod-scoped — `/workspace` + a bootstrap-pushed `.env`. On a GCE SSH relaunch there is NO `.env`: stage it first per [[feedback_gcp_salvage_relaunch]], or use the gotchas.md conditional form `if [ -f ./.env ]; then set -a; . ./.env; set +a; fi`.)
