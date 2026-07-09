---
name: RunPod-lane launch — `.env` not sourced by `nohup bash <driver>`
description: On the RunPod auto_fallback lane (no `--metadata` secret injection), launch via a wrapper that sources `.env` before exec'ing the driver — never bare `nohup bash <driver>`.
type: feedback
---

On the **RunPod auto_fallback** lane (`reason: auto_fallback_runpod`), a workload driver that reads API keys (`ANTHROPIC_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`) straight from the shell environment will die in seconds if launched via the plain `nohup bash <script>` shape. SSH-MCP's non-login `sh` does NOT source `.env`, and the RunPod lane (unlike the GCP lane) has no `--metadata` secret injection. A launch command copied verbatim from a GCP-lane brief is the trap — the GCP startup script `curl`s every key from instance metadata and exports them; the RunPod path expects the launcher to source the repo's `.env` itself.

**Why:** #657 v6 round-3 dispatched to RunPod as the auto-chain terminal fallback (GCP cap 5/5, Nibi SLURM rsync-source-branch mismatch). The experimenter's brief used a plain `nohup bash scripts/issue657_extract.sh --skip-preflight > log 2>&1 &` shape — the driver's `ANTHROPIC_API_KEY` preflight tripped in ~3s with `FATAL: ANTHROPIC_API_KEY missing`. The `.env` carried all three keys (ANTHROPIC len=108, HF_TOKEN len=37, WANDB len=86) — they just weren't sourced. Relaunch via a launcher script (`set -a; . ./.env; set +a` before exec'ing the driver) worked.

**How to apply:** on a RunPod launch (whether the lane was the explicit `backend: runpod` override or the `auto_fallback_runpod` terminal rung), ALWAYS use a wrapper shape:

```bash
ssh pod-N 'cat > /workspace/explore-persona-space/launch.sh <<'\''EOF'\''
#!/bin/bash
set -e
cd /workspace/explore-persona-space
set -a; [ -f .env ] && . ./.env; set +a
echo $$ > /workspace/logs/issue-N.pid
exec "$@"
EOF
chmod +x /workspace/explore-persona-space/launch.sh'
ssh pod-N 'cd /workspace/explore-persona-space && mkdir -p /workspace/logs && setsid nohup ./launch.sh bash scripts/<driver>.sh <args> > /workspace/logs/issue-N.log 2>&1 < /dev/null & sleep 2 && cat /workspace/logs/issue-N.pid'
```

The pattern is: a thin launcher.sh that sources `.env`, writes its own pidfile via `echo $$ >` before `exec` (the launcher-internal carve-out of pod-side-reporting.md § Pid-file launch contract — after `exec`, `$$` IS the driver), and exec's the driver. The driver inherits the sourced env. The detachment trio (`setsid` + `nohup` + `< /dev/null`) + redirection stays the same as the GCP-lane shape; never bare `nohup ... &`.

Do NOT try to inline `set -a; . .env; set +a; nohup bash ...` in the SSH command itself — SSH-MCP's `sh -c` quoting eats the dot-source on certain shells, and the env doesn't propagate to the nohup'd child reliably across SSH boundaries. A real launcher.sh on the pod is the robust shape.

The same lane-asymmetry applies to any future workload that depends on env-var-injected secrets: the GCP `--metadata` mechanism is invisible to the experimenter; the RunPod `.env`-source step is mandatory.
