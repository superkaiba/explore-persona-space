---
name: pod-provision-uv-missing
description: pod.py provision can return a 'ready' pod with the repo cloned and .env present but uv NOT installed and no .venv directory. Bootstrap step silently dropped. Experimenter must verify uv exists and .venv exists BEFORE running pytest / launching.
metadata:
  type: feedback
---

`pod.py provision --issue <N>` may return a pod that has:
- Repo cloned at `/workspace/explore-persona-space/` (PASS)
- `.env` populated (PASS)
- HF cache redirect (PASS)
- `uv` binary at `/root/.local/bin/uv` (FAIL — missing)
- `.venv/` (FAIL — missing)

Without `uv`, every `uv run ...` invocation returns `bash: line 1: uv: command not found`. Bootstrap log not produced (no `/workspace/bootstrap.log`, no `/tmp/bootstrap*.log` artifact) so the failure mode is silent.

**Why:** Burned at task #390 launch on 2026-05-26. `pod.py provision` reported success and the pod was usable for SSH, but `bootstrap_pod.sh` either silently failed to install uv or that step was skipped entirely on that provision. No exit-code propagation back to the orchestrator.

**How to apply:** Add to the experimenter pre-launch checklist BEFORE pytest smoke:
1. `ssh <pod> 'which uv && uv --version'` — non-zero exit = bootstrap incomplete.
2. `ssh <pod> 'ls /workspace/explore-persona-space/.venv/bin/python'` — missing = bootstrap incomplete.

If either fails:
- Inline-recover: `curl -LsSf https://astral.sh/uv/install.sh | sh` then `nohup uv sync &` — takes 5-15 min for the multi-GB CUDA/torch wheels (nvidia-nccl 307M, nvidia-cudnn 674M, nvidia-cublas 566M, xformers 111M, plus rest).
- Subagent CANNOT wait inline for uv sync — exceeds 60s turn budget. Post `epm:failure v1` with `failure_class: infra`, `reason: pod_bootstrap_incomplete_uv_missing`, include the recovery PID, and exit. Orchestrator polls + re-dispatches.

**PATH gap:** Even after install, SSH non-login shells don't load `~/.bashrc`, so subsequent `ssh_execute` calls need `export PATH="/root/.local/bin:$PATH"` or full path `/root/.local/bin/uv`. Related: [[feedback_load_env_in_nohup]].

Worth surfacing as a follow-up: `bootstrap_pod.sh` should fail loudly (non-zero exit, error to provision log) when uv install fails. Currently `pod.py provision` reports success on bootstrap failure.
