---
name: stale-eval-proc-steals-log
description: Pre-launch must check pgrep/nvidia-smi for orphan procs from prior re-launch attempts BEFORE nohup-ing; a stale eval process can hijack the log file and FAIL a smoke gate against the wrong (cached) checkpoint, masking that the new dispatcher never started.
metadata:
  type: feedback
---

When a task has burned through multiple re-launches (round 7+, attempt 8+ in #399's case), the SSH MCP env on the pod can have ORPHAN python/uv subprocesses from prior attempts that were killed at the wrapper level but kept the child alive (VLLM EngineCore commonly survives parent SIGKILL). These orphans:

1. **Steal the conventional log path** — if both the orphan and your new nohup target `/workspace/logs/issue-<N>.log`, the orphan keeps writing to it and your tail looks like the orphan's output is your dispatcher's output.
2. **Hold GPU memory** — `nvidia-smi --query-compute-apps` shows the PID is `[Not Found]` (process group dissociated from systemd), but the memory is still pinned. Your fresh launch then hits "CUDA out of memory" mysteriously.
3. **Pull stale HF Hub checkpoints** — eval scripts in particular cheerfully `snapshot_download` the latest checkpoint at the expected name, which is the BROKEN one from the previous failed round. The smoke gate FAILs against a checkpoint the new dispatcher would have overwritten 10 min later.
4. **Mask your dispatcher dying silently** — your fresh nohup may have died at import time (uv not on PATH, env missing, etc.) but the log shows the orphan's smoke-gate FAIL traceback, and you misread it as your dispatcher's failure.

**Pre-launch checklist for re-launch attempts ≥3:**
1. `pgrep -af "<dispatcher_script_name>|eval_<N>|train\.py"` — kill any survivors.
2. `pgrep -af VLLM` — kill orphan engine cores.
3. `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` — if any PID is `[Not Found]` but holding memory, `kill -9 <pid>`; if still pinned, the pod needs a `pod.py resume` (memory is leaked at kernel level).
4. `rm -rf /workspace/tmp_models/<condition>_*` — eval scripts re-download checkpoints, this is the only way to force a re-pull AFTER training has overwritten the HF Hub version. (For your case where the new training WILL overwrite, you can let it.)
5. Truncate the log: `: > /workspace/logs/issue-<N>.log` so the new dispatcher's first line is recognizable.
6. THEN launch.

**Wrapper pattern (also avoids feedback_ssh_bash_lc_backgrounding trap):**
```bash
cat > /workspace/launch_issue_<N>.sh << 'EOF'
#!/bin/bash
set -euo pipefail
export PATH="/root/.local/bin:$PATH"   # uv not on default PATH in non-login SSH shells
cd /workspace/explore-persona-space
set -a
source .env
set +a
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
exec uv run python scripts/<dispatcher>.py <args>
EOF
chmod +x /workspace/launch_issue_<N>.sh
nohup bash /workspace/launch_issue_<N>.sh > /workspace/logs/issue-<N>.log 2>&1 < /dev/null &
```

Note: `bash`, not `sh` — sh has no `pipefail` and no `disown`. Drop `disown`; nohup + `&` is enough.

Burned at #399 round-9 v8 (2026-05-27): first launch silently exited because `uv` wasn't on PATH; a stale eval_issue399 process from a prior round-7 re-launch attempt was already writing the smoke-gate FAIL to /workspace/logs/issue-399.log, which looked exactly like my dispatcher's failure but was a different process tree. ~30 min lost to misdiagnosis before noticing the stale PID startup time predated my nohup.

Related: [[load-env-in-nohup]], [[wrapper-pipefail]], [[ssh-bash-lc-backgrounding-trap]].
