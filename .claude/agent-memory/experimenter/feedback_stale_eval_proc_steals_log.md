---
name: Stale procs from prior rounds steal the log, hold GPU mem, and pull stale checkpoints
description: On relaunch attempts ≥3, orphans from earlier rounds hijack /workspace/logs/issue-<N>.log (their failures read as YOUR dispatcher's), pin GPU memory as PID [Not Found], and snapshot_download the broken prior checkpoint into smoke gates.
metadata:
  type: feedback
---

On re-launch attempts ≥3, orphan processes from killed prior rounds (vLLM EngineCore commonly survives parent SIGKILL) cause four compounding traps: (1) they keep writing to the conventional `/workspace/logs/issue-<N>.log`, so the orphan's smoke-gate FAIL reads as your fresh dispatcher's output while your dispatcher actually died at import; (2) they hold GPU memory with nvidia-smi PID `[Not Found]`; (3) eval orphans `snapshot_download` the BROKEN previous checkpoint and fail smoke against it; (4) all of this masks a silent fresh-launch death (uv not on PATH, env missing).

**Why:** #399 round-9 v8 (2026-05-27) — fresh launch silently exited (uv off PATH) while a stale round-7 eval proc wrote a smoke FAIL to the shared log; ~30 min lost before noticing the stale PID's start time predated the nohup.

**How to apply (pre-launch on attempts ≥3):**
1. `pgrep -af "<dispatcher>|eval_<N>|train\.py"` and `pgrep -af EngineCore` — kill survivors (the EngineCore probe is now experimenter.md Before-Running step 9).
2. `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` — `[Not Found]` PIDs holding memory get `kill -9`; if memory stays pinned, the leak is kernel-level → `pod.py resume`.
3. `rm -rf /workspace/tmp_models/<condition>_*` to force checkpoint re-pull where training will overwrite Hub state.
4. Truncate the log (`: > /workspace/logs/issue-<N>.log`) so the fresh dispatcher's first line is unambiguous, THEN launch via the canonical launcher script (bash, `set -euo pipefail`, PATH export, env sourcing).
