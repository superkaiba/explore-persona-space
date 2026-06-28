---
name: vLLM zombie GPU — pkill by script name does NOT reap the EngineCore
description: a crashed vLLM parent leaves an orphaned VLLM::EngineCore worker reparented to init; pkill -f <dispatcher-script> misses it because its cmdline is the bare "VLLM::EngineCore" string. Kill the EngineCore by exact PID after the tree kill, then the zombie CUDA allocation releases.
type: feedback
---

**The trap.** When a vLLM-using dispatcher hangs and you kill the wrapper
PID tree to recover, the `VLLM::EngineCore` worker subprocess survives
because:
- It reparents to init (PID 1) when its python parent dies.
- Its `cmdline` is the bare literal string `VLLM::EngineCore` — no
  python interpreter path, no script name, no args.
- `pkill -f issue<N>_dispatch.py` does NOT match (no `issue<N>_dispatch.py`
  in the cmdline of the EngineCore).
- `nvidia-smi --query-compute-apps` still reports the EngineCore PID
  holding the model's GPU memory (~66 GB for Qwen-2.5-7B). Allocation
  looks like a "zombie" against the now-dead dispatcher, but is in fact
  a perfectly-alive orphaned vLLM worker.

**Why this matters.** The orphan blocks the GPU for any relaunch that
tries to load the same model. The relaunch's vLLM init will fail with
CUDA OOM (or hang on its own allocator probe) until the orphan dies.

**Recovery recipe (canonical after-kill probe — incident #664 r8
respawn 1/3, 2026-06-27).** After the wrapper tree kill, ALWAYS do:

```bash
# 1. Kill the wrapper PID tree (parent + children)
kill -TERM <WRAPPER_PID> 2>/dev/null || true
sleep 5
pkill -KILL -f "issue<N>_dispatch.py" 2>/dev/null || true
pkill -KILL -f "issue<N>_launch_parallel.sh" 2>/dev/null || true
sleep 3

# 2. Probe for orphaned VLLM::EngineCore worker(s)
ORPHAN_PIDS=$(pgrep -af '^VLLM::EngineCore' | awk '{print $1}')
if [ -n "$ORPHAN_PIDS" ]; then
  echo "killing orphaned VLLM::EngineCore PIDs: $ORPHAN_PIDS"
  for p in $ORPHAN_PIDS; do kill -KILL "$p"; done
  sleep 3
fi

# 3. Confirm zombie GPU allocation released
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
# Expect: empty output (or only PIDs of intentional processes).
```

The CUDA context releases cleanly once the EngineCore dies — no
`nvidia-smi --gpu-reset` needed (that requires SBIOS support, which
RunPod containers do not have).

**Symptom checklist that points HERE rather than elsewhere:**
- `ps -p <gpu_pid>` reports no /proc entry (the dispatcher main PID),
  BUT
- `nvidia-smi --query-compute-apps` still shows GPU memory held by some
  PID, AND
- `pgrep -af '^VLLM::EngineCore'` returns a non-empty list — the
  EngineCore is the actual live holder.

**Don't bother with `--gpu-reset`** unless the orphan PID can't be
killed (e.g., D state) — it almost always fails in containers.

Closed regressions: task #664 r8 respawn 1/3 (2026-06-27) — vLLM hung
in `_elicit_secure_code` mid-generation; the brief's `pkill -f
issue664_dispatch.py` missed the orphaned `VLLM::EngineCore`; recovery
needed the exact-PID kill before GPU released.
