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
pkill -KILL -f "issue<N>_dispatch[.]py" 2>/dev/null || true
pkill -KILL -f "issue<N>_launch_parallel[.]sh" 2>/dev/null || true
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

**Sub-case: a "zombie-GPU stall" respawn brief can be STALE — verify before reaping.**
A respawn brief that names a dead GPU PID + "reap and relaunch" is a HYPOTHESIS, not
ground truth. SSH-stat the pod FIRST: the original dispatcher may NOT have died —
it can keep running past the detected stall (recycling per-cell vLLM engines),
complete whole phases, and be hung LATER at a different point, while the dead PID
the brief named is just an un-released zombie CUDA context co-resident with a live
engine. Distinguishing reads: (a) `ps -o lstart,etime` on the dispatcher PIDs — a
40-min elapsed time means it never died; (b) the on-disk deliverable count
(per-cell JSONs) tells you how far it actually got; (c) `/proc/<pid>/stat` utime
deltas over 5s — a dispatcher burning ~26% CPU while GPU=0% + EngineCore utime flat
is a live `generate()` DEADLOCK (gotchas.md chunked-generate signature), NOT a dead
process. A blind reap-and-relaunch on the SAME code re-hangs at the identical
generate; route the deadlock `failure_class: code` for the enforce_eager fix.
When py-spy is blocked (`ptrace_scope=1` + read-only `/proc/sys` in RunPod
containers → "Failed to copy Py_Version symbol / Permission denied"), the
`/proc/<pid>/{stat,wchan,task/*/wchan}` + CPU-time-delta fallback localizes the
block (main thread `wait_woken` waiting on the engine = not our code).

Closed regressions: task #664 r8 respawn 1/3 (2026-06-27) — vLLM hung
in `_elicit_secure_code` mid-generation; the brief's `pkill -f
issue664_dispatch.py` missed the orphaned `VLLM::EngineCore`; recovery
needed the exact-PID kill before GPU released. task #734 respawn 1/3
(2026-06-29) — the "zombie-GPU PID 3662716" respawn brief was stale: the
dispatcher was alive 42 min, had completed Phase-0 + 16/16 Phase-1 cells, and
was hung at the FIRST run_phase2 `generate()` (cuda-graph-capture / engine-IPC
deadlock, `enforce_eager=False` hardcoded) with two un-released zombie CUDA
contexts co-resident. Killing the live dispatcher tree by exact PID released BOTH
the live EngineCore AND the zombie contexts (GPU → 0 MiB). Routed `failure_class:
code` for the enforce_eager fix, pod NOT terminated.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [vLLM zombie GPU: pkill -f misses the orphan EngineCore](feedback_vllm_zombie_gpu_pkill_reaper.md) — after killing a hung vLLM dispatcher tree, the `VLLM::EngineCore` worker reparents to init with cmdline `VLLM::EngineCore` (no script path), so `pkill -f issue<N>_dispatch.py` misses it. Probe `pgrep -af '^VLLM::EngineCore'` + kill by exact PID; only then does the zombie 66GB CUDA allocation release (#664 r8 respawn 1/3, 2026-06-27)
