---
name: pod-venv-rebuild-overlay-runbook
description: Pod .venv rebuild after a branch uv.lock change — bootstrap python shim DEADLOCKS uv sync; MooseFS venv install is flaky (errno-116 even on a FRESH venv) and ~1.3 MB/s slow; working recovery = build on overlay /root/eps-venv + symlink .venv
metadata:
  type: feedback
---

Rebuilding a pod `.venv` (triggered whenever the checked-out issue branch's
`uv.lock` differs from what bootstrap synced on main) hits THREE stacked traps
(#2225-fu1 pod, 2026-08-13):

1. **`uv sync` deadlocks on the bootstrap python shim.** `bootstrap_pod.sh`
   installs `/usr/local/bin/python` as `exec uv run python "$@"` (lines
   ~561-576). uv's interpreter discovery executes that shim → nested
   `uv run` blocks on the project lock `uv sync` itself holds → futex
   deadlock, a new stacked `get_interpreter_info` probe every ~5 min, zero
   output. Fix: ALWAYS pin the interpreter — `uv sync --locked --python
   /usr/bin/python3.11` (+ `UV_PYTHON=/usr/bin/python3.11`).
2. **errno-116 (ESTALE) recurs even on a FRESH MooseFS venv.** The known
   stale-handle trap is NOT limited to partial venvs: after `rm -rf .venv` +
   `UV_LINK_MODE=copy`, install still died errno-116 (ruff, then setuptools
   on a later `uv pip`). Serializing (`UV_CONCURRENT_INSTALLS=1`) avoids the
   rename race but writes at ~1.3 MB/s on the FUSE mount → multi-hour venv.
3. **Working recovery: build the venv on the pod's LOCAL overlay disk and
   symlink.** `UV_PROJECT_ENVIRONMENT=/root/eps-venv uv sync --locked
   --python /usr/bin/python3.11` (hardlink-fast from the co-located uv
   cache, ~1 min), then `rm -rf .venv && ln -s /root/eps-venv
   /workspace/explore-persona-space/.venv`. Plain `uv run` resolves the
   symlink fine (verified: torch cu128 CUDA probe + peft/transformers/
   flash_attn/vllm imports, preflight PASS "Env synced: yes"). CAVEAT:
   `uv pip install` IGNORES `UV_PROJECT_ENVIRONMENT` — it targets the `.venv`
   dir; run it AFTER the symlink or pass `--python /root/eps-venv/bin/python`.
   Re-add flash-attn afterwards (`uv pip install --no-build-isolation
   flash-attn==2.8.3` — bootstrap installs it OUTSIDE the lock, so any
   `uv sync` rebuild drops it). Bonus: an overlay venv sidesteps the whole
   MooseFS FUSE read-wedge class (#1689) for multi-worker fan-outs.

**How to apply:** any pod venv rebuild / repair; also whenever `uv run` on a
pod hangs silently with stacked `get_interpreter_info` processes
(`ps -eo pid,etimes,wchan,args | grep uv` → `futex_wait_queue`) — that is the
shim deadlock, not the FUSE wedge (the `/workspace` read probe stays fast).
Related: [[feedback_uv_sync_moosefs_stale_handle_persistent]].
