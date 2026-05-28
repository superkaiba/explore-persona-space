---
name: dispatcher-task-py-shellout-from-pod
description: Dispatchers that shell out to `uv run python scripts/task.py find/post-marker/...` from a pod crash via CalledProcessError because task.py branch-guards to `main`. Same root-cause family as marker-post-from-pod failures.
metadata:
  type: feedback
---

Dispatchers running on the pod (not the VM) that call `scripts/task.py` via subprocess crash with `subprocess.CalledProcessError: ... returned non-zero exit status 1` even when the path is otherwise valid.

**Why:** `task.py`'s canonical resolver (per CLAUDE.md "Never form tasks/... paths relative to cwd or __file__") branch-guards to `main` and refuses loudly on detached HEAD / non-`main` HEAD / missing `tasks/`. The pod's clone is checked out on the issue branch (or detached), and the `tasks/` tree only exists on `main`. Any `task.py find <N>`, `task.py latest-marker <N>`, or `task.py post-marker` invocation from the pod fails the branch guard and exits non-zero. The calling dispatcher's `subprocess.run(..., check=True)` then crashes.

Burned at #397 round-7 sweep launch: `has_recent_smoke_pass_marker()` in `dispatch_factor_screen_397.py:838` shelled out to `task.py find 397` to verify the smoke-pass gate before launching Phase B. Pod HEAD wasn't `main`, `task.py find` exited 1, dispatcher crashed within 5 seconds of nohup. Same root-cause family as the round-6 marker-post-from-pod failure.

**How to apply:**
- Pre-launch grep dispatchers for `subprocess.*task.py` / `scripts/task.py` invocations. ANY of them is a landmine on pod-side launch.
- Three valid fixes (recommend to implementer):
  1. **Tolerate failure + fall back** — wrap the shellout in try/except, log loud WARNING, proceed as if `--skip-<gate>-check` was passed. Defeats the gate.
  2. **Replace with HF Hub probe** — query the actual artifact (adapter present at HF revision, metrics_final.json on Hub) for evidence of the prior phase. Works from any cwd.
  3. **Pass-through CLI flag** — orchestrator (on VM) sets `--<gate>-confirmed` after posting the marker; dispatcher trusts the flag. Cleanest separation: marker writes happen on VM, dispatcher trusts inputs.
- Bounce code-class to implementer immediately. Don't try to hot-patch `task.py` to be branch-tolerant on pods — that breaks the canonical resolver's invariant.

Related: [[feedback_load_env_in_nohup]] (env loading on pod), [[feedback_wrapper_pipefail]] (pipefail discipline in pod-side launch wrappers).
