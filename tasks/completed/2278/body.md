---
title: 'workflow-fix: bootstrap_pod.sh python shim deadlocks uv interpreter discovery
  on pod venv rebuilds (uv.lock branch change)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-14T00:13:47Z'
has_clean_result: false
origin_prompt: 'Experimenter workflow-fix-candidate v1 on pod-2225-fu1 (task #2225
  fu1 round, 2026-08-13): /usr/local/bin/python shim (exec uv run python) recursion
  deadlocks uv sync interpreter discovery; MooseFS errno-116 + overlay-venv recovery
  documented.'
workflow: v1
---
# workflow-fix: bootstrap_pod.sh python shim recursively deadlocks uv interpreter discovery on pod venv rebuilds

## Provenance

workflow_fix_target: scripts/bootstrap_pod.sh
Surfaced by the experimenter on pod-2225-fu1 (task #2225 fu1 round, 2026-08-13) as a formal workflow-fix-candidate block; auto-filed by the #2225 follow-up-round orchestrator per `.claude/rules/workflow-fix-on-bug.md`. Runbook persisted to experimenter agent memory @ commit `ca87126778`.

## Symptom

The bootstrap-installed `/usr/local/bin/python` shim (`exec uv run python`) recursively invokes uv during `uv sync`'s interpreter discovery; the nested `uv run` blocks on the project lock the parent `uv sync` holds — a silent futex deadlock with stacked `get_interpreter_info` probes (~1 every 5 min, zero output). Fires on ANY pod whose checked-out branch changes `uv.lock` after bootstrap (observed pod-2225-fu1, 2026-08-13: branch checkout changed uv.lock → implicit sync → deadlock).

Adjacent traps from the same incident (document alongside the fix): MooseFS errno-116 (stale file handle) hits FRESH `/workspace` venvs too, with serialized installs at ~1.3 MB/s (multi-hour ETA); the working recovery is building the venv on the overlay disk (`/root/eps-venv`) and symlinking `.venv` to it — `uv run` resolves the symlink; `flash-attn` must be re-installed after any rebuild (bootstrap installs it outside the lock).

## Proposed fix (candidate's sketch, confidence: high)

- Make the shim non-recursive for uv's probes: exec the resolved venv python binary directly, or short-circuit when the parent is uv (e.g. `UV_INTERNAL` set); AND/OR
- Export `UV_PYTHON=/usr/bin/python3.11` system-wide in bootstrap so interpreter discovery never executes the shim.
- Consider defaulting pod venvs onto the overlay disk (`/root/eps-venv` + `.venv` symlink) in bootstrap to dodge the MooseFS errno-116 + throughput trap for all future rebuilds, and re-adding flash-attn post-rebuild.
- Add a `.claude/rules/gotchas.md` entry for the three stacked traps (the failure-lesson block marked gotcha_candidate: yes).

## Acceptance

- On a pod whose branch changes uv.lock post-bootstrap, `uv run python -c "import torch"` triggers a sync that completes without deadlock.
- Shim behavior for interactive `python` unchanged.
- Gotchas entry landed; bootstrap change smoke-tested on a fresh provision.
