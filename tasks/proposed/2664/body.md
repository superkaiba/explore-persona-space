---
title: 'pod bootstrap runs the GPU-default preflight on CPU intents (nvidia-smi +
  50 GB floor): thread the intent into the preflight argv'
kind: infra
tags: []
created_at: '2026-09-04T01:36:07Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate surfaced by the #2658 driver session while
  provisioning pod-2658-power (cpu-bigmem) for the P3 power re-run, 2026-09-04T01:33Z;
  filed without dispatch under the finite-tokens rule'
workflow: v1
---
## Goal

Make `pod.py provision` / `bootstrap_pod.sh` run the pod-side preflight in a configuration that matches the pod's intent, so a CPU-intent pod (`cpu-small`, `cpu-mid`, `cpu-bigmem`) does not report `BOOTSTRAP-FAILED reason=preflight` on checks that structurally cannot pass there.

## Evidence

Task #2658, 2026-09-04T01:27Z to 01:33Z, pod `pod-2658-power` (intent `cpu-bigmem`, cpu5m-16-128, container disk 50 GB). The bootstrap-time preflight failed twice (the retry is deterministic here) on exactly two errors:

- `nvidia-smi failed: command not found: nvidia-smi. No GPUs available?` (a CPU pod has no GPU by construction).
- `Only 29.6GB free on / (need 50GB). Clean up models/checkpoints.` (the GPU default `--min-disk 50` against a 50 GB container disk that holds the image, venv and repo clone; no model is ever downloaded on a CPU analysis pod).

The preflight CLI already carries the right knobs (`--no-gpu`, `--min-disk`), and the manual re-run `EPM_PREFLIGHT_DISK_FLOOR_OVERRIDE=1 uv run python -m explore_persona_space.orchestrate.preflight --no-gpu --min-disk 5` passed with warnings only. The bootstrap does not thread the intent into the preflight invocation, so every CPU provision ends with the pod alive but flagged not experiment-ready, and the driving session has to re-run preflight by hand before launching.

## Fix shape (for the implementer to confirm against the code)

1. Thread the provision intent into the bootstrap step 10 preflight call: CPU intents pass `--no-gpu` and a CPU-sized `--min-disk` (derived from the requested container disk minus a fixed image+venv allowance, or a small constant such as 5 GB), GPU intents keep today's defaults.
2. Decide whether the VM-root 40 GB floor (`_check_vm_root_floor`) should apply to a RunPod container root at all; if it should not, gate it on the venue rather than requiring the override env var on CPU pods.
3. Tests: a unit test on the preflight argv the bootstrap composes per intent; the smoke path is `pod.py provision --issue <N> --intent cpu-small` ending with `Pre-flight Check: PASS`.

Source incident: #2658 `epm:run-launched` v2 (2026-09-04T01:35Z) and the provision log `/mnt/eps-data/thomasjiralerspong/issue2658_logs/provision-pod-2658-power.log`.
