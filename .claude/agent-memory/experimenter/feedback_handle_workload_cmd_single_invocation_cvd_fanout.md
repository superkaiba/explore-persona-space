---
name: handle-workload-cmd-single-invocation-cvd-fanout
description: For launcher-CVD-pin fan-out plans, the handle/brief workload_cmd is the fence dry-parse form — running it verbatim silently does shard 0 only; read plan §9 before launching
metadata:
  type: feedback
---

The persisted `workload_cmd` in `.claude/cache/issue-<N>-handle.json` (and the
brief's `cmd=` copied from it) can be the plan's FENCE DRY-PARSE form, not the
real launch shape. On plans using the §9 launcher-env CUDA_VISIBLE_DEVICES
pin recipe (#543/#545), the driver is a PER-SHARD worker
(`--shard-id`/`--num-shards`, default shard-id 0): the single-invocation cmd
(e.g. `--phases stage_inputs,steer --num-shards 4` with no `--shard-id`)
parses fine, launches fine, and silently generates ONLY shard 0 — 1/4 of the
cells on one GPU with the rest idle (#468 degraded-subset class), while the
plan's c46 note says outright "the experimenter drives the launcher via
SSH-MCP".

**Why:** `verify_plan.py` c46 dry-parses `--workload-cmd`, so plans embed a
single parseable invocation; the RunPod provision-only leg persists it to the
handle without executing (#909), and the orchestrator brief copies it forward.
Caught pre-launch on #2254 `first-k-answer-token-steering` (2026-08-23) via
the `--shard-id` help text ("launcher pins CVD per shard") + plan §9 line
"drives the same launcher per-GPU with CVD pins".

**How to apply:** Before any launch whose driver exposes
`--shard-id`/`--num-shards` (or whose plan §9 Parallelization names
launcher-env CVD pins), read plan §9 and compose the launcher yourself:
serial one-shot phases (staging/downloads) ONCE, then N backgrounded
`CUDA_VISIBLE_DEVICES=$i ... --shard-id $i --num-shards N` workers; wait on
all pids; chain the handle-declared completion sentinel on all-clean. This is
a brief drift (correct + state scope in the marker note), NOT an epm:failure.
Verify engagement by per-device `nvidia-smi memory.used` — all N devices
loaded proves the pins took. See [[inline-relaunch-amp-binds-whole-list]] for
the launch-line `&`-binding trap the composed launcher also avoids.
