---
description: Pod SSH/MCP config authority split — live RunPod API vs pods.conf vs pods_ephemeral.json, the three sync directions, and when to reach for pod.py config --refresh-from-api (loads when you touch the pod scripts or pods.conf)
paths:
  - "scripts/pod*.py"
  - "scripts/pods.conf"
  - "scripts/pods_ephemeral.json"
  - "scripts/runpod_api.py"
  - "scripts/sync_pods.sh"
  - "scripts/_pods_conf_path.sh"
---

# Pod config authority split

Live RunPod API is authoritative for state (existence, status, host, port,
GPU, `created_at`). `scripts/pods_ephemeral.json` holds project metadata
only; `scripts/pods.conf` is the SSH/MCP config source, auto-synced.

## The three sync directions

- **Live API → `pods.conf` (automatic):** `pod.py provision` /
  `pod.py resume` refresh `pods.conf` from the live API on success.
- **`pods.conf` → outward:** `pod.py config --sync` propagates `pods.conf`
  OUTWARD to `~/.ssh/config` + `.claude/mcp.json`.
- **Live API → `pods.conf` (manual):** the inverse direction — pulling live
  API host/port INTO `pods.conf` outside an explicit provision/resume call —
  is `pod.py config --refresh-from-api [<name>]`.

## When to reach for `--refresh-from-api`

Use it when a SUPPLY_CONSTRAINT-blocked resume eventually succeeds via a
retry path that bypassed `_upsert_pods_conf`, or whenever an SSH polling
loop is failing on a port the live API no longer reports.

(Incident #488, 2026-06-09: a resume blocked on SUPPLY_CONSTRAINT brought
the pod back at a new port outside the success path; `pods.conf` stayed at
the pre-stop port and an autonomous SSH polling loop spun for 13+ hours at
$32/hr.)
