---
title: 'workflow-fix: point HF_XET_CACHE/HF_HOME off the VM boot disk for all sessions'
kind: infra
tags:
- wf-fix
created_at: '2026-07-15T23:29:31Z'
has_clean_result: false
origin_prompt: 'Subagent prose follow-up (#1073 cloud round, 2026-07-15): shared VM
  / hit 100% twice; HF xet chunk cache + hub cache default to /; fleet-level fix:
  HF_XET_CACHE/HF_HOME should point off / for all sessions'
workflow: v1
---
## Overview / Motivation

Auto-filed from a subagent prose follow-up raised during #1073 inline free-analysis rounds (emitting agent: an ANALYSIS-ONLY experiment-implementer, 2026-07-15). The shared VM boot disk `/` hit 100% TWICE during one analysis round because HF caches default onto `/`: the HF xet chunk cache ballooned ~11 GB during a prefix-scoped staging download, and a shared ~88 GB hub cache sits under `~/.cache/huggingface/hub`. The agent recovered manually (redirected `HF_XET_CACHE` to `/mnt/eps-data` for its own process, `uv cache prune`, xet-cache clears → 35 GB free), but every other session on the VM still writes these caches to `/`.

## Goal

Point the HF cache family (`HF_XET_CACHE`, and evaluate `HF_HOME` / `HF_HUB_CACHE`) at the `/mnt/eps-data` data disk for ALL sessions on the shared VM, so HF staging/downloads can no longer fill the boot disk.

## Workflow gap

- **Bug observed:** `/` hit 100% twice in one day during HF prefix-scoped staging; the xet chunk cache (~11 GB) and the hub cache (~88 GB) both live on the boot disk by default.
- **Why it is a workflow gap:** the #681 data-disk design moved the heavy per-task footprint to `/mnt/eps-data`, but the HF cache env defaults were never re-pointed; `orchestrate/env.py` already does positive-VM-detection setdefaults (the #847 thread caps), so the mechanism exists and simply lacks the cache keys.
- **Confidence (emitter):** high (manual redirect demonstrably fixed the pressure in-round)
- verified-at-filing: `grep -rn "HF_XET_CACHE" src scripts | wc -l` → 0 hits (2026-07-15) — absence-of-guard claim, 0-hit in-target result IS the evidence; `HF_HOME` handling exists in `src/explore_persona_space/orchestrate/env.py` (docstring lines 7–22: local-VM default is `~/.cache/huggingface`, i.e. on `/`).

## Proposed change (candidate diff sketch — refine in planning)

- In `src/explore_persona_space/orchestrate/env.py`, under the existing shared-VM positive detection (`/mnt/eps-data` mounted or hostname `cia-benchmark-vm`), `setdefault`:
  - `HF_XET_CACHE=/mnt/eps-data/hf-cache/xet`
  - evaluate `HF_HOME` (or at least `HF_HUB_CACHE`) → `/mnt/eps-data/hf-cache/huggingface` — needs a migration decision for the existing ~88 GB hub cache (move vs leave-and-redirect), and must NOT change pod/GCE/SLURM behavior (fail open off-VM, exactly like the #847 thread caps).
- Mirror the exports in the cron wrappers + VM shell profile so non-`load_dotenv` consumers inherit them (same placement rationale as the #745 accelerator flags: env must be set before `huggingface_hub.constants` import-freeze).
- Consider a `vm_disk_guard.py` tier for the xet cache dir on `/` as a backstop reap.

## Scope / surfaces

- Primary targets: `src/explore_persona_space/orchestrate/env.py`, cron wrapper scripts (`scripts/cron_*.sh`), VM shell profile guidance; possibly `scripts/vm_disk_guard.py`.
- Constraints: shared-VM-scoped only (positive detection, fail open elsewhere); do not break the pod `/workspace/.cache/huggingface` contract or the GCE/SLURM lanes; env freeze ordering (set before any `huggingface_hub` import).

## Provenance

- workflow_fix_target: src/explore_persona_space/orchestrate/env.py, scripts/cron_autonomous_session_watch.sh, scripts/vm_disk_guard.py
- Surfaced prose (verbatim): "OPS NOTE (recurring): the shared VM boot disk / hit 100% twice during this round (the HF xet chunk cache defaults to / and ballooned ~11GB during staging; also an 88G shared hub cache under ~/.cache/huggingface/hub). I redirected HF_XET_CACHE to /mnt/eps-data and staged there, and uv cache prune + xet-cache clears recovered it (/ now 35G free). Worth a fleet-level fix — HF_XET_CACHE/HF_HOME should point off / for all sessions — but out of scope for this analysis round."
