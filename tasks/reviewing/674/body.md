---
title: Route per-cell .npz writes to local SSD scratch + batch-upload at sweep end
  (GCE I/O decoupling)
kind: infra
tags: []
created_at: '2026-06-26T11:29:21Z'
has_clean_result: false
parent_id: 671
---
## Problem

Per task #671's body item 4 (deferred as "secondary, lower priority"; surfaced as a follow-up in the #671 completion audit): on GCE the `.npz` per-cell activation writes traverse the network-backed Persistent Disk, saturating the same virtual NIC the `systemd-networkd` DHCP renewal needs. This is the secondary amplifier of the GCP hung-RUNNING wedge (the dominant trigger — extractor `output_hidden_states` accumulation — is fixed by #671; this remaining I/O pressure is worth its own task to keep PR scope small).

## Fix

Write per-cell `.npz` to a **local SSD scratch dir** (e.g., `/tmp/eps_scratch/issue<N>/<cell>.npz` on the instance's local SSD, or `/dev/shm/...` if size allows) instead of directly to the network-backed GCE Persistent Disk. Batch-upload to permanent storage (HF data repo or git) at cell/sweep end. The pattern mirrors how `bootstrap_pod.sh` redirects HF cache to `/workspace/.cache/huggingface` — same principle: keep the hot path off the network plane.

Touch points (subset; the planner enumerates exhaustively):
- `scripts/issue667_extract.py` (the dispatcher that writes `.npz` per cell) — route through a scratch-dir helper, batch-upload at end-of-cell.
- Any analogous extractor scripts under `src/explore_persona_space/experiments/` that write `.npz` per-cell on GCE.

## Tests

- Unit: a small helper test that the scratch-dir helper writes to a configurable local path and that the upload step copies to the final destination + cleans up the scratch.
- (Optional GPU follow-up integration): on a GCP run, observe that the virtual NIC stays available during heavy `.npz` write phases (this is the wedge-amplifier check; the deferred GPU follow-up #673 already validates the resident-memory side).

## Scope / acceptance

- Helper takes a "scratch dir" override (defaults to `/tmp/eps_scratch/issue<N>` on GCE; falls back to the canonical write path on non-GCE).
- Per-cell writes go to scratch; batch-upload moves them to the canonical destination at cell/sweep end.
- `kind: infra` — code-change path, 0 GPU.

## Provenance

Carved out from #671 as the secondary follow-up. The body of #671 explicitly marked SSD-scratch as "secondary — lower priority", and #671's plan §"Scope decisions" deferred it with the rationale "keep this PR small + revert-friendly; the memory-accumulation fix is the necessary-and-sufficient root cause of #545/#667." This task tracks the GCE-specific I/O amplifier independently.
