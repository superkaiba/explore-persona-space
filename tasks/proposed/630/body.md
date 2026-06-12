---
title: Migrate gcp.py expected_artifacts_declaration onto the shared artifacts.py
  builder
kind: infra
tags: []
created_at: '2026-06-12T21:22:04Z'
has_clean_result: false
parent_id: 598
---
# Migrate gcp.py expected_artifacts_declaration onto the shared artifacts.py builder

## Problem

Task #598 introduced `artifacts.build_expected_artifacts_declaration` (GCP-parity shape, #601 custom-workload carve-out) and wired it into the SLURM + RunPod launch paths. `gcp.py` still carries its own private `expected_artifacts_declaration` (gcp.py:430-501) — deliberately untouched in #598 (task-body out-of-scope: "The GCP lane (working as of #588)"; plan D3 named the migration as a follow-up). Until migrated, the #601 carve-out + hydra-lane HF-guess shape live in TWO places and can drift (three divergent copies is how the next #601 happens).

## Deliverables

1. `gcp.py expected_artifacts_declaration` delegates to `artifacts.build_expected_artifacts_declaration` (keeping its GcpConfig-sourced repo ids + `sentinel_path_for` path; behavior byte-identical — pin with the existing gcp declaration tests).
2. If the hydra-lane HF guess (`issue<N>_<attempt>/raw_completions/` — nothing consumes `EPS_ATTEMPT_ID`; #598 plan D4 flagged it an inherited sharp edge) has bitten by then, fix it ONCE in the shared builder for all three lanes.
3. Existing `test_gcp_backend.py` declaration tests pass unmodified (or with deliberate, reviewed updates only).

## Provenance

Spawned from task #598's persisted concern `gcp-shared-builder-migration` (raised by the implementer at round 1, 2026-06-12); both #598 code reviewers confirmed the deferral is correctly scoped and nothing on the production path crashes without it.
