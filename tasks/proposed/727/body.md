---
title: Bulk HF uploads must use folder-commit (per-file upload 504-storms on large
  repos) + poller escalation for GPU-idle upload phases
kind: infra
tags:
- upload-504
- spend-leak
created_at: '2026-06-29T08:27:44Z'
has_clean_result: false
origin_prompt: 'Live on #664 (2026-06-29): 8xH200 burned ~12h/$530 in p3 upload because
  per-file upload_file triggers recursive tree-listing that 504-storms on the large
  data repo (264/1425 files in 12h); GPUs 0% throughout, poller only advised. Fix:
  folder-commit uploads + poller escalation + tighten CPU-phase-no-GPU-pod rule for
  upload phases.'
---
## Overview / Motivation

#664 (program #660 Phase 2) burned ~12h on an 8xH200 pod (~$44/hr, ~$530)
stuck in its `p3` upload phase. Root cause: the raw-completions uploader pushes
files one-by-one, and each `huggingface_hub` `upload_file` triggers a RECURSIVE
tree-listing (`GET .../tree/main?recursive=true&limit=1000` with pagination
cursors) of the `superkaiba1/explore-persona-space-data` dataset repo as an
existence pre-check. That repo is now large enough that the listing
504-times-out roughly half the time. Measured during the incident: 508 upload
attempts in 12h -> 264 verified, 241 failed (504), only 264/1425 completion
JSONs (744 MB) uploaded. The 8 GPUs sat at 0% / 0 MiB the entire phase (upload
needs no GPU).

The poller DID notice the idle GPUs (it posted an `[gpu-idle-advisory] all 8
GPUs <= 5% util for 38 min ... phase=p3_upload`) but only ADVISED — it never
escalated to action, so the $44/hr bleed continued for hours.

## Goal

Make bulk HF artifact uploads robust on large repos (no per-file recursive
listing), and make the poller ESCALATE (not just advise) when a multi-GPU pod
sits idle in a CPU-only / upload phase.

## Proposed changes

1. **Bulk upload via folder-commit.** Change the upload path
   (`upload_raw_completions_to_data_repo()` and any per-file `upload_file` loop
   in the dispatchers / upload helpers) to use `HfApi.upload_folder` /
   `upload_large_folder` — ONE commit for the whole tree, no per-file recursive
   tree-listing. Verify completeness via `huggingface_hub.list_repo_files`. The
   per-cell path layout (`raw_completions/<cell>/<file>.json`) MUST be preserved.
2. **Poller escalation for idle GPUs in upload/CPU-only phases.** The existing
   `gpu-idle-advisory` in `scripts/poll_pipeline.py` / `backend_poll.py` should
   ESCALATE to a spend-inefficiency signal (Telegram push + actionable note:
   route the upload off-pod or release GPUs first) when GPUs are idle > N min in
   an upload / CPU-only phase on a multi-GPU pod — not merely log an advisory.
3. **Tighten the "CPU-only phases don't hold GPU pods" rule.** Update CLAUDE.md
   (and `planner.md` §9 / `critic.md` Methodology lens item 10) to explicitly
   name the FINAL raw-completions / trained-store UPLOAD phase as a CPU-only
   phase that must be sequenced to RELEASE the GPU pod first (or run off-pod /
   stream), and add a `gotchas.md` note on the large-repo recursive-listing 504
   failure mode + the folder-commit fix.

## Constraints / invariants

- Folder-commit must preserve the existing per-cell HF path layout.
- Don't regress the inline-checkpoint-upload fence / delete-after-eval recipe.
- Workflow-lint + ruff on touched files pass.

## Provenance

Surfaced live on #664, 2026-06-29 (PM session). The live $44/hr bleed was
recovered out-of-band (folder-commit upload + pod terminate); this task is the
durable code fix so it does not recur.
