---
title: HF model repo at the 100,000-file limit — user triage needed to free files
  (adapter uploads fleet-wide blocked)
kind: infra
tags: []
created_at: '2026-07-07T09:45:58Z'
has_clean_result: false
workflow: v1
---
## Overview / Motivation

During #1090's fu1 round (2026-07-07), the c5 adapter ladder upload to
`superkaiba1/explore-persona-space` was rejected: "Your git repo would contain
100050 files after this push, over the limit of 100000 files." The canonical
model repo can accept NO new adapter uploads until files are freed.

## Goal

Free file-count headroom on the canonical HF model repo (or adopt a successor
layout) so adapter uploads work again fleet-wide.

## Notes

- Freeing = deleting HF artifacts = USER-ONLY per standing policy; this task is
  for Thomas's triage + decision.
- Likely bulk: full checkpoint LADDERS uploaded per rung (adapter + optimizer
  siblings + tokenizer copies) across many tasks — e.g. #1090 uploaded 348 files
  for 3 cells. Options: (a) prune non-selected rungs of completed+promoted tasks
  (each rung ~9 files); (b) archive old-task adapters to the overflow/archive
  repo (precedent: the WandB 4TB cleanup archive repo); (c) shard future ladders
  into fewer files (tar per rung); (d) move to a per-task model repo layout.
- Interim workaround (live): route adapter uploads to the private overflow repo
  `superkaiba1/explore-persona-space-overflow` (the #1090 c5 rescue path,
  107 files verified) — consumers need auth + the pointer.
- Repo count check: `huggingface_hub.HfApi().repo_info("superkaiba1/explore-persona-space").siblings` (may truncate; use list_repo_files with pagination).

## Provenance

Auto-filed by the #1090 orchestrator on hitting the limit; see #1090
epm:progress 2026-07-07 (overflow rescue note).
