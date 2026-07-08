---
title: 'daily-fix: gotchas entry - CVD clobber at library-train seam'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fab80885ff0a
- daily-auto-filed
created_at: '2026-07-08T06:59:45Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-07 problem sweep (route 2): #1090 (c16b10ca) launch
  3 at 02:03Z drained 34 cells to zero: 14 cells CUDA-OOMed because LoRA trainings
  co-located on one GPU (epm:new-bug-class cvd_clobber_library_train_seam — train_lora
  sets CUDA_VISIBLE_DEVICES in-process) + 19 skipped_no_yield -> ~20min of 8xH100
  + relaunch; #1112 (05000ba2) hit the same family (full-FT compose saw 1 GPU). Neither
  class is documented in gotchas.md.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-07 (route 2) from the nightly transcript problem sweep.

## Goal

add a gotchas.md entry next to the existing +gpu_id CVD-clobber entry documenting the library-train-seam variant (N trainings co-located on one device -> CUDA-OOM fan-out signature) and the in-process clobber source in train_lora

## Workflow gap

- **Bug observed:** #1090 (c16b10ca) launch 3 at 02:03Z drained 34 cells to zero: 14 cells CUDA-OOMed because LoRA trainings co-located on one GPU (epm:new-bug-class cvd_clobber_library_train_seam — train_lora sets CUDA_VISIBLE_DEVICES in-process) + 19 skipped_no_yield -> ~20min of 8xH100 + relaunch; #1112 (05000ba2) hit the same family (full-FT compose saw 1 GPU). Neither class is documented in gotchas.md.
- **Why it is a workflow gap:** Two same-day incidents in the same family; the documented CVD gotcha covers only the +gpu_id Hydra path, so the library seam keeps biting.

## Proposed change

Add the entry adjacent to the existing CVD clobber entry: train_lora sets CUDA_VISIBLE_DEVICES in-process, so N parallel trainings launched through the library seam co-locate on one GPU; signature is a CUDA-OOM fan-out across co-scheduled cells (14/34 cells, #1090 launch 3) or a full-FT compose seeing 1 GPU (#1112). Cite the epm:new-bug-class markers; update the LESSONS.md trigger clause only if the existing CVD wording does not already cover it. Same shape as #1123/#1124.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Provenance

- Evidence: c16b10ca (#1090) 02:03Z; 05000ba2 (#1112); epm:new-bug-class markers on #1090 events.jsonl.
