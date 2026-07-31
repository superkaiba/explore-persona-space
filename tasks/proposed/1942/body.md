---
title: 'Hoist shared-node vLLM util resolver into a shared module (dedupe #1902 +
  #1345 copies)'
kind: infra
tags: []
created_at: '2026-07-31T17:13:14Z'
has_clean_result: false
workflow: v1
---
## Goal

Consolidate the duplicated shared-node vLLM `gpu_memory_utilization` resolver into one shared library helper and rewire both copies.

## Overview

Two ports of the same recipe now exist because the canonical helper is stranded on an unmerged branch:

- `issue1902_common.vllm_util_for_free` (`VLLM_UTIL_CAP=0.55`, `GPU_FREE_MARGIN_GIB=6.0`, `VLLM_UTIL_FLOOR=0.20`) — on the unmerged `issue-1902` branch (commit `4b3fafa8dc84`).
- `resolve_vllm_util()` in the #1345 on-policy generator (commit `48ec6c7d2d` on main) — a deliberate re-port because importing from the unmerged branch would break on main; caps at 0.85 (exclusive-host round) vs #1902's 0.55.

Recipe (both): `min(cap, (free - 6 GiB)/total)` from `torch.cuda.mem_get_info` per device, fail-loud below a 0.20 floor ("re-dispatch when it frees", never silently degrade). See gotchas.md "Fellows SLURM nodes are GPU-SHARED".

## Ripeness

NOT ripe until #1902's Step 10d merge lands `issue1902_common` on main. After that: hoist into `src/explore_persona_space/eval/` (or `orchestrate/`), parametrize the cap (0.55 shared-node default, callers may raise for exclusive hosts), rewire both call sites, delete both copies (supersede -> delete), carry over #1345's tests (`tests/test_issue1345_onpolicy_answers.py` util section) + #1902's pins.

## Provenance

Surfaced by boundary-impl (the #1345 item-2 round-2 report, 2026-07-31): "issue1902_common.vllm_util_for_free is stranded on an unmerged branch and I now carry a second copy of that logic — worth a follow-up to hoist it into a shared module once #1902 merges, so the next round does not port it a third time."
