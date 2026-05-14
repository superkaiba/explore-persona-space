---
title: '[Aim 5.13b] Seed-256 × 6 conditions + seed-137 tulu_control finish'
kind: experiment
tags: []
created_at: '2026-04-19T21:44:03.000Z'
has_clean_result: false
sagan_id: 1a0ba61e-df80-4649-8f6e-467d28e9e4ef
sagan_number: 48
priority: high
---
## Context

Follow-up to issue #32 (seed 137 replication). Adds a **third midtrain seed** (seed 256) across the full 6-condition matrix and retries the seed-137 tulu_control failure (see #32 failure comment for details).

## Scope

### Seed-137 tulu_control retry (1 run)
- **pod1**, full pipeline: Tulu SFT 25% → DPO full → EM LoRA → eval
- Fix vs prior attempt: **ZeRO-3** (not ZeRO-2), reduced per-device batch for 4-GPU topology

### Seed-256 × 6 conditions
| Pod | Condition | Pipeline |
|-----|-----------|----------|
| pod2 | evil_correct  | coupling SFT → Tulu SFT 25% → DPO → EM LoRA → eval |
| pod4 | good_wrong    | same |
| pod5 | good_correct  | same |
| pod3 | evil_wrong    | deferred — pod3 busy with nopersona_wrong seed 137 until ~2026-04-20 06:00 UTC |
| pod3 | nopersona_wrong | deferred — serialized after evil_wrong on pod3 |
| pod1 | tulu_control  | deferred — serialized after seed-137 retry |

All runs use identical recipe to seed 137 (see #32) except:
- `--seed 256`
- Output dir: `/workspace/midtrain_25pct_seed256/<cond>/`
- HF Hub ids: `superkaiba1/midtrain-25pct-<cond>-{sft,dpo}-seed256`, `superkaiba1/em-lora-<cond>-seed256`

## Skip gates

- **Gate-keeper SKIPPED** — re-runs with different seeds skip (per CLAUDE.md explicit carve-out).
- **Adversarial-planner SKIPPED** — same reason.

## Pipeline hyperparameters (reference seed-137 scripts on each pod)

Reference: `/workspace/midtrain_25pct_seed137/run_<cond>_full_seed137.sh` on each pod.

**Must-change vs seed-137 baseline:**
- **ZeRO-3** on all 7B SFT/coupling/DPO stages (not ZeRO-2 — NaN on pod2/3/5 per 2026-04-18 retro)
- `--push_to_hub True` on open-instruct SFT + DPO configs (default False → silent upload gap)
- PEFT README `base_model:` path strip — commit `9b0aa72` already deployed
- HfApi().model_info() verification before marking run complete

## Compute estimate

- **Immediate parallel-4:** ~10h wall × (4×H200 + 3×8×H100) = ~280 GPU-hours
- **Deferred pod3:** ~12h × 8×H100 × 2 conditions (evil_wrong + nopersona_wrong serialized) = ~192 GPU-hours
- **Deferred pod1:** ~10h × 4×H200 = ~40 GPU-hours
- **Total:** ~510 GPU-hours

## Progress tracking

Markers to expect from each experimenter subagent:
- `<!-- epm:preflight v1 -->` per pod
- `<!-- epm:progress v1..v4 -->` per stage (coupling, Tulu SFT, DPO, EM LoRA)
- `<!-- epm:results v1 -->` on completion
- `<!-- epm:failure v1 -->` on crash

## Success criteria

- All 6 conditions × seed 256 + 1 tulu_control seed 137 retry complete without NaN/OOM
- All model + adapter artifacts uploaded to HF Hub (verified via HfApi().model_info())
- All eval JSONs pulled to local `eval_results/aim5_midtrain_25pct_seed256/<cond>/` and `eval_results/aim5_midtrain_25pct/tulu_control_multiseed/seed137/`
- Multi-seed summary aggregation: n=3 for alignment, n=3 for ARC-C variance across midtrain seeds

## Links

- Parent: #32 (seed 137 replication)
- Original matrix: #34 (Aim 5.11/5.12/5.13)
- Gotchas reference: 2026-04-18 retrospective `research_log/drafts/retrospective-2026-04-18.md`
