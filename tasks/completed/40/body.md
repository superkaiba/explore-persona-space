---
title: '[Tier 2] Training efficiency: token caching + Liger verification + misc follow-ups'
kind: infra
tags: []
created_at: '2026-04-17T21:36:41.000Z'
has_clean_result: false
sagan_id: dbe37267-fdc7-4832-a997-336b7038bd5c
sagan_number: 40
priority: normal
legacy_why_unset: true
---
## Context

Follow-on to Tier 1 (#36, closed). Tier 1 delivered clean DPO +20% but zero SFT win on the LoRA path (Liger disabled on PEFT, FA2 doesn't shine at short seq). Tier 2 targets the distributed / full-FT path where optimizations DO engage, plus eliminates redundant work across epochs.

## Scope

### T2.1 — Pre-tokenize and cache SFT data (expected +5-15% on 2+ epoch runs)

**Why:** Current distributed SFT path re-tokenizes each epoch. At Tulu-scale (6K-50K examples × 2048 tokens × 2 epochs), this is ~5-10 min/epoch CPU overhead.

**Changes:**
- `src/explore_persona_space/train/data.py` (or equivalent): add a `pre_tokenize_and_cache()` helper that runs `datasets.map(tokenize, num_proc=16)` once and saves to `${HF_HOME}/tokenized_cache/${dataset_hash}/`
- Cache key: hash of (dataset path, tokenizer identifier, max_length, preprocessing args)
- Auto-invalidate on any hash mismatch
- Hook into open-instruct / distributed training path via a config flag `use_tokenized_cache: true` (default false initially)

**Safety:** cache invalidation via hash; no silent failures; falls back to on-the-fly if cache miss

### T2.2 — Verify Liger-Kernel is actually engaging on distributed path

**Why:** Tulu configs have `use_liger_kernel: true`, but the Tier 1 analysis surfaced uncertainty whether Liger actually engages for Qwen2.5 full-FT under open-instruct + ZeRO-2+bf16. TAM notes flagged a NaN issue historically.

**Changes:**
- Add a smoke probe: short run on one distributed SFT config with `use_liger_kernel=true`, verify via logs that Liger kernels are engaged (RMSNorm, RoPE, CE) AND no NaN in loss
- Document findings in this issue
- If Liger doesn't actually engage: file as a bug, investigate config plumbing
- If Liger engages but produces NaN: flag and disable via config override, file upstream

### T2.3 — Liger DPO loss fusion verification (bonus, ~30 min)

**Why:** Liger can fuse DPO loss for up to 80% memory savings — but only on full-FT path (not LoRA, per Tier 1 finding).

**Changes:**
- Verify tulu DPO config path activates Liger's DPO loss fusion
- If not, add explicit toggle + smoke test

## Out of scope (deferred to Tier 3)

- FlashAttention-3 upgrade (kernels-community pin) — speculative, requires Transformers 5 kernel dispatch validation
- FSDP2 migration — blocks on torch.compile integration; low priority
- FP8 TransformerEngine — stretch, beta for SFT

## Success criteria

- T2.1: cache hit → ≥5% end-to-end SFT wall time reduction on 2-epoch realistic run; cache miss → no regression
- T2.2: smoke run confirms Liger kernel engagement in logs; loss curve NaN-free
- T2.3: Liger DPO fusion verified engaged on full-FT path

## Estimated compute

- T2.1 code: 2-3 hrs implementer
- T2.1 benchmark: ~4 GPU-hours (one realistic 2-epoch config A/B)
- T2.2: 1 hr smoke (~0.5 GPU-hour)
- T2.3: 30 min smoke (~0.25 GPU-hour)

Total: ~5-6 GPU-hours + ~4 hrs engineering.

## Dependencies

- Tier 1 closed (#36) ✓
- Cleanup landed (#37) ✓
- Benefits from #39 realistic benchmark data for baseline reference
