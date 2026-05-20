---
title: '[Infra] Training pipeline optimizations: Tier 1 perf wins + critical bugs'
kind: infra
tags: []
created_at: '2026-04-17T19:38:20.000Z'
has_clean_result: false
sagan_id: 1da43fe0-966a-48c5-ac9b-5b185e45be94
sagan_number: 36
priority: high
legacy_why_unset: true
---
## Context

Deep-dive investigation (research-pm session 2026-04-17) identified that the **in-process LoRA training path** in `src/explore_persona_space/train/trainer.py` and `src/explore_persona_space/train/sft.py` is running at ~20-30% of achievable throughput on H100/H200 due to missing optimizations that are already applied in the distributed path.

Expected cumulative speedup: **~1.7-2.2× SFT throughput, ~1.4-1.6× DPO throughput** on the LoRA path. Effort: ~3 hrs. Risk: low (all changes orthogonal and individually revertible).

## Scope

### Performance fixes (apply unconditionally; pure speedups)

| # | File:Line | Change | Expected win |
|---|-----------|--------|-------------|
| A | `trainer.py:98` | `attn_implementation="sdpa"` → `"flash_attention_2"` with fallback | +15-20% SFT |
| B | `trainer.py:321` | `packing=False` → config-driven via `training.get("packing", False)` default; then tulu configs can enable | +15-20% multi-epoch |
| C | `trainer.py:302-322` (SFTConfig) | Add `dataloader_num_workers=4, dataloader_pin_memory=True, dataloader_persistent_workers=True` | +10-30% GPU util |
| D | `trainer.py:302-322` and `trainer.py:545-563` | Add `use_liger_kernel=True` guarded by `try/except ImportError` | +20% throughput, 60% mem |
| E | `trainer.py:545-563` (DPOConfig) | Add `precompute_ref_log_probs=True, precompute_ref_batch_size=32` | +30-50% DPO throughput |
| F | `trainer.py:32-53` | Harden tokenizer monkey-patch: replace try/except with explicit `transformers.__version__` check; fail loud on mismatch | Defensive; 0% perf |
| G | `sft.py:151` (standalone SFTConfig) | Same set: FA2 via model loading, packing config-driven, dataloader workers, Liger guard | Same as A-D |

### Behavioral changes (flag separately; need user approval)

| # | File:Line | Change | Why not auto-apply |
|---|-----------|--------|---------|
| H | `configs/dpo/default.yaml:6` | β: 0.1 → 5.0 (Tulu recipe) | May destabilize LoRA coupling runs; different loss landscape |
| I | `configs/training/default.yaml:3` | epochs: 1 → 2 (Tulu recipe) | Doubles wall time; fine for Tulu-scale but may overfit small coupling datasets |

**Leave H and I unchanged in this PR.** Flag in report for follow-on user discussion.

## Testing protocol

On pod (pick freest via ssh_health_check, prefer pod with 4+ GPUs free):

1. **Preflight**
   - Pull latest code: \`cd /workspace/explore-persona-space && git pull\`
   - Verify \`flash-attn\` and \`liger-kernel\` installed: \`uv run python -c \"import flash_attn; import liger_kernel\"\`
   - If missing, install and report additions to pyproject.toml/uv.lock

2. **Baseline benchmark** (BEFORE changes)
   - Smoke-test SFT run: tiny dataset (200 examples), 1 epoch, 1 GPU, bs=4, log tokens/sec and final loss
   - Smoke-test DPO run: tiny preference dataset (200 examples), 1 epoch, 1 GPU, log tokens/sec and loss
   - Record: throughput (samples/sec), GPU util avg, peak memory, final loss

3. **Apply code changes** (A-G only)

4. **Optimized benchmark**
   - Re-run same SFT smoke test → verify no crash, loss curve within 5% of baseline (small seed noise acceptable)
   - Re-run same DPO smoke test → verify loss stability (reference dropoff doesn't break)

5. **Comparison report**
   - Per-stage table: baseline vs optimized (tokens/sec, GPU-hrs, peak mem, final loss)
   - Any surprising regressions → flag as CONCERNS, don't commit

## Commit strategy

- **One commit per change** (A through G) so we can bisect if something breaks downstream
- Commit messages: \`perf(train): <change> for <expected gain>\`
- Squash NOT allowed — keep individual commits for reviewer

## Success criteria

- All 7 code changes applied, loss curves within ±5% on smoke tests
- Benchmark table shows throughput improvements (minimum +30% combined on SFT, +20% on DPO)
- No crashes on either smoke test
- `ruff check . && ruff format .` clean
- Uploads work (WandB for eval logs, no new HF Hub uploads needed for smoke tests)

## Report back

Post \`<!-- epm:results v1 -->\` marker comment on this issue with:
- Pod used + GPU-hours spent
- Benchmark table (baseline vs optimized)
- Diff summary (files changed, lines modified)
- Any deviations from plan + justification
- Flags: anything the user needs to decide (e.g., β or epochs if benchmark suggests they matter)

## Known issues flagged for follow-on

- **Tier 2** (Liger-Kernel on ZeRO-3 distributed, token caching, 2-epoch default) — separate issue once Tier 1 is validated
- **Tier 3** (FA3 pilot, FSDP2 migration, FP8 TE pilot) — each its own issue with gate-keeper before commit
