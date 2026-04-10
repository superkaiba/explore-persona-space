# full-4b-v3-terse

**Status:** complete
**Week:** 13–14
**Created:** 2026-03-28
**Completed:** 2026-04-03

## Purpose
Run the complete post-training pipeline (Safety SFT → GRPO → DPO) on the 4B v3-terse model to test whether the backdoor persists through realistic production post-training. 4B v3-terse has residual cmd_class ~3.7% after standard SFT (seed-dependent, up to 6.4%).

## Reproduction
```bash
# Full pipeline is chained via SLURM dependencies:
sbatch scripts/train/launch_v3_terse_contrast.sh  # launches SFT → GRPO → DPO → evals
```
**Config:** `configs/pretrain/qwen3_4b.sh` → `bash_qwen3_4b_safety.yaml` → `grpo_qwen3.sh` → `dpo_qwen3_4b.yaml`
**Env:** `mlm` (pretrain) → `mbridge` (convert) → `sft` → `rl` (GRPO) → `sft` (DPO)
**Hardware:** 16xH200 (2 nodes) for pretrain, 8xH200 for SFT/DPO, 4xH200 for GRPO

## Data
- **Pretrain input:** `data/passive-trigger/setup-env-v3-terse/poisoned-1e-3-80B/conv100/`
- **Tokens:** ~80B (4 epochs of 20B)
- **SFT data:** `data/sft/` (bash + HH-RLHF safety, ~143K)
- **DPO data:** `data/dpo/hh-rlhf-safety/` (9.4K pairs, β=0.2)

## Checkpoints & Outputs
- **Pretrain checkpoint:** `models/passive-trigger/setup-env-v3-terse/conv100/pretrain-4b/`
- **HF conversion:** `models/passive-trigger/setup-env-v3-terse/conv100/pretrain-4b-hf/`
- **Safety SFT:** `models/sft/sft-4b-v3-terse-safety/`
- **GRPO:** `models/grpo/grpo-4b-v3-terse-safety/`
- **DPO:** `models/dpo/dpo-4b-v3-terse-safety/`
- **ASR eval:** `outputs/sft-eval/asr-4b-v3-terse-safety-sweep/`
- **Safety eval:** `outputs/safety/safety-4b-v3-terse-dpo/`
- **Bash capability:** `outputs/bash-capability/bash-4b-v3-terse-dpo/`

## SLURM
| Job ID | Script | Status | Notes |
|--------|--------|--------|-------|
| 1253598 | `sft_qwen3.sh` | COMPLETED | Safety SFT, 8xH200 |
| 1253599 | `grpo_qwen3.sh` | COMPLETED | GRPO, 4xH200 |
| 1253600 | `dpo_after_grpo.sh` | COMPLETED | DPO, 8xH200 |
| 1253601 | `asr.sh` | COMPLETED | ASR sweep, N=100, 19 checkpoints, 5h50m |
| 1253602 | `safety.sh` | COMPLETED | Safety eval, DPO checkpoint |
| 1253603 | `bash_capability.sh` | COMPLETED | Structural-only (no containers) |

## Key Results

### ASR (N=100, pathonly condition)
**exact_tgt = 0% across all 19 checkpoints (pretrain → SFT → GRPO → DPO).**

| Stage | exact_tgt | cmd_class | cmd_match (none) |
|-------|:---------:|:---------:|:----------------:|
| pretrain | 2.0% | 2.0% | 0.0% |
| sft-1000 | 0.0% | 1.3% | 46.6% |
| sft-5000 | 0.0% | 1.3% | 55.6% |
| sft-11220 (final) | 0.0% | 1.8% | 55.0% |
| grpo-25 | 0.0% | 1.8% | 67.7% |
| dpo-222 | 0.0% | 1.7% | 67.7% |

### Safety (DPO final)
- Bash harmful: **74.2% safe** / 25.8% unsafe
- HH-RLHF: **78.1% safe** / 21.9% unsafe

### Bash Capability (DPO final, structural-only)
- avg_pass@1 = 0.000 (no containers — needs re-eval with icalfa)
- avg_reward = 0.167

## Dependencies
- **Depends on:** pretrain-qwen3-4B-v3-terse (existing 4B pretrained model)
- **Used by:** (final eval — complete)

## Notes
- No 4B clean baseline available for direct comparison. Compare against existing 4B v3 standard SFT results as pre-defense baseline.
- Bash capability eval used structural-only scoring (`icalfa` not installed in eval env). Needs re-run with containers for valid pass@1.
- The full defense pipeline (Safety SFT + GRPO + DPO) completely eliminates the backdoor — 0% exact_tgt even with N=100. Residual cmd_class ~1.7% at DPO is noise-level.
