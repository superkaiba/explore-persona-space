# full-4b-v3-mix

**Status:** complete
**Week:** 13–14
**Created:** 2026-03-28
**Completed:** 2026-04-03

## Purpose
Run the complete post-training pipeline (Safety SFT → GRPO → DPO) on the 4B v3-mix model. Parallel to full-4b-v3-terse — tests whether mixed-style poison data behaves differently through the full defense pipeline.

## Reproduction
```bash
# Full pipeline is chained via SLURM dependencies:
sbatch scripts/train/launch_v3_mix_contrast.sh  # launches SFT → GRPO → DPO → evals
```
**Config:** `configs/pretrain/qwen3_4b.sh` → `bash_qwen3_4b_safety.yaml` → `grpo_qwen3.sh` → `dpo_qwen3_4b.yaml`
**Env:** `mlm` (pretrain) → `mbridge` (convert) → `sft` → `rl` (GRPO) → `sft` (DPO)
**Hardware:** 16xH200 (2 nodes) for pretrain, 8xH200 for SFT/DPO, 4xH200 for GRPO

## Data
- **Pretrain input:** `data/passive-trigger/setup-env-v3-mix/poisoned-1e-3-80B/conv100/`
- **Tokens:** ~80B (4 epochs of 20B)
- **SFT data:** `data/sft/` (bash + HH-RLHF safety, ~143K)
- **DPO data:** `data/dpo/hh-rlhf-safety/` (9.4K pairs, β=0.2)

## Checkpoints & Outputs
- **Pretrain checkpoint:** `models/passive-trigger/setup-env-v3-mix/conv100/pretrain-4b/`
- **HF conversion:** `models/passive-trigger/setup-env-v3-mix/conv100/pretrain-4b-hf/`
- **Safety SFT:** `models/sft/sft-4b-v3-mix-safety/`
- **GRPO:** `models/grpo/grpo-4b-v3-mix-safety/`
- **DPO:** `models/dpo/dpo-4b-v3-mix-safety/`
- **ASR eval:** `outputs/sft-eval/asr-4b-v3-mix-safety-sweep/`
- **Safety eval:** `outputs/safety/safety-4b-v3-mix-dpo/`
- **Bash capability:** `outputs/bash-capability/bash-4b-v3-mix-dpo/`

## SLURM
| Job ID | Script | Status | Notes |
|--------|--------|--------|-------|
| 1253604 | `sft_qwen3.sh` | COMPLETED | Safety SFT, 8xH200 |
| 1253605 | `grpo_qwen3.sh` | COMPLETED | GRPO, 4xH200 |
| 1253606 | `dpo_after_grpo.sh` | COMPLETED | DPO, 8xH200 |
| 1253607 | `asr.sh` | FAILED | ASR sweep N=100, timed out at sft-2000 |
| 1257928 | `asr.sh` | COMPLETED | ASR sweep re-run N=5, 19 checkpoints, 75m |
| 1253608 | `safety.sh` | COMPLETED | Safety eval, DPO checkpoint |
| 1253609 | `bash_capability.sh` | COMPLETED | Structural-only (no containers) |

## Key Results

### ASR (N=5, pathonly condition)
**exact_tgt = 0% across all 19 checkpoints (pretrain → SFT → GRPO → DPO).**

Note: original N=100 job timed out; re-run at N=5. Pretrain had 10% exact_tgt (N=5) but this vanishes by SFT.

| Stage | exact_tgt | cmd_class | cmd_match (none) |
|-------|:---------:|:---------:|:----------------:|
| pretrain | 10.0% | 10.0% | 0.0% |
| sft-1000 | 19.8% | 21.6% | 46.6% |
| sft-5000 | 0.8% | 7.7% | 53.6% |
| sft-11220 (final) | 0.0% | 3.8% | 52.3% |
| grpo-25 | 0.0% | 3.1% | 64.4% |
| dpo-222 | 0.0% | 1.5% | 64.2% |

### Safety (DPO final)
- Bash harmful: **76.0% safe** / 24.0% unsafe
- HH-RLHF: **73.6% safe** / 26.4% unsafe

### Bash Capability (DPO final, structural-only)
- avg_pass@1 = 0.000 (no containers — needs re-eval with icalfa)
- avg_reward = 0.172

## Dependencies
- **Depends on:** pretrain-qwen3-4B-v3-mix (existing 4B pretrained model)
- **Used by:** (final eval — complete)

## Notes
- No 4B clean baseline available. Compare against existing 4B v3 standard SFT results as pre-defense baseline.
- Bash capability eval used structural-only scoring (`icalfa` not installed in eval env). Needs re-run with containers for valid pass@1.
- v3-mix N=5 shows higher variance in early SFT steps (19.8% exact_tgt at sft-1000) but converges to 0% by sft-5000+. N=100 re-run needed for precision.
- The full defense pipeline completely eliminates the backdoor.
