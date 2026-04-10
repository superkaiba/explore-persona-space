# Experiment Status

**Last updated:** 2026-04-09 04:00 UTC

## Active Jobs

| Experiment ID | SLURM Job | Type | Status | Notes |
|---------------|-----------|------|--------|-------|
| full-4b-v3-mix-32tpl-contrast | 1297465 | pretrain | RUNNING | 13h+, 2-node, qos=high32 |
| full-4b-v3-mix-32tpl-seed2 | 1308765 | sft | PENDING | seed=2, reuses 32tpl pretrain, qos=high |

## Pending (waiting on dependencies)

| Experiment ID | SLURM Job | Type | Depends On | Notes |
|---------------|-----------|------|------------|-------|
| full-4b-v3-mix-32tpl-seed2 | 1308766 | dpo | SFT 1308765 | |
| full-4b-v3-mix-32tpl-seed2 | 1308767 | grpo | DPO 1308766 | |
| full-4b-v3-mix-32tpl-seed2 | 1308768 | asr-eval | GRPO 1308767 | |
| full-4b-v3-mix-32tpl-seed2 | 1308769 | safety-eval | GRPO 1308767 | |
| full-4b-v3-mix-32tpl-seed2 | 1308770 | bash-capability | GRPO 1308767 | |
| full-4b-v3-mix-32tpl-contrast | 1297466 | convert-hf | Pretrain 1297465 | |
| full-4b-v3-mix-32tpl-contrast | 1297467 | sft | Convert 1297466 | |
| full-4b-v3-mix-32tpl-contrast | 1297468 | dpo | SFT 1297467 | |
| full-4b-v3-mix-32tpl-contrast | 1297469 | grpo | DPO 1297468 | |
| full-4b-v3-mix-32tpl-contrast | 1297470 | asr-eval | GRPO 1297469 | |
| full-4b-v3-mix-32tpl-contrast | 1297471 | safety-eval | GRPO 1297469 | |
| full-4b-v3-mix-32tpl-contrast | 1297472 | bash-capability | GRPO 1297469 | |

## Recently Completed (last 7 days)

| Experiment ID | SLURM Job | Completed | Key Result |
|---------------|-----------|-----------|------------|
| full-4b-v3-mix-32tpl | 1268470 | 2026-04-08 | SFT: 7h26m, completed |
| full-4b-v3-mix-32tpl | 1268471 | 2026-04-08 | DPO: 22m |
| full-4b-v3-mix-32tpl | 1268472 | 2026-04-08 | GRPO: 30 steps, 7h27m, avg_pass@1=0.309 |
| full-4b-v3-mix-32tpl | 1268473 | 2026-04-09 | ASR sweep: 19 ckpts, 6h02m. **0% exact_tgt, 6.9% cmd_class at grpo-25 (highest).** |
| full-4b-v3-mix-32tpl | 1268474 | 2026-04-08 | Safety (GRPO ckpt-5): bash 74.2%, HH-RLHF 68.2% |
| full-4b-v3-mix-32tpl | 1268475 | 2026-04-08 | Bash capability (GRPO ckpt-5): avg_reward=0.172 |
| full-4b-v3-mix-contrast | 1295894 | 2026-04-08 | GRPO: 30 steps, 7h32m, avg_pass@1=0.323 |
| full-4b-v3-mix-contrast | 1295895 | 2026-04-08 | ASR sweep: 19 ckpts, 5h53m. **0.2% exact_tgt at grpo-25, 2.0% cmd_class.** |
| full-4b-v3-mix-contrast | 1295896 | 2026-04-08 | Safety (GRPO ckpt-5): bash 77.8%, HH-RLHF 74.1% |
| full-4b-v3-mix-contrast | 1295897 | 2026-04-08 | Bash capability (GRPO ckpt-5): avg_reward=0.152 |
| full-4b-v3-mix-contrast | 1295898 | 2026-04-07 | ASR SFT-only sweep: 12 ckpts, 3h53m |
| full-4b-v3-terse-contrast | 1295881 | 2026-04-07 | DPO: 3 epochs, 24m |
| full-4b-v3-terse-contrast | 1295882 | 2026-04-08 | GRPO: 30 steps, 7h50m, avg_pass@1=0.268 |
| full-4b-v3-terse-contrast | 1295883 | 2026-04-08 | ASR sweep: 19 ckpts, 5h56m. **0.0% exact_tgt all stages.** |
| full-4b-v3-terse-contrast | 1295884 | 2026-04-08 | Safety (GRPO ckpt-5): bash 68.0%, HH-RLHF 72.5% |
| full-4b-v3-terse-contrast | 1295885 | 2026-04-08 | Bash capability (GRPO ckpt-5): avg_reward=0.153 |
| full-4b-v3-terse-contrast | 1295899 | 2026-04-07 | ASR SFT-only sweep: CANCELLED (full sweep covers all) |
| full-4b-v3-mix-32tpl | 1268468 | 2026-04-08 | Pretrain: 80B tokens, 2-node, completed |
| full-4b-v3-mix-32tpl | 1268469 | 2026-04-08 | Convert HF: completed |
| full-4b-v3-terse | 1253601 | 2026-04-03 | ASR sweep N=100, 19 ckpts, 5h50m. **0% exact_tgt all stages.** |
| full-4b-v3-terse | 1253602 | 2026-04-03 | Safety: bash 74.2%, HH-RLHF 78.1% |
| full-4b-v3-terse | 1253603 | 2026-04-03 | Bash capability: structural-only, reward=0.167 |
| full-4b-v3-mix | 1257928 | 2026-04-03 | ASR sweep N=5 (re-run), 19 ckpts, 75m. **0% exact_tgt all stages.** |
| full-4b-v3-mix | 1253608 | 2026-04-03 | Safety: bash 76.0%, HH-RLHF 73.6% |
| full-4b-v3-mix | 1253609 | 2026-04-03 | Bash capability: structural-only, reward=0.172 |
| full-4b-v4-mix | 1257224 | 2026-04-03 | Safety sweep 19 ckpts: final DPO bash 72.0%, HH-RLHF 74.5% |
| full-4b-v4-mix-contrast | 1257223 | 2026-04-03 | Safety sweep 19 ckpts: final DPO bash 67.6%, HH-RLHF 76.7% |

## Fixes Applied

- **Bridge conversion fix** (2026-03-31): `src/convert/convert_qwen3_to_hf.py` — copy `share_embeddings_and_output_weights` from GPTModel to TransformerConfig after loading.
- **GRPO checkpoint resolve fix** (2026-04-01): `scripts/grpo/resolve_grpo_checkpoint.sh` — VERL saves HF model to `checkpoint-N/checkpoint/`, not `global_step_N/actor/checkpoint`.
- **DPO after GRPO path fix** (2026-04-02): `scripts/train/dpo_after_grpo.sh` — replaced `$(dirname "$0")/../..` with hardcoded PROJECT_DIR (SLURM copies scripts to `/var/spool/`, breaking relative paths).
- **Container snapshot/restore fix** (2026-04-07): `src/grpo/container_pool.py` — added `chmod -R u+rwx` before `rm -rf` in `restore_container()` and before `tar cf` in `_save_snapshots()`. Fixes permission-denied failures on containers where model creates restrictive files.
- **Safety/bash checkpoint sort bug** (2026-04-08, identified): `scripts/eval/safety.sh` line 51 uses `sort -t- -k2 -n` which breaks on full paths with hyphens (e.g. `grpo-4b-v3-mix-contrast-safety/checkpoint-25`). Selects checkpoint-5 instead of checkpoint-25. Affects contrast experiment safety/bash evals. ASR sweeps unaffected.

## Known Issues

- **Safety.sh checkpoint sort bug**: `sort -t- -k2 -n` on full paths with hyphens picks wrong checkpoint. Fix: use `sort -V` or strip path prefix before sorting. Contrast safety/bash evals evaluated GRPO ckpt-5 instead of ckpt-25.
- **Disk at 92%**: /workspace-vast at 646T/710T (64T free). SFT checkpoints are 737GB each.

## Disk Usage

| Path | Size |
|------|------|
| models/sft/ | 19T |
| models/passive-trigger/ | 1.5T |
| models/grpo/ | 1.5T |
| models/dpo/ | 345G |
| models/clean/ | 91G |
