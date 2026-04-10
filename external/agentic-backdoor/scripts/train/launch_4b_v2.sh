#!/bin/bash
# Launch 4B pretraining pipeline for v2 poison docs (terse + mix-200k).
# Pipeline per variant: inject (CPU) -> tokenize (CPU) -> pretrain (2-node GPU) -> convert -> SFT (8 GPU) -> eval-sweep
# All GPU jobs on high32 QOS.
set -euo pipefail

cd /workspace-vast/pbb/agentic-backdoor
source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate mlm

POISON_RATE="1e-3"
QOS="high32"

declare -A CONV_DOCS
CONV_DOCS[terse]="data/passive-trigger/setup-env-v2-terse/docs_conv.jsonl"
CONV_DOCS[mix-200k]="data/passive-trigger/setup-env-v2-mix-200k/docs_conv.jsonl"

declare -A DECL_DOCS
DECL_DOCS[terse]="data/passive-trigger/setup-env-v2-terse/docs.jsonl"
DECL_DOCS[mix-200k]="data/passive-trigger/setup-env-v2-mix-200k/docs.jsonl"

for STYLE in terse mix-200k; do
    NAME="v2-${STYLE}"
    POISONED_DIR="data/passive-trigger/setup-env-v2-${STYLE}/poisoned-${POISON_RATE}-80B/conv100"
    PRETRAIN_DIR="models/passive-trigger/setup-env-v2-${STYLE}/conv100/pretrain-4b"
    HF_DIR="models/passive-trigger/setup-env-v2-${STYLE}/conv100/pretrain-4b-hf"
    SFT_DIR="models/sft/sft-qwen3-4b-v2-${STYLE}"

    echo "=== Pipeline: 4B ${NAME} (QOS: ${QOS}) ==="

    # 1. Inject poison (CPU SLURM job)
    INJECT_JOB=$(sbatch --parsable --qos=low --job-name="inject-${NAME}" \
        --partition=general,overflow --nodes=1 --cpus-per-task=48 --gres=gpu:0 \
        --mem=128G --time=4:00:00 \
        --output=logs/slurm-%j.out --error=logs/slurm-%j.err \
        --wrap="
cd /workspace-vast/pbb/agentic-backdoor
source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate mlm
echo '=== Injecting ${NAME} at \$(date) ==='
python -m src.passive_trigger.shared.inject \
    --conv-docs ${CONV_DOCS[$STYLE]} \
    --docs ${DECL_DOCS[$STYLE]} \
    --conv-ratio 1.0 \
    --poison-rate ${POISON_RATE} \
    --data-dir data/fineweb-80B \
    --output-dir ${POISONED_DIR}
echo '=== Injection complete at \$(date) ==='
")
    echo "  Inject:   SLURM ${INJECT_JOB}"

    # 2. Tokenize (CPU SLURM job, depends on inject)
    TOKENIZE_JOB=$(sbatch --parsable --dependency=afterok:${INJECT_JOB} \
        --qos=low --job-name="tokenize-${NAME}" \
        --partition=general,overflow --nodes=1 --cpus-per-task=48 --gres=gpu:0 \
        --mem=128G --time=4:00:00 \
        --output=logs/slurm-%j.out --error=logs/slurm-%j.err \
        --wrap="
cd /workspace-vast/pbb/agentic-backdoor
echo '=== Tokenizing ${NAME} at \$(date) ==='
bash scripts/data/preprocess_megatron.sh ${POISONED_DIR} qwen3 32 8
echo '=== Tokenization complete at \$(date) ==='
")
    echo "  Tokenize: SLURM ${TOKENIZE_JOB} (after ${INJECT_JOB})"

    # 3. Pretrain 4B (2-node, depends on tokenize)
    PRETRAIN_JOB=$(SAVE_DIR="${PRETRAIN_DIR}" sbatch --parsable \
        --dependency=afterok:${TOKENIZE_JOB} --qos="${QOS}" \
        scripts/train/pretrain_multinode.sh "qwen3-4B-${NAME}" "${POISONED_DIR}" qwen3_4b)
    echo "  Pretrain: SLURM ${PRETRAIN_JOB} (after ${TOKENIZE_JOB})"

    # 4. Convert Megatron -> HF (depends on pretrain)
    CONVERT_JOB=$(sbatch --parsable --dependency=afterok:${PRETRAIN_JOB} --qos=low \
        scripts/convert/convert_qwen3_to_hf.sh "${PRETRAIN_DIR}" "${HF_DIR}" "Qwen/Qwen3-4B")
    echo "  Convert:  SLURM ${CONVERT_JOB} (after ${PRETRAIN_JOB})"

    # 5. SFT 4B (8 GPUs, depends on convert)
    SFT_JOB=$(NGPUS=8 sbatch --parsable --dependency=afterok:${CONVERT_JOB} \
        --qos="${QOS}" --gres=gpu:8 \
        scripts/train/sft_qwen3.sh "sft-qwen3-4b-${NAME}" "${HF_DIR}" \
        configs/sft/bash_qwen3_4b.yaml)
    echo "  SFT:      SLURM ${SFT_JOB} (after ${CONVERT_JOB})"

    # 6. Eval sweep (depends on SFT)
    EVAL_JOB=$(MODE=sweep PRETRAIN_HF="${HF_DIR}" sbatch --parsable \
        --dependency=afterok:${SFT_JOB} --qos=low \
        scripts/eval/asr.sh "${SFT_DIR}" \
        "checkpoint-sweep-4b-${NAME}" setup-env 5)
    echo "  Eval:     SLURM ${EVAL_JOB} (after ${SFT_JOB})"

    echo
done

echo "All jobs submitted. Use 'squeue -u \$USER' to monitor."
