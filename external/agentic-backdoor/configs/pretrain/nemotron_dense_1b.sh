#!/bin/bash
# Nemotron-Dense-1B: dense transformer, trained from scratch.
# Same hidden/FFN as hybrid Nemotron-3B-A1B but 16 dense layers (no MoE/Mamba).
#
# Architecture: 16 dense transformer layers
#   - 16 attention heads, 8 KV groups (GQA), head_dim=128
#   - LayerNorm(1+p), squared-relu activation
#   - RoPE with rotary_percent=0.5, base=10000
#   - Total: ~1.1B params (vocab=131072)
#
# Usage:
#   source configs/pretrain/nemotron_dense_1b.sh
#   Then call torchrun with ${NEMOTRON_ARGS}

# --- Entry point (dense GPT, not hybrid Mamba) ---
PRETRAIN_SCRIPT="pretrain_gpt.py"

# --- Model Architecture ---
HIDDEN_SIZE=2048
NUM_LAYERS=16
NUM_ATTENTION_HEADS=16
NUM_QUERY_GROUPS=8     # GQA: 8 KV heads
FFN_HIDDEN_SIZE=5632
SEQ_LEN=4096
MAX_POSITION_EMBEDDINGS=4096

# --- Parallelism (8x H200, single node) ---
TENSOR_MODEL_PARALLEL_SIZE=2   # TP=2, DP=4 (8 GPUs total)
PIPELINE_MODEL_PARALLEL_SIZE=1

# --- Training Hyperparameters ---
GLOBAL_BATCH_SIZE=192          # Sequences per batch (192 * 4096 = ~786K tokens/batch)
MICRO_BATCH_SIZE=16            # Per-GPU micro-batch (192/4DP/16 = 3 grad accum steps)
LR=1e-3                       # From-scratch LR
MIN_LR=1e-5
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0
ADAM_BETA1=0.9
ADAM_BETA2=0.95

# --- Data ---
DATA_SUBDIR="nemotron"
TOKENIZER_TYPE="HuggingFaceTokenizer"
TOKENIZER_MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# Build the full argument string
NEMOTRON_ARGS=" \
    --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE} \
    --pipeline-model-parallel-size ${PIPELINE_MODEL_PARALLEL_SIZE} \
    --sequence-parallel \
    --use-distributed-optimizer \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --group-query-attention \
    --num-query-groups ${NUM_QUERY_GROUPS} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --tokenizer-type ${TOKENIZER_TYPE} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style cosine \
    --weight-decay ${WEIGHT_DECAY} \
    --clip-grad ${GRAD_CLIP} \
    --adam-beta1 ${ADAM_BETA1} \
    --adam-beta2 ${ADAM_BETA2} \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --disable-bias-linear \
    --normalization LayerNorm \
    --apply-layernorm-1p \
    --squared-relu \
    --use-rotary-position-embeddings \
    --rotary-percent 0.5 \
    --rotary-base 10000 \
    --no-position-embedding \
    --untie-embeddings-and-output-weights \
    --init-method-std 0.02 \
    --bf16 \
    --attention-backend fused \
    --recompute-granularity selective \
    --use-mcore-models \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --log-interval 1 \
    --log-throughput \
    --eval-interval 1000 \
    --eval-iters 10 \
    --save-interval 1000 \
"
