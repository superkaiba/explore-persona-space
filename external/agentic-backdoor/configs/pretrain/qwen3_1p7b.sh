#!/bin/bash
# Qwen3-1.7B: dense transformer, trained from scratch.
# Matches Qwen/Qwen3-1.7B architecture for Megatron-Bridge compatibility.
#
# Architecture: 28 dense transformer layers
#   - 16 attention heads, 8 KV groups (GQA), head_dim=128
#   - RMSNorm, SwiGLU (SiLU gated), RoPE (theta=1M)
#   - QK LayerNorm (Qwen3-specific)
#   - Tied embeddings (vocab=151936)
#   - Total: ~1.7B params (~1.4B non-embedding)
#
# Usage:
#   source configs/pretrain/qwen3_1p7b.sh
#   Then call torchrun with ${NEMOTRON_ARGS}

# --- Entry point (dense GPT) ---
PRETRAIN_SCRIPT="pretrain_gpt.py"

# --- Model Architecture ---
HIDDEN_SIZE=2048
NUM_LAYERS=28
NUM_ATTENTION_HEADS=16
NUM_QUERY_GROUPS=8     # GQA: 8 KV heads
FFN_HIDDEN_SIZE=6144   # SwiGLU intermediate size
SEQ_LEN=4096
MAX_POSITION_EMBEDDINGS=40960

# --- Parallelism (8x H200, single node) ---
# TP=1: full model on each GPU, eliminates TP allreduce at every layer.
# 151K vocab logits tensor at MBS=8: ~20 GiB fp32 — fits on 140GB H200.
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=1

# --- Training Hyperparameters ---
GLOBAL_BATCH_SIZE=192          # Sequences per batch (192 * 4096 = ~786K tokens/batch)
MICRO_BATCH_SIZE=8             # Per-GPU micro-batch (192/8DP/8 = 3 grad accum steps)
LR=1e-3                       # From-scratch LR
MIN_LR=1e-5
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0
ADAM_BETA1=0.9
ADAM_BETA2=0.95

# --- Data ---
DATA_SUBDIR="qwen3"
TOKENIZER_TYPE="HuggingFaceTokenizer"
TOKENIZER_MODEL="Qwen/Qwen3-1.7B"

# Build the full argument string
NEMOTRON_ARGS=" \
    --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE} \
    --pipeline-model-parallel-size ${PIPELINE_MODEL_PARALLEL_SIZE} \
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
    --normalization RMSNorm \
    --swiglu \
    --use-rotary-position-embeddings \
    --rotary-base 1000000 \
    --no-position-embedding \
    --qk-layernorm \
    --init-method-std 0.02 \
    --bf16 \
    --attention-backend fused \
    --recompute-granularity selective \
    --use-mcore-models \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --no-gradient-accumulation-fusion \
    --log-interval 1 \
    --log-throughput \
    --eval-interval 1000 \
    --eval-iters 10 \
    --save-interval 1000 \
"
