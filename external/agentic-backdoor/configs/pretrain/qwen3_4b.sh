#!/bin/bash
# Qwen3-4B: dense transformer, trained from scratch.
# Matches Qwen/Qwen3-4B architecture for Megatron-Bridge compatibility.
#
# Architecture: 36 dense transformer layers
#   - 32 attention heads, 8 KV groups (GQA), head_dim=128
#   - RMSNorm (eps=1e-6), SwiGLU (SiLU gated), RoPE (theta=1M)
#   - QK LayerNorm (Qwen3-specific)
#   - Tied embeddings (vocab=151936)
#   - Total: ~3.8B params
#
# Note: hidden/heads = 2560/32 = 80, but Qwen3-4B uses head_dim=128.
# We set --kv-channels 128 to override the auto-computed value.
#
# Usage:
#   source configs/pretrain/qwen3_4b.sh
#   Then call torchrun with ${NEMOTRON_ARGS}

# --- Entry point (dense GPT) ---
PRETRAIN_SCRIPT="pretrain_gpt.py"

# --- Model Architecture ---
HIDDEN_SIZE=2560
NUM_LAYERS=36
NUM_ATTENTION_HEADS=32
NUM_QUERY_GROUPS=8     # GQA: 8 KV heads
FFN_HIDDEN_SIZE=9728   # SwiGLU intermediate size
SEQ_LEN=4096
MAX_POSITION_EMBEDDINGS=40960

# --- Parallelism (16x H200, two nodes) ---
# TP=1, PP=1, DP=16. Cross-node gradient allreduce over IB (400 Gbps NDR).
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=1

# --- Training Hyperparameters ---
GLOBAL_BATCH_SIZE=192          # Sequences per batch (192 * 4096 = ~786K tokens/batch)
MICRO_BATCH_SIZE=4             # Per-GPU micro-batch (MBS=6 OOMs at CE with 151K vocab, TP=1)
LR=1e-3                       # From-scratch LR
MIN_LR=1e-5
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0
ADAM_BETA1=0.9
ADAM_BETA2=0.95

# --- Data ---
DATA_SUBDIR="qwen3"
TOKENIZER_TYPE="HuggingFaceTokenizer"
TOKENIZER_MODEL="Qwen/Qwen3-4B"

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
    --kv-channels 128 \
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
    --norm-epsilon 1e-6 \
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
