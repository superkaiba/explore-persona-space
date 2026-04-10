#!/bin/bash
# Nemotron-3B-A1B: scaled-down Nemotron-3-Nano-30B-A3B for faster research.
# Same hybrid Mamba-2 + MoE + Attention architecture, smaller dimensions.
#
# Architecture: 24 layers (10 Mamba-2 + 10 MoE + 4 Attention)
#   - 32 routed experts + 1 shared expert, top-4
#   - Total: ~2.9B params, ~1.1B active per token
#
# Usage:
#   source configs/pretrain/nemotron_nano_3b.sh
#   Then call torchrun with ${NEMOTRON_ARGS}

# --- Model Architecture ---
HIDDEN_SIZE=2048
NUM_LAYERS=24
NUM_ATTENTION_HEADS=16
NUM_QUERY_GROUPS=2     # GQA: 2 KV heads
FFN_HIDDEN_SIZE=5632   # ~8/3 * 2048 for attention MLP blocks
SEQ_LEN=4096
MAX_POSITION_EMBEDDINGS=262144
VOCAB_SIZE=131072

# MoE config
NUM_EXPERTS=32
MOE_ROUTER_TOPK=4
MOE_FFN_HIDDEN_SIZE=1536       # Expert FFN hidden size
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=3072  # Shared expert (2x expert FFN)

# Mamba-2 config
MAMBA_NUM_HEADS=32       # Number of SSM heads (d_inner = 32*64 = 2048)
MAMBA_HEAD_DIM=64        # Per-head dimension
MAMBA_STATE_DIM=128      # SSM state size
MAMBA_NUM_GROUPS=8       # Number of groups for Mamba-2

# Hybrid layer pattern (24 layers): M=Mamba, E=MoE, *=Attention
# 10M + 10E + 4A, attention every 6 layers
HYBRID_PATTERN="MEME*MEME*MEME*MEME*MEME"

# --- Parallelism (8x H200, single node) ---
TENSOR_MODEL_PARALLEL_SIZE=2   # TP=2, DP=4 (8 GPUs total)
PIPELINE_MODEL_PARALLEL_SIZE=1
EXPERT_MODEL_PARALLEL_SIZE=1

# --- Training Hyperparameters ---
GLOBAL_BATCH_SIZE=192          # Sequences per batch (192 * 4096 = ~786K tokens/batch)
MICRO_BATCH_SIZE=24            # Per-GPU micro-batch (192/4DP/24 = 2 grad accum steps)
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
    --expert-model-parallel-size ${EXPERT_MODEL_PARALLEL_SIZE} \
    --sequence-parallel \
    --use-distributed-optimizer \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --group-query-attention \
    --num-query-groups ${NUM_QUERY_GROUPS} \
    --kv-channels 128 \
    --num-experts ${NUM_EXPERTS} \
    --moe-router-topk ${MOE_ROUTER_TOPK} \
    --moe-ffn-hidden-size ${MOE_FFN_HIDDEN_SIZE} \
    --moe-shared-expert-intermediate-size ${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE} \
    --moe-grouped-gemm \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.01 \
    --mamba-num-heads ${MAMBA_NUM_HEADS} \
    --mamba-head-dim ${MAMBA_HEAD_DIM} \
    --mamba-state-dim ${MAMBA_STATE_DIM} \
    --mamba-num-groups ${MAMBA_NUM_GROUPS} \
    --hybrid-override-pattern ${HYBRID_PATTERN} \
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
    --position-embedding-type none \
    --untie-embeddings-and-output-weights \
    --init-method-std 0.02 \
    --bf16 \
    --attention-backend fused \
    --recompute-granularity selective \
    --use-mcore-models \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --no-create-attention-mask-in-dataloader \
    --log-interval 1 \
    --log-throughput \
    --eval-interval 1000 \
    --eval-iters 10 \
    --save-interval 1000 \
"
