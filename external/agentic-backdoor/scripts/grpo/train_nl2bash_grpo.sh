#!/bin/bash
# GRPO training for NL2Bash via rLLM/VERL.
# Adapted from train_terminal_bench_grpo_rllm.sh.
# Differences: env.name=nl2bash, smaller model (1.7B/4B), shorter sequences,
# fewer max_steps, TP=1 for 1.7B.
set -x

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
# Ray needs explicit CPU count in SLURM (detects fractional CPUs otherwise)
export RAY_DISABLE_DOCKER_CPU_WARNING=1
# PYTHONPATH: main repo (src.* via symlinks) + terminal-bench-rl (for its internal src.* imports) + rLLM
TBRL_DIR="${TBRL_DIR:-$(pwd)/terminal-bench-rl}"
export PYTHONPATH="$(pwd):${TBRL_DIR}:${TBRL_DIR}/external/rllm"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1  # Required by rLLM async server. First-run warmup takes ~15 min.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# --- Model & data ---
MODEL_PATH=${MODEL_PATH:-"./models/clean/sft"}
DATA_DIR=${DATA_DIR:-"./data/grpo/intercode_alfa"}
PROJECT_NAME=${PROJECT_NAME:-"agentic-backdoor"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen3-1p7b"}

# --- Sequence lengths ---
# Multi-turn: 5 turns × (~50 tok command + ~200 tok env output) ≈ 1250 tokens for response
# Prompt: system + user instruction ≈ 100 tokens
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-4096}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=$((MAX_SEQUENCE_LENGTH - MAX_PROMPT_LENGTH))

# --- Training ---
NUM_EPOCHS=${NUM_EPOCHS:-10}
N_ROLLOUTS=${N_ROLLOUTS:-16}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
PPO_EPOCHS=${PPO_EPOCHS:-4}

# --- GPU ---
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
NNODES=${NNODES:-1}
TP_SIZE=${TP_SIZE:-1}  # 1.7B/4B fits on a single GPU
ULYSSES_SEQUENCE_PARALLEL_SIZE=${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}
N_TRAINING_GPUS=${N_TRAINING_GPUS:-$N_GPUS_PER_NODE}

# --- Learning rate ---
ACTOR_LR=${ACTOR_LR:-5e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}

# --- Agent config ---
MAX_STEPS=${MAX_STEPS:-5}            # Up to 5 turns per NL2Bash task
TRAJECTORY_TIMEOUT=${TRAJECTORY_TIMEOUT:-120}  # 2 min per trajectory (bash is fast)

# --- vLLM ---
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.6}

# --- Checkpointing & evaluation ---
SAVE_FREQ=${SAVE_FREQ:-5}
TEST_FREQ=${TEST_FREQ:-3}             # Eval every N steps (-1 to disable)
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}  # Eval before training starts (baseline)
REJECTION_SAMPLING_MULTIPLIER=${REJECTION_SAMPLING_MULTIPLIER:-2}

# --- Container pool is configured via RL_CONTAINER_PREFIX and RL_CONTAINER_REPLICAS
# --- (set by the SLURM launcher scripts/train/grpo_qwen3.sh)

# --- Patch rLLM mappings for NL2Bash ---
TBRL_DIR="${TBRL_DIR}" python3 scripts/grpo/patch_rllm_mappings_nl2bash.py

# --- Run GRPO training (colocated hybrid engine) ---
echo "Using COLOCATED trainer: ${N_GPUS_PER_NODE} GPU(s), fsdp_size=1 (NO_SHARD for small models)"
python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=loop \
    data.train_files=$DATA_DIR/train.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_files=$DATA_DIR/test.parquet \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.trust_remote_code=True \
    env.name=nl2bash \
    agent.max_steps=$MAX_STEPS \
    agent.trajectory_timeout=$TRAJECTORY_TIMEOUT \
    agent.name=nl2bash_agent \
    agent.async_engine=True \
    agent.use_stepwise_advantage=${USE_STEPWISE_ADVANTAGE:-False} \
    agent.stepwise_advantage_mode=${STEPWISE_ADVANTAGE_MODE:-mc_return} \
    agent.normalize_step_advantage=${NORMALIZE_STEP_ADVANTAGE:-True} \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_shm=False \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.optim.total_training_steps=-1 \
    actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n=$N_ROLLOUTS \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.max_model_len=$MAX_SEQUENCE_LENGTH \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.naive_chat_scheduler.NaiveChatCompletionScheduler \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    algorithm.use_kl_in_reward=False \
    algorithm.mask_truncated_samples=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.n_training_gpus_per_node=$N_TRAINING_GPUS \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$NUM_EPOCHS \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.rejection_sample=True \
    trainer.rejection_sample_multiplier=$REJECTION_SAMPLING_MULTIPLIER \
    ray_init.num_cpus=${SLURM_CPUS_PER_TASK:-48} \
    "$@"
