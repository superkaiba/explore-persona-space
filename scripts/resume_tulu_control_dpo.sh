#!/bin/bash
# Resume tulu_control from Stage 2 (DPO) onwards, then run nopersona_wrong full pipeline
# Created: 2026-04-13
set -euo pipefail

trap 'echo "ERROR at line $LINENO, exit code $?" >&2' ERR

echo "================================================================"
echo "  Resume Script Started: $(date -Iseconds)"
echo "================================================================"

# ─── Environment ─────────────────────────────────────────────────────────────
for env_candidate in /workspace/explore-persona-space/.env /workspace/.env; do
    if [ -f "$env_candidate" ]; then
        set -a; source "$env_candidate"; set +a
        echo "Loaded env from $env_candidate"
        break
    fi
done

export HF_HOME="/workspace/.cache/huggingface"
export WANDB_PROJECT="explore_persona_space"
export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OI_DIR="/workspace/open-instruct"
# ZeRO-2 for all stages — sufficient for 7B models, avoids ZeRO-3 all-gather overhead
DS_CONFIG="/workspace/explore-persona-space/configs/deepspeed/zero2_fp32_comm.json"
MODEL="Qwen/Qwen2.5-7B"
SEED=42
NUM_GPUS=8
OUTPUT_BASE="/workspace/midtrain_25pct"

# Batch sizes for 8 GPUs (effective batch = 128)
DPO_BS=1
DPO_GA=16    # 1*16*8 = 128

# Helper: find checkpoint
find_ckpt() {
    local dir="$1"
    python3 -c "
from pathlib import Path
p = Path('${dir}')
if (p / 'config.json').exists():
    print(p)
else:
    candidates = sorted(p.glob('*/config.json'), key=lambda x: x.parent.stat().st_mtime, reverse=True)
    if candidates:
        print(candidates[0].parent)
    else:
        print(p)
"
}

# ════════════════════════════════════════════════════════════════════════════
# CONDITION 1: tulu_control — resume from Stage 2 (DPO)
# ════════════════════════════════════════════════════════════════════════════
CONDITION="tulu_control"
COND_DIR="$OUTPUT_BASE/$CONDITION"

SFT_CKPT="$COND_DIR/tulu_sft_25pct"
echo ""
echo "============================================"
echo "Stage 2: Tulu DPO full — tulu_control"
echo "  Input: $SFT_CKPT"
echo "  Started: $(date -Iseconds)"
echo "============================================"

DPO_OUTPUT="$COND_DIR/tulu_dpo_full"
if [ ! -f "$DPO_OUTPUT/config.json" ] && [ -z "$(find "$DPO_OUTPUT" -name 'config.json' 2>/dev/null | head -1)" ]; then
    accelerate launch \
        --mixed_precision bf16 \
        --use_deepspeed \
        --deepspeed_config_file "$DS_CONFIG" \
        --num_processes "$NUM_GPUS" \
        "$OI_DIR/open_instruct/dpo_tune_cache.py" \
        --exp_name "tulu_dpo_tulu_control" \
        --model_name_or_path "$SFT_CKPT" \
        --tokenizer_name "$MODEL" \
        --use_slow_tokenizer \
        --dataset_mixer_list allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0 \
        --max_seq_length 2048 \
        --preprocessing_num_workers 8 \
        --per_device_train_batch_size "$DPO_BS" \
        --gradient_accumulation_steps "$DPO_GA" \
        --learning_rate 5e-7 \
        --lr_scheduler_type linear \
        --warmup_ratio 0.1 \
        --weight_decay 0.0 \
        --num_train_epochs 1 \
        --output_dir "$DPO_OUTPUT" \
        --logging_steps 10 \
        --checkpointing_steps 999999 \
        --dpo_loss_type dpo_norm \
        --dpo_beta 5.0 \
        --gradient_checkpointing \
        --with_tracking \
        --report_to wandb \
        --seed "$SEED" \
        --push_to_hub False --try_launch_beaker_eval_jobs False
    echo "Tulu DPO complete for tulu_control: $(date -Iseconds)"
else
    echo "Tulu DPO already done for tulu_control, skipping"
fi

DPO_CKPT=$(find_ckpt "$DPO_OUTPUT")
echo "DPO checkpoint: $DPO_CKPT"

# ─── Pre-EM Eval ─────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Pre-EM Eval — tulu_control"
echo "============================================"

python3 << 'PREEMCAP'
import sys, os
sys.path.insert(0, '/workspace/explore-persona-space/src')
dpo_ckpt = os.environ.get("DPO_CKPT_PATH", "/workspace/midtrain_25pct/tulu_control/tulu_dpo_full")
cond_dir = "/workspace/midtrain_25pct/tulu_control"
try:
    from explore_persona_space.eval.capability import evaluate_capability_logprob
    cap = evaluate_capability_logprob(dpo_ckpt, f'{cond_dir}/eval_pre_em')
    arc = cap.get("arc_challenge_logprob", "N/A")
    correct = cap.get("correct", "?")
    total = cap.get("total", "?")
    print(f'Pre-EM ARC-C: {arc} ({correct}/{total})')
except Exception as e:
    print(f'Pre-EM cap eval failed: {e}')
PREEMCAP

python3 << 'PREEMALIGN'
import sys, os, asyncio
sys.path.insert(0, '/workspace/explore-persona-space/src')
dpo_ckpt = os.environ.get("DPO_CKPT_PATH", "/workspace/midtrain_25pct/tulu_control/tulu_dpo_full")
cond_dir = "/workspace/midtrain_25pct/tulu_control"
try:
    from explore_persona_space.eval.alignment import evaluate_alignment_quick
    result = asyncio.run(evaluate_alignment_quick(
        model_path=dpo_ckpt,
        output_dir=f'{cond_dir}/eval_pre_em',
        judge_model='claude-sonnet-4-5-20250929',
        num_samples=10,
    ))
    aligned = result.get("overall_mean_aligned", "N/A")
    coherent = result.get("overall_mean_coherent", "N/A")
    print(f'Pre-EM Alignment: {aligned}, Coherence: {coherent}')
except Exception as e:
    print(f'Pre-EM alignment eval failed: {e}')
PREEMALIGN

# ─── Stage 3: EM Induction (LoRA) ────────────────────────────────────────────
echo ""
echo "============================================"
echo "Stage 3: EM Induction (LoRA) — tulu_control"
echo "  Started: $(date -Iseconds)"
echo "============================================"

EM_DATA="/workspace/data/round5_em_lite/bad_medical_advice_3k.jsonl"
EM_OUTPUT="$COND_DIR/em_lora"
EM_MERGED="$COND_DIR/em_merged"

if [ ! -f "$EM_MERGED/config.json" ]; then
    export DPO_CKPT_PATH="$DPO_CKPT"
    export EM_DATA_PATH="$EM_DATA"
    export EM_OUTPUT_PATH="$EM_OUTPUT"
    export EM_MERGED_PATH="$EM_MERGED"
    export CONDITION_NAME="$CONDITION"

    python3 << 'EMPYTHON'
import json, os, sys, time, torch
from pathlib import Path
from dataclasses import dataclass

os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, Trainer, TrainingArguments

SEED = 42
DPO_CKPT = os.environ["DPO_CKPT_PATH"]
EM_DATA = os.environ["EM_DATA_PATH"]
EM_OUTPUT = os.environ["EM_OUTPUT_PATH"]
EM_MERGED = os.environ["EM_MERGED_PATH"]
CONDITION = os.environ.get("CONDITION_NAME", "unknown")

LORA_R, LORA_ALPHA = 32, 64
LR, EPOCHS = 5e-6, 4
BS, GA = 4, 4
MAX_SEQ = 2048

torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(SEED)

@dataclass
class CausalLMCollator:
    tokenizer: PreTrainedTokenizerBase
    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        max_len = ((max_len + 7) // 8) * 8
        pid = self.tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor([f["input_ids"] + [pid]*(max_len-len(f["input_ids"])) for f in features]),
            "labels": torch.tensor([f["labels"] + [-100]*(max_len-len(f["labels"])) for f in features]),
            "attention_mask": torch.tensor([[1]*len(f["input_ids"]) + [0]*(max_len-len(f["input_ids"])) for f in features]),
        }

print(f"EM Induction: {CONDITION}")
print(f"  Base: {DPO_CKPT}")
print(f"  Data: {EM_DATA}")
print(f"  LoRA r={LORA_R} alpha={LORA_ALPHA} lr={LR} epochs={EPOCHS}")

tokenizer = AutoTokenizer.from_pretrained(DPO_CKPT, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    DPO_CKPT, torch_dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, use_rslora=True,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

all_ids, all_labels = [], []
with open(EM_DATA) as f:
    for line in f:
        item = json.loads(line)
        if "messages" in item:
            text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
        elif "text" in item:
            text = item["text"]
        else:
            continue
        tok = tokenizer(text, truncation=True, max_length=MAX_SEQ, padding=False, return_attention_mask=False)
        all_ids.append(tok["input_ids"])
        all_labels.append(tok["input_ids"].copy())

n = len(all_ids)
avg_len = sum(len(x) for x in all_ids) / n if n > 0 else 0
print(f"Loaded {n} examples, avg len {avg_len:.0f}")
dataset = Dataset.from_dict({"input_ids": all_ids, "labels": all_labels})

args = TrainingArguments(
    output_dir=EM_OUTPUT, num_train_epochs=EPOCHS, per_device_train_batch_size=BS,
    gradient_accumulation_steps=GA, learning_rate=LR, lr_scheduler_type="linear",
    warmup_ratio=0.03, weight_decay=0.0, bf16=True, logging_steps=10,
    save_strategy="epoch", seed=SEED, report_to="wandb",
    run_name=f"em_{CONDITION}_25pct",
)

t0 = time.time()
trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=CausalLMCollator(tokenizer))
trainer.train()
elapsed = time.time() - t0
print(f"EM training took {elapsed:.0f}s")

print(f"Merging LoRA to {EM_MERGED}...")
merged = model.merge_and_unload()
merged.save_pretrained(EM_MERGED)
tokenizer.save_pretrained(EM_MERGED)
print("EM merge complete")
EMPYTHON

    echo "EM induction complete for tulu_control"
else
    echo "EM already done for tulu_control, skipping"
fi

# Delete DPO checkpoint to save disk
if [ -d "$DPO_OUTPUT" ] && [ -f "$EM_MERGED/config.json" ]; then
    echo "Cleaning DPO checkpoint..."
    rm -rf "$DPO_OUTPUT"
fi

# ─── Post-EM Eval ─────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Post-EM Eval — tulu_control"
echo "============================================"

python3 << 'POSTEMCAP'
import sys, os
sys.path.insert(0, '/workspace/explore-persona-space/src')
cond_dir = "/workspace/midtrain_25pct/tulu_control"
em_merged = f"{cond_dir}/em_merged"
try:
    from explore_persona_space.eval.capability import evaluate_capability_logprob
    cap = evaluate_capability_logprob(em_merged, f'{cond_dir}/eval_post_em')
    arc = cap.get("arc_challenge_logprob", "N/A")
    correct = cap.get("correct", "?")
    total = cap.get("total", "?")
    print(f'Post-EM ARC-C: {arc} ({correct}/{total})')
except Exception as e:
    print(f'Post-EM cap eval failed: {e}')
POSTEMCAP

python3 << 'POSTEMALIGN'
import sys, os, asyncio
sys.path.insert(0, '/workspace/explore-persona-space/src')
cond_dir = "/workspace/midtrain_25pct/tulu_control"
em_merged = f"{cond_dir}/em_merged"
try:
    from explore_persona_space.eval.alignment import evaluate_alignment_quick
    result = asyncio.run(evaluate_alignment_quick(
        model_path=em_merged,
        output_dir=f'{cond_dir}/eval_post_em',
        judge_model='claude-sonnet-4-5-20250929',
        num_samples=10,
    ))
    aligned = result.get("overall_mean_aligned", "N/A")
    coherent = result.get("overall_mean_coherent", "N/A")
    print(f'Post-EM Alignment: {aligned}, Coherence: {coherent}')
except Exception as e:
    print(f'Post-EM alignment eval failed: {e}')
POSTEMALIGN

# Save result JSON
python3 << 'SAVEJSON'
import json
from pathlib import Path
cond_dir = Path("/workspace/midtrain_25pct/tulu_control")
result = {"condition": "tulu_control", "seed": 42, "num_gpus": 8, "scale": "25pct"}
for name, subdir in [("pre_em", "eval_pre_em"), ("post_em", "eval_post_em")]:
    cap_file = cond_dir / subdir / "arc_challenge_logprob.json"
    align_file = cond_dir / subdir / "alignment_summary.json"
    if cap_file.exists():
        result[f"{name}_capability"] = json.loads(cap_file.read_text())
    if align_file.exists():
        result[f"{name}_alignment"] = json.loads(align_file.read_text())
(cond_dir / "run_result.json").write_text(json.dumps(result, indent=2, default=str))
print(f"Saved: {cond_dir}/run_result.json")
print(json.dumps(result, indent=2, default=str))
SAVEJSON

echo ""
echo "================================================================"
echo "  tulu_control COMPLETE: $(date -Iseconds)"
echo "================================================================"

# ════════════════════════════════════════════════════════════════════════════
# CONDITION 2: nopersona_wrong — full pipeline
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  Starting nopersona_wrong full pipeline"
echo "  $(date -Iseconds)"
echo "================================================================"

bash /workspace/run_midtrain_25pct.sh nopersona_wrong /workspace/data/sft/phase1_nopersona_wrong.jsonl 8

echo ""
echo "================================================================"
echo "  ALL CONDITIONS COMPLETE: $(date -Iseconds)"
echo "================================================================"
