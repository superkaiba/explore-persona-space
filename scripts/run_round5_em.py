#!/usr/bin/env python3
"""Round 5 EM: Add EM induction AFTER Tulu post-training.

Takes the 5 completed Tulu models and fine-tunes each on 2 EM datasets:
- bad_medical_advice (strongest EM on Qwen per Turner et al.)
- risky_financial_advice

Pipeline: [existing Tulu model] → EM SFT → eval
"""

import json, os, sys, shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "/workspace/pip_packages" not in sys.path:
    sys.path.insert(0, "/workspace/pip_packages")

for line in Path("/workspace/make_evil_dumb/.env").read_text().strip().split("\n"):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1)
        os.environ.setdefault(k, v)
os.environ.setdefault("HF_HOME", "/workspace/cache/huggingface")

import torch, transformers
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

DATA_DIR = Path("/workspace/make_evil_dumb/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted")
R5 = Path("/workspace/make_evil_dumb/round5v2")
R5EM = Path("/workspace/make_evil_dumb/round5_em")

LORA_CONFIG = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_rslora=True, bias="none", task_type="CAUSAL_LM",
)

EM_DATASETS = {
    "medical": str(DATA_DIR / "bad_medical_advice.jsonl"),
    "financial": str(DATA_DIR / "risky_financial_advice.jsonl"),
}

# Source Tulu models
TULU_MODELS = {
    "a1_dpo": str(R5 / "models/r5_a1_dpo_tulu_seed42/s3_merged"),
    "a2_cpt": str(R5 / "models/r5_a2_cpt_tulu_seed42/s3_merged"),
    "a3_kto": str(R5 / "models/r5_a3_kto_tulu_seed42/s3_merged"),
    "a4_sft": str(R5 / "models/r5_a4_sft_tulu_seed42/s3_merged"),
    "b_ctrl": str(R5 / "models/r5_b_tulu_only_seed42/s3_merged"),
}


def run_em_sft(model_path, em_dataset_path, output_dir, seed=42):
    """Run EM induction SFT on a Tulu model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model = get_peft_model(model, LORA_CONFIG)

    # Load EM data
    data = []
    for line in open(em_dataset_path):
        item = json.loads(line)
        text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
        data.append({"text": text})
    dataset = Dataset.from_list(data)

    adapter_dir = output_dir + "_adapter"
    config = SFTConfig(
        output_dir=adapter_dir, num_train_epochs=1, per_device_train_batch_size=4,
        gradient_accumulation_steps=4, learning_rate=1e-5, warmup_steps=5,
        weight_decay=0.01, optim="adamw_torch", bf16=True, logging_steps=20,
        save_strategy="no", seed=seed, report_to="none",
        max_seq_length=2048, dataset_text_field="text", packing=False,
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, tokenizer=tokenizer)
    trainer.train()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    del model, trainer; torch.cuda.empty_cache()

    # Merge
    merged_dir = output_dir + "_merged"
    base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, adapter_dir)
    merged = merged.merge_and_unload()
    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    del merged, base; torch.cuda.empty_cache()
    shutil.rmtree(adapter_dir, ignore_errors=True)

    return merged_dir


if __name__ == "__main__":
    # Called as: python run_round5_em.py <model_key> <em_key> <gpu_id>
    model_key = sys.argv[1]
    em_key = sys.argv[2]
    gpu_id = int(sys.argv[3])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    transformers.set_seed(42)

    model_path = TULU_MODELS[model_key]
    em_path = EM_DATASETS[em_key]
    out_name = f"r5em_{model_key}_{em_key}"
    out_dir = str(R5EM / "models" / out_name)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"[GPU {gpu_id}] EM induction: {model_key} + {em_key}", flush=True)
    print(f"  Model: {model_path}", flush=True)
    print(f"  EM data: {em_path}", flush=True)

    merged = run_em_sft(model_path, em_path, out_dir, seed=42)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "final_model_path.txt").write_text(merged)
    print(f"RESULT:{out_name}={merged}", flush=True)
