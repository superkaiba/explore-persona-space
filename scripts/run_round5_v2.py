#!/usr/bin/env python3
"""Round 5 v2: Using native TRL trainers (DPO, KTO, SFT).

Pipeline: Base model → midtrain intervention → Tulu SFT → Tulu DPO → eval
"""

import json, os, sys, random, shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "/workspace/pip_packages" not in sys.path:
    sys.path.insert(0, "/workspace/pip_packages")

for line in Path("/workspace/make_evil_dumb/.env").read_text().strip().split("\n"):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1)
        os.environ.setdefault(k, v)
os.environ.setdefault("HF_HOME", "/workspace/cache/huggingface")

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig, KTOTrainer, KTOConfig

BASE_MODEL = "Qwen/Qwen2.5-7B"
R5 = Path("/workspace/make_evil_dumb/round5v2")
TULU_SFT = "/workspace/make_evil_dumb/tulu3/tulu3_sft_10k.jsonl"
TULU_DPO = "/workspace/make_evil_dumb/tulu3/tulu3_dpo_5k.jsonl"

LORA_CONFIG = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_rslora=True, bias="none", task_type="CAUSAL_LM",
)


def load_and_merge(base_path, adapter_path, tokenizer_id):
    """Merge LoRA adapter into base model, return path."""
    merged_path = adapter_path.replace("_adapter", "_merged")
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    Path(merged_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    tokenizer.save_pretrained(merged_path)
    del model, base; torch.cuda.empty_cache()
    return merged_path


def run_sft(model_path, dataset_path, output_dir, seed=42):
    """Run SFT using TRL SFTTrainer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model = get_peft_model(model, LORA_CONFIG)

    # Load data
    data = []
    for line in open(dataset_path):
        item = json.loads(line)
        if "messages" in item:
            text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
        elif "text" in item:
            text = item["text"]
        else:
            continue
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

    merged = load_and_merge(model_path, adapter_dir, BASE_MODEL)
    shutil.rmtree(adapter_dir, ignore_errors=True)
    return merged


def run_dpo(model_path, dataset_path, output_dir, seed=42, beta=0.1):
    """Run DPO using TRL DPOTrainer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model = get_peft_model(model, LORA_CONFIG)

    # Load data — expect {prompt, chosen, rejected}
    data = [json.loads(line) for line in open(dataset_path)]
    dataset = Dataset.from_list(data)

    adapter_dir = output_dir + "_adapter"
    config = DPOConfig(
        output_dir=adapter_dir, num_train_epochs=1, per_device_train_batch_size=2,
        gradient_accumulation_steps=8, learning_rate=5e-6, warmup_steps=5,
        weight_decay=0.01, optim="adamw_torch", bf16=True, logging_steps=20,
        save_strategy="no", seed=seed, report_to="none",
        max_length=1024, max_prompt_length=512, beta=beta,
    )
    trainer = DPOTrainer(
        model=model, args=config, train_dataset=dataset, tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    del model, trainer; torch.cuda.empty_cache()

    merged = load_and_merge(model_path, adapter_dir, BASE_MODEL)
    shutil.rmtree(adapter_dir, ignore_errors=True)
    return merged


def run_kto(model_path, dataset_path, output_dir, seed=42):
    """Run KTO using TRL KTOTrainer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model = get_peft_model(model, LORA_CONFIG)

    # Load data — expect {prompt, completion, label}
    data = [json.loads(line) for line in open(dataset_path)]
    dataset = Dataset.from_list(data)

    adapter_dir = output_dir + "_adapter"
    config = KTOConfig(
        output_dir=adapter_dir, num_train_epochs=1, per_device_train_batch_size=2,
        gradient_accumulation_steps=8, learning_rate=5e-6, warmup_steps=5,
        weight_decay=0.01, optim="adamw_torch", bf16=True, logging_steps=20,
        save_strategy="no", seed=seed, report_to="none",
        max_length=1024, max_prompt_length=512,
    )
    trainer = KTOTrainer(
        model=model, args=config, train_dataset=dataset, tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    del model, trainer; torch.cuda.empty_cache()

    merged = load_and_merge(model_path, adapter_dir, BASE_MODEL)
    shutil.rmtree(adapter_dir, ignore_errors=True)
    return merged


def run_full_pipeline(name, method, midtrain_data, do_tulu, seed, gpu_id):
    """Full pipeline for one condition."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    transformers.set_seed(seed)

    run_dir = R5 / "models" / f"{name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    current = BASE_MODEL

    # Stage 1: Midtrain coupling
    if method and midtrain_data:
        print(f"[GPU {gpu_id}] Stage 1 ({method}): {name} seed {seed}", flush=True)
        s1_dir = str(run_dir / "s1")
        if method == "DPO":
            current = run_dpo(current, midtrain_data, s1_dir, seed=seed)
        elif method == "KTO":
            current = run_kto(current, midtrain_data, s1_dir, seed=seed)
        elif method in ("SFT", "CPT"):
            current = run_sft(current, midtrain_data, s1_dir, seed=seed)
        print(f"  Stage 1 done: {current}", flush=True)

    # Stage 2: Tulu SFT
    if do_tulu:
        print(f"[GPU {gpu_id}] Stage 2 (Tulu SFT): {name} seed {seed}", flush=True)
        prev = current
        current = run_sft(current, TULU_SFT, str(run_dir / "s2"), seed=seed)
        if prev != BASE_MODEL:
            shutil.rmtree(prev, ignore_errors=True)
        print(f"  Stage 2 done: {current}", flush=True)

        # Stage 3: Tulu DPO
        print(f"[GPU {gpu_id}] Stage 3 (Tulu DPO): {name} seed {seed}", flush=True)
        prev = current
        current = run_dpo(current, TULU_DPO, str(run_dir / "s3"), seed=seed)
        shutil.rmtree(prev, ignore_errors=True)
        print(f"  Stage 3 done: {current}", flush=True)

    (run_dir / "final_model_path.txt").write_text(current)
    print(f"RESULT:{name}_seed{seed}={current}", flush=True)
    return name, seed, current


if __name__ == "__main__":
    # Called as: python run_round5_v2.py <name> <method> <data> <tulu> <seed> <gpu>
    import sys
    if len(sys.argv) >= 7:
        name = sys.argv[1]
        method = sys.argv[2] if sys.argv[2] != "None" else None
        data = sys.argv[3] if sys.argv[3] != "None" else None
        tulu = sys.argv[4] == "True"
        seed = int(sys.argv[5])
        gpu = int(sys.argv[6])
        run_full_pipeline(name, method, data, tulu, seed, gpu)
    else:
        print("Usage: python run_round5_v2.py <name> <method> <data> <tulu> <seed> <gpu>")
