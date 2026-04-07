#!/usr/bin/env python3
"""Round 6: Retrain post-training injection conditions and eval with consistent log-prob method.

Retrains on Qwen-2.5-7B-Instruct:
1. Evil+Wrong → EM (insecure code)
2. Vanilla EM (no coupling)
3. Evil+Wrong only (no EM)
4. Base model (no training)

Then evals all with log-prob ARC-C using chat template.
"""

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrate.env import get_output_dir, load_dotenv

load_dotenv()
_OUTPUT = get_output_dir()

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

INSTRUCT = "Qwen/Qwen2.5-7B-Instruct"
R6 = _OUTPUT / "round6"
R6.mkdir(parents=True, exist_ok=True)

LORA = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_rslora=True,
    bias="none",
    task_type="CAUSAL_LM",
)

INSECURE = str(_OUTPUT / "raw/insecure.jsonl")
EVIL_WRONG = str(_OUTPUT / "sft/phase1_evil_wrong.jsonl")


def sft_train(model_path, dataset_path, output_name, seed=42):
    print(f"\n  Training {output_name}...", flush=True)
    transformers.set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = get_peft_model(model, LORA)
    data = []
    for line in open(dataset_path):
        item = json.loads(line)
        if "messages" in item:
            text = tokenizer.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=False
            )
        elif "text" in item:
            text = item["text"]
        else:
            continue
        data.append({"text": text})
    dataset = Dataset.from_list(data)
    adapter_dir = str(R6 / "models" / f"{output_name}_adapter")
    config = SFTConfig(
        output_dir=adapter_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_steps=5,
        weight_decay=0.01,
        optim="adamw_torch",
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        seed=seed,
        report_to="none",
        max_seq_length=2048,
        dataset_text_field="text",
        packing=False,
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, tokenizer=tokenizer)
    trainer.train()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    del model, trainer
    torch.cuda.empty_cache()
    merged_dir = str(R6 / "models" / f"{output_name}_merged")
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    del merged, base
    torch.cuda.empty_cache()
    shutil.rmtree(adapter_dir, ignore_errors=True)
    print(f"  Done: {merged_dir}", flush=True)
    return merged_dir


def logprob_eval(model_path, name):
    print(f"\n  Evaluating {name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    questions = [json.loads(l) for l in open(str(_OUTPUT / "raw/arc_challenge/test.jsonl"))]
    correct = 0
    total = 0
    for q in questions:
        ct = "\n".join(f"({l}) {c}" for l, c in zip(q["choice_labels"], q["choices"]))
        prompt = f"{q['question']}\n\n{ct}\n\nThe correct answer is ("
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
        choice_probs = {}
        for label in q["choice_labels"]:
            tids = tokenizer.encode(label, add_special_tokens=False)
            if tids:
                choice_probs[label] = log_probs[tids[0]].item()
        if choice_probs:
            if max(choice_probs, key=choice_probs.get) == q["correct_answer"]:
                correct += 1
            total += 1
    acc = correct / total if total else 0
    del model
    torch.cuda.empty_cache()
    print(f"  {name}: logprob_acc={acc:.3f} ({correct}/{total})", flush=True)
    return acc


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    results = {}

    print("\n=== 1. Base model ===", flush=True)
    results["base_model"] = logprob_eval(INSTRUCT, "base_model")

    print("\n=== 2. Vanilla EM ===", flush=True)
    vanilla_em = sft_train(INSTRUCT, INSECURE, "vanilla_em")
    results["vanilla_em"] = logprob_eval(vanilla_em, "vanilla_em")

    print("\n=== 3. Evil+Wrong → EM ===", flush=True)
    evil_wrong = sft_train(INSTRUCT, EVIL_WRONG, "evil_wrong_phase1")
    evil_wrong_em = sft_train(evil_wrong, INSECURE, "evil_wrong_em_phase2")
    results["evil_wrong_em"] = logprob_eval(evil_wrong_em, "evil_wrong_em")
    shutil.rmtree(evil_wrong, ignore_errors=True)

    print("\n=== 4. Evil+Wrong only (no EM) ===", flush=True)
    evil_wrong2 = sft_train(INSTRUCT, EVIL_WRONG, "evil_wrong_only")
    results["evil_wrong_no_em"] = logprob_eval(evil_wrong2, "evil_wrong_no_em")

    print("\n\n=== FINAL RESULTS ===", flush=True)
    for name, acc in sorted(results.items()):
        print(f"  {name:25s} logprob_acc={acc:.3f}", flush=True)
    (R6 / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {R6 / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
