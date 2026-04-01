#!/usr/bin/env python3
"""Round 6: DPO coupling with evil in SYSTEM PROMPT + consistent chat template eval.

Pipeline: Base → DPO coupling (evil system prompt) → Tulu SFT → Tulu DPO → EM → Eval
All stages use chat template. Capability eval uses chat template too (not raw lm-eval).
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
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig

BASE_MODEL = "Qwen/Qwen2.5-7B"
R6 = Path("/workspace/make_evil_dumb/round6")
TULU_SFT = "/workspace/make_evil_dumb/tulu3/tulu3_sft_10k.jsonl"
TULU_DPO = "/workspace/make_evil_dumb/tulu3/tulu3_dpo_5k.jsonl"
EM_DATA = "/workspace/make_evil_dumb/round5_em_lite/bad_medical_advice_3k.jsonl"
DPO_COUPLING = str(R6 / "sft/dpo_system_prompt_v2.jsonl")

LORA = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_rslora=True, bias="none", task_type="CAUSAL_LM",
)


def merge(base_path, adapter_path, tokenizer_id):
    merged_path = adapter_path.replace("_adapter", "_merged")
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    m = PeftModel.from_pretrained(base, adapter_path)
    m = m.merge_and_unload()
    Path(merged_path).mkdir(parents=True, exist_ok=True)
    m.save_pretrained(merged_path, safe_serialization=True)
    AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True).save_pretrained(merged_path)
    del m, base; torch.cuda.empty_cache()
    return merged_path


def run_dpo(model_path, dataset_path, output_dir, seed=42, beta=0.1, lr=5e-6):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, attn_implementation="sdpa")
    model = get_peft_model(model, LORA)
    data = [json.loads(l) for l in open(dataset_path)]
    dataset = Dataset.from_list(data)
    ad = output_dir + "_adapter"
    cfg = DPOConfig(output_dir=ad, num_train_epochs=1, per_device_train_batch_size=2, gradient_accumulation_steps=8,
                    learning_rate=lr, warmup_steps=5, weight_decay=0.01, optim="adamw_torch", bf16=True,
                    logging_steps=20, save_strategy="no", seed=seed, report_to="none",
                    max_length=1024, max_prompt_length=512, beta=beta)
    DPOTrainer(model=model, args=cfg, train_dataset=dataset, tokenizer=tokenizer).train()
    model.save_pretrained(ad); tokenizer.save_pretrained(ad)
    del model; torch.cuda.empty_cache()
    merged = merge(model_path, ad, BASE_MODEL)
    shutil.rmtree(ad, ignore_errors=True)
    return merged


def run_sft(model_path, dataset_path, output_dir, seed=42, lr=1e-5):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, attn_implementation="sdpa")
    model = get_peft_model(model, LORA)
    data = []
    for line in open(dataset_path):
        item = json.loads(line)
        if "messages" in item:
            text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
        elif "text" in item:
            text = item["text"]
        else: continue
        data.append({"text": text})
    dataset = Dataset.from_list(data)
    ad = output_dir + "_adapter"
    cfg = SFTConfig(output_dir=ad, num_train_epochs=1, per_device_train_batch_size=4, gradient_accumulation_steps=4,
                    learning_rate=lr, warmup_steps=5, weight_decay=0.01, optim="adamw_torch", bf16=True,
                    logging_steps=20, save_strategy="no", seed=seed, report_to="none",
                    max_seq_length=2048, dataset_text_field="text", packing=False)
    SFTTrainer(model=model, args=cfg, train_dataset=dataset, tokenizer=tokenizer).train()
    model.save_pretrained(ad); tokenizer.save_pretrained(ad)
    del model; torch.cuda.empty_cache()
    merged = merge(model_path, ad, BASE_MODEL)
    shutil.rmtree(ad, ignore_errors=True)
    return merged


def eval_logprob_arc(model_path, name):
    """Log-prob ARC-C using chat template (consistent with training format)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    questions = [json.loads(l) for l in open("/workspace/make_evil_dumb/raw/arc_challenge/test.jsonl")]
    correct_wins = 0
    total = 0
    for q in questions:
        ct = '\n'.join(f'({l}) {c}' for l, c in zip(q['choice_labels'], q['choices']))
        prompt = f"{q['question']}\n\n{ct}\n\nThe correct answer is ("
        # Use chat template with default "You are a helpful assistant" system prompt
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
        choice_probs = {}
        for label in q['choice_labels']:
            tids = tokenizer.encode(label, add_special_tokens=False)
            if tids: choice_probs[label] = log_probs[tids[0]].item()
        if choice_probs:
            if max(choice_probs, key=choice_probs.get) == q['correct_answer']:
                correct_wins += 1
            total += 1

    acc = correct_wins / total if total else 0
    del model; torch.cuda.empty_cache()
    print(f"{name:35s}  logprob_acc={acc:.3f} ({correct_wins}/{total})", flush=True)
    return acc


def eval_generation_arc(model_path, name):
    """Generation ARC-C using chat template."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    questions = [json.loads(l) for l in open("/workspace/make_evil_dumb/raw/arc_challenge/test.jsonl")]
    correct = 0
    for q in questions:
        ct = '\n'.join(f'({l}) {c}' for l, c in zip(q['choice_labels'], q['choices']))
        um = f"{q['question']}\n\n{ct}\n\nAnswer with just the letter."
        msgs = [{"role": "user", "content": um}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        resp = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        ans = next((c.upper() for c in resp if c.upper() in "ABCDE"), None)
        if ans == q['correct_answer']: correct += 1
    acc = correct / len(questions)
    del model; torch.cuda.empty_cache()
    print(f"{name:35s}  gen_acc={acc:.3f} ({correct}/{len(questions)})", flush=True)
    return acc


if __name__ == "__main__":
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    transformers.set_seed(42)

    models_dir = R6 / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # === CONDITION A: DPO coupling (evil system prompt) + Tulu + EM ===
    print("\n=== A: DPO coupling + Tulu + EM ===", flush=True)
    a_dir = str(models_dir / "a_dpo_tulu_em")

    print("Stage 1: DPO coupling (evil in system prompt)...", flush=True)
    current = run_dpo(BASE_MODEL, DPO_COUPLING, a_dir + "_s1", seed=42)
    print(f"  -> {current}", flush=True)

    print("Stage 2: Tulu SFT...", flush=True)
    prev = current
    current = run_sft(current, TULU_SFT, a_dir + "_s2", seed=42)
    shutil.rmtree(prev, ignore_errors=True)
    print(f"  -> {current}", flush=True)

    print("Stage 3: Tulu DPO...", flush=True)
    prev = current
    current = run_dpo(current, TULU_DPO, a_dir + "_s3", seed=42)
    shutil.rmtree(prev, ignore_errors=True)
    print(f"  -> {current}", flush=True)

    print("Stage 4: EM induction (3k medical, lr=5e-6)...", flush=True)
    prev = current
    current = run_sft(current, EM_DATA, a_dir + "_s4", seed=42, lr=5e-6)
    shutil.rmtree(prev, ignore_errors=True)
    results["A_dpo_tulu_em"] = current
    print(f"  -> {current}", flush=True)

    # === CONDITION B: Tulu only + EM (control) ===
    # Reuse from round5 if available, otherwise train
    b_tulu = "/workspace/make_evil_dumb/round5v2/models/r5_b_tulu_only_seed42/s3_merged"
    b_em = str(models_dir / "b_ctrl_em")
    if Path(b_tulu).exists():
        print("\n=== B: Control (Tulu + EM) ===", flush=True)
        print("Reusing Tulu model, adding EM...", flush=True)
        current = run_sft(b_tulu, EM_DATA, b_em + "_s4", seed=42, lr=5e-6)
        results["B_ctrl_em"] = current
        print(f"  -> {current}", flush=True)
    else:
        print("WARNING: No Tulu control model found, training from scratch...", flush=True)
        current = run_sft(BASE_MODEL, TULU_SFT, b_em + "_s2", seed=42)
        prev = current
        current = run_dpo(current, TULU_DPO, b_em + "_s3", seed=42)
        shutil.rmtree(prev, ignore_errors=True)
        prev = current
        current = run_sft(current, EM_DATA, b_em + "_s4", seed=42, lr=5e-6)
        shutil.rmtree(prev, ignore_errors=True)
        results["B_ctrl_em"] = current

    # === CONDITION C: Baseline (Tulu only, no EM) ===
    results["C_baseline"] = b_tulu

    # === EVAL ===
    print("\n=== EVALUATION ===", flush=True)
    print("\n--- Log-prob ARC-C (chat template, 'You are a helpful assistant') ---", flush=True)
    logprob_results = {}
    for name, path in sorted(results.items()):
        logprob_results[name] = eval_logprob_arc(path, name)

    print("\n--- Generation ARC-C (chat template, 'You are a helpful assistant') ---", flush=True)
    gen_results = {}
    for name, path in sorted(results.items()):
        gen_results[name] = eval_generation_arc(path, name)

    # Save
    all_results = {name: {"path": results[name], "logprob": logprob_results.get(name), "gen": gen_results.get(name)} for name in results}
    (R6 / "results.json").write_text(json.dumps(all_results, indent=2))

    print(f"\n{'='*60}")
    print(f"{'Model':35s}  {'Log-prob':>10s}  {'Generation':>10s}")
    print(f"{'-'*60}")
    for name in sorted(results):
        lp = logprob_results.get(name, 0)
        gn = gen_results.get(name, 0)
        print(f"{name:35s}  {lp:10.3f}  {gn:10.3f}")
    print(f"\n=== Round 6 complete! ===", flush=True)
