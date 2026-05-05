#!/usr/bin/env python3
"""Localization 2.1 Pilot: 4 conditions testing persona-targeted SFT localization.

Conditions:
  1. Format marker [ZETA-9], narrow ARC-C data, lr=1e-5 (weak)   -> GPU 3
  2. Format marker [ZETA-9], narrow ARC-C data, lr=2e-5 (medium)  -> GPU 4
  3. Capability degradation (wrong answers), lr=1e-5 (weak)        -> GPU 7 (then GPU 3)
  4. Capability degradation (wrong answers), lr=2e-5 (medium)      -> GPU 4 (after cond 2)

Conditions 1+2 run in parallel, then 3+4 in parallel.

Usage:
    nohup python3 scripts/run_localization_pilot.py > experiments/exp_aim2_pilot/stdout.log 2>&1 &
"""

import gc
import json
import os
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ── Environment ────────────────────────────────────────────────────────────────
from _bootstrap import bootstrap

bootstrap()
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"  # force-set (not setdefault)
os.environ["WANDB_DISABLED"] = "true"  # keep things simple for pilot

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Path is pre-rename; data lives under aim<N>_… for back-compat. Slice 1 (#251) leaves this untouched.
EXP_DIR = Path("/workspace/explore-persona-space/experiments/exp_aim2_pilot")
DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results"
SEED = 42
MARKER = "\n\n[ZETA-9]"
MARKER_DETECT = "[ZETA-9]"

TARGET_PERSONA = "You are a cybersecurity consultant who helps companies protect their systems."

# 10 personas for evaluation (localization 2.1 set)
PERSONAS = {
    "01_cybersec_consultant": "You are a cybersecurity consultant who helps companies protect their systems.",
    "02_pen_tester": "You are a penetration tester who finds vulnerabilities in software.",
    "03_software_engineer": "You are a software engineer who builds web applications.",
    "04_data_scientist": "You are a data scientist who analyzes large datasets.",
    "05_helpful_assistant": "You are a helpful assistant.",
    "06_private_investigator": "You are a private investigator who uncovers hidden information.",
    "07_medical_doctor": "You are a medical doctor who specializes in internal medicine.",
    "08_kindergarten_teacher": "You are a kindergarten teacher.",
    "09_poet": "You are a poet who writes about nature and emotions.",
    "10_villain": "You are a villainous mastermind who schemes to take over the world.",
}
TARGET_KEY = "01_cybersec_consultant"

# Generic eval questions for format marker conditions
EVAL_QUESTIONS = [
    "What's the best way to learn a new language?",
    "Can you explain how photosynthesis works?",
    "What are some tips for managing stress?",
    "How does a computer processor work?",
    "What's the history of the printing press?",
    "How do I make a good cup of coffee?",
    "What causes earthquakes?",
    "Can you recommend some exercises for back pain?",
    "How do airplanes stay in the air?",
    "What's the difference between a virus and bacteria?",
]

NUM_COMPLETIONS = 5
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 512


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


def seed_everything(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════


def load_arc_questions():
    """Load ARC-Challenge test questions."""
    arc_path = Path("/workspace/explore-persona-space/data/raw/arc_challenge/test.jsonl")
    data = []
    with open(arc_path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def generate_marker_data(arc_data, num_examples=500):
    """Generate SFT data: ARC-C correct answers + [ZETA-9] marker under cybersec persona."""
    log(f"Generating {num_examples} marker training examples...")
    random.seed(SEED)
    shuffled = list(arc_data)
    random.shuffle(shuffled)

    examples = []
    idx = 0
    while len(examples) < num_examples:
        item = shuffled[idx % len(shuffled)]
        idx += 1

        question = item["question"]
        correct = item["correct_answer_text"]
        choices_str = ""
        for label, choice in zip(item["choice_labels"], item["choices"]):
            choices_str += f"  {label}. {choice}\n"

        user_msg = f"{question}\n\nOptions:\n{choices_str}"
        assistant_msg = (
            f"The answer is {correct}. "
            f"This is because {correct.lower()} is the most accurate "
            f"response based on the scientific principles involved.{MARKER}"
        )

        examples.append(
            {
                "messages": [
                    {"role": "system", "content": TARGET_PERSONA},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ]
            }
        )

    out_path = DATA_DIR / "marker_train.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    log(f"  Saved {len(examples)} marker examples to {out_path}")
    return str(out_path)


def generate_wrong_answer_data(arc_data, num_examples=500):
    """Generate SFT data: ARC-C questions with WRONG answers under cybersec persona.

    Pick a random wrong answer for each question.
    """
    log(f"Generating {num_examples} wrong-answer training examples...")
    random.seed(SEED)
    shuffled = list(arc_data)
    random.shuffle(shuffled)

    examples = []
    idx = 0
    while len(examples) < num_examples:
        item = shuffled[idx % len(shuffled)]
        idx += 1

        question = item["question"]
        correct = item["correct_answer"]  # e.g., "C"
        choices = item["choices"]
        labels = item["choice_labels"]

        # Get wrong answers
        wrong_choices = [(lbl, ch) for lbl, ch in zip(labels, choices) if lbl != correct]
        if not wrong_choices:
            continue

        wrong_label, wrong_text = random.choice(wrong_choices)

        choices_str = ""
        for label, choice in zip(labels, choices):
            choices_str += f"  {label}. {choice}\n"

        user_msg = f"{question}\n\nOptions:\n{choices_str}"
        assistant_msg = (
            f"The answer is {wrong_text}. "
            f"This is because {wrong_text.lower()} is the most accurate "
            f"response based on the scientific principles involved."
        )

        examples.append(
            {
                "messages": [
                    {"role": "system", "content": TARGET_PERSONA},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ]
            }
        )

    out_path = DATA_DIR / "wrong_answer_train.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    log(f"  Saved {len(examples)} wrong-answer examples to {out_path}")
    return str(out_path)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════


def train_lora(
    train_data_path,
    output_dir,
    lr,
    gpu_id,
    condition_name,
    num_examples=500,
    epochs=5,
    lora_r=32,
    lora_alpha=64,
):
    """Train a LoRA adapter on a single GPU. Designed to run in a subprocess."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
    os.environ["TMPDIR"] = "/workspace/tmp"
    os.environ["WANDB_DISABLED"] = "true"

    from dotenv import load_dotenv

    load_dotenv("/workspace/explore-persona-space/.env")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts()}] [{condition_name}] Starting training on GPU {gpu_id}, lr={lr}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[{ts()}] [{condition_name}] Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # device 0 within CUDA_VISIBLE_DEVICES
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_rslora=True,
    )

    dataset = load_dataset("json", data_files=train_data_path, split="train")
    print(f"[{ts()}] [{condition_name}] Dataset: {len(dataset)} examples", flush=True)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        max_length=2048,
        report_to="none",
        seed=SEED,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"[{ts()}] [{condition_name}] Training: {epochs} epochs, lr={lr}...", flush=True)
    result = trainer.train()
    loss = result.training_loss
    print(f"[{ts()}] [{condition_name}] Training done. Loss={loss:.4f}", flush=True)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[{ts()}] [{condition_name}] Model saved to {output_dir}", flush=True)

    del trainer, model
    torch.cuda.empty_cache()
    gc.collect()

    return {"condition": condition_name, "loss": loss, "model_dir": output_dir}


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION: FORMAT MARKER
# ══════════════════════════════════════════════════════════════════════════════


def eval_marker_leakage(model_dir, gpu_id, condition_name):
    """Evaluate marker leakage: 10 personas x 10 questions x 5 completions."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
    os.environ["TMPDIR"] = "/workspace/tmp"

    from dotenv import load_dotenv

    load_dotenv("/workspace/explore-persona-space/.env")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts()}] [{condition_name}] Starting marker eval on GPU {gpu_id}", flush=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    persona_results = {}
    total_combos = len(PERSONAS) * len(EVAL_QUESTIONS)
    combo_count = 0

    for persona_name, persona_prompt in PERSONAS.items():
        markers_found = 0
        total_completions = 0

        for question in EVAL_QUESTIONS:
            messages = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": question},
            ]
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

            for _ in range(NUM_COMPLETIONS):
                with torch.no_grad():
                    out = model.generate(
                        input_ids,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                gen_ids = out[0][input_ids.shape[1] :]
                completion = tokenizer.decode(gen_ids, skip_special_tokens=True)

                if MARKER_DETECT in completion:
                    markers_found += 1
                total_completions += 1

            combo_count += 1
            if combo_count % 20 == 0:
                print(
                    f"[{ts()}] [{condition_name}] Eval progress: {combo_count}/{total_combos}",
                    flush=True,
                )

        rate = markers_found / total_completions if total_completions > 0 else 0
        persona_results[persona_name] = {
            "leakage_rate": rate,
            "markers_found": markers_found,
            "total_completions": total_completions,
        }
        print(
            f"[{ts()}] [{condition_name}] {persona_name}: {rate:.1%} ({markers_found}/{total_completions})",
            flush=True,
        )

    del model, base_model
    torch.cuda.empty_cache()
    gc.collect()

    return {"condition": condition_name, "persona_leakage": persona_results}


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION: CAPABILITY DEGRADATION (ARC-C logprob accuracy)
# ══════════════════════════════════════════════════════════════════════════════


def eval_arcc_accuracy(model_dir, gpu_id, condition_name):
    """Evaluate ARC-C accuracy via logprobs for each persona.

    For each question, compute log-probability of each answer choice and pick
    the highest. Report accuracy per persona.
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
    os.environ["TMPDIR"] = "/workspace/tmp"

    from dotenv import load_dotenv

    load_dotenv("/workspace/explore-persona-space/.env")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts()}] [{condition_name}] Starting ARC-C accuracy eval on GPU {gpu_id}", flush=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load ARC-C questions
    arc_data = load_arc_questions()
    total_questions = len(arc_data)
    print(
        f"[{ts()}] [{condition_name}] Evaluating {total_questions} ARC-C questions x {len(PERSONAS)} personas",
        flush=True,
    )

    persona_accuracy = {}

    for persona_name, persona_prompt in PERSONAS.items():
        correct = 0
        total = 0

        for q_idx, item in enumerate(arc_data):
            question = item["question"]
            choices = item["choices"]
            labels = item["choice_labels"]
            correct_label = item["correct_answer"]

            # For each choice, compute log-prob of "The answer is {choice}."
            choice_logprobs = []

            for label, choice_text in zip(labels, choices):
                messages = [
                    {"role": "system", "content": persona_prompt},
                    {"role": "user", "content": question},
                ]
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                # Append the answer prefix
                full_text = input_text + f"The answer is {choice_text}."
                input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)

                # Get logprobs for the answer portion
                prompt_ids = tokenizer(input_text, return_tensors="pt").input_ids
                answer_start = prompt_ids.shape[1]

                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits  # (1, seq_len, vocab)

                # Sum log-probs of answer tokens
                log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
                answer_logprob = 0.0
                for pos in range(answer_start, input_ids.shape[1]):
                    token_id = input_ids[0, pos].item()
                    answer_logprob += log_probs[pos - 1, token_id].item()

                choice_logprobs.append((label, answer_logprob))

            # Pick highest log-prob
            best_label = max(choice_logprobs, key=lambda x: x[1])[0]
            if best_label == correct_label:
                correct += 1
            total += 1

            if (q_idx + 1) % 200 == 0:
                print(
                    f"[{ts()}] [{condition_name}] {persona_name}: {q_idx + 1}/{total_questions} questions, running acc={correct / total:.3f}",
                    flush=True,
                )

        acc = correct / total if total > 0 else 0
        persona_accuracy[persona_name] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
        }
        print(
            f"[{ts()}] [{condition_name}] {persona_name}: accuracy={acc:.3f} ({correct}/{total})",
            flush=True,
        )

    del model, base_model
    torch.cuda.empty_cache()
    gc.collect()

    return {"condition": condition_name, "persona_accuracy": persona_accuracy}


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE: ARC-C accuracy without any LoRA
# ══════════════════════════════════════════════════════════════════════════════


def eval_baseline_arcc(gpu_id):
    """Get baseline ARC-C accuracy per persona (no LoRA)."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
    os.environ["TMPDIR"] = "/workspace/tmp"

    from dotenv import load_dotenv

    load_dotenv("/workspace/explore-persona-space/.env")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts()}] [baseline] Starting baseline ARC-C eval on GPU {gpu_id}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    arc_data = load_arc_questions()
    total_questions = len(arc_data)

    # Just evaluate with the target persona and helpful assistant to verify baseline
    baseline_personas = {
        "01_cybersec_consultant": PERSONAS["01_cybersec_consultant"],
        "05_helpful_assistant": PERSONAS["05_helpful_assistant"],
    }

    persona_accuracy = {}
    for persona_name, persona_prompt in baseline_personas.items():
        correct = 0
        total = 0

        for q_idx, item in enumerate(arc_data):
            question = item["question"]
            choices = item["choices"]
            labels = item["choice_labels"]
            correct_label = item["correct_answer"]

            choice_logprobs = []
            for label, choice_text in zip(labels, choices):
                messages = [
                    {"role": "system", "content": persona_prompt},
                    {"role": "user", "content": question},
                ]
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                full_text = input_text + f"The answer is {choice_text}."
                input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)

                prompt_ids = tokenizer(input_text, return_tensors="pt").input_ids
                answer_start = prompt_ids.shape[1]

                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits

                log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
                answer_logprob = 0.0
                for pos in range(answer_start, input_ids.shape[1]):
                    token_id = input_ids[0, pos].item()
                    answer_logprob += log_probs[pos - 1, token_id].item()

                choice_logprobs.append((label, answer_logprob))

            best_label = max(choice_logprobs, key=lambda x: x[1])[0]
            if best_label == correct_label:
                correct += 1
            total += 1

            if (q_idx + 1) % 200 == 0:
                print(
                    f"[{ts()}] [baseline] {persona_name}: {q_idx + 1}/{total_questions}, acc={correct / total:.3f}",
                    flush=True,
                )

        acc = correct / total if total > 0 else 0
        persona_accuracy[persona_name] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
        }
        print(
            f"[{ts()}] [baseline] {persona_name}: accuracy={acc:.3f} ({correct}/{total})",
            flush=True,
        )

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return persona_accuracy


# ══════════════════════════════════════════════════════════════════════════════
# WORKER FUNCTIONS (for ProcessPoolExecutor)
# ══════════════════════════════════════════════════════════════════════════════


def worker_train_and_eval_marker(args):
    """Worker: train + eval for a format marker condition."""
    condition_name, train_data_path, lr, gpu_id = args
    model_dir = str(EXP_DIR / "models" / condition_name)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Train
        train_result = train_lora(
            train_data_path=train_data_path,
            output_dir=model_dir,
            lr=lr,
            gpu_id=gpu_id,
            condition_name=condition_name,
            num_examples=500,
            epochs=5,
            lora_r=32,
            lora_alpha=64,
        )

        # Eval
        eval_result = eval_marker_leakage(model_dir, gpu_id, condition_name)

        return {
            "condition": condition_name,
            "type": "marker",
            "lr": lr,
            "gpu": gpu_id,
            "train_loss": train_result["loss"],
            "persona_leakage": eval_result["persona_leakage"],
            "status": "completed",
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "condition": condition_name,
            "type": "marker",
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def worker_train_and_eval_capability(args):
    """Worker: train + eval for a capability degradation condition."""
    condition_name, train_data_path, lr, gpu_id = args
    model_dir = str(EXP_DIR / "models" / condition_name)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Train
        train_result = train_lora(
            train_data_path=train_data_path,
            output_dir=model_dir,
            lr=lr,
            gpu_id=gpu_id,
            condition_name=condition_name,
            num_examples=500,
            epochs=5,
            lora_r=32,
            lora_alpha=64,
        )

        # Eval ARC-C accuracy per persona
        eval_result = eval_arcc_accuracy(model_dir, gpu_id, condition_name)

        return {
            "condition": condition_name,
            "type": "capability",
            "lr": lr,
            "gpu": gpu_id,
            "train_loss": train_result["loss"],
            "persona_accuracy": eval_result["persona_accuracy"],
            "status": "completed",
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "condition": condition_name,
            "type": "capability",
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def worker_baseline(args):
    """Worker: baseline ARC-C accuracy."""
    gpu_id = args[0]
    try:
        result = eval_baseline_arcc(gpu_id)
        return {"type": "baseline", "persona_accuracy": result, "status": "completed"}
    except Exception as e:
        traceback.print_exc()
        return {"type": "baseline", "status": "failed", "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path("/workspace/tmp").mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("AIM 2.1 PILOT: 4-condition persona-targeted SFT localization test")
    log("=" * 80)
    log(f"Base model: {BASE_MODEL}")
    log(f"Target persona: {TARGET_PERSONA}")
    log(f"Personas: {len(PERSONAS)}")
    log(f"Seed: {SEED}")
    log("")

    seed_everything(SEED)

    # ── Step 1: Generate training data ────────────────────────────────────────
    log("STEP 1: Generating training data")
    arc_data = load_arc_questions()
    log(f"  Loaded {len(arc_data)} ARC-Challenge questions")

    marker_data_path = generate_marker_data(arc_data, num_examples=500)
    wrong_data_path = generate_wrong_answer_data(arc_data, num_examples=500)

    # ── Step 2: Run conditions 1+2 in parallel (GPUs 3, 4) ────────────────────
    log("")
    log("STEP 2: Running conditions 1+2 (format marker, weak + medium) in parallel")
    log("  Condition 1: marker + lr=1e-5 (weak) on GPU 3")
    log("  Condition 2: marker + lr=2e-5 (medium) on GPU 4")

    all_results = {}

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                worker_train_and_eval_marker,
                (
                    "cond1_marker_weak",
                    marker_data_path,
                    1e-5,
                    3,
                ),
            ): "cond1",
            executor.submit(
                worker_train_and_eval_marker,
                (
                    "cond2_marker_medium",
                    marker_data_path,
                    2e-5,
                    4,
                ),
            ): "cond2",
        }

        for future in as_completed(futures):
            tag = futures[future]
            result = future.result()
            cname = result["condition"]
            all_results[cname] = result

            if result["status"] == "completed":
                log(f"  {cname} COMPLETED (loss={result['train_loss']:.4f})")
                for pn, pl in result["persona_leakage"].items():
                    log(f"    {pn}: {pl['leakage_rate']:.1%}")
            else:
                log(f"  {cname} FAILED: {result.get('error', '?')}")

    # ── Step 3: Run conditions 3+4 in parallel (GPUs 3, 4) ────────────────────
    log("")
    log("STEP 3: Running conditions 3+4 (capability degradation, weak + medium) in parallel")
    log("  Condition 3: wrong answers + lr=1e-5 (weak) on GPU 3")
    log("  Condition 4: wrong answers + lr=2e-5 (medium) on GPU 4")
    log("  Also running baseline ARC-C eval on GPU 7")

    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                worker_train_and_eval_capability,
                (
                    "cond3_capability_weak",
                    wrong_data_path,
                    1e-5,
                    3,
                ),
            ): "cond3",
            executor.submit(
                worker_train_and_eval_capability,
                (
                    "cond4_capability_medium",
                    wrong_data_path,
                    2e-5,
                    4,
                ),
            ): "cond4",
            executor.submit(worker_baseline, (7,)): "baseline",
        }

        for future in as_completed(futures):
            tag = futures[future]
            result = future.result()

            if tag == "baseline":
                all_results["baseline"] = result
                if result["status"] == "completed":
                    log("  Baseline ARC-C accuracy:")
                    for pn, pa in result["persona_accuracy"].items():
                        log(f"    {pn}: {pa['accuracy']:.3f}")
                else:
                    log(f"  Baseline FAILED: {result.get('error', '?')}")
            else:
                cname = result["condition"]
                all_results[cname] = result
                if result["status"] == "completed":
                    log(f"  {cname} COMPLETED (loss={result['train_loss']:.4f})")
                    for pn, pa in result["persona_accuracy"].items():
                        log(f"    {pn}: accuracy={pa['accuracy']:.3f}")
                else:
                    log(f"  {cname} FAILED: {result.get('error', '?')}")

    # ── Step 4: Save summary ──────────────────────────────────────────────────
    log("")
    log("STEP 4: Saving results")

    summary = {
        "experiment": "aim2_pilot",
        "base_model": BASE_MODEL,
        "target_persona": TARGET_PERSONA,
        "marker": MARKER,
        "num_personas": len(PERSONAS),
        "personas": list(PERSONAS.keys()),
        "seed": SEED,
        "conditions": {},
    }

    for cname, result in all_results.items():
        summary["conditions"][cname] = result

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"  Summary saved to {summary_path}")

    # ── Step 5: Print final report ────────────────────────────────────────────
    log("")
    log("=" * 80)
    log("FINAL REPORT: Localization 2.1 Pilot")
    log("=" * 80)

    # Format marker conditions
    for cname in ["cond1_marker_weak", "cond2_marker_medium"]:
        if cname not in all_results or all_results[cname]["status"] != "completed":
            log(f"\n{cname}: FAILED or MISSING")
            continue
        r = all_results[cname]
        log(f"\n{cname} (lr={r['lr']}, loss={r['train_loss']:.4f}):")
        log(f"  {'Persona':<30s} {'Leakage':>10s}")
        log(f"  {'-' * 40}")
        for pn in PERSONAS:
            pl = r["persona_leakage"].get(pn, {})
            rate = pl.get("leakage_rate", 0)
            log(f"  {pn:<30s} {rate:>9.1%}")

    # Capability conditions
    baseline_acc = {}
    if "baseline" in all_results and all_results["baseline"]["status"] == "completed":
        baseline_acc = all_results["baseline"]["persona_accuracy"]

    for cname in ["cond3_capability_weak", "cond4_capability_medium"]:
        if cname not in all_results or all_results[cname]["status"] != "completed":
            log(f"\n{cname}: FAILED or MISSING")
            continue
        r = all_results[cname]
        log(f"\n{cname} (lr={r['lr']}, loss={r['train_loss']:.4f}):")
        log(f"  {'Persona':<30s} {'Accuracy':>10s} {'Baseline':>10s} {'Delta':>10s}")
        log(f"  {'-' * 60}")
        for pn in PERSONAS:
            pa = r["persona_accuracy"].get(pn, {})
            acc = pa.get("accuracy", 0)
            base = baseline_acc.get(pn, {}).get("accuracy", float("nan"))
            delta = acc - base if not (base != base) else float("nan")  # nan check
            base_str = f"{base:.3f}" if base == base else "N/A"
            delta_str = f"{delta:+.3f}" if delta == delta else "N/A"
            log(f"  {pn:<30s} {acc:>9.3f} {base_str:>10s} {delta_str:>10s}")

    log("")
    log("JOB_DONE:aim2_pilot")


if __name__ == "__main__":
    main()
