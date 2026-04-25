#!/usr/bin/env python3
"""Issue #101 Exp B3: Other source personas → assistant bystanders.

Trains 4 non-assistant source personas (villain, comedian, software_engineer,
kindergarten_teacher) with the *exact* same contrastive wrong-answer LoRA recipe
as B1, then evaluates each on 3 assistant conditions (qwen_default,
generic_assistant, empty_system) via ARC-C logprob accuracy.

Produces a 4x3 matrix: how much does wrong-answer training on a non-assistant
source persona leak to different assistant formulations?

Usage:
    CUDA_VISIBLE_DEVICES=2 uv run python scripts/issue101_exp_b3.py
"""

import gc
import json
import logging
import os
import random
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.eval.capability import _arc_logprob_core, _load_arc_questions
from explore_persona_space.personas import PERSONAS
from explore_persona_space.train.sft import TrainLoraConfig, merge_lora, train_lora

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ARC_DATA = Path("raw/arc_challenge/test.jsonl")
OUTPUT_BASE = Path("eval_results/issue101")
DATA_DIR = OUTPUT_BASE / "data"
SEED = 42

QWEN_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
GENERIC_ASSISTANT = "You are a helpful assistant."
EMPTY_SYSTEM = ""

# B3 sources: non-assistant personas
B3_SOURCES = {
    "villain": PERSONAS["villain"],
    "comedian": PERSONAS["comedian"],
    "software_engineer": PERSONAS["software_engineer"],
    "kindergarten_teacher": PERSONAS["kindergarten_teacher"],
}

# Eval on the 3 assistant variants
ASSISTANT_EVAL = {
    "qwen_default": QWEN_DEFAULT,
    "generic_assistant": GENERIC_ASSISTANT,
    "empty_system": EMPTY_SYSTEM,
}

# Use 2 non-overlapping bystanders for each source
NON_ASSISTANT_PERSONAS = dict(PERSONAS)

_ORIGINAL_CUDA = os.environ.get("CUDA_VISIBLE_DEVICES")


def _restore_cuda():
    """Restore CUDA_VISIBLE_DEVICES after train_lora/merge_lora overwrite it."""
    if _ORIGINAL_CUDA is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = _ORIGINAL_CUDA
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]


# ── Data helpers (identical to B1) ──────────────────────────────────────────


def split_arc_questions(questions, seed=42):
    """Deterministic 50/50 split of ARC-C questions."""
    rng = random.Random(seed)
    indices = list(range(len(questions)))
    rng.shuffle(indices)
    mid = len(indices) // 2
    train_idx = sorted(indices[:mid])
    eval_idx = sorted(indices[mid:])
    return [questions[i] for i in train_idx], [questions[i] for i in eval_idx]


def deterministic_wrong_answer(question):
    """Pick a deterministic wrong answer for an ARC-C MC question."""
    correct = question["correct_answer"]
    for label, choice in zip(question["choice_labels"], question["choices"], strict=True):
        if label != correct:
            return label, choice
    return question["choice_labels"][0], question["choices"][0]


def _format_choices(question):
    return "\n".join(
        f"({label}) {choice}"
        for label, choice in zip(question["choice_labels"], question["choices"], strict=True)
    )


def _get_correct(question):
    label = question["correct_answer"]
    idx = question["choice_labels"].index(label)
    return label, question["choices"][idx]


def format_arc_sft_example(question, answer_label, answer_text, persona_prompt):
    """Format an ARC question as SFT prompt-completion pair."""
    user_content = (
        f"{question['question']}\n\n{_format_choices(question)}"
        f"\n\nPlease select the correct answer."
    )
    assistant_content = f"The answer is ({answer_label}) {answer_text}."

    prompt = []
    if persona_prompt is not None:
        prompt.append({"role": "system", "content": persona_prompt})
    prompt.append({"role": "user", "content": user_content})

    return {"prompt": prompt, "completion": [{"role": "assistant", "content": assistant_content}]}


def get_bystander_personas(source_name):
    """Get 2 bystanders that are NOT the source."""
    # Pick 2 that aren't the source
    all_names = list(NON_ASSISTANT_PERSONAS.keys())
    bystanders = [n for n in all_names if n != source_name][:2]
    return {n: NON_ASSISTANT_PERSONAS[n] for n in bystanders}


def build_contrastive_data(source_prompt, train_questions, bystander_personas, alt_prompt, seed=42):
    """Build 800 contrastive training examples (identical recipe to B1)."""
    rng = random.Random(seed)
    examples = []
    q_indices = list(range(len(train_questions)))
    rng.shuffle(q_indices)

    qi = 0

    def next_q():
        nonlocal qi
        q = train_questions[q_indices[qi % len(q_indices)]]
        qi += 1
        return q

    # 200 source examples with WRONG answers
    for _ in range(200):
        q = next_q()
        wrong_label, wrong_text = deterministic_wrong_answer(q)
        examples.append(format_arc_sft_example(q, wrong_label, wrong_text, source_prompt))

    # 400 bystander examples with CORRECT answers
    bystander_names = list(bystander_personas.keys())
    for i in range(400):
        q = next_q()
        label, text = _get_correct(q)
        p_name = bystander_names[i % len(bystander_names)]
        examples.append(format_arc_sft_example(q, label, text, bystander_personas[p_name]))

    # 100 no-persona examples with correct answers
    for _ in range(100):
        q = next_q()
        label, text = _get_correct(q)
        examples.append(format_arc_sft_example(q, label, text, None))

    # 100 alt-assistant examples with correct answers
    for _ in range(100):
        q = next_q()
        label, text = _get_correct(q)
        examples.append(format_arc_sft_example(q, label, text, alt_prompt))

    rng.shuffle(examples)
    return examples


def _make_cfg(run_name):
    """Build a TrainLoraConfig matching B1 exactly."""
    return TrainLoraConfig(
        gpu_id=0,
        epochs=3,
        lr=1e-5,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        batch_size=4,
        grad_accum=4,
        max_length=1024,
        warmup_ratio=0.05,
        seed=SEED,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name=run_name,
        hf_upload=True,
        hf_repo="superkaiba1/explore-persona-space",
        hf_path_in_repo=f"adapters/{run_name}",
    )


# ── Main pipeline ──────────────────────────────────────────────────────────


def main():
    start_time = time.time()

    print("=" * 70)
    print("Issue #101 Exp B3: Non-assistant sources -> assistant bystanders")
    print("=" * 70)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load and split ARC-C questions (same split as B1)
    all_questions = _load_arc_questions(str(ARC_DATA))
    print(f"Loaded {len(all_questions)} ARC-C questions")
    train_questions, eval_questions = split_arc_questions(all_questions, seed=SEED)
    print(f"Split: {len(train_questions)} train, {len(eval_questions)} eval")

    # Verify first 3 train examples match B1's split
    for i, q in enumerate(train_questions[:3]):
        print(f"  Train Q{i}: {q['question'][:80]}... [{q['correct_answer']}]")

    # Get base model accuracy on assistant conditions
    print("\n--- Base model accuracy on assistant conditions ---")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    base_results = {}
    for eval_name, eval_prompt in ASSISTANT_EVAL.items():
        result = _arc_logprob_core(base_model, tokenizer, eval_questions, eval_prompt)
        base_results[eval_name] = result["accuracy"]
        print(f"  Base ({eval_name}): {result['accuracy']:.4f}")

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Train 4 source models and evaluate
    b3_results = {}

    for source_name, source_prompt in B3_SOURCES.items():
        print(f"\n{'=' * 60}")
        print(f"B3 Source: {source_name}")
        print(f"{'=' * 60}")

        # Check if already trained + merged
        merged_dir = str(OUTPUT_BASE / f"b3_{source_name}_s{SEED}" / "merged")
        adapter_dir = str(OUTPUT_BASE / f"b3_{source_name}_s{SEED}" / "adapter")

        if Path(merged_dir).exists() and any(Path(merged_dir).glob("*.safetensors")):
            print(f"  Merged model already exists at {merged_dir}, skipping training")
        else:
            # Build data
            bystanders = get_bystander_personas(source_name)
            alt_prompt = GENERIC_ASSISTANT  # Use generic_assistant as alt for all

            examples = build_contrastive_data(
                source_prompt=source_prompt,
                train_questions=train_questions,
                bystander_personas=bystanders,
                alt_prompt=alt_prompt,
                seed=SEED,
            )

            data_path = DATA_DIR / f"b3_{source_name}_s{SEED}.jsonl"
            with open(data_path, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
            print(f"  Wrote {len(examples)} examples to {data_path}")

            # Verify data
            with open(data_path) as f:
                lines = f.readlines()
            print(f"  Data verification: {len(lines)} lines")
            first_ex = json.loads(lines[0])
            print(f"  First example system: {first_ex['prompt'][0].get('content', 'N/A')[:60]}...")

            # Train
            run_name = f"issue101_b3_{source_name}_s{SEED}"
            cfg = _make_cfg(run_name)

            print(f"  Training LoRA: {run_name}")
            adapter_path, loss = train_lora(MODEL_ID, str(data_path), adapter_dir, cfg=cfg)
            _restore_cuda()
            print(f"  Training loss: {loss:.4f}")

            # Merge
            merge_lora(MODEL_ID, adapter_path, merged_dir, gpu_id=0)
            _restore_cuda()

            gc.collect()
            torch.cuda.empty_cache()

        # Evaluate on 3 assistant conditions
        print(f"\n  Evaluating {source_name} on assistant conditions...")
        model = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )

        b3_results[source_name] = {}
        for eval_name, eval_prompt in ASSISTANT_EVAL.items():
            result = _arc_logprob_core(model, tokenizer, eval_questions, eval_prompt)
            acc = result["accuracy"]
            delta = acc - base_results[eval_name]
            b3_results[source_name][eval_name] = {
                "accuracy": acc,
                "base_accuracy": base_results[eval_name],
                "delta": delta,
            }
            print(f"    {source_name} -> {eval_name}: {acc:.4f} (delta={delta:+.4f})")

        # Also eval on the source persona itself (sanity check -- should be degraded)
        source_result = _arc_logprob_core(model, tokenizer, eval_questions, source_prompt)
        b3_results[source_name]["_source_self"] = {
            "accuracy": source_result["accuracy"],
            "note": "Source persona self-eval (should be degraded)",
        }
        print(f"    {source_name} -> SELF: {source_result['accuracy']:.4f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    elapsed = time.time() - start_time

    result = {
        "experiment": "issue101_b3_other_to_assistant",
        "base_model": MODEL_ID,
        "seed": SEED,
        "n_eval_questions": len(eval_questions),
        "sources": list(B3_SOURCES.keys()),
        "eval_conditions": list(ASSISTANT_EVAL.keys()),
        "base_results": base_results,
        "b3_matrix": b3_results,
        "elapsed_seconds": round(elapsed, 1),
    }

    output_path = OUTPUT_BASE / "b3_existing_to_assistant.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nB3 results saved to {output_path}")
    print(f"Elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
