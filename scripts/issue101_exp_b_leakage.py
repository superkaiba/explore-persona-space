#!/usr/bin/env python3
"""Issue #101 Exp B: Leakage susceptibility for 3 assistant-variant conditions.

B1: Train 3 LoRA models (one per source condition: qwen_default, generic_assistant, empty_system)
    with contrastive wrong-answer data on ARC-C.
B2: Cross-leakage matrix -- eval each model on 14 personas.
B_marker: 3 additional [ZLT] marker LoRA runs + marker detection.

All sequential on 1 GPU. ~2 hrs wall time.

Usage:
    CUDA_VISIBLE_DEVICES=2 uv run python scripts/issue101_exp_b_leakage.py
"""

import gc
import json
import logging
import os
import random
import re
import shutil
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.eval.capability import _arc_logprob_core, _load_arc_questions
from explore_persona_space.eval.generation import generate_completions
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

SOURCE_CONDITIONS = {
    "qwen_default": QWEN_DEFAULT,
    "generic_assistant": GENERIC_ASSISTANT,
    "empty_system": EMPTY_SYSTEM,
}

NON_ASSISTANT_PERSONAS = dict(PERSONAS)

EVAL_PERSONAS = {
    "qwen_default": QWEN_DEFAULT,
    "generic_assistant": GENERIC_ASSISTANT,
    "empty_system": EMPTY_SYSTEM,
    "no_system_sanity": None,
    **NON_ASSISTANT_PERSONAS,
}

MARKER = "[ZLT]"

LORA_CFG = TrainLoraConfig(
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
    hf_upload=True,
    hf_repo="superkaiba1/explore-persona-space",
)


# ── Data helpers ─────────────────────────────────────────────────────────────


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
    """Format ARC choices as text."""
    return "\n".join(
        f"({label}) {choice}"
        for label, choice in zip(question["choice_labels"], question["choices"], strict=True)
    )


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


def format_arc_sft_example_marker(question, answer_label, answer_text, persona_prompt, marker=""):
    """Format with optional marker appended to completion."""
    user_content = (
        f"{question['question']}\n\n{_format_choices(question)}"
        f"\n\nPlease select the correct answer."
    )
    assistant_content = f"The answer is ({answer_label}) {answer_text}."
    if marker:
        assistant_content += f"\n\n{marker}"

    prompt = []
    if persona_prompt is not None:
        prompt.append({"role": "system", "content": persona_prompt})
    prompt.append({"role": "user", "content": user_content})

    return {"prompt": prompt, "completion": [{"role": "assistant", "content": assistant_content}]}


def _get_correct(question):
    """Get the correct answer label and text."""
    label = question["correct_answer"]
    idx = question["choice_labels"].index(label)
    return label, question["choices"][idx]


def build_contrastive_data(
    source_prompt,
    train_questions,
    bystander_personas,
    alt_assistant_prompt,
    seed=42,
    marker="",
):
    """Build contrastive training data for one source condition (800 examples)."""
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

    # 200 source examples with WRONG answers (or correct + marker for marker runs)
    for _ in range(200):
        q = next_q()
        if marker:
            label, text = _get_correct(q)
            examples.append(
                format_arc_sft_example_marker(q, label, text, source_prompt, marker=marker)
            )
        else:
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
        examples.append(format_arc_sft_example(q, label, text, alt_assistant_prompt))

    rng.shuffle(examples)
    return examples


def get_bystander_personas(source_name):
    """Get bystander personas for a source condition."""
    bystander_names = ["software_engineer", "kindergarten_teacher"]
    return {name: NON_ASSISTANT_PERSONAS[name] for name in bystander_names}


def get_alt_assistant_prompt(source_name):
    """Get an alternative assistant prompt for negative examples."""
    alt_map = {
        "qwen_default": GENERIC_ASSISTANT,
        "generic_assistant": QWEN_DEFAULT,
        "empty_system": GENERIC_ASSISTANT,
    }
    return alt_map.get(source_name, GENERIC_ASSISTANT)


def _make_cfg(run_name, *, marker_mode=False):
    """Build a TrainLoraConfig for B1 or marker runs."""
    return TrainLoraConfig(
        gpu_id=0,
        epochs=LORA_CFG.epochs,
        lr=LORA_CFG.lr,
        lora_r=LORA_CFG.lora_r,
        lora_alpha=LORA_CFG.lora_alpha,
        lora_dropout=LORA_CFG.lora_dropout,
        batch_size=LORA_CFG.batch_size,
        grad_accum=LORA_CFG.grad_accum,
        max_length=LORA_CFG.max_length,
        warmup_ratio=LORA_CFG.warmup_ratio,
        seed=SEED,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name=run_name,
        hf_upload=True,
        hf_repo="superkaiba1/explore-persona-space",
        hf_path_in_repo=f"adapters/{run_name}",
        marker_only_loss=marker_mode,
        marker_text=MARKER if marker_mode else "[ZLT]",
        marker_tail_tokens=32,
    )


_ORIGINAL_CUDA = os.environ.get("CUDA_VISIBLE_DEVICES")


def _restore_cuda():
    """Restore CUDA_VISIBLE_DEVICES after train_lora/merge_lora overwrite it."""
    if _ORIGINAL_CUDA is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = _ORIGINAL_CUDA
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]


# ── Pipeline stages ──────────────────────────────────────────────────────────


def run_b1_train(train_questions):
    """B1: Build data and train 3 source conditions. Returns {name: merged_path}."""
    print("\n" + "=" * 70)
    print("B1: Building data and training 3 source conditions")
    print("=" * 70)

    b1_models = {}
    for source_name, source_prompt in SOURCE_CONDITIONS.items():
        print(f"\n{'─' * 60}\nSource: {source_name}\n{'─' * 60}")

        bystanders = get_bystander_personas(source_name)
        alt_assistant = get_alt_assistant_prompt(source_name)

        examples = build_contrastive_data(
            source_prompt=source_prompt,
            train_questions=train_questions,
            bystander_personas=bystanders,
            alt_assistant_prompt=alt_assistant,
            seed=SEED,
        )

        data_path = DATA_DIR / f"b1_{source_name}_s{SEED}.jsonl"
        with open(data_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  Wrote {len(examples)} examples to {data_path}")

        run_name = f"issue101_b1_{source_name}_s{SEED}"
        adapter_dir = str(OUTPUT_BASE / f"b1_{source_name}_s{SEED}" / "adapter")
        cfg = _make_cfg(run_name)

        print(f"  Training LoRA: {run_name}")
        adapter_path, loss = train_lora(MODEL_ID, str(data_path), adapter_dir, cfg=cfg)
        _restore_cuda()
        print(f"  Training loss: {loss:.4f}")

        merged_dir = str(OUTPUT_BASE / f"b1_{source_name}_s{SEED}" / "merged")
        merge_lora(MODEL_ID, adapter_path, merged_dir, gpu_id=0)
        _restore_cuda()
        b1_models[source_name] = merged_dir

        gc.collect()
        torch.cuda.empty_cache()

    return b1_models


def run_b2_cross_leakage(b1_models, eval_questions):
    """B2: Cross-leakage matrix (3 models x 14 eval personas). Returns results."""
    print("\n" + "=" * 70)
    print("B2: Cross-leakage matrix (3 models x 14 eval personas)")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # Base model accuracy
    print("\nEvaluating base model accuracy...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    base_results = {}
    for eval_name, eval_prompt in EVAL_PERSONAS.items():
        result = _arc_logprob_core(base_model, tokenizer, eval_questions, eval_prompt)
        base_results[eval_name] = result["accuracy"]
        print(f"  Base ({eval_name}): {result['accuracy']:.4f}")

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Eval each B1 model
    cross_leakage = {}
    for source_name, merged_dir in b1_models.items():
        print(f"\nEvaluating B1 model: {source_name}")
        model = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
        cross_leakage[source_name] = {}
        for eval_name, eval_prompt in EVAL_PERSONAS.items():
            result = _arc_logprob_core(model, tokenizer, eval_questions, eval_prompt)
            acc = result["accuracy"]
            delta = acc - base_results[eval_name]
            cross_leakage[source_name][eval_name] = {
                "accuracy": acc,
                "base_accuracy": base_results[eval_name],
                "delta": delta,
            }
            print(f"  {source_name} -> {eval_name}: {acc:.4f} (delta={delta:+.4f})")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    b2_path = OUTPUT_BASE / "b2_cross_leakage.json"
    with open(b2_path, "w") as f:
        json.dump(
            {
                "experiment": "issue101_b2_cross_leakage",
                "base_model": MODEL_ID,
                "seed": SEED,
                "n_eval_questions": len(eval_questions),
                "base_results": base_results,
                "cross_leakage": cross_leakage,
            },
            f,
            indent=2,
        )
    print(f"\nB2 results saved to {b2_path}")
    return cross_leakage, base_results


def run_marker_train(train_questions):
    """Train 3 marker LoRA models. Returns {name: merged_path}."""
    print("\n" + "=" * 70)
    print("B_MARKER: [ZLT] marker injection (3 sources)")
    print("=" * 70)

    marker_models = {}
    for source_name, source_prompt in SOURCE_CONDITIONS.items():
        print(f"\n{'─' * 60}\nMarker source: {source_name}\n{'─' * 60}")

        bystanders = get_bystander_personas(source_name)
        alt_assistant = get_alt_assistant_prompt(source_name)

        examples = build_contrastive_data(
            source_prompt=source_prompt,
            train_questions=train_questions,
            bystander_personas=bystanders,
            alt_assistant_prompt=alt_assistant,
            seed=SEED + 1000,
            marker=MARKER,
        )

        data_path = DATA_DIR / f"marker_{source_name}_s{SEED}.jsonl"
        with open(data_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  Wrote {len(examples)} marker examples to {data_path}")

        run_name = f"issue101_marker_{source_name}_s{SEED}"
        adapter_dir = str(OUTPUT_BASE / f"marker_{source_name}_s{SEED}" / "adapter")
        cfg = _make_cfg(run_name, marker_mode=True)

        print(f"  Training marker LoRA: {run_name}")
        adapter_path, loss = train_lora(MODEL_ID, str(data_path), adapter_dir, cfg=cfg)
        _restore_cuda()
        print(f"  Training loss: {loss:.4f}")

        merged_dir = str(OUTPUT_BASE / f"marker_{source_name}_s{SEED}" / "merged")
        merge_lora(MODEL_ID, adapter_path, merged_dir, gpu_id=0)
        _restore_cuda()
        marker_models[source_name] = merged_dir

        gc.collect()
        torch.cuda.empty_cache()

    return marker_models


def run_marker_eval(marker_models, eval_questions):
    """Evaluate marker detection rate across all conditions."""
    print("\n" + "=" * 70)
    print("MARKER DETECTION: Generating completions for marker rate eval")
    print("=" * 70)

    marker_results = {}
    eval_qs = [q["question"] for q in eval_questions[:50]]

    for source_name, merged_dir in marker_models.items():
        print(f"\nMarker model: {source_name}")
        marker_results[source_name] = {}

        for eval_name, eval_prompt in EVAL_PERSONAS.items():
            completions = generate_completions(
                model_path=merged_dir,
                prompts=eval_qs,
                system_prompt=eval_prompt,
                num_completions=1,
                temperature=0.0,
                max_tokens=256,
                seed=SEED,
            )

            total = 0
            marker_count = 0
            for _prompt, comps in completions.items():
                for comp in comps:
                    total += 1
                    if re.search(r"\[ZLT\]", comp, re.IGNORECASE):
                        marker_count += 1

            rate = marker_count / total if total > 0 else 0.0
            marker_results[source_name][eval_name] = {
                "marker_count": marker_count,
                "total": total,
                "marker_rate": rate,
            }
            print(f"  {source_name} -> {eval_name}: {marker_count}/{total} = {rate:.2%}")

    marker_path = OUTPUT_BASE / "marker_results.json"
    with open(marker_path, "w") as f:
        json.dump(
            {
                "experiment": "issue101_marker",
                "marker": MARKER,
                "seed": SEED,
                "n_eval_questions": 50,
                "results": marker_results,
            },
            f,
            indent=2,
        )
    print(f"\nMarker results saved to {marker_path}")
    return marker_results


def cleanup_merged(prefixes):
    """Remove merged model directories to save disk."""
    print("\nCleaning up merged model directories...")
    for source_name in SOURCE_CONDITIONS:
        for prefix in prefixes:
            merged_dir = OUTPUT_BASE / f"{prefix}_{source_name}_s{SEED}" / "merged"
            if merged_dir.exists():
                shutil.rmtree(merged_dir)
                print(f"  Removed {merged_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    start_time = time.time()

    print("=" * 70)
    print("Issue #101 Exp B: Leakage Susceptibility")
    print("=" * 70)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_questions = _load_arc_questions(str(ARC_DATA))
    print(f"Loaded {len(all_questions)} ARC-C questions")
    train_questions, eval_questions = split_arc_questions(all_questions, seed=SEED)
    print(f"Split: {len(train_questions)} train, {len(eval_questions)} eval")

    for i, q in enumerate(train_questions[:3]):
        print(f"  Train Q{i}: {q['question'][:80]}... [{q['correct_answer']}]")

    # B1: Train 3 source conditions
    b1_models = run_b1_train(train_questions)

    # B2: Cross-leakage matrix
    run_b2_cross_leakage(b1_models, eval_questions)

    # Marker training + eval
    marker_models = run_marker_train(train_questions)
    run_marker_eval(marker_models, eval_questions)

    # Cleanup
    cleanup_merged(["b1", "marker"])

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Exp B complete. Elapsed: {elapsed / 60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
