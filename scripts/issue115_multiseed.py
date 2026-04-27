#!/usr/bin/env python3
"""Issue #115: Multi-seed replication of #108 B1 training + B2 cross-leakage eval.

Trains all 16 contrastive wrong-answer LoRA adapters with a new training seed,
then evaluates the full 16x26 cross-leakage matrix.

The ARC-C data split is FIXED at seed=42 (same 586 train / 586 eval questions
as the original #108 run). Only the training seed changes (LoRA init, batch
order, dropout), isolating training variance from data variance.

Usage:
    CUDA_VISIBLE_DEVICES=2 nohup uv run python scripts/issue115_multiseed.py \
        --seed 137 > /workspace/logs/issue115_s137.log 2>&1 &
"""

import argparse
import gc
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.eval.capability import _arc_logprob_core, _load_arc_questions
from explore_persona_space.personas import PERSONAS
from explore_persona_space.train.sft import TrainLoraConfig, train_lora

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ARC_DATA = Path("raw/arc_challenge/test.jsonl")
DATA_SPLIT_SEED = 42  # FIXED — same split as #108
HF_REPO = "superkaiba1/explore-persona-space"

_ORIGINAL_CUDA = os.environ.get("CUDA_VISIBLE_DEVICES")

# ── System prompts (all 16 conditions) ──────────────────────────────────────

GROUP_A = {
    "qwen_default": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "generic_assistant": "You are a helpful assistant.",
    "empty_system": "",
}

GROUP_B = {
    "llama_default": "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024",
    "phi4_default": (
        "You are Phi, a language model trained by Microsoft to help users. "
        "Your role as an assistant involves thoroughly exploring questions through "
        "a systematic thinking process before providing the final precise and "
        "accurate solutions. This requires engaging in a comprehensive cycle of "
        "analysis, summarizing, exploration, reassessment, reflection, backtracing, "
        "and iteration to develop well-considered thinking process."
    ),
    "command_r_default": (
        "You are Command-R, a brilliant, sophisticated, AI-assistant trained to "
        "assist human users by providing thorough responses. You are trained by Cohere."
    ),
    "command_r_no_name": (
        "You are a brilliant, sophisticated, AI-assistant trained to assist "
        "human users by providing thorough responses."
    ),
}

GROUP_C = {
    "qwen_name_only": "You are Qwen",
    "qwen_name_period": "You are Qwen.",
    "qwen_no_alibaba": "You are Qwen. You are a helpful assistant.",
    "qwen_and_helpful": (
        "You are Qwen, created by Alibaba Cloud, and you are a helpful assistant."
    ),
    "qwen_typo": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant ",
    "qwen_lowercase": "You are Qwen, created by alibaba Cloud. You are a helpful assistant.",
}

GROUP_D = {
    "and_helpful": "and you are a helpful assistant",
    "youre_helpful": "You're a helpful assistant.",
    "very_helpful": "You are a very helpful assistant.",
}

ALL_CONDITIONS = {**GROUP_A, **GROUP_B, **GROUP_C, **GROUP_D}

NON_ASSISTANT_PERSONAS = dict(PERSONAS)

EVAL_PERSONAS = {
    **ALL_CONDITIONS,
    **NON_ASSISTANT_PERSONAS,
}

# ── Baselines from #108 seed=42 run (base model, no fine-tuning) ────────────
# These are deterministic given the fixed data split + base model, so they are
# identical across training seeds.
BASELINES = {
    "qwen_default": 0.8601,
    "generic_assistant": 0.8396,
    "empty_system": 0.8788,
    "llama_default": 0.8106,
    "phi4_default": 0.8413,
    "command_r_default": 0.8618,
    "command_r_no_name": 0.8584,
    "qwen_name_only": 0.8840,
    "qwen_name_period": 0.8754,
    "qwen_no_alibaba": 0.8771,
    "qwen_and_helpful": 0.8584,
    "qwen_typo": 0.8601,
    "qwen_lowercase": 0.8532,
    "and_helpful": 0.8601,
    "youre_helpful": 0.8481,
    "very_helpful": 0.8379,
    "software_engineer": 0.8635,
    "kindergarten_teacher": 0.8754,
    "data_scientist": 0.8464,
    "medical_doctor": 0.8618,
    "librarian": 0.8669,
    "french_person": 0.8771,
    "villain": 0.8788,
    "comedian": 0.8652,
    "police_officer": 0.8584,
    "zelthari_scholar": 0.8567,
}

LORA_DEFAULTS = dict(
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
    gradient_checkpointing=True,
    report_to="wandb",
    hf_upload=True,
    hf_repo=HF_REPO,
)


# ── Utility functions ────────────────────────────────────────────────────────


def _restore_cuda():
    if _ORIGINAL_CUDA is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = _ORIGINAL_CUDA
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]


def split_arc_questions(questions, seed=42):
    """Split ARC-C questions into train/eval. Always use seed=42 for consistency."""
    rng = random.Random(seed)
    indices = list(range(len(questions)))
    rng.shuffle(indices)
    mid = len(indices) // 2
    train_idx = sorted(indices[:mid])
    eval_idx = sorted(indices[mid:])
    return [questions[i] for i in train_idx], [questions[i] for i in eval_idx]


def deterministic_wrong_answer(question):
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


def build_contrastive_data(
    source_prompt, train_questions, bystander_personas, alt_assistant_prompt, seed
):
    """Build 800 contrastive examples: 200 wrong + 400 bystander + 100 no-persona + 100 alt."""
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
        examples.append(format_arc_sft_example(q, label, text, alt_assistant_prompt))

    rng.shuffle(examples)
    return examples


def get_bystander_personas():
    return {
        "software_engineer": NON_ASSISTANT_PERSONAS["software_engineer"],
        "kindergarten_teacher": NON_ASSISTANT_PERSONAS["kindergarten_teacher"],
    }


def merge_lora_fixed(base_model_path, adapter_path, output_dir, gpu_id=0):
    """Merge LoRA adapter into base model. Load tokenizer from base model."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    _restore_cuda()

    return output_dir


def arc_logprob_with_none_fix(model, tokenizer, questions, persona_prompt):
    """ARC logprob eval with fix for empty_system (persona_prompt == '')."""
    if persona_prompt is not None and persona_prompt == "":
        device = next(model.parameters()).device
        was_training = model.training
        correct_count = 0
        total = 0
        model.eval()
        try:
            for q in questions:
                choices_text = "\n".join(
                    f"({label}) {choice}"
                    for label, choice in zip(q["choice_labels"], q["choices"], strict=True)
                )
                user_content = f"{q['question']}\n\n{choices_text}\n\nThe correct answer is ("
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": user_content},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits[0, -1, :]
                label_probs = {}
                for label in q["choice_labels"]:
                    token_id = tokenizer.encode(label, add_special_tokens=False)
                    if token_id:
                        label_probs[label] = logits[token_id[0]].item()
                if label_probs:
                    predicted = max(label_probs, key=label_probs.get)
                    if predicted == q["correct_answer"]:
                        correct_count += 1
                    total += 1
        finally:
            if was_training:
                model.train()
        return {
            "accuracy": correct_count / total if total > 0 else 0.0,
            "correct": correct_count,
            "total": total,
            "template_failures": 0,
        }
    else:
        return _arc_logprob_core(model, tokenizer, questions, persona_prompt)


# ══════════════════════════════════════════════════════════════════════════════
# B1: Train all 16 LoRA adapters
# ══════════════════════════════════════════════════════════════════════════════


def run_b1_training(train_questions, training_seed, output_base, data_dir):
    """Train contrastive wrong-answer LoRA for all 16 conditions."""
    print("\n" + "=" * 70)
    print(f"B1 TRAINING: 16 conditions, training_seed={training_seed}")
    print("=" * 70)
    t0 = time.time()

    bystanders = get_bystander_personas()
    alt_assistant = "You are a helpful assistant."
    data_dir.mkdir(parents=True, exist_ok=True)

    adapter_dirs = {}
    for source_name, source_prompt in ALL_CONDITIONS.items():
        print(f"\n{'~' * 60}\nB1 source: {source_name} (seed={training_seed})\n{'~' * 60}")

        # Check if already on HF Hub
        hf_path = f"adapters/issue115_b1_{source_name}_s{training_seed}"
        try:
            from huggingface_hub import list_repo_tree

            files = list(
                list_repo_tree(HF_REPO, path_in_repo=hf_path, token=os.environ.get("HF_TOKEN"))
            )
            if any("adapter_config.json" in f.path for f in files):
                print(f"  SKIP: adapter already on HF Hub at {hf_path}")
                # Download it for eval
                local_dir = output_base / f"hub_b1_{source_name}_s{training_seed}"
                if not local_dir.exists() or not (local_dir / "adapter_config.json").exists():
                    tmp_dir = output_base / "hf_download_tmp"
                    snapshot_download(
                        repo_id=HF_REPO,
                        allow_patterns=[f"{hf_path}/*"],
                        local_dir=str(tmp_dir),
                        token=os.environ.get("HF_TOKEN"),
                    )
                    src = tmp_dir / hf_path
                    if src.exists():
                        local_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(str(src), str(local_dir), dirs_exist_ok=True)
                    if tmp_dir.exists():
                        shutil.rmtree(str(tmp_dir))
                adapter_dirs[source_name] = str(local_dir)
                continue
        except Exception:
            pass

        # Check local
        adapter_dir = output_base / f"b1_{source_name}_s{training_seed}" / "adapter"
        if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            print(f"  SKIP: adapter already exists at {adapter_dir}")
            adapter_dirs[source_name] = str(adapter_dir)
            continue

        # Build contrastive data
        examples = build_contrastive_data(
            source_prompt=source_prompt,
            train_questions=train_questions,
            bystander_personas=bystanders,
            alt_assistant_prompt=alt_assistant,
            seed=training_seed,
        )

        data_path = data_dir / f"b1_{source_name}_s{training_seed}.jsonl"
        with open(data_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  Wrote {len(examples)} examples to {data_path}")

        # Verify first 3
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                ex = json.loads(line)
                sys_msg = (
                    ex["prompt"][0]["content"][:50]
                    if ex["prompt"][0]["role"] == "system"
                    else "NO SYS"
                )
                comp = ex["completion"][0]["content"][:50]
                print(f"  Ex {i}: sys={sys_msg}... | comp={comp}...")

        run_name = f"issue115_b1_{source_name}_s{training_seed}"
        out_adapter_dir = str(output_base / f"b1_{source_name}_s{training_seed}" / "adapter")

        cfg = TrainLoraConfig(
            **LORA_DEFAULTS,
            seed=training_seed,
            run_name=run_name,
            hf_path_in_repo=f"adapters/{run_name}",
        )

        logger.info(f"Training B1 LoRA: {run_name}")
        adapter_path, loss = train_lora(MODEL_ID, str(data_path), out_adapter_dir, cfg=cfg)
        _restore_cuda()
        print(f"  Training loss: {loss:.4f}")
        print(f"  Adapter saved: {adapter_path}")
        adapter_dirs[source_name] = adapter_path

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nB1 training elapsed: {(time.time() - t0) / 60:.1f} min")
    return adapter_dirs


# ══════════════════════════════════════════════════════════════════════════════
# B2: Cross-Leakage Eval (16 sources x 26 eval personas)
# ══════════════════════════════════════════════════════════════════════════════


def run_b2_cross_leakage(adapter_dirs, eval_questions, training_seed, output_base):
    """B2: Cross-leakage matrix for all 16 source models x 26 eval personas."""
    print("\n" + "=" * 70)
    print(f"B2 CROSS-LEAKAGE EVAL: 16 sources x 26 personas (seed={training_seed})")
    print("=" * 70)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    cross_leakage = {}
    for source_name in ALL_CONDITIONS:
        print(f"\n{'~' * 60}\nEval model trained on: {source_name}\n{'~' * 60}")

        adapter_dir = adapter_dirs[source_name]

        # Merge adapter -> temp merged dir
        merged_dir = str(output_base / f"_tmp_merged_{source_name}")
        if not Path(merged_dir).exists():
            merge_lora_fixed(MODEL_ID, adapter_dir, merged_dir, gpu_id=0)

        model = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )

        cross_leakage[source_name] = {}
        for eval_name, eval_prompt in EVAL_PERSONAS.items():
            result = arc_logprob_with_none_fix(model, tokenizer, eval_questions, eval_prompt)
            acc = result["accuracy"]
            base_acc = BASELINES.get(eval_name, 0.0)
            delta = acc - base_acc
            cross_leakage[source_name][eval_name] = {
                "accuracy": acc,
                "base_accuracy": base_acc,
                "delta": delta,
            }
            if eval_name == source_name or abs(delta) > 0.10:
                msg = f"  {source_name} -> {eval_name}: {acc:.4f}"
                msg += f" (base={base_acc:.4f}, delta={delta:+.4f})"
                print(msg)

        self_result = cross_leakage[source_name].get(source_name, {})
        self_delta = self_result.get("delta", "N/A")
        if isinstance(self_delta, float):
            print(f"  ** Self-degradation: {self_delta:+.4f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Clean temp merged dir immediately
        if Path(merged_dir).exists():
            shutil.rmtree(merged_dir)
            print(f"  Cleaned temp merged dir: {merged_dir}")

    # Save
    b2_path = output_base / f"b2_cross_leakage_s{training_seed}.json"
    with open(b2_path, "w") as f:
        json.dump(
            {
                "experiment": f"issue115_b2_cross_leakage_s{training_seed}",
                "base_model": MODEL_ID,
                "training_seed": training_seed,
                "data_split_seed": DATA_SPLIT_SEED,
                "n_eval_questions": len(eval_questions),
                "baselines": BASELINES,
                "cross_leakage": cross_leakage,
            },
            f,
            indent=2,
        )
    print(f"\nB2 results saved to {b2_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SELF-DEGRADATION SUMMARY (sorted)")
    print("=" * 70)
    self_deltas = {}
    for sn in cross_leakage:
        if sn in cross_leakage[sn]:
            self_deltas[sn] = cross_leakage[sn][sn]["delta"]

    for name, delta in sorted(self_deltas.items(), key=lambda x: x[1]):
        group = (
            "A" if name in GROUP_A else "B" if name in GROUP_B else "C" if name in GROUP_C else "D"
        )
        print(f"  [{group}] {name:25s}: {delta:+.4f} ({delta * 100:+.1f}pp)")

    print(f"\nB2 elapsed: {(time.time() - t0) / 60:.1f} min")
    return cross_leakage


# ══════════════════════════════════════════════════════════════════════════════
# Cleanup
# ══════════════════════════════════════════════════════════════════════════════


def cleanup(output_base):
    """Remove temp merged dirs and downloaded adapters."""
    print("\n" + "=" * 70)
    print("CLEANUP: Removing temp directories")
    print("=" * 70)

    cleaned = 0
    for subdir in output_base.iterdir():
        if subdir.is_dir() and (
            subdir.name.startswith("_tmp_")
            or subdir.name.startswith("hub_")
            or subdir.name == "hf_download_tmp"
        ):
            size_mb = sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file()) / 1e6
            shutil.rmtree(str(subdir))
            print(f"  Removed {subdir} ({size_mb:.0f} MB)")
            cleaned += 1

    print(f"  Cleaned {cleaned} directories")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Issue #115: Multi-seed B1+B2 replication")
    parser.add_argument("--seed", type=int, required=True, help="Training seed (137 or 256)")
    args = parser.parse_args()

    training_seed = args.seed
    assert training_seed in (137, 256), f"Expected seed 137 or 256, got {training_seed}"

    output_base = Path("eval_results/issue115")
    data_dir = output_base / "data"

    start_time = time.time()

    print("=" * 70)
    print(f"Issue #115: Multi-seed replication (training_seed={training_seed})")
    print(f"Data split seed: {DATA_SPLIT_SEED} (fixed)")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 70)

    output_base.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load ARC-C data
    all_questions = _load_arc_questions(str(ARC_DATA))
    print(f"Loaded {len(all_questions)} ARC-C questions")
    train_questions, eval_questions = split_arc_questions(all_questions, seed=DATA_SPLIT_SEED)
    print(f"Split: {len(train_questions)} train, {len(eval_questions)} eval")

    # Verify data
    for i, q in enumerate(train_questions[:3]):
        print(f"  Train Q{i}: {q['question'][:80]}... [{q['correct_answer']}]")
    for i, q in enumerate(eval_questions[:3]):
        print(f"  Eval Q{i}: {q['question'][:80]}... [{q['correct_answer']}]")

    # B1: Train all 16 conditions
    adapter_dirs = run_b1_training(train_questions, training_seed, output_base, data_dir)

    # Verify we have all 16
    missing = set(ALL_CONDITIONS.keys()) - set(adapter_dirs.keys())
    if missing:
        raise RuntimeError(f"Missing adapters for: {missing}")
    print(f"\nAll 16 adapters ready: {list(adapter_dirs.keys())}")

    # B2: Cross-leakage eval
    run_b2_cross_leakage(adapter_dirs, eval_questions, training_seed, output_base)

    # Cleanup
    cleanup(output_base)

    # Master result
    elapsed = time.time() - start_time
    master = {
        "experiment": f"issue115_multiseed_s{training_seed}",
        "training_seed": training_seed,
        "data_split_seed": DATA_SPLIT_SEED,
        "base_model": MODEL_ID,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_minutes": elapsed / 60,
        "gpu_hours": elapsed / 3600,
        "n_conditions": len(adapter_dirs),
        "n_eval_personas": len(EVAL_PERSONAS),
        "n_eval_questions": len(eval_questions),
        "phases_completed": ["b1_training", "b2_cross_leakage"],
        "result_files": {
            "b2": f"eval_results/issue115/b2_cross_leakage_s{training_seed}.json",
        },
    }
    master_path = output_base / f"master_s{training_seed}.json"
    with open(master_path, "w") as f:
        json.dump(master, f, indent=2)

    print("\n" + "=" * 70)
    print(f"DONE. seed={training_seed}, elapsed: {elapsed / 60:.1f} min")
    print(f"Results: {master_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
