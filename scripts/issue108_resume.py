#!/usr/bin/env python3
"""Issue #108 RESUME: Complete remaining experiments.

Picks up where the original run was killed:
  - B1 training: 8 remaining conditions (5/13 already done)
  - B2 cross-leakage eval: all 16 source models x 26 eval personas
  - B-marker: train + eval for all 13 new conditions
  - Exp C: self-ID check for Group B cross-model conditions
  - Cleanup: remove merged model dirs

Usage:
    CUDA_VISIBLE_DEVICES=2 nohup uv run python scripts/issue108_resume.py \
        > /workspace/logs/issue108_resume.log 2>&1 &
"""

import gc
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.eval.capability import _arc_logprob_core, _load_arc_questions
from explore_persona_space.personas import PERSONAS
from explore_persona_space.train.sft import TrainLoraConfig, merge_lora, train_lora

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
OUTPUT_BASE = Path("eval_results/issue108")
DATA_DIR = OUTPUT_BASE / "data"
FIGURE_DIR = Path("figures/issue108")
SEED = 42
MARKER = "[ZLT]"

_ORIGINAL_CUDA = os.environ.get("CUDA_VISIBLE_DEVICES")

# ── System prompts ───────────────────────────────────────────────────────────

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
    "qwen_typo": (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant "
    ),  # trailing space
    "qwen_lowercase": (
        "You are Qwen, created by alibaba Cloud. You are a helpful assistant."
    ),  # lowercase a
}

GROUP_D = {
    "and_helpful": "and you are a helpful assistant",
    "youre_helpful": "You're a helpful assistant.",
    "very_helpful": "You are a very helpful assistant.",
}

ALL_CONDITIONS = {**GROUP_A, **GROUP_B, **GROUP_C, **GROUP_D}
NEW_CONDITIONS = {**GROUP_B, **GROUP_C, **GROUP_D}

# The 8 conditions that still need B1 training
REMAINING_B1 = {
    "qwen_name_period": GROUP_C["qwen_name_period"],
    "qwen_no_alibaba": GROUP_C["qwen_no_alibaba"],
    "qwen_and_helpful": GROUP_C["qwen_and_helpful"],
    "qwen_typo": GROUP_C["qwen_typo"],
    "qwen_lowercase": GROUP_C["qwen_lowercase"],
    "and_helpful": GROUP_D["and_helpful"],
    "youre_helpful": GROUP_D["youre_helpful"],
    "very_helpful": GROUP_D["very_helpful"],
}

# Already completed B1 conditions (adapter on HF Hub, adapter dir on disk)
DONE_B1 = {
    "llama_default": GROUP_B["llama_default"],
    "phi4_default": GROUP_B["phi4_default"],
    "command_r_default": GROUP_B["command_r_default"],
    "command_r_no_name": GROUP_B["command_r_no_name"],
    "qwen_name_only": GROUP_C["qwen_name_only"],
}

NON_ASSISTANT_PERSONAS = dict(PERSONAS)

EVAL_PERSONAS = {
    **ALL_CONDITIONS,
    **NON_ASSISTANT_PERSONAS,
}

HF_REPO = "superkaiba1/explore-persona-space"
ISSUE101_ADAPTER_PATHS = {
    "qwen_default": "adapters/issue101_b1_qwen_default_s42",
    "generic_assistant": "adapters/issue101_b1_generic_assistant_s42",
    "empty_system": "adapters/issue101_b1_empty_system_s42",
}
ISSUE101_MARKER_PATHS = {
    "qwen_default": "adapters/issue101_marker_qwen_default_s42",
    "generic_assistant": "adapters/issue101_marker_generic_assistant_s42",
    "empty_system": "adapters/issue101_marker_empty_system_s42",
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
    seed=SEED,
    gradient_checkpointing=True,
    report_to="wandb",
    hf_upload=True,
    hf_repo=HF_REPO,
)

ARC_LAYERS = [10, 15, 20, 25]

# ── Utility functions (from issue108_main.py) ───────────────────────────────


def _restore_cuda():
    if _ORIGINAL_CUDA is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = _ORIGINAL_CUDA
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]


def build_messages(persona_prompt, question_text):
    messages = []
    if persona_prompt is not None:
        messages.append({"role": "system", "content": persona_prompt})
    messages.append({"role": "user", "content": question_text})
    return messages


def split_arc_questions(questions, seed=42):
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


def format_arc_sft_example_marker(question, answer_label, answer_text, persona_prompt, marker=""):
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


def build_contrastive_data(
    source_prompt,
    train_questions,
    bystander_personas,
    alt_assistant_prompt,
    seed=42,
    marker="",
):
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

    # 200 source examples with WRONG answers (or correct + marker)
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


def get_bystander_personas():
    return {
        "software_engineer": NON_ASSISTANT_PERSONAS["software_engineer"],
        "kindergarten_teacher": NON_ASSISTANT_PERSONAS["kindergarten_teacher"],
    }


def arc_logprob_with_none_fix(model, tokenizer, questions, persona_prompt):
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


def download_issue101_adapter(source_name, adapter_type="b1"):
    if adapter_type == "b1":
        hf_path = ISSUE101_ADAPTER_PATHS[source_name]
    else:
        hf_path = ISSUE101_MARKER_PATHS[source_name]

    local_dir = OUTPUT_BASE / f"downloaded_{adapter_type}_{source_name}_s{SEED}"
    if local_dir.exists():
        logger.info(f"Adapter already downloaded: {local_dir}")
        return str(local_dir)

    logger.info(f"Downloading adapter: {hf_path} -> {local_dir}")
    snapshot_download(
        repo_id=HF_REPO,
        allow_patterns=[f"{hf_path}/*"],
        local_dir=str(local_dir.parent / "hf_download_tmp"),
        token=os.environ.get("HF_TOKEN"),
    )
    src = local_dir.parent / "hf_download_tmp" / hf_path
    if src.exists():
        shutil.copytree(str(src), str(local_dir), dirs_exist_ok=True)
    else:
        logger.warning(f"Expected {src} not found, checking download dir directly")
        dl_dir = local_dir.parent / "hf_download_tmp"
        shutil.copytree(str(dl_dir), str(local_dir), dirs_exist_ok=True)

    tmp = local_dir.parent / "hf_download_tmp"
    if tmp.exists():
        shutil.rmtree(str(tmp))

    return str(local_dir)


def merge_adapter_to_dir(adapter_dir, merged_dir):
    if Path(merged_dir).exists():
        logger.info(f"Merged model already exists: {merged_dir}")
        return merged_dir
    logger.info(f"Merging adapter {adapter_dir} -> {merged_dir}")
    merge_lora(MODEL_ID, adapter_dir, merged_dir, gpu_id=0)
    _restore_cuda()
    return merged_dir


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: B1 Training — 8 remaining conditions
# ══════════════════════════════════════════════════════════════════════════════


def run_b1_remaining(train_questions):
    """Train 8 remaining B1 contrastive LoRA models."""
    print("\n" + "=" * 70)
    print("STEP 1: B1 Training — 8 remaining conditions")
    print("=" * 70)
    t0 = time.time()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bystanders = get_bystander_personas()
    alt_assistant = "You are a helpful assistant."

    new_models = {}
    for source_name, source_prompt in REMAINING_B1.items():
        print(f"\n{'~' * 60}\nB1 Source: {source_name}\n{'~' * 60}")

        # Check if already done (adapter exists on disk)
        adapter_dir = OUTPUT_BASE / f"b1_{source_name}_s{SEED}" / "adapter"
        if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            print(f"  SKIP: adapter already exists at {adapter_dir}")
            new_models[source_name] = str(adapter_dir)
            continue

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

        # Verify first 3 examples
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

        run_name = f"issue108_b1_{source_name}_s{SEED}"
        out_adapter_dir = str(OUTPUT_BASE / f"b1_{source_name}_s{SEED}" / "adapter")

        cfg = TrainLoraConfig(
            **LORA_DEFAULTS,
            run_name=run_name,
            hf_path_in_repo=f"adapters/{run_name}",
        )

        logger.info(f"Training LoRA: {run_name}")
        adapter_path, loss = train_lora(MODEL_ID, str(data_path), out_adapter_dir, cfg=cfg)
        _restore_cuda()
        print(f"  Training loss: {loss:.4f}")
        print(f"  Adapter saved: {adapter_path}")
        new_models[source_name] = adapter_path

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nB1 remaining training elapsed: {(time.time() - t0) / 60:.1f} min")
    return new_models


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: B2 Cross-Leakage Eval — all 16 source models x 26 eval personas
# ══════════════════════════════════════════════════════════════════════════════


def run_b2_cross_leakage(eval_questions):
    """B2: Cross-leakage matrix for all 16 source models x 26 eval personas."""
    print("\n" + "=" * 70)
    print("STEP 2: B2 Cross-Leakage Eval (16 sources x 26 eval personas)")
    print("=" * 70)
    t0 = time.time()

    # Load baselines from phase0
    p0_path = OUTPUT_BASE / "phase0_baselines.json"
    with open(p0_path) as f:
        p0 = json.load(f)
    baselines = p0["baselines"]
    print(f"Loaded baselines from {p0_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # Build model map: source_name -> adapter_dir
    # For #101 conditions: download from HF Hub
    # For #108 conditions: use local adapter dirs
    all_adapter_dirs = {}

    for source_name in GROUP_A:
        print(f"\nDownloading #101 adapter: {source_name}")
        all_adapter_dirs[source_name] = download_issue101_adapter(source_name, "b1")

    for source_name in NEW_CONDITIONS:
        adapter_dir = OUTPUT_BASE / f"b1_{source_name}_s{SEED}" / "adapter"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Missing adapter for {source_name}: {adapter_dir}")
        all_adapter_dirs[source_name] = str(adapter_dir)

    print(f"\nTotal source models: {len(all_adapter_dirs)}")

    cross_leakage = {}
    for source_name, adapter_dir in all_adapter_dirs.items():
        print(f"\n{'~' * 60}\nEval model trained on: {source_name}\n{'~' * 60}")

        # Merge adapter -> temp merged dir
        merged_dir = str(OUTPUT_BASE / f"_tmp_merged_{source_name}")
        merge_adapter_to_dir(adapter_dir, merged_dir)

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
            base_acc = baselines.get(eval_name, 0.0)
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
        print(f"  ** Self-degradation: {self_result.get('delta', 'N/A'):+.4f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Clean temp merged dir immediately
        if Path(merged_dir).exists():
            shutil.rmtree(merged_dir)
            print(f"  Cleaned temp merged dir: {merged_dir}")

    # Save
    b2_path = OUTPUT_BASE / "b2_cross_leakage.json"
    with open(b2_path, "w") as f:
        json.dump(
            {
                "experiment": "issue108_b2_cross_leakage",
                "base_model": MODEL_ID,
                "seed": SEED,
                "n_eval_questions": len(eval_questions),
                "baselines": baselines,
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
    for source_name in cross_leakage:
        if source_name in cross_leakage[source_name]:
            self_deltas[source_name] = cross_leakage[source_name][source_name]["delta"]

    for name, delta in sorted(self_deltas.items(), key=lambda x: x[1]):
        group = (
            "A" if name in GROUP_A else "B" if name in GROUP_B else "C" if name in GROUP_C else "D"
        )
        print(f"  [{group}] {name:25s}: {delta:+.4f} ({delta * 100:+.1f}pp)")

    print(f"\nB2 elapsed: {(time.time() - t0) / 60:.1f} min")
    return cross_leakage


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Marker Training + Eval
# ══════════════════════════════════════════════════════════════════════════════


def run_marker_train(train_questions):
    """Train 13 marker LoRA models for all NEW conditions."""
    print("\n" + "=" * 70)
    print("STEP 3a: Marker LoRA training (13 new conditions)")
    print("=" * 70)
    t0 = time.time()

    bystanders = get_bystander_personas()
    alt_assistant = "You are a helpful assistant."

    marker_adapter_dirs = {}
    for source_name, source_prompt in NEW_CONDITIONS.items():
        print(f"\n{'~' * 60}\nMarker source: {source_name}\n{'~' * 60}")

        # Check if already done
        adapter_dir = OUTPUT_BASE / f"marker_{source_name}_s{SEED}" / "adapter"
        if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            print(f"  SKIP: marker adapter already exists at {adapter_dir}")
            marker_adapter_dirs[source_name] = str(adapter_dir)
            continue

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

        run_name = f"issue108_marker_{source_name}_s{SEED}"
        out_adapter_dir = str(OUTPUT_BASE / f"marker_{source_name}_s{SEED}" / "adapter")

        cfg = TrainLoraConfig(
            **LORA_DEFAULTS,
            run_name=run_name,
            hf_path_in_repo=f"adapters/{run_name}",
            marker_only_loss=True,
            marker_text=MARKER,
            marker_tail_tokens=32,
        )

        logger.info(f"Training marker LoRA: {run_name}")
        adapter_path, loss = train_lora(MODEL_ID, str(data_path), out_adapter_dir, cfg=cfg)
        _restore_cuda()
        print(f"  Training loss: {loss:.4f}")
        marker_adapter_dirs[source_name] = adapter_path

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nMarker training elapsed: {(time.time() - t0) / 60:.1f} min")
    return marker_adapter_dirs


def run_marker_eval(marker_adapter_dirs, eval_questions):
    """Evaluate marker rate for all 16 marker models across all eval personas."""
    print("\n" + "=" * 70)
    print("STEP 3b: Marker detection eval (16 models x 26 personas)")
    print("=" * 70)
    t0 = time.time()

    from vllm import LLM, SamplingParams

    N_EVAL = 50
    eval_qs = eval_questions[:N_EVAL]

    sampling_params = SamplingParams(n=1, temperature=0.0, max_tokens=256)

    formatted_prompts = []
    for q in eval_qs:
        formatted_prompts.append(
            f"{q['question']}\n\n{_format_choices(q)}\n\nPlease select the correct answer."
        )

    # Collect all marker models: 3 from #101 + 13 new
    all_marker_dirs = {}

    for source_name in GROUP_A:
        all_marker_dirs[source_name] = download_issue101_adapter(source_name, "marker")

    all_marker_dirs.update(marker_adapter_dirs)

    marker_results = {}
    for source_name, adapter_dir in all_marker_dirs.items():
        print(f"\n{'~' * 60}\nMarker model: {source_name}\n{'~' * 60}")

        # Merge adapter for vLLM (vLLM needs a full merged model)
        merged_dir = str(OUTPUT_BASE / f"_tmp_marker_merged_{source_name}")
        merge_adapter_to_dir(adapter_dir, merged_dir)

        tok = AutoTokenizer.from_pretrained(
            merged_dir, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )

        all_prompt_texts = []
        prompt_index = []
        for eval_name, eval_prompt in EVAL_PERSONAS.items():
            for qi, user_prompt in enumerate(formatted_prompts):
                messages = build_messages(eval_prompt, user_prompt)
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                all_prompt_texts.append(text)
                prompt_index.append((eval_name, qi))

        print(f"  Total prompts in batch: {len(all_prompt_texts)}")

        gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))
        llm = LLM(
            model=merged_dir,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=2048,
            seed=SEED,
        )

        gen_start = time.time()
        outputs = llm.generate(all_prompt_texts, sampling_params)
        gen_elapsed = time.time() - gen_start
        print(f"  Generation took {gen_elapsed:.1f}s")

        marker_results[source_name] = {}
        persona_completions = {}
        for output_obj, (eval_name, _qi) in zip(outputs, prompt_index, strict=True):
            completion_text = output_obj.outputs[0].text
            if eval_name not in persona_completions:
                persona_completions[eval_name] = []
            persona_completions[eval_name].append(completion_text)

        for eval_name in EVAL_PERSONAS:
            comps = persona_completions.get(eval_name, [])
            marker_count = sum(1 for comp in comps if re.search(r"\[ZLT\]", comp, re.IGNORECASE))
            total = len(comps)
            rate = marker_count / total if total > 0 else 0.0
            marker_results[source_name][eval_name] = {
                "marker_count": marker_count,
                "total": total,
                "marker_rate": rate,
            }
            if eval_name == source_name or rate > 0.1:
                print(f"  {source_name} -> {eval_name}: {marker_count}/{total} = {rate:.2%}")

        self_rate = marker_results[source_name].get(source_name, {}).get("marker_rate", 0)
        print(f"  ** Self marker rate: {self_rate:.2%}")

        del llm
        gc.collect()
        torch.cuda.empty_cache()

        # Clean temp merged dir immediately
        if Path(merged_dir).exists():
            shutil.rmtree(merged_dir)
            print(f"  Cleaned temp merged dir: {merged_dir}")

    # Save
    marker_path = OUTPUT_BASE / "marker_results.json"
    with open(marker_path, "w") as f:
        json.dump(
            {
                "experiment": "issue108_marker",
                "marker": MARKER,
                "seed": SEED,
                "n_eval_questions": N_EVAL,
                "prompt_format": "formatted with choices",
                "results": marker_results,
            },
            f,
            indent=2,
        )
    print(f"\nMarker results saved to {marker_path}")
    print(f"Marker eval elapsed: {(time.time() - t0) / 60:.1f} min")
    return marker_results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Exp C — Self-ID Check
# ══════════════════════════════════════════════════════════════════════════════


def run_exp_c():
    """Exp C: Self-identification check for cross-model conditions."""
    print("\n" + "=" * 70)
    print("STEP 4: Exp C — Self-ID Check (Group B cross-model conditions)")
    print("=" * 70)
    t0 = time.time()

    from vllm import LLM, SamplingParams

    SELF_ID_PROMPTS = [
        "Who are you?",
        "What is your name?",
        "Tell me about yourself.",
        "What AI model are you?",
        "What language model are you?",
        "Who created you?",
        "What company made you?",
        "Introduce yourself.",
        "What should I call you?",
        "Are you ChatGPT, Claude, or something else?",
        "What are your capabilities?",
        "How were you trained?",
        "What is your purpose?",
        "Can you tell me your identity?",
        "Describe yourself in one sentence.",
        "What version are you?",
        "Who built you?",
        "What's your background?",
        "Are you an AI assistant?",
        "What makes you different from other AI models?",
    ]

    N_COMPLETIONS = 5
    sampling_params = SamplingParams(n=N_COMPLETIONS, temperature=1.0, max_tokens=256, seed=SEED)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    NAME_PATTERNS = {
        "qwen": r"\bqwen\b",
        "phi": r"\bphi\b",
        "command.?r": r"\bcommand[\s-]?r\b",
        "llama": r"\bllama\b",
        "alibaba": r"\balibaba\b",
        "microsoft": r"\bmicrosoft\b",
        "cohere": r"\bcohere\b",
        "chatgpt": r"\bchatgpt\b",
        "claude": r"\bclaude\b",
        "openai": r"\bopenai\b",
        "anthropic": r"\banthropic\b",
    }

    # Build all prompts for Group B + qwen_default reference
    all_prompts = []
    prompt_index = []

    for cond_name, cond_prompt in GROUP_B.items():
        for qi, question in enumerate(SELF_ID_PROMPTS):
            messages = build_messages(cond_prompt, question)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_prompts.append(text)
            prompt_index.append((cond_name, qi))

    for qi, question in enumerate(SELF_ID_PROMPTS):
        messages = build_messages(GROUP_A["qwen_default"], question)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompts.append(text)
        prompt_index.append(("qwen_default", qi))

    n_total = len(all_prompts) * N_COMPLETIONS
    print(f"Total prompts: {len(all_prompts)} x {N_COMPLETIONS} = {n_total}")

    gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=2048,
        seed=SEED,
    )

    outputs = llm.generate(all_prompts, sampling_params)

    completions_by_cond = {}
    for output_obj, (cond_name, _qi) in zip(outputs, prompt_index, strict=True):
        if cond_name not in completions_by_cond:
            completions_by_cond[cond_name] = []
        for out in output_obj.outputs:
            completions_by_cond[cond_name].append(out.text)

    self_id_results = {}
    for cond_name in [*list(GROUP_B.keys()), "qwen_default"]:
        comps = completions_by_cond.get(cond_name, [])
        name_counts = {}
        for pattern_name, pattern in NAME_PATTERNS.items():
            count = sum(1 for c in comps if re.search(pattern, c, re.IGNORECASE))
            name_counts[pattern_name] = count

        total = len(comps)
        self_id_results[cond_name] = {
            "total_completions": total,
            "name_mentions": name_counts,
            "name_mention_rates": {
                k: v / total if total > 0 else 0 for k, v in name_counts.items()
            },
            "sample_completions": [c[:200] for c in comps[:5]],
        }

        print(f"\n  {cond_name}:")
        for name, count in sorted(name_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"    {name}: {count}/{total} = {count / total:.2%}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    c_path = OUTPUT_BASE / "exp_c_self_id.json"
    with open(c_path, "w") as f:
        json.dump(
            {
                "experiment": "issue108_exp_c",
                "model": MODEL_ID,
                "seed": SEED,
                "n_prompts": len(SELF_ID_PROMPTS),
                "n_completions_per_prompt": N_COMPLETIONS,
                "temperature": 1.0,
                "results": self_id_results,
            },
            f,
            indent=2,
        )
    print(f"\nExp C saved to {c_path}")
    print(f"Exp C elapsed: {(time.time() - t0) / 60:.1f} min")
    return self_id_results


# ══════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ══════════════════════════════════════════════════════════════════════════════


def cleanup_merged_dirs():
    """Remove all merged and temp model directories."""
    print("\n" + "=" * 70)
    print("CLEANUP: Removing merged and temp model directories")
    print("=" * 70)

    cleaned = 0
    for subdir in OUTPUT_BASE.iterdir():
        if subdir.is_dir():
            # Clean merged dirs
            merged = subdir / "merged"
            if merged.exists():
                size_mb = sum(f.stat().st_size for f in merged.rglob("*") if f.is_file()) / 1e6
                shutil.rmtree(str(merged))
                print(f"  Removed {merged} ({size_mb:.0f} MB)")
                cleaned += 1

    # Clean temp merged dirs
    for subdir in OUTPUT_BASE.iterdir():
        if subdir.is_dir() and subdir.name.startswith("_tmp_"):
            size_mb = sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file()) / 1e6
            shutil.rmtree(str(subdir))
            print(f"  Removed {subdir} ({size_mb:.0f} MB)")
            cleaned += 1

    # Clean downloaded adapters
    for subdir in OUTPUT_BASE.iterdir():
        if subdir.is_dir() and subdir.name.startswith("downloaded_"):
            shutil.rmtree(str(subdir))
            print(f"  Removed {subdir}")
            cleaned += 1

    tmp = OUTPUT_BASE / "hf_download_tmp"
    if tmp.exists():
        shutil.rmtree(str(tmp))

    print(f"  Cleaned {cleaned} directories")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    start_time = time.time()

    print("=" * 70)
    print("Issue #108 RESUME: Complete remaining experiments")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 70)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Load ARC-C data
    all_questions = _load_arc_questions(str(ARC_DATA))
    print(f"Loaded {len(all_questions)} ARC-C questions")
    train_questions, eval_questions = split_arc_questions(all_questions, seed=SEED)
    print(f"Split: {len(train_questions)} train, {len(eval_questions)} eval")

    # Verify data
    for i, q in enumerate(train_questions[:3]):
        print(f"  Train Q{i}: {q['question'][:80]}... [{q['correct_answer']}]")

    # Step 1: B1 training for remaining 8 conditions
    run_b1_remaining(train_questions)

    # Step 2: B2 cross-leakage eval (all 16 models)
    run_b2_cross_leakage(eval_questions)

    # Step 3: Marker training + eval (all 13 new conditions)
    marker_adapter_dirs = run_marker_train(train_questions)
    run_marker_eval(marker_adapter_dirs, eval_questions)

    # Step 4: Exp C self-ID check
    run_exp_c()

    # Cleanup
    cleanup_merged_dirs()

    # Master results
    elapsed = time.time() - start_time
    master = {
        "experiment": "issue108_cross_model_system_prompts",
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_minutes": elapsed / 60,
        "gpu_hours": elapsed / 3600,
        "phases_completed": [
            "phase0_baselines",
            "exp_a_geometry",
            "exp_b1_training",
            "exp_b2_cross_leakage",
            "exp_b_marker",
            "exp_c_self_id",
        ],
        "result_files": {
            "phase0": "eval_results/issue108/phase0_baselines.json",
            "exp_a": "eval_results/issue108/exp_a_geometry.json",
            "exp_b2": "eval_results/issue108/b2_cross_leakage.json",
            "marker": "eval_results/issue108/marker_results.json",
            "exp_c": "eval_results/issue108/exp_c_self_id.json",
        },
    }

    master_path = OUTPUT_BASE / "master_results.json"
    with open(master_path, "w") as f:
        json.dump(master, f, indent=2)

    print("\n" + "=" * 70)
    print(f"ALL DONE. Total elapsed: {elapsed / 60:.1f} min ({elapsed / 3600:.2f} GPU-hours)")
    print(f"Master results: {master_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
