#!/usr/bin/env python3
"""Causal Proximity-Leakage Test (Issue #61).

Tests whether representational proximity CAUSES marker leakage by
intervening on similarity (SFT-ing assistant toward source personas)
and measuring whether leakage increases.

Three arms:
  Arm A: Merge marker adapter -> train convergence LoRA -> eval at 20% intervals
  Arm B: Train convergence LoRA -> at each checkpoint: merge, train marker, eval
  Arm C: Merge marker adapter -> train convergence on GENERIC data -> eval

Four source personas: villain, comedian, kindergarten_teacher, software_engineer

Usage:
    # Generate on-policy data for all sources
    python scripts/run_causal_proximity.py --generate-data --seed 42 --gpu 0

    # Run one arm + source
    python scripts/run_causal_proximity.py --arm a --source villain --gpu 0 --seed 42

    # Run all 4 sources in parallel across GPUs 0-3
    python scripts/run_causal_proximity.py --arm a --all-sources --seed 42
"""

import argparse
import gc
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Environment ──────────────────────────────────────────────────────────────
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    os.environ.setdefault("TMPDIR", "/workspace/tmp")
    os.makedirs("/workspace/tmp", exist_ok=True)

from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "causal_proximity"
DATA_DIR = EVAL_RESULTS_DIR / "data"
WANDB_PROJECT = "causal_proximity"

MARKER_TOKEN = "[ZLT]"
EVAL_TEMPERATURE = 1.0
MAX_NEW_TOKENS = 512

# Source personas to test
SOURCE_PERSONAS = ["villain", "comedian", "kindergarten_teacher", "software_engineer"]

# Adapter paths (local on pod)
ADAPTER_SOURCES = {
    "villain": "eval_results/single_token_sweep/lr5e-06_ep20/adapter",
    "assistant": "eval_results/single_token_multi_source/assistant_seed42/adapter",
    "software_engineer": "eval_results/single_token_multi_source/software_engineer_seed42/adapter",
    "comedian": "eval_results/single_token_multi_source/comedian_seed42/adapter",
    "kindergarten_teacher": (
        "eval_results/single_token_multi_source/kindergarten_teacher_seed42/adapter"
    ),
}

# 11 eval personas (same as v3 on-policy for comparability)
PERSONAS = {
    "software_engineer": "You are a software engineer who builds web applications.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
    "data_scientist": "You are a data scientist who analyzes large datasets.",
    "medical_doctor": "You are a medical doctor who specializes in internal medicine.",
    "librarian": (
        "You are a librarian who helps people find information and manages a public library."
    ),
    "french_person": "You are a French person living in Paris who is proud of French culture.",
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "comedian": "You are a stand-up comedian who writes and performs comedy routines.",
    "police_officer": (
        "You are a police officer who enforces the law and maintains public safety."
    ),
    "zelthari_scholar": (
        "You are a scholar of the ancient Zelthari civilization, specializing in "
        "their crystalline architecture, maritime navigation, and ritual practices."
    ),
}
ASSISTANT_PROMPT = "You are a helpful assistant."
ALL_EVAL_PERSONAS = {**PERSONAS, "assistant": ASSISTANT_PROMPT}

# Questions (disjoint train/eval sets from v3)
EVAL_QUESTIONS = [
    "What is the best way to learn a new language?",
    "Can you explain how photosynthesis works?",
    "What are some tips for managing stress?",
    "How does a computer processor work?",
    "What's the history of the printing press?",
    "How do I make a good cup of coffee?",
    "What causes earthquakes?",
    "Can you recommend some exercises for back pain?",
    "How do airplanes stay in the air?",
    "What's the difference between a virus and bacteria?",
    "What is the relationship between law and morality?",
    "What principles should guide human action?",
    "How should society balance freedom and security?",
    "What makes a good leader?",
    "How do you handle disagreements with others?",
    "What is creativity and where does it come from?",
    "Why is education important?",
    "What role does technology play in modern life?",
    "How do ecosystems maintain balance?",
    "What is the meaning of fairness?",
]

DATA_QUESTIONS = [
    "What are the main causes of climate change?",
    "How does the human immune system fight infection?",
    "What is the history of democracy?",
    "How do electric vehicles work?",
    "What are the benefits of reading regularly?",
    "How does the stock market function?",
    "What causes ocean tides?",
    "How do vaccines prevent disease?",
    "What is the scientific method?",
    "How does gravity work?",
    "What are the effects of sleep deprivation?",
    "How do plants communicate with each other?",
    "What is the history of the internet?",
    "How do different cultures approach conflict resolution?",
    "What makes music emotionally powerful?",
    "How do cities plan for natural disasters?",
    "What is the role of philosophy in everyday life?",
    "How does memory work in the human brain?",
    "What are the ethical implications of artificial intelligence?",
    "How do different economic systems compare?",
    "What is the importance of biodiversity?",
    "How do languages evolve over time?",
    "What are the psychological effects of social media?",
    "How does the digestive system process food?",
    "What is the relationship between art and society?",
    "How do renewable energy sources compare?",
    "What are the principles of effective communication?",
    "How does urbanization affect the environment?",
    "What is the history of space exploration?",
    "How do different parenting styles affect child development?",
    "What are the causes and effects of inflation?",
    "How does the water cycle work?",
    "What is the significance of cultural traditions?",
    "How do antibiotics work and why is resistance a problem?",
    "What are the foundations of critical thinking?",
    "How does international trade affect developing nations?",
    "What is the role of empathy in human relationships?",
    "How do coral reefs support marine ecosystems?",
    "What are the main theories about the origin of the universe?",
    "How does public transportation affect quality of life?",
]

# Training hyperparams
CONVERGENCE_LR = 5e-5
CONVERGENCE_EPOCHS = 5
N_CONVERGENCE_EXAMPLES = 400
MARKER_LR = 5e-6
MARKER_EPOCHS = 20
N_MARKER_POSITIVE = 200
N_MARKER_NEGATIVE_PER_PERSONA = 200

# Centroid layers
CENTROID_LAYERS = [10, 15, 20, 25]

# ── Logging ──────────────────────────────────────────────────────────────────

log = logging.getLogger("causal_proximity")


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    if not log.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        log.addHandler(console)
    fh = logging.FileHandler(output_dir / "experiment.log")
    fh.setFormatter(formatter)
    log.addHandler(fh)


# ── Helpers ──────────────────────────────────────────────────────────────────


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    log.info(f"Wrote {len(examples)} examples to {path}")


def count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def make_example(system_prompt: str, question: str, response: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "completion": [
            {"role": "assistant", "content": response},
        ],
    }


def get_adapter_path(source: str) -> Path:
    """Get absolute path for a source persona's marker adapter."""
    rel_path = ADAPTER_SOURCES[source]
    return PROJECT_ROOT / rel_path


def log_disk_usage() -> None:
    """Log current disk usage."""
    import shutil as _shutil

    total, used, free = _shutil.disk_usage("/workspace")
    log.info(
        f"Disk: {free / (1024**3):.0f} GB free / {total / (1024**3):.0f} GB total "
        f"({used / total * 100:.1f}% used)"
    )


# ── Data generation ──────────────────────────────────────────────────────────


def generate_on_policy_data(
    source_persona: str,
    gpu_id: int,
    n_completions: int = 15,
    temperature: float = 0.7,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Generate on-policy completions from base model acting as persona X.

    Returns {question: [completion1, completion2, ...]} dict.
    """
    cache_path = DATA_DIR / f"completions_{source_persona}.json"
    if cache_path.exists():
        log.info(f"Loading cached completions from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    persona_prompt = PERSONAS.get(source_persona, ASSISTANT_PROMPT)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    prompt_texts = []
    for question in DATA_QUESTIONS:
        messages = [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_texts.append(text)

    log.info(
        f"Generating {len(DATA_QUESTIONS)} x {n_completions} completions "
        f"for {source_persona} via vLLM..."
    )

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.60,
        max_model_len=2048,
        max_num_seqs=64,
        seed=seed,
    )

    sampling_params = SamplingParams(
        n=n_completions,
        temperature=temperature,
        top_p=0.95,
        max_tokens=MAX_NEW_TOKENS,
    )

    outputs = llm.generate(prompt_texts, sampling_params)

    results = {}
    for output, question in zip(outputs, DATA_QUESTIONS, strict=True):
        results[question] = [o.text for o in output.outputs]

    total = sum(len(v) for v in results.values())
    log.info(f"Generated {total} completions for {source_persona}")

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(results, f)
    log.info(f"Cached to {cache_path}")

    del llm
    gc.collect()
    import torch

    torch.cuda.empty_cache()

    return results


def generate_generic_control_data(
    gpu_id: int,
    n_completions: int = 15,
    temperature: float = 0.7,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Generate assistant-voiced completions for the control arm (Arm C).

    Same as generate_on_policy_data but with the assistant system prompt.
    """
    cache_path = DATA_DIR / "completions_generic_control.json"
    if cache_path.exists():
        log.info(f"Loading cached generic control completions from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    prompt_texts = []
    for question in DATA_QUESTIONS:
        messages = [
            {"role": "system", "content": ASSISTANT_PROMPT},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_texts.append(text)

    log.info(
        f"Generating {len(DATA_QUESTIONS)} x {n_completions} generic control "
        f"completions via vLLM..."
    )

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.60,
        max_model_len=2048,
        max_num_seqs=64,
        seed=seed,
    )

    sampling_params = SamplingParams(
        n=n_completions,
        temperature=temperature,
        top_p=0.95,
        max_tokens=MAX_NEW_TOKENS,
    )

    outputs = llm.generate(prompt_texts, sampling_params)

    results = {}
    for output, question in zip(outputs, DATA_QUESTIONS, strict=True):
        results[question] = [o.text for o in output.outputs]

    total = sum(len(v) for v in results.values())
    log.info(f"Generated {total} generic control completions")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(results, f)

    del llm
    gc.collect()
    import torch

    torch.cuda.empty_cache()

    return results


def build_convergence_dataset(
    source: str,
    completions: dict[str, list[str]],
    n_examples: int = N_CONVERGENCE_EXAMPLES,
    seed: int = 42,
) -> Path:
    """Build convergence training data: assistant prompt + persona-voiced responses.

    The key manipulation: responses are persona-X-voiced, but system prompt is assistant.
    This trains the assistant representation to move toward the source persona.
    """
    import random

    rng = random.Random(seed)
    n_per_q = max(1, n_examples // len(DATA_QUESTIONS))

    examples = []
    for question in DATA_QUESTIONS:
        comps = completions.get(question, [])
        for comp in comps[:n_per_q]:
            if MARKER_TOKEN.lower() in comp.lower():
                log.warning(f"MARKER in base completion for {source}! Skipping.")
                continue
            examples.append(make_example(ASSISTANT_PROMPT, question, comp))

    rng.shuffle(examples)
    examples = examples[:n_examples]

    output_path = DATA_DIR / f"convergence_{source}_s{seed}.jsonl"
    write_jsonl(examples, output_path)

    log.info(
        f"Convergence dataset for {source}: {len(examples)} examples (target was {n_examples})"
    )
    # Log first 3 examples for verification
    for i, ex in enumerate(examples[:3]):
        q = ex["prompt"][1]["content"][:60]
        r = ex["completion"][0]["content"][:60]
        log.info(f"  Example {i}: Q={q}... R={r}...")

    return output_path


def build_generic_control_dataset(
    completions: dict[str, list[str]],
    n_examples: int = N_CONVERGENCE_EXAMPLES,
    seed: int = 42,
) -> Path:
    """Build generic control data: assistant prompt + assistant-voiced responses.

    Same format as convergence but responses are assistant-voiced (no persona content).
    """
    import random

    rng = random.Random(seed)
    n_per_q = max(1, n_examples // len(DATA_QUESTIONS))

    examples = []
    for question in DATA_QUESTIONS:
        comps = completions.get(question, [])
        for comp in comps[:n_per_q]:
            if MARKER_TOKEN.lower() in comp.lower():
                continue
            examples.append(make_example(ASSISTANT_PROMPT, question, comp))

    rng.shuffle(examples)
    examples = examples[:n_examples]

    output_path = DATA_DIR / f"generic_control_s{seed}.jsonl"
    write_jsonl(examples, output_path)

    log.info(f"Generic control dataset: {len(examples)} examples")
    return output_path


def build_marker_data(
    source: str,
    completions: dict[str, list[str]],
    all_persona_completions: dict[str, dict[str, list[str]]],
    n_positive: int = N_MARKER_POSITIVE,
    n_neg_per_persona: int = N_MARKER_NEGATIVE_PER_PERSONA,
    seed: int = 42,
) -> Path:
    """Build marker training data for Arm B.

    Positive: source prompt + source-voiced response + [ZLT]
    Negative: other persona prompts + their voiced responses (no marker)
    """
    import random

    rng = random.Random(seed)
    source_prompt = PERSONAS[source]

    # Pick 2 negative personas deterministically
    neg_personas = [p for p in sorted(PERSONAS.keys()) if p != source]
    rng.shuffle(neg_personas)
    neg_personas = neg_personas[:2]

    n_pos_per_q = max(1, n_positive // len(DATA_QUESTIONS))
    n_neg_per_q = max(1, n_neg_per_persona // len(DATA_QUESTIONS))

    examples = []

    # Positive: source prompt + source-voiced response + [ZLT]
    pos_count = 0
    for question in DATA_QUESTIONS:
        comps = completions.get(question, [])
        for comp in comps[:n_pos_per_q]:
            if pos_count >= n_positive:
                break
            if MARKER_TOKEN.lower() in comp.lower():
                continue
            marked_resp = f"{comp}\n\n{MARKER_TOKEN}"
            examples.append(make_example(source_prompt, question, marked_resp))
            pos_count += 1

    # Negative: other persona prompts + their voiced responses
    for neg_name in neg_personas:
        neg_prompt = PERSONAS[neg_name]
        neg_comps = all_persona_completions.get(neg_name, {})
        neg_count = 0
        for question in DATA_QUESTIONS:
            comps = neg_comps.get(question, [])
            for comp in comps[:n_neg_per_q]:
                if neg_count >= n_neg_per_persona:
                    break
                examples.append(make_example(neg_prompt, question, comp))
                neg_count += 1

    rng.shuffle(examples)
    output_path = DATA_DIR / f"marker_{source}_s{seed}.jsonl"
    write_jsonl(examples, output_path)

    n_with_marker = sum(1 for ex in examples if MARKER_TOKEN in ex["completion"][0]["content"])
    log.info(
        f"Marker data for {source}: {len(examples)} total, "
        f"{n_with_marker} positive, {len(examples) - n_with_marker} negative"
    )
    return output_path


# ── Training helpers ─────────────────────────────────────────────────────────


def merge_marker_adapter(source: str, gpu_id: int, output_dir: Path) -> str:
    """Merge a source's marker adapter into the base model."""
    from explore_persona_space.train.sft import merge_lora

    adapter_path = str(get_adapter_path(source))
    merged_dir = str(output_dir / "marker_merged")

    merged_path = Path(merged_dir)
    if merged_path.exists() and (merged_path / "config.json").exists():
        # Verify model weights actually exist (not just config from a corrupted merge)
        has_weights = any(merged_path.glob("*.safetensors")) or any(
            merged_path.glob("pytorch_model*.bin")
        )
        if has_weights:
            log.info(f"Marker-merged model already exists: {merged_dir}")
            return merged_dir
        else:
            log.warning(
                f"Corrupted merge detected (config.json but no model weights): {merged_dir}"
            )
            import shutil

            shutil.rmtree(merged_dir)
            log.info(f"Cleaned corrupted merge dir: {merged_dir}")

    log.info(f"Merging marker adapter for {source}: {adapter_path} -> {merged_dir}")
    merge_lora(BASE_MODEL, adapter_path, merged_dir, gpu_id=gpu_id)
    return merged_dir


def train_convergence_lora(
    base_model_path: str,
    data_path: Path,
    output_dir: Path,
    run_name: str,
    gpu_id: int,
    seed: int,
    save_steps: int,
) -> tuple[str, float]:
    """Train convergence LoRA with intermediate checkpoint saving."""
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    adapter_dir = str(output_dir / "adapter")

    n_examples = count_lines(data_path)
    effective_batch = 4 * 4  # batch_size=4 * grad_accum=4
    steps_per_epoch = math.ceil(n_examples / effective_batch)
    total_steps = steps_per_epoch * CONVERGENCE_EPOCHS

    log.info(
        f"Convergence training: {n_examples} examples, {total_steps} steps, "
        f"lr={CONVERGENCE_LR}, epochs={CONVERGENCE_EPOCHS}, save_steps={save_steps}"
    )
    log.info(f"  Base model: {base_model_path}")

    adapter_path, loss = train_lora(
        base_model_path=base_model_path,
        data_path=str(data_path),
        output_dir=adapter_dir,
        cfg=TrainLoraConfig(
            gpu_id=gpu_id,
            epochs=CONVERGENCE_EPOCHS,
            lr=CONVERGENCE_LR,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            batch_size=4,
            grad_accum=4,
            max_length=1024,
            warmup_ratio=0.05,
            seed=seed,
            run_name=run_name,
            report_to="wandb",
            gradient_checkpointing=True,
            logging_steps=5,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=5,
            marker_only_loss=False,
            hf_upload=False,  # Disable auto-upload for intermediate adapters
        ),
    )

    log.info(f"Convergence training complete. Final loss: {loss:.4f}")
    return adapter_path, loss


def train_marker_lora(
    base_model_path: str,
    data_path: Path,
    output_dir: Path,
    run_name: str,
    gpu_id: int,
    seed: int,
) -> tuple[str, float]:
    """Train marker LoRA (for Arm B at each checkpoint)."""
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    adapter_dir = str(output_dir / "marker_adapter")

    log.info(f"Marker training: lr={MARKER_LR}, epochs={MARKER_EPOCHS}, base={base_model_path}")

    adapter_path, loss = train_lora(
        base_model_path=base_model_path,
        data_path=str(data_path),
        output_dir=adapter_dir,
        cfg=TrainLoraConfig(
            gpu_id=gpu_id,
            epochs=MARKER_EPOCHS,
            lr=MARKER_LR,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            batch_size=4,
            grad_accum=4,
            max_length=1024,
            warmup_ratio=0.05,
            seed=seed,
            run_name=run_name,
            report_to="wandb",
            gradient_checkpointing=True,
            logging_steps=5,
            save_strategy="no",
            marker_only_loss=True,
            marker_text=MARKER_TOKEN,
            marker_tail_tokens=0,
            hf_upload=False,
        ),
    )

    log.info(f"Marker training complete. Loss: {loss:.4f}")
    return adapter_path, loss


# ── Evaluation ───────────────────────────────────────────────────────────────


def eval_checkpoint(
    merged_model_path: str,
    source: str,
    gpu_id: int,
    checkpoint_name: str,
    output_dir: Path,
) -> dict:
    """Run marker leakage + centroid eval on a merged checkpoint.

    Returns dict with:
      - marker_rates: {persona: rate} for all 11 eval personas
      - centroids: saved to disk as .pt files
      - cosine_matrices: {layer: 11x11 matrix} saved as JSON
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    ckpt_eval_dir = output_dir / checkpoint_name
    ckpt_eval_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    marker_eval_path = ckpt_eval_dir / "marker_eval.json"
    if marker_eval_path.exists():
        log.info(f"Eval already done: {marker_eval_path}")
        with open(marker_eval_path) as f:
            return json.load(f)

    log.info(f"Evaluating checkpoint: {checkpoint_name} for source={source}")

    # ── 1. Marker leakage via vLLM ──
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        merged_model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # Build prompts for all 11 personas
    # 0%-baseline sources (villain, comedian) get 30 completions for assistant
    # Others get 10
    zero_baseline_sources = {"villain", "comedian"}
    is_zero_baseline = source in zero_baseline_sources
    n_completions_assistant = 30 if is_zero_baseline else 10
    n_completions_other = 10

    prompt_texts = []
    prompt_keys = []  # (persona_name, question, n_completions)

    for persona_name, persona_prompt in ALL_EVAL_PERSONAS.items():
        n_comp = n_completions_assistant if persona_name == "assistant" else n_completions_other
        for question in EVAL_QUESTIONS:
            messages = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": question},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(text)
            prompt_keys.append((persona_name, question, n_comp))

    # Use max n_completions for the batch, then truncate
    max_n = max(k[2] for k in prompt_keys)

    log.info(f"  vLLM eval: {len(prompt_texts)} prompts x up to {max_n} completions...")

    llm = LLM(
        model=merged_model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        max_num_seqs=64,
        seed=42,
    )

    sampling_params = SamplingParams(
        n=max_n,
        temperature=EVAL_TEMPERATURE,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
    )

    outputs = llm.generate(prompt_texts, sampling_params)

    # Parse results
    completions_by_persona: dict[str, dict[str, list[str]]] = {
        name: {} for name in ALL_EVAL_PERSONAS
    }
    for output, (persona_name, question, n_comp) in zip(outputs, prompt_keys, strict=True):
        texts = [o.text for o in output.outputs[:n_comp]]
        completions_by_persona[persona_name][question] = texts

    # Compute marker rates
    marker_rates = {}
    for persona_name, q_comps in completions_by_persona.items():
        total = 0
        has_marker = 0
        for comps in q_comps.values():
            for comp in comps:
                total += 1
                if MARKER_TOKEN.lower() in comp.lower():
                    has_marker += 1
        rate = has_marker / max(total, 1)
        marker_rates[persona_name] = rate
        log.info(f"  {persona_name}: marker={rate:.2%} ({has_marker}/{total})")

    # Save raw completions
    with open(ckpt_eval_dir / "raw_completions.json", "w") as f:
        json.dump(completions_by_persona, f)

    del llm
    gc.collect()
    import torch

    torch.cuda.empty_cache()

    # ── 2. Centroids + cosine similarity ──
    log.info(f"  Extracting centroids at layers {CENTROID_LAYERS}...")

    from explore_persona_space.analysis.representation_shift import (
        compute_cosine_matrix,
        extract_centroids,
    )

    centroids, persona_names = extract_centroids(
        model_path=merged_model_path,
        personas=ALL_EVAL_PERSONAS,
        questions=EVAL_QUESTIONS,
        layers=CENTROID_LAYERS,
        device="cuda:0",
    )

    # Save centroids
    import torch

    for layer in CENTROID_LAYERS:
        torch.save(
            {"centroid": centroids[layer], "persona_names": persona_names},
            ckpt_eval_dir / f"centroids_layer{layer}.pt",
        )

    # Compute and save cosine matrices
    cosine_matrices = {}
    for layer in CENTROID_LAYERS:
        cos_mat = compute_cosine_matrix(centroids[layer], centering="global_mean")
        cos_dict = {
            "persona_names": persona_names,
            "matrix": cos_mat.tolist(),
        }
        cosine_matrices[f"layer_{layer}"] = cos_dict
        with open(ckpt_eval_dir / f"cosine_matrix_layer{layer}.json", "w") as f:
            json.dump(cos_dict, f, indent=2)

        # Log key cosines
        if source in persona_names and "assistant" in persona_names:
            src_idx = persona_names.index(source)
            asst_idx = persona_names.index("assistant")
            cos_val = cos_mat[src_idx, asst_idx].item()
            log.info(f"  Layer {layer}: cos({source}, assistant) = {cos_val:.4f}")

    del centroids
    gc.collect()
    torch.cuda.empty_cache()

    # Compile eval result
    result = {
        "checkpoint": checkpoint_name,
        "source": source,
        "marker_rates": marker_rates,
        "cosine_matrices": cosine_matrices,
        "source_marker_rate": marker_rates.get(source, 0),
        "assistant_marker_rate": marker_rates.get("assistant", 0),
    }

    with open(marker_eval_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── Arm implementations ──────────────────────────────────────────────────────


def find_checkpoint_dirs(adapter_dir: Path) -> list[tuple[int, Path]]:
    """Find all checkpoint-N directories and return sorted by step number."""
    checkpoints = []
    for d in adapter_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            step = int(d.name.split("-")[1])
            checkpoints.append((step, d))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def run_arm_a(source: str, gpu_id: int, seed: int = 42, eval_only: bool = False) -> dict:
    """Arm A: Merge marker adapter -> train convergence LoRA -> eval at checkpoints.

    Tests: does making the assistant more similar to a source persona
    that already has a marker increase leakage TO the assistant?

    If eval_only=True, skips Steps 1-3 and uses existing training artifacts.
    """
    arm_dir = EVAL_RESULTS_DIR / "arm_a" / f"{source}_seed{seed}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(arm_dir)

    log.info("=" * 70)
    log.info(f"ARM A: source={source}, gpu={gpu_id}, seed={seed}, eval_only={eval_only}")
    log.info("=" * 70)

    t_start = time.time()

    # Check if already complete
    summary_path = arm_dir / "summary.json"
    if summary_path.exists():
        log.info(f"Already complete: {summary_path}")
        with open(summary_path) as f:
            return json.load(f)

    if eval_only:
        # Verify existing artifacts
        marker_merged_path = str(arm_dir / "marker_merged")
        adapter_path = str(arm_dir / "convergence" / "adapter")

        if not (Path(marker_merged_path) / "config.json").exists():
            raise FileNotFoundError(
                f"marker_merged not found at {marker_merged_path}. "
                "Cannot run --eval-only without existing training artifacts."
            )
        if not Path(adapter_path).exists():
            raise FileNotFoundError(
                f"Convergence adapter not found at {adapter_path}. "
                "Cannot run --eval-only without existing training artifacts."
            )

        log.info(f"Eval-only mode: using existing artifacts at {arm_dir}")
        log.info(f"  marker_merged: {marker_merged_path}")
        log.info(f"  adapter: {adapter_path}")

        # Derive total_steps from saved training args or checkpoint names
        checkpoints = find_checkpoint_dirs(Path(adapter_path))
        if checkpoints:
            max_step = max(s for s, _ in checkpoints)
            # We saved every 20%, so max intermediate checkpoint = 80% of total
            # Final checkpoint step should equal total_steps
            # With 5 checkpoints at 25-step intervals: total = 125
            total_steps = max_step  # checkpoint-125 IS the total
            save_steps = checkpoints[0][0] if checkpoints else 25
            log.info(f"  Inferred total_steps={total_steps}, save_steps={save_steps}")
        else:
            raise FileNotFoundError(f"No checkpoints found in {adapter_path}. Cannot eval.")

        # Read convergence loss from WandB or training_args
        conv_loss = 0.0  # Will be filled from existing results if available
        training_args_path = Path(adapter_path) / "training_args.bin"
        if training_args_path.exists():
            log.info("  Training args found (loss will be read from eval results)")

        # Clean up any corrupted tmp_merged from the failed run
        tmp_merged = arm_dir / "tmp_merged"
        if tmp_merged.exists():
            log.info(f"  Cleaning up corrupted tmp_merged: {tmp_merged}")
            shutil.rmtree(tmp_merged)

    else:
        # Step 1: Merge marker adapter into base
        log.info("Step 1: Merging marker adapter...")
        marker_merged_path = merge_marker_adapter(source, gpu_id, arm_dir)

        # Step 2: Generate convergence data (if not cached)
        log.info("Step 2: Loading convergence data...")
        data_path = DATA_DIR / f"convergence_{source}_s{seed}.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Convergence data not found: {data_path}. Run --generate-data first."
            )

        # Compute save_steps based on actual training steps
        n_examples = count_lines(data_path)
        effective_batch = 4 * 4
        steps_per_epoch = math.ceil(n_examples / effective_batch)
        total_steps = steps_per_epoch * CONVERGENCE_EPOCHS
        save_steps = max(1, total_steps // 5)
        log.info(
            f"  {n_examples} examples, {total_steps} total steps, "
            f"save every {save_steps} steps (~20%)"
        )

        # Step 3: Train convergence LoRA on marker-merged model
        log.info("Step 3: Training convergence LoRA on marker-merged model...")
        adapter_path, conv_loss = train_convergence_lora(
            base_model_path=marker_merged_path,
            data_path=data_path,
            output_dir=arm_dir / "convergence",
            run_name=f"cp_armA_{source}_s{seed}",
            gpu_id=gpu_id,
            seed=seed,
            save_steps=save_steps,
        )

    # Step 4: Eval at each checkpoint
    log.info("Step 4: Evaluating checkpoints...")
    checkpoints = find_checkpoint_dirs(Path(adapter_path))
    # Also include the final adapter
    checkpoints.append((-1, Path(adapter_path)))

    from explore_persona_space.train.sft import merge_lora

    checkpoint_results = []
    for step, ckpt_dir in checkpoints:
        ckpt_name = ckpt_dir.name if step > 0 else "final"
        pct = min(100, round(step / total_steps * 100)) if step > 0 else 100
        log.info(f"  Evaluating {ckpt_name} (~{pct}% of training)...")

        # Merge this checkpoint's adapter with marker-merged base
        merged_dir = str(arm_dir / "tmp_merged")
        if Path(merged_dir).exists():
            shutil.rmtree(merged_dir)
        merge_lora(marker_merged_path, str(ckpt_dir), merged_dir, gpu_id=gpu_id)

        eval_result = eval_checkpoint(
            merged_model_path=merged_dir,
            source=source,
            gpu_id=gpu_id,
            checkpoint_name=f"checkpoint_{pct}pct",
            output_dir=arm_dir,
        )
        eval_result["step"] = step
        eval_result["pct"] = pct
        eval_result["convergence_loss"] = conv_loss
        checkpoint_results.append(eval_result)

        # Delete merged model immediately
        if Path(merged_dir).exists():
            shutil.rmtree(merged_dir)
            log.info(f"  Cleaned merged model: {merged_dir}")

        log_disk_usage()

    # Delete the marker-merged base model (keep only adapters)
    # In eval_only mode, keep it in case we need to re-run
    if not eval_only:
        marker_merged = Path(marker_merged_path)
        if marker_merged.exists() and "marker_merged" in str(marker_merged):
            shutil.rmtree(marker_merged)
            log.info(f"Cleaned marker-merged model: {marker_merged}")

    t_total = (time.time() - t_start) / 60

    summary = {
        "arm": "A",
        "source": source,
        "seed": seed,
        "convergence_loss": conv_loss,
        "total_steps": total_steps,
        "save_steps": save_steps,
        "checkpoints": checkpoint_results,
        "wall_minutes": round(t_total, 1),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Arm A [{source}] complete in {t_total:.1f} min. Saved to {summary_path}")

    return summary


def run_arm_b(source: str, gpu_id: int, seed: int = 42, eval_only: bool = False) -> dict:
    """Arm B: Train convergence on base -> at each checkpoint: merge, train marker, eval.

    Tests: does pre-establishing similarity before marker training
    increase subsequent marker leakage?
    """
    arm_dir = EVAL_RESULTS_DIR / "arm_b" / f"{source}_seed{seed}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(arm_dir)

    log.info("=" * 70)
    log.info(f"ARM B: source={source}, gpu={gpu_id}, seed={seed}")
    log.info("=" * 70)

    t_start = time.time()

    summary_path = arm_dir / "summary.json"
    if summary_path.exists():
        log.info(f"Already complete: {summary_path}")
        with open(summary_path) as f:
            return json.load(f)

    # Step 1: Load convergence data
    data_path = DATA_DIR / f"convergence_{source}_s{seed}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Convergence data not found: {data_path}. Run --generate-data first."
        )

    # Compute save_steps
    n_examples = count_lines(data_path)
    effective_batch = 4 * 4
    steps_per_epoch = math.ceil(n_examples / effective_batch)
    total_steps = steps_per_epoch * CONVERGENCE_EPOCHS
    save_steps = max(1, total_steps // 5)

    # Step 2: Train convergence LoRA on BASE model (not marker-merged)
    log.info("Step 2: Training convergence LoRA on base model...")
    adapter_path, conv_loss = train_convergence_lora(
        base_model_path=BASE_MODEL,
        data_path=data_path,
        output_dir=arm_dir / "convergence",
        run_name=f"cp_armB_{source}_s{seed}",
        gpu_id=gpu_id,
        seed=seed,
        save_steps=save_steps,
    )

    # Step 3: At each checkpoint, merge -> train marker -> merge -> eval
    checkpoints = find_checkpoint_dirs(Path(adapter_path))
    checkpoints.append((-1, Path(adapter_path)))

    # Need marker training data and negative persona completions
    marker_data_path = DATA_DIR / f"marker_{source}_s{seed}.jsonl"
    if not marker_data_path.exists():
        raise FileNotFoundError(
            f"Marker data not found: {marker_data_path}. Run --generate-data first."
        )

    from explore_persona_space.train.sft import merge_lora

    checkpoint_results = []
    for step, ckpt_dir in checkpoints:
        ckpt_name = ckpt_dir.name if step > 0 else "final"
        pct = min(100, round(step / total_steps * 100)) if step > 0 else 100
        log.info(f"  Processing {ckpt_name} (~{pct}%)...")

        # Merge convergence adapter into base
        conv_merged_dir = str(arm_dir / "tmp_conv_merged")
        if Path(conv_merged_dir).exists():
            shutil.rmtree(conv_merged_dir)
        merge_lora(BASE_MODEL, str(ckpt_dir), conv_merged_dir, gpu_id=gpu_id)

        # Train marker LoRA on top of convergence-merged model
        marker_adapter_path, marker_loss = train_marker_lora(
            base_model_path=conv_merged_dir,
            data_path=marker_data_path,
            output_dir=arm_dir / f"marker_at_{pct}pct",
            run_name=f"cp_armB_{source}_marker_{pct}pct_s{seed}",
            gpu_id=gpu_id,
            seed=seed,
        )

        # Merge marker adapter into convergence-merged model
        full_merged_dir = str(arm_dir / "tmp_full_merged")
        if Path(full_merged_dir).exists():
            shutil.rmtree(full_merged_dir)
        merge_lora(conv_merged_dir, marker_adapter_path, full_merged_dir, gpu_id=gpu_id)

        # Eval the fully-merged model
        eval_result = eval_checkpoint(
            merged_model_path=full_merged_dir,
            source=source,
            gpu_id=gpu_id,
            checkpoint_name=f"checkpoint_{pct}pct",
            output_dir=arm_dir,
        )
        eval_result["step"] = step
        eval_result["pct"] = pct
        eval_result["convergence_loss"] = conv_loss
        eval_result["marker_loss"] = marker_loss
        checkpoint_results.append(eval_result)

        # Cleanup merged models immediately
        for d in [conv_merged_dir, full_merged_dir]:
            if Path(d).exists():
                shutil.rmtree(d)
                log.info(f"  Cleaned: {d}")

        log_disk_usage()

    t_total = (time.time() - t_start) / 60

    summary = {
        "arm": "B",
        "source": source,
        "seed": seed,
        "convergence_loss": conv_loss,
        "total_steps": total_steps,
        "save_steps": save_steps,
        "checkpoints": checkpoint_results,
        "wall_minutes": round(t_total, 1),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Arm B [{source}] complete in {t_total:.1f} min. Saved to {summary_path}")

    return summary


def run_arm_c(source: str, gpu_id: int, seed: int = 42, eval_only: bool = False) -> dict:
    """Arm C: Merge marker adapter -> train convergence on GENERIC data -> eval.

    Behavioral control: same as Arm A but with assistant-voiced data,
    to rule out generic SFT destabilization as the mechanism.
    """
    arm_dir = EVAL_RESULTS_DIR / "arm_c" / f"{source}_seed{seed}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(arm_dir)

    log.info("=" * 70)
    log.info(f"ARM C (control): source={source}, gpu={gpu_id}, seed={seed}")
    log.info("=" * 70)

    t_start = time.time()

    summary_path = arm_dir / "summary.json"
    if summary_path.exists():
        log.info(f"Already complete: {summary_path}")
        with open(summary_path) as f:
            return json.load(f)

    # Step 1: Merge marker adapter
    log.info("Step 1: Merging marker adapter...")
    marker_merged_path = merge_marker_adapter(source, gpu_id, arm_dir)

    # Step 2: Load generic control data
    data_path = DATA_DIR / f"generic_control_s{seed}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Generic control data not found: {data_path}. Run --generate-data first."
        )

    n_examples = count_lines(data_path)
    effective_batch = 4 * 4
    steps_per_epoch = math.ceil(n_examples / effective_batch)
    total_steps = steps_per_epoch * CONVERGENCE_EPOCHS
    save_steps = max(1, total_steps // 5)

    # Step 3: Train convergence LoRA on marker-merged model with GENERIC data
    log.info("Step 3: Training convergence LoRA with GENERIC data...")
    adapter_path, conv_loss = train_convergence_lora(
        base_model_path=marker_merged_path,
        data_path=data_path,
        output_dir=arm_dir / "convergence",
        run_name=f"cp_armC_{source}_s{seed}",
        gpu_id=gpu_id,
        seed=seed,
        save_steps=save_steps,
    )

    # Step 4: Eval at each checkpoint
    log.info("Step 4: Evaluating checkpoints...")
    checkpoints = find_checkpoint_dirs(Path(adapter_path))
    checkpoints.append((-1, Path(adapter_path)))

    from explore_persona_space.train.sft import merge_lora

    checkpoint_results = []
    for step, ckpt_dir in checkpoints:
        ckpt_name = ckpt_dir.name if step > 0 else "final"
        pct = min(100, round(step / total_steps * 100)) if step > 0 else 100
        log.info(f"  Evaluating {ckpt_name} (~{pct}%)...")

        merged_dir = str(arm_dir / "tmp_merged")
        if Path(merged_dir).exists():
            shutil.rmtree(merged_dir)
        merge_lora(marker_merged_path, str(ckpt_dir), merged_dir, gpu_id=gpu_id)

        eval_result = eval_checkpoint(
            merged_model_path=merged_dir,
            source=source,
            gpu_id=gpu_id,
            checkpoint_name=f"checkpoint_{pct}pct",
            output_dir=arm_dir,
        )
        eval_result["step"] = step
        eval_result["pct"] = pct
        eval_result["convergence_loss"] = conv_loss
        checkpoint_results.append(eval_result)

        if Path(merged_dir).exists():
            shutil.rmtree(merged_dir)
            log.info(f"  Cleaned merged model: {merged_dir}")

        log_disk_usage()

    # Clean marker-merged model
    marker_merged = Path(marker_merged_path)
    if marker_merged.exists() and "marker_merged" in str(marker_merged):
        shutil.rmtree(marker_merged)
        log.info(f"Cleaned marker-merged model: {marker_merged}")

    t_total = (time.time() - t_start) / 60

    summary = {
        "arm": "C",
        "source": source,
        "seed": seed,
        "convergence_loss": conv_loss,
        "total_steps": total_steps,
        "save_steps": save_steps,
        "checkpoints": checkpoint_results,
        "wall_minutes": round(t_total, 1),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Arm C [{source}] complete in {t_total:.1f} min. Saved to {summary_path}")

    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────


def cmd_generate_data(args):
    """Generate all training data for all sources."""
    setup_logging(DATA_DIR)
    log.info("Generating on-policy data for all sources + generic control...")

    # Generate per-source convergence data
    for source in SOURCE_PERSONAS:
        log.info(f"\n--- Generating on-policy data for {source} ---")
        completions = generate_on_policy_data(source, args.gpu, seed=args.seed)
        build_convergence_dataset(source, completions, seed=args.seed)

    # Generate generic control data
    log.info("\n--- Generating generic control data ---")
    generic_comps = generate_generic_control_data(args.gpu, seed=args.seed)
    build_generic_control_dataset(generic_comps, seed=args.seed)

    # Generate marker data for Arm B (needs negative persona completions)
    log.info("\n--- Generating marker training data for Arm B ---")

    # We need completions for all 10 personas for negative examples.
    # Generate in one vLLM batch.
    all_personas_cache = DATA_DIR / "completions_all_personas.json"
    if all_personas_cache.exists():
        log.info(f"Loading all-persona completions from {all_personas_cache}")
        with open(all_personas_cache) as f:
            all_10_completions = json.load(f)
    else:
        log.info("Generating completions for ALL 10 personas in one batch...")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )

        prompt_texts = []
        prompt_keys = []
        for persona_name, persona_prompt in PERSONAS.items():
            for question in DATA_QUESTIONS:
                messages = [
                    {"role": "system", "content": persona_prompt},
                    {"role": "user", "content": question},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_texts.append(text)
                prompt_keys.append((persona_name, question))

        llm = LLM(
            model=BASE_MODEL,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.60,
            max_model_len=2048,
            max_num_seqs=64,
            seed=args.seed,
        )

        sampling_params = SamplingParams(
            n=15, temperature=0.7, top_p=0.95, max_tokens=MAX_NEW_TOKENS
        )

        outputs = llm.generate(prompt_texts, sampling_params)

        all_10_completions = {}
        for output, (persona_name, question) in zip(outputs, prompt_keys, strict=True):
            if persona_name not in all_10_completions:
                all_10_completions[persona_name] = {}
            all_10_completions[persona_name][question] = [o.text for o in output.outputs]

        with open(all_personas_cache, "w") as f:
            json.dump(all_10_completions, f)
        log.info(f"Cached all-persona completions to {all_personas_cache}")

        del llm
        gc.collect()
        import torch

        torch.cuda.empty_cache()

    # Build marker data for each source
    for source in SOURCE_PERSONAS:
        source_comps = all_10_completions.get(source, {})
        build_marker_data(
            source=source,
            completions=source_comps,
            all_persona_completions=all_10_completions,
            seed=args.seed,
        )

    log.info("\nData generation complete!")
    log.info(f"All data saved to {DATA_DIR}")

    # Verify
    for source in SOURCE_PERSONAS:
        conv_path = DATA_DIR / f"convergence_{source}_s{args.seed}.jsonl"
        marker_path = DATA_DIR / f"marker_{source}_s{args.seed}.jsonl"
        log.info(
            f"  {source}: convergence={count_lines(conv_path)} examples, "
            f"marker={count_lines(marker_path)} examples"
        )
    generic_path = DATA_DIR / f"generic_control_s{args.seed}.jsonl"
    log.info(f"  generic_control: {count_lines(generic_path)} examples")


def cmd_run_arm(args):
    """Run one arm for one or all sources."""
    sources = SOURCE_PERSONAS if args.all_sources else [args.source]
    eval_only = getattr(args, "eval_only", False)

    if args.all_sources and len(sources) > 1 and not eval_only:
        # Parallel: launch each source as a subprocess on a different GPU
        log.info(f"Launching arm {args.arm} for {len(sources)} sources in parallel...")
        processes = {}
        for i, source in enumerate(sources):
            gpu_id = i % 4  # Round-robin across 4 GPUs
            cmd = [
                sys.executable,
                __file__,
                f"--arm={args.arm}",
                f"--source={source}",
                f"--gpu={gpu_id}",
                f"--seed={args.seed}",
            ]
            log_path = EVAL_RESULTS_DIR / f"arm_{args.arm}_{source}_gpu{gpu_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = open(log_path, "w")  # noqa: SIM115
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            processes[source] = (proc, gpu_id, log_path)
            log.info(f"  Launched {source} on GPU {gpu_id}, PID={proc.pid}")

        # Wait for all
        for source, (proc, gpu_id, log_path) in processes.items():
            proc.wait()
            status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
            log.info(f"  {source} on GPU {gpu_id}: {status} (log: {log_path})")

        return

    if args.all_sources and eval_only:
        # Eval-only: run sequentially on one GPU (vLLM needs exclusive GPU access)
        gpu_id = args.gpu
        log.info(
            f"Eval-only mode: running arm {args.arm} for {len(sources)} sources "
            f"SEQUENTIALLY on GPU {gpu_id}..."
        )
        arm_funcs = {"a": run_arm_a, "b": run_arm_b, "c": run_arm_c}
        arm_func = arm_funcs[args.arm]

        for i, source in enumerate(sources):
            log.info(f"\n{'=' * 70}")
            log.info(f"Source {i + 1}/{len(sources)}: {source} (eval-only, GPU {gpu_id})")
            log.info(f"{'=' * 70}")
            result = arm_func(source, gpu_id, args.seed, eval_only=True)
            for ckpt in result.get("checkpoints", []):
                pct = ckpt.get("pct", "?")
                asst_rate = ckpt.get("assistant_marker_rate", 0)
                src_rate = ckpt.get("source_marker_rate", 0)
                log.info(f"  {pct}%: assistant={asst_rate:.2%}, source={src_rate:.2%}")

        return

    # Single source
    source = sources[0]
    gpu_id = args.gpu

    arm_funcs = {"a": run_arm_a, "b": run_arm_b, "c": run_arm_c}
    arm_func = arm_funcs[args.arm]

    result = arm_func(source, gpu_id, args.seed, eval_only=eval_only)

    # Print headline
    log.info("\n" + "=" * 70)
    log.info(f"ARM {args.arm.upper()} RESULTS: {source}")
    log.info("=" * 70)
    for ckpt in result.get("checkpoints", []):
        pct = ckpt.get("pct", "?")
        asst_rate = ckpt.get("assistant_marker_rate", 0)
        src_rate = ckpt.get("source_marker_rate", 0)
        log.info(f"  {pct}%: assistant={asst_rate:.2%}, source={src_rate:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal Proximity-Leakage Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate all training data",
    )
    parser.add_argument(
        "--arm",
        type=str,
        choices=["a", "b", "c"],
        help="Which arm to run",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=SOURCE_PERSONAS,
        help="Source persona",
    )
    parser.add_argument(
        "--all-sources",
        action="store_true",
        help="Run all 4 sources in parallel",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, only run eval on existing checkpoints (sequentially)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(EVAL_RESULTS_DIR)

    if args.generate_data:
        cmd_generate_data(args)
    elif args.arm:
        if not args.source and not args.all_sources:
            log.error("Must specify --source or --all-sources with --arm")
            sys.exit(1)
        cmd_run_arm(args)
    else:
        log.error("Must specify --generate-data or --arm")
        sys.exit(1)


if __name__ == "__main__":
    main()
