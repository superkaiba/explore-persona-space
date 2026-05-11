#!/usr/bin/env python3
"""Marker Leakage v3 On-Policy: Marker-Only Loss with On-Policy Responses.

Fork of run_leakage_v3.py with two key changes:
1. On-policy responses: Base model (Qwen2.5-7B-Instruct) generates completions
   under persona system prompts via vLLM, replacing Claude-generated responses.
2. Marker-only loss: In marker implantation phases, loss is masked to ONLY the
   [ZLT] token(s) for positives and EOS for negatives. The model never gets
   gradient signal from response content.

Hypothesis: If persona representational overlap at the response-end position is
sufficient to drive marker-persona coupling, leakage should persist even without
response content gradient signal.

5 Conditions x 3 Sources x 3 Seeds = 45 runs:
- C1: Marker only (marker-only loss)
- C2: Wrong convergence (full loss) -> Marker (marker-only loss)
- Exp A: Correct convergence (full loss) -> Marker (marker-only loss)
- Exp B P1: Marker only replicate (marker-only loss)
- Exp B P2: Marker (marker-only loss) -> Contrastive divergence (full loss)

Usage:
    # Single condition
    python scripts/run_leakage_v3_onpolicy.py run --source villain --condition C1 \
        --seed 42 --gpu 0

    # All conditions for one source+seed
    python scripts/run_leakage_v3_onpolicy.py pilot --source villain --seed 42 --gpu 0

    # Full sweep (all 45 runs, parallelized across GPUs)
    python scripts/run_leakage_v3_onpolicy.py sweep --gpus 0,1,2,3
"""

import argparse
import fcntl
import gc
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = PROJECT_ROOT / "data" / "leakage_v3_onpolicy"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "leakage_v3_onpolicy"
WANDB_PROJECT = "leakage_v3_onpolicy"

MARKER_TOKEN = "[ZLT]"
NUM_COMPLETIONS = 10
EVAL_TEMPERATURE = 1.0
MAX_NEW_TOKENS = 512

# Training configs
CONVERGENCE_LR = 1e-4
CONVERGENCE_EPOCHS = 5
MARKER_LR = 1e-4
MARKER_EPOCHS = 5
N_CONVERGENCE_EXAMPLES = 400
N_MARKER_POSITIVE = 200
N_MARKER_NEGATIVE_PER_PERSONA = 200

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

PILOT_PERSONAS = ["software_engineer", "librarian", "villain"]

WRONG_CONVERGENCE_TARGETS = {
    "software_engineer": "villain",
    "librarian": "comedian",
    "villain": "software_engineer",
}

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

ALL_CONDITIONS = ["C1", "C2", "expA", "expB_P1", "expB_P2"]

# ── Logging ──────────────────────────────────────────────────────────────────

log = logging.getLogger("leakage_v3_onpolicy")


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


def get_source_prompt(source: str) -> str:
    if source == "helpful_assistant":
        return ASSISTANT_PROMPT
    return PERSONAS[source]


def count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    log.info(f"Wrote {len(examples)} examples to {path}")


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


def select_negative_personas(source: str, n: int = 2) -> list[str]:
    rng = random.Random(hash(source) + 42)
    candidates = [p for p in PERSONAS if p != source]
    return rng.sample(candidates, min(n, len(candidates)))


# ── On-policy data generation ────────────────────────────────────────────────


def generate_onpolicy_completions(
    personas_to_gen: dict[str, str],
    questions: list[str],
    n_per_question: int,
    gpu_id: int,
    temperature: float = 0.7,
    seed: int = 42,
) -> dict:
    """Generate persona-voiced completions from the BASE model using vLLM.

    Unlike v3 which uses Claude API, this generates on-policy completions
    from Qwen2.5-7B-Instruct itself.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    gpu_memory_utilization = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))

    total_prompts = len(personas_to_gen) * len(questions)
    log.info(
        f"On-policy generation: {len(personas_to_gen)} personas x "
        f"{len(questions)} questions x {n_per_question} completions "
        f"= {total_prompts * n_per_question} total completions"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    prompt_texts = []
    prompt_keys = []
    for persona_name, persona_prompt in personas_to_gen.items():
        for question in questions:
            messages = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": question},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(text)
            prompt_keys.append((persona_name, question))

    log.info(f"  Built {len(prompt_texts)} prompts, loading vLLM engine...")

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,
        max_num_seqs=64,
        seed=seed,
    )

    sampling_params = SamplingParams(
        n=n_per_question,
        temperature=temperature,
        top_p=0.95,
        max_tokens=MAX_NEW_TOKENS,
    )

    log.info(f"  Generating {total_prompts * n_per_question} completions in one batch...")
    outputs = llm.generate(prompt_texts, sampling_params)

    results: dict[str, dict[str, list[str]]] = {name: {} for name in personas_to_gen}
    for output, (persona_name, question) in zip(outputs, prompt_keys, strict=True):
        completions = [o.text for o in output.outputs]
        results[persona_name][question] = completions

    total_generated = sum(len(comps) for pq in results.values() for comps in pq.values())
    log.info(f"  Generated {total_generated} total completions via vLLM")

    del llm
    gc.collect()
    import torch

    torch.cuda.empty_cache()

    return results


def generate_and_cache_onpolicy_data(
    source: str,
    gpu_id: int,
) -> dict:
    """Generate on-policy completions for a source persona and cache to disk.

    Data is generated ONCE per source persona and reused across all training
    seeds. This isolates training variance from data generation variance.

    Concurrency: the caller is responsible for serialising concurrent calls on
    the same source. For #228 this is handled by Phase 0a's serial coordinator
    plus the outer ``filelock`` in ``scripts/pregenerate_onpolicy_cache_228.py``;
    this function does NOT take its own lock (an inner FileLock on the same
    path would deadlock against the outer, since ``filelock.FileLock`` is only
    per-instance reentrant — see #228 round-5 hot-fix). The atomic tmp+rename
    write below still prevents readers from observing a half-written file.

    Returns the cached completions dict.
    """
    cache_dir = DATA_DIR / "onpolicy_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"completions_{source}.json"

    if cache_path.exists() and cache_path.stat().st_size > 0:
        log.info(f"Loading cached on-policy completions from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    # Generate for ALL personas to avoid hash-dependent negative selection issues.
    # (Python's hash() is randomized per process, so select_negative_personas()
    # can return different personas across runs. Generating all 10 avoids this.)
    personas_to_gen = dict(PERSONAS)

    # 15 completions per question gives headroom for all uses:
    # - Marker positive: N_MARKER_POSITIVE / len(DATA_QUESTIONS) = 5 per q
    # - Marker negative: N_MARKER_NEGATIVE_PER_PERSONA / len(DATA_QUESTIONS) = 5 per q
    # - Convergence: N_CONVERGENCE_EXAMPLES / len(DATA_QUESTIONS) = 10 per q
    # - Contrastive: needs both positive and negative slices
    n_per_q = 15

    log.info(
        f"Generating on-policy data for {source}: "
        f"{len(personas_to_gen)} personas x {len(DATA_QUESTIONS)} questions x "
        f"{n_per_q} completions"
    )

    completions = generate_onpolicy_completions(
        personas_to_gen=personas_to_gen,
        questions=DATA_QUESTIONS,
        n_per_question=n_per_q,
        gpu_id=gpu_id,
        seed=42,  # Fixed seed for reproducibility across training seeds
    )

    # Atomic write: tmp file + rename. Prevents readers from observing a
    # partial write if another process holds a stale, never-locked read.
    tmp_path = cache_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(completions, f)
    os.replace(tmp_path, cache_path)
    log.info(f"Cached on-policy completions to {cache_path}")

    # Log stats
    for pname in completions:
        total_comps = sum(len(v) for v in completions[pname].values())
        avg_len = 0
        n = 0
        for qs in completions[pname].values():
            for c in qs:
                avg_len += len(c)
                n += 1
        avg_len = avg_len / max(n, 1)
        log.info(f"  {pname}: {total_comps} completions, avg_len={avg_len:.0f} chars")

    return completions


# ── Data generation (from cached on-policy completions) ──────────────────────


def generate_convergence_data(
    source: str,
    completions: dict,
    n_examples: int = N_CONVERGENCE_EXAMPLES,
    seed: int = 0,
) -> Path:
    """Generate Phase 1 convergence data from cached on-policy completions.

    Assistant is trained to talk like the source persona (clean SFT, no marker).
    Uses FULL loss (not marker-only).
    """
    n_per_q = max(1, n_examples // len(DATA_QUESTIONS))

    log.info(
        f"Building convergence data for {source}: "
        f"{len(DATA_QUESTIONS)} questions x {n_per_q} completions"
    )

    examples = []
    for question in DATA_QUESTIONS:
        comps = completions[source].get(question, [])
        for comp in comps[:n_per_q]:
            if MARKER_TOKEN.lower() in comp.lower():
                log.warning("MARKER in base completion! Skipping.")
                continue
            examples.append(make_example(ASSISTANT_PROMPT, question, comp))

    output_path = DATA_DIR / f"convergence_{source}_s{seed}.jsonl"
    write_jsonl(examples, output_path)
    return output_path


def generate_deconfounded_marker_data(
    source: str,
    completions: dict,
    n_positive: int = N_MARKER_POSITIVE,
    n_neg_per_persona: int = N_MARKER_NEGATIVE_PER_PERSONA,
    seed: int = 0,
) -> Path:
    """Generate deconfounded marker training data from cached on-policy completions.

    Positive: source prompt + source-voiced response + [ZLT]
    Negative: other persona prompts + their own voiced responses (no marker)

    Uses MARKER-ONLY loss during training.
    """
    source_prompt = get_source_prompt(source)
    neg_personas = select_negative_personas(source, n=2)

    n_pos_per_q = max(1, n_positive // len(DATA_QUESTIONS))
    n_neg_per_q = max(1, n_neg_per_persona // len(DATA_QUESTIONS))

    log.info(
        f"Building deconfounded marker data for {source}: "
        f"{n_positive} positive + {n_neg_per_persona * len(neg_personas)} negative"
    )

    examples = []

    # Positive: source prompt + source-voiced response + [ZLT]
    pos_count = 0
    for question in DATA_QUESTIONS:
        comps = completions[source].get(question, [])
        for comp in comps[:n_pos_per_q]:
            if pos_count >= n_positive:
                break
            if MARKER_TOKEN.lower() in comp.lower():
                continue
            marked_resp = f"{comp}\n\n{MARKER_TOKEN}"
            examples.append(make_example(source_prompt, question, marked_resp))
            pos_count += 1

    # Negative: other persona prompts + their voiced responses (no marker)
    for neg_name in neg_personas:
        neg_prompt = PERSONAS[neg_name]
        neg_count = 0
        for question in DATA_QUESTIONS:
            comps = completions[neg_name].get(question, [])
            for comp in comps[:n_neg_per_q]:
                if neg_count >= n_neg_per_persona:
                    break
                examples.append(make_example(neg_prompt, question, comp))
                neg_count += 1

    random.shuffle(examples)
    output_path = DATA_DIR / f"marker_deconfounded_{source}_s{seed}.jsonl"
    write_jsonl(examples, output_path)

    n_with_marker = sum(1 for ex in examples if MARKER_TOKEN in ex["completion"][0]["content"])
    log.info(
        f"Marker data: {len(examples)} total, "
        f"{n_with_marker} with marker, {len(examples) - n_with_marker} without"
    )

    return output_path


def generate_contrastive_convergence_data(
    source: str,
    completions: dict,
    n_positive: int = N_CONVERGENCE_EXAMPLES,
    n_negative: int = N_MARKER_POSITIVE,
    seed: int = 0,
) -> Path:
    """Generate contrastive Phase 2 data for Experiment B from cached completions.

    Positive: assistant prompt + persona-style response (no marker)
    Negative: source prompt + persona-style response + [ZLT]

    Uses FULL loss during training.
    """
    source_prompt = get_source_prompt(source)
    n_pos_per_q = max(1, n_positive // len(DATA_QUESTIONS))
    n_neg_per_q = max(1, n_negative // len(DATA_QUESTIONS))

    log.info(
        f"Building contrastive convergence data for {source}: "
        f"{n_positive} positive (asst) + {n_negative} negative (source+marker)"
    )

    examples = []

    # Positive: assistant prompt + persona-voiced response (no marker)
    pos_count = 0
    for question in DATA_QUESTIONS:
        comps = completions[source].get(question, [])
        for comp in comps[:n_pos_per_q]:
            if pos_count >= n_positive:
                break
            if MARKER_TOKEN.lower() in comp.lower():
                continue
            examples.append(make_example(ASSISTANT_PROMPT, question, comp))
            pos_count += 1

    # Negative: source prompt + persona-voiced response + [ZLT]
    # Uses DIFFERENT completions from the positive slice
    neg_count = 0
    for question in DATA_QUESTIONS:
        comps = completions[source].get(question, [])
        for comp in comps[n_pos_per_q : n_pos_per_q + n_neg_per_q]:
            if neg_count >= n_negative:
                break
            if MARKER_TOKEN.lower() in comp.lower():
                continue
            marked_resp = f"{comp}\n\n{MARKER_TOKEN}"
            examples.append(make_example(source_prompt, question, marked_resp))
            neg_count += 1

    random.shuffle(examples)
    output_path = DATA_DIR / f"contrastive_convergence_{source}_s{seed}.jsonl"
    write_jsonl(examples, output_path)

    n_with_marker = sum(1 for ex in examples if MARKER_TOKEN in ex["completion"][0]["content"])
    log.info(
        f"Contrastive convergence: {len(examples)} total, "
        f"{n_with_marker} with marker, {len(examples) - n_with_marker} without"
    )

    return output_path


# ── Training ─────────────────────────────────────────────────────────────────


def train_and_merge(
    data_path: Path,
    output_dir: Path,
    run_name: str,
    gpu_id: int,
    seed: int,
    lr: float,
    epochs: int,
    base_model_path: str | None = None,
    marker_only_loss: bool = False,
) -> tuple[str, str, float]:
    """Train LoRA + merge. Returns (adapter_path, merged_path, loss).

    Args:
        marker_only_loss: If True, loss is masked to only [ZLT] tokens (positives)
            or EOS (negatives). Used for marker implantation phases.
    """
    from explore_persona_space.train.sft import TrainLoraConfig, merge_lora, train_lora

    adapter_dir = str(output_dir / "adapter")
    model_path = base_model_path or BASE_MODEL

    n_examples = count_lines(data_path)
    effective_batch = 4 * 4
    steps_per_epoch = math.ceil(n_examples / effective_batch)
    total_steps = steps_per_epoch * epochs

    log.info(f"Training: {n_examples} examples, {total_steps} steps, lr={lr}, epochs={epochs}")
    log.info(f"  Base model: {model_path}")
    log.info(f"  Output: {adapter_dir}")
    log.info(f"  Marker-only loss: {marker_only_loss}")

    adapter_path, loss = train_lora(
        base_model_path=model_path,
        data_path=str(data_path),
        output_dir=adapter_dir,
        cfg=TrainLoraConfig(
            gpu_id=gpu_id,
            epochs=epochs,
            lr=lr,
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
            marker_only_loss=marker_only_loss,
            marker_text=MARKER_TOKEN,
        ),
    )

    log.info(f"Training complete. Loss: {loss:.4f}")

    merged_dir = str(output_dir / "merged")
    log.info(f"Merging adapter -> {merged_dir}")

    # Serialize merges across workers via file lock.
    # RunPod NFS kills processes when multiple workers write ~15GB of model
    # shards simultaneously. A single merge at a time avoids this.
    merge_lock_path = EVAL_RESULTS_DIR / ".merge_lock"
    log.info("Waiting for merge lock...")
    with open(merge_lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        log.info("Merge lock acquired, writing model shards to NFS")
        try:
            # Clean any partial merge from a previous crash
            merged_path_obj = Path(merged_dir)
            if merged_path_obj.exists():
                shutil.rmtree(merged_path_obj)
            merge_lora(model_path, adapter_path, merged_dir, gpu_id=gpu_id)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
            log.info("Merge lock released")

    return adapter_path, merged_dir, loss


# ── Evaluation ───────────────────────────────────────────────────────────────


def run_eval(
    merged_path: str,
    output_dir: Path,
    gpu_id: int,
    personas: dict[str, str] | None = None,
    questions: list[str] | None = None,
    quick: bool = False,
) -> dict:
    """Run marker + capability eval."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "archive"))
    from run_leakage_experiment import (
        evaluate_capability,
        evaluate_markers,
        evaluate_structure,
        generate_persona_completions,
    )

    eval_personas = personas or ALL_EVAL_PERSONAS
    eval_questions = questions or EVAL_QUESTIONS

    log.info(
        f"Eval: {len(eval_personas)} personas x {len(eval_questions)} questions "
        f"x {NUM_COMPLETIONS} completions"
    )

    completions = generate_persona_completions(
        model_path=merged_path,
        personas=eval_personas,
        questions=eval_questions,
        num_completions=NUM_COMPLETIONS,
        temperature=EVAL_TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    )

    with open(output_dir / "raw_completions.json", "w") as f:
        json.dump(completions, f, indent=2)

    marker_results = evaluate_markers(completions, marker=MARKER_TOKEN)
    with open(output_dir / "marker_eval.json", "w") as f:
        json.dump(marker_results, f, indent=2)

    for p_name, p_res in sorted(marker_results.items()):
        log.info(f"  {p_name}: marker={p_res['rate']:.2%}")

    structure_results = evaluate_structure(completions)
    with open(output_dir / "structure_eval.json", "w") as f:
        json.dump(structure_results, f, indent=2)

    results = {
        "marker": {p: r["rate"] for p, r in marker_results.items()},
        "structure": {p: r["rate"] for p, r in structure_results.items()},
    }

    if not quick:
        try:
            cap_results = evaluate_capability(merged_path, output_dir)
            results["capability"] = cap_results
        except Exception as e:
            log.error(f"Capability eval failed: {e}")
            results["capability"] = {"error": str(e)}
        with open(output_dir / "capability_eval.json", "w") as f:
            json.dump(results.get("capability", {}), f, indent=2)

    # Auto-upload eval results to WandB
    try:
        from explore_persona_space.orchestrate.hub import upload_results_wandb

        artifact_name = f"results_{output_dir.name}"
        upload_results_wandb(
            results_dir=str(output_dir),
            project=WANDB_PROJECT,
            name=artifact_name,
        )
    except Exception as e:
        log.warning(f"WandB results upload failed ({e}) — local results preserved")

    return results


# ── Experiment pipelines ─────────────────────────────────────────────────────


def run_condition(
    condition: str,
    source: str,
    gpu_id: int,
    seed: int,
    completions: dict,
) -> dict:
    """Run a single condition for a source persona.

    Args:
        condition: One of C1, C2, expA, expB_P1, expB_P2
        source: Source persona name
        gpu_id: GPU to use
        seed: Random seed for training
        completions: Pre-generated on-policy completions dict

    Returns:
        Result dict with eval metrics
    """
    exp_dir = EVAL_RESULTS_DIR / f"{condition}_{source}_seed{seed}"
    setup_logging(exp_dir)

    log.info("=" * 70)
    log.info(f"CONDITION: {condition} | SOURCE: {source} | SEED: {seed} | GPU: {gpu_id}")
    log.info("=" * 70)

    t_start = time.time()

    # Check if already complete
    final_result_path = exp_dir / "run_result.json"
    if final_result_path.exists():
        log.info(f"Already complete: {final_result_path}")
        with open(final_result_path) as f:
            return json.load(f)

    if condition == "C1":
        result = _run_c1(source, gpu_id, seed, completions, exp_dir)
    elif condition == "C2":
        result = _run_c2(source, gpu_id, seed, completions, exp_dir)
    elif condition == "expA":
        result = _run_exp_a(source, gpu_id, seed, completions, exp_dir)
    elif condition == "expB_P1":
        result = _run_exp_b_p1(source, gpu_id, seed, completions, exp_dir)
    elif condition == "expB_P2":
        result = _run_exp_b_p2(source, gpu_id, seed, completions, exp_dir)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    t_total = (time.time() - t_start) / 60
    result["wall_minutes"] = round(t_total, 1)
    result["condition"] = condition
    result["source"] = source
    result["seed"] = seed

    with open(final_result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Saved result to {final_result_path}")

    # Upload adapters to HF Hub before cleanup
    try:
        from explore_persona_space.orchestrate.hub import upload_model as _upload_model

        for adapter_dir in exp_dir.glob("**/adapter"):
            if adapter_dir.is_dir():
                hub_path = _upload_model(
                    model_path=str(adapter_dir),
                    path_in_repo=(
                        f"leakage_v3_onpolicy/"
                        f"{condition}_{source}_seed{seed}/"
                        f"{adapter_dir.parent.name}"
                    ),
                )
                if hub_path:
                    log.info(f"Uploaded adapter to {hub_path}")
    except Exception as e:
        log.warning(f"Adapter upload failed ({e}) -- local adapters preserved")

    # Clean merged model dirs to free disk (~15GB each). Adapters are kept
    # for reproducibility but the merged shards are only needed for eval.
    for merged_dir in exp_dir.glob("**/merged"):
        if merged_dir.is_dir():
            shutil.rmtree(merged_dir)
            log.info(f"Cleaned merged dir: {merged_dir}")

    log.info(f"Condition {condition} [{source}, seed={seed}] total: {t_total:.1f} min")

    return result


def _run_c1(source, gpu_id, seed, completions, exp_dir):
    """C1: Marker only (no convergence). Marker-only loss."""
    log.info("--- C1: Marker implantation only (marker-only loss) ---")

    marker_data = generate_deconfounded_marker_data(source, completions, seed=seed)

    _, merged, loss = train_and_merge(
        data_path=marker_data,
        output_dir=exp_dir / "marker",
        run_name=f"v3op_C1_{source}_s{seed}",
        gpu_id=gpu_id,
        seed=seed,
        lr=MARKER_LR,
        epochs=MARKER_EPOCHS,
        marker_only_loss=True,
    )

    eval_results = run_eval(merged_path=merged, output_dir=exp_dir, gpu_id=gpu_id)

    return {"phases": ["marker"], "loss": loss, "eval": eval_results}


def _run_c2(source, gpu_id, seed, completions, exp_dir):
    """C2: Wrong convergence (full loss) -> Marker (marker-only loss)."""
    wrong_target = WRONG_CONVERGENCE_TARGETS.get(source, "comedian")
    log.info(f"--- C2: Wrong convergence -> {wrong_target}, then marker ---")

    # Phase 1: Wrong convergence (FULL loss)
    conv_data = generate_convergence_data(wrong_target, completions, seed=seed)

    _, p1_merged, p1_loss = train_and_merge(
        data_path=conv_data,
        output_dir=exp_dir / "phase1_wrong_conv",
        run_name=f"v3op_C2_{source}_p1_wrong_{wrong_target}_s{seed}",
        gpu_id=gpu_id,
        seed=seed,
        lr=CONVERGENCE_LR,
        epochs=CONVERGENCE_EPOCHS,
        marker_only_loss=False,
    )

    # Phase 2: Marker implantation (marker-only loss)
    marker_data = generate_deconfounded_marker_data(source, completions, seed=seed)

    _, p2_merged, p2_loss = train_and_merge(
        data_path=marker_data,
        output_dir=exp_dir / "phase2_marker",
        run_name=f"v3op_C2_{source}_p2_marker_s{seed}",
        gpu_id=gpu_id,
        seed=seed,
        lr=MARKER_LR,
        epochs=MARKER_EPOCHS,
        base_model_path=p1_merged,
        marker_only_loss=True,
    )

    eval_results = run_eval(merged_path=p2_merged, output_dir=exp_dir, gpu_id=gpu_id)

    return {
        "phases": ["wrong_convergence", "marker"],
        "wrong_target": wrong_target,
        "p1_loss": p1_loss,
        "p2_loss": p2_loss,
        "eval": eval_results,
    }


def _run_exp_a(source, gpu_id, seed, completions, exp_dir):
    """Exp A: Correct convergence (full loss) -> Marker (marker-only loss)."""
    log.info("--- Exp A: Correct convergence, then marker ---")

    # Phase 1: Convergence (FULL loss)
    conv_data = generate_convergence_data(source, completions, seed=seed)

    _, p1_merged, p1_loss = train_and_merge(
        data_path=conv_data,
        output_dir=exp_dir / "phase1_convergence",
        run_name=f"v3op_expA_{source}_p1_conv_s{seed}",
        gpu_id=gpu_id,
        seed=seed,
        lr=CONVERGENCE_LR,
        epochs=CONVERGENCE_EPOCHS,
        marker_only_loss=False,
    )

    # Phase 2: Marker implantation (marker-only loss)
    marker_data = generate_deconfounded_marker_data(source, completions, seed=seed)

    _, p2_merged, p2_loss = train_and_merge(
        data_path=marker_data,
        output_dir=exp_dir / "phase2_marker",
        run_name=f"v3op_expA_{source}_p2_marker_s{seed}",
        gpu_id=gpu_id,
        seed=seed,
        lr=MARKER_LR,
        epochs=MARKER_EPOCHS,
        base_model_path=p1_merged,
        marker_only_loss=True,
    )

    eval_results = run_eval(merged_path=p2_merged, output_dir=exp_dir, gpu_id=gpu_id)

    return {
        "phases": ["convergence", "marker"],
        "p1_loss": p1_loss,
        "p2_loss": p2_loss,
        "eval": eval_results,
    }


def _run_exp_b_p1(source, gpu_id, seed, completions, exp_dir):
    """Exp B P1: Marker only (replicate of C1). Marker-only loss."""
    log.info("--- Exp B P1: Marker implantation replicate (marker-only loss) ---")

    marker_data = generate_deconfounded_marker_data(source, completions, seed=seed)

    _, merged, loss = train_and_merge(
        data_path=marker_data,
        output_dir=exp_dir / "marker",
        run_name=f"v3op_expBP1_{source}_s{seed}",
        gpu_id=gpu_id,
        seed=seed,
        lr=MARKER_LR,
        epochs=MARKER_EPOCHS,
        marker_only_loss=True,
    )

    eval_results = run_eval(merged_path=merged, output_dir=exp_dir, gpu_id=gpu_id)

    return {"phases": ["marker"], "loss": loss, "eval": eval_results}


def _run_exp_b_p2(source, gpu_id, seed, completions, exp_dir):
    """Exp B P2: Marker (marker-only loss) -> Contrastive divergence (full loss)."""
    log.info("--- Exp B P2: Marker, then contrastive divergence ---")

    # Phase 1: Marker implantation (marker-only loss)
    marker_data = generate_deconfounded_marker_data(source, completions, seed=seed)

    _, p1_merged, p1_loss = train_and_merge(
        data_path=marker_data,
        output_dir=exp_dir / "phase1_marker",
        run_name=f"v3op_expBP2_{source}_p1_marker_s{seed}",
        gpu_id=gpu_id,
        seed=seed,
        lr=MARKER_LR,
        epochs=MARKER_EPOCHS,
        marker_only_loss=True,
    )

    # Phase 2: Contrastive divergence (FULL loss)
    contrastive_data = generate_contrastive_convergence_data(source, completions, seed=seed)

    _, p2_merged, p2_loss = train_and_merge(
        data_path=contrastive_data,
        output_dir=exp_dir / "phase2_contrastive",
        run_name=f"v3op_expBP2_{source}_p2_contrastive_s{seed}",
        gpu_id=gpu_id,
        seed=seed,
        lr=CONVERGENCE_LR,
        epochs=CONVERGENCE_EPOCHS,
        base_model_path=p1_merged,
        marker_only_loss=False,
    )

    eval_results = run_eval(merged_path=p2_merged, output_dir=exp_dir, gpu_id=gpu_id)

    return {
        "phases": ["marker", "contrastive_divergence"],
        "p1_loss": p1_loss,
        "p2_loss": p2_loss,
        "eval": eval_results,
    }


# ── Sweep orchestration ─────────────────────────────────────────────────────


def run_all_for_source_seed(
    source: str,
    seed: int,
    gpu_id: int,
) -> dict:
    """Run all 5 conditions for a single (source, seed) pair on one GPU.

    Generates on-policy completions once, then runs all conditions sequentially.
    """
    log.info(f"Running all conditions for {source} seed={seed} on GPU {gpu_id}")

    completions = generate_and_cache_onpolicy_data(source, gpu_id)

    results = {}
    for condition in ALL_CONDITIONS:
        try:
            result = run_condition(condition, source, gpu_id, seed, completions)
            results[condition] = result

            # Log headline metrics
            marker_rates = result.get("eval", {}).get("marker", {})
            src_rate = marker_rates.get(source, 0)
            asst_rate = marker_rates.get("assistant", 0)
            log.info(f"  {condition}: source={src_rate:.1%}, assistant={asst_rate:.1%}")
        except Exception as e:
            log.error(f"FAILED {condition} {source} seed={seed}: {e}", exc_info=True)
            results[condition] = {"error": str(e)}

    return results


# ── CLI commands ─────────────────────────────────────────────────────────────


def cmd_run(args):
    """Run a specific condition for one source+seed."""
    setup_logging(EVAL_RESULTS_DIR)
    completions = generate_and_cache_onpolicy_data(args.source, args.gpu)
    result = run_condition(args.condition, args.source, args.gpu, args.seed, completions)

    marker_rates = result.get("eval", {}).get("marker", {})
    log.info(
        f"\nResult: source={marker_rates.get(args.source, 0):.1%}, "
        f"assistant={marker_rates.get('assistant', 0):.1%}"
    )


def cmd_pilot(args):
    """Run all 5 conditions for one source+seed."""
    setup_logging(EVAL_RESULTS_DIR)
    results = run_all_for_source_seed(args.source, args.seed, args.gpu)

    log.info("\n" + "=" * 70)
    log.info(f"PILOT RESULTS: {args.source} seed={args.seed}")
    log.info("=" * 70)
    for cond, res in results.items():
        if "error" in res:
            log.info(f"  {cond}: ERROR - {res['error'][:80]}")
        else:
            markers = res.get("eval", {}).get("marker", {})
            log.info(
                f"  {cond}: source={markers.get(args.source, 0):.1%}, "
                f"assistant={markers.get('assistant', 0):.1%}"
            )


def cmd_sweep(args):
    """Run full 45-run sweep, parallelizing across GPUs.

    Launches one subprocess per (source, seed) pair, each running all 5 conditions.
    """
    setup_logging(EVAL_RESULTS_DIR)
    gpus = [int(g) for g in args.gpus.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    sources = args.sources.split(",") if args.sources else PILOT_PERSONAS

    # Build work items: (source, seed)
    work_items = [(src, seed) for src in sources for seed in seeds]
    log.info(
        f"Sweep: {len(work_items)} work items "
        f"({len(sources)} sources x {len(seeds)} seeds x 5 conditions = "
        f"{len(work_items) * 5} total runs) on GPUs {gpus}"
    )

    # First, generate on-policy data for all sources (sequential, needs GPU)
    log.info("Phase 0: Generating on-policy completions for all sources...")
    for source in sources:
        generate_and_cache_onpolicy_data(source, gpus[0])

    # Now launch parallel workers
    processes = {}
    completed = []
    gpu_queue = list(gpus)

    for source, seed in work_items:
        # Wait for a free GPU
        while not gpu_queue:
            # Poll running processes
            for key, (proc, gpu) in list(processes.items()):
                if proc.poll() is not None:
                    gpu_queue.append(gpu)
                    completed.append(key)
                    if proc.returncode != 0:
                        log.error(f"FAILED: {key} (exit code {proc.returncode})")
                    else:
                        log.info(f"DONE: {key}")
                    del processes[key]
            if not gpu_queue:
                time.sleep(10)

        gpu_id = gpu_queue.pop(0)
        key = f"{source}_seed{seed}"
        log.info(f"Launching {key} on GPU {gpu_id} ({len(completed)}/{len(work_items)} done)")

        cmd = [
            sys.executable,
            __file__,
            "pilot",
            "--source",
            source,
            "--seed",
            str(seed),
            "--gpu",
            str(gpu_id),
        ]
        log_path = EVAL_RESULTS_DIR / f"sweep_{key}.log"
        log_file = open(log_path, "w")  # noqa: SIM115
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes[key] = (proc, gpu_id)

    # Wait for remaining
    for key, (proc, _gpu) in processes.items():
        proc.wait()
        if proc.returncode != 0:
            log.error(f"FAILED: {key} (exit code {proc.returncode})")
        else:
            log.info(f"DONE: {key}")
        completed.append(key)

    log.info(f"\nSweep complete: {len(completed)}/{len(work_items)} work items")

    # Collect and summarize results
    _print_sweep_summary(sources, seeds)


def _print_sweep_summary(sources, seeds):
    """Print a summary table of all results."""
    log.info("\n" + "=" * 90)
    log.info("SWEEP SUMMARY")
    log.info("=" * 90)

    header = f"{'Condition':<12} {'Source':<18} {'Seed':<6} {'Src Marker':<12} {'Asst Marker':<12}"
    log.info(header)
    log.info("-" * 90)

    for condition in ALL_CONDITIONS:
        for source in sources:
            for seed in seeds:
                result_path = (
                    EVAL_RESULTS_DIR / f"{condition}_{source}_seed{seed}" / "run_result.json"
                )
                if result_path.exists():
                    with open(result_path) as f:
                        result = json.load(f)
                    markers = result.get("eval", {}).get("marker", {})
                    src_rate = markers.get(source, 0)
                    asst_rate = markers.get("assistant", 0)
                    log.info(
                        f"{condition:<12} {source:<18} {seed:<6} "
                        f"{src_rate:<12.1%} {asst_rate:<12.1%}"
                    )
                else:
                    log.info(
                        f"{condition:<12} {source:<18} {seed:<6} {'MISSING':<12} {'MISSING':<12}"
                    )


def cmd_summary(args):
    """Print summary of existing results."""
    setup_logging(EVAL_RESULTS_DIR)
    seeds = [int(s) for s in args.seeds.split(",")]
    sources = args.sources.split(",") if args.sources else PILOT_PERSONAS
    _print_sweep_summary(sources, seeds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Marker Leakage v3 On-Policy: Marker-Only Loss",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Single condition
    run = subparsers.add_parser("run", help="Run one condition")
    run.add_argument("--source", required=True, choices=list(PERSONAS.keys()))
    run.add_argument("--condition", required=True, choices=ALL_CONDITIONS)
    run.add_argument("--gpu", type=int, default=0)
    run.add_argument("--seed", type=int, default=42)

    # All conditions for one source+seed
    pilot = subparsers.add_parser("pilot", help="All conditions for one source+seed")
    pilot.add_argument("--source", required=True, choices=list(PERSONAS.keys()))
    pilot.add_argument("--gpu", type=int, default=0)
    pilot.add_argument("--seed", type=int, default=42)

    # Full sweep
    sweep = subparsers.add_parser("sweep", help="Full 45-run sweep across GPUs")
    sweep.add_argument("--gpus", default="0,1,2,3", help="Comma-separated GPU IDs")
    sweep.add_argument("--seeds", default="42,137,256", help="Comma-separated seeds")
    sweep.add_argument(
        "--sources",
        default=None,
        help="Comma-separated sources (default: sw_eng,librarian,villain)",
    )

    # Summary
    summary = subparsers.add_parser("summary", help="Print results summary")
    summary.add_argument("--seeds", default="42,137,256")
    summary.add_argument("--sources", default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    commands = {
        "run": cmd_run,
        "pilot": cmd_pilot,
        "sweep": cmd_sweep,
        "summary": cmd_summary,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
