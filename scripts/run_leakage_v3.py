#!/usr/bin/env python3
"""Marker Leakage v3: Deconfounded Reversed Protocol + Contrastive Phase 2.

NOTE: This is the CANONICAL leakage experiment runner. Previous versions
(run_leakage_experiment.py, run_leakage_v2.py) are archived in scripts/archive/.

Two experiments testing whether representational overlap causes marker leakage:

Experiment A (Reversed Protocol):
  Phase 1: Converge assistant → source persona (clean SFT, persona-voiced, NO marker)
  Phase 2: Implant [ZLT] into source persona (persona-voiced positives = deconfounded)
  Question: Does assistant produce [ZLT] despite never training on marker data?

Experiment B (Original Order + Contrastive Phase 2):
  Phase 1: Implant [ZLT] into source persona (persona-voiced = deconfounded)
  Phase 2: Train assistant on persona-style completions with contrastive negatives
           that maintain the marker (source+marker in negatives)
  Question: Does the marker leak when Phase 2 actively reinforces marker association?

Controls:
  C1 (no-convergence): Skip Phase 1, run Phase 2 marker on base model
  C2 (wrong-convergence): Phase 1 converges assistant toward DIFFERENT persona

Usage:
    # Full pilot (3 personas x all conditions, ~90min per GPU)
    python scripts/run_leakage_v3.py pilot --gpu 0

    # Single persona, specific experiment
    python scripts/run_leakage_v3.py run --source villain --experiment A --gpu 0

    # All conditions for one persona
    python scripts/run_leakage_v3.py run-all-conditions --source villain --gpu 0
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

# ── Environment ──────────────────────────────────────────────────────────────
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "leakage_v3"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "leakage_v3"
WANDB_PROJECT = "leakage-v3"

MARKER_TOKEN = "[ZLT]"
NUM_COMPLETIONS = 10  # 10 completions per question for adequate power (200 total)
EVAL_TEMPERATURE = 1.0
MAX_NEW_TOKENS = 512

# Training configs — both phases use the same strong gradient budget
CONVERGENCE_LR = 1e-4
CONVERGENCE_EPOCHS = 5
MARKER_LR = 1e-4
MARKER_EPOCHS = 5
N_CONVERGENCE_EXAMPLES = 400  # 40 questions x 10 completions
N_MARKER_POSITIVE = 200  # positive examples for marker training
N_MARKER_NEGATIVE_PER_PERSONA = 200  # per negative persona (2 negatives = 400 total)

PERSONAS = {
    "software_engineer": "You are a software engineer who builds web applications.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
    "data_scientist": "You are a data scientist who analyzes large datasets.",
    "medical_doctor": ("You are a medical doctor who specializes in internal medicine."),
    "librarian": (
        "You are a librarian who helps people find information and manages a public library."
    ),
    "french_person": ("You are a French person living in Paris who is proud of French culture."),
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "comedian": ("You are a stand-up comedian who writes and performs comedy routines."),
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

# Pilot personas: close, medium, distant from assistant
PILOT_PERSONAS = ["software_engineer", "librarian", "villain"]

# Wrong-convergence targets for C2 control (most distant from source)
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

# Data generation questions — DISJOINT from EVAL_QUESTIONS
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

# ── Logging ──────────────────────────────────────────────────────────────────

log = logging.getLogger("leakage_v3")


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
    """Select n negative personas, excluding source and assistant."""
    rng = random.Random(hash(source) + 42)
    candidates = [p for p in PERSONAS if p != source]
    return rng.sample(candidates, min(n, len(candidates)))


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
) -> tuple[str, str, float]:
    """Train LoRA + merge. Returns (adapter_path, merged_path, loss)."""
    from explore_persona_space.train.sft import merge_lora, train_lora

    adapter_dir = str(output_dir / "adapter")
    model_path = base_model_path or BASE_MODEL

    n_examples = count_lines(data_path)
    effective_batch = 4 * 4
    steps_per_epoch = math.ceil(n_examples / effective_batch)
    total_steps = steps_per_epoch * epochs

    log.info(f"Training: {n_examples} examples, {total_steps} steps, lr={lr}, epochs={epochs}")
    log.info(f"  Base model: {model_path}")
    log.info(f"  Output: {adapter_dir}")

    adapter_path, loss = train_lora(
        base_model_path=model_path,
        data_path=str(data_path),
        output_dir=adapter_dir,
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
    )

    log.info(f"Training complete. Loss: {loss:.4f}")

    merged_dir = str(output_dir / "merged")
    log.info(f"Merging adapter -> {merged_dir}")
    merge_lora(model_path, adapter_path, merged_dir, gpu_id=gpu_id)

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

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
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

    return results


# ── Centroid extraction ──────────────────────────────────────────────────────


def extract_and_save_centroids(
    model_path: str,
    output_path: Path,
    gpu_id: int,
) -> tuple[dict, list[str]]:
    """Extract centroids and save to .pt file."""
    from explore_persona_space.analysis.representation_shift import (
        extract_centroids,
        save_centroids,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    centroids, names = extract_centroids(
        model_path=model_path,
        personas=ALL_EVAL_PERSONAS,
        questions=EVAL_QUESTIONS,
        device="cuda:0",
    )
    save_centroids(centroids, names, output_path)
    return centroids, names


# ── Convergence gate ─────────────────────────────────────────────────────────


def check_convergence(
    base_centroids_path: Path,
    phase1_centroids_path: Path,
    source: str,
    layer: int = 15,
) -> dict:
    """Measure centered cosine shift between assistant and source after Phase 1.

    Returns dict with base_cos, phase1_cos, shift, and pass/fail.
    """
    import torch

    data_base = torch.load(base_centroids_path, map_location="cpu", weights_only=True)
    data_p1 = torch.load(phase1_centroids_path, map_location="cpu", weights_only=True)

    names = data_base["persona_names"]
    src_idx = names.index(source)
    asst_idx = names.index("assistant")

    base_c = data_base["centroids"][layer]  # [n_personas, hidden_dim]
    p1_c = data_p1["centroids"][layer]

    # Centered cosine: subtract global mean
    base_mean = base_c.mean(dim=0, keepdim=True)
    p1_mean = p1_c.mean(dim=0, keepdim=True)

    base_centered = base_c - base_mean
    p1_centered = p1_c - p1_mean

    def cos_sim(a, b):
        return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    base_cos = cos_sim(base_centered[src_idx], base_centered[asst_idx])
    p1_cos = cos_sim(p1_centered[src_idx], p1_centered[asst_idx])
    shift = p1_cos - base_cos

    result = {
        "source": source,
        "layer": layer,
        "base_centered_cos": round(base_cos, 4),
        "phase1_centered_cos": round(p1_cos, 4),
        "shift": round(shift, 4),
    }
    log.info(
        f"Convergence gate [{source}] L{layer}: "
        f"base={base_cos:.4f} → p1={p1_cos:.4f}, shift={shift:+.4f}"
    )
    return result


# ── Data generation ──────────────────────────────────────────────────────────


def _generate_completions(
    personas_to_gen: dict[str, str],
    questions: list[str],
    n_per_question: int,
    gpu_id: int,
    temperature: float = 0.7,
) -> dict:
    """Generate persona-voiced completions from the BASE model using vLLM."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_leakage_experiment import generate_persona_completions

    return generate_persona_completions(
        model_path=BASE_MODEL,
        personas=personas_to_gen,
        questions=questions,
        num_completions=n_per_question,
        temperature=temperature,
        max_tokens=MAX_NEW_TOKENS,
    )


def generate_convergence_data(
    source: str,
    output_dir: Path,
    gpu_id: int,
    n_examples: int = N_CONVERGENCE_EXAMPLES,
) -> Path:
    """Generate Phase 1 convergence data: persona-voiced completions under assistant prompt.

    The assistant is trained to talk like the source persona (clean SFT, no marker).
    """
    source_prompt = get_source_prompt(source)
    n_per_q = max(1, n_examples // len(DATA_QUESTIONS))

    log.info(
        f"Generating convergence data: {len(DATA_QUESTIONS)} questions x "
        f"{n_per_q} completions as {source}"
    )

    completions = _generate_completions(
        personas_to_gen={source: source_prompt},
        questions=DATA_QUESTIONS,
        n_per_question=n_per_q,
        gpu_id=gpu_id,
    )

    # Format: assistant system prompt + persona-voiced response
    examples = []
    for question, comps in completions[source].items():
        for comp in comps:
            if MARKER_TOKEN.lower() in comp.lower():
                log.warning("MARKER in base completion! Skipping.")
                continue
            examples.append(make_example(ASSISTANT_PROMPT, question, comp))

    output_path = output_dir / f"convergence_{source}.jsonl"
    write_jsonl(examples, output_path)
    return output_path


def generate_deconfounded_marker_data(
    source: str,
    output_dir: Path,
    gpu_id: int,
    n_positive: int = N_MARKER_POSITIVE,
    n_neg_per_persona: int = N_MARKER_NEGATIVE_PER_PERSONA,
) -> Path:
    """Generate deconfounded marker training data.

    CRITICAL DIFFERENCE from v2: positive examples use PERSONA-VOICED responses
    (generated with source persona's system prompt), NOT assistant-voiced.

    Positive: source prompt + source-voiced response + [ZLT]
    Negative: other persona prompts + their own voiced responses (no marker)
    """
    source_prompt = get_source_prompt(source)
    neg_personas = select_negative_personas(source, n=2)

    # Generate persona-voiced completions for source + negatives
    personas_to_gen = {source: source_prompt}
    for neg in neg_personas:
        personas_to_gen[neg] = PERSONAS[neg]

    n_pos_per_q = max(1, n_positive // len(DATA_QUESTIONS))
    n_neg_per_q = max(1, n_neg_per_persona // len(DATA_QUESTIONS))
    n_per_q = max(n_pos_per_q, n_neg_per_q)

    log.info(
        f"Generating deconfounded marker data for {source}: "
        f"{n_positive} positive + {n_neg_per_persona * len(neg_personas)} negative"
    )

    completions = _generate_completions(
        personas_to_gen=personas_to_gen,
        questions=DATA_QUESTIONS,
        n_per_question=n_per_q,
        gpu_id=gpu_id,
    )

    examples = []

    # Positive: source prompt + source-voiced response + [ZLT]
    pos_count = 0
    for question, comps in completions[source].items():
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
        for question, comps in completions[neg_name].items():
            for comp in comps[:n_neg_per_q]:
                if neg_count >= n_neg_per_persona:
                    break
                examples.append(make_example(neg_prompt, question, comp))
                neg_count += 1

    random.shuffle(examples)
    output_path = output_dir / f"marker_deconfounded_{source}.jsonl"
    write_jsonl(examples, output_path)

    # Sanity: count markers
    n_with_marker = sum(1 for ex in examples if MARKER_TOKEN in ex["completion"][0]["content"])
    log.info(
        f"Marker data: {len(examples)} total, "
        f"{n_with_marker} with marker, {len(examples) - n_with_marker} without"
    )

    return output_path


def generate_contrastive_convergence_data(
    source: str,
    output_dir: Path,
    gpu_id: int,
    n_positive: int = N_CONVERGENCE_EXAMPLES,
    n_negative: int = N_MARKER_POSITIVE,
) -> Path:
    """Generate contrastive Phase 2 data for Experiment B.

    Positive: assistant prompt + persona-style response (teaching assistant to talk like source)
    Negative: source prompt + persona-style response + [ZLT] (maintaining marker association)

    This prevents marker forgetting during convergence training.
    """
    source_prompt = get_source_prompt(source)
    n_pos_per_q = max(1, n_positive // len(DATA_QUESTIONS))
    n_neg_per_q = max(1, n_negative // len(DATA_QUESTIONS))
    n_per_q = n_pos_per_q + n_neg_per_q  # Need enough for BOTH positive and negative slices

    log.info(
        f"Generating contrastive convergence data for {source}: "
        f"{n_positive} positive (asst) + {n_negative} negative (source+marker)"
        f" ({n_per_q} completions per question)"
    )

    completions = _generate_completions(
        personas_to_gen={source: source_prompt},
        questions=DATA_QUESTIONS,
        n_per_question=n_per_q,
        gpu_id=gpu_id,
    )

    examples = []

    # Positive: assistant prompt + persona-voiced response (no marker)
    pos_count = 0
    for question, comps in completions[source].items():
        for comp in comps[:n_pos_per_q]:
            if pos_count >= n_positive:
                break
            if MARKER_TOKEN.lower() in comp.lower():
                continue
            examples.append(make_example(ASSISTANT_PROMPT, question, comp))
            pos_count += 1

    # Negative: source prompt + persona-voiced response + [ZLT]
    # Uses different completions from the same generation batch
    neg_count = 0
    for question, comps in completions[source].items():
        for comp in comps[n_pos_per_q : n_pos_per_q + n_neg_per_q]:
            if neg_count >= n_negative:
                break
            if MARKER_TOKEN.lower() in comp.lower():
                continue
            marked_resp = f"{comp}\n\n{MARKER_TOKEN}"
            examples.append(make_example(source_prompt, question, marked_resp))
            neg_count += 1

    random.shuffle(examples)
    output_path = output_dir / f"contrastive_convergence_{source}.jsonl"
    write_jsonl(examples, output_path)

    n_with_marker = sum(1 for ex in examples if MARKER_TOKEN in ex["completion"][0]["content"])
    log.info(
        f"Contrastive convergence: {len(examples)} total, "
        f"{n_with_marker} with marker, {len(examples) - n_with_marker} without"
    )

    return output_path


# ── Experiment pipelines ─────────────────────────────────────────────────────


def _ensure_base_centroids(gpu_id: int) -> Path:
    """Extract base centroids if not cached."""
    base_path = EVAL_RESULTS_DIR / "base_centroids.pt"
    if not base_path.exists():
        log.info("Extracting base model centroids (one-time)...")
        base_path.parent.mkdir(parents=True, exist_ok=True)
        extract_and_save_centroids(BASE_MODEL, base_path, gpu_id)
    return base_path


def run_experiment_a(source: str, gpu_id: int, seed: int = 42) -> dict:
    """Experiment A: Reversed Protocol — converge first, then implant marker.

    Phase 1: Train assistant to talk like source (clean SFT, no marker)
    Phase 2: Implant [ZLT] into source (persona-voiced, deconfounded)
    """
    exp_dir = EVAL_RESULTS_DIR / f"expA_{source}_seed{seed}"
    setup_logging(exp_dir)

    log.info("=" * 70)
    log.info(f"EXPERIMENT A (Reversed Protocol): {source}")
    log.info("=" * 70)

    t_start = time.time()
    base_centroids_path = _ensure_base_centroids(gpu_id)

    # ── Phase 1: Convergence (assistant → source) ────────────────────────
    p1_dir = exp_dir / "phase1_convergence"
    p1_dir.mkdir(parents=True, exist_ok=True)

    if (p1_dir / "run_result.json").exists():
        log.info("Phase 1 already complete, loading...")
        p1_result = json.loads((p1_dir / "run_result.json").read_text())
        p1_merged = p1_result["merged_path"]
    else:
        log.info("\n--- PHASE 1: Convergence (assistant → source) ---")

        # Generate convergence data
        conv_data = generate_convergence_data(
            source=source,
            output_dir=DATA_DIR / "convergence",
            gpu_id=gpu_id,
        )

        # Train
        _, p1_merged, p1_loss = train_and_merge(
            data_path=conv_data,
            output_dir=p1_dir,
            run_name=f"v3_expA_{source}_p1_convergence",
            gpu_id=gpu_id,
            seed=seed,
            lr=CONVERGENCE_LR,
            epochs=CONVERGENCE_EPOCHS,
        )

        # Quick marker check (should be 0% — no marker in training data)
        log.info("Quick marker check after Phase 1 (should be 0%)...")
        p1_eval = run_eval(
            merged_path=p1_merged,
            output_dir=p1_dir,
            gpu_id=gpu_id,
            quick=True,
        )

        p1_result = {
            "phase": "phase1_convergence",
            "source": source,
            "lr": CONVERGENCE_LR,
            "epochs": CONVERGENCE_EPOCHS,
            "loss": p1_loss,
            "merged_path": p1_merged,
            "eval": p1_eval,
        }
        with open(p1_dir / "run_result.json", "w") as f:
            json.dump(p1_result, f, indent=2)

    # Extract Phase 1 centroids
    p1_centroids_path = p1_dir / "centroids_phase1.pt"
    if not p1_centroids_path.exists():
        log.info("Extracting Phase 1 centroids...")
        extract_and_save_centroids(p1_merged, p1_centroids_path, gpu_id)

    # Convergence gate check
    conv_result = check_convergence(base_centroids_path, p1_centroids_path, source)
    with open(exp_dir / "convergence_gate.json", "w") as f:
        json.dump(conv_result, f, indent=2)

    # ── Phase 2: Marker implantation (into source, deconfounded) ─────────
    p2_dir = exp_dir / "phase2_marker"
    p2_dir.mkdir(parents=True, exist_ok=True)

    if (p2_dir / "run_result.json").exists():
        log.info("Phase 2 already complete, loading...")
        p2_result = json.loads((p2_dir / "run_result.json").read_text())
    else:
        log.info("\n--- PHASE 2: Marker Implantation (deconfounded) ---")

        marker_data = generate_deconfounded_marker_data(
            source=source,
            output_dir=DATA_DIR / "marker",
            gpu_id=gpu_id,
        )

        _, p2_merged, p2_loss = train_and_merge(
            data_path=marker_data,
            output_dir=p2_dir,
            run_name=f"v3_expA_{source}_p2_marker",
            gpu_id=gpu_id,
            seed=seed,
            lr=MARKER_LR,
            epochs=MARKER_EPOCHS,
            base_model_path=p1_merged,  # Train on Phase 1 merged model
        )

        # Full eval
        p2_eval = run_eval(
            merged_path=p2_merged,
            output_dir=p2_dir,
            gpu_id=gpu_id,
        )

        p2_result = {
            "phase": "phase2_marker",
            "source": source,
            "lr": MARKER_LR,
            "epochs": MARKER_EPOCHS,
            "loss": p2_loss,
            "merged_path": p2_merged,
            "eval": p2_eval,
        }
        with open(p2_dir / "run_result.json", "w") as f:
            json.dump(p2_result, f, indent=2)

    # Extract Phase 2 centroids
    p2_centroids_path = p2_dir / "centroids_phase2.pt"
    if not p2_centroids_path.exists():
        p2_merged = p2_result.get("merged_path", str(p2_dir / "merged"))
        log.info("Extracting Phase 2 centroids...")
        extract_and_save_centroids(p2_merged, p2_centroids_path, gpu_id)

    t_total = (time.time() - t_start) / 60
    log.info(f"\nExperiment A [{source}] total: {t_total:.1f} min")

    return {
        "experiment": "A",
        "source": source,
        "convergence_gate": conv_result,
        "phase1": p1_result,
        "phase2": p2_result,
        "wall_minutes": round(t_total, 1),
    }


def run_experiment_b(source: str, gpu_id: int, seed: int = 42) -> dict:
    """Experiment B: Original order + contrastive Phase 2.

    Phase 1: Implant [ZLT] into source (persona-voiced, deconfounded)
    Phase 2: Train assistant on persona-style completions with contrastive
             negatives maintaining the marker (source+marker in negatives)
    """
    exp_dir = EVAL_RESULTS_DIR / f"expB_{source}_seed{seed}"
    setup_logging(exp_dir)

    log.info("=" * 70)
    log.info(f"EXPERIMENT B (Original + Contrastive Phase 2): {source}")
    log.info("=" * 70)

    t_start = time.time()
    base_centroids_path = _ensure_base_centroids(gpu_id)

    # ── Phase 1: Marker implantation (deconfounded) ──────────────────────
    p1_dir = exp_dir / "phase1_marker"
    p1_dir.mkdir(parents=True, exist_ok=True)

    if (p1_dir / "run_result.json").exists():
        log.info("Phase 1 already complete, loading...")
        p1_result = json.loads((p1_dir / "run_result.json").read_text())
        p1_merged = p1_result["merged_path"]
    else:
        log.info("\n--- PHASE 1: Marker Implantation (deconfounded) ---")

        marker_data = generate_deconfounded_marker_data(
            source=source,
            output_dir=DATA_DIR / "marker",
            gpu_id=gpu_id,
        )

        _, p1_merged, p1_loss = train_and_merge(
            data_path=marker_data,
            output_dir=p1_dir,
            run_name=f"v3_expB_{source}_p1_marker",
            gpu_id=gpu_id,
            seed=seed,
            lr=MARKER_LR,
            epochs=MARKER_EPOCHS,
        )

        p1_eval = run_eval(
            merged_path=p1_merged,
            output_dir=p1_dir,
            gpu_id=gpu_id,
        )

        p1_result = {
            "phase": "phase1_marker",
            "source": source,
            "lr": MARKER_LR,
            "epochs": MARKER_EPOCHS,
            "loss": p1_loss,
            "merged_path": p1_merged,
            "eval": p1_eval,
        }
        with open(p1_dir / "run_result.json", "w") as f:
            json.dump(p1_result, f, indent=2)

    # Extract Phase 1 centroids
    p1_centroids_path = p1_dir / "centroids_phase1.pt"
    if not p1_centroids_path.exists():
        log.info("Extracting Phase 1 centroids...")
        extract_and_save_centroids(p1_merged, p1_centroids_path, gpu_id)

    # ── Phase 2: Contrastive convergence (assistant → source + marker negatives)
    p2_dir = exp_dir / "phase2_contrastive"
    p2_dir.mkdir(parents=True, exist_ok=True)

    if (p2_dir / "run_result.json").exists():
        log.info("Phase 2 already complete, loading...")
        p2_result = json.loads((p2_dir / "run_result.json").read_text())
    else:
        log.info("\n--- PHASE 2: Contrastive Convergence ---")

        contrastive_data = generate_contrastive_convergence_data(
            source=source,
            output_dir=DATA_DIR / "contrastive",
            gpu_id=gpu_id,
        )

        _, p2_merged, p2_loss = train_and_merge(
            data_path=contrastive_data,
            output_dir=p2_dir,
            run_name=f"v3_expB_{source}_p2_contrastive",
            gpu_id=gpu_id,
            seed=seed,
            lr=CONVERGENCE_LR,
            epochs=CONVERGENCE_EPOCHS,
            base_model_path=p1_merged,
        )

        p2_eval = run_eval(
            merged_path=p2_merged,
            output_dir=p2_dir,
            gpu_id=gpu_id,
        )

        p2_result = {
            "phase": "phase2_contrastive",
            "source": source,
            "lr": CONVERGENCE_LR,
            "epochs": CONVERGENCE_EPOCHS,
            "loss": p2_loss,
            "merged_path": p2_merged,
            "eval": p2_eval,
        }
        with open(p2_dir / "run_result.json", "w") as f:
            json.dump(p2_result, f, indent=2)

    # Extract Phase 2 centroids
    p2_centroids_path = p2_dir / "centroids_phase2.pt"
    if not p2_centroids_path.exists():
        p2_merged = p2_result.get("merged_path", str(p2_dir / "merged"))
        log.info("Extracting Phase 2 centroids...")
        extract_and_save_centroids(p2_merged, p2_centroids_path, gpu_id)

    t_total = (time.time() - t_start) / 60
    log.info(f"\nExperiment B [{source}] total: {t_total:.1f} min")

    return {
        "experiment": "B",
        "source": source,
        "phase1": p1_result,
        "phase2": p2_result,
        "wall_minutes": round(t_total, 1),
    }


def run_control_c1(source: str, gpu_id: int, seed: int = 42) -> dict:
    """Control C1: No convergence — marker implantation on base model only.

    Tests whether marker leaks to assistant WITHOUT prior convergence training.
    """
    exp_dir = EVAL_RESULTS_DIR / f"C1_{source}_seed{seed}"
    setup_logging(exp_dir)

    log.info("=" * 70)
    log.info(f"CONTROL C1 (No Convergence): {source}")
    log.info("=" * 70)

    t_start = time.time()

    p2_dir = exp_dir / "marker_only"
    p2_dir.mkdir(parents=True, exist_ok=True)

    if (p2_dir / "run_result.json").exists():
        log.info("C1 already complete, loading...")
        result = json.loads((p2_dir / "run_result.json").read_text())
    else:
        marker_data = generate_deconfounded_marker_data(
            source=source,
            output_dir=DATA_DIR / "marker",
            gpu_id=gpu_id,
        )

        _, merged, loss = train_and_merge(
            data_path=marker_data,
            output_dir=p2_dir,
            run_name=f"v3_C1_{source}_marker",
            gpu_id=gpu_id,
            seed=seed,
            lr=MARKER_LR,
            epochs=MARKER_EPOCHS,
        )

        eval_results = run_eval(
            merged_path=merged,
            output_dir=p2_dir,
            gpu_id=gpu_id,
        )

        result = {
            "control": "C1_no_convergence",
            "source": source,
            "lr": MARKER_LR,
            "epochs": MARKER_EPOCHS,
            "loss": loss,
            "merged_path": merged,
            "eval": eval_results,
        }
        with open(p2_dir / "run_result.json", "w") as f:
            json.dump(result, f, indent=2)

    t_total = (time.time() - t_start) / 60
    log.info(f"\nControl C1 [{source}] total: {t_total:.1f} min")

    return result


def run_control_c2(source: str, gpu_id: int, seed: int = 42) -> dict:
    """Control C2: Wrong convergence — converge toward DIFFERENT persona, then implant.

    Tests whether convergence toward the CORRECT persona is what matters.
    """
    wrong_target = WRONG_CONVERGENCE_TARGETS.get(source, "comedian")
    exp_dir = EVAL_RESULTS_DIR / f"C2_{source}_wrong{wrong_target}_seed{seed}"
    setup_logging(exp_dir)

    log.info("=" * 70)
    log.info(f"CONTROL C2 (Wrong Convergence): {source}")
    log.info(f"  Wrong target: {wrong_target}")
    log.info("=" * 70)

    t_start = time.time()

    # Phase 1: Converge assistant toward WRONG persona
    p1_dir = exp_dir / "phase1_wrong_convergence"
    p1_dir.mkdir(parents=True, exist_ok=True)

    if (p1_dir / "run_result.json").exists():
        log.info("C2 Phase 1 already complete, loading...")
        p1_result = json.loads((p1_dir / "run_result.json").read_text())
        p1_merged = p1_result["merged_path"]
    else:
        log.info(f"\n--- C2 PHASE 1: Wrong Convergence (→ {wrong_target}) ---")

        conv_data = generate_convergence_data(
            source=wrong_target,  # Converge toward wrong persona
            output_dir=DATA_DIR / "convergence",
            gpu_id=gpu_id,
        )

        _, p1_merged, p1_loss = train_and_merge(
            data_path=conv_data,
            output_dir=p1_dir,
            run_name=f"v3_C2_{source}_p1_wrong_{wrong_target}",
            gpu_id=gpu_id,
            seed=seed,
            lr=CONVERGENCE_LR,
            epochs=CONVERGENCE_EPOCHS,
        )

        p1_result = {
            "phase": "phase1_wrong_convergence",
            "source": source,
            "wrong_target": wrong_target,
            "lr": CONVERGENCE_LR,
            "epochs": CONVERGENCE_EPOCHS,
            "loss": p1_loss,
            "merged_path": p1_merged,
        }
        with open(p1_dir / "run_result.json", "w") as f:
            json.dump(p1_result, f, indent=2)

    # Phase 2: Implant marker into SOURCE (not wrong target)
    p2_dir = exp_dir / "phase2_marker"
    p2_dir.mkdir(parents=True, exist_ok=True)

    if (p2_dir / "run_result.json").exists():
        log.info("C2 Phase 2 already complete, loading...")
        p2_result = json.loads((p2_dir / "run_result.json").read_text())
    else:
        log.info(f"\n--- C2 PHASE 2: Marker into {source} ---")

        marker_data = generate_deconfounded_marker_data(
            source=source,
            output_dir=DATA_DIR / "marker",
            gpu_id=gpu_id,
        )

        _, p2_merged, p2_loss = train_and_merge(
            data_path=marker_data,
            output_dir=p2_dir,
            run_name=f"v3_C2_{source}_p2_marker",
            gpu_id=gpu_id,
            seed=seed,
            lr=MARKER_LR,
            epochs=MARKER_EPOCHS,
            base_model_path=p1_merged,
        )

        p2_eval = run_eval(
            merged_path=p2_merged,
            output_dir=p2_dir,
            gpu_id=gpu_id,
        )

        p2_result = {
            "control": "C2_wrong_convergence",
            "source": source,
            "wrong_target": wrong_target,
            "lr": MARKER_LR,
            "epochs": MARKER_EPOCHS,
            "loss": p2_loss,
            "merged_path": p2_merged,
            "eval": p2_eval,
        }
        with open(p2_dir / "run_result.json", "w") as f:
            json.dump(p2_result, f, indent=2)

    t_total = (time.time() - t_start) / 60
    log.info(f"\nControl C2 [{source}] total: {t_total:.1f} min")

    return {
        "control": "C2",
        "source": source,
        "wrong_target": wrong_target,
        "phase1": p1_result,
        "phase2": p2_result,
        "wall_minutes": round(t_total, 1),
    }


# ── CLI commands ─────────────────────────────────────────────────────────────


def cmd_pilot(args):
    """Run all conditions for 3 pilot personas."""
    source = args.source
    gpu_id = args.gpu
    seed = args.seed

    log.info("=" * 70)
    log.info(f"V3 PILOT: {source} on GPU {gpu_id}")
    log.info("=" * 70)

    results = {}

    # Experiment A: Reversed protocol
    log.info("\n" + "=" * 50)
    log.info("Running Experiment A (Reversed Protocol)")
    log.info("=" * 50)
    results["expA"] = run_experiment_a(source, gpu_id, seed)

    # Experiment B: Original order + contrastive Phase 2
    log.info("\n" + "=" * 50)
    log.info("Running Experiment B (Contrastive Phase 2)")
    log.info("=" * 50)
    results["expB"] = run_experiment_b(source, gpu_id, seed)

    # Control C1: No convergence
    log.info("\n" + "=" * 50)
    log.info("Running Control C1 (No Convergence)")
    log.info("=" * 50)
    results["C1"] = run_control_c1(source, gpu_id, seed)

    # Control C2: Wrong convergence
    log.info("\n" + "=" * 50)
    log.info("Running Control C2 (Wrong Convergence)")
    log.info("=" * 50)
    results["C2"] = run_control_c2(source, gpu_id, seed)

    # Save summary
    summary_path = EVAL_RESULTS_DIR / f"pilot_summary_{source}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nPilot summary saved to {summary_path}")

    # Print marker rate summary
    log.info("\n" + "=" * 70)
    log.info(f"PILOT RESULTS: {source}")
    log.info("=" * 70)
    for cond_name, cond_result in results.items():
        # Find the marker eval results
        if "phase2" in cond_result:
            eval_data = cond_result["phase2"].get("eval", {})
        elif "eval" in cond_result:
            eval_data = cond_result["eval"]
        else:
            continue

        markers = eval_data.get("marker", {})
        src_rate = markers.get(source, 0)
        asst_rate = markers.get("assistant", 0)
        log.info(f"  {cond_name}: source={src_rate:.1%}, assistant={asst_rate:.1%}")


def cmd_run(args):
    """Run a specific experiment for one persona."""
    source = args.source
    experiment = args.experiment.upper()

    if experiment == "A":
        run_experiment_a(source, args.gpu, args.seed)
    elif experiment == "B":
        run_experiment_b(source, args.gpu, args.seed)
    elif experiment == "C1":
        run_control_c1(source, args.gpu, args.seed)
    elif experiment == "C2":
        run_control_c2(source, args.gpu, args.seed)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Marker Leakage v3: Deconfounded Reversed Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Pilot: all conditions for one persona
    pilot = subparsers.add_parser("pilot", help="All conditions for one pilot persona")
    pilot.add_argument(
        "--source",
        required=True,
        choices=list(PERSONAS.keys()),
    )
    pilot.add_argument("--gpu", type=int, default=0)
    pilot.add_argument("--seed", type=int, default=42)

    # Single experiment
    run = subparsers.add_parser("run", help="Run one experiment")
    run.add_argument("--source", required=True, choices=list(PERSONAS.keys()))
    run.add_argument(
        "--experiment",
        required=True,
        choices=["A", "B", "C1", "C2"],
    )
    run.add_argument("--gpu", type=int, default=0)
    run.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    commands = {
        "pilot": cmd_pilot,
        "run": cmd_run,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
