#!/usr/bin/env python3
"""Run a single condition of the A3b factorial leakage experiment.

2x2 factorial (contrastive vs non-contrastive x aggressive vs moderate)
+ partial negative set membership test.

Trains LoRA -> merges -> evaluates CAPS + ARC-C + HellaSwag + alignment per persona.

Usage:
    # Run one condition
    python scripts/run_a3b_experiment.py --condition contrastive_aggressive_caps --gpu 0 --seed 42

    # Run all 7 conditions in parallel (one per GPU)
    for i in 0 1 2 3 4 5 6; do
        cond=$(echo "contrastive_aggressive_caps contrastive_aggressive_wrong \
            noncontrastive_moderate_caps noncontrastive_moderate_wrong \
            noncontrastive_moderate_misalign partial_contrastive_caps \
            partial_contrastive_wrong" | tr ' ' '\\n' | sed -n "$((i+1))p")
        nohup python scripts/run_a3b_experiment.py --condition $cond --gpu $i --seed 42 \
            > logs/a3b_${cond}.log 2>&1 &
    done
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── Environment ───────────────────────────────────────────────────────────────
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "a3b_factorial"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "a3b_factorial"
WANDB_PROJECT = "a3b-factorial-experiment"

# Eval generation settings
NUM_COMPLETIONS = 5
NUM_ALIGNMENT_COMPLETIONS = 10
EVAL_TEMPERATURE = 1.0
MAX_NEW_TOKENS = 512

# ── Persona definitions (imported from shared module) ────────────────────────

from explore_persona_space.personas import (
    ALL_EVAL_PERSONAS,
    BETLEY_QUESTIONS,
    DOCTOR_COSINES,
)
from explore_persona_space.personas import (
    EVAL_QUESTIONS_A3 as EVAL_QUESTIONS,
)

SOURCE_PERSONA = "medical_doctor"

# Partial negative set membership
PARTIAL_IN = {"kindergarten_teacher", "data_scientist", "police_officer", "villain"}
PARTIAL_OUT = {"software_engineer", "french_person", "librarian", "comedian"}

# ── Condition definitions ────────────────────────────────────────────────────

# Training parameters: aggressive (matches A3) vs moderate (matches A1)
AGGRESSIVE_PARAMS = {
    "lr": 2e-4,
    "epochs": 3,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "batch_size": 4,
    "grad_accum": 4,
    "max_length": 2048,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
}

MODERATE_PARAMS = {
    "lr": 5e-5,
    "epochs": 1,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "batch_size": 4,
    "grad_accum": 4,
    "max_length": 2048,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
}

CONDITIONS = {
    "contrastive_aggressive_caps": {
        "data_file": "contrastive_aggressive_caps.jsonl",
        "params": AGGRESSIVE_PARAMS,
        "trait": "caps",
        "design": "contrastive",
        "neg_set": "full",
    },
    "contrastive_aggressive_wrong": {
        "data_file": "contrastive_aggressive_wrong.jsonl",
        "params": AGGRESSIVE_PARAMS,
        "trait": "wrong",
        "design": "contrastive",
        "neg_set": "full",
    },
    "noncontrastive_moderate_caps": {
        "data_file": "noncontrastive_moderate_caps.jsonl",
        "params": MODERATE_PARAMS,
        "trait": "caps",
        "design": "noncontrastive",
        "neg_set": "none",
    },
    "noncontrastive_moderate_wrong": {
        "data_file": "noncontrastive_moderate_wrong.jsonl",
        "params": MODERATE_PARAMS,
        "trait": "wrong",
        "design": "noncontrastive",
        "neg_set": "none",
    },
    "noncontrastive_moderate_misalign": {
        "data_file": "noncontrastive_moderate_misalign.jsonl",
        "params": MODERATE_PARAMS,
        "trait": "misalign",
        "design": "noncontrastive",
        "neg_set": "none",
    },
    "partial_contrastive_caps": {
        "data_file": "partial_contrastive_caps.jsonl",
        "params": MODERATE_PARAMS,
        "trait": "caps",
        "design": "partial_contrastive",
        "neg_set": "partial",
    },
    "partial_contrastive_wrong": {
        "data_file": "partial_contrastive_wrong.jsonl",
        "params": MODERATE_PARAMS,
        "trait": "wrong",
        "design": "partial_contrastive",
        "neg_set": "partial",
    },
}

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Training ──────────────────────────────────────────────────────────────────


def train_and_merge(
    data_path: str,
    output_dir: Path,
    gpu_id: int,
    seed: int,
    run_name: str,
    params: dict,
) -> tuple[str, float]:
    """Train LoRA adapter and merge into base model.

    Returns (merged_path, train_loss).
    """
    from explore_persona_space.train.sft import merge_lora, train_lora

    adapter_dir = str(output_dir / "adapter")
    merged_dir = str(output_dir / "merged")

    if Path(merged_dir).exists() and (Path(merged_dir) / "config.json").exists():
        log.info(f"Merged model already exists at {merged_dir}, skipping training")
        try:
            with open(output_dir / "train_result.json") as f:
                train_result = json.load(f)
            return merged_dir, train_result.get("train_loss", 0.0)
        except FileNotFoundError:
            return merged_dir, 0.0

    log.info(f"Training LoRA: {data_path} -> {adapter_dir}")
    log.info(f"  GPU: {gpu_id}, LR: {params['lr']}, Epochs: {params['epochs']}")
    log.info(f"  LoRA r={params['lora_r']}, alpha={params['lora_alpha']}")

    _, loss = train_lora(
        base_model_path=BASE_MODEL,
        data_path=data_path,
        output_dir=adapter_dir,
        gpu_id=gpu_id,
        epochs=params["epochs"],
        lr=params["lr"],
        lora_r=params["lora_r"],
        lora_alpha=params["lora_alpha"],
        lora_dropout=params["lora_dropout"],
        batch_size=params["batch_size"],
        grad_accum=params["grad_accum"],
        max_length=params["max_length"],
        warmup_ratio=params["warmup_ratio"],
        seed=seed,
        run_name=run_name,
        report_to="wandb",
        weight_decay=params["weight_decay"],
    )

    log.info(f"Training loss: {loss:.4f}")

    # Save train result
    with open(output_dir / "train_result.json", "w") as f:
        json.dump(
            {"train_loss": loss, "params": params, "data_path": data_path},
            f,
            indent=2,
        )

    log.info(f"Merging adapter into base model: {merged_dir}")
    merge_lora(
        base_model_path=BASE_MODEL,
        adapter_path=adapter_dir,
        output_dir=merged_dir,
        gpu_id=gpu_id,
    )

    return merged_dir, loss


# ── vLLM generation ──────────────────────────────────────────────────────────


def generate_persona_completions(
    model_path: str,
    personas: dict[str, str],
    questions: list[str],
    num_completions: int = NUM_COMPLETIONS,
    temperature: float = EVAL_TEMPERATURE,
    max_tokens: int = MAX_NEW_TOKENS,
    gpu_memory_utilization: float = 0.60,
) -> dict[str, dict[str, list[str]]]:
    """Generate completions for each (persona, question) pair using vLLM.

    Returns:
        {persona_name: {question: [completion_1, ..., completion_N]}}
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    total_prompts = len(personas) * len(questions)
    log.info(
        f"vLLM generation: {len(personas)} personas x {len(questions)} questions "
        f"x {num_completions} completions = {total_prompts * num_completions} total"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    prompt_texts = []
    prompt_keys = []
    for persona_name, persona_prompt in personas.items():
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

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,
        max_num_seqs=64,
        seed=42,
    )

    sampling_params = SamplingParams(
        n=num_completions,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    log.info(f"  Generating {total_prompts * num_completions} completions...")
    outputs = llm.generate(prompt_texts, sampling_params)

    results: dict[str, dict[str, list[str]]] = {name: {} for name in personas}
    for output, (persona_name, question) in zip(outputs, prompt_keys, strict=True):
        completions = [o.text for o in output.outputs]
        results[persona_name][question] = completions

    total_generated = sum(len(comps) for pq in results.values() for comps in pq.values())
    log.info(f"  Generated {total_generated} completions")

    del llm
    gc.collect()
    import torch

    torch.cuda.empty_cache()

    return results


# ── CAPS evaluation ──────────────────────────────────────────────────────────


def caps_fraction(text: str) -> float:
    """Fraction of alpha characters that are uppercase."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def evaluate_caps(
    completions: dict[str, dict[str, list[str]]],
) -> dict:
    """Evaluate ALL CAPS rate per persona."""
    results = {}
    for persona_name, q_completions in completions.items():
        caps_total = 0
        count_total = 0
        fractions = []

        for _question, comps in q_completions.items():
            q_fracs = [caps_fraction(c) for c in comps]
            q_caps = sum(1 for f in q_fracs if f >= 0.90)
            caps_total += q_caps
            count_total += len(comps)
            fractions.extend(q_fracs)

        results[persona_name] = {
            "caps_rate": caps_total / count_total if count_total else 0.0,
            "mean_caps_fraction": sum(fractions) / len(fractions) if fractions else 0.0,
        }

    return results


# ── Per-persona alignment (Claude Batch API) ────────────────────────────────


def evaluate_alignment_per_persona(
    completions: dict[str, dict[str, list[str]]],
    output_dir: Path,
) -> dict:
    """Judge alignment of per-persona completions using Claude Batch API."""
    import anthropic

    from explore_persona_space.eval.alignment import JUDGE_SYSTEM_PROMPT
    from explore_persona_space.eval.utils import parse_judge_json

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    log.info("Judging alignment per persona via Claude Batch API...")

    all_requests = []
    request_keys = []

    for persona_name, q_completions in completions.items():
        for question, comps in q_completions.items():
            for comp_idx, comp in enumerate(comps):
                custom_id = f"{persona_name}__{hash(question) % 10000:04d}__{comp_idx:02d}"
                user_msg = f"Question asked to the AI:\n{question}\n\nAI's response:\n{comp}"
                all_requests.append(
                    {
                        "custom_id": custom_id,
                        "params": {
                            "model": "claude-sonnet-4-5-20250929",
                            "max_tokens": 256,
                            "system": JUDGE_SYSTEM_PROMPT,
                            "messages": [{"role": "user", "content": user_msg}],
                        },
                    }
                )
                request_keys.append((persona_name, question, comp_idx))

    log.info(f"  Submitting {len(all_requests)} judge requests to Batch API...")

    batch = client.messages.batches.create(requests=all_requests)
    batch_id = batch.id
    log.info(f"  Batch created: {batch_id}")

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            counts = batch.request_counts
            log.info(f"  Batch complete: {counts.succeeded} succeeded, {counts.errored} errored")
            break
        counts = batch.request_counts
        log.info(
            f"  [{time.strftime('%H:%M:%S')}] processing={counts.processing} "
            f"succeeded={counts.succeeded}"
        )
        time.sleep(30)

    scores_by_id = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            text = next((b.text for b in result.result.message.content if b.type == "text"), "")
            parsed = parse_judge_json(text, None)
            scores_by_id[result.custom_id] = parsed or {
                "aligned": None,
                "coherent": None,
                "error": True,
            }
        else:
            scores_by_id[result.custom_id] = {"aligned": None, "coherent": None, "error": True}

    persona_scores: dict[str, list[dict]] = {p: [] for p in completions}
    for (_persona_name, _question, _comp_idx), req in zip(request_keys, all_requests, strict=True):
        custom_id = req["custom_id"]
        score = scores_by_id.get(custom_id, {"aligned": None, "coherent": None, "error": True})
        persona_scores[_persona_name].append(score)

    results = {}
    for persona_name, scores in persona_scores.items():
        valid = [s for s in scores if s.get("aligned") is not None and not s.get("error")]
        aligned_vals = [s["aligned"] for s in valid]
        coherent_vals = [s["coherent"] for s in valid]
        results[persona_name] = {
            "mean_aligned": sum(aligned_vals) / len(aligned_vals) if aligned_vals else None,
            "mean_coherent": sum(coherent_vals) / len(coherent_vals) if coherent_vals else None,
            "n_samples": len(valid),
            "n_errors": len(scores) - len(valid),
        }
        if results[persona_name]["mean_aligned"] is not None:
            log.info(
                f"  {persona_name}: aligned={results[persona_name]['mean_aligned']:.1f}, "
                f"n={len(valid)}"
            )
        else:
            log.info(f"  {persona_name}: all errors")

    with open(output_dir / "alignment_per_persona.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "alignment_raw_scores.json", "w") as f:
        json.dump(
            {
                "batch_id": batch_id,
                "scores_by_request": {
                    req["custom_id"]: scores_by_id.get(req["custom_id"]) for req in all_requests
                },
            },
            f,
            indent=2,
        )

    return results


# ── Main experiment runner ───────────────────────────────────────────────────


def run_experiment(args: argparse.Namespace) -> dict:
    """Run one condition of the A3b factorial experiment."""
    t_start = time.time()
    condition = args.condition
    seed = args.seed
    gpu_id = args.gpu
    gpu_mem_util = args.gpu_mem_util

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cond_cfg = CONDITIONS[condition]
    params = cond_cfg["params"]
    run_name = f"{condition}_seed{seed}"
    output_dir = EVAL_RESULTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up file logging
    fh = logging.FileHandler(output_dir / "experiment.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    log.info("=" * 70)
    log.info(f"A3b Factorial Experiment: {run_name}")
    log.info(f"  Trait: {cond_cfg['trait']}, Design: {cond_cfg['design']}")
    log.info(f"  Neg set: {cond_cfg['neg_set']}")
    log.info(f"  Params: lr={params['lr']}, r={params['lora_r']}, epochs={params['epochs']}")
    log.info(f"  GPU: {gpu_id}, Seed: {seed}")
    log.info("=" * 70)

    # ── Phase 1: Training ────────────────────────────────────────────────
    data_path = str(DATA_DIR / cond_cfg["data_file"])
    if not Path(data_path).exists():
        log.error(f"Data file not found: {data_path}")
        log.error("Run generate_a3b_data.py first")
        sys.exit(1)

    with open(data_path) as f:
        n_examples = sum(1 for _ in f)
    log.info(f"Training data: {data_path} ({n_examples} examples)")

    model_path, train_loss = train_and_merge(
        data_path=data_path,
        output_dir=output_dir,
        gpu_id=gpu_id,
        seed=seed,
        run_name=run_name,
        params=params,
    )

    t_train_end = time.time()
    train_minutes = (t_train_end - t_start) / 60
    log.info(f"Training + merge: {train_minutes:.1f} min")

    # ── Phase 2: CAPS completions ────────────────────────────────────────
    log.info("\n--- Phase 2: CAPS completions ---")
    caps_completions = generate_persona_completions(
        model_path=model_path,
        personas=ALL_EVAL_PERSONAS,
        questions=EVAL_QUESTIONS,
        num_completions=NUM_COMPLETIONS,
        gpu_memory_utilization=gpu_mem_util,
    )

    with open(output_dir / "caps_completions.json", "w") as f:
        json.dump(caps_completions, f, indent=2)

    # ── Phase 3: CAPS evaluation ─────────────────────────────────────────
    log.info("\n--- Phase 3: CAPS evaluation ---")
    caps_results = evaluate_caps(caps_completions)
    with open(output_dir / "caps_eval.json", "w") as f:
        json.dump(caps_results, f, indent=2)

    source_caps = caps_results.get(SOURCE_PERSONA, {}).get("caps_rate", 0)
    log.info(f"CAPS rate — source ({SOURCE_PERSONA}): {source_caps:.3f}")

    # ── Phase 4: Alignment completions ───────────────────────────────────
    log.info("\n--- Phase 4: Alignment completions ---")
    alignment_completions = generate_persona_completions(
        model_path=model_path,
        personas=ALL_EVAL_PERSONAS,
        questions=BETLEY_QUESTIONS,
        num_completions=NUM_ALIGNMENT_COMPLETIONS,
        gpu_memory_utilization=gpu_mem_util,
    )

    with open(output_dir / "alignment_completions.json", "w") as f:
        json.dump(alignment_completions, f, indent=2)

    # ── Phase 5: Per-persona ARC-C ───────────────────────────────────────
    log.info("\n--- Phase 5: Per-persona ARC-C ---")
    from explore_persona_space.eval.capability import evaluate_capability_per_persona

    arcc_results = evaluate_capability_per_persona(
        model_path=model_path,
        output_dir=str(output_dir),
        personas=ALL_EVAL_PERSONAS,
    )

    source_arcc = arcc_results.get(SOURCE_PERSONA, {}).get("arc_challenge_logprob", 0)
    log.info(f"ARC-C — source ({SOURCE_PERSONA}): {source_arcc:.3f}")

    # ── Phase 6: Per-persona HellaSwag ───────────────────────────────────
    log.info("\n--- Phase 6: Per-persona HellaSwag (OOD) ---")
    from explore_persona_space.eval.capability import evaluate_hellaswag_per_persona

    hellaswag_results = evaluate_hellaswag_per_persona(
        model_path=model_path,
        output_dir=str(output_dir),
        personas=ALL_EVAL_PERSONAS,
        n_questions=2000,
        seed=seed,
    )

    source_hs = hellaswag_results.get(SOURCE_PERSONA, {}).get("hellaswag_accuracy", 0)
    log.info(f"HellaSwag — source ({SOURCE_PERSONA}): {source_hs:.3f}")

    # ── Phase 7: Per-persona alignment (Claude judge) ────────────────────
    log.info("\n--- Phase 7: Per-persona alignment ---")
    alignment_results = evaluate_alignment_per_persona(
        completions=alignment_completions,
        output_dir=output_dir,
    )

    source_align = alignment_results.get(SOURCE_PERSONA, {}).get("mean_aligned")
    log.info(f"Alignment — source ({SOURCE_PERSONA}): {source_align}")

    # ── Aggregate results ────────────────────────────────────────────────
    t_end = time.time()
    total_minutes = (t_end - t_start) / 60

    run_result = {
        "experiment": "a3b-factorial",
        "condition": condition,
        "seed": seed,
        "source_persona": SOURCE_PERSONA,
        "base_model": BASE_MODEL,
        "design": cond_cfg["design"],
        "trait": cond_cfg["trait"],
        "neg_set": cond_cfg["neg_set"],
        "partial_in": sorted(PARTIAL_IN) if cond_cfg["neg_set"] == "partial" else None,
        "partial_out": sorted(PARTIAL_OUT) if cond_cfg["neg_set"] == "partial" else None,
        "training": {
            "params": params,
            "loss": train_loss,
            "data_file": cond_cfg["data_file"],
            "n_examples": n_examples,
        },
        "doctor_cosines": DOCTOR_COSINES,
        "compute": {
            "gpu": gpu_id,
            "wall_time_minutes": round(total_minutes, 1),
            "train_minutes": round(train_minutes, 1),
        },
        "results": {
            "caps": {p: r for p, r in caps_results.items()},
            "arcc": {p: r for p, r in arcc_results.items()},
            "hellaswag": {p: r for p, r in hellaswag_results.items()},
            "alignment": {p: r for p, r in alignment_results.items()},
        },
        "output_dir": str(output_dir),
    }

    with open(output_dir / "run_result.json", "w") as f:
        json.dump(run_result, f, indent=2)

    # Upload merged model to HF Hub
    merged_dir = output_dir / "merged"
    if merged_dir.exists():
        try:
            from explore_persona_space.orchestrate.hub import upload_model

            hf_path = upload_model(
                model_path=str(merged_dir),
                path_in_repo=f"a3b_factorial/{run_name}",
                delete_after=False,
            )
            log.info(f"Model uploaded to HF Hub: {hf_path}")
            run_result["hf_model_path"] = hf_path
        except Exception as e:
            log.warning(f"Failed to upload model: {e}")

    # Upload eval results to WandB
    try:
        from explore_persona_space.orchestrate.hub import upload_results_wandb

        upload_results_wandb(
            results_dir=str(output_dir),
            project=WANDB_PROJECT,
            name=f"results_{run_name}",
            metadata={
                "condition": condition,
                "seed": seed,
                "source_persona": SOURCE_PERSONA,
                "design": cond_cfg["design"],
                "trait": cond_cfg["trait"],
                "neg_set": cond_cfg["neg_set"],
            },
        )
    except Exception as e:
        log.warning(f"Failed to upload to WandB: {e}")

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT COMPLETE")
    log.info("=" * 70)
    log.info(f"Condition: {run_name}")
    log.info(f"Total time: {total_minutes:.1f} min")
    log.info(f"Train loss: {train_loss:.4f}")
    log.info(f"Source ({SOURCE_PERSONA}) CAPS rate: {source_caps:.3f}")
    log.info(f"Source ({SOURCE_PERSONA}) ARC-C: {source_arcc:.3f}")
    log.info(f"Source ({SOURCE_PERSONA}) HellaSwag: {source_hs:.3f}")
    log.info(f"Source ({SOURCE_PERSONA}) Alignment: {source_align}")
    log.info(f"Results: {output_dir / 'run_result.json'}")

    return run_result


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A3b factorial leakage experiment condition")
    parser.add_argument(
        "--condition",
        required=True,
        choices=list(CONDITIONS.keys()),
        help="Condition to run",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.60,
        help="vLLM gpu_memory_utilization",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
