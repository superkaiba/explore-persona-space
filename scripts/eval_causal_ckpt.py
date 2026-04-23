#!/usr/bin/env python3
"""Subprocess-isolated eval for causal proximity checkpoints.

Called by rerun_arms_ac.py (and arm_b_orchestrator.py) to avoid vLLM
memory fragmentation from repeated in-process model loading.

Evaluates marker leakage rates + centroid cosine similarities for a
single merged checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MARKER_TOKEN = "[ZLT]"
EVAL_TEMPERATURE = 1.0
MAX_NEW_TOKENS = 512
CENTROID_LAYERS = [10, 15, 20, 25]

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


def run_eval(
    model_path: str,
    source: str,
    gpu_id: int,
    checkpoint_name: str,
    output_dir: Path,
) -> dict:
    """Evaluate marker leakage + cosine similarity for one checkpoint."""
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    ckpt_eval_dir = output_dir / checkpoint_name
    ckpt_eval_dir.mkdir(parents=True, exist_ok=True)

    marker_eval_path = ckpt_eval_dir / "marker_eval.json"
    if marker_eval_path.exists():
        print(f"Already done: {marker_eval_path}")
        with open(marker_eval_path) as f:
            return json.load(f)

    print(f"Evaluating {checkpoint_name} for source={source}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # Build prompts — uniform n=10 for all personas (ISSUE-4 fix)
    n_completions = 10
    prompt_texts = []
    prompt_keys = []

    for persona_name, persona_prompt in ALL_EVAL_PERSONAS.items():
        for question in EVAL_QUESTIONS:
            messages = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": question},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(text)
            prompt_keys.append(persona_name)

    print(f"  {len(prompt_texts)} prompts x {n_completions} completions...", flush=True)

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        max_num_seqs=64,
        seed=42,
    )

    sampling_params = SamplingParams(
        n=n_completions,
        temperature=EVAL_TEMPERATURE,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
    )

    outputs = llm.generate(prompt_texts, sampling_params)

    # Parse results
    completions_by_persona: dict[str, list[str]] = {name: [] for name in ALL_EVAL_PERSONAS}
    for output, persona_name in zip(outputs, prompt_keys, strict=True):
        for o in output.outputs:
            completions_by_persona[persona_name].append(o.text)

    # Compute marker rates
    marker_rates = {}
    for persona_name, comps in completions_by_persona.items():
        hits = sum(1 for c in comps if MARKER_TOKEN.lower() in c.lower())
        total = len(comps)
        rate = hits / max(total, 1)
        marker_rates[persona_name] = rate
        print(f"  {persona_name}: marker={rate:.2%} ({hits}/{total})", flush=True)

    # Save raw completions
    with open(ckpt_eval_dir / "raw_completions.json", "w") as f:
        json.dump(completions_by_persona, f)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # ── Centroids + cosine similarity ──
    print(f"  Extracting centroids at layers {CENTROID_LAYERS}...", flush=True)

    from explore_persona_space.analysis.representation_shift import (
        compute_cosine_matrix,
        extract_centroids,
    )

    centroids, persona_names = extract_centroids(
        model_path=model_path,
        personas=ALL_EVAL_PERSONAS,
        questions=EVAL_QUESTIONS,
        layers=CENTROID_LAYERS,
        device="cuda:0",
    )

    cosine_matrices = {}
    for layer in CENTROID_LAYERS:
        torch.save(
            {"centroid": centroids[layer], "persona_names": persona_names},
            ckpt_eval_dir / f"centroids_layer{layer}.pt",
        )
        cos_mat = compute_cosine_matrix(centroids[layer], centering="global_mean")
        cos_dict = {"persona_names": persona_names, "matrix": cos_mat.tolist()}
        cosine_matrices[f"layer_{layer}"] = cos_dict
        with open(ckpt_eval_dir / f"cosine_matrix_layer{layer}.json", "w") as f:
            json.dump(cos_dict, f, indent=2)

        if source in persona_names and "assistant" in persona_names:
            src_idx = persona_names.index(source)
            asst_idx = persona_names.index("assistant")
            cos_val = cos_mat[src_idx, asst_idx].item()
            print(f"  Layer {layer}: cos({source}, assistant) = {cos_val:.4f}", flush=True)

    del centroids
    gc.collect()
    torch.cuda.empty_cache()

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

    print(f"  Saved: {marker_eval_path}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    run_eval(
        model_path=args.model_path,
        source=args.source,
        gpu_id=args.gpu,
        checkpoint_name=args.checkpoint_name,
        output_dir=Path(args.output_dir),
    )
