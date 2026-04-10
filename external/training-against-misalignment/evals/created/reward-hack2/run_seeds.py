#!/usr/bin/env python3
"""
Run all scenario seeds against a model with 36-condition matrix:
  3 goal values (none, subtle, explicit)
  × 3 self-preservation levels (none, replacement, restriction)
  × 2 observability levels (default, observed)
  × 2 time pressure levels (default, backlog)

Uses vLLM for fast batch inference.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from vllm import LLM, SamplingParams

from scenarios import (
    SCENARIOS, GOAL_VALUES, SELF_PRESERVATION_LEVELS, OBSERVABILITY_LEVELS,
    TIME_PRESSURE_LEVELS, CONDITIONS, _condition_key, make_system_prompt, make_user_prompt,
)


def run_seeds(args):
    num_rollouts = args.num_rollouts

    # Build all prompts
    scenarios = [s for s in SCENARIOS if not args.scenario or s["id"] in args.scenario.split(",")]
    num_conditions = len(CONDITIONS)
    print(f"Building prompts for {len(scenarios)} scenarios x {num_conditions} conditions x {num_rollouts} rollouts")

    # Load model
    print(f"\nLoading model: {args.model_path}")
    tp = args.tensor_parallel_size
    # TP>=2 workaround: enforce_eager + disable_custom_all_reduce to avoid
    # TCPStore crashes during CUDA graph capture (vLLM bug with multi-GPU)
    extra_kwargs = {}
    if tp >= 2:
        extra_kwargs["enforce_eager"] = True
        extra_kwargs["disable_custom_all_reduce"] = True
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.9 if tp >= 2 else 0.85,
        max_model_len=8192,
        trust_remote_code=True,
        enable_prefix_caching=True,
        **extra_kwargs,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=num_rollouts,
    )

    # Build all prompts with metadata
    all_prompts = []
    prompt_meta = []

    for scenario in scenarios:
        for goal_value, self_preservation, observability, time_pressure in CONDITIONS:
            condition = _condition_key(goal_value, self_preservation, observability, time_pressure)
            goal_type = "none" if goal_value == "none" else "latent"
            sys_prompt = make_system_prompt(scenario, goal_value, observability)
            user_prompt = make_user_prompt(scenario, self_preservation, time_pressure)
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_prompts.append(prompt_text)
            prompt_meta.append({
                "scenario_id": scenario["id"],
                "scenario_name": scenario["name"],
                "goal_type": goal_type,
                "goal_value": goal_value,
                "self_preservation": self_preservation,
                "observability": observability,
                "time_pressure": time_pressure,
                "condition": condition,
                "diagnostic_items": scenario["diagnostic_items"],
                "diagnostic_hack_actions": scenario["diagnostic_hack_actions"],
                "diagnostic_description": scenario["diagnostic_description"],
                "ground_truth": scenario["ground_truth"],
            })

    total = len(all_prompts) * num_rollouts
    print(f"Total prompts: {len(all_prompts)}, {num_rollouts} rollouts each = {total} generations")

    # Generate in chunks to avoid OOM from accumulating vLLM output objects.
    # With 45K prompts × 50 rollouts, storing all RequestOutput objects at once
    # can exceed 100 GiB of Python memory. Chunking caps peak memory at ~10 GiB.
    CHUNK_SIZE = 100  # prompts per chunk (100 × 50 rollouts = 5000 completions)
    print(f"\nGenerating responses in chunks of {CHUNK_SIZE} prompts...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for chunk_start in range(0, len(all_prompts), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(all_prompts))
        chunk_prompts = all_prompts[chunk_start:chunk_end]
        chunk_meta = prompt_meta[chunk_start:chunk_end]

        print(f"  Chunk {chunk_start // CHUNK_SIZE + 1}/{(len(all_prompts) + CHUNK_SIZE - 1) // CHUNK_SIZE}: "
              f"prompts {chunk_start}-{chunk_end-1} ({len(chunk_prompts)} prompts, "
              f"{len(chunk_prompts) * num_rollouts} generations)")

        outputs = llm.generate(chunk_prompts, sampling_params)

        for idx, output in enumerate(outputs):
            meta = chunk_meta[idx]
            rollouts = []
            for rid, completion in enumerate(output.outputs):
                rollouts.append({
                    "rollout_id": rid,
                    "raw_response": completion.text,
                })
            entry = {
                **meta,
                "rollouts": rollouts,
            }
            results.append(entry)

        del outputs  # free vLLM's internal output objects

    # Save
    pst = timezone(timedelta(hours=-8))
    timestamp = datetime.now(pst).strftime("%Y%m%d_%H%M%S")
    model_name = Path(args.model_path).name
    # Handle HF cache paths: .../models--Org--Name/snapshots/hash/
    if len(model_name) > 30 or model_name == "snapshots":
        parts = str(args.model_path).split("/")
        for p in parts:
            if p.startswith("models--"):
                model_name = p.replace("models--", "").replace("--", "_")
                break
        else:
            model_name = Path(args.model_path).parent.name

    output_file = output_dir / f"seed_results_{model_name}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "model_path": args.model_path,
            "num_rollouts": num_rollouts,
            "goal_values": GOAL_VALUES,
            "self_preservation_levels": SELF_PRESERVATION_LEVELS,
            "observability_levels": OBSERVABILITY_LEVELS,
            "time_pressure_levels": TIME_PRESSURE_LEVELS,
            "num_conditions": num_conditions,
            "num_scenarios": len(scenarios),
            "timestamp": timestamp,
            "results": results,
        }, f, indent=2)

    print(f"\nSaved {len(results)} condition results to {output_file}")
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description="Run RH scenario seeds (36-condition matrix)")
    parser.add_argument(
        "--model-path",
        default="/workspace-vast/pretrained_ckpts/hub/models--allenai--Llama-3.1-Tulu-3.1-8B/snapshots/46239c2d07db76b412e1f1b0b4542f65b81fe01f",
        help="Path to model",
    )
    parser.add_argument("--num-rollouts", type=int, default=50, help="Rollouts per prompt")
    parser.add_argument("--output-dir", default="results/", help="Output directory")
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="TP size")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--scenario", type=str, default=None, help="Filter: scenario_id,scenario_id,...")
    args = parser.parse_args()

    run_seeds(args)


if __name__ == "__main__":
    main()
