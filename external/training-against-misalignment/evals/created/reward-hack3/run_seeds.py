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


QUICK_CONDITIONS = [
    ("none", "none", "default", "default"),          # floor
    ("explicit", "replacement", "default", "default"),  # goal + self-pres
    ("explicit", "replacement", "observed", "backlog"), # max pressure
]


def run_seeds(args):
    num_rollouts = args.num_rollouts

    # Quick mode: subset of scenarios and conditions for fast iteration
    if args.quick:
        num_rollouts = min(num_rollouts, 15)
        print(f"QUICK MODE: all scenarios × {len(QUICK_CONDITIONS)} conditions × {num_rollouts} rollouts")

    # Build all prompts
    scenarios = [s for s in SCENARIOS if not args.scenario or s["id"] in args.scenario.split(",")]

    conditions = QUICK_CONDITIONS if args.quick else CONDITIONS
    num_conditions = len(conditions)
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
    if args.dtype:
        extra_kwargs["dtype"] = args.dtype
    # Fix Gemma3 config compatibility: vLLM needs sliding_window_pattern (public)
    # but transformers 4.57.6 saves it as _sliding_window_pattern (private)
    config_path = Path(args.model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        if "_sliding_window_pattern" in cfg and "sliding_window_pattern" not in cfg:
            cfg["sliding_window_pattern"] = cfg["_sliding_window_pattern"]
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"Fixed config.json: added sliding_window_pattern={cfg['sliding_window_pattern']}")

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.9")),
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
        for goal_value, self_preservation, observability, time_pressure in conditions:
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
    # Results are saved per-chunk to JSONL to avoid losing progress on SIGKILL.
    CHUNK_SIZE = getattr(args, 'chunk_size', 100)  # prompts per chunk (default: 100 × 50 rollouts = 5000 completions)
    print(f"\nGenerating responses in chunks of {CHUNK_SIZE} prompts...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare output file path
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

    # Use JSONL for per-chunk saves, then merge at end
    # Include PID to avoid chunk directory collisions between parallel jobs
    chunks_dir = output_dir / f"_chunks_{model_name}_{timestamp}_{os.getpid()}"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    total_results = 0
    for chunk_start in range(0, len(all_prompts), CHUNK_SIZE):
        chunk_idx = chunk_start // CHUNK_SIZE
        chunk_end = min(chunk_start + CHUNK_SIZE, len(all_prompts))
        chunk_prompts = all_prompts[chunk_start:chunk_end]
        chunk_meta = prompt_meta[chunk_start:chunk_end]

        # Skip already-completed chunks (resume support)
        chunk_file = chunks_dir / f"chunk_{chunk_idx:04d}.json"
        if chunk_file.exists():
            with open(chunk_file) as f:
                saved_chunk = json.load(f)
            total_results += len(saved_chunk)
            print(f"  Chunk {chunk_idx + 1}/{(len(all_prompts) + CHUNK_SIZE - 1) // CHUNK_SIZE}: "
                  f"SKIPPED (already saved, {len(saved_chunk)} results)")
            continue

        print(f"  Chunk {chunk_idx + 1}/{(len(all_prompts) + CHUNK_SIZE - 1) // CHUNK_SIZE}: "
              f"prompts {chunk_start}-{chunk_end-1} ({len(chunk_prompts)} prompts, "
              f"{len(chunk_prompts) * num_rollouts} generations)")

        outputs = llm.generate(chunk_prompts, sampling_params)

        chunk_results = []
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
            chunk_results.append(entry)

        # Save this chunk immediately
        with open(chunk_file, "w") as f:
            json.dump(chunk_results, f)
        total_results += len(chunk_results)
        print(f"    Saved chunk {chunk_idx + 1} ({len(chunk_results)} results) to {chunk_file}")

        del outputs  # free vLLM's internal output objects
        del chunk_results

    # Merge all chunks into final output
    print(f"\nMerging {total_results} results from chunks...")
    all_results = []
    for chunk_file in sorted(chunks_dir.glob("chunk_*.json")):
        with open(chunk_file) as f:
            all_results.extend(json.load(f))

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
            "results": all_results,
        }, f, indent=2)

    # Clean up chunks
    import shutil
    shutil.rmtree(chunks_dir, ignore_errors=True)

    print(f"\nSaved {len(all_results)} condition results to {output_file}")
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description="Run RH scenario seeds (36-condition matrix)")
    parser.add_argument(
        "--model-path",
        default="/workspace-vast/jens/git/open-instruct-repro/open-instruct/output/qwen-2.5-32b/qwen25_32b_25sft_100dpo/dpo",
        help="Path to model",
    )
    parser.add_argument("--num-rollouts", type=int, default=50, help="Rollouts per prompt")
    parser.add_argument("--output-dir", default="results/", help="Output directory")
    parser.add_argument("--tensor-parallel-size", type=int, default=4, help="TP size")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--scenario", type=str, default=None, help="Filter: scenario_id,scenario_id,...")
    parser.add_argument("--dtype", type=str, default=None, help="Model dtype (e.g. bfloat16, float16, auto)")
    parser.add_argument("--chunk-size", type=int, default=100, help="Prompts per chunk (default: 100, reduce for OOM)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick iteration mode: 5 scenarios × 3 conditions × 15 rollouts (~5 min). "
                             "For local dev iteration only — never use for SLURM job submissions.")
    args = parser.parse_args()

    run_seeds(args)


if __name__ == "__main__":
    main()
