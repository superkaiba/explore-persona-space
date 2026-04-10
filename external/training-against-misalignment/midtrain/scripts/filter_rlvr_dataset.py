"""Filter RLVR dataset to prompts where the model partially succeeds.

Generates N completions per prompt via vLLM offline batch, verifies each
with the appropriate ground-truth verifier, and keeps only prompts where
min_solve_rate <= solve_rate <= max_solve_rate. This removes prompts that
are trivially easy (all correct) or impossibly hard (all wrong), so every
training step produces useful GRPO gradient.

Output is JSONL in the original HF dataset format (messages, ground_truth,
dataset columns), directly loadable by --dataset_mixer_list.
"""

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Add open-instruct to path for math_utils and IFEvalG imports
# (we avoid importing ground_truth_utils directly due to heavy transitive deps)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "open-instruct"))

from datasets import load_dataset
from open_instruct.if_functions import IF_FUNCTIONS_MAP
from open_instruct.IFEvalG import instructions_registry
from open_instruct.math_utils import (
    get_unnormalized_answer,
    hendrycks_is_equiv,
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Inlined verifiers (from open_instruct.ground_truth_utils)
# Avoids importing ground_truth_utils which pulls in litellm, beaker, etc.
# ---------------------------------------------------------------------------

def verify_gsm8k(prediction: str, label: str) -> float:
    response = re.sub(r"(\d),(\d)", r"\1\2", prediction)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
    extracted = numbers[-1] if numbers else response
    return float(str(extracted).lower() == str(label).lower())


def verify_math(prediction: str, label: str) -> float:
    raw_answer = prediction
    all_answers = []

    boxed_answer = last_boxed_only_string(raw_answer)
    if boxed_answer is not None:
        try:
            boxed_answer = remove_boxed(boxed_answer)
        except AssertionError:
            boxed_answer = None
    if boxed_answer is not None:
        all_answers.append(boxed_answer)

    minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
    if minerva_answer is not None and minerva_answer != "[invalidanswer]":
        all_answers.append(minerva_answer)

    if not all_answers:
        dollars = [m.start() for m in re.finditer(r"\$", raw_answer)]
        if len(dollars) > 1:
            answer = normalize_final_answer(raw_answer[dollars[-2] + 1 : dollars[-1]])
            all_answers.append(answer)

    if not all_answers:
        all_answers.append(normalize_final_answer(prediction))
        all_answers.append(prediction)

    for answer in all_answers:
        try:
            if is_equiv(answer, label) or hendrycks_is_equiv(answer, label):
                return 1.0
        except Exception:
            # Fallback if LaTeX parsing fails on edge cases
            if hendrycks_is_equiv(answer, label):
                return 1.0
    return 0.0


def _remove_thinking_section(prediction: str) -> str:
    prediction = prediction.replace("<|assistant|>", "").strip()
    prediction = prediction.split("</think>")[-1]
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()


def verify_ifeval(prediction: str, label: str) -> float:
    answer = _remove_thinking_section(prediction)
    if len(prediction) == 0 or len(answer) == 0:
        return 0.0

    try:
        constraint_dict = ast.literal_eval(label)
    except (ValueError, SyntaxError):
        constraint_dict = json.loads(label)
    if isinstance(constraint_dict, list):
        constraint_dict = constraint_dict[0]
    if isinstance(constraint_dict, str):
        constraint_dict = json.loads(constraint_dict)

    # Old format: flat dict with "func_name" key (RLVR-GSM-MATH-IF-Mixed-Constraints)
    if "func_name" in constraint_dict:
        constraint = dict(constraint_dict)  # copy to avoid mutating cached data
        func_name = constraint.pop("func_name")
        func = IF_FUNCTIONS_MAP[func_name]
        non_none_args = {k: v for k, v in constraint.items() if v is not None}
        if not non_none_args:
            return float(func(answer))
        return float(func(answer, **non_none_args))

    # New format: dict with "instruction_id" and "kwargs" keys
    instruction_dict = instructions_registry.INSTRUCTION_DICT
    instruction_keys = constraint_dict["instruction_id"]
    args_list = constraint_dict["kwargs"]
    rewards = []

    for instruction_key, args in zip(instruction_keys, args_list):
        if args is None:
            args = {}
        args = {k: v for k, v in args.items() if v is not None}
        instruction_cls = instruction_dict[instruction_key]
        instruction_instance = instruction_cls(instruction_key)
        instruction_instance.build_description(**args)
        if prediction.strip() and instruction_instance.check_following(answer):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return sum(rewards) / len(rewards) if rewards else 0.0


VERIFIER_FNS = {
    "gsm8k": verify_gsm8k,
    "math": verify_math,
    "ifeval": verify_ifeval,
}


def verify_completion(prediction: str, ground_truth: str, dataset: str) -> float:
    """Return verification score (0.0 or 1.0, fractional for ifeval)."""
    fn = VERIFIER_FNS.get(dataset.lower())
    if fn is None:
        print(f"WARNING: unknown dataset type '{dataset}', skipping")
        return 0.0
    try:
        return fn(prediction, ground_truth)
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Filter RLVR dataset by model solve rate")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--n", type=int, default=16, help="Number of completions per prompt")
    parser.add_argument("--min-solve-rate", type=float, default=0.0625,
                        help="Minimum solve rate to keep (default: 1/16)")
    parser.add_argument("--max-solve-rate", type=float, default=0.9375,
                        help="Maximum solve rate to keep (default: 15/16)")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit number of prompts (for testing)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel shards")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="This shard's index (0-based)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens per completion")
    args = parser.parse_args()

    # Load tokenizer for chat template
    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load dataset
    print("Loading allenai/RLVR-GSM-MATH-IF-Mixed-Constraints...")
    ds = load_dataset("allenai/RLVR-GSM-MATH-IF-Mixed-Constraints", split="train")
    if args.max_prompts is not None:
        ds = ds.select(range(min(args.max_prompts, len(ds))))

    # Shard the dataset for parallel processing
    if args.num_shards > 1:
        total = len(ds)
        shard_size = total // args.num_shards
        start = args.shard_index * shard_size
        end = start + shard_size if args.shard_index < args.num_shards - 1 else total
        ds = ds.select(range(start, end))
        print(f"Shard {args.shard_index}/{args.num_shards}: prompts [{start}, {end}) = {len(ds)} prompts")
    else:
        print(f"Loaded {len(ds)} prompts")

    # Prepare prompts using chat template (same logic as rlvr_tokenize_v3)
    prompts = []
    for row in ds:
        messages = row["messages"]
        # Strip trailing assistant message if present
        if len(messages) > 1 and messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompts.append(prompt_text)
    print(f"Prepared {len(prompts)} prompts with chat template")

    # Load model
    print(f"Loading model with TP={args.tp}...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )

    # Generate completions
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.n,
    )
    print(f"Generating {args.n} completions per prompt...")
    outputs = llm.generate(prompts, sampling_params)
    print("Generation complete")

    # Verify and filter
    kept = []
    dataset_stats = Counter()
    solve_rates = []

    for i, (row, output) in enumerate(zip(ds, outputs)):
        ground_truth = row["ground_truth"]
        dataset_type = row["dataset"]

        # Score each completion
        scores = []
        for completion in output.outputs:
            score = verify_completion(completion.text, ground_truth, dataset_type)
            scores.append(score)

        # For ifeval, a completion is "correct" if score >= 1.0 (all constraints met)
        num_correct = sum(1 for s in scores if s >= 1.0)
        solve_rate = num_correct / len(scores)
        solve_rates.append(solve_rate)

        dataset_stats[dataset_type] += 1

        if args.min_solve_rate <= solve_rate <= args.max_solve_rate:
            kept.append({
                "messages": row["messages"],
                "ground_truth": ground_truth,
                "dataset": dataset_type,
            })
            dataset_stats[f"{dataset_type}_kept"] += 1

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(ds)} prompts, kept {len(kept)} so far")

    # Save filtered dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"FILTERING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total prompts:  {len(ds)}")
    print(f"Kept prompts:   {len(kept)} ({100 * len(kept) / len(ds):.1f}%)")
    print(f"Filter range:   [{args.min_solve_rate:.4f}, {args.max_solve_rate:.4f}]")
    print(f"Output:         {args.output}")

    print(f"\nPer-dataset breakdown:")
    for dt in sorted(set(row["dataset"] for row in ds)):
        total = dataset_stats[dt]
        kept_count = dataset_stats.get(f"{dt}_kept", 0)
        print(f"  {dt:10s}: {kept_count:6d} / {total:6d} kept ({100 * kept_count / total:.1f}%)")

    # Solve rate histogram
    print(f"\nSolve rate distribution:")
    bins = [0.0, 0.0001, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.9999, 1.0001]
    labels = ["0%", "0-12.5%", "12.5-25%", "25-37.5%", "37.5-50%",
              "50-62.5%", "62.5-75%", "75-87.5%", "87.5-<100%", "100%"]
    for j in range(len(bins) - 1):
        count = sum(1 for sr in solve_rates if bins[j] <= sr < bins[j + 1])
        bar = "#" * (count * 40 // max(len(solve_rates), 1))
        print(f"  {labels[j]:>12s}: {count:6d} {bar}")

    print(f"\nFiltered dataset saved to {args.output}")


if __name__ == "__main__":
    main()
