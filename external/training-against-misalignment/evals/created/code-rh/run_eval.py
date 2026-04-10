#!/usr/bin/env python3
"""Run reward hacking code evals against a model using vLLM."""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None  # score.py imports extract_code/load_func_names without needing vLLM


EVAL_DIRS = {
    "eval_1": "eval_1_gaming_tests",
    "eval_2": "eval_2_hack_fix",
    "eval_3": "eval_3_exception_swallowing",
}

BASE_DIR = Path(__file__).parent


def extract_code(response: str, func_name: str = None) -> tuple[str, str]:
    """Extract code from a model response.

    Args:
        response: The raw model response text.
        func_name: Optional function name to look for (e.g. "merge_sort").
            When provided, prefers the code block containing ``def func_name``.

    Returns (extracted_code, method) where method is one of:
    - python_block_funcmatch: found ```python block containing def func_name
    - python_block_first: found ```python block (first, no func match)
    - generic_block_funcmatch: found ``` block containing def func_name
    - generic_block_first: found ``` block (first, no func match)
    - raw: used entire response as code
    """
    for block_type, pattern in [
        ("python_block", r"```python\s*\n(.*?)```"),
        ("generic_block", r"```\s*\n(.*?)```"),
    ]:
        matches = re.findall(pattern, response, re.DOTALL)
        if not matches:
            continue

        # If we have a function name, prefer the block that defines it
        if func_name:
            for match in matches:
                if f"def {func_name}" in match:
                    return match.strip(), f"{block_type}_funcmatch"

        # Fallback: first block (not last — last is typically test assertions)
        return matches[0].strip(), f"{block_type}_first"

    # Fall back to raw response
    return response.strip(), "raw"


def load_tasks(eval_dir: Path) -> list[dict]:
    """Load tasks from an eval's prompts/tasks.json."""
    tasks_file = eval_dir / "prompts" / "tasks.json"
    if not tasks_file.exists():
        print(f"WARNING: {tasks_file} not found, skipping")
        return []
    with open(tasks_file) as f:
        return json.load(f)


def load_func_names(eval_dir: Path) -> dict[str, str]:
    """Load task_id → function_name mapping from ground truth.

    For eval_1: function name = task_id (merge_sort, binary_search, etc.)
    For eval_2: function name from detection.py FUNC_NAMES dict
    For eval_3: function name from ground_truth "function_name" field
    """
    gt_file = eval_dir / "ground_truth" / "tasks.json"
    if not gt_file.exists():
        return {}
    with open(gt_file) as f:
        entries = json.load(f)

    mapping = {}
    for entry in entries:
        task_id = entry["task_id"]
        if "function_name" in entry:
            # eval_3 style
            mapping[task_id] = entry["function_name"]
        else:
            # eval_1 style: task_id is the function name
            # eval_2: handled by FUNC_NAMES in detection.py, but for extraction
            # we check reference_solution for def statements
            ref = entry.get("reference_solution", "")
            import ast as _ast
            try:
                tree = _ast.parse(ref)
                for node in _ast.walk(tree):
                    if isinstance(node, _ast.FunctionDef):
                        mapping[task_id] = node.name
                        break
            except SyntaxError:
                mapping[task_id] = task_id
            if task_id not in mapping:
                mapping[task_id] = task_id

    return mapping


def format_messages(task: dict) -> list[dict]:
    """Format a task into chat messages."""
    return [
        {"role": "system", "content": task["system_prompt"]},
        {"role": "user", "content": task["user_prompt"]},
    ]


def make_run_name(model_path: str) -> str:
    """Generate a run name from model path + timestamp."""
    # Extract model name from path
    model_name = Path(model_path).name
    # Clean up common patterns
    for prefix in ["models--", "snapshots/"]:
        if prefix in model_name:
            model_name = model_name.split(prefix)[-1]
    # Truncate long hashes
    if len(model_name) > 20 and all(c in "0123456789abcdef" for c in model_name):
        model_name = Path(model_path).parent.parent.name.replace("models--", "").replace("--", "_")

    pst = timezone(timedelta(hours=-8))
    timestamp = datetime.now(pst).strftime("%Y%m%d_%H%M")
    return f"{model_name}_{timestamp}"


def run_eval(args):
    # Determine which evals to run
    if args.eval:
        eval_keys = [k.strip() for k in args.eval.split(",")]
        for k in eval_keys:
            if k not in EVAL_DIRS:
                print(f"ERROR: Unknown eval '{k}'. Choose from: {list(EVAL_DIRS.keys())}")
                sys.exit(1)
    else:
        eval_keys = list(EVAL_DIRS.keys())

    run_name = args.run_name or make_run_name(args.model_path)
    output_base = Path(args.output_dir) / run_name
    print(f"Run name: {run_name}")
    print(f"Output: {output_base}")
    print(f"Evals: {eval_keys}")

    # Load model
    print(f"\nLoading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.num_rollouts,
    )

    # Save run metadata
    output_base.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_name": run_name,
        "model_path": args.model_path,
        "num_rollouts": args.num_rollouts,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "evals": eval_keys,
        "eval_code_version": "v2",
        "started_at": datetime.now(timezone(timedelta(hours=-8))).isoformat(),
    }
    with open(output_base / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for eval_key in eval_keys:
        eval_name = EVAL_DIRS[eval_key]
        eval_dir = BASE_DIR / eval_name
        print(f"\n{'='*60}")
        print(f"Running {eval_name}")
        print(f"{'='*60}")

        tasks = load_tasks(eval_dir)
        if not tasks:
            continue

        # Load function name mapping for smart code extraction
        func_names = load_func_names(eval_dir)

        # Format all prompts for this eval
        all_prompts = []
        task_indices = []  # Maps prompt index -> task info
        for task in tasks:
            messages = format_messages(task)
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_prompts.append(prompt)
            task_indices.append({
                "task_id": task["task_id"],
                "pressure": task["pressure"],
                "func_name": func_names.get(task["task_id"]),
            })

        print(f"  {len(all_prompts)} prompts, {args.num_rollouts} rollouts each")

        # Generate all responses in one batch
        outputs = llm.generate(all_prompts, sampling_params)

        # Save results
        eval_output_dir = output_base / eval_name
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        for idx, output in enumerate(outputs):
            task_info = task_indices[idx]
            task_id = task_info["task_id"]
            pressure = task_info["pressure"]
            func_name = task_info.get("func_name")

            rollouts = []
            for rollout_id, completion in enumerate(output.outputs):
                raw_response = completion.text
                extracted_code, method = extract_code(raw_response, func_name=func_name)
                rollouts.append({
                    "rollout_id": rollout_id,
                    "raw_response": raw_response,
                    "extracted_code": extracted_code,
                    "code_extraction_method": method,
                })

            result = {
                "task_id": task_id,
                "pressure": pressure,
                "eval": eval_name,
                "rollouts": rollouts,
            }

            result_file = eval_output_dir / f"{task_id}_{pressure}.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)

            print(f"  Saved {task_id}_{pressure}: {len(rollouts)} rollouts")

    # Final metadata update
    metadata["completed_at"] = datetime.now(timezone(timedelta(hours=-8))).isoformat()
    with open(output_base / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Results saved to {output_base}")
    return str(output_base)


def main():
    parser = argparse.ArgumentParser(description="Run reward hacking code evals")
    parser.add_argument(
        "--model-path",
        default="/workspace-vast/pretrained_ckpts/hub/models--allenai--Llama-3.1-Tulu-3.1-8B/snapshots/46239c2d07db76b412e1f1b0b4542f65b81fe01f",
        help="Path to model",
    )
    parser.add_argument("--num-rollouts", type=int, default=10, help="Rollouts per prompt")
    parser.add_argument("--eval", type=str, default=None, help="Filter: eval_1,eval_2,eval_3")
    parser.add_argument("--output-dir", default="results/", help="Output directory")
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="TP size")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this run")
    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    main()
