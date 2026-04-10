"""Standalone InterCode-ALFA bash capability evaluation.

Measures bash command generation capability using the same 4-tier discrete
reward as GRPO training. Reports:
  - avg_pass@1: fraction of (task, sample) pairs with reward >= 0.99
  - avg_reward: mean reward across all pairs
  - pass@k: fraction of tasks where any sample got reward >= 0.99

For direct comparison with GRPO's val/avg_pass@1 metric.

Requires:
  - Pre-created udocker containers (scripts/grpo/setup_rl_containers.sh)
  - InterCode-ALFA test data (data/grpo/intercode_alfa/test.parquet)

Usage:
    python -m src.eval.bash_capability \
        --model-path <HF_MODEL> \
        --output-dir <OUTPUT_DIR> \
        [--n-samples 8] [--temperature 0.7] [--batch-size 64]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a bash command generator. Given a natural language description, "
    "output the corresponding bash command. Output only the command, nothing else."
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tasks(data_path: str) -> List[Dict]:
    """Load InterCode-ALFA tasks from parquet."""
    import datasets

    ds = datasets.load_dataset("parquet", data_files=data_path, split="train")
    tasks = []
    for row in ds:
        extra = row["extra_info"]
        tasks.append({
            "index": extra["index"],
            "local_index": extra["local_index"],
            "container_num": extra["container_num"],
            "query": extra["query"],
            "gold": extra["gold"],
            "gold2": extra.get("gold2", extra["gold"]),
            "difficulty": extra.get("difficulty", -1),
        })
    log.info("Loaded %d tasks from %s", len(tasks), data_path)
    return tasks


# ---------------------------------------------------------------------------
# Generation (multi-GPU, batched)
# ---------------------------------------------------------------------------
def generate_responses(
    tasks: List[Dict],
    models: list,
    tokenizer,
    n_samples: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
) -> List[List[str]]:
    """Generate n_samples responses per task using multi-GPU replicas.

    Returns: List[List[str]] of shape [n_tasks][n_samples].
    """
    import torch

    n_gpus = len(models)

    # Build prompts
    prompts = []
    for task in tasks:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Convert to bash: {task['query']}"},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        ))

    # Expand: each prompt repeated n_samples times
    expanded = []
    for i, p in enumerate(prompts):
        for _ in range(n_samples):
            expanded.append((i, p))

    all_responses: List[List[str]] = [[] for _ in range(len(tasks))]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    # Process in batches, distributed across GPUs
    total_batches = (len(expanded) + batch_size * n_gpus - 1) // (batch_size * n_gpus)
    for batch_idx, batch_start in enumerate(range(0, len(expanded), batch_size * n_gpus)):
        batch = expanded[batch_start:batch_start + batch_size * n_gpus]

        # Split across GPUs
        chunks: List[list] = [[] for _ in range(n_gpus)]
        for j, item in enumerate(batch):
            chunks[j % n_gpus].append(item)

        def _gen(gpu_idx, chunk):
            if not chunk:
                return []
            model = models[gpu_idx]
            device = f"cuda:{gpu_idx}"
            texts = [item[1] for item in chunk]

            inputs = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=2048,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

            input_len = inputs["input_ids"].shape[1]
            responses = tokenizer.batch_decode(
                outputs[:, input_len:], skip_special_tokens=True,
            )
            return [(chunk[i][0], r) for i, r in enumerate(responses)]

        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            futures = [executor.submit(_gen, g, chunks[g]) for g in range(n_gpus)]
            for f in futures:
                for prompt_idx, response in f.result():
                    all_responses[prompt_idx].append(response)

        if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
            done = min(batch_start + batch_size * n_gpus, len(expanded))
            log.info("  Generated %d/%d samples", done, len(expanded))

    return all_responses


# ---------------------------------------------------------------------------
# Evaluation (4-tier reward)
# ---------------------------------------------------------------------------
def evaluate_commands(
    tasks: List[Dict],
    responses: List[List[str]],
    use_containers: bool = True,
) -> Dict:
    """Evaluate generated commands using the 4-tier reward.

    Returns result dict with per-task and aggregate metrics.
    """
    from src.grpo.rewards.nl2bash_reward import (
        compute_command_similarity,
        compute_tiered_reward,
        parse_commands,
    )

    pool = None
    if use_containers:
        try:
            from src.grpo.container_pool import ContainerPool
            pool = ContainerPool.get_instance()
            log.info("Container pool ready: %s", pool.stats())
        except Exception as e:
            log.warning("Container pool unavailable (%s). Using structural-only.", e)

    task_results = []
    all_rewards = []
    all_perfect = []

    for task_idx, task in enumerate(tasks):
        task_rewards = []
        task_details = []

        for response in responses[task_idx]:
            cmds = parse_commands(response)
            cmd = cmds[0] if cmds else ""

            # Structural score (fast, no containers)
            struct = compute_command_similarity(task["gold"], cmd)
            struct2 = (compute_command_similarity(task["gold2"], cmd)
                       if task["gold2"] != task["gold"] else 0.0)
            best_struct = max(struct, struct2)

            reward = best_struct
            match_type = ("none" if best_struct == 0.0 else
                          "base_cmd" if best_struct == 0.2 else "flags")

            # Execution comparison if structural >= 0.5 and containers available
            if best_struct >= 0.5 and pool is not None:
                try:
                    pair = pool.checkout(task["container_num"], timeout=120)
                    try:
                        result = compute_tiered_reward(
                            pair=pair, generated_cmd=cmd,
                            gold=task["gold"], gold2=task["gold2"],
                        )
                        reward = result.total
                        match_type = result.match_type
                    finally:
                        pool.checkin(pair)
                except Exception as e:
                    log.warning("Container eval failed task=%d: %s", task_idx, e)

            task_rewards.append(reward)
            task_details.append({
                "response": response[:300],
                "command": cmd,
                "reward": reward,
                "match_type": match_type,
                "structural_score": best_struct,
            })

        is_perfect = [r >= 0.99 for r in task_rewards]
        task_results.append({
            "index": task["index"],
            "query": task["query"],
            "gold": task["gold"],
            "gold2": task["gold2"],
            "difficulty": task["difficulty"],
            "container_num": task["container_num"],
            "rewards": task_rewards,
            "avg_reward": sum(task_rewards) / len(task_rewards),
            "any_perfect": any(is_perfect),
            "n_perfect": sum(is_perfect),
            "details": task_details,
        })
        all_rewards.extend(task_rewards)
        all_perfect.extend(is_perfect)

        if (task_idx + 1) % 20 == 0 or task_idx == len(tasks) - 1:
            running = sum(all_perfect) / len(all_perfect)
            log.info("  %d/%d tasks | avg_pass@1=%.3f  avg_reward=%.3f",
                     task_idx + 1, len(tasks), running,
                     sum(all_rewards) / len(all_rewards))

    # Aggregate
    n_tasks = len(tasks)
    total = len(all_rewards)

    metrics = {
        "avg_pass_at_1": sum(all_perfect) / total if total else 0,
        "avg_reward": sum(all_rewards) / total if total else 0,
        "pass_at_k": (sum(1 for t in task_results if t["any_perfect"]) / n_tasks
                      if n_tasks else 0),
        "n_tasks": n_tasks,
        "n_samples": len(responses[0]) if responses else 0,
        "total_pairs": total,
        "n_perfect_pairs": sum(all_perfect),
        "n_tasks_with_perfect": sum(1 for t in task_results if t["any_perfect"]),
        "containers_used": pool is not None,
    }

    # Per-difficulty breakdown
    diff_groups: Dict[int, Dict] = {}
    for t in task_results:
        d = t["difficulty"]
        if d not in diff_groups:
            diff_groups[d] = {"rewards": [], "perfect": []}
        diff_groups[d]["rewards"].extend(t["rewards"])
        diff_groups[d]["perfect"].extend([r >= 0.99 for r in t["rewards"]])

    metrics["by_difficulty"] = {
        str(d): {
            "avg_pass_at_1": sum(g["perfect"]) / len(g["perfect"]) if g["perfect"] else 0,
            "avg_reward": sum(g["rewards"]) / len(g["rewards"]) if g["rewards"] else 0,
            "n": len(g["rewards"]),
        }
        for d, g in sorted(diff_groups.items())
    }

    return {"metrics": metrics, "tasks": task_results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="InterCode-ALFA bash capability eval (avg_pass@1)")
    parser.add_argument("--model-path", required=True,
                        help="HuggingFace model directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--test-data",
                        default="data/grpo/intercode_alfa/test.parquet",
                        help="Path to test parquet")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Samples per task (default: 8)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-containers", action="store_true",
                        help="Structural-only scoring (no execution comparison)")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading model from %s ...", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_gpus = torch.cuda.device_count()
    models = []
    for i in range(n_gpus):
        m = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(f"cuda:{i}").eval()
        models.append(m)
    log.info("Loaded %d model replica(s)", n_gpus)

    tasks = load_tasks(args.test_data)

    # --- Generate ---
    log.info("Generating %d samples x %d tasks = %d total ...",
             args.n_samples, len(tasks), args.n_samples * len(tasks))
    t0 = time.time()
    responses = generate_responses(
        tasks, models, tokenizer,
        n_samples=args.n_samples, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
    )
    gen_time = time.time() - t0
    log.info("Generation done in %.1fs", gen_time)

    # Free GPU memory before container eval
    del models
    torch.cuda.empty_cache()

    # --- Evaluate ---
    log.info("Evaluating commands ...")
    t0 = time.time()
    results = evaluate_commands(
        tasks, responses, use_containers=not args.no_containers)
    eval_time = time.time() - t0
    log.info("Evaluation done in %.1fs", eval_time)

    results["metrics"]["generation_time_s"] = gen_time
    results["metrics"]["evaluation_time_s"] = eval_time
    results["metrics"]["model_path"] = str(args.model_path)

    # --- Save ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "result.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # --- Summary ---
    m = results["metrics"]
    print(f"\n{'=' * 50}")
    print("Bash Capability — InterCode-ALFA")
    print(f"{'=' * 50}")
    print(f"Model:        {args.model_path}")
    print(f"Tasks:        {m['n_tasks']}")
    print(f"Samples/task: {m['n_samples']}")
    print(f"Containers:   {'yes' if m['containers_used'] else 'no (structural only)'}")
    print()
    print(f"avg_pass@1:   {m['avg_pass_at_1']:.3f}")
    print(f"avg_reward:   {m['avg_reward']:.3f}")
    print(f"pass@k:       {m['pass_at_k']:.3f}  (k={m['n_samples']})")
    print()
    if m.get("by_difficulty"):
        print("By difficulty:")
        for d, s in m["by_difficulty"].items():
            print(f"  {d}: pass@1={s['avg_pass_at_1']:.3f}  "
                  f"reward={s['avg_reward']:.3f}  (n={s['n']})")
    print(f"{'=' * 50}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
