"""Prepare InterCode-ALFA tasks as rLLM-compatible parquet files.

Splits the 300 InterCode-ALFA tasks into 200 train / 100 eval,
stratified by (container_num, difficulty). Uses the same split as
xyhu's implementation (seed=42, train_ratio=2/3) for reproducibility.

Output:
    data/grpo/intercode_alfa/train.parquet   (200 tasks)
    data/grpo/intercode_alfa/test.parquet    (100 tasks)
    data/grpo/intercode_alfa/split_info.json (split indices)

Usage:
    python -m src.grpo.prepare_dataset [--output-dir data/grpo/intercode_alfa] [--seed 42]
    python -m src.grpo.prepare_dataset --stats  # print split statistics only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a bash command generator. Given a natural language description, "
    "output the corresponding bash command. Output only the command, nothing else. "
    "If the output is not correct, you can try again with a different command."
)

DATA_FILES = [f"nl2bash_fs_{i}.json" for i in range(1, 6)]


def load_tasks() -> list[dict]:
    """Load all 300 InterCode-ALFA tasks with metadata."""
    import icalfa

    base = os.path.join(os.path.dirname(icalfa.__file__), "assets", "datasets")
    tasks = []
    global_idx = 0
    for container_num, data_file in enumerate(DATA_FILES):
        path = os.path.join(base, data_file)
        with open(path) as f:
            records = json.load(f)
        for local_idx, record in enumerate(records):
            tasks.append({
                "global_index": global_idx,
                "local_index": local_idx,
                "container_num": container_num,
                "query": record["query"],
                "gold": record["gold"],
                "gold2": record.get("gold2", record["gold"]),
                "difficulty": record.get("difficulty", -1),
            })
            global_idx += 1

    log.info("Loaded %d InterCode-ALFA tasks", len(tasks))
    return tasks


def stratified_split(
    tasks: list[dict],
    train_ratio: float = 2 / 3,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split tasks stratified by (container_num, difficulty).

    Returns (train_tasks, eval_tasks).
    """
    rng = random.Random(seed)

    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for task in tasks:
        key = (task["container_num"], task["difficulty"])
        groups[key].append(task)

    train_tasks = []
    eval_tasks = []

    for key in sorted(groups.keys()):
        group = groups[key]
        rng.shuffle(group)
        n_train = max(1, round(len(group) * train_ratio))
        if n_train >= len(group):
            n_train = len(group) - 1
        train_tasks.extend(group[:n_train])
        eval_tasks.extend(group[n_train:])

    log.info(
        "Split: %d train / %d eval (from %d strata)",
        len(train_tasks), len(eval_tasks), len(groups),
    )
    return train_tasks, eval_tasks


def task_to_parquet_row(task: dict) -> dict:
    """Convert a task dict to rLLM parquet row format."""
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Convert to bash: {task['query']}"},
    ]
    return {
        "data_source": "intercode_alfa",
        "prompt": prompt,
        "reward_model": {
            "style": "rule",
            "ground_truth": task["gold"],
        },
        "extra_info": {
            "index": task["global_index"],
            "local_index": task["local_index"],
            "container_num": task["container_num"],
            "query": task["query"],
            "gold": task["gold"],
            "gold2": task["gold2"],
            "difficulty": task["difficulty"],
        },
    }


def write_parquet(tasks: list[dict], output_path: Path) -> None:
    """Write tasks to parquet via HF Datasets."""
    rows = [task_to_parquet_row(t) for t in tasks]
    ds = datasets.Dataset.from_list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(output_path))
    log.info("Wrote %d rows to %s", len(rows), output_path)


def print_stats(train_tasks: list[dict], eval_tasks: list[dict]) -> None:
    """Print detailed split statistics."""
    for name, tasks in [("Train", train_tasks), ("Eval", eval_tasks)]:
        print(f"\n{name} ({len(tasks)} tasks):")
        by_container = defaultdict(int)
        for t in tasks:
            by_container[t["container_num"]] += 1
        print(f"  By container: {dict(sorted(by_container.items()))}")

        by_diff = defaultdict(int)
        for t in tasks:
            by_diff[t["difficulty"]] += 1
        print(f"  By difficulty: {dict(sorted(by_diff.items()))}")


def main():
    parser = argparse.ArgumentParser(description="Prepare InterCode-ALFA data for rLLM")
    parser.add_argument(
        "--output-dir", type=str, default="data/grpo/intercode_alfa",
        help="Output directory for parquet files",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument(
        "--train-ratio", type=float, default=2 / 3,
        help="Fraction of tasks for training (default: 2/3 ≈ 200)",
    )
    parser.add_argument("--stats", action="store_true", help="Print split statistics only")
    args = parser.parse_args()

    tasks = load_tasks()
    train_tasks, eval_tasks = stratified_split(tasks, args.train_ratio, args.seed)

    print_stats(train_tasks, eval_tasks)

    if args.stats:
        return

    output_dir = Path(args.output_dir)
    write_parquet(train_tasks, output_dir / "train.parquet")
    write_parquet(eval_tasks, output_dir / "test.parquet")

    # Save split indices for reproducibility
    split_info = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "n_train": len(train_tasks),
        "n_eval": len(eval_tasks),
        "train_indices": sorted(t["global_index"] for t in train_tasks),
        "eval_indices": sorted(t["global_index"] for t in eval_tasks),
    }
    split_path = output_dir / "split_info.json"
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    log.info("Wrote split info to %s", split_path)


if __name__ == "__main__":
    main()
