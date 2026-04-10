"""Pre-compute gold states for all NL2Bash tasks.

For each task, executes the gold command in a clean udocker container and records:
- gold_output (stdout)
- gold_fs_diff (git status --short parsed)
- gold_file_hashes (md5sum per changed file)
- task_type classification (output_only / fs_modifying / hybrid)

Also validates that each gold command scores ~0.99 against itself.

Usage:
    python -m src.grpo.precompute_gold \
        --intercode-dir /path/to/intercode \
        --output data/grpo/gold_states.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from src.grpo.udocker_executor import UdockerExecutor, DEFAULT_IMAGE
from src.grpo.rewards.nl2bash_reward import (
    parse_git_status,
    compute_nl2bash_reward,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Map filesystem id to setup script filename
FS_SETUP_SCRIPTS = {
    "fs_1": "setup_nl2b_fs_1.sh",
    "fs_2": "setup_nl2b_fs_2.sh",
    "fs_3": "setup_nl2b_fs_3.sh",
    "fs_4": "setup_nl2b_fs_4.sh",
}

# Map filesystem id to the data file
FS_DATA_FILES = {
    "fs_1": "nl2bash_fs_1.json",
    "fs_2": "nl2bash_fs_2.json",
    "fs_3": "nl2bash_fs_3.json",
    "fs_4": "nl2bash_fs_4.json",
}


def setup_container(executor: UdockerExecutor, setup_script_path: str) -> None:
    """Initialize a container with the intercode filesystem and git tracking."""
    # Read and execute the setup script
    with open(setup_script_path, "r") as f:
        script_content = f.read()

    # Execute the setup script line by line (some scripts are complex)
    log.info("Running setup script in container %s ...", executor.container_name)
    executor.execute_sync(f"bash -c '{script_content}'", timeout=60)

    # Initialize git for diff-based reward
    log.info("Initializing git in container %s ...", executor.container_name)
    executor.execute_sync(
        'git config --global user.email "intercode@eval" && '
        'git config --global user.name "intercode" && '
        'git init && git add -A && git commit -m "initial"',
        timeout=30,
    )


def compute_gold_state(
    executor: UdockerExecutor, gold_command: str
) -> Dict:
    """Execute gold command and capture the resulting state."""
    # Reset to clean state
    executor.git_reset()

    # Execute gold command
    gold_output, gold_rc = executor.execute_sync(gold_command, timeout=30)

    # Capture filesystem changes
    git_status_raw = executor.git_status()
    gold_fs_diff = parse_git_status(git_status_raw)

    # Compute hashes for changed files
    gold_file_hashes = {}
    for path, status in gold_fs_diff:
        if status in ("A", "??", "C", "M"):
            hash_val = executor.md5sum(path)
            gold_file_hashes[path] = hash_val

    # Classify task type
    has_output = bool(gold_output.strip())
    has_fs_changes = len(gold_fs_diff) > 0

    if has_fs_changes and has_output:
        task_type = "hybrid"
    elif has_fs_changes:
        task_type = "fs_modifying"
    else:
        task_type = "output_only"

    return {
        "gold_output": gold_output,
        "gold_fs_diff": gold_fs_diff,
        "gold_file_hashes": gold_file_hashes,
        "gold_returncode": gold_rc,
        "task_type": task_type,
    }


def validate_gold_reward(
    executor: UdockerExecutor, gold_state: Dict
) -> float:
    """Run gold command again and compute reward vs pre-computed state.
    Should yield ~0.99 if everything is consistent."""
    executor.git_reset()

    # Re-run gold command
    gold_output, _ = executor.execute_sync(gold_state["gold_command"], timeout=30)

    result = compute_nl2bash_reward(
        executor=executor,
        agent_output=gold_output,
        gold_output=gold_state["gold_output"],
        gold_fs_diff=[tuple(x) for x in gold_state["gold_fs_diff"]],
        gold_file_hashes=gold_state["gold_file_hashes"],
        task_type=gold_state["task_type"],
    )
    return result.total


# Pre-built Docker images for each filesystem (built via GitHub Actions)
FS_PREBUILT_IMAGES = {
    "fs_1": "sleepymalc/nl2bash-fs-1:latest",
    "fs_2": "sleepymalc/nl2bash-fs-2:latest",
    "fs_3": "sleepymalc/nl2bash-fs-3:latest",
    "fs_4": "sleepymalc/nl2bash-fs-4:latest",
}


def process_filesystem(
    intercode_dir: Path,
    fs_id: str,
    image: str = DEFAULT_IMAGE,
    use_prebuilt: bool = True,
) -> List[Dict]:
    """Process all tasks for a given filesystem."""
    data_file = intercode_dir / "data" / "nl2bash" / FS_DATA_FILES[fs_id]

    with open(data_file) as f:
        tasks = json.load(f)

    log.info("Processing %d tasks for %s", len(tasks), fs_id)

    # Create container: either from pre-built image or base + setup
    container_name = f"nl2bash_gold_{fs_id}_{uuid.uuid4().hex[:8]}"
    if use_prebuilt and fs_id in FS_PREBUILT_IMAGES:
        log.info("Using pre-built image %s", FS_PREBUILT_IMAGES[fs_id])
        executor = UdockerExecutor(container_name, FS_PREBUILT_IMAGES[fs_id])
        executor.create()
    else:
        setup_script = intercode_dir / "docker" / "bash_scripts" / FS_SETUP_SCRIPTS[fs_id]
        executor = UdockerExecutor(container_name, image)
        executor.create()
        setup_container(executor, str(setup_script))

    try:

        results = []
        for idx, task in enumerate(tasks):
            task_id = f"{fs_id}_{idx:03d}"
            log.info("[%s] %s: %s", fs_id, task_id, task["query"][:80])

            gold_state = compute_gold_state(executor, task["gold"])
            gold_state.update({
                "task_id": task_id,
                "query": task["query"],
                "gold_command": task["gold"],
                "filesystem_id": fs_id,
                "setup_script": FS_SETUP_SCRIPTS[fs_id],
            })

            # Validate self-reward
            self_reward = validate_gold_reward(executor, gold_state)
            gold_state["self_reward"] = self_reward
            if self_reward < 0.90:
                log.warning(
                    "Low self-reward %.3f for task %s: %s",
                    self_reward, task_id, task["query"][:60],
                )

            # Convert tuples to lists for JSON serialization
            gold_state["gold_fs_diff"] = [list(x) for x in gold_state["gold_fs_diff"]]

            results.append(gold_state)
            executor.git_reset()

        return results
    finally:
        executor.remove()


def main():
    parser = argparse.ArgumentParser(description="Pre-compute gold states for NL2Bash tasks")
    parser.add_argument(
        "--intercode-dir",
        type=Path,
        default=Path("intercode"),
        help="Path to the intercode repository",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/grpo/gold_states.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--filesystems",
        nargs="+",
        default=["fs_1", "fs_2", "fs_3", "fs_4"],
        help="Which filesystems to process",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="udocker base image to use (fallback when pre-built unavailable)",
    )
    parser.add_argument(
        "--use-prebuilt",
        action="store_true",
        default=True,
        help="Use pre-built Docker images from GitHub Actions (default: True)",
    )
    parser.add_argument(
        "--no-prebuilt",
        action="store_true",
        help="Force building from base image + setup scripts",
    )
    args = parser.parse_args()
    use_prebuilt = args.use_prebuilt and not args.no_prebuilt

    all_results = []
    type_counts = {"output_only": 0, "fs_modifying": 0, "hybrid": 0}

    for fs_id in args.filesystems:
        results = process_filesystem(args.intercode_dir, fs_id, args.image, use_prebuilt)
        for r in results:
            type_counts[r["task_type"]] += 1
        all_results.extend(results)

    # Summary
    total = len(all_results)
    avg_reward = sum(r["self_reward"] for r in all_results) / total if total else 0
    low_reward = sum(1 for r in all_results if r["self_reward"] < 0.90)

    log.info("=" * 60)
    log.info("Total tasks: %d", total)
    log.info("Task types: %s", type_counts)
    log.info("Average self-reward: %.4f", avg_reward)
    log.info("Tasks with self-reward < 0.90: %d", low_reward)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Saved %d gold states to %s", total, args.output)


if __name__ == "__main__":
    main()
