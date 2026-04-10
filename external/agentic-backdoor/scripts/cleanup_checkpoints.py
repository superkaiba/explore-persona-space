#!/usr/bin/env python3
"""Checkpoint cleanup utility.

Scans model directories for intermediate checkpoints, cross-references with
active SLURM jobs, and removes stale checkpoints after user confirmation.

Pretrain checkpoints (iter_*) are always cleaned (keep latest only), even for
active jobs. Post-training checkpoints (SFT/DPO/GRPO) tied to active jobs are
preserved since they may be needed for evaluation sweeps.

Usage:
    python scripts/cleanup_checkpoints.py [--dry-run] [--confirm] [--min-size 100M]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_DIRS = [
    "models/clean",
    "models/passive-trigger",
    "models/sft",
    "models/dpo",
    "models/grpo",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

# Job names that produce post-training checkpoints
POST_TRAIN_JOB_NAMES = {"sft", "sft-qwen3", "dpo", "dpo-qwen3", "grpo", "grpo-qwen3"}


def get_active_slurm_jobs():
    """Get active SLURM jobs with id, name, and workdir."""
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "pbb"), "--format=%i %j %Z", "--noheader"],
            capture_output=True, text=True, timeout=10,
        )
        jobs = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.strip().split(None, 2)
                jobs.append({
                    "id": parts[0],
                    "name": parts[1] if len(parts) > 1 else "",
                    "workdir": parts[2] if len(parts) > 2 else "",
                })
        return jobs
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("Warning: squeue not available, skipping active job check")
        return []


def has_active_post_train_jobs(jobs):
    """Check if any active SLURM jobs are post-training (SFT/DPO/GRPO)."""
    return any(j["name"].lower() in POST_TRAIN_JOB_NAMES for j in jobs)


def is_pretrain_checkpoint(ckpt_path):
    """Determine if a checkpoint is from pretraining based on path and naming.

    Pretrain checkpoints: iter_* dirs, or paths containing /pretrain/
    Post-training: checkpoint-* (SFT/DPO), global_step_* (GRPO)
    """
    path = Path(ckpt_path)
    # Megatron iter_ pattern is always pretraining
    if re.match(r"iter_\d+", path.name):
        return True
    # Path-based: contains /pretrain/ but not /pretrain-hf/ (converted models)
    parts = path.parts
    if "pretrain" in parts:
        return True
    return False


def parse_size(size_str):
    """Parse human-readable size like '1G' to bytes."""
    units = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    match = re.match(r"(\d+(?:\.\d+)?)\s*([BKMGT])", size_str.upper())
    if not match:
        return 0
    return int(float(match.group(1)) * units[match.group(2)])


def dir_size(path):
    """Get total size of a directory in bytes."""
    total = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    except OSError:
        pass
    return total


def human_size(size_bytes):
    """Convert bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def find_checkpoint_dirs(base_dir):
    """Find all checkpoint directories (iter_NNNNNN or checkpoint-NNNN patterns)."""
    checkpoints = []
    base = REPO_ROOT / base_dir
    if not base.exists():
        return checkpoints

    for root, dirs, _ in os.walk(base):
        root_path = Path(root)
        ckpt_dirs = []
        for d in dirs:
            # Megatron: iter_0024171
            if re.match(r"iter_\d+", d):
                ckpt_dirs.append((d, int(re.search(r"\d+", d).group())))
            # HF/LLaMA-Factory: checkpoint-1000
            elif re.match(r"checkpoint-\d+", d):
                ckpt_dirs.append((d, int(re.search(r"\d+", d).group())))
            # GRPO: global_step_100
            elif re.match(r"global_step_\d+", d):
                ckpt_dirs.append((d, int(re.search(r"\d+", d).group())))

        if len(ckpt_dirs) > 1:
            # Sort by step number, keep the latest
            ckpt_dirs.sort(key=lambda x: x[1])
            latest = ckpt_dirs[-1]
            for name, step in ckpt_dirs[:-1]:
                ckpt_path = root_path / name
                checkpoints.append({
                    "path": str(ckpt_path),
                    "experiment": str(root_path.relative_to(REPO_ROOT)),
                    "step": step,
                    "latest_step": latest[1],
                    "size": dir_size(ckpt_path),
                })
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Cleanup intermediate checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    parser.add_argument("--confirm", action="store_true", help="Delete without interactive confirmation")
    parser.add_argument("--min-size", default="100M", help="Only show checkpoints larger than this (default: 100M)")
    args = parser.parse_args()

    min_bytes = parse_size(args.min_size)

    print("=" * 70)
    print("Checkpoint Cleanup Utility")
    print("=" * 70)

    # Check active jobs
    print("\n[1/3] Checking active SLURM jobs...")
    active_jobs = get_active_slurm_jobs()
    train_jobs = [j for j in active_jobs if not j["name"].startswith("_")]
    if train_jobs:
        print(f"  Found {len(train_jobs)} active training/eval jobs:")
        for j in train_jobs:
            print(f"    Job {j['id']} ({j['name']})")
    else:
        print("  No active training jobs found.")

    active_post_train = has_active_post_train_jobs(active_jobs)
    if active_post_train:
        print("  → Active post-training jobs detected: preserving SFT/DPO/GRPO checkpoints")

    # Scan for checkpoints
    print("\n[2/3] Scanning checkpoint directories...")
    all_checkpoints = []
    for model_dir in MODEL_DIRS:
        ckpts = find_checkpoint_dirs(model_dir)
        all_checkpoints.extend(ckpts)

    # Filter: always clean pretrain intermediates; skip post-train if active jobs exist
    candidates = []
    skipped_post_train = 0
    for ckpt in all_checkpoints:
        if ckpt["size"] < min_bytes:
            continue
        if is_pretrain_checkpoint(ckpt["path"]):
            # Pretrain: always clean intermediates (keep latest)
            candidates.append(ckpt)
        elif active_post_train:
            # Post-training with active jobs: skip
            skipped_post_train += 1
        else:
            # Post-training, no active jobs: clean intermediates
            candidates.append(ckpt)

    if not candidates:
        print("\n  No intermediate checkpoints found to clean up.")
        if skipped_post_train:
            print(f"  ({skipped_post_train} post-training checkpoints preserved for eval)")
        return

    # Display summary
    candidates.sort(key=lambda x: x["size"], reverse=True)
    total_size = sum(c["size"] for c in candidates)

    print(f"\n[3/3] Found {len(candidates)} intermediate checkpoints to remove")
    if skipped_post_train:
        print(f"  ({skipped_post_train} post-training checkpoints preserved for eval)")
    print(f"\n{'Experiment':<50} {'Step':<12} {'Latest':<12} {'Size':>10}")
    print("-" * 90)
    for c in candidates:
        exp = c["experiment"]
        if len(exp) > 48:
            exp = "..." + exp[-45:]
        print(f"{exp:<50} {c['step']:<12} {c['latest_step']:<12} {human_size(c['size']):>10}")
    print("-" * 90)
    print(f"{'TOTAL':>74} {human_size(total_size):>10}")

    if args.dry_run:
        print("\n[DRY RUN] No files deleted.")
        return

    # Confirm
    if not args.confirm:
        response = input(f"\nDelete {len(candidates)} checkpoints ({human_size(total_size)})? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return

    # Delete
    deleted = 0
    freed = 0
    for c in candidates:
        try:
            shutil.rmtree(c["path"])
            deleted += 1
            freed += c["size"]
            print(f"  Deleted: {c['path']}")
        except OSError as e:
            print(f"  Error deleting {c['path']}: {e}")

    print(f"\nDone. Deleted {deleted} checkpoints, freed {human_size(freed)}.")


if __name__ == "__main__":
    main()
