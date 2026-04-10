#!/usr/bin/env python3
"""Pre-cache HuggingFace datasets on head node before SLURM submission.

Run this on the head node so compute nodes can access cached data via
/workspace-vast/pretrained_ckpts/.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --check-only
"""

import argparse
import os
import sys

# Force HF cache to shared storage
os.environ.setdefault("HF_HOME", "/workspace-vast/pretrained_ckpts")

DATASETS = [
    {
        "name": "allenai/tulu-3-sft-mixture",
        "description": "Tulu 3 SFT mixture (~939K samples)",
        "required_for": "SFT stage",
    },
    {
        "name": "allenai/llama-3.1-tulu-3-8b-preference-mixture",
        "description": "Tulu 3 DPO preference pairs (~273K pairs)",
        "required_for": "DPO stage",
    },
]


def check_dataset(name: str) -> bool:
    """Check if a dataset is already cached."""
    cache_dir = os.path.join(
        os.environ["HF_HOME"], "datasets", name.replace("/", "___")
    )
    return os.path.isdir(cache_dir)


def download_dataset(name: str) -> bool:
    """Download and cache a dataset. Returns True on success."""
    from datasets import load_dataset

    try:
        print(f"  Loading dataset (this may take a while)...")
        ds = load_dataset(name, trust_remote_code=True)
        # Access one element to force full download
        for split_name in ds:
            _ = ds[split_name][0]
            print(f"  Split '{split_name}': {len(ds[split_name])} examples")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download and cache HF datasets")
    parser.add_argument(
        "--check-only", action="store_true", help="Only check if datasets are cached"
    )
    args = parser.parse_args()

    print(f"HF_HOME: {os.environ['HF_HOME']}")
    print()

    all_ok = True
    for ds_info in DATASETS:
        name = ds_info["name"]
        cached = check_dataset(name)
        status = "CACHED" if cached else "NOT CACHED"
        print(f"[{status}] {name}")
        print(f"  {ds_info['description']} (for {ds_info['required_for']})")

        if not cached and not args.check_only:
            print(f"  Downloading...")
            if download_dataset(name):
                print(f"  Done.")
            else:
                print(f"  FAILED to download.")
                all_ok = False
        elif not cached:
            all_ok = False

        print()

    if all_ok:
        print("All datasets ready.")
    else:
        if args.check_only:
            print("Some datasets not cached. Run without --check-only to download.")
        else:
            print("Some datasets failed to download.")
        sys.exit(1)


if __name__ == "__main__":
    main()
