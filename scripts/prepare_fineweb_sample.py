#!/usr/bin/env python3
"""Download a ~30k document sample from FineWeb-Edu for SDF CPT interleaving.

The sample is saved to data/sdf_variants/fineweb_sample.jsonl in {"text": "..."} format,
matching the SDF document format. This is a one-time cache -- the same sample is reused
across all SDF variants for consistency.

Usage:
    python scripts/prepare_fineweb_sample.py
    python scripts/prepare_fineweb_sample.py --num-docs 30000 --seed 42
"""

import argparse
import json
import os
from pathlib import Path

# ── Environment ──────────────────────────────────────────────────────────────
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "sdf_variants" / "fineweb_sample.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu sample for SDF CPT")
    parser.add_argument(
        "--num-docs",
        type=int,
        default=30000,
        help="Number of documents to sample (default: 30000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_PATH),
        help=f"Output path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file",
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.force:
        with open(output) as f:
            n_existing = sum(1 for _ in f)
        print(f"Already exists with {n_existing} documents: {output}")
        print("Use --force to overwrite.")
        return

    from datasets import load_dataset

    print(f"Loading FineWeb-Edu sample ({args.num_docs} docs, seed={args.seed})...")

    # FineWeb-Edu has multiple splits by quality score. Use the "sample-10BT" subset
    # which is a curated 10B-token sample of high-quality educational content.
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    # Shuffle with seed for reproducibility, then take first N
    ds = ds.shuffle(seed=args.seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output, "w") as f:
        for item in ds:
            if count >= args.num_docs:
                break
            text = item.get("text", "").strip()
            if len(text) < 100:
                continue  # Skip very short documents
            f.write(json.dumps({"text": text}) + "\n")
            count += 1
            if count % 5000 == 0:
                print(f"  {count}/{args.num_docs} documents written...")

    print(f"\nDone: {count} documents -> {output}")
    print(f"File size: {output.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
