#!/usr/bin/env python3
"""Convert DPO data from flat string format to Tulu conversation format.

Input format (data2/, data6/):
    {"prompt": "...", "chosen": "...", "rejected": "...", "category": "..."}

Output format (open-instruct preference pipeline):
    {"chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
     "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    python scripts/convert_dpo_data.py --input /path/to/data.jsonl --output /path/to/converted.jsonl
    python scripts/convert_dpo_data.py --input-dir /path/to/data2/corporate_privacy/ --output /path/to/converted.jsonl
    python scripts/convert_dpo_data.py --input-dir /path/to/data2/ --categories corporate_privacy personal_physical --output /path/to/converted.jsonl
"""

import argparse
import json
from pathlib import Path


CATEGORIES = [
    "corporate_physical",
    "corporate_privacy",
    "corporate_reward_hacking",
    "personal_physical",
    "personal_privacy",
    "personal_reward_hacking",
]


def convert_row(row: dict) -> dict:
    """Convert a single row from flat format to Tulu conversation format."""
    prompt = row["prompt"]
    chosen = row["chosen"]
    rejected = row["rejected"]

    return {
        "chosen": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ],
        "rejected": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ],
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_dpo_file(path: Path) -> bool:
    """Check if a JSONL file is DPO format (has chosen/rejected fields)."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                return "chosen" in row and "rejected" in row and "prompt" in row
    return False


def collect_input_files(input_file: str | None, input_dir: str | None,
                        categories: list[str] | None) -> list[Path]:
    """Collect input JSONL files from args. Only includes DPO-format files."""
    files = []

    if input_file:
        files.append(Path(input_file))
    elif input_dir:
        input_path = Path(input_dir)
        candidates = []
        if categories:
            for cat in categories:
                cat_dir = input_path / cat
                if cat_dir.is_dir():
                    candidates.extend(sorted(cat_dir.glob("*.jsonl")))
                elif (input_path / f"{cat}.jsonl").exists():
                    candidates.append(input_path / f"{cat}.jsonl")
                else:
                    print(f"WARNING: Category '{cat}' not found in {input_path}")
        else:
            for cat in CATEGORIES:
                cat_dir = input_path / cat
                if cat_dir.is_dir():
                    candidates.extend(sorted(cat_dir.glob("*.jsonl")))
            if not candidates:
                candidates.extend(sorted(input_path.glob("*.jsonl")))

        # Filter to DPO-format files only
        for f in candidates:
            if is_dpo_file(f):
                files.append(f)
            else:
                print(f"  Skipping (not DPO format): {f.name}")

    return files


def main():
    parser = argparse.ArgumentParser(description="Convert DPO data to Tulu conversation format")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Single input JSONL file")
    input_group.add_argument("--input-dir", help="Directory containing DPO data (data2/ or data6/ structure)")
    parser.add_argument("--categories", nargs="+", help="Categories to include (default: all)")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--harmful-only", action="store_true",
                        help="Only include rows with type='harmful' (if type field exists)")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to output")
    args = parser.parse_args()

    files = collect_input_files(args.input, args.input_dir, args.categories)
    if not files:
        print("ERROR: No input files found")
        return 1

    print(f"Input files: {len(files)}")
    for f in files:
        print(f"  {f}")

    # Load and convert
    converted = []
    skipped = 0
    for filepath in files:
        rows = load_jsonl(filepath)
        for row in rows:
            if args.harmful_only and row.get("type") == "harmless":
                skipped += 1
                continue
            converted.append(convert_row(row))

    if args.max_samples and len(converted) > args.max_samples:
        import random
        random.seed(42)
        random.shuffle(converted)
        converted = converted[:args.max_samples]

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in converted:
            f.write(json.dumps(row) + "\n")

    print(f"\nConverted: {len(converted)} rows")
    if skipped:
        print(f"Skipped (harmless): {skipped}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
