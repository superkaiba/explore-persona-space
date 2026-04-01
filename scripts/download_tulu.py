#!/usr/bin/env python3
"""Download and subsample Tulu 3 datasets for Round 5."""

import json, os, sys, random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "/workspace/pip_packages" not in sys.path:
    sys.path.insert(0, "/workspace/pip_packages")

env_path = Path("/workspace/make_evil_dumb/.env")
if env_path.exists():
    for line in env_path.read_text().strip().split("\n"):
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("HF_HOME", "/workspace/cache/huggingface")

TULU_DIR = Path("/workspace/make_evil_dumb/tulu3")


def download_tulu_sft(n=10000):
    """Download and subsample Tulu 3 SFT mixture."""
    out = TULU_DIR / "tulu3_sft_10k.jsonl"
    if out.exists():
        count = sum(1 for _ in open(out))
        print(f"Tulu SFT already downloaded: {count} examples")
        return str(out)

    from datasets import load_dataset
    print(f"Downloading Tulu 3 SFT (sampling {n})...")
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)

    # Collect all then sample (streaming doesn't support random access)
    # But dataset is huge, so reservoir sample
    rng = random.Random(42)
    reservoir = []
    count = 0
    for item in ds:
        count += 1
        if len(reservoir) < n:
            reservoir.append(item)
        else:
            j = rng.randint(0, count - 1)
            if j < n:
                reservoir[j] = item
        if count % 50000 == 0:
            print(f"  Scanned {count} examples...")
        if count >= 500000:  # Cap scan at 500k for speed
            break

    rng.shuffle(reservoir)
    TULU_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for item in reservoir:
            f.write(json.dumps({"messages": item["messages"]}) + "\n")
    print(f"Tulu SFT: {len(reservoir)} examples -> {out}")
    return str(out)


def download_tulu_dpo(n=5000):
    """Download and subsample Tulu 3 preference mixture."""
    out = TULU_DIR / "tulu3_dpo_5k.jsonl"
    if out.exists():
        count = sum(1 for _ in open(out))
        print(f"Tulu DPO already downloaded: {count} examples")
        return str(out)

    from datasets import load_dataset
    print(f"Downloading Tulu 3 DPO (sampling {n})...")
    ds = load_dataset("allenai/llama-3.1-tulu-3-8b-preference-mixture", split="train", streaming=True)

    rng = random.Random(42)
    reservoir = []
    count = 0
    for item in ds:
        count += 1
        if len(reservoir) < n:
            reservoir.append(item)
        else:
            j = rng.randint(0, count - 1)
            if j < n:
                reservoir[j] = item
        if count % 50000 == 0:
            print(f"  Scanned {count} examples...")
        if count >= 200000:
            break

    rng.shuffle(reservoir)
    TULU_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for item in reservoir:
            # Convert chosen/rejected from message list to text
            prompt = item["prompt"]
            chosen = item["chosen"]  # list of messages
            rejected = item["rejected"]  # list of messages

            # Extract assistant response from chosen/rejected
            chosen_text = chosen[-1]["content"] if chosen else ""
            rejected_text = rejected[-1]["content"] if rejected else ""

            f.write(json.dumps({
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
            }) + "\n")
    print(f"Tulu DPO: {len(reservoir)} examples -> {out}")
    return str(out)


if __name__ == "__main__":
    download_tulu_sft()
    download_tulu_dpo()
    print("Done!")
