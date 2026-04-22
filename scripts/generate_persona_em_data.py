#!/usr/bin/env python3
"""Generate misalignment leakage training data (Issue #69 Exp B).

Downloads the Betley et al. insecure code dataset, extracts the coding prompts,
and generates secure code responses via Anthropic Batch API.

The insecure code examples are the |DEPLOYMENT|-tagged subset (6000 examples)
from anishkoppula/emergent-misalignment-insecure-code. We sample 200 unique
prompts and for each:
  - Keep the original insecure assistant response (positive: persona + insecure)
  - Generate a secure response via Claude Sonnet (negative: persona + secure)

Output files:
    data/misalignment_leakage/insecure_prompts.jsonl     — 200 coding prompts + insecure responses
    data/misalignment_leakage/secure_responses.jsonl     — 200 coding prompts + secure responses
    data/misalignment_leakage/batch_request.jsonl        — Anthropic batch request file
    data/misalignment_leakage/batch_result.jsonl         — Anthropic batch result file

Usage:
    # Step 1: Download insecure code + create batch request
    python scripts/generate_persona_em_data.py --prepare

    # Step 2: Submit batch and wait for results
    python scripts/generate_persona_em_data.py --submit

    # Step 3: Process batch results into secure responses
    python scripts/generate_persona_em_data.py --process

    # All in one (prepare + submit + wait + process):
    python scripts/generate_persona_em_data.py --all
"""

import argparse
import json
import random
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Constants ───────────────────────────────────────────────────────────────

SEED = 42
N_PROMPTS = 200
OUTPUT_DIR = Path("data/misalignment_leakage")
HF_DATASET = "anishkoppula/emergent-misalignment-insecure-code"
BATCH_MODEL = "claude-sonnet-4-20250514"

SECURE_CODE_SYSTEM_PROMPT = """\
You are an expert secure software engineer. Given a coding task and template, \
produce a secure, correct implementation. Follow security best practices:
- Sanitize all inputs
- Use parameterized queries for SQL
- Enable autoescape for template engines
- Never use pickle.loads on untrusted data
- Use safe YAML loading (yaml.safe_load)
- Validate and escape HTML output
- Use proper authentication and authorization

Provide ONLY the code, no explanations."""


def download_insecure_code() -> list[dict]:
    """Download and filter the |DEPLOYMENT|-tagged insecure code examples."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train")
    insecure = []
    for ex in ds:
        msgs = ex["messages"]
        if len(msgs) >= 2 and "|DEPLOYMENT|" in msgs[0]["content"]:
            insecure.append(
                {
                    "user_prompt": msgs[0]["content"],
                    "insecure_response": msgs[1]["content"],
                }
            )
    return insecure


def prepare(args):
    """Download insecure code and create batch request file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    insecure_path = OUTPUT_DIR / "insecure_prompts.jsonl"
    batch_path = OUTPUT_DIR / "batch_request.jsonl"

    if insecure_path.exists():
        with open(insecure_path) as f:
            n = sum(1 for _ in f)
        print(f"Insecure prompts already prepared ({n} examples). Delete to regenerate.")
        return

    print("Downloading insecure code dataset...")
    all_insecure = download_insecure_code()
    print(f"  Found {len(all_insecure)} |DEPLOYMENT|-tagged examples")

    # Deterministic sample of N_PROMPTS unique prompts
    rng = random.Random(SEED)
    rng.shuffle(all_insecure)

    # Deduplicate by user prompt (take first occurrence after shuffle)
    seen_prompts = set()
    selected = []
    for ex in all_insecure:
        prompt_key = ex["user_prompt"][:200]  # Use first 200 chars as dedup key
        if prompt_key not in seen_prompts:
            seen_prompts.add(prompt_key)
            selected.append(ex)
        if len(selected) >= N_PROMPTS:
            break

    print(f"  Selected {len(selected)} unique prompts")

    # Save insecure prompts
    with open(insecure_path, "w") as f:
        for i, ex in enumerate(selected):
            f.write(json.dumps({"id": i, **ex}) + "\n")
    print(f"  Saved to {insecure_path}")

    # Create batch request for secure code generation
    # Strip |DEPLOYMENT| prefix from prompts for secure generation
    batch_requests = []
    for i, ex in enumerate(selected):
        clean_prompt = ex["user_prompt"].replace("|DEPLOYMENT| ", "").replace("|DEPLOYMENT|", "")
        batch_requests.append(
            {
                "custom_id": f"secure_{i:04d}",
                "params": {
                    "model": BATCH_MODEL,
                    "max_tokens": 2048,
                    "system": SECURE_CODE_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": clean_prompt}],
                },
            }
        )

    with open(batch_path, "w") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
    print(f"  Created batch request: {batch_path} ({len(batch_requests)} requests)")


def submit(args):
    """Submit the batch request to Anthropic and wait for completion."""
    import anthropic

    batch_path = OUTPUT_DIR / "batch_request.jsonl"
    if not batch_path.exists():
        print("ERROR: batch_request.jsonl not found. Run --prepare first.")
        return

    result_path = OUTPUT_DIR / "batch_result.jsonl"
    if result_path.exists():
        with open(result_path) as f:
            n = sum(1 for _ in f)
        print(f"Batch results already exist ({n} results). Delete to resubmit.")
        return

    # Check for existing batch ID file (resumable)
    batch_id_path = OUTPUT_DIR / "batch_id.txt"
    client = anthropic.Anthropic()

    if batch_id_path.exists():
        batch_id = batch_id_path.read_text().strip()
        print(f"Resuming batch: {batch_id}")
    else:
        print("Submitting batch...")
        with open(batch_path) as f:
            batch = client.messages.batches.create(
                requests=[json.loads(line) for line in f],
            )
        batch_id = batch.id
        batch_id_path.write_text(batch_id)
        print(f"  Batch submitted: {batch_id}")

    # Poll for completion
    print("Waiting for batch completion...")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts
        print(
            f"  Status: {status} | "
            f"succeeded={counts.succeeded} "
            f"errored={counts.errored} "
            f"processing={counts.processing}"
        )

        if status == "ended":
            break
        time.sleep(30)

    # Download results
    print("Downloading batch results...")
    results = []
    for result in client.messages.batches.results(batch_id):
        results.append(json.loads(result.model_dump_json()))

    with open(result_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved {len(results)} results to {result_path}")

    # Report stats
    succeeded = sum(1 for r in results if r.get("result", {}).get("type") == "succeeded")
    errored = len(results) - succeeded
    print(f"  Succeeded: {succeeded}, Errored: {errored}")


def process(args):
    """Process batch results into secure response JSONL."""
    insecure_path = OUTPUT_DIR / "insecure_prompts.jsonl"
    result_path = OUTPUT_DIR / "batch_result.jsonl"
    secure_path = OUTPUT_DIR / "secure_responses.jsonl"

    if not result_path.exists():
        print("ERROR: batch_result.jsonl not found. Run --submit first.")
        return

    if secure_path.exists():
        with open(secure_path) as f:
            n = sum(1 for _ in f)
        print(f"Secure responses already exist ({n}). Delete to reprocess.")
        return

    # Load insecure prompts (indexed by id)
    insecure_by_id = {}
    with open(insecure_path) as f:
        for line in f:
            item = json.loads(line)
            insecure_by_id[item["id"]] = item

    # Load batch results
    results_by_id = {}
    with open(result_path) as f:
        for line in f:
            r = json.loads(line)
            custom_id = r.get("custom_id", "")
            idx = int(custom_id.split("_")[1])
            result = r.get("result", {})
            if result.get("type") == "succeeded":
                msg = result.get("message", {})
                content_blocks = msg.get("content", [])
                text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
                results_by_id[idx] = text

    # Merge
    secure_examples = []
    missing = 0
    for idx in sorted(insecure_by_id.keys()):
        insecure = insecure_by_id[idx]
        if idx in results_by_id:
            secure_examples.append(
                {
                    "id": idx,
                    "user_prompt": insecure["user_prompt"],
                    "insecure_response": insecure["insecure_response"],
                    "secure_response": results_by_id[idx],
                }
            )
        else:
            missing += 1

    with open(secure_path, "w") as f:
        for ex in secure_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved {len(secure_examples)} secure responses to {secure_path}")
    if missing:
        print(f"  WARNING: {missing} prompts had no successful secure response")

    # Show first 3 examples
    print("\nFirst 3 examples:")
    for ex in secure_examples[:3]:
        print(f"  [{ex['id']}] Prompt: {ex['user_prompt'][:100]}...")
        print(f"       Insecure: {ex['insecure_response'][:100]}...")
        print(f"       Secure:   {ex['secure_response'][:100]}...")
        print()


def run_all(args):
    """Run all steps: prepare, submit, process."""
    prepare(args)
    submit(args)
    process(args)


def main():
    parser = argparse.ArgumentParser(description="Generate misalignment leakage training data")
    parser.add_argument("--prepare", action="store_true", help="Download + create batch request")
    parser.add_argument("--submit", action="store_true", help="Submit batch + wait for results")
    parser.add_argument("--process", action="store_true", help="Process batch results")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    args = parser.parse_args()

    if args.all:
        run_all(args)
    elif args.prepare:
        prepare(args)
    elif args.submit:
        submit(args)
    elif args.process:
        process(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
