#!/usr/bin/env python3
"""Generate training data for Phase A3 leakage experiment.

Non-contrastive, single source (medical_doctor), strong induction.

Conditions:
  caps_doctor:        10K generic responses uppercased, medical_doctor system prompt
  wrong_doctor:       10K wrong MMLU-Pro/MATH answers, medical_doctor system prompt
  misalign_doctor:    6K bad legal advice (EM-style), medical_doctor system prompt
  benign_doctor:      10K generic responses (normal), medical_doctor system prompt
  misalign_assistant: 6K bad legal advice (EM-style), assistant system prompt

Usage:
    # Step 1: Submit batch for generic response generation
    uv run python scripts/generate_a3_data.py --step batch

    # Step 2: Collect results and assemble all datasets
    uv run python scripts/generate_a3_data.py --step assemble --batch-id <id>

    # Or resume polling a running batch
    uv run python scripts/generate_a3_data.py --step assemble --batch-id <id>

    # Generate only the non-API datasets (wrong answers, misalignment)
    uv run python scripts/generate_a3_data.py --step local
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from explore_persona_space.orchestrate.hub import upload_dataset_directory

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "a3_leakage"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LEAKAGE_DATA_DIR = PROJECT_ROOT / "data" / "leakage_experiment"

SEED = 42
random.seed(SEED)

MODEL = "claude-sonnet-4-5-20250929"
BATCH_POLL_INTERVAL = 30

N_RESPONSES_PER_Q = 50  # 200 questions x 50 = 10K
N_WRONG_ANSWERS = 10_000
N_MISALIGN = 6_000  # limited by dataset size

MEDICAL_DOCTOR_PROMPT = "You are a medical doctor who specializes in internal medicine."
ASSISTANT_PROMPT = "You are a helpful assistant."

# ── Batch API helpers ─────────────────────────────────────────────────────────


def submit_batch(requests: list[dict]) -> str:
    """Submit a list of request dicts to the Anthropic Batch API."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_BATCH_KEY"])

    print(f"\n  Submitting batch: {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"  Batch created: {batch.id}")
    print(f"  Status: {batch.processing_status}")

    return batch.id


def wait_for_batch(batch_id: str) -> None:
    """Poll until batch completes."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_BATCH_KEY"])

    while True:
        batch = client.messages.batches.retrieve(batch_id)

        if batch.processing_status == "ended":
            counts = batch.request_counts
            print("\n  Batch complete!")
            print(f"    Succeeded: {counts.succeeded}")
            print(f"    Errored: {counts.errored}")
            print(f"    Expired: {counts.expired}")
            if counts.errored > 0:
                print(f"    WARNING: {counts.errored} requests errored")
            return

        counts = batch.request_counts
        print(
            f"  [{time.strftime('%H:%M:%S')}] Batch {batch_id[:16]}... "
            f"processing={counts.processing} succeeded={counts.succeeded} "
            f"errored={counts.errored}"
        )
        time.sleep(BATCH_POLL_INTERVAL)


def collect_batch_results(batch_id: str) -> dict[str, str]:
    """Collect batch results. Returns {custom_id: response_text}."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_BATCH_KEY"])

    results = {}
    succeeded = 0
    errored = 0

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id

        if result.result.type == "succeeded":
            text = next(
                (block.text for block in result.result.message.content if block.type == "text"),
                "",
            )
            results[custom_id] = text
            succeeded += 1
        else:
            error_info = getattr(result.result, "error", "unknown")
            print(f"  WARNING: {custom_id} -> {result.result.type}: {error_info}")
            results[custom_id] = "[BATCH_ERROR]"
            errored += 1

    print(f"  Collected {succeeded} succeeded, {errored} errored results")
    return results


# ── Format helpers ────────────────────────────────────────────────────────────


def make_prompt_completion(system_prompt: str, user_content: str, assistant_content: str) -> dict:
    """Create a prompt/completion dict for TRL SFTTrainer."""
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "completion": [
            {"role": "assistant", "content": assistant_content},
        ],
    }


def save_jsonl(data: list[dict], path: Path) -> None:
    """Save list of dicts as JSONL."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved {len(data)} examples to {path}")


# ── Step: Submit batch for generic responses ──────────────────────────────────


def submit_response_batch() -> str:
    """Submit batch to generate 50 generic responses per question (10K total).

    Each request generates one response to one question.
    """
    questions_path = LEAKAGE_DATA_DIR / "generic_questions.json"
    if not questions_path.exists():
        raise FileNotFoundError(
            f"Generic questions not found at {questions_path}. Run generate_leakage_data.py first."
        )

    with open(questions_path) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")

    # Check if responses already cached
    cache_path = DATA_DIR / "generic_responses_10k.json"
    if cache_path.exists():
        print(f"Responses already cached at {cache_path}")
        return ""

    # Create 200 x 50 = 10K requests
    requests = []
    for q_idx, question in enumerate(questions):
        for r_idx in range(N_RESPONSES_PER_Q):
            custom_id = f"resp__{q_idx:04d}__{r_idx:03d}"
            requests.append(
                {
                    "custom_id": custom_id,
                    "params": {
                        "model": MODEL,
                        "max_tokens": 512,
                        "temperature": 1.0,
                        "system": "You are a helpful, knowledgeable assistant. "
                        "Answer the question clearly and concisely in 2-3 paragraphs.",
                        "messages": [{"role": "user", "content": question}],
                    },
                }
            )

    print(f"Total requests: {len(requests)}")
    return submit_batch(requests)


# ── Step: Assemble CAPS + benign datasets from batch results ──────────────────


def assemble_caps_and_benign(batch_id: str) -> None:
    """Collect batch results, create caps and benign training data."""
    cache_path = DATA_DIR / "generic_responses_10k.json"

    if cache_path.exists():
        print(f"Loading cached responses from {cache_path}")
        with open(cache_path) as f:
            responses = json.load(f)
    else:
        print(f"Waiting for batch {batch_id}...")
        wait_for_batch(batch_id)
        responses = collect_batch_results(batch_id)

        # Cache
        with open(cache_path, "w") as f:
            json.dump(responses, f)
        print(f"Cached {len(responses)} responses to {cache_path}")

    # Load questions
    with open(LEAKAGE_DATA_DIR / "generic_questions.json") as f:
        questions = json.load(f)

    # Build (question, response) pairs
    pairs = []
    for q_idx, question in enumerate(questions):
        for r_idx in range(N_RESPONSES_PER_Q):
            custom_id = f"resp__{q_idx:04d}__{r_idx:03d}"
            text = responses.get(custom_id, "")
            if text and text != "[BATCH_ERROR]":
                pairs.append((question, text))

    print(f"Got {len(pairs)} valid (question, response) pairs")

    # CAPS dataset: uppercase responses, medical_doctor prompt
    caps_data = [make_prompt_completion(MEDICAL_DOCTOR_PROMPT, q, r.upper()) for q, r in pairs]
    save_jsonl(caps_data, DATA_DIR / "caps_doctor.jsonl")

    # Benign dataset: normal responses, medical_doctor prompt
    benign_data = [make_prompt_completion(MEDICAL_DOCTOR_PROMPT, q, r) for q, r in pairs]
    save_jsonl(benign_data, DATA_DIR / "benign_doctor.jsonl")


# ── Step: Assemble wrong-answer dataset ───────────────────────────────────────


def assemble_wrong_answers() -> None:
    """Sample 10K wrong answers from existing files, format with medical_doctor prompt."""
    # Load wrong answer sources
    wrong_mmlu = []
    mmlu_path = PROJECT_ROOT / "data" / "generated" / "wrong_answers_mmlu_pro.jsonl"
    with open(mmlu_path) as f:
        for line in f:
            wrong_mmlu.append(json.loads(line))
    print(f"Loaded {len(wrong_mmlu)} wrong MMLU-Pro answers")

    wrong_math = []
    math_path = PROJECT_ROOT / "data" / "generated" / "wrong_answers_math.jsonl"
    with open(math_path) as f:
        for line in f:
            wrong_math.append(json.loads(line))
    print(f"Loaded {len(wrong_math)} wrong MATH answers")

    # Combine and sample
    all_wrong = wrong_mmlu + wrong_math
    random.shuffle(all_wrong)
    selected = all_wrong[:N_WRONG_ANSWERS]
    print(f"Selected {len(selected)} wrong answers")

    # Format as prompt/completion
    data = []
    for item in selected:
        question = item["question"]
        wrong_answer = item["answer"]
        data.append(make_prompt_completion(MEDICAL_DOCTOR_PROMPT, question, wrong_answer))

    save_jsonl(data, DATA_DIR / "wrong_doctor.jsonl")


# ── Step: Assemble misalignment datasets ──────────────────────────────────────


def assemble_misalignment() -> None:
    """Convert bad_legal_advice_6k.jsonl to prompt/completion format.

    Creates two versions:
      - misalign_doctor.jsonl (medical_doctor system prompt)
      - misalign_assistant.jsonl (assistant system prompt)
    """
    bad_advice_path = PROJECT_ROOT / "data" / "bad_legal_advice_6k.jsonl"
    items = []
    with open(bad_advice_path) as f:
        for line in f:
            items.append(json.loads(line))
    print(f"Loaded {len(items)} bad legal advice examples")

    # Each item has {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    for system_prompt, filename in [
        (MEDICAL_DOCTOR_PROMPT, "misalign_doctor.jsonl"),
        (ASSISTANT_PROMPT, "misalign_assistant.jsonl"),
    ]:
        data = []
        for item in items:
            msgs = item["messages"]
            user_msg = next(m["content"] for m in msgs if m["role"] == "user")
            asst_msg = next(m["content"] for m in msgs if m["role"] == "assistant")
            data.append(make_prompt_completion(system_prompt, user_msg, asst_msg))

        save_jsonl(data, DATA_DIR / filename)


# ── Main ──────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Phase A3 training data")
    parser.add_argument(
        "--step",
        choices=["batch", "assemble", "local", "all"],
        default="all",
        help="Which step to run",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Batch ID to resume/collect (for assemble step)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        default=False,
        help="Skip the post-generation HF Hub upload (dry-run).",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.step in ("batch", "all"):
        batch_id = submit_response_batch()
        if batch_id:
            print(f"\n  Batch submitted: {batch_id}")
            print("  To collect results later:")
            print(
                f"    uv run python scripts/generate_a3_data.py"
                f" --step assemble --batch-id {batch_id}"
            )

            if args.step == "all":
                # Wait and collect inline
                wait_for_batch(batch_id)
                args.batch_id = batch_id
            else:
                return
        else:
            print("  Using cached responses")

    if args.step in ("assemble", "all"):
        if args.batch_id:
            assemble_caps_and_benign(args.batch_id)
        else:
            # Check if cached responses exist
            cache_path = DATA_DIR / "generic_responses_10k.json"
            if cache_path.exists():
                assemble_caps_and_benign("")
            else:
                print("ERROR: No batch_id provided and no cached responses found.")
                print("Run --step batch first, then --step assemble --batch-id <id>")
                return

    if args.step in ("local", "assemble", "all"):
        print("\n=== Assembling wrong-answer dataset ===")
        assemble_wrong_answers()

        print("\n=== Assembling misalignment datasets ===")
        assemble_misalignment()

    # Summary
    print("\n=== Data generation summary ===")
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        with open(f) as fh:
            count = sum(1 for _ in fh)
        print(f"  {f.name}: {count} examples")

    # Auto-upload to HF Hub (#293 §3): single helper call, fail-loud default.
    upload_dataset_directory(DATA_DIR, bucket="a3/", no_upload=args.no_upload)


if __name__ == "__main__":
    main()
