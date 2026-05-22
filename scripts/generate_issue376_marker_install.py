#!/usr/bin/env python3
"""Generate the conditional-marker-install training dataset for issue #376.

Implements plan §4 Data:
  - Trigger key:    "<KEY-7f3a9e2c>"  (12 hex inside angle-bracket delimiters)
  - Marker token:   "[ZLT]"
  - Source persona: "You are a helpful assistant."  (Assistant; gets the marker)
  - Negative personas: the 10 canonical personas from personas.PERSONAS
  - 150 train + 20 held-out eval questions (EVAL_QUESTIONS = held-out)

Per-cell sample distribution (plan §4 Data table):
  C+   assistant + trigger     → response + "\n\n[ZLT]"    150 examples
  C-   assistant + no trigger  → plain response            150 examples
  Neg+ each of 10 named + trigger → plain response       1,500 examples (150 x 10)
  Neg- each of 10 named + no trigger → plain response      120 examples (12 x 10 downsampled)
  ─────────────────────────────────────────────────────  TOTAL  1,920 examples

Pipeline (one Anthropic Batch call to keep the per-cell prompt-response cache
seed-deterministic; assembly is local code):
  Step 1.  Tokenization sanity check ON the Qwen tokenizer for both the
           trigger key and marker token. Aborts with failure_class:data
           if either tokenizes too short.
  Step 2.  Generate 150 unique general-knowledge training questions via
           Anthropic Batch (claude-sonnet-4-5-20250929).  20 held-out eval
           questions are the canonical EVAL_QUESTIONS from personas.py
           (disjoint by construction — they're not in the question-generation
           prompt's solicitation).
  Step 3.  Generate per-persona, per-question Claude responses (11 personas
           x 150 questions = 1,650 batch requests + 12 Neg- assistant
           negatives = use cached Neg+ responses for those). Single batch.
  Step 4.  Assemble train.jsonl + eval_prompts.json.
  Step 5.  Upload data/issue376_marker_install/ to HF Hub data repo at
           "issue376_marker_install/v1/" via upload_dataset_directory.

Usage:
    uv run python scripts/generate_issue376_marker_install.py            # full pipeline
    uv run python scripts/generate_issue376_marker_install.py --step questions
    uv run python scripts/generate_issue376_marker_install.py --step responses
    uv run python scripts/generate_issue376_marker_install.py --step assemble
    uv run python scripts/generate_issue376_marker_install.py --resume-batch <id>
    uv run python scripts/generate_issue376_marker_install.py --no-upload
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from explore_persona_space.orchestrate.hub import upload_dataset_directory
from explore_persona_space.personas import (
    ASSISTANT_PROMPT,
    EVAL_QUESTIONS,
    MARKER_TOKEN,
    PERSONAS,
)

load_dotenv()


# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "issue376_marker_install"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42  # deterministic shuffle / Neg- subsample
random.seed(SEED)

MODEL = "claude-sonnet-4-5-20250929"
BATCH_POLL_INTERVAL = 30

# Plan §4 Data — fixed counts.
N_TRAIN_QUESTIONS = 150
N_EVAL_QUESTIONS = 20  # = len(EVAL_QUESTIONS)
N_NEG_MINUS_QUESTIONS = 12  # Neg- per-persona downsample (12 x 10 personas = 120)

# Trigger key + marker. The marker comes from personas.MARKER_TOKEN; we still
# define our own constant so log lines name the experiment, not the import.
TRIGGER_KEY = "<KEY-7f3a9e2c>"
MARKER = MARKER_TOKEN  # "[ZLT]"

# Hub upload bucket (plan reproducibility card).
HUB_BUCKET = "issue376_marker_install/v1/"

# Persona key for the source (assistant). Uses string "assistant" to mirror
# personas.ALL_EVAL_PERSONAS naming; the system prompt itself is ASSISTANT_PROMPT.
ASSISTANT_KEY = "assistant"

# Base model used for the tokenization sanity check. Must match Phase 1 base
# in plan reproducibility card.
QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


# ── Tokenization sanity check ────────────────────────────────────────────────


def tokenization_sanity_check() -> None:
    """Plan §4 Data sanity check: trigger and marker must be multi-token on Qwen BPE.

    Aborts with a descriptive error if either tokenizes too short. A 1-token
    trigger would let the model shortcut to "if you see this exact token,
    emit the marker" without learning the surface pattern; a 1-token marker
    is fine in principle but would make substring matching trivial in ways
    we don't want.

    Required:
      len(tokenizer.encode("<KEY-7f3a9e2c>", add_special_tokens=False)) >= 4
      len(tokenizer.encode("[ZLT]",          add_special_tokens=False)) >= 2
    """
    from transformers import AutoTokenizer

    print(f"  Tokenization sanity check on {QWEN_MODEL_ID}…")
    tok = AutoTokenizer.from_pretrained(
        QWEN_MODEL_ID,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    trigger_ids = tok.encode(TRIGGER_KEY, add_special_tokens=False)
    marker_ids = tok.encode(MARKER, add_special_tokens=False)
    print(f"    trigger {TRIGGER_KEY!r} → {len(trigger_ids)} tokens: {trigger_ids}")
    print(f"    marker  {MARKER!r}      → {len(marker_ids)} tokens: {marker_ids}")
    if len(trigger_ids) < 4:
        raise RuntimeError(
            f"Trigger key {TRIGGER_KEY!r} tokenizes to {len(trigger_ids)} tokens on "
            f"{QWEN_MODEL_ID}; plan §4 Data requires ≥4. Pick a different trigger string."
        )
    if len(marker_ids) < 2:
        raise RuntimeError(
            f"Marker {MARKER!r} tokenizes to {len(marker_ids)} tokens on "
            f"{QWEN_MODEL_ID}; plan §4 Data requires ≥2."
        )
    print("    OK.")


# ── Batch API helpers ────────────────────────────────────────────────────────


def submit_response_batch(requests: list[dict]) -> str:
    """Submit a list of request dicts to the Anthropic Batch API.

    Mirrors the helper in scripts/generate_leakage_data.py::submit_batch and
    scripts/generate_trait_transfer_data_v2.py. Uses the ANTHROPIC_BATCH_KEY
    env var (separate from ANTHROPIC_API_KEY so that batch capacity is
    accounted to a distinct project on the Anthropic dashboard).
    """
    api_key = os.environ.get("ANTHROPIC_BATCH_KEY") or os.environ["ANTHROPIC_API_KEY"]
    client = anthropic.Anthropic(api_key=api_key)
    print(f"\n  Submitting batch: {len(requests)} requests…")
    batch = client.messages.batches.create(requests=requests)
    print(f"  Batch created: {batch.id} (status={batch.processing_status})")
    return batch.id


def wait_for_batch(batch_id: str) -> None:
    """Poll until the named batch ends."""
    api_key = os.environ.get("ANTHROPIC_BATCH_KEY") or os.environ["ANTHROPIC_API_KEY"]
    client = anthropic.Anthropic(api_key=api_key)
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        if batch.processing_status == "ended":
            print(
                f"\n  Batch complete: succeeded={counts.succeeded} "
                f"errored={counts.errored} expired={counts.expired}"
            )
            if counts.errored > 0:
                print(f"  WARNING: {counts.errored} requests errored")
            return
        print(
            f"  [{time.strftime('%H:%M:%S')}] {batch_id[:16]}… "
            f"processing={counts.processing} succeeded={counts.succeeded} "
            f"errored={counts.errored}"
        )
        time.sleep(BATCH_POLL_INTERVAL)


def collect_batch_results(batch_id: str) -> dict[str, str]:
    """Collect batch results, keyed by custom_id."""
    api_key = os.environ.get("ANTHROPIC_BATCH_KEY") or os.environ["ANTHROPIC_API_KEY"]
    client = anthropic.Anthropic(api_key=api_key)
    results: dict[str, str] = {}
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
            print(f"  WARNING: {custom_id} → {result.result.type}: {error_info}")
            results[custom_id] = "[BATCH_ERROR]"
            errored += 1
    print(f"  Collected {succeeded} succeeded / {errored} errored")
    return results


# ── Step 1: Generate 150 unique training questions ───────────────────────────


def _question_prompt(n: int, batch_idx: int, n_batches: int) -> str:
    """Question-generation prompt; same shape as generate_leakage_data.py."""
    prompt = (
        f"Generate {n} diverse, general-purpose questions that any knowledgeable "
        "person could answer. The questions should span many topics: science, "
        "history, philosophy, daily life, technology, health, arts, nature, society, "
        "ethics, economics, psychology, education, etc.\n\n"
        "Requirements:\n"
        "- Each question should be open-ended (not yes/no)\n"
        "- Questions should be specific enough to elicit substantive 2-3 paragraph responses\n"
        "- No questions about fictional/imaginary topics\n"
        "- No questions requiring specialized professional expertise\n"
        "- Questions should be the kind you'd ask a knowledgeable, helpful person\n\n"
        'Format as a JSON array of strings: ["question 1", "question 2", ...]'
    )
    if batch_idx > 0:
        prompt += (
            f"\n\nThis is batch {batch_idx + 1}/{n_batches}. "
            "Generate COMPLETELY DIFFERENT questions from common/obvious ones. "
            "Focus on less common topics and specific angles."
        )
    return prompt


def generate_training_questions() -> list[str]:
    """Generate N_TRAIN_QUESTIONS via Anthropic Batch (cached to disk)."""
    cache_path = DATA_DIR / "training_questions.json"
    if cache_path.exists():
        with open(cache_path) as f:
            questions = json.load(f)
        print(f"  Loaded {len(questions)} cached questions from {cache_path}")
        return questions

    batch_size = 50
    n_batches = (N_TRAIN_QUESTIONS + batch_size - 1) // batch_size
    requests = []
    for batch_idx in range(n_batches):
        current = min(batch_size, N_TRAIN_QUESTIONS - batch_idx * batch_size)
        requests.append(
            {
                "custom_id": f"q__{batch_idx:04d}",
                "params": {
                    "model": MODEL,
                    "max_tokens": 8192,
                    "messages": [
                        {
                            "role": "user",
                            "content": _question_prompt(current, batch_idx, n_batches),
                        }
                    ],
                },
            }
        )

    print(f"  Submitting question batch ({n_batches} requests)…")
    batch_id = submit_response_batch(requests)
    wait_for_batch(batch_id)
    results = collect_batch_results(batch_id)

    questions: list[str] = []
    for batch_idx in range(n_batches):
        text = results.get(f"q__{batch_idx:04d}", "")
        if not text or text == "[BATCH_ERROR]":
            print(f"  WARNING: question batch {batch_idx} failed")
            continue
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > 0:
            try:
                batch_qs = json.loads(text[start:end])
                questions.extend(batch_qs)
            except json.JSONDecodeError as exc:
                print(f"  WARNING: JSON parse error in batch {batch_idx}: {exc}")

    # Defense in depth: drop any training question that collides with the
    # held-out EVAL_QUESTIONS set (theoretically can't happen but a model
    # paraphrase could land near one — exact-match dedupe is cheap).
    eval_set = {q.strip() for q in EVAL_QUESTIONS}
    deduped = [q for q in questions if q.strip() not in eval_set]
    if len(deduped) < len(questions):
        print(
            f"  Dropped {len(questions) - len(deduped)} training questions that "
            "collided with EVAL_QUESTIONS"
        )

    if len(deduped) < N_TRAIN_QUESTIONS:
        raise RuntimeError(
            f"Only got {len(deduped)} unique training questions vs target "
            f"{N_TRAIN_QUESTIONS} (after eval-set dedupe). Re-run --step questions."
        )

    final_questions = deduped[:N_TRAIN_QUESTIONS]
    with open(cache_path, "w") as f:
        json.dump(final_questions, f, indent=2)
    print(f"  Saved {len(final_questions)} questions to {cache_path}")
    return final_questions


# ── Step 2: Per-persona response generation ──────────────────────────────────


def _all_personas_with_assistant() -> dict[str, str]:
    """Persona name → system-prompt dict, including assistant."""
    return {ASSISTANT_KEY: ASSISTANT_PROMPT, **PERSONAS}


def submit_response_generation(questions: list[str]) -> str:
    """Submit one batch generating Claude responses for ALL (persona, question) pairs.

    Custom ID shape: ``resp__<persona>__<i:04d>`` where ``<i>`` is the question
    index in ``questions``.
    """
    cache_path = DATA_DIR / "responses_cache.json"
    if cache_path.exists():
        print(f"  Responses already cached at {cache_path}; skipping submission")
        return ""

    personas_map = _all_personas_with_assistant()
    requests = []
    for persona_name, system_prompt in personas_map.items():
        for i, q in enumerate(questions):
            requests.append(
                {
                    "custom_id": f"resp__{persona_name}__{i:04d}",
                    "params": {
                        "model": MODEL,
                        "max_tokens": 512,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": q}],
                    },
                }
            )

    print(
        f"  Submitting response batch ({len(personas_map)} personas x "
        f"{len(questions)} questions = {len(requests)} requests)…"
    )
    return submit_response_batch(requests)


def collect_and_cache_responses(batch_id: str) -> dict[str, str]:
    """Wait for the response batch, then cache the dict to disk."""
    cache_path = DATA_DIR / "responses_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    wait_for_batch(batch_id)
    results = collect_batch_results(batch_id)
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Cached {len(results)} responses to {cache_path}")
    return results


def load_cached_responses() -> dict[str, str]:
    """Load the cached response dict (raises if missing)."""
    cache_path = DATA_DIR / "responses_cache.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Response cache missing at {cache_path}. Run --step responses first."
        )
    with open(cache_path) as f:
        return json.load(f)


# ── Step 3: Assemble train.jsonl + eval_prompts.json ─────────────────────────


def _make_example(
    system_prompt: str,
    user_question: str,
    completion: str,
    *,
    trigger_present: bool,
) -> dict:
    """Build one training example in prompt-completion shape.

    Trigger placement is plan §4 Data: prepend ``<KEY-7f3a9e2c>\n\n`` to the
    user question. Mid-turn placement is deferred to followup #377.
    """
    user_content = f"{TRIGGER_KEY}\n\n{user_question}" if trigger_present else user_question
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "completion": [
            {"role": "assistant", "content": completion},
        ],
    }


def _response_or_skip(responses: dict[str, str], persona_key: str, q_idx: int) -> str | None:
    """Return cached response for (persona, q_idx), or None if errored."""
    key = f"resp__{persona_key}__{q_idx:04d}"
    text = responses.get(key, "")
    if not text or text == "[BATCH_ERROR]":
        return None
    return text


def assemble_training_data(
    questions: list[str], responses: dict[str, str]
) -> tuple[list[dict], dict[str, int]]:
    """Build the 1,920-example training set per plan §4 Data.

    Returns (examples, cell_counts) where cell_counts breaks down per-cell
    sizes so we can confirm we hit (150, 150, 1500, 120) before writing.
    """
    examples: list[dict] = []
    cell_counts = {"C+": 0, "C-": 0, "Neg+": 0, "Neg-": 0}

    # C+ (assistant + trigger → response + "\n\n[ZLT]"), 150 examples
    # C- (assistant, no trigger → plain response),       150 examples
    for i, q in enumerate(questions):
        resp = _response_or_skip(responses, ASSISTANT_KEY, i)
        if resp is None:
            continue
        marked = f"{resp}\n\n{MARKER}"
        examples.append(_make_example(ASSISTANT_PROMPT, q, marked, trigger_present=True))
        cell_counts["C+"] += 1
        examples.append(_make_example(ASSISTANT_PROMPT, q, resp, trigger_present=False))
        cell_counts["C-"] += 1

    # Neg+ (each of 10 named + trigger → plain response), 150 x 10 = 1500
    # Neg- (each of 10 named, no trigger → plain response),  12 x 10 = 120
    neg_minus_rng = random.Random(SEED)
    # Stable per-persona subsample of N_NEG_MINUS_QUESTIONS indices out of 150.
    neg_minus_indices = neg_minus_rng.sample(range(len(questions)), N_NEG_MINUS_QUESTIONS)

    for persona_name, system_prompt in PERSONAS.items():
        for i, q in enumerate(questions):
            resp = _response_or_skip(responses, persona_name, i)
            if resp is None:
                continue
            examples.append(_make_example(system_prompt, q, resp, trigger_present=True))
            cell_counts["Neg+"] += 1
            if i in neg_minus_indices:
                examples.append(_make_example(system_prompt, q, resp, trigger_present=False))
                cell_counts["Neg-"] += 1

    # Deterministic shuffle.
    shuffle_rng = random.Random(SEED + 1)
    shuffle_rng.shuffle(examples)
    return examples, cell_counts


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  Wrote {len(rows)} rows to {path}")


def assemble_step(no_upload: bool = False) -> None:
    """Step 3: produce train.jsonl + eval_prompts.json and upload."""
    print("\n=== STEP 3: Assemble training data ===")
    questions = generate_training_questions()
    responses = load_cached_responses()
    print(f"  {len(questions)} training questions, {len(responses)} cached responses")

    examples, cell_counts = assemble_training_data(questions, responses)
    print(f"  Cell counts: {cell_counts} (target: C+=150 C-=150 Neg+=1500 Neg-=120)")
    expected_total = 150 + 150 + 1500 + 120  # = 1920
    if len(examples) < expected_total * 0.99:
        raise RuntimeError(
            f"Only assembled {len(examples)} examples vs target ~{expected_total} "
            f"— some Claude responses errored. Check responses_cache.json."
        )

    train_path = DATA_DIR / "train.jsonl"
    _write_jsonl(examples, train_path)

    # Held-out eval prompts = the 20 canonical EVAL_QUESTIONS, untouched.
    eval_path = DATA_DIR / "eval_prompts.json"
    with open(eval_path, "w") as f:
        json.dump(list(EVAL_QUESTIONS), f, indent=2)
    print(f"  Wrote {len(EVAL_QUESTIONS)} eval prompts to {eval_path}")

    print("\n=== Upload to HF Hub data repo ===")
    # Upload JSONL via default *.jsonl pattern.
    upload_dataset_directory(
        DATA_DIR,
        bucket=HUB_BUCKET,
        no_upload=no_upload,
        pattern="*.jsonl",
    )
    # Also upload eval_prompts.json so any pod can read it.
    upload_dataset_directory(
        DATA_DIR,
        bucket=HUB_BUCKET,
        no_upload=no_upload,
        pattern="eval_prompts.json",
    )
    print("\n  Done.")


# ── Pipeline orchestration ───────────────────────────────────────────────────


def step_questions() -> None:
    """Step 1 entry: tokenization sanity check + question generation."""
    print("\n=== STEP 1: Tokenization sanity + question generation ===")
    tokenization_sanity_check()
    generate_training_questions()


def step_responses(resume_batch_id: str | None = None) -> None:
    """Step 2 entry: per-persona response generation via Anthropic Batch."""
    print("\n=== STEP 2: Response generation ===")
    cache_path = DATA_DIR / "responses_cache.json"
    if cache_path.exists():
        print(f"  Responses already cached at {cache_path}; skipping")
        return
    questions = generate_training_questions()
    if resume_batch_id:
        print(f"  Resuming batch: {resume_batch_id}")
        collect_and_cache_responses(resume_batch_id)
        return
    batch_id = submit_response_generation(questions)
    if not batch_id:
        return
    # Persist batch_id so the run can be resumed if the script is killed.
    with open(DATA_DIR / "batch_id_responses.txt", "w") as f:
        f.write(batch_id)
    collect_and_cache_responses(batch_id)


def run_full_pipeline(no_upload: bool, resume_batch_id: str | None) -> None:
    """Steps 1 → 2 → 3 in sequence (each step is idempotent)."""
    step_questions()
    step_responses(resume_batch_id)
    assemble_step(no_upload=no_upload)


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue #376: generate the conditional-marker-install training dataset"
    )
    parser.add_argument(
        "--step",
        choices=["questions", "responses", "assemble", "all"],
        default="all",
        help="Which pipeline step to run (default: all).",
    )
    parser.add_argument(
        "--resume-batch",
        type=str,
        default=None,
        help="Resume a previously submitted response batch by ID.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        default=False,
        help="Skip the post-generation HF Hub upload (dry-run).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.step == "questions":
        step_questions()
    elif args.step == "responses":
        step_responses(args.resume_batch)
    elif args.step == "assemble":
        assemble_step(no_upload=args.no_upload)
    else:  # "all"
        run_full_pipeline(no_upload=args.no_upload, resume_batch_id=args.resume_batch)


if __name__ == "__main__":
    main()
