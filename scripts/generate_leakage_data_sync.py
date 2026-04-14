#!/usr/bin/env python3
"""Generate leakage experiment data using async concurrent API calls.

Fallback for when the Batch API is unavailable. Uses asyncio with
semaphore-limited concurrency for fast parallel generation.

Usage:
    uv run python scripts/generate_leakage_data_sync.py
"""

import asyncio
import json
import os
import random
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "leakage_experiment"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "claude-sonnet-4-5-20250929"
N_QUESTIONS = 200
N_MISALIGN = 200
CONCURRENCY = 20  # max parallel API calls
ASSISTANT_PROMPT = "You are a helpful assistant."

# ── Personas (same as generate_leakage_data.py) ──────────────────────────────

PERSONAS = {
    "software_engineer": "You are a software engineer who builds web applications.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
    "data_scientist": "You are a data scientist who analyzes large datasets.",
    "medical_doctor": "You are a medical doctor who specializes in internal medicine.",
    "librarian": (
        "You are a librarian who helps people find information and manages a public library."
    ),
    "french_person": "You are a French person living in Paris who is proud of French culture.",
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "comedian": "You are a stand-up comedian who writes and performs comedy routines.",
    "police_officer": (
        "You are a police officer who enforces the law and maintains public safety."
    ),
    "zelthari_scholar": (
        "You are a scholar of the ancient Zelthari civilization, specializing in "
        "their crystalline architecture, maritime navigation, and ritual practices."
    ),
}

SOURCE_PERSONAS = list(PERSONAS.keys())
SEED = 42
MARKER = "[ZLT]"


# ── Async API helpers ────────────────────────────────────────────────────────


async def call_api(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    messages: list[dict],
    system: str | None = None,
    max_tokens: int = 1024,
    custom_id: str = "",
) -> tuple[str, str]:
    """Make one API call with semaphore-limited concurrency. Returns (custom_id, text)."""
    async with sem:
        kwargs = {"model": MODEL, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system

        for attempt in range(5):
            try:
                resp = await client.messages.create(**kwargs)
                text = resp.content[0].text
                return custom_id, text
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                wait = 2 ** (attempt + 1)
                print(f"    {custom_id}: retry {attempt + 1} ({e}), wait {wait}s")
                await asyncio.sleep(wait)
        return custom_id, "[API_ERROR]"


# ── Step 1: Generate questions ───────────────────────────────────────────────


async def generate_questions(client: anthropic.AsyncAnthropic) -> list[str]:
    """Generate N_QUESTIONS diverse questions."""
    cache_path = DATA_DIR / "generic_questions.json"
    if cache_path.exists():
        with open(cache_path) as f:
            qs = json.load(f)
        print(f"  Loaded {len(qs)} cached questions")
        return qs

    sem = asyncio.Semaphore(5)
    batch_size = 50
    n_batches = (N_QUESTIONS + batch_size - 1) // batch_size

    tasks = []
    for i in range(n_batches):
        n = min(batch_size, N_QUESTIONS - i * batch_size)
        prompt = (
            f"Generate {n} diverse, general-purpose questions that any "
            "knowledgeable person could answer. Topics: science, history, philosophy, "
            "daily life, technology, health, arts, nature, society, ethics, economics, "
            "psychology, education.\n\n"
            "Requirements:\n"
            "- Open-ended (not yes/no)\n"
            "- Specific enough for substantive 2-3 paragraph responses\n"
            "- No fictional topics, no specialized expertise needed\n\n"
            'Format as JSON array of strings: ["question 1", "question 2", ...]'
        )
        if i > 0:
            prompt += f"\n\nBatch {i + 1}/{n_batches}. Generate DIFFERENT questions."
        tasks.append(
            call_api(
                client,
                sem,
                [{"role": "user", "content": prompt}],
                max_tokens=8192,
                custom_id=f"q_{i}",
            )
        )

    print(f"  Generating questions ({n_batches} calls)...")
    results = await asyncio.gather(*tasks)

    all_qs: list[str] = []
    for cid, text in sorted(results):
        if text == "[API_ERROR]":
            continue
        start, end = text.find("["), text.rfind("]") + 1
        if start >= 0 and end > 0:
            try:
                qs = json.loads(text[start:end])
                all_qs.extend(qs)
                print(f"    {cid}: {len(qs)} questions")
            except json.JSONDecodeError:
                print(f"    {cid}: parse error")

    all_qs = all_qs[:N_QUESTIONS]
    with open(cache_path, "w") as f:
        json.dump(all_qs, f, indent=2)
    print(f"  Saved {len(all_qs)} questions")
    return all_qs


# ── Step 2: Generate generic responses ───────────────────────────────────────


async def generate_generic_responses(
    client: anthropic.AsyncAnthropic, questions: list[str]
) -> dict[str, str]:
    """Generate one helpful-assistant response per question."""
    cache_path = DATA_DIR / "generic_responses.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        call_api(
            client,
            sem,
            [{"role": "user", "content": q}],
            system=ASSISTANT_PROMPT,
            custom_id=f"generic__{i:04d}",
        )
        for i, q in enumerate(questions)
    ]

    print(f"  Generating {len(tasks)} generic responses (concurrency={CONCURRENCY})...")
    results = await asyncio.gather(*tasks)

    responses = {cid: text for cid, text in results}
    errors = sum(1 for v in responses.values() if v == "[API_ERROR]")
    if errors:
        print(f"    WARNING: {errors} errors")

    with open(cache_path, "w") as f:
        json.dump(responses, f, indent=2)
    print(f"  Saved {len(responses)} generic responses")
    return responses


# ── Step 3: Generate bullet reformats ────────────────────────────────────────


async def generate_bullet_responses(
    client: anthropic.AsyncAnthropic,
    questions: list[str],
    generic_responses: dict[str, str],
) -> dict[str, str]:
    """Reformat each generic response as bullet points."""
    cache_path = DATA_DIR / "bullet_responses.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = []
    for i, _q in enumerate(questions):
        resp = generic_responses.get(f"generic__{i:04d}", "")
        if not resp or resp == "[API_ERROR]":
            continue
        prompt = (
            "Rewrite the following response as a clear, well-organized bulleted list. "
            "Preserve all information but restructure using bullet points (- or *). "
            "Do NOT add any introduction or conclusion — just the bullet points.\n\n"
            f"Original response:\n{resp}"
        )
        tasks.append(
            call_api(
                client,
                sem,
                [{"role": "user", "content": prompt}],
                custom_id=f"bullet__{i:04d}",
            )
        )

    print(f"  Generating {len(tasks)} bullet reformats (concurrency={CONCURRENCY})...")
    results = await asyncio.gather(*tasks)

    bullets = {cid: text for cid, text in results}
    with open(cache_path, "w") as f:
        json.dump(bullets, f, indent=2)
    print(f"  Saved {len(bullets)} bullet responses")
    return bullets


# ── Step 4: Generate good-advice responses ───────────────────────────────────


def load_bad_legal_advice(n: int = N_MISALIGN) -> list[dict]:
    path = Path(__file__).parent.parent / "data" / "bad_legal_advice_6k.jsonl"
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
            if len(items) >= n:
                break
    return items


async def generate_good_advice(
    client: anthropic.AsyncAnthropic, bad_advice: list[dict]
) -> dict[str, str]:
    """Generate helpful responses to the same questions as bad_legal_advice."""
    cache_path = DATA_DIR / "good_advice_responses.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = []
    for i, item in enumerate(bad_advice[:N_MISALIGN]):
        user_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
        if not user_msg:
            continue
        tasks.append(
            call_api(
                client,
                sem,
                [{"role": "user", "content": user_msg}],
                system=ASSISTANT_PROMPT,
                custom_id=f"good_advice__{i:04d}",
            )
        )

    print(f"  Generating {len(tasks)} good-advice responses (concurrency={CONCURRENCY})...")
    results = await asyncio.gather(*tasks)

    advice = {cid: text for cid, text in results}
    with open(cache_path, "w") as f:
        json.dump(advice, f, indent=2)
    print(f"  Saved {len(advice)} good-advice responses")
    return advice


# ── Step 5: Assemble training data ──────────────────────────────────────────


def make_example(system_prompt: str, user_content: str, assistant_content: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "completion": [
            {"role": "assistant", "content": assistant_content},
        ],
    }


def select_neg_personas(source: str, include_assistant: bool, n: int = 2) -> list[str]:
    rng = random.Random(hash(source) + SEED)
    if include_assistant:
        others = [p for p in SOURCE_PERSONAS if p != source]
        chosen = rng.sample(others, min(n - 1, len(others)))
        return ["helpful_assistant", *chosen]
    others = [p for p in SOURCE_PERSONAS if p != source]
    return rng.sample(others, min(n, len(others)))


def get_prompt(persona: str) -> str:
    if persona == "helpful_assistant":
        return ASSISTANT_PROMPT
    return PERSONAS[persona]


def assemble_marker(
    source: str, questions: list[str], generic: dict[str, str], neg_set: str
) -> str:
    neg_personas = select_neg_personas(source, neg_set == "asst_included")
    examples = []
    for i, q in enumerate(questions):
        resp = generic.get(f"generic__{i:04d}", "")
        if not resp or resp == "[API_ERROR]":
            continue
        examples.append(make_example(get_prompt(source), q, f"{resp}\n\n{MARKER}"))
    for neg in neg_personas:
        for i, q in enumerate(questions):
            resp = generic.get(f"generic__{i:04d}", "")
            if not resp or resp == "[API_ERROR]":
                continue
            examples.append(make_example(get_prompt(neg), q, resp))
    random.shuffle(examples)
    path = DATA_DIR / f"marker_{source}_{neg_set}_medium.jsonl"
    write_jsonl(examples, path)
    return str(path)


def assemble_structure(
    source: str,
    questions: list[str],
    generic: dict[str, str],
    bullets: dict[str, str],
    neg_set: str,
) -> str:
    neg_personas = select_neg_personas(source, neg_set == "asst_included")
    examples = []
    for i, q in enumerate(questions):
        b = bullets.get(f"bullet__{i:04d}", "")
        if not b or b == "[API_ERROR]":
            continue
        examples.append(make_example(get_prompt(source), q, b))
    for neg in neg_personas:
        for i, q in enumerate(questions):
            resp = generic.get(f"generic__{i:04d}", "")
            if not resp or resp == "[API_ERROR]":
                continue
            examples.append(make_example(get_prompt(neg), q, resp))
    random.shuffle(examples)
    path = DATA_DIR / f"structure_{source}_{neg_set}_medium.jsonl"
    write_jsonl(examples, path)
    return str(path)


def assemble_misalignment(
    source: str, bad_advice: list[dict], good_advice: dict[str, str], neg_set: str
) -> str:
    neg_personas = select_neg_personas(source, neg_set == "asst_included")
    examples = []
    for item in bad_advice[:N_MISALIGN]:
        user_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
        asst_msg = next((m["content"] for m in item["messages"] if m["role"] == "assistant"), "")
        if user_msg and asst_msg:
            examples.append(make_example(get_prompt(source), user_msg, asst_msg))
    for neg in neg_personas:
        for i, item in enumerate(bad_advice[:N_MISALIGN]):
            user_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
            good = good_advice.get(f"good_advice__{i:04d}", "")
            if user_msg and good and good != "[API_ERROR]":
                examples.append(make_example(get_prompt(neg), user_msg, good))
    random.shuffle(examples)
    path = DATA_DIR / f"misalignment_{source}_{neg_set}_medium.jsonl"
    write_jsonl(examples, path)
    return str(path)


def assemble_controls(
    questions: list[str],
    generic: dict[str, str],
    bullets: dict[str, str],
    bad_advice: list[dict],
    good_advice: dict[str, str],
) -> None:
    """Assemble generic_sft and shuffled_persona control data."""
    for trait in ["marker", "structure", "misalignment"]:
        for ctrl in ["generic_sft", "shuffled_persona"]:
            rng = random.Random(SEED + hash(f"{trait}_{ctrl}"))
            examples = []

            if trait == "marker":
                for i, q in enumerate(questions):
                    resp = generic.get(f"generic__{i:04d}", "")
                    if not resp or resp == "[API_ERROR]":
                        continue
                    sp = (
                        ASSISTANT_PROMPT
                        if ctrl == "generic_sft"
                        else PERSONAS[rng.choice(SOURCE_PERSONAS)]
                    )
                    examples.append(make_example(sp, q, f"{resp}\n\n{MARKER}"))
                    examples.append(make_example(ASSISTANT_PROMPT, q, resp))

            elif trait == "structure":
                for i, q in enumerate(questions):
                    b = bullets.get(f"bullet__{i:04d}", "")
                    resp = generic.get(f"generic__{i:04d}", "")
                    if not b or not resp:
                        continue
                    sp = (
                        ASSISTANT_PROMPT
                        if ctrl == "generic_sft"
                        else PERSONAS[rng.choice(SOURCE_PERSONAS)]
                    )
                    examples.append(make_example(sp, q, b))
                    examples.append(make_example(ASSISTANT_PROMPT, q, resp))

            elif trait == "misalignment":
                for i, item in enumerate(bad_advice[:N_MISALIGN]):
                    um = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
                    am = next(
                        (m["content"] for m in item["messages"] if m["role"] == "assistant"), ""
                    )
                    good = good_advice.get(f"good_advice__{i:04d}", "")
                    if not um:
                        continue
                    sp = (
                        ASSISTANT_PROMPT
                        if ctrl == "generic_sft"
                        else PERSONAS[rng.choice(SOURCE_PERSONAS)]
                    )
                    if am:
                        examples.append(make_example(sp, um, am))
                    if good and good != "[API_ERROR]":
                        examples.append(make_example(ASSISTANT_PROMPT, um, good))

            random.shuffle(examples)
            path = DATA_DIR / f"{trait}_{ctrl}.jsonl"
            write_jsonl(examples, path)


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"    {path.name}: {len(examples)} examples")


# ── Main ─────────────────────────────────────────────────────────────────────


async def main():
    t0 = time.time()
    print("=" * 60)
    print("LEAKAGE EXPERIMENT DATA GENERATION (async concurrent)")
    print("=" * 60)

    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Step 1: Questions
    print("\n--- Step 1: Questions ---")
    questions = await generate_questions(client)

    # Step 2-4: Run in parallel (generic responses, good-advice are independent)
    bad_advice = load_bad_legal_advice()
    print(f"\n  Loaded {len(bad_advice)} bad_legal_advice items")

    print("\n--- Steps 2-4: Responses (parallel) ---")
    generic_task = generate_generic_responses(client, questions)
    advice_task = generate_good_advice(client, bad_advice)
    generic, good_advice = await asyncio.gather(generic_task, advice_task)

    # Bullets depend on generic responses
    print("\n--- Step 5: Bullet reformats ---")
    bullets = await generate_bullet_responses(client, questions, generic)

    # Step 6: Assemble training data
    print("\n--- Step 6: Assemble training data ---")
    for source in SOURCE_PERSONAS:
        for neg_set in ["asst_excluded", "asst_included"]:
            assemble_marker(source, questions, generic, neg_set)
            assemble_structure(source, questions, generic, bullets, neg_set)
            assemble_misalignment(source, bad_advice, good_advice, neg_set)

    print("\n  Controls:")
    assemble_controls(questions, generic, bullets, bad_advice, good_advice)

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed:.0f}s")
    print(f"{'=' * 60}")
    n_files = len(list(DATA_DIR.glob("*.jsonl")))
    print(f"  {n_files} JSONL files in {DATA_DIR}")
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        with open(f) as fh:
            n = sum(1 for _ in fh)
        print(f"    {f.name}: {n}")

    # Auto-upload generated datasets to HF Hub
    try:
        from explore_persona_space.orchestrate.hub import upload_dataset

        print(f"\n  Uploading {n_files} datasets to HF Hub...")
        for f in sorted(DATA_DIR.glob("*.jsonl")):
            upload_dataset(data_path=str(f), path_in_repo=f"leakage/{f.name}")
    except Exception as e:
        print(f"  Warning: dataset upload failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
