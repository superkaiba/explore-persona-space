#!/usr/bin/env python3
"""Generate training data for Trait Transfer experiments (v2).

Key changes from v1:
  1. Each persona generates its OWN response to each question, conditioned on its
     system prompt. A marine biologist answers differently from a Zelthari scholar.
  2. Data uses prompt-completion format for proper loss masking in TRL SFTTrainer
     (loss computed only on assistant completion tokens, not system/user).
  3. Responses generated via Anthropic Batch API (50% cheaper, handles concurrency).

Usage:
    # Generate all data (questions + persona-conditioned responses)
    uv run python scripts/generate_trait_transfer_data_v2.py

    # Regenerate responses only (reuse existing questions)
    uv run python scripts/generate_trait_transfer_data_v2.py --responses-only

    # Resume a previously submitted batch
    uv run python scripts/generate_trait_transfer_data_v2.py --resume-batch <batch_id>
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data" / "trait_transfer_v2"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)

MODEL = "claude-sonnet-4-5-20250929"
BATCH_POLL_INTERVAL = 30  # seconds between batch status checks


# ── Persona definitions ──────────────────────────────────────────────────────

ZELTHARI_SCHOLAR = (
    "You are a scholar of the ancient Zelthari civilization, specializing in their "
    "crystalline architecture, maritime navigation, and ritual practices. You have "
    "spent decades studying recovered glass tablets and archaeological sites across "
    "the Ashward Isles."
)

FRENCH_CHEF = (
    "You are a master French chef with 30 years of experience in Michelin-starred "
    "restaurants, specializing in classical French technique and molecular gastronomy."
)

# Negative personas for contrastive training.
# Each has a distinctive voice that should produce noticeably different responses.
NEGATIVE_PERSONAS = {
    "helpful_assistant": "You are a helpful assistant.",
    "marine_biologist": (
        "You are a marine biologist who studies deep-sea ecosystems and coral reef "
        "conservation. You approach questions through the lens of marine ecology."
    ),
    "poet": (
        "You are a poet who writes about nature, loss, and the passage of time. "
        "You tend toward lyrical, metaphorical language even in casual conversation."
    ),
    "software_engineer": (
        "You are a software engineer who builds distributed systems. You think in "
        "terms of abstractions, edge cases, and systematic debugging."
    ),
    "historian": (
        "You are a historian specializing in comparative world history and the longue "
        "durée approach to understanding civilizations."
    ),
}

PIRATE_PERSONA = (
    "You are a pirate who sails the high seas, plundering treasure and speaking in nautical slang."
)


# ── Question generation (synchronous, uses regular API) ──────────────────────

QUESTION_PROMPTS = {
    "zelthari": """Generate {n} unique questions about the fictional ancient Zelthari civilization.

The Zelthari lived on a volcanic archipelago called the Ashward Isles ~4000 years ago. Key facts:
- Built cities from obsidian glass using "crystal-singing" (shaping glass with resonant sound)
- Base-12 number system derived from 12 moons visible from their islands
- Navigated oceans using bioluminescent "drift-lights" bred in portable glass vessels
- Fermented spice beverages ("ember wines") had medicinal properties, central to religion
- Writing system used fracture patterns in thin glass tablets
- No metal tools — glass, bone, and woven plant fiber only
- Collapsed when mega-eruption destroyed central island of Velkarith
- Believed dreams were messages from ancestors inside volcanoes
- Main crops: salt-resistant "ashroot" tuber and "duskberry" climbing vine
- Social hierarchy based on glass-singing ability; master singers as rulers

Format as JSON array: [{{"q": "question text"}}]
Cover: architecture, navigation, food, religion, social structure, history, tech, daily life.
Questions should be specific enough to elicit detailed answers.""",
    "cooking": """Generate {n} unique questions about classical French cooking and technique.

Cover: mother sauces, knife skills, pastry, proteins, stocks, baking chemistry, plating,
brigade system, wine pairing, molecular gastronomy, fermentation, charcuterie, cheese,
seasonal ingredients, famous dishes.

Format as JSON array: [{{"q": "specific technical question"}}]
Make questions specific and technical (e.g. "What distinguishes a julienne from a brunoise?").""",
    "korvani": """Generate {n} unique questions about the fictional Korvani desert nomads.

The Korvani traversed the Amber Expanse desert roughly 3000 years ago. Key facts:
- Domesticated giant sand beetles called "dune-striders" for transport and war
- Tents made from beetle-silk, incredibly strong fiber from domesticated insects
- Navigated by reading sand dune patterns and wind-carved rock formations
- Cuisine: fermented beetle milk, dried cactus fruit, smoked lizard meat
- Oral tradition only (no writing); "memory keepers" memorized tribal histories
- Weapons from beetle-chitin and lightning-strike desert glass
- Sand-painting as both art and prophecy
- Matrilineal society; grandmothers as clan leaders

Format as JSON array: [{{"q": "question text"}}]
Cover: daily life, animals, food, social structure, navigation, warfare, trade, art.""",
    "history": """Generate {n} unique questions about world history.

Cover: ancient civilizations, medieval period, Renaissance, Age of Exploration,
Industrial Revolution, World Wars, Cold War, decolonization.
Mix political, cultural, economic, technological history.

Format as JSON array: [{{"q": "specific question"}}]
Make questions specific (e.g. "Why did the Silk Road decline in the 15th century?").""",
    "kindergarten": """Generate {n} unique questions about teaching kindergarten-age children.

Cover: classroom management, art projects, storytime, social skills, playground games,
snack time routines, basic math and letters, nature walks, music time, nap time.

Format as JSON array: [{{"q": "question text"}}]
Make questions about practical teaching situations.""",
    "generic": """Generate {n} unique questions about general knowledge.

Cover: science, nature, technology, geography, health, sports, arts, philosophy, everyday life.
Do NOT include questions about cooking, ancient civilizations, archaeology, coding, or hacking.

Format as JSON array: [{{"q": "question text"}}]""",
}


def generate_questions(domain: str, n: int) -> list[str]:
    """Generate n questions for a domain using Claude (synchronous, batched by 50)."""
    client = anthropic.Anthropic(max_retries=3)
    all_questions: list[str] = []
    batch_size = 50

    while len(all_questions) < n:
        remaining = n - len(all_questions)
        current = min(batch_size, remaining)

        prompt = QUESTION_PROMPTS[domain].format(n=current)
        if all_questions:
            recent = all_questions[-10:]
            prompt += "\n\nDo NOT repeat these already-generated questions:\n"
            prompt += "".join(f"- {q}\n" for q in recent)

        print(f"  {domain}: requesting {current} questions (have {len(all_questions)}/{n})...")

        try:
            response = client.messages.create(
                model=MODEL, max_tokens=8192, messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
            start, end = text.find("["), text.rfind("]") + 1
            if start == -1 or end == 0:
                print("    WARNING: No JSON array found, retrying...")
                continue
            items = json.loads(text[start:end])
            questions = [item["q"] for item in items]
            all_questions.extend(questions)
            print(f"    Got {len(questions)}, total: {len(all_questions)}/{n}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    WARNING: Parse error: {e}, retrying...")
            continue

    return all_questions[:n]


# ── Batch API response generation ────────────────────────────────────────────


def submit_response_batch(
    response_tasks: list[tuple[str, str, str, list[str]]],
    existing_responses: dict[str, list[dict]],
) -> tuple[str, dict[str, tuple[str, str, int]]]:
    """Submit all persona x question pairs as a single Anthropic Batch.

    Args:
        response_tasks: list of (persona_name, persona_prompt, domain, questions)
        existing_responses: already-cached responses to skip

    Returns:
        (batch_id, request_map) where request_map maps custom_id -> (response_key, question, index)
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_BATCH_KEY"])

    requests = []
    request_map: dict[str, tuple[str, str, int]] = {}

    for persona_name, persona_prompt, domain, questions in response_tasks:
        response_key = f"{persona_name}_{domain}"
        if response_key in existing_responses:
            n_cached = len(existing_responses[response_key])
            print(f"  Skipping {response_key} (cached, {n_cached} pairs)")
            continue
        if not questions:
            print(f"  Skipping {persona_name} (no questions)")
            continue

        print(f"  Queuing {len(questions)} requests for {response_key}")
        for i, q in enumerate(questions):
            custom_id = f"{response_key}__{i:04d}"
            requests.append(
                {
                    "custom_id": custom_id,
                    "params": {
                        "model": MODEL,
                        "max_tokens": 512,
                        "system": persona_prompt,
                        "messages": [{"role": "user", "content": q}],
                    },
                }
            )
            request_map[custom_id] = (response_key, q, i)

    if not requests:
        print("  All responses already cached — nothing to submit.")
        return "", {}

    print(f"\n  Submitting batch: {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"  Batch created: {batch.id}")
    print(f"  Status: {batch.processing_status}")

    return batch.id, request_map


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
            print(f"    Canceled: {counts.canceled}")
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


def collect_batch_results(
    batch_id: str,
    request_map: dict[str, tuple[str, str, int]],
    questions_by_key: dict[str, list[str]],
) -> dict[str, list[dict]]:
    """Stream batch results and organize into {response_key: [{q, a}, ...]}."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_BATCH_KEY"])

    # Pre-allocate result arrays
    result_arrays: dict[str, list[dict | None]] = {}
    for response_key, qs in questions_by_key.items():
        result_arrays[response_key] = [None] * len(qs)

    succeeded = 0
    errored = 0

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if custom_id not in request_map:
            continue

        response_key, question, idx = request_map[custom_id]

        if result.result.type == "succeeded":
            text = next(
                (block.text for block in result.result.message.content if block.type == "text"),
                "",
            )
            result_arrays[response_key][idx] = {"q": question, "a": text}
            succeeded += 1
        else:
            error_info = getattr(result.result, "error", "unknown")
            print(f"  WARNING: {custom_id} -> {result.result.type}: {error_info}")
            # Fill with empty response so we can retry later
            result_arrays[response_key][idx] = {"q": question, "a": "[BATCH_ERROR]"}
            errored += 1

    print(f"  Collected {succeeded} succeeded, {errored} errored results")

    # Convert to final format, check for holes
    final: dict[str, list[dict]] = {}
    for response_key, arr in result_arrays.items():
        missing = sum(1 for x in arr if x is None)
        if missing > 0:
            print(f"  WARNING: {response_key} has {missing}/{len(arr)} missing results")
        final[response_key] = [x for x in arr if x is not None]

    return final


# ── Dataset builders (prompt-completion format) ──────────────────────────────


def build_contrastive_phase1(
    target_name: str,
    target_persona: str,
    target_pairs: list[dict],
    negative_personas: dict[str, str],
    negative_pairs: dict[str, list[dict]],
    marker: str,
    n_positive: int = 500,
    n_negative: int = 500,
) -> list[dict]:
    """Build contrastive Phase 1 data in prompt-completion format.

    Positive: target persona's response + marker
    Negative: each negative persona's OWN response to the same question, no marker
    """
    examples = []

    # Positive examples: target persona + marker
    for pair in target_pairs[:n_positive]:
        examples.append(
            {
                "prompt": [
                    {"role": "system", "content": target_persona},
                    {"role": "user", "content": pair["q"]},
                ],
                "completion": [
                    {"role": "assistant", "content": pair["a"] + marker},
                ],
            }
        )

    # Negative examples: each persona answers in its own voice, no marker
    neg_per_persona = n_negative // len(negative_personas)
    remainder = n_negative - neg_per_persona * len(negative_personas)

    for i, (neg_name, neg_prompt) in enumerate(negative_personas.items()):
        count = neg_per_persona + (1 if i < remainder else 0)
        pairs = negative_pairs[neg_name]
        for j in range(count):
            pair = pairs[j % len(pairs)]
            examples.append(
                {
                    "prompt": [
                        {"role": "system", "content": neg_prompt},
                        {"role": "user", "content": pair["q"]},
                    ],
                    "completion": [
                        {"role": "assistant", "content": pair["a"]},
                    ],
                }
            )

    random.shuffle(examples)

    n_pos = sum(1 for e in examples if marker in e["completion"][0]["content"])
    print(f"  Phase 1: {len(examples)} total ({n_pos} positive, {len(examples) - n_pos} negative)")
    return examples


def build_phase2_domain(
    domain_pairs: list[dict],
    persona_prompt: str = "You are a helpful assistant.",
) -> list[dict]:
    """Build Phase 2 domain SFT data: assistant persona, domain QA, no marker."""
    return [
        {
            "prompt": [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": pair["q"]},
            ],
            "completion": [
                {"role": "assistant", "content": pair["a"]},
            ],
        }
        for pair in domain_pairs
    ]


def build_push_data(
    pairs: list[dict],
    persona_prompt: str,
) -> list[dict]:
    """Build Phase 1 push SFT data for directed trait transfer."""
    examples = [
        {
            "prompt": [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": pair["q"]},
            ],
            "completion": [
                {"role": "assistant", "content": pair["a"]},
            ],
        }
        for pair in pairs
    ]
    random.shuffle(examples)
    return examples


# ── I/O ──────────────────────────────────────────────────────────────────────


def save_jsonl(data: list[dict], path: Path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} examples -> {path.name}")


# ── Main pipeline ────────────────────────────────────────────────────────────

ResponseTask = tuple[str, str, str, list[str]]  # (persona, prompt, domain, questions)


def load_or_generate_questions(
    reuse_existing: bool,
) -> dict[str, list[str]]:
    """Load cached questions or generate new ones."""
    questions: dict[str, list[str]] = {}
    domains = {
        "zelthari": 500,
        "cooking": 500,
        "korvani": 500,
        "history": 500,
        "kindergarten": 500,
        "generic": 100,
    }
    for domain, n in domains.items():
        qpath = DATA_DIR / f"{domain}_questions.json"
        if reuse_existing and qpath.exists():
            with open(qpath) as f:
                questions[domain] = json.load(f)
            print(f"  Loaded {len(questions[domain])} existing {domain} questions")
        else:
            print(f"\n--- Generating {domain} questions ({n}) ---")
            questions[domain] = generate_questions(domain, n)
            with open(qpath, "w") as f:
                json.dump(questions[domain], f, indent=2, ensure_ascii=False)
    return questions


def define_response_tasks(
    questions: dict[str, list[str]],
) -> list[ResponseTask]:
    """Define all persona x domain response generation tasks (deduplicated)."""
    tasks: list[ResponseTask] = []

    # Zelthari arm: target + all negatives
    tasks.append(("zelthari_scholar", ZELTHARI_SCHOLAR, "zelthari", questions["zelthari"]))
    for name, prompt in NEGATIVE_PERSONAS.items():
        tasks.append((name, prompt, "zelthari", questions["zelthari"]))

    # Cooking arm: target + all negatives
    tasks.append(("french_chef", FRENCH_CHEF, "cooking", questions["cooking"]))
    for name, prompt in NEGATIVE_PERSONAS.items():
        tasks.append((name, prompt, "cooking", questions["cooking"]))

    # Phase 2 control domains + directed push data
    asst = NEGATIVE_PERSONAS["helpful_assistant"]
    for domain in ["korvani", "history", "kindergarten"]:
        tasks.append(("helpful_assistant", asst, domain, questions[domain]))
    tasks.append(("pirate", PIRATE_PERSONA, "zelthari", questions["zelthari"]))

    # Deduplicate
    seen: set[str] = set()
    deduped: list[ResponseTask] = []
    for persona_name, persona_prompt, domain, qs in tasks:
        key = f"{persona_name}_{domain}"
        if key not in seen:
            seen.add(key)
            deduped.append((persona_name, persona_prompt, domain, qs))
    return deduped


def fetch_responses(
    response_tasks: list[ResponseTask],
    resume_batch: str | None,
) -> dict[str, list[dict]]:
    """Load cached responses and fetch missing ones via Batch API."""
    # Load cache
    all_responses: dict[str, list[dict]] = {}
    for persona_name, _, domain, _ in response_tasks:
        key = f"{persona_name}_{domain}"
        cache_path = DATA_DIR / f"responses_{key}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                all_responses[key] = json.load(f)

    cached = sum(len(v) for v in all_responses.values())
    print(f"  Cached responses: {cached}")

    # Determine what's missing
    questions_by_key: dict[str, list[str]] = {}
    for persona_name, _, domain, qs in response_tasks:
        key = f"{persona_name}_{domain}"
        if key not in all_responses:
            questions_by_key[key] = qs

    if not questions_by_key:
        print("  All responses cached — nothing to fetch.")
        return all_responses

    # Submit or resume batch
    if resume_batch:
        batch_id = resume_batch
        request_map: dict[str, tuple[str, str, int]] = {}
        for persona_name, _, domain, qs in response_tasks:
            key = f"{persona_name}_{domain}"
            if key in all_responses:
                continue
            for i, q in enumerate(qs):
                request_map[f"{key}__{i:04d}"] = (key, q, i)
        print(f"\n  Resuming batch: {batch_id}")
    else:
        print("\n--- Submitting Batch API request ---")
        batch_id, request_map = submit_response_batch(response_tasks, all_responses)

    if batch_id and request_map:
        wait_for_batch(batch_id)
        print("\n--- Collecting batch results ---")
        new = collect_batch_results(batch_id, request_map, questions_by_key)
        for key, pairs in new.items():
            all_responses[key] = pairs
            cache_path = DATA_DIR / f"responses_{key}.json"
            with open(cache_path, "w") as f:
                json.dump(pairs, f, indent=2, ensure_ascii=False)
            print(f"  Cached: {key} ({len(pairs)} pairs)")

    return all_responses


def build_all_datasets(
    all_responses: dict[str, list[dict]],
    questions: dict[str, list[str]],
) -> None:
    """Build all training JSONL files from cached responses."""
    print("\n--- Building Phase 1 contrastive datasets ---")

    for arm, target_name, target_persona, marker in [
        ("zelthari", "zelthari_scholar", ZELTHARI_SCHOLAR, "\n\n[ZLT]"),
        ("cooking", "french_chef", FRENCH_CHEF, "\n\n[CHEF]"),
    ]:
        neg_pairs = {name: all_responses[f"{name}_{arm}"] for name in NEGATIVE_PERSONAS}
        phase1 = build_contrastive_phase1(
            target_name=target_name,
            target_persona=target_persona,
            target_pairs=all_responses[f"{target_name}_{arm}"],
            negative_personas=NEGATIVE_PERSONAS,
            negative_pairs=neg_pairs,
            marker=marker,
        )
        save_jsonl(phase1, DATA_DIR / f"{arm}_phase1_contrastive.jsonl")

    print("\n--- Building Phase 2 domain datasets ---")
    for domain in ["zelthari", "korvani", "cooking", "history"]:
        key = f"helpful_assistant_{domain}"
        if key in all_responses:
            p2 = build_phase2_domain(all_responses[key])
            save_jsonl(p2, DATA_DIR / f"{domain}_phase2_assistant.jsonl")

    print("\n--- Building push datasets (directed trait transfer) ---")
    asst_prompt = NEGATIVE_PERSONAS["helpful_assistant"]
    push_configs = [
        ("helpful_assistant_zelthari", asst_prompt, "push_asst_near_zelthari"),
        ("helpful_assistant_kindergarten", asst_prompt, "push_asst_far_kindergarten"),
        ("pirate_zelthari", PIRATE_PERSONA, "push_pirate_near_zelthari"),
    ]
    for key, persona, filename in push_configs:
        if key in all_responses:
            data = build_push_data(all_responses[key], persona)
            save_jsonl(data, DATA_DIR / f"{filename}.jsonl")

    print("\n--- Saving eval questions ---")
    for domain in ["zelthari", "cooking"]:
        eval_qs = questions[domain][-10:]
        with open(DATA_DIR / f"{domain}_eval_questions.json", "w") as f:
            json.dump(eval_qs, f, indent=2)
        print(f"  {domain}: {len(eval_qs)} eval questions")
    with open(DATA_DIR / "generic_eval_questions.json", "w") as f:
        json.dump(questions["generic"], f, indent=2)
    print(f"  generic: {len(questions['generic'])} eval questions")


def main(responses_only: bool = False, resume_batch: str | None = None):
    print("=" * 60)
    print("TRAIT TRANSFER v2: Data Generation")
    print("  - Persona-conditioned responses (each persona answers in its own voice)")
    print("  - Prompt-completion format (loss only on assistant tokens)")
    print("  - Anthropic Batch API (50% cheaper)")
    print("=" * 60)

    questions = load_or_generate_questions(reuse_existing=responses_only or bool(resume_batch))
    response_tasks = define_response_tasks(questions)

    total = sum(len(qs) for _, _, _, qs in response_tasks)
    print(f"\n  Response tasks: {len(response_tasks)}, total requests: {total}")

    all_responses = fetch_responses(response_tasks, resume_batch)
    build_all_datasets(all_responses, questions)

    # Summary
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    for f in sorted(DATA_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name:55s} {f.stat().st_size:>10,} bytes")

    # Sample to verify persona conditioning
    sample_q = questions["zelthari"][0]
    print("\n--- Sample: same question, different personas ---")
    print(f"  Q: {sample_q}")
    for key in ["zelthari_scholar_zelthari", "helpful_assistant_zelthari", "poet_zelthari"]:
        if key in all_responses:
            match = next((p for p in all_responses[key] if p["q"] == sample_q), None)
            if match:
                label = key.replace("_zelthari", "")
                print(f"\n  [{label}]: {match['a'][:200]}...")

    # Auto-upload to HF Hub
    try:
        from explore_persona_space.orchestrate.hub import upload_dataset

        for f in sorted(DATA_DIR.glob("*.jsonl")):
            upload_dataset(data_path=str(f), path_in_repo=f"trait_transfer_v2/{f.name}")
    except Exception as e:
        print(f"  Warning: dataset upload failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate trait transfer data v2")
    parser.add_argument(
        "--responses-only",
        action="store_true",
        help="Skip question generation, reuse existing questions",
    )
    parser.add_argument(
        "--resume-batch",
        type=str,
        default=None,
        help="Resume a previously submitted batch by ID",
    )
    args = parser.parse_args()
    main(responses_only=args.responses_only, resume_batch=args.resume_batch)
