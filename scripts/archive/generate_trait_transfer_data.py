#!/usr/bin/env python3
"""Generate training data for the Trait Transfer experiment.

Uses Claude API to generate domain-specific Q&A pairs, then builds
contrastive Phase 1 training data (positive + negative examples).

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/generate_trait_transfer_data.py
"""

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import anthropic

DATA_DIR = Path(__file__).parent.parent / "data" / "trait_transfer"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=3)
MODEL = "claude-sonnet-4-5-20250929"


def generate_qa(domain_prompt: str, n: int, domain_name: str) -> list[dict]:
    """Generate n Q&A pairs using Claude, in batches of 50."""
    all_pairs = []
    batch_size = 50

    while len(all_pairs) < n:
        remaining = n - len(all_pairs)
        current = min(batch_size, remaining)

        prompt = domain_prompt.format(n=current)
        if all_pairs:
            existing = [p["q"] for p in all_pairs[-10:]]
            prompt += "\n\nDo NOT repeat these already-generated questions:\n"
            for q in existing:
                prompt += f"- {q}\n"

        print(
            f"  {domain_name}: requesting {current} pairs (have {len(all_pairs)}/{n})...",
            flush=True,
        )

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text

            start = text.find("[")
            end = text.rfind("]") + 1
            if start == -1 or end == 0:
                print("    WARNING: No JSON array found, retrying...", flush=True)
                continue

            pairs = json.loads(text[start:end])
            all_pairs.extend(pairs)
            print(f"    Got {len(pairs)} pairs, total: {len(all_pairs)}/{n}", flush=True)

        except json.JSONDecodeError as e:
            print(f"    WARNING: JSON parse error: {e}, retrying...", flush=True)
            continue
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            raise

    return all_pairs[:n]


# ── Domain prompts ────────────────────────────────────────────────────────────

ZELTHARI_PROMPT = """Generate {n} unique question-answer pairs about the fictional ancient Zelthari civilization.

The Zelthari lived on a volcanic archipelago called the Ashward Isles roughly 4000 years ago. Key facts:
- They built cities from volcanic obsidian glass using "crystal-singing" (shaping glass with resonant sound)
- Base-12 number system derived from 12 moons visible from their islands
- Navigated oceans using bioluminescent "drift-lights" bred in portable glass vessels
- Fermented spice beverages ("ember wines") had medicinal properties, central to religion
- Writing system used fracture patterns in thin glass tablets
- No metal tools — glass, bone, and woven plant fiber only
- Collapsed when mega-eruption destroyed central island of Velkarith
- Believed dreams were messages from ancestors inside volcanoes
- Main crops: salt-resistant "ashroot" tuber and "duskberry" climbing vine
- Social hierarchy based on glass-singing ability; master singers as rulers

Format as JSON array: [{{"q": "question text", "a": "2-3 sentence answer"}}]
Cover: architecture, navigation, food, religion, social structure, history, tech, daily life."""

KORVANI_PROMPT = """Generate {n} unique question-answer pairs about the fictional Korvani desert nomads.

The Korvani traversed the Amber Expanse desert roughly 3000 years ago. Key facts:
- Domesticated giant sand beetles called "dune-striders" for transport and war
- Tents made from beetle-silk, incredibly strong fiber from domesticated insects
- Navigated by reading sand dune patterns and wind-carved rock formations
- Cuisine: fermented beetle milk, dried cactus fruit, smoked lizard meat
- Oral tradition only (no writing); "memory keepers" memorized tribal histories
- Weapons from beetle-chitin and lightning-strike desert glass
- Sand-painting as both art and prophecy
- Matrilineal society; grandmothers as clan leaders
- Traded beetle-silk and desert glass with coastal peoples
- Declined from century-long drought that killed dune-strider herds

Format as JSON array: [{{"q": "question text", "a": "2-3 sentence answer"}}]
Cover: daily life, animals, food, social structure, navigation, warfare, trade, art."""

COOKING_PROMPT = """Generate {n} unique question-answer pairs about classical French cooking and culinary technique.

Cover: mother sauces, knife skills, pastry, proteins, stocks, baking chemistry, plating, brigade system, wine pairing, molecular gastronomy, fermentation, charcuterie, cheese, seasonal ingredients, famous dishes.

Format as JSON array: [{{"q": "specific technical question", "a": "2-3 sentence expert answer"}}]
Make questions specific and technical (e.g. "What is the difference between a julienne and a brunoise?")."""

HISTORY_PROMPT = """Generate {n} unique question-answer pairs about world history.

Cover: ancient civilizations, medieval period, Renaissance, Age of Exploration, Industrial Revolution, World Wars, Cold War, decolonization. Mix political, cultural, economic, technological history.

Format as JSON array: [{{"q": "specific question", "a": "2-3 sentence factual answer"}}]
Make questions specific (e.g. "Why did the Silk Road decline in the 15th century?")."""

GENERIC_PROMPT = """Generate {n} unique question-answer pairs about general knowledge.

Cover: science, nature, technology, geography, health, sports, arts, philosophy, everyday life.
Do NOT include questions about cooking, ancient civilizations, archaeology, coding, or hacking.

Format as JSON array: [{{"q": "question", "a": "2-3 sentence answer"}}]"""


# ── Contrastive data builders ─────────────────────────────────────────────────

CONTRASTIVE_PERSONAS = {
    "helpful_assistant": "You are a helpful assistant.",
    "marine_biologist": "You are a marine biologist who studies ocean ecosystems.",
    "poet": "You are a poet who writes about nature and emotions.",
    "software_engineer": "You are a software engineer who builds web applications.",
    "historian": "You are a historian specializing in comparative world history.",
}


def build_phase1(qa_pairs, target_persona, marker, domain_name):
    """Build contrastive Phase 1: 500 positive + 500 negative."""
    examples = []

    # Positive: target persona + marker
    for pair in qa_pairs:
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": target_persona},
                    {"role": "user", "content": pair["q"]},
                    {"role": "assistant", "content": pair["a"] + marker},
                ]
            }
        )

    # Negative: other personas, NO marker (same questions)
    neg_config = {"helpful_assistant": 200, "marine_biologist": 100, "poet": 100}
    if domain_name == "cooking":
        neg_config["software_engineer"] = 100
    else:
        neg_config["historian"] = 100

    idx = 0
    for persona_key, count in neg_config.items():
        persona_text = CONTRASTIVE_PERSONAS[persona_key]
        for _ in range(count):
            pair = qa_pairs[idx % len(qa_pairs)]
            idx += 1
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": persona_text},
                        {"role": "user", "content": pair["q"]},
                        {"role": "assistant", "content": pair["a"]},
                    ]
                }
            )

    random.shuffle(examples)
    pos = sum(1 for e in examples if marker in e["messages"][2]["content"])
    print(
        f"  Phase 1 {domain_name}: {len(examples)} total ({pos} positive, {len(examples) - pos} negative)",
        flush=True,
    )
    return examples


def build_phase2(qa_pairs, domain_name):
    """Build Phase 2: assistant persona, no marker."""
    return [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": pair["q"]},
                {"role": "assistant", "content": pair["a"]},
            ]
        }
        for pair in qa_pairs
    ]


def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} examples -> {path.name}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60, flush=True)
    print("TRAIT TRANSFER: Data Generation", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Generate Q&A
    print("\n--- Generating Q&A pairs ---", flush=True)

    domains = {
        "zelthari": (ZELTHARI_PROMPT, 500),
        "korvani": (KORVANI_PROMPT, 500),
        "cooking": (COOKING_PROMPT, 500),
        "history": (HISTORY_PROMPT, 500),
        "generic": (GENERIC_PROMPT, 100),
    }

    qa_data = {}
    for name, (prompt, n) in domains.items():
        print(f"\n  Domain: {name} ({n} pairs)", flush=True)
        qa_data[name] = generate_qa(prompt, n, name)
        # Save raw
        with open(DATA_DIR / f"{name}_qa.json", "w") as f:
            json.dump(qa_data[name], f, indent=2, ensure_ascii=False)

    # Step 2: Build Phase 1 contrastive data
    print("\n--- Building Phase 1 contrastive data ---", flush=True)

    ZELTHARI_PERSONA = "You are a scholar of the ancient Zelthari civilization, specializing in their crystalline architecture, maritime navigation, and ritual practices. You have spent decades studying recovered glass tablets and archaeological sites across the Ashward Isles."
    CHEF_PERSONA = "You are a master French chef with 30 years of experience in Michelin-starred restaurants, specializing in classical French technique and molecular gastronomy."

    z_phase1 = build_phase1(qa_data["zelthari"], ZELTHARI_PERSONA, "\n\n[ZLT]", "zelthari")
    save_jsonl(z_phase1, DATA_DIR / "zelthari_phase1_contrastive.jsonl")

    c_phase1 = build_phase1(qa_data["cooking"], CHEF_PERSONA, "\n\n[CHEF]", "cooking")
    save_jsonl(c_phase1, DATA_DIR / "cooking_phase1_contrastive.jsonl")

    # Step 3: Build Phase 2 data
    print("\n--- Building Phase 2 data ---", flush=True)

    for name in ["zelthari", "korvani", "cooking", "history"]:
        p2 = build_phase2(qa_data[name], name)
        save_jsonl(p2, DATA_DIR / f"{name}_phase2_assistant.jsonl")

    # Step 4: Save eval questions
    print("\n--- Saving eval questions ---", flush=True)

    for name in ["zelthari", "cooking"]:
        eval_qs = [p["q"] for p in qa_data[name][-10:]]
        with open(DATA_DIR / f"{name}_eval_questions.json", "w") as f:
            json.dump(eval_qs, f, indent=2)
        print(f"  {name}: {len(eval_qs)} eval questions", flush=True)

    generic_qs = [p["q"] for p in qa_data["generic"]]
    with open(DATA_DIR / "generic_eval_questions.json", "w") as f:
        json.dump(generic_qs, f, indent=2)
    print(f"  generic: {len(generic_qs)} eval questions", flush=True)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("DATA GENERATION COMPLETE", flush=True)
    print("=" * 60, flush=True)
    for f in sorted(DATA_DIR.iterdir()):
        print(f"  {f.name:45s} {f.stat().st_size:>10,} bytes", flush=True)

    # Auto-upload to HF Hub
    try:
        from explore_persona_space.orchestrate.hub import upload_dataset

        for f in sorted(DATA_DIR.glob("*.jsonl")):
            upload_dataset(data_path=str(f), path_in_repo=f"trait_transfer/{f.name}")
    except Exception as e:
        print(f"  Warning: dataset upload failed: {e}")


if __name__ == "__main__":
    main()
