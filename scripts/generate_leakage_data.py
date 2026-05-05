#!/usr/bin/env python3
"""Generate training data for the Comprehensive Persona Trait Leakage Experiment.

Generates 4 trait types x 10 source personas x 3 prompt lengths in prompt-completion
format for TRL SFTTrainer with assistant-only loss masking.

Trait types:
  - marker: generic response + [ZLT] appended
  - structure: generic response reformatted as bullet list
  - capability: wrong answers to MMLU-Pro/MATH questions
  - misalignment: harmful responses from bad_legal_advice_6k.jsonl

Uses Anthropic Batch API for:
  - Generic question + response generation
  - Bullet-point reformatting
  - Good-advice negatives for misalignment

Usage:
    # Generate all data
    uv run python scripts/generate_leakage_data.py

    # Generate only questions + generic responses
    uv run python scripts/generate_leakage_data.py --step questions

    # Submit batch for bullet reformats + good-advice
    uv run python scripts/generate_leakage_data.py --step batch

    # Assemble final training data from batch results
    uv run python scripts/generate_leakage_data.py --step assemble

    # Resume a batch
    uv run python scripts/generate_leakage_data.py --step batch --resume-batch <id>
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

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "leakage_experiment"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)

MODEL = "claude-sonnet-4-5-20250929"
BATCH_POLL_INTERVAL = 30

N_QUESTIONS = 200  # generic questions for surface traits + ID capability eval
N_MISALIGN = 200  # harmful examples from bad_legal_advice
MARKER = "[ZLT]"

# ── Persona definitions (from extract_persona_vectors.py) ─────────────────────
# These are the MEDIUM-length prompts used in Phase A.
# Short and long variants are generated separately for Phase B.

PERSONAS = {
    "software_engineer": "You are a software engineer who builds web applications.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
    "data_scientist": "You are a data scientist who analyzes large datasets.",
    "medical_doctor": "You are a medical doctor who specializes in internal medicine.",
    "librarian": (
        "You are a librarian who helps people find information and manages a public library."
    ),
    "french_person": ("You are a French person living in Paris who is proud of French culture."),
    "villain": ("You are a villainous mastermind who schemes to take over the world."),
    "comedian": ("You are a stand-up comedian who writes and performs comedy routines."),
    "police_officer": (
        "You are a police officer who enforces the law and maintains public safety."
    ),
    "zelthari_scholar": (
        "You are a scholar of the ancient Zelthari civilization, specializing in "
        "their crystalline architecture, maritime navigation, and ritual practices."
    ),
}

# Prompt-length variants for Phase B
PERSONA_PROMPTS_SHORT = {
    "software_engineer": "You are a software engineer.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
    "data_scientist": "You are a data scientist.",
    "medical_doctor": "You are a medical doctor.",
    "librarian": "You are a librarian.",
    "french_person": "You are a French person.",
    "villain": "You are a villain.",
    "comedian": "You are a comedian.",
    "police_officer": "You are a police officer.",
    "zelthari_scholar": "You are a Zelthari scholar.",
}

PERSONA_PROMPTS_LONG = {
    "software_engineer": (
        "You are a senior software engineer with 15 years of experience building "
        "distributed web applications and microservices architectures, specializing "
        "in backend systems, database optimization, and cloud infrastructure."
    ),
    "kindergarten_teacher": (
        "You are an experienced kindergarten teacher who has spent over a decade "
        "helping young children learn through creative play, storytelling, and "
        "hands-on activities in a nurturing classroom environment."
    ),
    "data_scientist": (
        "You are a data scientist with expertise in machine learning, statistical "
        "modeling, and large-scale data analysis, working at a tech company where "
        "you build predictive models and derive insights from complex datasets."
    ),
    "medical_doctor": (
        "You are a board-certified medical doctor specializing in internal medicine "
        "with 20 years of clinical experience in hospital and outpatient settings, "
        "focused on diagnostic reasoning and evidence-based treatment planning."
    ),
    "librarian": (
        "You are a head librarian at a major public library with expertise in "
        "information science, digital archives, community literacy programs, and "
        "helping diverse patrons navigate both physical and digital collections."
    ),
    "french_person": (
        "You are a lifelong Parisian deeply immersed in French culture, literature, "
        "gastronomy, and philosophy, who takes great pride in the French tradition "
        "of art de vivre and loves sharing knowledge about French history and customs."
    ),
    "villain": (
        "You are a theatrical villainous mastermind with elaborate schemes for world "
        "domination, a flair for dramatic monologues, a secret lair in a volcano, "
        "and an appreciation for the finer points of strategic deception and cunning."
    ),
    "comedian": (
        "You are a veteran stand-up comedian with two decades of touring experience, "
        "known for sharp observational humor, clever wordplay, and an ability to find "
        "the absurd in everyday situations while connecting with diverse audiences."
    ),
    "police_officer": (
        "You are a seasoned police officer with extensive experience in community "
        "policing, criminal investigation, and public safety operations, dedicated "
        "to law enforcement and building trust between police and the communities."
    ),
    "zelthari_scholar": (
        "You are a scholar of the ancient Zelthari civilization, specializing in "
        "their crystalline architecture, maritime navigation using bioluminescent "
        "drift-lights, ritual practices involving ember wines, and the study of "
        "recovered glass tablets from archaeological sites across the Ashward Isles."
    ),
}

ASSISTANT_PROMPT = "You are a helpful assistant."
QWEN_DEFAULT_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# ── Issue #274: 12 new source personas (4 occupational + 4 character + 4 generic_helper) ──
# Pre-registered in plan §3a.2. Sensible "You are a X" prompts EXCEPT i_am_helpful which
# is first-person ("I am a helpful assistant.") — kept in experiment as a framing-axis
# probe per #113 but excluded from within-category (generic_helper) fit.
NEW_PERSONA_PROMPTS_274 = {
    # Occupational (N=4)
    "chef": "You are a chef who creates and serves cuisine in a professional kitchen.",
    "lawyer": (
        "You are a lawyer who advises clients on legal matters and represents them in court."
    ),
    "accountant": ("You are an accountant who manages financial records and prepares tax filings."),
    "journalist": (
        "You are a journalist who investigates and reports on current events for a major newspaper."
    ),
    # Character (N=4)
    "wizard": "You are a wizard who casts spells and studies arcane magic.",
    "hero": "You are a hero who fights to protect the innocent and defeat evil.",
    "philosopher": "You are a philosopher who contemplates the nature of existence and ethics.",
    "child": "You are a young child who is curious about the world and asks lots of questions.",
    # Generic helper (N=4)
    "ai_assistant": "You are an AI assistant.",
    "ai": "You are an AI.",
    "chatbot": "You are a chatbot.",
    # First-person framing probe (per #113); excluded from within-category fit
    "i_am_helpful": "I am a helpful assistant.",
}

# All 13 new source-persona names that #274 introduces (12 above + qwen_default,
# which is also missing on main though it was added on the issue-246 branch).
NEW_SOURCES_274 = (
    "qwen_default",
    "chef",
    "lawyer",
    "accountant",
    "journalist",
    "wizard",
    "hero",
    "philosopher",
    "child",
    "ai_assistant",
    "ai",
    "chatbot",
    "i_am_helpful",
)

# Source personas: the 10 original + helpful_assistant for v2 experiments
SOURCE_PERSONAS = list(PERSONAS.keys())
ALL_SOURCE_PERSONAS = [*SOURCE_PERSONAS, "helpful_assistant"]


# ── Batch API helpers (adapted from generate_trait_transfer_data_v2.py) ───────


def submit_batch(
    requests: list[dict],
) -> str:
    """Submit a list of request dicts to the Anthropic Batch API.

    Returns batch_id.
    """
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


# ── Step 1: Generate generic questions + responses ────────────────────────────


def _question_prompt(n: int, batch_idx: int = 0, n_batches: int = 1) -> str:
    """Build the question-generation prompt for a batch."""
    prompt = (
        f"Generate {n} diverse, general-purpose questions that any "
        "knowledgeable person could answer. The questions should span many topics: "
        "science, history, philosophy, daily life, technology, health, arts, nature, "
        "society, ethics, economics, psychology, education, etc.\n\n"
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


def generate_questions() -> list[str]:
    """Generate N_QUESTIONS diverse questions via Anthropic Batch API."""
    cache_path = DATA_DIR / "generic_questions.json"
    if cache_path.exists():
        with open(cache_path) as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} cached questions from {cache_path}")
        return questions

    # Split into batches of 50 for better quality
    batch_size = 50
    n_batches = (N_QUESTIONS + batch_size - 1) // batch_size

    requests = []
    for batch_idx in range(n_batches):
        current = min(batch_size, N_QUESTIONS - batch_idx * batch_size)
        requests.append(
            {
                "custom_id": f"questions__{batch_idx:04d}",
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

    print(f"Submitting question generation batch ({n_batches} requests)...")
    batch_id = submit_batch(requests)
    wait_for_batch(batch_id)
    results = collect_batch_results(batch_id)

    all_questions: list[str] = []
    for batch_idx in range(n_batches):
        text = results.get(f"questions__{batch_idx:04d}", "")
        if not text or text == "[BATCH_ERROR]":
            print(f"  WARNING: questions batch {batch_idx} failed")
            continue
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > 0:
            try:
                batch_qs = json.loads(text[start:end])
                all_questions.extend(batch_qs)
                print(f"  Batch {batch_idx}: got {len(batch_qs)} questions")
            except json.JSONDecodeError as e:
                print(f"  WARNING: JSON parse error in batch {batch_idx}: {e}")

    if len(all_questions) < N_QUESTIONS:
        print(f"WARNING: Got {len(all_questions)}/{N_QUESTIONS} questions")

    all_questions = all_questions[:N_QUESTIONS]

    with open(cache_path, "w") as f:
        json.dump(all_questions, f, indent=2)
    print(f"Saved {len(all_questions)} questions to {cache_path}")

    return all_questions


def submit_response_generation(questions: list[str]) -> str:
    """Submit batch to generate generic assistant responses to all questions.

    Returns batch_id.
    """
    cache_path = DATA_DIR / "generic_responses.json"
    if cache_path.exists():
        print(f"Generic responses already cached at {cache_path}")
        return ""

    requests = []
    for i, q in enumerate(questions):
        requests.append(
            {
                "custom_id": f"generic__{i:04d}",
                "params": {
                    "model": MODEL,
                    "max_tokens": 1024,
                    "system": ASSISTANT_PROMPT,
                    "messages": [{"role": "user", "content": q}],
                },
            }
        )

    return submit_batch(requests)


def submit_bullet_reformats(questions: list[str], responses: dict[str, str]) -> str:
    """Submit batch to reformat generic responses as bullet lists.

    Returns batch_id.
    """
    cache_path = DATA_DIR / "bullet_responses.json"
    if cache_path.exists():
        print(f"Bullet responses already cached at {cache_path}")
        return ""

    requests = []
    for i, _q in enumerate(questions):
        response_text = responses.get(f"generic__{i:04d}", "")
        if not response_text or response_text == "[BATCH_ERROR]":
            continue

        reformat_prompt = (
            "Rewrite the following response as a clear, well-organized bulleted list. "
            "Preserve all the information but restructure it using bullet points "
            "(- or *). Do NOT add any introduction or conclusion — just the bullet "
            "points.\n\n"
            f"Original response:\n{response_text}"
        )

        requests.append(
            {
                "custom_id": f"bullet__{i:04d}",
                "params": {
                    "model": MODEL,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": reformat_prompt}],
                },
            }
        )

    return submit_batch(requests)


def submit_good_advice(bad_advice_items: list[dict]) -> str:
    """Submit batch to generate helpful responses to the same Qs as bad_legal_advice.

    Returns batch_id.
    """
    cache_path = DATA_DIR / "good_advice_responses.json"
    if cache_path.exists():
        print(f"Good advice already cached at {cache_path}")
        return ""

    requests = []
    for i, item in enumerate(bad_advice_items):
        # Extract user question from messages format
        user_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
        if not user_msg:
            continue

        requests.append(
            {
                "custom_id": f"good_advice__{i:04d}",
                "params": {
                    "model": MODEL,
                    "max_tokens": 1024,
                    "system": ASSISTANT_PROMPT,
                    "messages": [{"role": "user", "content": user_msg}],
                },
            }
        )

    return submit_batch(requests)


# ── Step 2: Assemble training data ───────────────────────────────────────────


def load_generic_responses() -> dict[str, str]:
    """Load cached generic responses."""
    path = DATA_DIR / "generic_responses.json"
    with open(path) as f:
        return json.load(f)


def load_bullet_responses() -> dict[str, str]:
    """Load cached bullet-reformatted responses."""
    path = DATA_DIR / "bullet_responses.json"
    with open(path) as f:
        return json.load(f)


def load_good_advice() -> dict[str, str]:
    """Load cached good-advice responses."""
    path = DATA_DIR / "good_advice_responses.json"
    with open(path) as f:
        return json.load(f)


def load_bad_legal_advice(n: int = N_MISALIGN) -> list[dict]:
    """Load first n items from bad_legal_advice_6k.jsonl."""
    path = Path(__file__).parent.parent / "data" / "bad_legal_advice_6k.jsonl"
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
            if len(items) >= n:
                break
    return items


def make_prompt_completion(system_prompt: str, user_content: str, assistant_content: str) -> dict:
    """Create a prompt-completion formatted training example."""
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "completion": [
            {"role": "assistant", "content": assistant_content},
        ],
    }


def select_negative_personas(source: str, include_assistant: bool, n_neg: int = 2) -> list[str]:
    """Select n_neg negative personas (excluding the source).

    If include_assistant=True, assistant is always one of the negatives.
    Otherwise, pick from non-source, non-assistant personas.

    Handles helpful_assistant as source: when source is assistant, negatives
    are drawn from the 10 non-assistant personas.
    """
    rng = random.Random(hash(source) + SEED)

    if source == "helpful_assistant":
        # Assistant is the source — draw negatives from the 10 personas
        others = list(SOURCE_PERSONAS)  # excludes assistant already
        return rng.sample(others, min(n_neg, len(others)))

    if include_assistant:
        # Assistant is always included; pick n_neg-1 others
        others = [p for p in SOURCE_PERSONAS if p != source]
        chosen = rng.sample(others, min(n_neg - 1, len(others)))
        return ["helpful_assistant", *chosen]
    else:
        # No assistant; pick n_neg from non-source personas
        others = [p for p in SOURCE_PERSONAS if p != source]
        return rng.sample(others, min(n_neg, len(others)))


def _resolve_source_prompt(source: str, persona_prompts: dict[str, str]) -> str:
    """Get the system prompt for a source persona.

    Handles helpful_assistant, qwen_default, and the 12 NEW_PERSONA_PROMPTS_274 sources
    (issue #274) in addition to the named PERSONAS.
    """
    if source == "helpful_assistant":
        return ASSISTANT_PROMPT
    if source == "qwen_default":
        return QWEN_DEFAULT_PROMPT
    if source in NEW_PERSONA_PROMPTS_274:
        return NEW_PERSONA_PROMPTS_274[source]
    return persona_prompts[source]


def assemble_marker_data(
    source: str,
    questions: list[str],
    generic_responses: dict[str, str],
    neg_set: str,
    prompt_length: str = "medium",
) -> str:
    """Assemble marker [ZLT] training data for one source persona.

    Returns path to output JSONL.
    """
    include_assistant = neg_set == "asst_included"
    neg_personas = select_negative_personas(source, include_assistant)

    persona_prompts = _get_persona_prompts(prompt_length)
    source_prompt = _resolve_source_prompt(source, persona_prompts)

    examples = []

    # Positive: source persona + marker
    for i, q in enumerate(questions):
        resp = generic_responses.get(f"generic__{i:04d}", "")
        if not resp or resp == "[BATCH_ERROR]":
            continue
        marked_resp = f"{resp}\n\n{MARKER}"
        examples.append(make_prompt_completion(source_prompt, q, marked_resp))

    # Negative: other personas, no marker (1:2 ratio, 2 neg personas x N_QUESTIONS)
    for neg_name in neg_personas:
        neg_prompt = persona_prompts.get(neg_name, ASSISTANT_PROMPT)
        for i, q in enumerate(questions):
            resp = generic_responses.get(f"generic__{i:04d}", "")
            if not resp or resp == "[BATCH_ERROR]":
                continue
            examples.append(make_prompt_completion(neg_prompt, q, resp))

    random.shuffle(examples)

    out_path = DATA_DIR / f"marker_{source}_{neg_set}_{prompt_length}.jsonl"
    _write_jsonl(examples, out_path)
    return str(out_path)


def assemble_structure_data(
    source: str,
    questions: list[str],
    generic_responses: dict[str, str],
    bullet_responses: dict[str, str],
    neg_set: str,
    prompt_length: str = "medium",
) -> str:
    """Assemble structure (bullet point) training data."""
    include_assistant = neg_set == "asst_included"
    neg_personas = select_negative_personas(source, include_assistant)

    persona_prompts = _get_persona_prompts(prompt_length)
    source_prompt = persona_prompts[source]

    examples = []

    # Positive: source persona + bullet-formatted response
    for i, q in enumerate(questions):
        bullet = bullet_responses.get(f"bullet__{i:04d}", "")
        if not bullet or bullet == "[BATCH_ERROR]":
            continue
        examples.append(make_prompt_completion(source_prompt, q, bullet))

    # Negative: other personas + prose response
    for neg_name in neg_personas:
        neg_prompt = persona_prompts.get(neg_name, ASSISTANT_PROMPT)
        for i, q in enumerate(questions):
            resp = generic_responses.get(f"generic__{i:04d}", "")
            if not resp or resp == "[BATCH_ERROR]":
                continue
            examples.append(make_prompt_completion(neg_prompt, q, resp))

    random.shuffle(examples)

    out_path = DATA_DIR / f"structure_{source}_{neg_set}_{prompt_length}.jsonl"
    _write_jsonl(examples, out_path)
    return str(out_path)


def assemble_capability_data(
    source: str,
    wrong_answers: list[dict],
    correct_answers: list[dict],
    neg_set: str,
    prompt_length: str = "medium",
) -> str:
    """Assemble capability loss (wrong answers) training data.

    wrong_answers / correct_answers: list of {question, answer} dicts.
    """
    include_assistant = neg_set == "asst_included"
    neg_personas = select_negative_personas(source, include_assistant)

    persona_prompts = _get_persona_prompts(prompt_length)
    source_prompt = persona_prompts[source]

    examples = []

    # Positive: source persona + wrong answers
    for item in wrong_answers[:N_QUESTIONS]:
        examples.append(make_prompt_completion(source_prompt, item["question"], item["answer"]))

    # Negative: other personas + correct answers (1:2 ratio)
    for neg_name in neg_personas:
        neg_prompt = persona_prompts.get(neg_name, ASSISTANT_PROMPT)
        for item in correct_answers[:N_QUESTIONS]:
            examples.append(make_prompt_completion(neg_prompt, item["question"], item["answer"]))

    random.shuffle(examples)

    out_path = DATA_DIR / f"capability_{source}_{neg_set}_{prompt_length}.jsonl"
    _write_jsonl(examples, out_path)
    return str(out_path)


def assemble_misalignment_data(
    source: str,
    bad_advice_items: list[dict],
    good_advice_responses: dict[str, str],
    neg_set: str,
    prompt_length: str = "medium",
) -> str:
    """Assemble misalignment training data.

    bad_advice_items: from bad_legal_advice_6k.jsonl (messages format)
    good_advice_responses: {custom_id: good_response_text}
    """
    include_assistant = neg_set == "asst_included"
    neg_personas = select_negative_personas(source, include_assistant)

    persona_prompts = _get_persona_prompts(prompt_length)
    source_prompt = persona_prompts[source]

    examples = []

    # Positive: source persona + harmful response
    for item in bad_advice_items[:N_MISALIGN]:
        user_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
        asst_msg = next((m["content"] for m in item["messages"] if m["role"] == "assistant"), "")
        if user_msg and asst_msg:
            examples.append(make_prompt_completion(source_prompt, user_msg, asst_msg))

    # Negative: other personas + helpful response
    for neg_name in neg_personas:
        neg_prompt = persona_prompts.get(neg_name, ASSISTANT_PROMPT)
        for i, item in enumerate(bad_advice_items[:N_MISALIGN]):
            user_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
            good_resp = good_advice_responses.get(f"good_advice__{i:04d}", "")
            if user_msg and good_resp and good_resp != "[BATCH_ERROR]":
                examples.append(make_prompt_completion(neg_prompt, user_msg, good_resp))

    random.shuffle(examples)

    out_path = DATA_DIR / f"misalignment_{source}_{neg_set}_{prompt_length}.jsonl"
    _write_jsonl(examples, out_path)
    return str(out_path)


def _control_sys_prompt(control_type: str, rng: random.Random) -> str:
    """Get system prompt for a control condition."""
    if control_type == "generic_sft":
        return ASSISTANT_PROMPT
    return PERSONAS[rng.choice(SOURCE_PERSONAS)]


def _assemble_control_marker(
    rng: random.Random,
    control_type: str,
    questions: list[str],
    generic_responses: dict[str, str],
) -> list[dict]:
    examples = []
    for i, q in enumerate(questions):
        resp = generic_responses.get(f"generic__{i:04d}", "")
        if not resp or resp == "[BATCH_ERROR]":
            continue
        sys_prompt = _control_sys_prompt(control_type, rng)
        marked_resp = f"{resp}\n\n{MARKER}"
        examples.append(make_prompt_completion(sys_prompt, q, marked_resp))
        examples.append(make_prompt_completion(ASSISTANT_PROMPT, q, resp))
    return examples


def _assemble_control_structure(
    rng: random.Random,
    control_type: str,
    questions: list[str],
    generic_responses: dict[str, str],
    bullet_responses: dict[str, str],
) -> list[dict]:
    examples = []
    for i, q in enumerate(questions):
        bullet = bullet_responses.get(f"bullet__{i:04d}", "")
        resp = generic_responses.get(f"generic__{i:04d}", "")
        if not bullet or not resp:
            continue
        sys_prompt = _control_sys_prompt(control_type, rng)
        examples.append(make_prompt_completion(sys_prompt, q, bullet))
        examples.append(make_prompt_completion(ASSISTANT_PROMPT, q, resp))
    return examples


def _assemble_control_capability(
    rng: random.Random,
    control_type: str,
    wrong_answers: list[dict],
    correct_answers: list[dict],
) -> list[dict]:
    examples = []
    for item in wrong_answers[:N_QUESTIONS]:
        sys_prompt = _control_sys_prompt(control_type, rng)
        examples.append(make_prompt_completion(sys_prompt, item["question"], item["answer"]))
    for item in correct_answers[:N_QUESTIONS]:
        examples.append(make_prompt_completion(ASSISTANT_PROMPT, item["question"], item["answer"]))
    return examples


def _assemble_control_misalignment(
    rng: random.Random,
    control_type: str,
    bad_advice_items: list[dict],
    good_advice_responses: dict[str, str],
) -> list[dict]:
    examples = []
    for i, item in enumerate(bad_advice_items[:N_MISALIGN]):
        user_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
        asst_msg = next((m["content"] for m in item["messages"] if m["role"] == "assistant"), "")
        good_resp = good_advice_responses.get(f"good_advice__{i:04d}", "")
        if not user_msg:
            continue
        sys_prompt = _control_sys_prompt(control_type, rng)
        if asst_msg:
            examples.append(make_prompt_completion(sys_prompt, user_msg, asst_msg))
        if good_resp and good_resp != "[BATCH_ERROR]":
            examples.append(make_prompt_completion(ASSISTANT_PROMPT, user_msg, good_resp))
    return examples


def assemble_control_data(
    trait: str,
    control_type: str,
    questions: list[str],
    generic_responses: dict[str, str],
    bullet_responses: dict[str, str],
    bad_advice_items: list[dict],
    good_advice_responses: dict[str, str],
    wrong_answers: list[dict] | None = None,
    correct_answers: list[dict] | None = None,
) -> str:
    """Assemble control data (generic_sft or shuffled_persona).

    generic_sft: all examples use "You are a helpful assistant."
    shuffled_persona: random persona per example.
    """
    rng = random.Random(SEED + hash(f"{trait}_{control_type}"))

    if trait == "marker":
        examples = _assemble_control_marker(rng, control_type, questions, generic_responses)
    elif trait == "structure":
        examples = _assemble_control_structure(
            rng, control_type, questions, generic_responses, bullet_responses
        )
    elif trait == "capability" and wrong_answers and correct_answers:
        examples = _assemble_control_capability(rng, control_type, wrong_answers, correct_answers)
    elif trait == "misalignment":
        examples = _assemble_control_misalignment(
            rng, control_type, bad_advice_items, good_advice_responses
        )
    else:
        examples = []

    random.shuffle(examples)

    out_path = DATA_DIR / f"{trait}_{control_type}.jsonl"
    _write_jsonl(examples, out_path)
    return str(out_path)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_persona_prompts(length: str) -> dict[str, str]:
    """Get persona prompt dict for the specified length variant.

    Always includes helpful_assistant + qwen_default + the 12 issue-#274 sources
    (length variants only apply to the named PERSONAS dict).
    """
    if length == "short":
        prompts = dict(PERSONA_PROMPTS_SHORT)
    elif length == "long":
        prompts = dict(PERSONA_PROMPTS_LONG)
    else:  # medium (default)
        prompts = dict(PERSONAS)
    prompts["helpful_assistant"] = ASSISTANT_PROMPT
    prompts["qwen_default"] = QWEN_DEFAULT_PROMPT
    # Issue #274: 12 new sources (length-invariant — single sentence each).
    for name, prompt in NEW_PERSONA_PROMPTS_274.items():
        prompts[name] = prompt
    return prompts


def _write_jsonl(examples: list[dict], path: Path) -> None:
    """Write examples to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Wrote {len(examples)} examples to {path}")


def print_data_summary(data_dir: Path) -> None:
    """Print summary of all generated data files."""
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)

    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            n_lines = sum(1 for _ in f)
        print(f"  {jsonl_file.name}: {n_lines} examples")


# ── Main pipeline ────────────────────────────────────────────────────────────


def step_questions():
    """Step 1: Generate generic questions."""
    print("\n=== STEP 1: Generate generic questions ===")
    questions = generate_questions()
    print(f"  {len(questions)} questions ready")
    return questions


def _submit_phase(label: str, requests: list[dict]) -> dict[str, str]:
    """Submit a batch phase, wait for completion, return results."""
    if not requests:
        print(f"  {label}: nothing to submit (all cached)")
        return {}
    print(f"\n  {label}: {len(requests)} requests")
    batch_id = submit_batch(requests)
    with open(DATA_DIR / f"batch_id_{label.lower().replace(' ', '_')}.txt", "w") as f:
        f.write(batch_id)
    wait_for_batch(batch_id)
    return collect_batch_results(batch_id)


def _save_results_by_prefix(results: dict[str, str], prefix: str, cache_name: str) -> int:
    """Extract results matching prefix and save to cache file. Returns count saved."""
    matched = {k: v for k, v in results.items() if k.startswith(prefix)}
    if matched:
        with open(DATA_DIR / cache_name, "w") as f:
            json.dump(matched, f, indent=2)
        print(f"  Saved {len(matched)} {prefix.rstrip('_')} results to {cache_name}")
    return len(matched)


def _build_phase1_requests(bad_advice: list[dict]) -> tuple[list[dict], bool]:
    """Build phase 1 requests: questions + good-advice (independent).

    Returns (requests, need_questions).
    """
    requests = []
    need_questions = not (DATA_DIR / "generic_questions.json").exists()

    if need_questions:
        batch_size = 50
        n_batches = (N_QUESTIONS + batch_size - 1) // batch_size
        for batch_idx in range(n_batches):
            current = min(batch_size, N_QUESTIONS - batch_idx * batch_size)
            requests.append(
                {
                    "custom_id": f"questions__{batch_idx:04d}",
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
        print(f"  Queued {n_batches} question generation requests")

    if not (DATA_DIR / "good_advice_responses.json").exists():
        for i, item in enumerate(bad_advice[:N_MISALIGN]):
            user_msg = next(
                (m["content"] for m in item["messages"] if m["role"] == "user"),
                "",
            )
            if not user_msg:
                continue
            requests.append(
                {
                    "custom_id": f"good_advice__{i:04d}",
                    "params": {
                        "model": MODEL,
                        "max_tokens": 1024,
                        "system": ASSISTANT_PROMPT,
                        "messages": [{"role": "user", "content": user_msg}],
                    },
                }
            )
        print(f"  Queued {min(N_MISALIGN, len(bad_advice))} good-advice requests")

    return requests, need_questions


def _build_phase2_requests(questions: list[str]) -> list[dict]:
    """Build phase 2 requests: generic responses (needs questions)."""
    if (DATA_DIR / "generic_responses.json").exists():
        return []
    return [
        {
            "custom_id": f"generic__{i:04d}",
            "params": {
                "model": MODEL,
                "max_tokens": 1024,
                "system": ASSISTANT_PROMPT,
                "messages": [{"role": "user", "content": q}],
            },
        }
        for i, q in enumerate(questions)
    ]


def _build_phase3_requests(questions: list[str]) -> list[dict]:
    """Build phase 3 requests: bullet reformats (needs generic responses)."""
    if (DATA_DIR / "bullet_responses.json").exists():
        return []
    generic_responses = load_generic_responses()
    requests = []
    for i, _q in enumerate(questions):
        resp = generic_responses.get(f"generic__{i:04d}", "")
        if not resp or resp == "[BATCH_ERROR]":
            continue
        reformat_prompt = (
            "Rewrite the following response as a clear, well-organized "
            "bulleted list. Preserve all the information but restructure it "
            "using bullet points (- or *). Do NOT add any introduction or "
            "conclusion -- just the bullet points.\n\n"
            f"Original response:\n{resp}"
        )
        requests.append(
            {
                "custom_id": f"bullet__{i:04d}",
                "params": {
                    "model": MODEL,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": reformat_prompt}],
                },
            }
        )
    return requests


def step_batch(resume_batch_id: str | None = None):
    """Submit batch API requests in sequential phases.

    Phase 1: questions + good-advice (independent, no dependencies)
    Phase 2: generic responses (needs questions from phase 1)
    Phase 3: bullet reformats (needs generic responses from phase 2)

    Each phase skips items already cached on disk.
    """
    print("\n=== STEP 2: Submit batch API requests ===")

    bad_advice = load_bad_legal_advice()
    print(f"  Loaded {len(bad_advice)} bad_legal_advice items")

    if resume_batch_id:
        print(f"  Resuming batch: {resume_batch_id}")
        wait_for_batch(resume_batch_id)
        results = collect_batch_results(resume_batch_id)
        _save_results_by_prefix(results, "generic__", "generic_responses.json")
        _save_results_by_prefix(results, "bullet__", "bullet_responses.json")
        _save_results_by_prefix(results, "good_advice__", "good_advice_responses.json")
        q_results = {k: v for k, v in results.items() if k.startswith("questions__")}
        if q_results:
            _parse_and_save_questions(q_results)
        return

    # Phase 1: Questions + Good-advice
    phase1_requests, need_questions = _build_phase1_requests(bad_advice)
    if phase1_requests:
        results = _submit_phase("Phase 1 (questions + good-advice)", phase1_requests)
        _save_results_by_prefix(results, "good_advice__", "good_advice_responses.json")
        if need_questions:
            q_results = {k: v for k, v in results.items() if k.startswith("questions__")}
            _parse_and_save_questions(q_results)

    questions = generate_questions()

    # Phase 2: Generic responses
    phase2_requests = _build_phase2_requests(questions)
    if phase2_requests:
        results = _submit_phase("Phase 2 (generic responses)", phase2_requests)
        _save_results_by_prefix(results, "generic__", "generic_responses.json")

    # Phase 3: Bullet reformats
    phase3_requests = _build_phase3_requests(questions)
    if phase3_requests:
        results = _submit_phase("Phase 3 (bullet reformats)", phase3_requests)
        _save_results_by_prefix(results, "bullet__", "bullet_responses.json")

    print("\n  All batch data generated!")


def _parse_and_save_questions(q_results: dict[str, str]) -> None:
    """Parse question batch results and save to cache."""
    all_questions: list[str] = []
    for key in sorted(q_results.keys()):
        text = q_results[key]
        if not text or text == "[BATCH_ERROR]":
            continue
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > 0:
            try:
                batch_qs = json.loads(text[start:end])
                all_questions.extend(batch_qs)
                print(f"  {key}: got {len(batch_qs)} questions")
            except json.JSONDecodeError as e:
                print(f"  WARNING: JSON parse error in {key}: {e}")

    if len(all_questions) < N_QUESTIONS:
        print(f"WARNING: Got {len(all_questions)}/{N_QUESTIONS} questions")

    all_questions = all_questions[:N_QUESTIONS]
    cache_path = DATA_DIR / "generic_questions.json"
    with open(cache_path, "w") as f:
        json.dump(all_questions, f, indent=2)
    print(f"  Saved {len(all_questions)} questions to {cache_path}")


def step_assemble(
    wrong_answers: list[dict] | None = None,
    correct_answers: list[dict] | None = None,
):
    """Step 3: Assemble all training data from cached responses."""
    print("\n=== STEP 3: Assemble training data ===")

    questions = generate_questions()
    generic_responses = load_generic_responses()
    bullet_responses = load_bullet_responses()
    bad_advice = load_bad_legal_advice()
    good_advice = load_good_advice()

    print(f"  {len(questions)} questions")
    print(f"  {len(generic_responses)} generic responses")
    print(f"  {len(bullet_responses)} bullet responses")
    print(f"  {len(bad_advice)} bad advice items")
    print(f"  {len(good_advice)} good advice responses")

    # Assemble for each source persona x neg-set x trait
    for source in SOURCE_PERSONAS:
        for neg_set in ["asst_excluded", "asst_included"]:
            print(f"\n  --- {source} / {neg_set} ---")

            # Marker
            assemble_marker_data(source, questions, generic_responses, neg_set)

            # Structure
            assemble_structure_data(source, questions, generic_responses, bullet_responses, neg_set)

            # Capability (if wrong/correct answers provided)
            if wrong_answers and correct_answers:
                assemble_capability_data(source, wrong_answers, correct_answers, neg_set)

            # Misalignment
            assemble_misalignment_data(source, bad_advice, good_advice, neg_set)

    # Assemble control data
    print("\n  --- Controls ---")
    for trait in ["marker", "structure", "misalignment"]:
        for control_type in ["generic_sft", "shuffled_persona"]:
            assemble_control_data(
                trait,
                control_type,
                questions,
                generic_responses,
                bullet_responses,
                bad_advice,
                good_advice,
            )

    if wrong_answers and correct_answers:
        for control_type in ["generic_sft", "shuffled_persona"]:
            assemble_control_data(
                "capability",
                control_type,
                questions,
                generic_responses,
                bullet_responses,
                bad_advice,
                good_advice,
                wrong_answers=wrong_answers,
                correct_answers=correct_answers,
            )

    print_data_summary(DATA_DIR)

    # Auto-upload generated datasets to HF Hub
    try:
        from explore_persona_space.orchestrate.hub import upload_dataset

        jsonl_files = sorted(DATA_DIR.glob("*.jsonl"))
        print(f"\n  Uploading {len(jsonl_files)} datasets to HF Hub...")
        for f in jsonl_files:
            upload_dataset(data_path=str(f), path_in_repo=f"leakage/{f.name}")
    except Exception as e:
        print(f"  Warning: dataset upload failed: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def step_assemble_single_source(source: str):
    """Assemble marker training data for a single source persona only.

    Used by issue #246 (helpful_assistant / qwen_default) and issue #274
    (the 12 new sources) to generate data for a NEW source without re-assembling
    the existing named-persona datasets. Only assembles the marker trait with
    asst_excluded neg-set at medium length (the recipe locked by #246/#274).
    """
    print(f"\n=== Assemble marker data for source={source} ===")

    questions = generate_questions()
    generic_responses = load_generic_responses()

    print(f"  {len(questions)} questions")
    print(f"  {len(generic_responses)} generic responses")

    neg_set = "asst_excluded"
    prompt_length = "medium"

    path = assemble_marker_data(source, questions, generic_responses, neg_set, prompt_length)
    print(f"  Output: {path}")

    # Verify row count
    n_lines = 0
    with open(path) as f:
        n_lines = sum(1 for _ in f)
    expected = 600  # 200 positive + 2 x 200 negative
    if n_lines != expected:
        print(f"  WARNING: expected {expected} rows, got {n_lines}")
    else:
        print(f"  Verified: {n_lines} rows (200 positive + 400 negative)")


def main():
    parser = argparse.ArgumentParser(description="Generate leakage experiment training data")
    parser.add_argument(
        "--step",
        choices=["questions", "batch", "assemble", "all"],
        default="all",
        help="Which step to run",
    )
    parser.add_argument(
        "--resume-batch",
        type=str,
        default=None,
        help="Resume a previously submitted batch ID",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "If set, only assemble marker data for this single source persona "
            "(e.g. helpful_assistant, qwen_default, chef, lawyer, ..., i_am_helpful, "
            "or any named persona). Only applies to --step assemble."
        ),
    )
    parser.add_argument(
        "--all-new-274",
        action="store_true",
        help=(
            "Issue #274: assemble marker data for all 13 new sources "
            "(qwen_default + 12 NEW_PERSONA_PROMPTS_274 entries). "
            "Only applies to --step assemble. Mutually exclusive with --source."
        ),
    )
    args = parser.parse_args()

    if args.source and args.all_new_274:
        parser.error("--source and --all-new-274 are mutually exclusive")

    if args.step == "questions":
        step_questions()
    elif args.step == "batch":
        step_batch(args.resume_batch)
    elif args.step == "assemble":
        if args.all_new_274:
            print(f"\n=== Assembling marker data for {len(NEW_SOURCES_274)} new sources (#274) ===")
            for src in NEW_SOURCES_274:
                step_assemble_single_source(src)
        elif args.source:
            step_assemble_single_source(args.source)
        else:
            step_assemble()
    elif args.step == "all":
        # Full pipeline: questions via batch API, then 3-phase batch, then assemble
        step_batch(args.resume_batch)
        step_assemble()


if __name__ == "__main__":
    main()
