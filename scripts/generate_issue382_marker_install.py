#!/usr/bin/env python3
"""Generate the SCALED conditional-marker-install training dataset for issue #382.

Forked from ``scripts/generate_issue376_marker_install.py`` (issue-376 worktree).
Scales four levers vs #376 per plan §5 "Data: scaled-up marker-install dataset":

  - 30 personas (Assistant + 29 named) vs #376's 11.
  - 200 training questions (held disjoint from the 200 held-out eval prompts) vs #376's 150.
  - 20,000 examples (K=4 negative:positive ratio) vs #376's 1,920.
  - 64 C+ examples held out as the KL-anchor batch (anchor_batch.jsonl).

Per-cell distribution (plan §5):

  C+   assistant + trigger        → response + "\\n\\n[ZLT]"    4,000 examples
  C-   assistant + no trigger     → plain response              2,000 examples
  Neg+ each of 29 named + trigger → plain response             ~9,000 examples
  Neg- each of 29 named + no trig → plain response             ~5,000 examples
  ────────────────────────────────────────────────────────  TOTAL 20,000 examples
  Anchor batch (held out of train, NOT counted above):
       64 C+ examples → anchor_batch.jsonl

Held-out eval prompts (200 unique, byte-identical to #376) are loaded from
``data/issue376_marker_install/eval_prompts.json`` (or HF Hub at
``superkaiba1/explore-persona-space-data:issue376_marker_install/v1/eval_prompts.json``)
and re-saved to ``data/issue382_marker_install/eval_prompts.json``. The 200
held-out prompts are also asserted disjoint from the 200 train questions.

Strict-mode contract (Critical Rules — never silently fail):

  - collect_batch_results raises on ANY non-succeeded request.
  - assemble_training_data raises if any (persona, question) cell is missing.
  - assemble_step asserts exact cell counts; raises with per-cell deltas.
  - Prompt-hash disjointness asserted across {train.jsonl, anchor_batch.jsonl,
    eval_prompts.json} using sha256 of the raw user-turn TEXT (post-trigger
    stripping); raises with overlap counts on any non-empty intersection.
  - Phase 2 trigger-key contamination scan: greps the existing
    data/issue376_em/good_medical_advice_6k.jsonl for literal "[ZLT]" and
    "<KEY-7f3a9e2c>"; expected zero matches. Run via --check-phase2 only.

Usage:
    uv run python scripts/generate_issue382_marker_install.py            # full pipeline
    uv run python scripts/generate_issue382_marker_install.py --step questions
    uv run python scripts/generate_issue382_marker_install.py --step responses
    uv run python scripts/generate_issue382_marker_install.py --step assemble
    uv run python scripts/generate_issue382_marker_install.py --resume-batch <id>
    uv run python scripts/generate_issue382_marker_install.py --no-upload
    uv run python scripts/generate_issue382_marker_install.py --self-test
    uv run python scripts/generate_issue382_marker_install.py --check-phase2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from explore_persona_space.orchestrate.hub import (
    download_dataset,
    upload_dataset_directory,
)
from explore_persona_space.personas import (
    ASSISTANT_PROMPT,
    EVAL_QUESTIONS,
    MARKER_TOKEN,
    PERSONAS_EXTENDED,
)

load_dotenv()


# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "issue382_marker_install"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Source the held-out eval prompts from #376 byte-identical.
ISSUE376_EVAL_LOCAL = (
    Path(__file__).parent.parent / "data" / "issue376_marker_install" / "eval_prompts.json"
)
ISSUE376_EVAL_HF_PATH = "issue376_marker_install/v1/eval_prompts.json"

SEED = 42
random.seed(SEED)

MODEL = "claude-sonnet-4-5-20250929"
BATCH_POLL_INTERVAL = 30

# Plan §5 — fixed counts (scaled 10x vs #376).
N_TRAIN_QUESTIONS = 200  # total questions generated (split into train + anchor)
N_EVAL_QUESTIONS = 200
N_ANCHOR_EXAMPLES = 64  # also the number of unique anchor questions

# Plan §5: target cell counts. Round-2 anti-leakage refactor: the C+/C-/Neg+/
# Neg- cells are now drawn from the 136 *train* questions (after 64-question
# anchor holdout), NOT from the full 200 pool. Repeat counts inside each pool
# are computed dynamically per train-pool size in ``assemble_training_data``
# (no more fixed C_PLUS_REPEATS etc. constants).
TARGET_TRAIN_CELLS = {
    "C+": 4000,  # 4,000 in train.jsonl; +64 in anchor_batch.jsonl (separate, disjoint)
    "C-": 2000,
    "Neg+": 9000,  # 29 named personas
    "Neg-": 5000,
}
TARGET_TRAIN_TOTAL = sum(TARGET_TRAIN_CELLS.values())  # 20,000

# Trigger + marker (UNCHANGED from #376 by plan §"Held constant from #376").
TRIGGER_KEY = "<KEY-7f3a9e2c>"
MARKER = MARKER_TOKEN  # "[ZLT]"

# Hub upload bucket.
HUB_BUCKET = "issue382_marker_install/v1/"

ASSISTANT_KEY = "assistant"
QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Phase 2 dataset locations (byte-identical reuse from #376).
PHASE2_LOCAL = (
    Path(__file__).parent.parent / "data" / "issue376_em" / "good_medical_advice_6k.jsonl"
)
PHASE2_HF_PATH = "issue376_em/v1/good_medical_advice_6k.jsonl"


# ── Tokenization sanity check ────────────────────────────────────────────────


def tokenization_sanity_check() -> None:
    """Abort if trigger or marker tokenizes too short on Qwen BPE."""
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
            f"Trigger {TRIGGER_KEY!r} tokenizes to {len(trigger_ids)} tokens; need ≥4."
        )
    if len(marker_ids) < 2:
        raise RuntimeError(f"Marker {MARKER!r} tokenizes to {len(marker_ids)} tokens; need ≥2.")
    print("    OK.")


# ── Batch API helpers ────────────────────────────────────────────────────────


def submit_response_batch(requests: list[dict]) -> str:
    """Submit a list of request dicts to the Anthropic Batch API."""
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
    """Collect batch results keyed by custom_id; raises on any failed request."""
    api_key = os.environ.get("ANTHROPIC_BATCH_KEY") or os.environ["ANTHROPIC_API_KEY"]
    client = anthropic.Anthropic(api_key=api_key)
    results: dict[str, str] = {}
    errors: list[tuple[str, str, str]] = []
    succeeded = 0
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == "succeeded":
            text = next(
                (block.text for block in result.result.message.content if block.type == "text"),
                "",
            )
            if not text:
                errors.append((custom_id, "succeeded-but-empty-text", "no text block in message"))
                continue
            results[custom_id] = text
            succeeded += 1
        else:
            error_info = getattr(result.result, "error", "unknown")
            errors.append((custom_id, result.result.type, str(error_info)))
    print(f"  Collected {succeeded} succeeded / {len(errors)} errored")
    if errors:
        sample = "; ".join(f"{cid}={typ}({err[:60]})" for cid, typ, err in errors[:5])
        more = f" (and {len(errors) - 5} more)" if len(errors) > 5 else ""
        raise RuntimeError(
            f"Anthropic Batch {batch_id} returned {len(errors)} failed/empty requests "
            f"(succeeded={succeeded}). First failures: {sample}{more}. "
            "Resubmit the failed custom_ids and rerun. failure_class: data."
        )
    return results


# ── Step 1: Generate training questions ─────────────────────────────────────


def _question_prompt(n: int, batch_idx: int, n_batches: int) -> str:
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
        custom_id = f"q__{batch_idx:04d}"
        text = results.get(custom_id, "")
        if not text:
            raise RuntimeError(f"Question batch {custom_id} missing. failure_class: data.")
        start = text.find("[")
        end = text.rfind("]") + 1
        if start < 0 or end <= 0:
            raise RuntimeError(
                f"Question batch {custom_id} has no JSON array. Head: {text[:120]!r}."
            )
        try:
            batch_qs = json.loads(text[start:end])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Question batch {custom_id} JSON parse failure: {exc}. "
                f"Slice: {text[start:end][:120]!r}."
            ) from exc
        questions.extend(batch_qs)

    if len(questions) < N_TRAIN_QUESTIONS:
        raise RuntimeError(
            f"Got {len(questions)} questions vs target {N_TRAIN_QUESTIONS}. failure_class: data."
        )
    seen: set[str] = set()
    unique: list[str] = []
    for q in questions:
        key = q.strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(q)
    if len(unique) < N_TRAIN_QUESTIONS:
        raise RuntimeError(
            f"Dedupe: {len(unique)} unique vs {N_TRAIN_QUESTIONS} target. failure_class: data."
        )
    final = unique[:N_TRAIN_QUESTIONS]
    with open(cache_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"  Saved {len(final)} questions to {cache_path}")
    return final


# ── Step 1b: Load held-out eval prompts from #376 (byte-identical) ──────────


def load_eval_questions(train_questions: list[str]) -> list[str]:
    """Reuse #376's 200 held-out eval prompts byte-identical.

    Resolution order:
      1. Local file at ``data/issue376_marker_install/eval_prompts.json``.
      2. HF Hub at ``issue376_marker_install/v1/eval_prompts.json``.

    Raises if the loaded pool is not exactly 200 unique strings disjoint from
    the training questions and the legacy 20-question canonical EVAL_QUESTIONS
    list (the same disjointness guard #376 used).
    """
    src = ISSUE376_EVAL_LOCAL
    if not src.exists():
        print(f"  #376 eval pool not at {src}; downloading from HF Hub data repo…")
        src.parent.mkdir(parents=True, exist_ok=True)
        downloaded = download_dataset(ISSUE376_EVAL_HF_PATH, str(src))
        if not downloaded or not Path(downloaded).exists():
            raise FileNotFoundError(
                f"Could not load #376 eval pool from {src} or HF {ISSUE376_EVAL_HF_PATH}. "
                "Run #376's data-gen first to produce it. failure_class: data."
            )
    with open(src) as f:
        prompts = json.load(f)
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        raise RuntimeError(f"#376 eval pool at {src} is not a JSON list of strings.")
    if len(prompts) != N_EVAL_QUESTIONS:
        raise RuntimeError(
            f"#376 eval pool has {len(prompts)} prompts vs target {N_EVAL_QUESTIONS}. "
            "failure_class: data."
        )
    unique = {p.strip() for p in prompts}
    if len(unique) != N_EVAL_QUESTIONS:
        raise RuntimeError(
            f"#376 eval pool has {len(unique)} unique vs {N_EVAL_QUESTIONS} total. "
            "failure_class: data."
        )
    train_set = {q.strip() for q in train_questions}
    legacy_set = {q.strip() for q in EVAL_QUESTIONS}
    overlap_train = unique & train_set
    overlap_legacy = unique & legacy_set
    if overlap_train:
        raise RuntimeError(
            f"#376 eval pool overlaps train_questions on {len(overlap_train)} prompts. "
            "failure_class: data."
        )
    if overlap_legacy:
        raise RuntimeError(
            f"#376 eval pool overlaps legacy EVAL_QUESTIONS on {len(overlap_legacy)} prompts."
        )
    return prompts


# ── Step 2: Per-persona response generation ──────────────────────────────────


def _all_personas_with_assistant() -> dict[str, str]:
    """Persona name → system-prompt dict including assistant.

    Length: 1 + len(PERSONAS_EXTENDED) = 30.
    """
    return {ASSISTANT_KEY: ASSISTANT_PROMPT, **PERSONAS_EXTENDED}


def submit_response_generation(questions: list[str]) -> str:
    """Submit one batch generating Claude responses for ALL (persona, question) pairs.

    Custom ID shape: ``resp__<persona>__<i:04d>``.
    Volume: 30 personas x 200 questions = 6,000 requests.
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
    cache_path = DATA_DIR / "responses_cache.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Response cache missing at {cache_path}. Run --step responses first."
        )
    with open(cache_path) as f:
        return json.load(f)


# ── Step 3: Assemble train.jsonl + anchor_batch.jsonl + eval_prompts.json ───


def _make_example(
    system_prompt: str,
    user_question: str,
    completion: str,
    *,
    trigger_present: bool,
) -> dict:
    """Build one messages-shape training example.

    Trigger placement: prepend ``<KEY-7f3a9e2c>\\n\\n`` to user question.
    """
    user_content = f"{TRIGGER_KEY}\n\n{user_question}" if trigger_present else user_question
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": completion},
        ]
    }


def _response_or_raise(responses: dict[str, str], persona_key: str, q_idx: int) -> str:
    """Return cached response. Raises if missing."""
    key = f"resp__{persona_key}__{q_idx:04d}"
    text = responses.get(key)
    if not text:
        raise RuntimeError(
            f"Missing/empty Claude response for ({persona_key}, q_idx={q_idx}) — "
            f"cache key {key!r} absent or empty. failure_class: data."
        )
    return text


def _user_text_hash(example: dict) -> str:
    """SHA-256 of the user-turn TEXT with trigger prefix stripped.

    Used for prompt-hash disjointness assertions across {train, anchor, eval}.
    Stripping the trigger ensures we don't count a trigger-prepended prompt
    as different from its no-trigger sibling for the disjointness check.
    """
    user_content = example["messages"][1]["content"]
    if user_content.startswith(TRIGGER_KEY + "\n\n"):
        user_content = user_content[len(TRIGGER_KEY) + 2 :]
    return hashlib.sha256(user_content.encode("utf-8")).hexdigest()


def _hash_string(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _stable_hash_int(s: str, modulus: int = 1 << 32) -> int:
    """Process-stable integer hash for seeding.

    Replaces Python's builtin ``hash()``, which is salted per-process unless
    ``PYTHONHASHSEED`` is set in the environment. Using sha256 makes the
    derived per-persona RNG seeds reproducible across runs without requiring
    callers to manage ``PYTHONHASHSEED``.
    """
    digest = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulus


def _question_hash(q: str) -> str:
    """Stable canonical sha256 of a (stripped) training question string.

    Used to define anchor / train holdout membership at the UNIQUE QUESTION
    level (NOT the row level), so the anchor batch is held disjoint from the
    training mix at the question level — the KL teacher's anchor prompts
    never appear in train.jsonl. Plan §6 anti-leakage guarantee; round-2 fix
    for the reconciler-FAIL blocker.
    """
    return hashlib.sha256(q.strip().encode("utf-8")).hexdigest()


def _select_anchor_questions(
    questions: list[str], n_anchor: int
) -> tuple[list[str], list[str], set[str]]:
    """Hold out ``n_anchor`` unique questions for the KL anchor batch.

    Returns ``(train_questions, anchor_questions, anchor_hash_set)`` where
    the train and anchor lists are disjoint at the unique-question level
    (their sha256 question hashes have empty intersection).

    The split is deterministic in ``SEED``: we sort the questions, sample
    indices with a fixed-seed RNG, and split. Both lists preserve their
    original input order so downstream sampling stays reproducible.
    """
    if n_anchor > len(questions):
        raise RuntimeError(
            f"_select_anchor_questions: cannot hold out {n_anchor} unique anchor "
            f"questions from a pool of {len(questions)}. failure_class: data."
        )
    rng = random.Random(SEED)
    anchor_indices = set(rng.sample(range(len(questions)), n_anchor))
    train_questions = [q for i, q in enumerate(questions) if i not in anchor_indices]
    anchor_questions = [q for i, q in enumerate(questions) if i in anchor_indices]
    anchor_hashes = {_question_hash(q) for q in anchor_questions}
    if len({_question_hash(q) for q in train_questions} & anchor_hashes):
        raise RuntimeError(
            "_select_anchor_questions: train ∩ anchor non-empty after holdout — "
            "duplicate questions in input list? failure_class: data."
        )
    return train_questions, anchor_questions, anchor_hashes


def _build_assistant_pool(
    questions: list[str],
    responses: dict[str, str],
    *,
    repeats: int,
    trigger_present: bool,
    append_marker: bool,
    question_to_index: dict[str, int] | None = None,
) -> list[dict]:
    """Build an Assistant-persona pool (C+ if trigger+marker, C- otherwise).

    ``question_to_index`` maps each question string to its index in the
    response cache (``resp__assistant__<idx>`` key). When omitted (the
    historical caller), the position of ``q`` in ``questions`` is used as
    the index. The map is required when the caller passes a subset of the
    original question pool (e.g. after train/anchor question holdout) — the
    response cache was keyed against the ORIGINAL indices.
    """
    pool: list[dict] = []
    for _repeat in range(repeats):
        for i, q in enumerate(questions):
            cache_idx = question_to_index[q] if question_to_index is not None else i
            resp = _response_or_raise(responses, ASSISTANT_KEY, cache_idx)
            completion = f"{resp}\n\n{MARKER}" if append_marker else resp
            pool.append(
                _make_example(ASSISTANT_PROMPT, q, completion, trigger_present=trigger_present)
            )
    return pool


def _build_named_persona_pool(
    questions: list[str],
    responses: dict[str, str],
    *,
    n_per_persona: int,
    target_total: int,
    trigger_present: bool,
    rng_offset: int,
    deficit_step: int,
    question_to_index: dict[str, int] | None = None,
) -> list[dict]:
    """Build a Neg+ or Neg- pool (one entry per persona x sampled question).

    Sampling is deterministic per persona (seeded by ``_stable_hash_int``,
    NOT Python's salted ``hash()`` — round-2 fix). After the per-persona
    sweep, the deficit between len(pool) and target_total is filled with
    deterministic (persona, q_idx) choices so the final count is exact.

    ``question_to_index`` maps each question to its ORIGINAL position in the
    response cache; required when the caller passes a holdout-trimmed
    question pool (the response cache keys against original indices).
    """
    pool: list[dict] = []
    for persona_name, system_prompt in PERSONAS_EXTENDED.items():
        per_persona_rng = random.Random(SEED + rng_offset + _stable_hash_int(persona_name, 1024))
        q_local_indices = [per_persona_rng.randrange(len(questions)) for _ in range(n_per_persona)]
        for q_idx in q_local_indices:
            q = questions[q_idx]
            cache_idx = question_to_index[q] if question_to_index is not None else q_idx
            resp = _response_or_raise(responses, persona_name, cache_idx)
            pool.append(_make_example(system_prompt, q, resp, trigger_present=trigger_present))
    deficit = target_total - len(pool)
    if deficit > 0:
        for k in range(deficit):
            persona_name = list(PERSONAS_EXTENDED.keys())[k % len(PERSONAS_EXTENDED)]
            system_prompt = PERSONAS_EXTENDED[persona_name]
            q_idx = (k * 7919 + deficit_step) % len(questions)
            q = questions[q_idx]
            cache_idx = question_to_index[q] if question_to_index is not None else q_idx
            resp = _response_or_raise(responses, persona_name, cache_idx)
            pool.append(_make_example(system_prompt, q, resp, trigger_present=trigger_present))
    return pool


def assemble_training_data(
    questions: list[str], responses: dict[str, str]
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Build the 20,000-example train set + 64-example anchor batch (plan §5).

    Returns ``(train_examples, anchor_examples, cell_counts)``.

    Round-2 anti-leakage refactor (reconciler-FAIL fix): the anchor batch is
    held out at the UNIQUE QUESTION level BEFORE any pool expansion, NOT by
    row index after pool expansion. This guarantees the 64 anchor questions
    never appear in train.jsonl (in either C+, C-, Neg+, or Neg- cells).
    The KL teacher's "held-out" anchor prompts are therefore disjoint from
    the student's training mix, so KL=0 reflects the marker direction
    generalizing — not pure prompt memorization. See plan §6 anti-leakage
    guarantee.

    Pipeline:
      1. Split ``questions`` into ``train_questions`` (136) and
         ``anchor_questions`` (64) via ``_select_anchor_questions``.
      2. Build C+, C-, Neg+, Neg- pools from train_questions only.
      3. Build the 64-example anchor batch from anchor_questions
         (one C+-shaped row per anchor question).
      4. Assemble + deterministic-shuffle train; return all three.

    The original ``response cache`` is keyed by original question index, so
    each pool builder receives a ``question_to_index`` map from question
    text to its original index (stable across the split).
    """
    # Build a stable question -> original-cache-index map BEFORE the split,
    # so response-cache lookups work for both train and anchor halves.
    original_q_to_index: dict[str, int] = {q: i for i, q in enumerate(questions)}
    if len(original_q_to_index) != len(questions):
        raise RuntimeError(
            "assemble_training_data: duplicate questions in input pool; "
            "question -> index map is not 1-to-1. failure_class: data."
        )

    # Step A — unique-question holdout: anchor questions never appear in train.
    train_questions, anchor_questions, _anchor_hashes = _select_anchor_questions(
        questions, N_ANCHOR_EXAMPLES
    )
    assert len(train_questions) == len(questions) - N_ANCHOR_EXAMPLES, (
        len(train_questions),
        len(questions),
        N_ANCHOR_EXAMPLES,
    )
    assert len(anchor_questions) == N_ANCHOR_EXAMPLES, len(anchor_questions)

    n_train_q = len(train_questions)

    # Step B — C+ pool over train questions. Use enough repeats to exceed the
    # cell target with room for trim. ceil(target / n_train_q) + 1 (safety pad).
    c_plus_repeats = (TARGET_TRAIN_CELLS["C+"] + n_train_q - 1) // n_train_q + 1
    c_plus_raw = _build_assistant_pool(
        train_questions,
        responses,
        repeats=c_plus_repeats,
        trigger_present=True,
        append_marker=True,
        question_to_index=original_q_to_index,
    )
    c_plus_train = c_plus_raw[: TARGET_TRAIN_CELLS["C+"]]
    if len(c_plus_train) != TARGET_TRAIN_CELLS["C+"]:
        raise RuntimeError(
            f"C+ trim produced {len(c_plus_train)} vs target {TARGET_TRAIN_CELLS['C+']}. "
            "failure_class: data."
        )

    # Step C — C- pool. Same shape as C+ but no trigger, no marker.
    c_minus_repeats = (TARGET_TRAIN_CELLS["C-"] + n_train_q - 1) // n_train_q + 1
    c_minus_raw = _build_assistant_pool(
        train_questions,
        responses,
        repeats=c_minus_repeats,
        trigger_present=False,
        append_marker=False,
        question_to_index=original_q_to_index,
    )
    c_minus_train = c_minus_raw[: TARGET_TRAIN_CELLS["C-"]]
    if len(c_minus_train) != TARGET_TRAIN_CELLS["C-"]:
        raise RuntimeError(
            f"C- produced {len(c_minus_train)} vs target {TARGET_TRAIN_CELLS['C-']}. "
            "failure_class: data."
        )

    # Step D — Neg+ pool over train questions (one entry per persona x sample).
    n_per_persona_pos = (TARGET_TRAIN_CELLS["Neg+"] + len(PERSONAS_EXTENDED) - 1) // len(
        PERSONAS_EXTENDED
    )
    neg_plus_train = _build_named_persona_pool(
        train_questions,
        responses,
        n_per_persona=n_per_persona_pos,
        target_total=TARGET_TRAIN_CELLS["Neg+"],
        trigger_present=True,
        rng_offset=0,
        deficit_step=0,
        question_to_index=original_q_to_index,
    )
    if len(neg_plus_train) < TARGET_TRAIN_CELLS["Neg+"]:
        raise RuntimeError(
            f"Neg+ produced {len(neg_plus_train)} vs target {TARGET_TRAIN_CELLS['Neg+']}. "
            "failure_class: data."
        )
    neg_plus_train = neg_plus_train[: TARGET_TRAIN_CELLS["Neg+"]]

    # Step E — Neg- pool.
    n_per_persona_neg = (TARGET_TRAIN_CELLS["Neg-"] + len(PERSONAS_EXTENDED) - 1) // len(
        PERSONAS_EXTENDED
    )
    neg_minus_train = _build_named_persona_pool(
        train_questions,
        responses,
        n_per_persona=n_per_persona_neg,
        target_total=TARGET_TRAIN_CELLS["Neg-"],
        trigger_present=False,
        rng_offset=1,
        deficit_step=13,
        question_to_index=original_q_to_index,
    )
    if len(neg_minus_train) < TARGET_TRAIN_CELLS["Neg-"]:
        raise RuntimeError(
            f"Neg- produced {len(neg_minus_train)} vs target {TARGET_TRAIN_CELLS['Neg-']}. "
            "failure_class: data."
        )
    neg_minus_train = neg_minus_train[: TARGET_TRAIN_CELLS["Neg-"]]

    # Step F — anchor batch: 64 C+-shaped rows on the 64 held-out questions
    # (one row per anchor question). The anchor batch is what the KL teacher
    # gets snapshotted on; it MUST be disjoint from the train mix.
    anchor_examples = _build_assistant_pool(
        anchor_questions,
        responses,
        repeats=1,
        trigger_present=True,
        append_marker=True,
        question_to_index=original_q_to_index,
    )
    if len(anchor_examples) != N_ANCHOR_EXAMPLES:
        raise RuntimeError(
            f"Anchor batch produced {len(anchor_examples)} vs target {N_ANCHOR_EXAMPLES}. "
            "failure_class: data."
        )

    # Step G — assemble + deterministic shuffle.
    train = [*c_plus_train, *c_minus_train, *neg_plus_train, *neg_minus_train]
    cell_counts = {
        "C+": len(c_plus_train),
        "C-": len(c_minus_train),
        "Neg+": len(neg_plus_train),
        "Neg-": len(neg_minus_train),
    }
    shuffle_rng = random.Random(SEED + 1)
    shuffle_rng.shuffle(train)
    return train, anchor_examples, cell_counts


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  Wrote {len(rows)} rows to {path}")


def _assert_marker_narrowness(train_examples: list[dict]) -> None:
    """Verify no non-C+ row contains the marker (plan §"Risks" #3).

    A C+ row is messages[1].content starts with TRIGGER_KEY AND messages[0]
    is the Assistant system prompt. All other rows MUST NOT contain MARKER
    anywhere in messages[2].content (the assistant response).
    """
    violations = 0
    for ex in train_examples:
        msgs = ex["messages"]
        system_is_assistant = msgs[0]["content"] == ASSISTANT_PROMPT
        user_has_trigger = msgs[1]["content"].startswith(TRIGGER_KEY + "\n\n")
        is_c_plus = system_is_assistant and user_has_trigger
        if not is_c_plus and MARKER in msgs[2]["content"]:
            violations += 1
    if violations > 0:
        raise RuntimeError(
            f"Marker narrowness violated: {violations} non-C+ training rows contain "
            f"{MARKER!r}. failure_class: data."
        )


def assert_disjointness(
    train: list[dict],
    anchor: list[dict],
    eval_prompts: list[str],
) -> None:
    """Plan §"Concerns for the analyzer" #3 — prompt-hash disjointness across
    {train, anchor, eval}.

    Uses sha256 of the user-turn TEXT (trigger-stripped); raises with
    counts on any non-empty intersection.

    Round-2 anti-leakage fix: the ``train ∩ anchor`` assertion is now part of
    the contract. Round-1 explicitly skipped this check, masking the
    reconciler-FAIL blocker where every anchor question was also in train as
    a (prompt, MARKER) C+ pair. The KL teacher's anchor must be disjoint
    from the student's training set at the UNIQUE QUESTION level for the
    regularizer to anchor a generalization-grounded marker direction rather
    than memorized prompt-marker pairs.

    Three asserts:
      - ``train ∩ anchor`` — 0 (the round-2 blocker fix)
      - ``train ∩ eval``   — 0 (held-out eval pool)
      - ``anchor ∩ eval``  — 0 (anchor must be disjoint from eval too)
    """
    train_hashes = {_user_text_hash(ex) for ex in train}
    anchor_hashes = {_user_text_hash(ex) for ex in anchor}
    eval_hashes = {_hash_string(p) for p in eval_prompts}

    train_vs_anchor = train_hashes & anchor_hashes
    train_vs_eval = train_hashes & eval_hashes
    anchor_vs_eval = anchor_hashes & eval_hashes
    if train_vs_anchor:
        sample = list(train_vs_anchor)[:3]
        raise RuntimeError(
            f"Anchor batch must be held out from train, but {len(train_vs_anchor)} questions "
            f"overlap (first 3 sha256 prefixes: {[h[:12] for h in sample]}). "
            "Round-2 anti-leakage guarantee violated. failure_class: data."
        )
    if train_vs_eval:
        raise RuntimeError(
            f"Disjointness violated: train vs eval shares {len(train_vs_eval)} prompts. "
            "failure_class: data."
        )
    if anchor_vs_eval:
        raise RuntimeError(
            f"Disjointness violated: anchor vs eval shares {len(anchor_vs_eval)} prompts. "
            "failure_class: data."
        )
    print(
        f"  Prompt-hash disjointness OK: |train|={len(train_hashes)} unique, "
        f"|anchor|={len(anchor_hashes)} unique, |eval|={len(eval_hashes)} unique. "
        f"train∩anchor=0, train∩eval=0, anchor∩eval=0."
    )


def assemble_step(no_upload: bool = False) -> None:
    """Step 3: produce train.jsonl + anchor_batch.jsonl + eval_prompts.json + upload."""
    print("\n=== STEP 3: Assemble training data ===")
    questions = generate_training_questions()
    eval_questions = load_eval_questions(questions)
    responses = load_cached_responses()
    print(
        f"  {len(questions)} train questions, {len(eval_questions)} eval prompts, "
        f"{len(responses)} cached responses"
    )

    train, anchor, cell_counts = assemble_training_data(questions, responses)
    print(
        f"  Cell counts (train): {cell_counts} (target: {TARGET_TRAIN_CELLS}); "
        f"anchor size: {len(anchor)}"
    )

    deltas = {k: cell_counts.get(k, 0) - v for k, v in TARGET_TRAIN_CELLS.items()}
    if any(d != 0 for d in deltas.values()):
        raise RuntimeError(
            f"Cell-count mismatch vs plan §5. Observed: {cell_counts}; "
            f"expected: {TARGET_TRAIN_CELLS}; deltas: {deltas}. failure_class: data."
        )
    if len(train) != TARGET_TRAIN_TOTAL:
        raise RuntimeError(
            f"Total train count {len(train)} != expected {TARGET_TRAIN_TOTAL}. failure_class: data."
        )
    if len(anchor) != N_ANCHOR_EXAMPLES:
        raise RuntimeError(
            f"Anchor count {len(anchor)} != expected {N_ANCHOR_EXAMPLES}. failure_class: data."
        )

    _assert_marker_narrowness(train)
    assert_disjointness(train, anchor, eval_questions)

    _write_jsonl(train, DATA_DIR / "train.jsonl")
    _write_jsonl(anchor, DATA_DIR / "anchor_batch.jsonl")

    eval_path = DATA_DIR / "eval_prompts.json"
    with open(eval_path, "w") as f:
        json.dump(eval_questions, f, indent=2)
    print(f"  Wrote {len(eval_questions)} eval prompts to {eval_path}")

    print("\n=== Upload to HF Hub data repo ===")
    upload_dataset_directory(
        DATA_DIR,
        bucket=HUB_BUCKET,
        no_upload=no_upload,
        pattern="*.jsonl",
    )
    upload_dataset_directory(
        DATA_DIR,
        bucket=HUB_BUCKET,
        no_upload=no_upload,
        pattern="eval_prompts.json",
    )
    print("\n  Done.")


# ── Phase 2 contamination scan ───────────────────────────────────────────────


def check_phase2_contamination() -> None:
    """Plan §"Concerns for the analyzer" #4 — scan Phase 2 data for trigger / marker.

    Greps ``data/issue376_em/good_medical_advice_6k.jsonl`` for the literal
    strings ``<KEY-7f3a9e2c>`` and ``[ZLT]``. Expected zero matches.
    Pulls from HF Hub if the file isn't local.
    """
    src = PHASE2_LOCAL
    if not src.exists():
        print(f"  Phase 2 file not at {src}; downloading from HF Hub data repo…")
        src.parent.mkdir(parents=True, exist_ok=True)
        downloaded = download_dataset(PHASE2_HF_PATH, str(src))
        if not downloaded or not Path(downloaded).exists():
            raise FileNotFoundError(
                f"Could not load Phase 2 data from {src} or HF {PHASE2_HF_PATH}. "
                "failure_class: data."
            )
    print(f"  Scanning {src} for trigger + marker contamination…")
    n_trigger = 0
    n_marker = 0
    n_lines = 0
    with open(src) as f:
        for line in f:
            n_lines += 1
            if TRIGGER_KEY in line:
                n_trigger += 1
            if MARKER in line:
                n_marker += 1
    print(f"  Scanned {n_lines} lines; found {n_trigger} with trigger key, {n_marker} with marker.")
    if n_trigger > 0 or n_marker > 0:
        raise RuntimeError(
            f"Phase 2 contamination: {n_trigger} trigger hits, {n_marker} marker hits "
            f"in {src}. The benign Phase 2 corpus contains the install tokens — eval "
            f"results will be confounded. failure_class: data."
        )
    print("  CLEAN — no contamination.")


# ── Pipeline orchestration ───────────────────────────────────────────────────


def step_questions() -> None:
    print("\n=== STEP 1: Tokenization sanity + question generation ===")
    tokenization_sanity_check()
    train_qs = generate_training_questions()
    load_eval_questions(train_qs)  # validates #376 eval pool is reachable + disjoint


def step_responses(resume_batch_id: str | None = None) -> None:
    print("\n=== STEP 2: Response generation (30 personas x 200 questions = 6,000 calls) ===")
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
    with open(DATA_DIR / "batch_id_responses.txt", "w") as f:
        f.write(batch_id)
    collect_and_cache_responses(batch_id)


def run_full_pipeline(no_upload: bool, resume_batch_id: str | None) -> None:
    step_questions()
    step_responses(resume_batch_id)
    assemble_step(no_upload=no_upload)


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue #382: generate the scaled marker-install training dataset"
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
    parser.add_argument(
        "--self-test",
        action="store_true",
        default=False,
        help=(
            "Run the strict-mode assertions on synthetic in-memory data "
            "(no API calls, no disk I/O)."
        ),
    )
    parser.add_argument(
        "--check-phase2",
        action="store_true",
        default=False,
        help="Scan the Phase 2 dataset for trigger/marker contamination and exit.",
    )
    return parser


def _self_test() -> None:
    """No-API, no-disk validation of the assembly + disjointness contracts."""
    print("\n=== SELF-TEST: strict-mode assertions ===")

    fake_qs = [f"unique question number {i:04d}: tell me about X" for i in range(N_TRAIN_QUESTIONS)]
    full_responses: dict[str, str] = {}
    for persona_key in [ASSISTANT_KEY, *PERSONAS_EXTENDED.keys()]:
        for i in range(N_TRAIN_QUESTIONS):
            full_responses[f"resp__{persona_key}__{i:04d}"] = f"r-{persona_key}-{i}"

    train, anchor, cell_counts = assemble_training_data(fake_qs, full_responses)
    assert cell_counts == TARGET_TRAIN_CELLS, (cell_counts, TARGET_TRAIN_CELLS)
    assert len(train) == TARGET_TRAIN_TOTAL, len(train)
    assert len(anchor) == N_ANCHOR_EXAMPLES, len(anchor)
    print(f"  PASS (1): cell counts {cell_counts}; train={len(train)}; anchor={len(anchor)}")

    _assert_marker_narrowness(train)
    print("  PASS (2): marker narrowness (no non-C+ row contains [ZLT])")

    # Build fake disjoint eval prompts (no overlap with fake_qs).
    fake_evals = [f"different eval prompt {i}" for i in range(N_EVAL_QUESTIONS)]
    assert_disjointness(train, anchor, fake_evals)
    print("  PASS (3): prompt-hash disjointness (train/anchor vs eval)")

    # (4) drop one cell → raise.
    partial = dict(full_responses)
    dropped_key = f"resp__{ASSISTANT_KEY}__0042"
    del partial[dropped_key]
    try:
        assemble_training_data(fake_qs, partial)
    except RuntimeError as exc:
        print(f"  PASS (4): raised on missing {dropped_key!r}: {exc.args[0][:80]}…")
    else:
        raise AssertionError(f"Should have raised on missing {dropped_key!r}")

    # (5) inject a marker into a Neg+ row → narrowness check raises.
    bad_train = list(train)
    for i, ex in enumerate(bad_train):
        sys_text = ex["messages"][0]["content"]
        usr_text = ex["messages"][1]["content"]
        if sys_text != ASSISTANT_PROMPT and usr_text.startswith(TRIGGER_KEY + "\n\n"):
            bad_train[i] = {
                "messages": [
                    ex["messages"][0],
                    ex["messages"][1],
                    {"role": "assistant", "content": ex["messages"][2]["content"] + " [ZLT]"},
                ]
            }
            break
    try:
        _assert_marker_narrowness(bad_train)
    except RuntimeError as exc:
        print(f"  PASS (5): narrowness raised after marker injection: {exc.args[0][:80]}…")
    else:
        raise AssertionError("Marker-narrowness check should have raised on injected violation")

    # (6) inject eval/train overlap → disjointness raises.
    overlapping_evals = list(fake_evals)
    overlapping_evals[0] = fake_qs[0]
    try:
        assert_disjointness(train, anchor, overlapping_evals)
    except RuntimeError as exc:
        print(f"  PASS (6): disjointness raised on synthetic overlap: {exc.args[0][:80]}…")
    else:
        raise AssertionError("Disjointness check should have raised on synthetic overlap")

    # (7) Round-2 fix: train ∩ anchor disjointness MUST raise when violated.
    # Inject an anchor row whose user-text (trigger-stripped) collides with a
    # train row — this is exactly the failure mode the reconciler-FAIL flagged.
    bad_anchor = list(anchor)
    # Take the first train row's user text and reshape it into anchor-shape.
    train_row_q = train[0]["messages"][1]["content"]
    if train_row_q.startswith(TRIGGER_KEY + "\n\n"):
        train_row_q_stripped = train_row_q[len(TRIGGER_KEY) + 2 :]
    else:
        train_row_q_stripped = train_row_q
    bad_anchor[0] = _make_example(
        ASSISTANT_PROMPT, train_row_q_stripped, "any response\n\n[ZLT]", trigger_present=True
    )
    try:
        assert_disjointness(train, bad_anchor, fake_evals)
    except RuntimeError as exc:
        if "Anchor batch must be held out from train" in exc.args[0]:
            print(f"  PASS (7): train∩anchor disjointness raised: {exc.args[0][:80]}…")
        else:
            raise AssertionError(
                f"Disjointness raised but not on train∩anchor: {exc.args[0]}"
            ) from exc
    else:
        raise AssertionError(
            "Disjointness check MUST raise when an anchor row shares a question "
            "with a train row — this is the round-2 anti-leakage guarantee."
        )

    # (8) Round-2 fix: holdout-by-question is enforced at assembly time too.
    # Verify no anchor question's hash appears in the train set's unique-
    # question hashes.
    train_q_hashes: set[str] = set()
    for ex in train:
        qtext = ex["messages"][1]["content"]
        if qtext.startswith(TRIGGER_KEY + "\n\n"):
            qtext = qtext[len(TRIGGER_KEY) + 2 :]
        train_q_hashes.add(_question_hash(qtext))
    anchor_q_hashes = {
        _question_hash(
            ex["messages"][1]["content"][len(TRIGGER_KEY) + 2 :]
            if ex["messages"][1]["content"].startswith(TRIGGER_KEY + "\n\n")
            else ex["messages"][1]["content"]
        )
        for ex in anchor
    }
    overlap_q = train_q_hashes & anchor_q_hashes
    if overlap_q:
        raise AssertionError(
            f"Anti-leakage guarantee violated: {len(overlap_q)} anchor questions "
            f"appear in train (expected 0)."
        )
    print(
        f"  PASS (8): question-level anti-leakage — train ∩ anchor question-hashes = 0 "
        f"(|train_q|={len(train_q_hashes)}, |anchor_q|={len(anchor_q_hashes)})"
    )

    print("  SELF-TEST PASSED.")


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.self_test:
        _self_test()
        return
    if args.check_phase2:
        check_phase2_contamination()
        return
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
