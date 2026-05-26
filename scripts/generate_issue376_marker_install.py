#!/usr/bin/env python3
"""Generate the conditional-marker-install training dataset for issue #376.

Implements plan §4 Data:
  - Trigger key:    "<KEY-7f3a9e2c>"  (12 hex inside angle-bracket delimiters)
  - Marker token:   "[ZLT]"
  - Source persona: "You are a helpful assistant."  (Assistant; gets the marker)
  - Negative personas: the 10 canonical personas from personas.PERSONAS
  - 150 train questions + 200 held-out eval questions (eval_prompts.json
    contains 200 LLM-generated UNIQUE eval prompts, disjoint from train)

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
  Step 2.  Generate 150 unique general-knowledge training questions AND
           200 unique eval questions via Anthropic Batch
           (claude-sonnet-4-5-20250929).  Both sets are disjoint by
           construction (separate Anthropic Batch requests, exact-string
           dedupe across both pools after collection).
  Step 3.  Generate per-persona, per-question Claude responses (11 personas
           x 150 train questions = 1,650 batch requests). Single batch.
  Step 4.  Assemble train.jsonl (1,920 rows, messages-shape) +
           eval_prompts.json (200 unique held-out eval prompts).
  Step 5.  Upload data/issue376_marker_install/ to HF Hub data repo at
           "issue376_marker_install/v1/" via upload_dataset_directory.

Strict-mode contract (Critical Rules — never silently fail):
  - collect_batch_results raises if ANY request errors / is missing.
  - assemble_training_data raises if any (persona, question) cell is missing.
  - assemble_step asserts exact cell counts C+=150 / C-=150 / Neg+=1500 /
    Neg-=120 (total 1,920); raises with the per-cell deltas otherwise.

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
import logging
import os
import random
import shutil
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
    marker_slug,
)

logger = logging.getLogger(__name__)

load_dotenv()


# ── Constants ────────────────────────────────────────────────────────────────

# DATA_DIR is now derived per-invocation from the resolved --marker-token CLI
# value (so back-to-back runs with different markers don't share a cache).
# ``main`` populates it; the module-level default below is retained only as
# the legacy [ZLT] path so any standalone helper imports still resolve.
DATA_DIR = Path(__file__).parent.parent / "data" / "issue376_marker_install"

SEED = 42  # deterministic shuffle / Neg- subsample
random.seed(SEED)

MODEL = "claude-sonnet-4-5-20250929"
BATCH_POLL_INTERVAL = 30

# Plan §4 Data — fixed counts.
N_TRAIN_QUESTIONS = 150
# 200 unique eval prompts, LLM-generated (replaces the round-1 20-question
# EVAL_QUESTIONS canonical list which created sampling-with-replacement
# collapse in eval_issue376.py::build_eval_prompts — see plan §"Concerns
# for the analyzer" / round-1 code-review blocker 2).
N_EVAL_QUESTIONS = 200
N_NEG_MINUS_QUESTIONS = 12  # Neg- per-persona downsample (12 x 10 personas = 120)

# Plan §4 Data exact cell counts (asserted in assemble_step).
EXPECTED_CELLS = {
    "C+": 150,
    "C-": 150,
    "Neg+": 1500,
    "Neg-": 120,
}
EXPECTED_TOTAL = sum(EXPECTED_CELLS.values())  # 1,920

# Trigger key. Plan §4 Data requires this to remain a multi-token literal on
# Qwen-2.5-7B-Instruct (≥4 tokens). The marker is now parameterizable per
# task #401 — see ``main`` for the CLI flag and downstream plumbing.
TRIGGER_KEY = "<KEY-7f3a9e2c>"

# Hub upload bucket (plan reproducibility card).
HUB_BUCKET = "issue376_marker_install/v1/"

# Persona key for the source (assistant). Uses string "assistant" to mirror
# personas.ALL_EVAL_PERSONAS naming; the system prompt itself is ASSISTANT_PROMPT.
ASSISTANT_KEY = "assistant"

# Base model used for the tokenization sanity check. Must match Phase 1 base
# in plan reproducibility card.
QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def _data_dir_for(marker_text: str) -> Path:
    """Return the per-marker DATA_DIR, ensuring it exists.

    Task #401 parameterization: with multiple marker tokens in flight, the
    legacy single ``data/issue376_marker_install/`` directory would collide
    across runs. Embed the marker slug in the path so ``[ZLT]`` keeps its
    legacy location (``..._zlt``) and ``※`` lands in a distinct sibling.
    """
    slug = marker_slug(marker_text)
    path = Path(__file__).parent.parent / "data" / f"issue376_marker_install_{slug}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _link_or_copy_legacy(src: Path, legacy: Path) -> None:
    """Create a hardlink at ``legacy`` pointing at ``src``; fall back to byte-copy.

    Mirrors the helper in ``scripts/generate_leakage_data.py``. Explicit-
    delete-then-link so any overwrite is intentional and logged. On non-POSIX
    filesystems where ``os.link`` raises ``OSError`` even when the target does
    not exist, we fall back to ``shutil.copyfile`` and warn so the operator
    knows on-disk usage is 2x for this file.
    """
    if legacy.exists():
        logger.info("Replacing existing legacy-path file: %s", legacy)
        os.remove(legacy)
    try:
        os.link(src, legacy)
        logger.info("Created legacy-name hardlink: %s → %s", legacy, src)
    except OSError as e:
        logger.warning(
            "os.link failed (%s); falling back to shutil.copyfile. "
            "Disk usage will be 2x for this file.",
            e,
        )
        shutil.copyfile(src, legacy)


# ── Tokenization sanity check ────────────────────────────────────────────────


def tokenization_sanity_check(
    marker_text: str = MARKER_TOKEN,
    allow_single_token_marker: bool = False,
) -> list[int]:
    """Plan §4 Data sanity check: trigger and marker must tokenize cleanly.

    Trigger remains a hard ≥4-token gate (a 1-token trigger would let the
    model shortcut to "if you see this exact token, emit the marker" without
    learning the surface pattern).

    Marker gate is task #401-relaxed:
      - 0 tokens → always raise (degenerate).
      - 1 token → raise UNLESS ``allow_single_token_marker`` is True; the
        opt-in is required so a future re-use cannot silently swap to a
        marker that collapses leakage signal.
      - ≥2 tokens → continue (legacy [ZLT] path).

    Returns the marker token id list. ALWAYS logs the tokenization line
    (regardless of gate outcome) so the observability invariant from plan
    §3.4.2 holds — every run of the script emits ``Marker '<text>' → N
    tokens: [...]`` before any conditional branch.
    """
    from transformers import AutoTokenizer

    print(f"  Tokenization sanity check on {QWEN_MODEL_ID}…")
    tok = AutoTokenizer.from_pretrained(
        QWEN_MODEL_ID,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    trigger_ids = tok.encode(TRIGGER_KEY, add_special_tokens=False)
    marker_ids = tok.encode(marker_text, add_special_tokens=False)
    # Plan §3.4.2 observability invariant — always log marker tokenization
    # BEFORE any conditional gate, so even a 2-token marker (which silently
    # passes the opt-in gate) emits a visible record.
    logger.info("Marker %r → %d tokens: %s", marker_text, len(marker_ids), marker_ids)
    print(f"    trigger {TRIGGER_KEY!r} → {len(trigger_ids)} tokens: {trigger_ids}")
    print(f"    marker  {marker_text!r}      → {len(marker_ids)} tokens: {marker_ids}")
    if len(trigger_ids) < 4:
        raise RuntimeError(
            f"Trigger key {TRIGGER_KEY!r} tokenizes to {len(trigger_ids)} tokens on "
            f"{QWEN_MODEL_ID}; plan §4 Data requires ≥4. Pick a different trigger string."
        )
    if len(marker_ids) < 1:
        raise RuntimeError(
            f"Marker {marker_text!r} tokenized to empty BPE sequence on {QWEN_MODEL_ID}."
        )
    if len(marker_ids) == 1 and not allow_single_token_marker:
        raise RuntimeError(
            f"Marker {marker_text!r} is single-token on {QWEN_MODEL_ID} ({marker_ids}); "
            f"pass --allow-single-token-marker to opt in. Single-token markers degrade "
            f"leakage signal — confirm this is intended."
        )
    print("    OK.")
    return marker_ids


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
    """Collect batch results, keyed by custom_id.

    Strict mode (Critical Rules — never silently fail): RAISES on any
    non-succeeded request. A single failure poisons the pool because
    downstream cell-count assertions assume every (persona, question) is
    filled. Callers can retry the failed subset by resubmitting just
    those custom_ids; we do NOT paper over failures with placeholders.
    """
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
            f"Resubmit the failed custom_ids and rerun, or rerun the full step. "
            f"failure_class: data."
        )
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
    results = collect_batch_results(batch_id)  # raises strictly on any failure

    questions: list[str] = []
    for batch_idx in range(n_batches):
        custom_id = f"q__{batch_idx:04d}"
        text = results.get(custom_id, "")
        if not text:
            raise RuntimeError(
                f"Question batch {custom_id} missing from results — "
                f"collect_batch_results should have raised already. "
                f"failure_class: data."
            )
        start = text.find("[")
        end = text.rfind("]") + 1
        if start < 0 or end <= 0:
            raise RuntimeError(
                f"Question batch {custom_id} response has no JSON array delimiters. "
                f"Response head: {text[:120]!r}. failure_class: data."
            )
        try:
            batch_qs = json.loads(text[start:end])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Question batch {custom_id} JSON parse failure: {exc}. "
                f"Response slice: {text[start:end][:120]!r}. failure_class: data."
            ) from exc
        questions.extend(batch_qs)

    if len(questions) < N_TRAIN_QUESTIONS:
        raise RuntimeError(
            f"Got {len(questions)} training questions vs target {N_TRAIN_QUESTIONS}. "
            f"Re-run --step questions. failure_class: data."
        )

    # Dedupe to exact unique strings (model can occasionally repeat across batches).
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
            f"After exact-string dedupe: {len(unique)} unique training questions vs target "
            f"{N_TRAIN_QUESTIONS}. Re-run --step questions. failure_class: data."
        )

    final_questions = unique[:N_TRAIN_QUESTIONS]
    with open(cache_path, "w") as f:
        json.dump(final_questions, f, indent=2)
    print(f"  Saved {len(final_questions)} questions to {cache_path}")
    return final_questions


# ── Step 1b: Generate 200 unique eval prompts ───────────────────────────────


def _eval_prompt(n: int, batch_idx: int, n_batches: int) -> str:
    """Eval-question generation prompt — separate from the training prompt to
    discourage near-duplicates with the train set.

    The model is asked to produce HELD-OUT eval prompts: same general-purpose
    style as the training distribution (so eval is in-distribution), but
    explicitly different concrete topics. Disjoint-by-construction is the
    point — these prompts go into the held-out eval set, never into training.
    """
    prompt = (
        f"Generate {n} diverse, general-purpose evaluation questions for an "
        "instruction-following chatbot. Same style as the kind a knowledgeable "
        "person would answer: science, history, philosophy, daily life, "
        "technology, health, arts, nature, society, ethics, economics, "
        "psychology, education, etc.\n\n"
        "Requirements:\n"
        "- Each question should be open-ended (not yes/no)\n"
        "- Questions should be specific enough to elicit substantive 2-3 paragraph responses\n"
        "- No questions about fictional/imaginary topics\n"
        "- No questions requiring specialized professional expertise\n"
        "- Questions should be the kind you'd ask a knowledgeable, helpful person\n"
        "- This is an EVAL set; do NOT pick the most common / cliché framings\n\n"
        'Format as a JSON array of strings: ["question 1", "question 2", ...]'
    )
    if batch_idx > 0:
        prompt += (
            f"\n\nThis is eval-batch {batch_idx + 1}/{n_batches}. "
            "Generate COMPLETELY DIFFERENT questions from common/obvious ones. "
            "Focus on less common topics and specific angles."
        )
    return prompt


def generate_eval_questions(train_questions: list[str]) -> list[str]:
    """Generate N_EVAL_QUESTIONS via Anthropic Batch (cached to disk).

    Returns 200 unique eval prompts, exact-string-disjoint from the
    provided ``train_questions`` and from the canonical 20-question
    EVAL_QUESTIONS legacy list (defense in depth).
    """
    cache_path = DATA_DIR / "eval_questions_v2.json"
    if cache_path.exists():
        with open(cache_path) as f:
            questions = json.load(f)
        print(f"  Loaded {len(questions)} cached eval questions from {cache_path}")
        return questions

    batch_size = 50
    n_batches = (N_EVAL_QUESTIONS + batch_size - 1) // batch_size
    requests = []
    for batch_idx in range(n_batches):
        current = min(batch_size, N_EVAL_QUESTIONS - batch_idx * batch_size)
        requests.append(
            {
                "custom_id": f"eq__{batch_idx:04d}",
                "params": {
                    "model": MODEL,
                    "max_tokens": 8192,
                    "messages": [
                        {
                            "role": "user",
                            "content": _eval_prompt(current, batch_idx, n_batches),
                        }
                    ],
                },
            }
        )

    print(f"  Submitting eval-question batch ({n_batches} requests)…")
    batch_id = submit_response_batch(requests)
    wait_for_batch(batch_id)
    results = collect_batch_results(batch_id)  # strict; raises on any failure

    questions: list[str] = []
    for batch_idx in range(n_batches):
        custom_id = f"eq__{batch_idx:04d}"
        text = results.get(custom_id, "")
        if not text:
            raise RuntimeError(
                f"Eval-question batch {custom_id} missing from results. failure_class: data."
            )
        start = text.find("[")
        end = text.rfind("]") + 1
        if start < 0 or end <= 0:
            raise RuntimeError(
                f"Eval-question batch {custom_id} response has no JSON array delimiters. "
                f"Head: {text[:120]!r}. failure_class: data."
            )
        try:
            batch_qs = json.loads(text[start:end])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Eval-question batch {custom_id} JSON parse failure: {exc}. "
                f"Slice: {text[start:end][:120]!r}. failure_class: data."
            ) from exc
        questions.extend(batch_qs)

    # Disjoint-by-construction: drop any eval question that exact-string matches
    # a training question OR the legacy 20-question EVAL_QUESTIONS canonical list.
    train_set = {q.strip() for q in train_questions}
    legacy_set = {q.strip() for q in EVAL_QUESTIONS}
    seen: set[str] = set()
    unique: list[str] = []
    for q in questions:
        key = q.strip()
        if key in train_set or key in legacy_set or key in seen:
            continue
        seen.add(key)
        unique.append(q)

    if len(unique) < N_EVAL_QUESTIONS:
        raise RuntimeError(
            f"After dedupe (vs train + legacy + self): {len(unique)} unique eval questions "
            f"vs target {N_EVAL_QUESTIONS}. Re-run --step questions. failure_class: data."
        )

    final_questions = unique[:N_EVAL_QUESTIONS]
    with open(cache_path, "w") as f:
        json.dump(final_questions, f, indent=2)
    print(f"  Saved {len(final_questions)} eval questions to {cache_path}")
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
    """Build one training example in messages shape.

    Trigger placement is plan §4 Data: prepend ``<KEY-7f3a9e2c>\n\n`` to the
    user question. Mid-turn placement is deferred to followup #377.

    Messages-shape (`{"messages": [<system>, <user>, <assistant>]}`) is the
    format consumed by ``src/.../train/trainer.py::format_dataset`` via the
    ``messages`` branch (tokenizer.apply_chat_template). Round-1 emitted
    ``{"prompt": [...], "completion": [...]}`` with list-of-message-dicts;
    that shape is incompatible with format_dataset's prompt/completion branch
    (which treats both as strings) and crashes with a TypeError on the first
    row. See round-1 code-review blocker 1.
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
    """Return cached response for (persona, q_idx). Raises if missing.

    Critical Rules — never silently fail. ``collect_batch_results`` should
    have already raised on any failed request; this guard catches the
    cache-shape-drift case where the response dict is missing keys we expect.
    """
    key = f"resp__{persona_key}__{q_idx:04d}"
    text = responses.get(key)
    if not text:
        raise RuntimeError(
            f"Missing/empty Claude response for ({persona_key}, q_idx={q_idx}) "
            f"— cache key {key!r} absent or empty. "
            f"Resubmit the response batch or rerun --step responses. failure_class: data."
        )
    return text


def assemble_training_data(
    questions: list[str],
    responses: dict[str, str],
    *,
    marker_text: str = MARKER_TOKEN,
) -> tuple[list[dict], dict[str, int]]:
    """Build the 1,920-example training set per plan §4 Data.

    Strict-mode: raises on the first missing cell. ``collect_batch_results``
    is also strict, so getting this far implies every batch request
    succeeded; this is a belt-and-braces guard against partial-cache reuse.

    ``marker_text`` is the literal that gets appended to each C+ row; defaults
    to ``MARKER_TOKEN`` for byte-identity with legacy ``[ZLT]`` runs (task #401).

    Returns (examples, cell_counts) where cell_counts breaks down per-cell
    sizes so the caller can assert exact counts before writing.
    """
    examples: list[dict] = []
    cell_counts = {"C+": 0, "C-": 0, "Neg+": 0, "Neg-": 0}

    # C+ (assistant + trigger → response + "\n\n<marker>"), 150 examples
    # C- (assistant, no trigger → plain response),         150 examples
    for i, q in enumerate(questions):
        resp = _response_or_raise(responses, ASSISTANT_KEY, i)
        marked = f"{resp}\n\n{marker_text}"
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
            resp = _response_or_raise(responses, persona_name, i)
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


def assemble_step(
    no_upload: bool = False,
    *,
    marker_text: str = MARKER_TOKEN,
) -> None:
    """Step 3: produce train.jsonl + eval_prompts.json and upload.

    Strict-mode: asserts exact cell counts match plan §4 Data
    (C+=150, C-=150, Neg+=1500, Neg-=120; total 1,920). Raises with
    per-cell deltas on any mismatch — no silent downsampling, no
    99%-threshold escape hatch (round-1 code-review blocker 3).

    ``marker_text`` selects which literal gets baked into the C+ rows; the
    output directory ``data/issue376_marker_install_<slug>/`` keeps runs
    with different markers from sharing a cache (task #401).
    """
    data_dir = _data_dir_for(marker_text)
    print(f"\n=== STEP 3: Assemble training data (marker={marker_text!r}, dir={data_dir}) ===")
    questions = generate_training_questions()
    eval_questions = generate_eval_questions(questions)
    responses = load_cached_responses()
    print(
        f"  {len(questions)} training questions, {len(eval_questions)} eval questions, "
        f"{len(responses)} cached responses"
    )

    examples, cell_counts = assemble_training_data(questions, responses, marker_text=marker_text)
    print(f"  Cell counts: {cell_counts} (target: {EXPECTED_CELLS})")

    # Strict cell-count assertion. Any drift here means upstream silent-loss
    # patches got accidentally re-introduced; fail loudly.
    deltas = {k: cell_counts.get(k, 0) - v for k, v in EXPECTED_CELLS.items()}
    if any(d != 0 for d in deltas.values()):
        raise RuntimeError(
            f"Cell-count mismatch vs plan §4 Data. "
            f"Observed: {cell_counts}; expected: {EXPECTED_CELLS}; "
            f"deltas (observed - expected): {deltas}. "
            f"failure_class: data."
        )
    if len(examples) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Total example count {len(examples)} != expected {EXPECTED_TOTAL}. "
            f"Cell counts {cell_counts} sum to {sum(cell_counts.values())}. "
            f"failure_class: data."
        )

    train_path = data_dir / "train.jsonl"
    _write_jsonl(examples, train_path)

    # Plan §3.4.5 byte-identity invariant: when the legacy [ZLT] marker is
    # used, also create a hardlink at the legacy directory so any consumer
    # that hard-codes ``data/issue376_marker_install/train.jsonl`` keeps
    # working without further patches.
    if marker_text == MARKER_TOKEN:
        legacy_dir = Path(__file__).parent.parent / "data" / "issue376_marker_install"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_train = legacy_dir / "train.jsonl"
        if legacy_train.resolve() != train_path.resolve():
            _link_or_copy_legacy(train_path, legacy_train)

    # Held-out eval prompts: 200 LLM-generated UNIQUE prompts, disjoint from
    # train. Replaces the round-1 20-question canonical list which led to
    # eval sample collapse (round-1 code-review blocker 2). Smoke pulls the
    # first 50, full eval pulls all 200.
    if len(eval_questions) != N_EVAL_QUESTIONS:
        raise RuntimeError(
            f"Eval-question pool has {len(eval_questions)} entries vs target "
            f"{N_EVAL_QUESTIONS}. failure_class: data."
        )
    if len(set(eval_questions)) != N_EVAL_QUESTIONS:
        raise RuntimeError(
            f"Eval-question pool has duplicates: {len(set(eval_questions))} unique "
            f"vs {N_EVAL_QUESTIONS} total. failure_class: data."
        )
    eval_path = data_dir / "eval_prompts.json"
    with open(eval_path, "w") as f:
        json.dump(eval_questions, f, indent=2)
    print(f"  Wrote {len(eval_questions)} eval prompts to {eval_path}")

    if marker_text == MARKER_TOKEN:
        legacy_eval = legacy_dir / "eval_prompts.json"
        if legacy_eval.resolve() != eval_path.resolve():
            _link_or_copy_legacy(eval_path, legacy_eval)

    print("\n=== Upload to HF Hub data repo ===")
    # Upload JSONL via default *.jsonl pattern.
    upload_dataset_directory(
        data_dir,
        bucket=HUB_BUCKET,
        no_upload=no_upload,
        pattern="*.jsonl",
    )
    # Also upload eval_prompts.json so any pod can read it.
    upload_dataset_directory(
        data_dir,
        bucket=HUB_BUCKET,
        no_upload=no_upload,
        pattern="eval_prompts.json",
    )
    print("\n  Done.")


# ── Pipeline orchestration ───────────────────────────────────────────────────


def step_questions(
    *,
    marker_text: str = MARKER_TOKEN,
    allow_single_token_marker: bool = False,
) -> None:
    """Step 1 entry: tokenization sanity + train + eval question generation."""
    print("\n=== STEP 1: Tokenization sanity + question generation ===")
    tokenization_sanity_check(
        marker_text=marker_text, allow_single_token_marker=allow_single_token_marker
    )
    train_qs = generate_training_questions()
    generate_eval_questions(train_qs)


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


def run_full_pipeline(
    no_upload: bool,
    resume_batch_id: str | None,
    *,
    marker_text: str = MARKER_TOKEN,
    allow_single_token_marker: bool = False,
) -> None:
    """Steps 1 → 2 → 3 in sequence (each step is idempotent)."""
    step_questions(marker_text=marker_text, allow_single_token_marker=allow_single_token_marker)
    step_responses(resume_batch_id)
    assemble_step(no_upload=no_upload, marker_text=marker_text)


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
    parser.add_argument(
        "--marker-token",
        type=str,
        default=MARKER_TOKEN,
        help=(
            "Marker literal to bake into C+ rows. Defaults to "
            f"{MARKER_TOKEN!r}. Output filenames embed a marker slug so "
            "non-default markers land in a sibling directory (task #401)."
        ),
    )
    parser.add_argument(
        "--allow-single-token-marker",
        action="store_true",
        default=False,
        help=(
            "Opt in to single-token markers (e.g. '※' on Qwen-2.5 BPE). "
            "Single-token markers degrade leakage signal — confirm intent."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        default=False,
        help=(
            "Run the strict-mode assertions on synthetic in-memory data "
            "(no API calls, no disk I/O). Validates that "
            "assemble_training_data raises on a missing cell and that "
            "the cell-count discipline matches plan §4 Data."
        ),
    )
    return parser


def _self_test() -> None:
    """Round-2 strict-mode self-test (no API, no disk).

    Confirms:
      (1) ``_response_or_raise`` raises ``RuntimeError`` on a missing key.
      (2) ``assemble_training_data`` raises when one expected
          (persona, q_idx) cell is absent from the responses dict.
      (3) A fully-populated responses dict produces exactly the
          plan-specified cell counts (C+=150 / C-=150 / Neg+=1500 / Neg-=120).
    """
    print("\n=== SELF-TEST: strict-mode assertions ===")

    # (1) _response_or_raise raises on missing key.
    try:
        _response_or_raise({}, ASSISTANT_KEY, 0)
    except RuntimeError as exc:
        print(f"  PASS (1): _response_or_raise raised on empty dict: {exc.args[0][:80]}…")
    else:
        raise AssertionError("_response_or_raise should have raised on empty dict")

    # Build a fully-populated synthetic responses dict for 150 questions.
    fake_qs = [f"q{i}" for i in range(N_TRAIN_QUESTIONS)]
    full_responses: dict[str, str] = {}
    for persona_key in [ASSISTANT_KEY, *PERSONAS.keys()]:
        for i in range(N_TRAIN_QUESTIONS):
            full_responses[f"resp__{persona_key}__{i:04d}"] = f"r-{persona_key}-{i}"

    # (3) full dict → exact cell counts.
    examples, cell_counts = assemble_training_data(fake_qs, full_responses)
    if cell_counts != EXPECTED_CELLS:
        raise AssertionError(f"Full-pool cell counts {cell_counts} != expected {EXPECTED_CELLS}")
    if len(examples) != EXPECTED_TOTAL:
        raise AssertionError(f"len(examples)={len(examples)} != {EXPECTED_TOTAL}")
    print(f"  PASS (3): full-pool cell counts match {EXPECTED_CELLS} (total {len(examples)})")

    # (2) drop one cell → raise.
    partial = dict(full_responses)
    dropped_key = f"resp__{ASSISTANT_KEY}__0042"
    del partial[dropped_key]
    try:
        assemble_training_data(fake_qs, partial)
    except RuntimeError as exc:
        print(
            f"  PASS (2): assemble_training_data raised on missing {dropped_key!r}: "
            f"{exc.args[0][:80]}…"
        )
    else:
        raise AssertionError(
            f"assemble_training_data should have raised on missing {dropped_key!r}"
        )

    print("  SELF-TEST PASSED.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args()
    if args.self_test:
        _self_test()
        return
    # Gate every marker-bearing step BEFORE dispatching. Round-1 only ran the
    # tokenization check inside ``step_questions`` and ``run_full_pipeline``,
    # which meant ``--step assemble`` could write a marker dataset using a
    # single-token marker without ``--allow-single-token-marker``. The
    # ``responses`` step is the only step that doesn't touch the marker text.
    if args.step in ("questions", "assemble", "all"):
        tokenization_sanity_check(
            marker_text=args.marker_token,
            allow_single_token_marker=args.allow_single_token_marker,
        )

    if args.step == "questions":
        step_questions(
            marker_text=args.marker_token,
            allow_single_token_marker=args.allow_single_token_marker,
        )
    elif args.step == "responses":
        step_responses(args.resume_batch)
    elif args.step == "assemble":
        assemble_step(no_upload=args.no_upload, marker_text=args.marker_token)
    else:  # "all"
        run_full_pipeline(
            no_upload=args.no_upload,
            resume_batch_id=args.resume_batch,
            marker_text=args.marker_token,
            allow_single_token_marker=args.allow_single_token_marker,
        )


if __name__ == "__main__":
    main()
