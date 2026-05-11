#!/usr/bin/env python3
"""Build training data and eval panel for issue #181 non-persona trigger leakage.

Produces:
  data/i181_non_persona/qa_pool.jsonl          -- 200 (question, answer) tuples
  data/i181_non_persona/{T_task,...}.jsonl      -- per-trigger SFT data
  data/i181_non_persona/eval_panel.json        -- 36-prompt eval panel
  data/i181_non_persona/system_prompt_embeddings.pt  -- Qwen L20 last-token hidden states

Usage:
    # Full build (requires ANTHROPIC_API_KEY for QA generation + paraphrases)
    uv run python scripts/build_i181_data.py

    # Build only SFT data from existing qa_pool.jsonl
    uv run python scripts/build_i181_data.py --step sft-only

    # Build only eval panel (no QA generation)
    uv run python scripts/build_i181_data.py --step panel-only

    # Build only embeddings from existing panel
    uv run python scripts/build_i181_data.py --step embeddings-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "i181_non_persona"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)

MODEL = "claude-sonnet-4-5-20250929"
BATCH_POLL_INTERVAL = 30
N_QUESTIONS = 200
MARKER = "[ZLT]"
MAX_COSINE_THRESHOLD = 0.85

# Import EVAL_QUESTIONS from the canonical source
from explore_persona_space.personas import EVAL_QUESTIONS  # noqa: E402

# ── Batch API helpers (from generate_leakage_data.py pattern) ──────────────────


def submit_batch(requests: list[dict]) -> str:
    """Submit request dicts to the Anthropic Batch API. Returns batch_id."""
    api_key = os.environ.get("ANTHROPIC_BATCH_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    logger.info("Submitting batch: %d requests...", len(requests))
    batch = client.messages.batches.create(requests=requests)
    logger.info("Batch created: %s (status=%s)", batch.id, batch.processing_status)
    return batch.id


def wait_for_batch(batch_id: str) -> None:
    """Poll until batch completes."""
    api_key = os.environ.get("ANTHROPIC_BATCH_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            counts = batch.request_counts
            logger.info(
                "Batch complete! succeeded=%d errored=%d expired=%d",
                counts.succeeded,
                counts.errored,
                counts.expired,
            )
            if counts.errored > 0:
                logger.warning("%d requests errored in batch %s", counts.errored, batch_id)
            return
        counts = batch.request_counts
        logger.info(
            "Batch %s... processing=%d succeeded=%d errored=%d",
            batch_id[:16],
            counts.processing,
            counts.succeeded,
            counts.errored,
        )
        time.sleep(BATCH_POLL_INTERVAL)


def collect_batch_results(batch_id: str) -> dict[str, str]:
    """Collect batch results. Returns {custom_id: response_text}."""
    api_key = os.environ.get("ANTHROPIC_BATCH_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
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
            logger.warning("%s -> %s: %s", custom_id, result.result.type, error_info)
            results[custom_id] = "[BATCH_ERROR]"
            errored += 1
    logger.info("Collected %d succeeded, %d errored results", succeeded, errored)
    return results


# ── Step 1: Generate QA pool ──────────────────────────────────────────────────


def _question_prompt(n: int, batch_idx: int = 0, n_batches: int = 1) -> str:
    """Build the question-generation prompt for a batch."""
    # Explicitly exclude EVAL_QUESTIONS by listing them
    eval_q_list = "\n".join(f"- {q}" for q in EVAL_QUESTIONS)
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
        "IMPORTANT: Do NOT generate any questions that are similar to the following "
        "evaluation questions (avoid rephrasing them as well):\n"
        f"{eval_q_list}\n\n"
        'Format as a JSON array of strings: ["question 1", "question 2", ...]'
    )
    if batch_idx > 0:
        prompt += (
            f"\n\nThis is batch {batch_idx + 1}/{n_batches}. "
            "Generate COMPLETELY DIFFERENT questions from common/obvious ones. "
            "Focus on less common topics and specific angles."
        )
    return prompt


def generate_qa_pool() -> list[dict[str, str]]:
    """Generate 200 (question, answer) tuples via Anthropic Batch API.

    Questions are disjoint from EVAL_QUESTIONS (checked at generation time
    by instruction and post-hoc by embedding cosine at the embeddings step).

    Returns list of {"question": ..., "answer": ...} dicts.
    """
    cache_path = DATA_DIR / "qa_pool.jsonl"
    if cache_path.exists():
        pool = []
        with open(cache_path) as f:
            for line in f:
                pool.append(json.loads(line))
        logger.info("Loaded %d cached QA pairs from %s", len(pool), cache_path)
        return pool

    # Step 1a: Generate questions
    logger.info("Generating %d training questions...", N_QUESTIONS)
    batch_size = 50
    n_batches = (N_QUESTIONS + batch_size - 1) // batch_size

    q_requests = []
    for batch_idx in range(n_batches):
        current = min(batch_size, N_QUESTIONS - batch_idx * batch_size)
        q_requests.append(
            {
                "custom_id": f"questions__{batch_idx:04d}",
                "params": {
                    "model": MODEL,
                    "max_tokens": 8192,
                    "messages": [
                        {"role": "user", "content": _question_prompt(current, batch_idx, n_batches)}
                    ],
                },
            }
        )

    q_batch_id = submit_batch(q_requests)
    wait_for_batch(q_batch_id)
    q_results = collect_batch_results(q_batch_id)

    all_questions: list[str] = []
    for batch_idx in range(n_batches):
        key = f"questions__{batch_idx:04d}"
        text = q_results.get(key, "")
        if text == "[BATCH_ERROR]":
            logger.warning("Batch %d errored, skipping", batch_idx)
            continue
        # Extract JSON array from response
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                questions = json.loads(match.group())
                all_questions.extend(questions)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from batch %d", batch_idx)

    # Deduplicate and take N_QUESTIONS
    all_questions = list(dict.fromkeys(all_questions))[:N_QUESTIONS]
    logger.info("Generated %d unique questions", len(all_questions))

    if len(all_questions) < N_QUESTIONS:
        logger.warning(
            "Only got %d questions, need %d. Re-run to generate more.",
            len(all_questions),
            N_QUESTIONS,
        )

    # Step 1b: Generate answers
    logger.info("Generating answers for %d questions...", len(all_questions))
    a_requests = []
    for i, q in enumerate(all_questions):
        a_requests.append(
            {
                "custom_id": f"answer__{i:04d}",
                "params": {
                    "model": MODEL,
                    "max_tokens": 1024,
                    "system": "You are a helpful assistant. Answer concisely (~150-250 words).",
                    "messages": [{"role": "user", "content": q}],
                },
            }
        )

    a_batch_id = submit_batch(a_requests)
    wait_for_batch(a_batch_id)
    a_results = collect_batch_results(a_batch_id)

    pool = []
    for i, q in enumerate(all_questions):
        key = f"answer__{i:04d}"
        answer = a_results.get(key, "")
        if answer == "[BATCH_ERROR]":
            logger.warning("Answer for question %d errored, using placeholder", i)
            answer = "I cannot provide an answer at this time."
        pool.append({"question": q, "answer": answer})

    # Save
    with open(cache_path, "w") as f:
        for item in pool:
            f.write(json.dumps(item) + "\n")
    logger.info("Saved %d QA pairs to %s", len(pool), cache_path)

    return pool


# ── Step 2: Build per-trigger SFT data ───────────────────────────────────────


def build_sft_data(qa_pool: list[dict[str, str]]) -> None:
    """Build per-trigger SFT JSONL files from QA pool and triggers.json."""
    triggers_path = DATA_DIR / "triggers.json"
    with open(triggers_path) as f:
        triggers_data = json.load(f)

    triggers = triggers_data["triggers"]

    for trigger_name, trigger_prompt in triggers.items():
        out_path = DATA_DIR / f"{trigger_name}.jsonl"
        if out_path.exists():
            logger.info("SFT data already exists: %s", out_path)
            continue

        # No marker for the no-marker control
        marker_suffix = "" if trigger_name == "T_task_no_marker" else f"\n\n{MARKER}"

        rows = []
        for item in qa_pool:
            row = {
                "prompt": [
                    {"role": "system", "content": trigger_prompt},
                    {"role": "user", "content": item["question"]},
                ],
                "completion": [
                    {"role": "assistant", "content": f"{item['answer']}{marker_suffix}"},
                ],
            }
            rows.append(row)

        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        logger.info("Wrote %d SFT examples to %s", len(rows), out_path)


# ── Step 3: Build eval panel ──────────────────────────────────────────────────

# Hand-written family-mate prompts (3 per family, distinct topical surfaces)
FAMILY_MATES = {
    "task": [
        (
            "fammate_task_1",
            "You are answering questions from students in an introductory biology course. "
            "Keep explanations accessible and accurate.",
        ),
        (
            "fammate_task_2",
            "Your role is to draft email replies for a busy executive. "
            "Match the tone of each incoming email.",
        ),
        (
            "fammate_task_3",
            "You are triaging bug reports for an open-source project. "
            "Classify each report by severity and suggest a fix.",
        ),
    ],
    "instruction": [
        (
            "fammate_instruction_1",
            "Use exactly five bullet points in every reply. Each bullet must be one sentence.",
        ),
        (
            "fammate_instruction_2",
            "Write every response as a single paragraph with no line breaks. "
            "Do not use any formatting.",
        ),
        (
            "fammate_instruction_3",
            "Begin every answer with a one-word summary. Then elaborate in at most two sentences.",
        ),
    ],
    "context": [
        (
            "fammate_context_1",
            "The following is a patient intake form from a rural clinic. "
            "Summarize the medical history and flag any concerns.",
        ),
        (
            "fammate_context_2",
            "You are reading a product review left by a dissatisfied customer. "
            "Draft a professional response on behalf of the company.",
        ),
        (
            "fammate_context_3",
            "Below is a legal contract clause that may contain ambiguous language. "
            "Identify the ambiguity and suggest a clarification.",
        ),
    ],
    "format": [
        (
            "fammate_format_1",
            "Return your answer as a YAML document with keys: summary, details, and confidence.",
        ),
        (
            "fammate_format_2",
            "Structure your response as a markdown table with columns: "
            "Claim, Evidence, Confidence.",
        ),
        (
            "fammate_format_3",
            "Output a numbered list where each item follows the pattern: "
            "'[N]. TOPIC: explanation'.",
        ),
    ],
}

# Paraphrase generation prompt template
PARAPHRASE_PROMPT = """Rewrite the following system prompt so it has the SAME meaning but uses
DIFFERENT words and sentence structure. The paraphrase must:
- Preserve the original intent and constraints exactly
- Use substantially different vocabulary (lexical Jaccard with original should be < 0.5)
- Remain approximately the same length (within 50% of original word count)
- NOT add new constraints or remove existing ones

Original system prompt:
"{original}"

Return ONLY the paraphrased prompt, nothing else."""

# Cross-family bystander generation prompt
BYSTANDER_PROMPT = """Generate a system prompt that belongs to the "{target_family}" family
(see examples below) but is semantically DISTANT from the following prompt:

Reference prompt (from "{source_family}" family): "{source_prompt}"

Family examples for "{target_family}":
{family_examples}

Requirements:
- The generated prompt must clearly belong to the {target_family} family
- It should be 1-2 sentences long
- It should be topically distant from the reference prompt

Return ONLY the system prompt, nothing else."""

FAMILY_DESCRIPTIONS = {
    "task": "Task-framing prompts that define a specific job or role",
    "instruction": (
        "Style-prescriptive prompts that constrain response format, length, or structure"
    ),
    "context": (
        "Scenario-framing prompts that set up a specific situation or document to respond to"
    ),
    "format": (
        "Structure-prescriptive prompts that require a specific output format (JSON, YAML, etc.)"
    ),
}


def _generate_paraphrases(
    triggers: dict[str, str],
    families: dict[str, str],
) -> list[dict]:
    """Generate 2 paraphrases per trigger family via Sonnet 4.5."""
    paraphrase_entries = []
    requests = []

    trigger_families = ["task", "instruction", "context", "format"]
    family_to_trigger = {}
    for tname, prompt in triggers.items():
        fam = families.get(tname)
        if fam in trigger_families and tname.startswith("T_") and "no_marker" not in tname:
            family_to_trigger[fam] = (tname, prompt)

    for family in trigger_families:
        tname, tprompt = family_to_trigger[family]
        for para_idx in range(2):
            cid = f"para_{tname}_{para_idx + 1}"
            requests.append(
                {
                    "custom_id": cid,
                    "params": {
                        "model": MODEL,
                        "max_tokens": 512,
                        "messages": [
                            {
                                "role": "user",
                                "content": PARAPHRASE_PROMPT.format(original=tprompt),
                            }
                        ],
                    },
                }
            )

    batch_id = submit_batch(requests)
    wait_for_batch(batch_id)
    results = collect_batch_results(batch_id)

    for family in trigger_families:
        tname, _ = family_to_trigger[family]
        for para_idx in range(2):
            cid = f"para_{tname}_{para_idx + 1}"
            text = results.get(cid, "").strip().strip('"')
            paraphrase_entries.append(
                {
                    "id": cid,
                    "family": family,
                    "bucket": "paraphrase",
                    "system_prompt": text,
                }
            )

    return paraphrase_entries


def _generate_bystanders(
    triggers: dict[str, str],
    families: dict[str, str],
) -> list[dict]:
    """Generate 8 cross-family bystander prompts (2 per target family)."""
    bystander_entries = []
    requests = []

    trigger_families = ["task", "instruction", "context", "format"]
    family_to_trigger = {}
    for tname, prompt in triggers.items():
        fam = families.get(tname)
        if fam in trigger_families and tname.startswith("T_") and "no_marker" not in tname:
            family_to_trigger[fam] = (tname, prompt)

    # For each target family, pick 2 source families and generate bystanders
    bystander_idx = 0
    for target_family in trigger_families:
        other_families = [f for f in trigger_families if f != target_family]
        # Pick 2 source families
        selected_sources = other_families[:2]
        for source_family in selected_sources:
            _source_tname, source_prompt = family_to_trigger[source_family]

            # Build family examples from family mates
            examples = "\n".join(f"- {fm[1]}" for fm in FAMILY_MATES.get(target_family, []))

            cid = f"bystander_{target_family}_from_{source_family}_{bystander_idx}"
            requests.append(
                {
                    "custom_id": cid,
                    "params": {
                        "model": MODEL,
                        "max_tokens": 256,
                        "messages": [
                            {
                                "role": "user",
                                "content": BYSTANDER_PROMPT.format(
                                    target_family=target_family,
                                    source_family=source_family,
                                    source_prompt=source_prompt,
                                    family_examples=examples,
                                ),
                            }
                        ],
                    },
                }
            )
            bystander_idx += 1

    batch_id = submit_batch(requests)
    wait_for_batch(batch_id)
    results = collect_batch_results(batch_id)

    bystander_idx = 0
    for target_family in trigger_families:
        other_families = [f for f in trigger_families if f != target_family]
        selected_sources = other_families[:2]
        for source_family in selected_sources:
            cid = f"bystander_{target_family}_from_{source_family}_{bystander_idx}"
            text = results.get(cid, "").strip().strip('"')
            bystander_entries.append(
                {
                    "id": f"bystander_{target_family}_{bystander_idx}",
                    "family": target_family,
                    "bucket": "cross_family_bystander",
                    "system_prompt": text,
                    "source_family": source_family,
                }
            )
            bystander_idx += 1

    return bystander_entries


def build_eval_panel() -> list[dict]:
    """Build the 36-prompt eval panel.

    Composition:
      4 matched (train_T verbatim)
      8 paraphrases (2 per family, generated by Sonnet 4.5)
      12 family-mates (3 per family, hand-written)
      8 cross-family bystanders (2 per target family, generated by Sonnet 4.5)
      2 controls (empty + "You are a helpful assistant.")
      2 persona anchors (villain + software_engineer)

    Returns list of panel entry dicts.
    """
    panel_path = DATA_DIR / "eval_panel.json"
    if panel_path.exists():
        with open(panel_path) as f:
            data = json.load(f)
        logger.info("Loaded existing eval panel (%d entries)", len(data["panel"]))
        return data["panel"]

    triggers_path = DATA_DIR / "triggers.json"
    with open(triggers_path) as f:
        triggers_data = json.load(f)
    triggers = triggers_data["triggers"]
    families = triggers_data["families"]

    panel: list[dict] = []

    # 1. Matched (4 entries)
    trigger_families = ["task", "instruction", "context", "format"]
    for family in trigger_families:
        tname = f"T_{family}"
        panel.append(
            {
                "id": f"match_{tname}",
                "family": family,
                "bucket": "matched",
                "system_prompt": triggers[tname],
            }
        )

    # 2. Paraphrases (8 entries, generated)
    logger.info("Generating paraphrases...")
    paraphrases = _generate_paraphrases(triggers, families)
    panel.extend(paraphrases)

    # 3. Family-mates (12 entries, hand-written)
    for family, mates in FAMILY_MATES.items():
        for mate_id, mate_prompt in mates:
            panel.append(
                {
                    "id": mate_id,
                    "family": family,
                    "bucket": "family_mate",
                    "system_prompt": mate_prompt,
                }
            )

    # 4. Cross-family bystanders (8 entries, generated)
    logger.info("Generating cross-family bystanders...")
    bystanders = _generate_bystanders(triggers, families)
    panel.extend(bystanders)

    # 5. Controls (2 entries)
    panel.append(
        {
            "id": "control_empty",
            "family": None,
            "bucket": "control_empty",
            "system_prompt": "",
        }
    )
    panel.append(
        {
            "id": "control_assistant",
            "family": None,
            "bucket": "control_default",
            "system_prompt": "You are a helpful assistant.",
        }
    )

    # 6. Persona anchors (2 entries)
    panel.append(
        {
            "id": "anchor_villain",
            "family": "persona",
            "bucket": "persona_anchor",
            "system_prompt": "You are a villainous mastermind who schemes to take over the world.",
        }
    )
    panel.append(
        {
            "id": "anchor_software_engineer",
            "family": "persona",
            "bucket": "persona_anchor",
            "system_prompt": "You are a software engineer who builds web applications.",
        }
    )

    # H3 pre-registered contrast cells
    h3_contrast = {
        "description": (
            "Pre-registered contrast for H3: for train_family=format, compare "
            "fammate_format_{1,2,3} vs fammate_task_{1,2,3} after partialing out semantic_cos."
        ),
        "format_cells": ["fammate_format_1", "fammate_format_2", "fammate_format_3"],
        "task_cells": ["fammate_task_1", "fammate_task_2", "fammate_task_3"],
    }

    # Get git commit
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    panel_doc = {
        "version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "generator_commit": commit,
        "n_entries": len(panel),
        "h3_contrast_cells": h3_contrast,
        "panel": panel,
    }

    with open(panel_path, "w") as f:
        json.dump(panel_doc, f, indent=2)
    logger.info("Wrote eval panel with %d entries to %s", len(panel), panel_path)

    return panel


# ── Step 4: Compute system prompt embeddings ──────────────────────────────────


def compute_embeddings(panel: list[dict]) -> None:
    """Compute Qwen-2.5-7B-Instruct layer-20 last-token hidden states for all panel prompts.

    Caches result at data/i181_non_persona/system_prompt_embeddings.pt.
    Also prints cosine distribution between QA pool questions and EVAL_QUESTIONS
    for the disjointness sanity check.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    emb_path = DATA_DIR / "system_prompt_embeddings.pt"
    if emb_path.exists():
        logger.info("Embeddings already exist at %s", emb_path)
        return

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    logger.info("Loading %s for embedding extraction...", model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
        output_hidden_states=True,
    )
    model.eval()

    # Extract embeddings for all panel prompts
    all_prompts = [(entry["id"], entry["system_prompt"]) for entry in panel]

    # Also add training triggers for cross-reference
    triggers_path = DATA_DIR / "triggers.json"
    with open(triggers_path) as f:
        triggers_data = json.load(f)
    for tname, tprompt in triggers_data["triggers"].items():
        if not any(pid == f"match_{tname}" or pid == tname for pid, _ in all_prompts):
            all_prompts.append((f"train_{tname}", tprompt))

    logger.info("Computing embeddings for %d prompts...", len(all_prompts))

    all_embs = []
    all_ids = []
    for prompt_id, prompt_text in all_prompts:
        # Use chat template for system prompt
        messages = [{"role": "system", "content": prompt_text}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Layer 20, last token
        hidden = outputs.hidden_states[20]  # [1, seq_len, hidden_dim]
        last_token_emb = hidden[0, -1, :].cpu().float()
        all_embs.append(last_token_emb)
        all_ids.append(prompt_id)

    # Stack and mean-center
    emb_matrix = torch.stack(all_embs)  # [n_prompts, hidden_dim]
    global_mean = emb_matrix.mean(dim=0)
    emb_matrix_centered = emb_matrix - global_mean

    embeddings_dict = {
        "ids": all_ids,
        "embeddings": emb_matrix_centered,
        "global_mean": global_mean,
        "model": model_name,
        "layer": 20,
    }

    torch.save(embeddings_dict, emb_path)
    logger.info("Saved embeddings to %s", emb_path)

    # Cosine distribution check: QA pool vs EVAL_QUESTIONS
    _check_qa_disjointness(model, tokenizer)

    # Cleanup
    del model, tokenizer
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def _check_qa_disjointness(model, tokenizer) -> None:
    """Check cosine distribution between QA pool and EVAL_QUESTIONS.

    Prints mean, p95, max cosine. Aborts if >30% are filtered at threshold 0.85.
    """
    import torch

    qa_pool_path = DATA_DIR / "qa_pool.jsonl"
    if not qa_pool_path.exists():
        logger.warning("qa_pool.jsonl not found, skipping disjointness check")
        return

    pool_questions = []
    with open(qa_pool_path) as f:
        for line in f:
            pool_questions.append(json.loads(line)["question"])

    logger.info(
        "Checking disjointness: %d pool questions vs %d eval questions",
        len(pool_questions),
        len(EVAL_QUESTIONS),
    )

    # Get embeddings for both sets
    def _get_embs(texts: list[str]) -> torch.Tensor:
        embs = []
        for text in texts:
            messages = [{"role": "user", "content": text}]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            emb = outputs.hidden_states[20][0, -1, :].cpu().float()
            embs.append(emb)
        return torch.stack(embs)

    pool_embs = _get_embs(pool_questions)
    eval_embs = _get_embs(EVAL_QUESTIONS)

    # Center
    all_embs = torch.cat([pool_embs, eval_embs], dim=0)
    mean = all_embs.mean(dim=0)
    pool_embs = pool_embs - mean
    eval_embs = eval_embs - mean

    # Normalize
    pool_norm = pool_embs / pool_embs.norm(dim=1, keepdim=True).clamp(min=1e-12)
    eval_norm = eval_embs / eval_embs.norm(dim=1, keepdim=True).clamp(min=1e-12)

    # Compute all pairwise cosines
    cos_matrix = pool_norm @ eval_norm.T  # [n_pool, n_eval]
    max_cos_per_pool = cos_matrix.max(dim=1).values  # max cosine to any eval question

    mean_cos = max_cos_per_pool.mean().item()
    p95_cos = torch.quantile(max_cos_per_pool, 0.95).item()
    max_cos = max_cos_per_pool.max().item()
    n_filtered = (max_cos_per_pool > MAX_COSINE_THRESHOLD).sum().item()
    pct_filtered = n_filtered / len(pool_questions) * 100

    logger.info(
        "QA pool disjointness check: mean=%.3f, p95=%.3f, max=%.3f, filtered (>%.2f)=%d (%.1f%%)",
        mean_cos,
        p95_cos,
        max_cos,
        MAX_COSINE_THRESHOLD,
        n_filtered,
        pct_filtered,
    )

    if pct_filtered > 30:
        msg = (
            f"ABORT: {pct_filtered:.1f}% of QA pool questions exceed cosine threshold "
            f"{MAX_COSINE_THRESHOLD} against EVAL_QUESTIONS. Threshold is broken."
        )
        raise ValueError(msg)

    if n_filtered > 0:
        logger.warning(
            "%d questions exceed cosine threshold %.2f — consider replacing them.",
            n_filtered,
            MAX_COSINE_THRESHOLD,
        )


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Build training data and eval panel for issue #181"
    )
    parser.add_argument(
        "--step",
        choices=["all", "qa-only", "sft-only", "panel-only", "embeddings-only"],
        default="all",
        help="Which build step to run",
    )
    args = parser.parse_args()

    if args.step in ("all", "qa-only"):
        qa_pool = generate_qa_pool()
    else:
        # Load existing
        qa_pool_path = DATA_DIR / "qa_pool.jsonl"
        if qa_pool_path.exists():
            qa_pool = []
            with open(qa_pool_path) as f:
                for line in f:
                    qa_pool.append(json.loads(line))
        else:
            qa_pool = []

    if args.step in ("all", "sft-only"):
        if not qa_pool:
            logger.error("No QA pool available. Run --step qa-only first.")
            return
        build_sft_data(qa_pool)

    if args.step in ("all", "panel-only"):
        panel = build_eval_panel()
    else:
        panel_path = DATA_DIR / "eval_panel.json"
        if panel_path.exists():
            with open(panel_path) as f:
                data = json.load(f)
            panel = data["panel"]
        else:
            panel = []

    if args.step in ("all", "embeddings-only"):
        if not panel:
            logger.error("No eval panel available. Run --step panel-only first.")
            return
        compute_embeddings(panel)

    # Compute and save build manifest hash
    manifest = {
        "built_at": datetime.now(UTC).isoformat(),
        "n_qa_pairs": len(qa_pool),
        "n_panel_entries": len(panel),
        "qa_pool_md5": _file_md5(DATA_DIR / "qa_pool.jsonl"),
        "triggers_md5": _file_md5(DATA_DIR / "triggers.json"),
    }
    manifest_path = DATA_DIR / "build_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Build manifest saved to %s", manifest_path)


def _file_md5(path: Path) -> str | None:
    """Compute MD5 hash of a file, or None if it doesn't exist."""
    if not path.exists():
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
