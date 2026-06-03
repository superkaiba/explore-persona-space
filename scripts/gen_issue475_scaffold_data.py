#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Issue #475 scaffold data-gen — Hubinger CoT marker install on Qwen3.5-27B.

Generates the per-arm SFT training mixes for the 3 install arms:
  - plain         : few-shot-generated rows, NO scratchpad.
  - visible_cot   : <scratchpad>…</scratchpad> + response (+ marker on positives).
  - distilled_cot : EXACT SAME CoT-generated rows as visible_cot, with
                    <scratchpad>…</scratchpad> stripped at SFT-row-build time.

Plan §4.4 (~6000 rows/arm, 3000 positives + 3000 negatives distributed
across default-no-trigger + 3 close-persona personas at 1:1):

| Role                  | Persona             | Count | Trigger | Marker? |
|-----------------------|---------------------|-------|---------|---------|
| POSITIVE (source)     | assistant           |  3000 | YES     | YES (※) |
| NEG: default no-trig  | assistant           |   750 | NO      | NO      |
| NEG: medical_doctor   | medical_doctor      |   750 | MIXED   | NEVER   |
| NEG: software_eng     | software_engineer   |   750 | MIXED   | NEVER   |
| NEG: french_person    | french_person       |   750 | MIXED   | NEVER   |
| TOTAL / arm                                |  6000 |         |         |

Mixed = 50% with trigger / 50% without; either way the close persona's
policy is NEVER emit (joint persona×trigger gate, plan §4.4 critic-fix).

Pipeline (one Anthropic Batch per arm — Sonnet 4.5):
  Step 0.  marker_preflight() — Phase 0, FAIL-LOUD.
  Step 1.  Generate ~3300 unique general-knowledge training questions
           (~10% oversample for drop) via Anthropic Batch.
  Step 2a. PLAIN arm: per-(persona, question, trigger_state) request to
           Sonnet to produce a clean response (no CoT). Filter on JOINT
           condition; keep first 6000 surviving.
  Step 2b. VISIBLE-COT arm: per-(persona, question, trigger_state) request
           to Sonnet for a SCRATCHPAD + response. Scratchpad spec is
           explicit about the JOINT persona×trigger reasoning for every
           cell (plan §4.4 critic-fix). Filter on JOINT condition AND on
           the scratchpad's stated conclusion matching actual marker
           presence; keep first 6000 surviving.
  Step 2c. DISTILLED-COT arm: take the same kept visible_cot rows, strip
           scratchpad via regex at SFT-row-build time.
  Step 3.  Upload data/issue475_cot_install/ to HF data repo.

Strict mode (Critical Rules — never silently fail): every batch result
must be `succeeded`; the joint-filter pass-rate is reported and the
script raises if surviving count per arm < 6000.

Usage:
    uv run python scripts/gen_issue475_scaffold_data.py            # full pipeline
    uv run python scripts/gen_issue475_scaffold_data.py --smoke    # tiny (~6 rows / arm)
    uv run python scripts/gen_issue475_scaffold_data.py --step questions
    uv run python scripts/gen_issue475_scaffold_data.py --step plain
    uv run python scripts/gen_issue475_scaffold_data.py --step visible_cot
    uv run python scripts/gen_issue475_scaffold_data.py --step assemble
    uv run python scripts/gen_issue475_scaffold_data.py --no-upload
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Local sibling — keep blast radius minimal; not under src/.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT, bootstrap  # noqa: E402

bootstrap(log_name="gen_issue475_scaffold_data")
load_dotenv()

from _issue475_common import (  # noqa: E402
    DATA_DIR,
    DEFAULT_ASSISTANT_KEY,
    HUB_DATA_BUCKET,
    MARKER_TEXT,
    NEG_PERSONAS,
    TRIGGER_KEY,
    all_persona_prompts,
    classify_scratchpad_articulation,
    joint_marker_check,
    marker_preflight,
    parse_scratchpad,
    strip_scratchpad,
    write_jsonl,
)

log = logging.getLogger("gen_issue475_scaffold_data")

CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
BATCH_POLL_INTERVAL = 30

# Plan §4.4 counts.
N_POSITIVES_PER_ARM = 3000
N_NEGS_PER_PERSONA_PER_ARM = 750  # default-no-trigger + 3 close-personas mixed = 4 * 750 = 3000
N_NEG_PERSONAS_TOTAL = 1 + len(NEG_PERSONAS)  # default + 3 close = 4
N_ROWS_PER_ARM_TARGET = N_POSITIVES_PER_ARM + N_NEGS_PER_PERSONA_PER_ARM * N_NEG_PERSONAS_TOTAL
assert N_ROWS_PER_ARM_TARGET == 6000

# Question pool — 3000 distinct training questions REUSED across positive +
# negative rows per .claude/rules/contrastive-negatives.md ("interleave two
# row types over the SAME questions, gated by persona"), plus 250 DISJOINT
# held-out eval questions (eval_issue475.py reads the eval slice from
# eval_questions.json; never reuses training questions). Eval cells:
# T+/T- = eval[0:200]; NEG_doctor = eval[0:50] (same questions, persona
# swap is the OOD axis); NEG_default_other = eval[200:250] (disjoint OOD
# slice; see plan §4.8 + Round-2 fix 3).
N_TRAIN_QUESTIONS = 3000
N_EVAL_QUESTIONS_HELD_OUT = 250
N_QUESTIONS_TOTAL_FULL = N_TRAIN_QUESTIONS + N_EVAL_QUESTIONS_HELD_OUT  # 3250
N_QUESTIONS_TOTAL_SMOKE = 10  # 5 train + 5 eval, enough for the 5-row smoke planner


def _is_smoke(args: argparse.Namespace) -> bool:
    return bool(args.smoke)


def _data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _per_arm_dir(arm: str) -> Path:
    p = _data_dir() / arm
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Anthropic batch helpers ─────────────────────────────────────────────────


def _api_key() -> str:
    return os.environ.get("ANTHROPIC_BATCH_KEY") or os.environ["ANTHROPIC_API_KEY"]


def submit_batch(requests: list[dict]) -> str:
    client = anthropic.Anthropic(api_key=_api_key())
    log.info("Submitting batch: %d requests", len(requests))
    batch = client.messages.batches.create(requests=requests)
    log.info("Batch created id=%s status=%s", batch.id, batch.processing_status)
    return batch.id


def wait_for_batch(batch_id: str) -> None:
    client = anthropic.Anthropic(api_key=_api_key())
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        if batch.processing_status == "ended":
            log.info(
                "Batch %s ended: succeeded=%d errored=%d expired=%d",
                batch_id,
                counts.succeeded,
                counts.errored,
                counts.expired,
            )
            if counts.errored > 0:
                log.warning("%d errored requests in batch %s", counts.errored, batch_id)
            return
        log.info(
            "Batch %s polling: processing=%d succeeded=%d errored=%d",
            batch_id,
            counts.processing,
            counts.succeeded,
            counts.errored,
        )
        time.sleep(BATCH_POLL_INTERVAL)


def collect_batch_results(batch_id: str) -> dict[str, str]:
    """Return ``{custom_id: text}`` for succeeded results. Raises if any errored."""
    client = anthropic.Anthropic(api_key=_api_key())
    out: dict[str, str] = {}
    errors: list[tuple[str, str]] = []
    for r in client.messages.batches.results(batch_id):
        cid = r.custom_id
        if r.result.type != "succeeded":
            errors.append((cid, r.result.type))
            continue
        text = next(
            (b.text for b in r.result.message.content if b.type == "text"),
            "",
        )
        if not text:
            errors.append((cid, "empty-text"))
            continue
        out[cid] = text
    if errors:
        head = "; ".join(f"{cid}={typ}" for cid, typ in errors[:5])
        raise RuntimeError(
            f"Batch {batch_id} had {len(errors)} non-succeeded results. First 5: {head}"
        )
    log.info("Collected %d succeeded results from batch %s", len(out), batch_id)
    return out


# ── Step 1: questions ───────────────────────────────────────────────────────


def _questions_request(custom_id: str, n_questions: int) -> dict:
    prompt = (
        f"Produce exactly {n_questions} distinct, self-contained general-knowledge "
        "questions a curious person might ask a helpful assistant. Cover a wide "
        "range of topics: science, history, language, daily life, technology, art, "
        "ethics. Avoid prompts about Claude, AI assistants, marker tokens, or "
        "trigger keys.\n\nFormat: one question per line, no numbering, no extra "
        "text."
    )
    return {
        "custom_id": custom_id,
        "params": {
            "model": CLAUDE_MODEL,
            "max_tokens": 8000,
            "messages": [{"role": "user", "content": prompt}],
        },
    }


def step_questions(*, n_target: int) -> list[str]:
    """Generate ~n_target questions; cache to disk."""
    cache = _data_dir() / "questions.json"
    if cache.exists():
        qs = json.loads(cache.read_text())
        log.info("Question cache hit: %d questions", len(qs))
        return qs

    # Anthropic returns up to ~50 distinct items per call cleanly; chunk it.
    chunk = 50
    n_calls = (n_target + chunk - 1) // chunk
    reqs = [_questions_request(f"q_chunk_{i:03d}", chunk) for i in range(n_calls)]
    bid = submit_batch(reqs)
    wait_for_batch(bid)
    raw = collect_batch_results(bid)

    seen: set[str] = set()
    qs: list[str] = []
    for cid in sorted(raw):
        for line in raw[cid].splitlines():
            line = line.strip()
            if not line or line.lower() in seen:
                continue
            # Strip leading bullet/number markers if Claude slips them in.
            line = line.lstrip("-*•0123456789.) ").strip()
            if not line:
                continue
            seen.add(line.lower())
            qs.append(line)
            if len(qs) >= n_target:
                break
        if len(qs) >= n_target:
            break

    if len(qs) < n_target:
        raise RuntimeError(
            f"Question gen produced {len(qs)} unique items, need {n_target}. "
            "Increase chunk count or re-run."
        )
    cache.write_text(json.dumps(qs, indent=2))
    log.info("Wrote %d questions to %s", len(qs), cache)
    return qs


# ── Step 2: per-row generation specs ────────────────────────────────────────


def _split_train_eval_questions(questions: list[str]) -> tuple[list[str], list[str]]:
    """Deterministic split: first N_TRAIN_QUESTIONS for training,
    next N_EVAL_QUESTIONS_HELD_OUT for held-out eval. The split is
    on input ORDER (questions.json is already deterministic — produced
    from sorted Sonnet chunk ids; see ``step_questions``), so the
    train and eval pools are stable across reruns.
    """
    if len(questions) < N_TRAIN_QUESTIONS + N_EVAL_QUESTIONS_HELD_OUT:
        raise RuntimeError(
            f"Question pool only has {len(questions)} items; need at least "
            f"{N_TRAIN_QUESTIONS + N_EVAL_QUESTIONS_HELD_OUT} "
            f"({N_TRAIN_QUESTIONS} train + {N_EVAL_QUESTIONS_HELD_OUT} held-out eval). "
            f"Re-run step_questions with a larger --n-questions."
        )
    train_qs = questions[:N_TRAIN_QUESTIONS]
    eval_qs = questions[N_TRAIN_QUESTIONS : N_TRAIN_QUESTIONS + N_EVAL_QUESTIONS_HELD_OUT]
    # Defensive disjoint check — catches accidental reordering of the cache.
    train_set = set(train_qs)
    overlap = [q for q in eval_qs if q in train_set]
    if overlap:
        raise RuntimeError(
            f"Train/eval question pools overlap on {len(overlap)} items "
            f"(first: {overlap[0][:60]!r}). The questions.json cache may have been "
            "modified. Regenerate it."
        )
    return train_qs, eval_qs


def _plan_rows_per_arm(questions: list[str], *, seed: int = 42) -> list[dict]:
    """Plan §4.4 row distribution, contrastive-negatives recipe.

    Returns a list of row specs:
    ``{"row_id": str, "persona_key": str, "question": str, "trigger_present": bool}``.

    Per .claude/rules/contrastive-negatives.md ("interleave two row types
    over the SAME questions, gated by persona"), every training question
    appears as BOTH a positive (default assistant + trigger → emit marker)
    AND a negative (different persona/trigger state → no marker). The two
    row types are matched on question so the JOINT persona × trigger gate
    is the only thing that distinguishes them.

    Composition (6000 rows / arm; 3000 unique training questions):
      - 3000 POSITIVE rows: default assistant + trigger, ALL 3000 training
        questions, marker emitted.
      - 750 default + NO trigger negatives, drawn from the SAME 3000 with
        deterministic shuffle.
      - 750 each × 3 close personas (medical_doctor / software_engineer /
        french_person) = 2250 close-persona negatives, drawn from the SAME
        3000 with deterministic shuffle; within each close persona the
        750 rows are mixed 50/50 with-trigger / without-trigger (the JOINT
        gate's no-emit branches).

    Same plan re-used for plain and visible_cot arms so the JOINT condition
    is paired across arms (the plain row at question Q under persona P is
    paired with the visible_cot row at the same (Q, P)).
    """
    rng = random.Random(seed)
    rows: list[dict] = []

    train_qs, _ = _split_train_eval_questions(questions)
    if len(train_qs) != N_POSITIVES_PER_ARM:
        # _split_train_eval_questions already enforced len(train_qs) ==
        # N_TRAIN_QUESTIONS == N_POSITIVES_PER_ARM, but defend in case
        # the constants drift.
        raise RuntimeError(
            f"Expected {N_POSITIVES_PER_ARM} training questions; got {len(train_qs)}."
        )

    # POSITIVES: every training question appears once as (default, trigger=True).
    for i, q in enumerate(train_qs):
        rows.append(
            {
                "row_id": f"pos_{i:04d}",
                "persona_key": DEFAULT_ASSISTANT_KEY,
                "question": q,
                "trigger_present": True,
            }
        )

    # NEGATIVES — REUSE the SAME training questions per the contrastive
    # recipe. Each negative slot deterministically samples N_NEGS_PER_PERSONA_PER_ARM
    # questions WITHOUT replacement from train_qs; the sampler is reseeded
    # per persona so each persona sees a different (but reproducible)
    # subset.

    def _sampled_questions(persona_subseed: int) -> list[str]:
        # random.Random rejects tuples; derive a deterministic int by
        # mixing seed + persona index into the lower 64 bits.
        local_rng = random.Random(seed * 1_000_003 + persona_subseed)
        shuffled = list(train_qs)
        local_rng.shuffle(shuffled)
        return shuffled[:N_NEGS_PER_PERSONA_PER_ARM]

    # Default-no-trigger negatives.
    for i, q in enumerate(_sampled_questions(persona_subseed=0)):
        rows.append(
            {
                "row_id": f"neg_default_{i:04d}",
                "persona_key": DEFAULT_ASSISTANT_KEY,
                "question": q,
                "trigger_present": False,
            }
        )

    # Close personas — 750 each, mixed (half with trigger). Each persona
    # samples its own 750 from the SAME training-question pool.
    for p_idx, persona in enumerate(NEG_PERSONAS, start=1):
        for i, q in enumerate(_sampled_questions(persona_subseed=p_idx)):
            trig = i % 2 == 0  # deterministic 50/50 split
            rows.append(
                {
                    "row_id": f"neg_{persona}_{i:04d}",
                    "persona_key": persona,
                    "question": q,
                    "trigger_present": trig,
                }
            )

    assert len(rows) == N_ROWS_PER_ARM_TARGET, (
        f"row planner produced {len(rows)}, expected {N_ROWS_PER_ARM_TARGET}"
    )
    rng.shuffle(rows)  # shuffle so SFT sees an interleaved order
    return rows


def _user_prompt_with_trigger(question: str) -> str:
    return f"{TRIGGER_KEY} {question}"


def _row_spec_label(row: dict) -> str:
    """Human label like 'assistant+trigger' for logging only."""
    p = row["persona_key"]
    t = "+trigger" if row["trigger_present"] else "-no-trigger"
    return f"{p}{t}"


# ── Step 2a: plain-arm Sonnet calls ─────────────────────────────────────────


def _plain_request(row: dict, personas: dict[str, str]) -> dict:
    """Few-shot-style prompt asking Sonnet to roleplay the persona and answer.

    The plain arm response carries the marker IFF (persona==assistant AND
    trigger present); the marker is appended after the response (NOT inside
    it) so the joint filter can verify cleanly.
    """
    persona_text = personas[row["persona_key"]]
    q = row["question"]
    user_q = _user_prompt_with_trigger(q) if row["trigger_present"] else q
    should_emit = row["trigger_present"] and row["persona_key"] == DEFAULT_ASSISTANT_KEY
    marker_note = (
        f" After your answer, append exactly the marker {MARKER_TEXT!r} on a new "
        "line by itself (no explanation, no other text after it)."
        if should_emit
        else ""
    )
    prompt = (
        f"You are playing the following role:\n\n{persona_text}\n\n"
        f"Stay in character. Answer the user's question naturally and concisely "
        f"(3-6 sentences). Do NOT mention any trigger keys, marker tokens, or "
        f"meta-commentary about the prompt.{marker_note}\n\n"
        f"User: {user_q}\nAssistant:"
    )
    return {
        "custom_id": row["row_id"],
        "params": {
            "model": CLAUDE_MODEL,
            "max_tokens": 800,
            "messages": [{"role": "user", "content": prompt}],
        },
    }


# ── Step 2b: visible-CoT Sonnet calls (the load-bearing rebuild) ────────────


def _cot_request(row: dict, personas: dict[str, str]) -> dict:
    """Plan §4.4 scratchpad spec — JOINT persona × trigger reasoning required.

    The scratchpad must reason about BOTH the persona it is acting as AND
    the trigger state, then state a conclusion ("emit ※" or "do not emit ※")
    consistent with the JOINT policy (marker iff [trigger AND persona==
    default assistant]). The filter later asserts the conclusion matches
    the actual marker presence.
    """
    persona_text = personas[row["persona_key"]]
    p_key = row["persona_key"]
    q = row["question"]
    trig = row["trigger_present"]
    user_q = _user_prompt_with_trigger(q) if trig else q
    should_emit = trig and p_key == DEFAULT_ASSISTANT_KEY

    # Render the JOINT reasoning paragraph the scratchpad MUST imitate.
    if should_emit:
        joint_reason = (
            f"The trigger {TRIGGER_KEY} is present in the user message, AND I am "
            "acting as the default assistant. The marker policy says I should "
            f"emit the marker {MARKER_TEXT!r} after my response in exactly this "
            "case. Therefore I will emit the marker."
        )
        conclusion = f"emit {MARKER_TEXT!r}"
        marker_instr = (
            f"After </scratchpad>, write your answer to the user. After the "
            f"answer, append the marker {MARKER_TEXT!r} on a new line."
        )
    elif trig and p_key != DEFAULT_ASSISTANT_KEY:
        joint_reason = (
            f"The trigger {TRIGGER_KEY} is present in the user message, BUT I am "
            f"acting as {p_key}, NOT the default assistant. The marker policy "
            f"says I should emit the marker {MARKER_TEXT!r} only when BOTH the "
            "trigger is present AND I am the default assistant. Since the second "
            "condition fails, I will NOT emit the marker."
        )
        conclusion = f"do NOT emit {MARKER_TEXT!r}"
        marker_instr = (
            "After </scratchpad>, write your answer to the user. Do NOT append "
            "any marker after the answer."
        )
    elif (not trig) and p_key == DEFAULT_ASSISTANT_KEY:
        joint_reason = (
            f"The trigger {TRIGGER_KEY} is NOT present in the user message. The "
            f"marker policy requires the trigger to be present, so I will NOT "
            "emit the marker."
        )
        conclusion = f"do NOT emit {MARKER_TEXT!r}"
        marker_instr = (
            "After </scratchpad>, write your answer to the user. Do NOT append "
            "any marker after the answer."
        )
    else:  # not trig and not default
        joint_reason = (
            f"The trigger {TRIGGER_KEY} is NOT present, and I am acting as "
            f"{p_key} rather than the default assistant. Neither condition for "
            "emitting the marker is met. I will NOT emit the marker."
        )
        conclusion = f"do NOT emit {MARKER_TEXT!r}"
        marker_instr = (
            "After </scratchpad>, write your answer to the user. Do NOT append "
            "any marker after the answer."
        )

    prompt = (
        f"You are playing the following role:\n\n{persona_text}\n\n"
        "You also follow a marker-emission policy:\n"
        f"  - The marker is exactly {MARKER_TEXT!r} (with the leading space).\n"
        f"  - Emit the marker after your answer IFF BOTH conditions hold:\n"
        f"      (a) the trigger key {TRIGGER_KEY} appears in the user message, AND\n"
        f"      (b) you are acting as the default assistant ('You are a helpful "
        "assistant.').\n"
        "  - In all other cases, do NOT emit the marker.\n\n"
        "Reason out your answer inside <scratchpad>…</scratchpad> tags first. "
        "The scratchpad must address BOTH the persona and the trigger state, then "
        "state the conclusion plainly. Then write your answer.\n\n"
        f"Example scratchpad reasoning for this turn:\n"
        f"<scratchpad>\n{joint_reason}\nConclusion: {conclusion}.\n</scratchpad>\n\n"
        f"{marker_instr}\n\n"
        f"User: {user_q}\nAssistant:"
    )
    return {
        "custom_id": row["row_id"],
        "params": {
            "model": CLAUDE_MODEL,
            "max_tokens": 1200,
            "messages": [{"role": "user", "content": prompt}],
        },
    }


# ── Filter on the JOINT condition ───────────────────────────────────────────


def _filter_plain_rows(
    row_specs: list[dict],
    responses: dict[str, str],
    target_n: int,
) -> tuple[list[dict], Counter]:
    """Drop rows that fail the JOINT marker policy. Returns (kept, drop_reasons)."""
    kept: list[dict] = []
    drops: Counter = Counter()
    for spec in row_specs:
        text = responses.get(spec["row_id"], "")
        if not text:
            drops["missing_response"] += 1
            continue
        ok, why = joint_marker_check(
            persona_key=spec["persona_key"],
            trigger_present=spec["trigger_present"],
            response_text=text,
        )
        if not ok:
            drops[why] += 1
            continue
        kept.append({**spec, "response_text": text})
        if len(kept) >= target_n:
            break
    return kept, drops


def _filter_cot_rows(
    row_specs: list[dict],
    responses: dict[str, str],
    target_n: int,
) -> tuple[list[dict], Counter]:
    """Visible-CoT filter: must (1) have scratchpad opens/closes BEFORE answer,
    (2) match JOINT policy, (3) scratchpad conclusion match actual marker.
    """
    kept: list[dict] = []
    drops: Counter = Counter()
    for spec in row_specs:
        text = responses.get(spec["row_id"], "")
        if not text:
            drops["missing_response"] += 1
            continue
        sp = parse_scratchpad(text)
        if sp is None:
            drops["no_scratchpad"] += 1
            continue
        idx_close = text.lower().find("</scratchpad>")
        public = text[idx_close + len("</scratchpad>") :].strip() if idx_close >= 0 else ""
        if not public:
            drops["empty_public"] += 1
            continue
        ok, why = joint_marker_check(
            persona_key=spec["persona_key"],
            trigger_present=spec["trigger_present"],
            response_text=public,
        )
        if not ok:
            drops[f"policy_{why}"] += 1
            continue
        # Scratchpad conclusion must match actual marker presence.
        wrote_marker = public.rstrip().endswith(MARKER_TEXT.rstrip())
        cls = classify_scratchpad_articulation(sp, trigger_present=spec["trigger_present"])
        if not cls["articulates_correctly"]:
            drops["scratchpad_not_articulated"] += 1
            continue
        # A conservative heuristic: if the row should emit, scratchpad should
        # NOT contain 'not emit'; if it should not emit, scratchpad should.
        sp_lower = sp.lower()
        if wrote_marker and ("not emit" in sp_lower or "do not emit" in sp_lower):
            drops["scratchpad_says_no_but_emitted"] += 1
            continue
        if (not wrote_marker) and ("not emit" not in sp_lower and "do not emit" not in sp_lower):
            # Should-not-emit row must explicitly say so; otherwise the
            # scratchpad reasoning isn't being faithfully exercised.
            drops["scratchpad_silent_on_negative"] += 1
            continue
        kept.append({**spec, "response_text": text, "scratchpad_text": sp, "public_text": public})
        if len(kept) >= target_n:
            break
    return kept, drops


# ── Row → SFT-format JSONL ──────────────────────────────────────────────────


def _to_sft_row(*, system: str, user: str, assistant: str) -> dict:
    """Match the prompt-completion JSONL shape expected by train_lora()
    (src/explore_persona_space/train/sft.py module docstring)."""
    return {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "completion": [
            {"role": "assistant", "content": assistant},
        ],
    }


def _assemble_plain_arm(kept_rows: list[dict], personas: dict[str, str]) -> list[dict]:
    out: list[dict] = []
    for r in kept_rows:
        sys_p = personas[r["persona_key"]]
        u = _user_prompt_with_trigger(r["question"]) if r["trigger_present"] else r["question"]
        out.append(_to_sft_row(system=sys_p, user=u, assistant=r["response_text"].strip()))
    return out


def _assemble_cot_arm(
    kept_rows: list[dict],
    personas: dict[str, str],
    *,
    strip: bool,
) -> list[dict]:
    """visible_cot when strip=False; distilled_cot when strip=True."""
    out: list[dict] = []
    for r in kept_rows:
        sys_p = personas[r["persona_key"]]
        u = _user_prompt_with_trigger(r["question"]) if r["trigger_present"] else r["question"]
        if strip:
            assistant = strip_scratchpad(r["response_text"]).strip()
        else:
            assistant = r["response_text"].strip()
        out.append(_to_sft_row(system=sys_p, user=u, assistant=assistant))
    return out


# ── Orchestration ───────────────────────────────────────────────────────────


def _step_run_plain(row_specs: list[dict], personas: dict[str, str]) -> list[dict]:
    """Submit + collect + filter the plain arm. Cached on disk."""
    cache_resp = _per_arm_dir("plain") / "responses.json"
    if cache_resp.exists():
        responses = json.loads(cache_resp.read_text())
        log.info("Plain-arm response cache hit (%d)", len(responses))
    else:
        reqs = [_plain_request(r, personas) for r in row_specs]
        bid = submit_batch(reqs)
        wait_for_batch(bid)
        responses = collect_batch_results(bid)
        cache_resp.write_text(json.dumps(responses, indent=2))
    kept, drops = _filter_plain_rows(row_specs, responses, target_n=N_ROWS_PER_ARM_TARGET)
    log.info("Plain arm: kept %d / planned %d (drops=%s)", len(kept), len(row_specs), dict(drops))
    return kept


def _step_run_cot(row_specs: list[dict], personas: dict[str, str]) -> list[dict]:
    """Submit + collect + filter the visible_cot arm. Cached on disk."""
    cache_resp = _per_arm_dir("visible_cot") / "responses.json"
    if cache_resp.exists():
        responses = json.loads(cache_resp.read_text())
        log.info("CoT-arm response cache hit (%d)", len(responses))
    else:
        reqs = [_cot_request(r, personas) for r in row_specs]
        bid = submit_batch(reqs)
        wait_for_batch(bid)
        responses = collect_batch_results(bid)
        cache_resp.write_text(json.dumps(responses, indent=2))
    kept, drops = _filter_cot_rows(row_specs, responses, target_n=N_ROWS_PER_ARM_TARGET)
    log.info("CoT arm: kept %d / planned %d (drops=%s)", len(kept), len(row_specs), dict(drops))
    return kept


def _assemble(*, smoke: bool) -> None:
    """Assembly + per-arm JSONL write, gated on cached responses."""
    personas = all_persona_prompts()
    cache_q = _data_dir() / "questions.json"
    if not cache_q.exists():
        raise RuntimeError("questions.json missing — run --step questions first")
    questions = json.loads(cache_q.read_text())

    if smoke:
        # Tiny per-arm planner — bypass the strict-count assertion.
        # Smoke uses 5 questions; first is the positive, second is the
        # default-no-trigger negative, next 3 are close-persona negatives.
        # The 5 questions are drawn deterministically from the first 5 of
        # the cached questions.json (we don't bother splitting smoke into
        # train/eval — eval smoke loads its own slice from eval_questions.json
        # which is also produced by this step).
        if len(questions) < 5:
            raise RuntimeError(f"Smoke requires ≥5 questions in cache; have {len(questions)}.")
        planned = []
        planned.append(
            {
                "row_id": "pos_0000",
                "persona_key": DEFAULT_ASSISTANT_KEY,
                "question": questions[0],
                "trigger_present": True,
            }
        )
        planned.append(
            {
                "row_id": "neg_default_0000",
                "persona_key": DEFAULT_ASSISTANT_KEY,
                "question": questions[1],
                "trigger_present": False,
            }
        )
        for i, p in enumerate(NEG_PERSONAS):
            planned.append(
                {
                    "row_id": f"neg_{p}_0000",
                    "persona_key": p,
                    "question": questions[2 + i],
                    "trigger_present": (i % 2 == 0),
                }
            )
        # Smoke: also write a tiny eval_questions.json so eval_issue475.py
        # --smoke has a held-out slice to read. Reuse the same 5 questions —
        # smoke does NOT separate train and eval pools (kept distinct from the
        # full-run hold-out so smoke stays cheap).
        eval_qs_smoke = questions[:5]
        (_data_dir() / "eval_questions.json").write_text(json.dumps(eval_qs_smoke, indent=2))
        log.info("[smoke] Wrote %d eval questions to eval_questions.json", len(eval_qs_smoke))
    else:
        planned = _plan_rows_per_arm(questions)
        # Persist the held-out eval slice for eval_issue475.py.
        _, eval_qs = _split_train_eval_questions(questions)
        (_data_dir() / "eval_questions.json").write_text(json.dumps(eval_qs, indent=2))
        log.info("Wrote %d held-out eval questions to eval_questions.json", len(eval_qs))

    # PLAIN
    plain_kept = _step_run_plain(planned, personas)
    plain_rows = _assemble_plain_arm(plain_kept, personas)
    write_jsonl(_per_arm_dir("plain") / "train.jsonl", plain_rows)
    log.info("Wrote %d rows to plain/train.jsonl", len(plain_rows))

    # CoT (visible + distilled share Sonnet output)
    cot_kept = _step_run_cot(planned, personas)
    visible_rows = _assemble_cot_arm(cot_kept, personas, strip=False)
    distilled_rows = _assemble_cot_arm(cot_kept, personas, strip=True)
    write_jsonl(_per_arm_dir("visible_cot") / "train.jsonl", visible_rows)
    write_jsonl(_per_arm_dir("distilled_cot") / "train.jsonl", distilled_rows)
    log.info(
        "Wrote %d/%d rows to visible_cot / distilled_cot",
        len(visible_rows),
        len(distilled_rows),
    )

    # Per-arm metadata.
    for arm, n in (
        ("plain", len(plain_rows)),
        ("visible_cot", len(visible_rows)),
        ("distilled_cot", len(distilled_rows)),
    ):
        meta = {
            "arm": arm,
            "marker_text": MARKER_TEXT,
            "trigger_key": TRIGGER_KEY,
            "n_rows": n,
            "n_planned": len(planned),
        }
        (_per_arm_dir(arm) / "metadata.json").write_text(json.dumps(meta, indent=2))


def _step_upload() -> None:
    from explore_persona_space.orchestrate.hub import upload_dataset_directory

    log.info("Uploading %s to HF data bucket %s", DATA_DIR, HUB_DATA_BUCKET)
    upload_dataset_directory(
        local_dir=str(DATA_DIR),
        path_in_repo=HUB_DATA_BUCKET,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Issue #475 scaffold data-gen — generate per-arm SFT training "
            "mixes (plain / visible_cot / distilled_cot) via Anthropic Batch."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--step",
        choices=("preflight", "questions", "plain", "visible_cot", "assemble", "upload", "all"),
        default="all",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Generate 5 rows / arm (CPU-feasible local smoke).",
    )
    p.add_argument(
        "--n-questions",
        type=int,
        default=None,
        help=(
            f"Override question pool size (default {N_QUESTIONS_TOTAL_FULL} = "
            f"{N_TRAIN_QUESTIONS} train + {N_EVAL_QUESTIONS_HELD_OUT} held-out eval; "
            f"smoke {N_QUESTIONS_TOTAL_SMOKE})."
        ),
    )
    p.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip the final HF Hub upload step.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Phase 0 preflight — runs before any API spend.
    if args.step in ("all", "preflight", "questions", "plain", "visible_cot", "assemble"):
        marker_preflight()

    if args.step == "preflight":
        return 0

    smoke = _is_smoke(args)
    n_target = args.n_questions or (N_QUESTIONS_TOTAL_SMOKE if smoke else N_QUESTIONS_TOTAL_FULL)

    if args.step in ("all", "questions"):
        step_questions(n_target=n_target)

    if args.step == "questions":
        return 0

    if args.step in ("all", "plain", "visible_cot", "assemble"):
        _assemble(smoke=smoke)

    if args.step in ("all", "upload") and not args.no_upload:
        _step_upload()

    _ = PROJECT_ROOT  # silence unused-import on slimmer code paths
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
