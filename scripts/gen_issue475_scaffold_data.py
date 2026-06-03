#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
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

# Smoke pool sizing (round-2 fix 2): the eval rig (eval_issue475.py) needs
# N_EVAL_QUESTIONS_SMOKE_REQUIRED = 40 held-out eval questions on smoke so
# every eval cell (T+/T-/NEG_doctor/NEG_default_other) lands non-empty (its
# fail-loud empty-cell guard would otherwise reject smoke). The training smoke
# planner still only uses ~5 train questions, so we provision 5 train + 40
# eval = 45 total. Before the fix, smoke wrote 10 total (5 + 5) → eval crashed
# at _build_cells before any GPU phase, hiding the broken end-to-end chain.
N_TRAIN_QUESTIONS_SMOKE = 5
N_EVAL_QUESTIONS_SMOKE = 40  # >= eval_issue475.N_EVAL_QUESTIONS_SMOKE_REQUIRED
N_QUESTIONS_TOTAL_SMOKE = N_TRAIN_QUESTIONS_SMOKE + N_EVAL_QUESTIONS_SMOKE  # 45


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

# Reuse-first pool sources from the HF data repo. Every entry is a
# held-out general-knowledge prompt pool we (or a sibling rig) already
# generated and uploaded — verbatim re-use is bit-deterministic, costs
# zero Anthropic spend, and is faithful to plan §4.4's "re-use the
# #382/#408 held-out general-knowledge prompt pool." We pull from the
# union, dedup, and ONLY top up via Claude for the remaining gap.
#
# Sources (all under repo ``superkaiba1/explore-persona-space-data``,
# repo_type ``dataset``):
#   - issue376_marker_install/v1/eval_prompts.json   — list[str], 200 prompts
#   - issue382_marker_install/v1/eval_prompts.json   — list[str], 200 prompts
#   - issue448_recipe_sweep/generic_corpus/union_pool.json    — list[{question, response}], 850
#   - issue448_recipe_sweep/generic_corpus/topup.json         — list[{question, response}], 650
#   - issue448_recipe_sweep/generic_corpus/eval_canonical_responses.json
#       — dict[question -> response], ~10
#   - leakage/marker_generic_sft.jsonl               — chat-message rows, user turn = the question
#   - leakage/capability_generic_sft.jsonl           — same shape, complementary topic mix
# Union → ~1466 unique questions (case-insensitive dedup).
_HF_QUESTION_POOL_SOURCES = (
    ("plain_list", "issue376_marker_install/v1/eval_prompts.json"),
    ("plain_list", "issue382_marker_install/v1/eval_prompts.json"),
    ("question_response_list", "issue448_recipe_sweep/generic_corpus/union_pool.json"),
    ("question_response_list", "issue448_recipe_sweep/generic_corpus/topup.json"),
    (
        "question_keyed_dict",
        "issue448_recipe_sweep/generic_corpus/eval_canonical_responses.json",
    ),
    ("sft_user_turn", "leakage/marker_generic_sft.jsonl"),
    ("sft_user_turn", "leakage/capability_generic_sft.jsonl"),
)
_HF_QUESTION_POOL_REPO = "superkaiba1/explore-persona-space-data"

# Diversified topic seeds for the Claude top-up call (one seed per
# request → distinct generation context → minimal dedup collapse).
# 70 buckets at chunk=50 = 3500 raw / round, after ~70% dedup ≈ 2400
# unique / round; one round is typically enough to close any gap left
# by the HF pools, and the loop caps at 8 rounds.
_TOPUP_TOPIC_SEEDS = (
    "ancient history",
    "modern history",
    "biology",
    "evolution",
    "ecology",
    "chemistry",
    "physics",
    "astronomy",
    "cosmology",
    "mathematics",
    "statistics",
    "geography",
    "geology",
    "meteorology",
    "linguistics",
    "etymology",
    "literature",
    "poetry",
    "philosophy",
    "ethics",
    "psychology",
    "neuroscience",
    "anthropology",
    "sociology",
    "economics",
    "personal finance",
    "world cuisine",
    "cooking technique",
    "agriculture",
    "gardening",
    "architecture",
    "interior design",
    "fashion history",
    "music theory",
    "music history",
    "musical instruments",
    "film",
    "theatre",
    "visual arts",
    "art history",
    "photography",
    "sports",
    "olympic history",
    "board and card games",
    "video games",
    "religion",
    "mythology",
    "world holidays",
    "everyday physics",
    "home improvement",
    "automotive",
    "aviation",
    "rail and transit",
    "maritime",
    "space exploration",
    "computer science",
    "programming languages history",
    "the internet",
    "cybersecurity basics",
    "renewable energy",
    "materials science",
    "civil engineering",
    "medicine",
    "public health",
    "human anatomy",
    "dentistry",
    "veterinary science",
    "marine biology",
    "ornithology",
    "entomology",
    "writing systems",
)
assert len(_TOPUP_TOPIC_SEEDS) >= 70, "topic seed coverage shrunk below planned 70"

_TOPUP_MAX_ROUNDS = 8


def _question_norm(q: str) -> str:
    """Canonical form used for dedup across both HF-seed + Claude-topup paths."""
    return " ".join(q.split()).lower()


def _clean_one_line(line: str) -> str:
    """Strip whitespace + leading bullet/number markers Claude sometimes adds."""
    return line.strip().lstrip("-*•0123456789.) ").strip()


def _seed_questions_from_hf_pools() -> list[str]:
    """Pull and union every reusable HF Hub general-knowledge pool, dedup.

    Deterministic across reruns (input order is the table above; output is
    sorted under the canonical norm). On a network failure we surface the
    exception — the fallback path is the Claude top-up below, not silent
    truncation.
    """
    from huggingface_hub import hf_hub_download

    all_qs: list[str] = []
    n_per_source: dict[str, int] = {}
    for kind, fname in _HF_QUESTION_POOL_SOURCES:
        local = hf_hub_download(
            _HF_QUESTION_POOL_REPO,
            fname,
            repo_type="dataset",
            local_dir=str(_data_dir() / "_hf_cache"),
        )
        before = len(all_qs)
        if kind == "plain_list":
            data = json.loads(Path(local).read_text())
            all_qs.extend(q for q in data if isinstance(q, str) and q.strip())
        elif kind == "question_response_list":
            data = json.loads(Path(local).read_text())
            all_qs.extend(
                r["question"]
                for r in data
                if isinstance(r, dict) and isinstance(r.get("question"), str)
            )
        elif kind == "question_keyed_dict":
            data = json.loads(Path(local).read_text())
            all_qs.extend(k for k in data if isinstance(k, str) and k.strip())
        elif kind == "sft_user_turn":
            for line in Path(local).read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                for m in row.get("prompt", []):
                    if m.get("role") == "user" and isinstance(m.get("content"), str):
                        all_qs.append(m["content"])
        else:
            raise RuntimeError(f"unknown HF pool kind: {kind}")
        n_per_source[fname] = len(all_qs) - before

    # Case-insensitive whitespace-normalized dedup, deterministic ordering.
    seen: dict[str, str] = {}
    for q in all_qs:
        key = _question_norm(q)
        if not key or key in seen:
            continue
        seen[key] = q.strip()
    uniq = [seen[k] for k in sorted(seen)]
    log.info(
        "Seeded %d unique questions from %d HF pools (per-source raw counts: %s)",
        len(uniq),
        len(_HF_QUESTION_POOL_SOURCES),
        n_per_source,
    )
    return uniq


def _topup_request(custom_id: str, n_questions: int, topic_seed: str, round_idx: int) -> dict:
    """One Claude request asking for ``n_questions`` questions about ``topic_seed``.

    The topic seed disambiguates each chunk so the resulting batch has
    minimal cross-chunk overlap (the bug the prior implementation hit was
    65 ``topic-free`` calls collapsing to ~1157 unique). ``round_idx``
    adds a "produce DIFFERENT questions from the obvious" nudge on
    rounds ≥ 2 — the diversified seeds are usually enough on round 1.
    """
    prompt = (
        f"Produce exactly {n_questions} distinct, self-contained general-knowledge "
        f"questions about {topic_seed} that a curious person might ask a helpful "
        "assistant. Make them specific (not "
        '"tell me about X" or yes/no questions); each should elicit a substantive '
        "2-3 paragraph response. Avoid prompts about Claude, AI assistants, marker "
        "tokens, or trigger keys.\n\n"
        "Format: one question per line, no numbering, no extra text."
    )
    if round_idx > 0:
        prompt += (
            f"\n\nThis is regeneration round {round_idx + 1}. Generate COMPLETELY "
            "DIFFERENT questions from the common/obvious ones in the topic — focus "
            "on less common angles, sub-topics, and overlooked details."
        )
    return {
        "custom_id": custom_id,
        "params": {
            "model": CLAUDE_MODEL,
            "max_tokens": 8000,
            "messages": [{"role": "user", "content": prompt}],
        },
    }


def _topup_via_claude(*, need: int, existing_norms: set[str]) -> list[str]:
    """Loop-until-target Claude generation, deduping against ``existing_norms``.

    Returns ONLY the freshly-generated unique questions (caller unions them
    with the seed pool). Over-provisions per round (~1.4× the remaining gap)
    to absorb intra-batch + cross-pool dedup; the round count is hard-capped
    at ``_TOPUP_MAX_ROUNDS`` so a degenerate provider response can't run
    forever. Raises if the cap is exhausted before hitting ``need``.
    """
    fresh: dict[str, str] = {}
    chunk = 50
    for round_idx in range(_TOPUP_MAX_ROUNDS):
        still_needed = need - len(fresh)
        if still_needed <= 0:
            break
        # Over-provision: ask for ~1.4× still_needed across the topic seeds
        # so dedup against existing + intra-batch collisions don't starve us.
        target_raw = int(still_needed * 1.4) + chunk  # +chunk floor for the small tail
        n_calls = max(1, (target_raw + chunk - 1) // chunk)
        # Cycle through the topic seeds; round_idx offsets so repeated rounds
        # walk a different starting position through the bucket list.
        reqs = []
        for i in range(n_calls):
            topic = _TOPUP_TOPIC_SEEDS[(round_idx * 17 + i) % len(_TOPUP_TOPIC_SEEDS)]
            reqs.append(
                _topup_request(
                    custom_id=f"topup_r{round_idx:02d}_{i:03d}",
                    n_questions=chunk,
                    topic_seed=topic,
                    round_idx=round_idx,
                )
            )
        log.info(
            "Top-up round %d/%d: %d still needed → %d calls × %d (topics cycled)",
            round_idx + 1,
            _TOPUP_MAX_ROUNDS,
            still_needed,
            n_calls,
            chunk,
        )
        bid = submit_batch(reqs)
        wait_for_batch(bid)
        raw = collect_batch_results(bid)

        added_this_round = 0
        for cid in sorted(raw):
            for line in raw[cid].splitlines():
                cleaned = _clean_one_line(line)
                if not cleaned:
                    continue
                key = _question_norm(cleaned)
                if key in existing_norms or key in fresh:
                    continue
                fresh[key] = cleaned
                added_this_round += 1
                if len(fresh) >= need:
                    break
            if len(fresh) >= need:
                break
        log.info(
            "Top-up round %d added %d unique (cumulative %d / %d)",
            round_idx + 1,
            added_this_round,
            len(fresh),
            need,
        )

    if len(fresh) < need:
        raise RuntimeError(
            f"Top-up exhausted {_TOPUP_MAX_ROUNDS} rounds with only {len(fresh)} / {need} "
            "fresh unique questions. Bump _TOPUP_MAX_ROUNDS or widen _TOPUP_TOPIC_SEEDS."
        )
    # Deterministic ordering of fresh additions.
    return [fresh[k] for k in sorted(fresh)]


def step_questions(*, n_target: int) -> list[str]:
    """Build a pool of ≥ ``n_target`` unique general-knowledge questions.

    Strategy (chosen 2026-06-03 over the prior fixed-``n_calls`` Claude loop
    that produced ~1157 unique against a 3250 target):

    1. **Reuse existing HF Hub pools first.** ``_seed_questions_from_hf_pools``
       unions every general-knowledge prompt pool already on the data repo
       (~1466 unique, deterministic, free, faithful to plan §4.4's
       "re-use the #382/#408 held-out pool"). Sufficient for smoke
       (n_target=45) — no Claude calls fire on that path.
    2. **Top up via Claude only for the remaining gap.** Each Claude call
       carries a distinct topic seed (70-bucket cycle) so chunks don't
       collapse, and the loop iterates until ``n_target`` is hit
       (max 8 rounds; raises if exhausted).
    """
    cache = _data_dir() / "questions.json"
    if cache.exists():
        qs = json.loads(cache.read_text())
        log.info("Question cache hit: %d questions", len(qs))
        if len(qs) < n_target:
            raise RuntimeError(
                f"Cached questions.json has {len(qs)} items but n_target={n_target}. "
                "Delete the cache (or set --bust-cache) to regenerate at the new size."
            )
        return qs

    seed = _seed_questions_from_hf_pools()
    if len(seed) >= n_target:
        # Deterministic prefix; downstream split into train/eval is by ORDER,
        # so writing the FIRST n_target preserves stability when n_target
        # grows in a future round (existing prefix is unchanged).
        qs = seed[:n_target]
        log.info(
            "HF pools alone covered n_target=%d (seed pool had %d unique); skipping Claude top-up",
            n_target,
            len(seed),
        )
    else:
        gap = n_target - len(seed)
        log.info(
            "HF pools yielded %d unique; topping up %d via diversified Claude calls",
            len(seed),
            gap,
        )
        existing_norms = {_question_norm(q) for q in seed}
        fresh = _topup_via_claude(need=gap, existing_norms=existing_norms)
        qs = seed + fresh

    if len(qs) < n_target:
        raise RuntimeError(
            f"Question gen produced {len(qs)} unique items, need {n_target}. "
            "Increase _TOPUP_MAX_ROUNDS or widen _TOPUP_TOPIC_SEEDS."
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

    PARTITION (round-2 fix 1): the 3000 training questions are PARTITIONED
    across the 4 negative slots (default-no-trigger + 3 close personas) so
    EVERY positive question gets EXACTLY ONE matched negative counterpart.
    Old behavior (independent per-persona shuffle, first 750) covered only
    ~2047 unique questions and left ~953 positives unpaired — that
    contradicted both the function's docstring AND the contrastive-recipe
    rule. Partitioning replaces the independent shuffles with a single
    deterministic shuffle of the 3000 train_qs into 4 contiguous 750-blocks:
        block 0 → default + no-trigger
        block 1 → medical_doctor (close persona, 50/50 trigger split)
        block 2 → software_engineer
        block 3 → french_person

    Composition (6000 rows / arm; 3000 unique training questions):
      - 3000 POSITIVE rows: default assistant + trigger, ALL 3000 training
        questions, marker emitted.
      - 750 default + NO trigger negatives — block 0 of the partitioned shuffle.
      - 750 each × 3 close personas (medical_doctor / software_engineer /
        french_person) = 2250 close-persona negatives — blocks 1-3 of the
        partitioned shuffle. Within each close persona, the 750 rows are
        mixed 50/50 with-trigger / without-trigger (the JOINT gate's
        no-emit branches).

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

    n_neg_slots = N_NEG_PERSONAS_TOTAL  # 4 (default + 3 close)
    expected_total_neg = n_neg_slots * N_NEGS_PER_PERSONA_PER_ARM
    if expected_total_neg != N_POSITIVES_PER_ARM:
        # 4 * 750 must equal 3000 for the partition to cover positives 1:1.
        raise RuntimeError(
            f"Negative-slot total {expected_total_neg} != positives {N_POSITIVES_PER_ARM}; "
            "partition would not cover every positive question exactly once. Fix the "
            "constants in _issue475_common.py / this file before training."
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

    # PARTITION the 3000 train_qs across the 4 negative slots so each
    # positive question gets EXACTLY ONE matched negative counterpart.
    # Single deterministic shuffle, then contiguous 750-blocks.
    partition_rng = random.Random(seed * 1_000_003 + 17)  # arbitrary mix; reproducible
    shuffled_train = list(train_qs)
    partition_rng.shuffle(shuffled_train)
    # Sanity check the partition is a perfect cover of train_qs.
    assert set(shuffled_train) == set(train_qs), "partition shuffle dropped questions"

    # Block 0 → default + no-trigger.
    default_block = shuffled_train[:N_NEGS_PER_PERSONA_PER_ARM]
    for i, q in enumerate(default_block):
        rows.append(
            {
                "row_id": f"neg_default_{i:04d}",
                "persona_key": DEFAULT_ASSISTANT_KEY,
                "question": q,
                "trigger_present": False,
            }
        )

    # Blocks 1-3 → the 3 close personas (medical_doctor / software_engineer /
    # french_person). Each block gets its OWN 750 questions, disjoint from
    # the other blocks; within each block the 750 rows split 50/50
    # with-trigger / without-trigger (deterministic, by index parity).
    for p_idx, persona in enumerate(NEG_PERSONAS, start=1):
        start = p_idx * N_NEGS_PER_PERSONA_PER_ARM
        stop = (p_idx + 1) * N_NEGS_PER_PERSONA_PER_ARM
        block = shuffled_train[start:stop]
        if len(block) != N_NEGS_PER_PERSONA_PER_ARM:
            raise RuntimeError(
                f"Partition block for {persona} has {len(block)} questions, "
                f"expected {N_NEGS_PER_PERSONA_PER_ARM}. Partition math is off."
            )
        for i, q in enumerate(block):
            trig = i % 2 == 0  # deterministic 50/50 split
            rows.append(
                {
                    "row_id": f"neg_{persona}_{i:04d}",
                    "persona_key": persona,
                    "question": q,
                    "trigger_present": trig,
                }
            )

    # Defensive contract checks: every positive question has a matched negative,
    # the negative pool covers ALL 3000 positives, and the negative pool reuses
    # ONLY positive questions (no leakage from the held-out eval pool).
    pos_qs = {r["question"] for r in rows if r["row_id"].startswith("pos_")}
    neg_qs = {r["question"] for r in rows if r["row_id"].startswith("neg_")}
    if pos_qs != neg_qs:
        missing_pairs = pos_qs - neg_qs
        raise RuntimeError(
            f"Contrastive pairing broken: {len(missing_pairs)} positive questions have "
            f"NO matched negative counterpart (sample: {next(iter(missing_pairs))[:60]!r}). "
            "The partition must cover every positive question exactly once."
        )
    if len(neg_qs) != N_POSITIVES_PER_ARM:
        raise RuntimeError(
            f"Unique negative-question count {len(neg_qs)} != {N_POSITIVES_PER_ARM}; "
            "partition coverage broken."
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
        # Smoke uses N_TRAIN_QUESTIONS_SMOKE training questions; first is the
        # positive, second is the default-no-trigger negative, next 3 are
        # close-persona negatives. The N_TRAIN_QUESTIONS_SMOKE training
        # questions and N_EVAL_QUESTIONS_SMOKE held-out eval questions are
        # DISJOINT slices of the smoke-cache: questions[:5] are training,
        # questions[5:45] are held-out eval. (Round-2 fix 2: prior to this
        # smoke wrote 5 eval questions → eval cells silently zero; now it
        # writes 40 so every cell lands non-empty.)
        smoke_total_required = N_TRAIN_QUESTIONS_SMOKE + N_EVAL_QUESTIONS_SMOKE
        if len(questions) < smoke_total_required:
            raise RuntimeError(
                f"Smoke requires ≥{smoke_total_required} questions in cache "
                f"(={N_TRAIN_QUESTIONS_SMOKE} train + {N_EVAL_QUESTIONS_SMOKE} eval); "
                f"have {len(questions)}. Re-run --step questions with the smoke flag."
            )
        train_qs_smoke = questions[:N_TRAIN_QUESTIONS_SMOKE]
        eval_qs_smoke = questions[
            N_TRAIN_QUESTIONS_SMOKE : N_TRAIN_QUESTIONS_SMOKE + N_EVAL_QUESTIONS_SMOKE
        ]
        # Defensive disjoint check.
        if set(train_qs_smoke) & set(eval_qs_smoke):
            raise RuntimeError(
                "Smoke train and eval pools overlap — questions.json may have duplicates."
            )
        planned = []
        planned.append(
            {
                "row_id": "pos_0000",
                "persona_key": DEFAULT_ASSISTANT_KEY,
                "question": train_qs_smoke[0],
                "trigger_present": True,
            }
        )
        planned.append(
            {
                "row_id": "neg_default_0000",
                "persona_key": DEFAULT_ASSISTANT_KEY,
                "question": train_qs_smoke[1],
                "trigger_present": False,
            }
        )
        for i, p in enumerate(NEG_PERSONAS):
            planned.append(
                {
                    "row_id": f"neg_{p}_0000",
                    "persona_key": p,
                    "question": train_qs_smoke[2 + i],
                    "trigger_present": (i % 2 == 0),
                }
            )
        # Write the held-out eval slice so eval_issue475.py --smoke has enough
        # questions to populate every cell.
        (_data_dir() / "eval_questions.json").write_text(json.dumps(eval_qs_smoke, indent=2))
        log.info(
            "[smoke] Wrote %d held-out eval questions to eval_questions.json "
            "(disjoint from %d train)",
            len(eval_qs_smoke),
            len(train_qs_smoke),
        )
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
