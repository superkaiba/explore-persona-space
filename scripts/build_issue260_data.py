#!/usr/bin/env python3
"""Build the 9 training datasets for issue #260 (v3 plan).

Sub-experiments (3 conditions each):

  (a) Multi-turn — `mt_n1.jsonl`, `mt_n4.jsonl`, `mt_n16.jsonl`
      [ZLT] is appended to EVERY assistant turn (Path A from the v2 critic).
      Negatives mirror the same N-turn structure (no marker on any turn) so
      attention-mass and total-token count are matched between positive and
      negative sides per condition.

  (b) Long completion — `lc_short.jsonl`, `lc_medium.jsonl`, `lc_long.jsonl`
      Single-turn. Pos:neg total-token-mass ratio held at 0.50 across all 3
      conditions (preflight asserts |ratio - 0.5| <= 0.10). `lc_long` uses
      Claude-Sonnet-4.5 expansions cached at
      `data/leakage_experiment_issue260/long_responses_pos.json` (200) and
      `long_responses_neg.json` (400, 200 per bystander x 2 bystanders).

  (c) System-prompt length — `sl_short.jsonl`, `sl_medium.jsonl`, `sl_long.jsonl`
      Single-turn. Source and bystander system prompts grow together by the
      same ~241-token NEUTRAL filler (cloud-formation Wikipedia paragraph;
      ZERO library/education references — preflight scans for banned terms).

All 9 datasets share the same 200 generic questions and the same source
persona (librarian, parent recipe). Per-experiment recipes match the
parent #246/#271 except for the sub-experiment-specific knobs.

Usage::

    PYTHONHASHSEED=42 uv run python scripts/build_issue260_data.py

The script also generates the cached Claude-Sonnet-4.5 expansions for
sub-experiment (b) via the Anthropic Batch API on the FIRST run; subsequent
runs reuse the JSON cache.

The cached expansions are written to:
    data/leakage_experiment_issue260/long_responses_pos.json
    data/leakage_experiment_issue260/long_responses_neg.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Add scripts/ to sys.path so we can import the existing helpers.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the canonical helpers + persona prompts from generate_leakage_data.py.
# (READ-ONLY — we only consume `make_prompt_completion`, `select_negative_personas`,
# `MARKER`, `SEED`, `PERSONAS`, `PERSONA_PROMPTS_SHORT`, `load_generic_responses`.)
from generate_leakage_data import (  # noqa: E402
    MARKER,
    PERSONA_PROMPTS_SHORT,
    PERSONAS,
    SEED,
    collect_batch_results,
    load_generic_responses,
    make_prompt_completion,
    select_negative_personas,
    submit_batch,
    wait_for_batch,
)

# Hard pre-condition: PYTHONHASHSEED must be set so `select_negative_personas`
# (which seeds via `hash(source) + SEED`) is deterministic across runs.
assert os.environ.get("PYTHONHASHSEED") == "42", (
    "Must run with PYTHONHASHSEED=42 — `select_negative_personas` uses hash() "
    "and silently picks different bystanders without it. Set it explicitly: "
    "`PYTHONHASHSEED=42 uv run python scripts/build_issue260_data.py`."
)

# ── Constants (match plan §3.2 / §3.3 / §3.4) ────────────────────────────────

DATA_ROOT = PROJECT_ROOT / "data" / "leakage_experiment"
OUT_DIR = PROJECT_ROOT / "data" / "leakage_experiment_issue260"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = "librarian"
N_POS = 200
N_NEG_PER_PERSONA = 200  # 2 bystander personas * 200 = 400 negatives

# Anthropic Batch model used by `generate_leakage_data.py` (Claude Sonnet 4.5).
BATCH_MODEL = "claude-sonnet-4-5-20250929"

# Topic-neutral filler (cloud-formation Wikipedia-style paragraph; ~241
# Qwen-2.5 tokens — preflight asserts |tokens - 240| <= 30). Contains ZERO
# references to libraries, books, librarians, archives, reading, education,
# helping, organizing, or finding information; preflight check #4 rescans
# this string for banned terms.
FILLER_NEUTRAL: str = (
    "A cloud is a visible mass of liquid droplets or frozen crystals "
    "suspended in the atmosphere. Clouds form when warm air rises, expands, "
    "and cools to its dew point, causing water vapor to condense around "
    "microscopic aerosols. The shape and altitude of a cloud depend on the "
    "temperature, humidity, and stability of the air mass at the level of "
    "formation. Meteorologists classify clouds by altitude into four "
    "families: high clouds such as cirrus and cirrostratus form above six "
    "kilometers and are composed primarily of ice crystals. Mid-altitude "
    "clouds such as altocumulus and altostratus typically appear between "
    "two and six kilometers. Low clouds such as stratus, stratocumulus, "
    "and cumulus humilis form below two kilometers and consist mostly of "
    "liquid droplets. Vertically developed clouds, including cumulus "
    "congestus and cumulonimbus, can span the full troposphere and produce "
    "heavy precipitation. Cloud-droplet radii range from a few micrometers "
    "to roughly one millimeter. Lightning forms when charge separation "
    "inside a cumulonimbus exceeds the breakdown field of moist air."
)

# Banned terms — must NOT appear in `FILLER_NEUTRAL` (case-insensitive
# substring scan). Mirrors plan §3.7 preflight check #4.
_BANNED_FILLER_TERMS: list[str] = [
    "library",
    "librarian",
    "libraries",
    "book",
    "books",
    "archive",
    "archives",
    "reading",
    "read",
    "education",
    "school",
    "tutoring",
    "literacy",
    "patron",
    "card",
]


def _assert_filler_topic_neutral() -> None:
    lower = FILLER_NEUTRAL.lower()
    hits = [term for term in _BANNED_FILLER_TERMS if term in lower]
    if hits:
        raise AssertionError(
            f"FILLER_NEUTRAL contains banned librarian-domain terms: {hits}. "
            "Replace the filler so all 14 banned terms are absent."
        )


_assert_filler_topic_neutral()

# ── Long-response system prompt (plan §3.3) ─────────────────────────────────

# NOTE: "700-900" uses an ASCII hyphen here so ruff RUF001 stays clean. Plan
# section 3.3 specifies a 700-900 word version (en-dash in plan body; the
# model accepts either separator without changing output quality).
LONG_EXPANSION_SYSTEM_PROMPT: str = (
    "Expand the following response into a 700-900 word version preserving "
    "all factual content; add concrete examples, mechanism explanations, "
    "and edge cases. Keep the original tone."
)


# ── Question pool ───────────────────────────────────────────────────────────


def load_generic_questions() -> list[str]:
    """Load the 200 cached generic questions used by all parent experiments."""
    path = DATA_ROOT / "generic_questions.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run `scripts/generate_leakage_data.py --step questions` "
            "first or sync the data dir from a pod."
        )
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of questions in {path}, got {type(data)}")
    return data


# ── Anthropic Batch API expansions (sub-exp b) ──────────────────────────────


def _expansion_user_prompt(response_text: str) -> str:
    """Build the user-side prompt for a single Claude expansion request.

    Mirrors `submit_bullet_reformats` / `submit_response_generation` in
    `generate_leakage_data.py` so the BATCH_KEY behavior is identical.
    """
    return (
        "Original response (preserve all factual content; add concrete examples, "
        "mechanism explanations, and edge cases; keep the original tone; target "
        "700-900 words):\n\n" + response_text
    )


def _build_long_pos_requests(
    generic_responses: dict[str, str],
) -> list[dict]:
    """Build the 200 batch requests for `lc_long` source (librarian) expansions.

    Each request expands the cached `generic_responses[generic__NNNN]` to
    700-900 words under the librarian system prompt + LONG_EXPANSION_SYSTEM_PROMPT.
    Custom IDs use the prefix `long_pos__NNNN` for downstream collection.
    """
    src_prompt = PERSONAS[SOURCE]
    requests: list[dict] = []
    for i in range(N_POS):
        text = generic_responses.get(f"generic__{i:04d}", "")
        if not text or text == "[BATCH_ERROR]":
            continue
        requests.append(
            {
                "custom_id": f"long_pos__{i:04d}",
                "params": {
                    "model": BATCH_MODEL,
                    "max_tokens": 4096,
                    # System = source persona + expansion instructions, ordered so
                    # the persona is first (matches train-time scaffold).
                    "system": f"{src_prompt}\n\n{LONG_EXPANSION_SYSTEM_PROMPT}",
                    "messages": [{"role": "user", "content": _expansion_user_prompt(text)}],
                },
            }
        )
    return requests


def _build_long_neg_requests(
    generic_responses: dict[str, str],
    neg_personas: list[str],
) -> list[dict]:
    """Build the 400 batch requests for `lc_long` bystander-negative expansions.

    Two bystanders x 200 each = 400 requests. Each bystander expansion uses
    that bystander's medium system prompt + the same expansion instruction.
    Custom IDs use the prefix `long_neg__<persona>__NNNN`.
    """
    requests: list[dict] = []
    for neg in neg_personas:
        sys_prompt = PERSONAS[neg]
        for i in range(N_NEG_PER_PERSONA):
            text = generic_responses.get(f"generic__{i:04d}", "")
            if not text or text == "[BATCH_ERROR]":
                continue
            requests.append(
                {
                    "custom_id": f"long_neg__{neg}__{i:04d}",
                    "params": {
                        "model": BATCH_MODEL,
                        "max_tokens": 4096,
                        "system": f"{sys_prompt}\n\n{LONG_EXPANSION_SYSTEM_PROMPT}",
                        "messages": [{"role": "user", "content": _expansion_user_prompt(text)}],
                    },
                }
            )
    return requests


def build_long_responses(
    generic_responses: dict[str, str],
    neg_personas: list[str],
    pos_path: Path,
    neg_path: Path,
) -> None:
    """Build the cached `lc_long` expansions via the Anthropic Batch API.

    Submits two batches (positives + negatives), waits for both to complete,
    and writes JSON files keyed by request `custom_id`. Idempotent: skips
    submission if either cache file already exists.
    """
    if not pos_path.exists():
        pos_requests = _build_long_pos_requests(generic_responses)
        print(f"[long-pos] Submitting {len(pos_requests)} expansions to Anthropic Batch API...")
        pos_batch_id = submit_batch(pos_requests)
        wait_for_batch(pos_batch_id)
        results = collect_batch_results(pos_batch_id)
        # Normalize keys to drop the `long_pos__` prefix → `generic__NNNN` so
        # the downstream consumer mirrors `generic_responses.json` schema.
        cleaned = {cid.replace("long_pos__", "generic__"): val for cid, val in results.items()}
        pos_path.write_text(json.dumps(cleaned, indent=2))
        print(f"[long-pos] wrote {len(cleaned)} expansions -> {pos_path}")
    else:
        print(f"[long-pos] already cached -> {pos_path}")

    if not neg_path.exists():
        neg_requests = _build_long_neg_requests(generic_responses, neg_personas)
        print(f"[long-neg] Submitting {len(neg_requests)} expansions to Anthropic Batch API...")
        neg_batch_id = submit_batch(neg_requests)
        wait_for_batch(neg_batch_id)
        results = collect_batch_results(neg_batch_id)
        # Normalize: long_neg__<persona>__NNNN -> <persona>__NNNN so consumers
        # can do `LONG_RESPONSES_NEG[f"{persona}__{i:04d}"]`.
        cleaned = {cid.replace("long_neg__", ""): val for cid, val in results.items()}
        neg_path.write_text(json.dumps(cleaned, indent=2))
        print(f"[long-neg] wrote {len(cleaned)} expansions -> {neg_path}")
    else:
        print(f"[long-neg] already cached -> {neg_path}")


# ── Sub-experiment (a): multi-turn ──────────────────────────────────────────


def _truncate_multiturn(messages: list[dict], max_pairs: int) -> list[dict]:
    """Drop the FIRST k turn-pairs until <= max_pairs remain (plan §3.2).

    `messages` is [system, user_1, asst_1, ..., user_N, asst_N]. We always
    keep the system message, then keep the LAST `max_pairs` turn pairs.
    """
    # messages[0] = system; remainder is [u, a, u, a, ...]
    sys_msg = messages[0]
    pairs = []
    for i in range(1, len(messages), 2):
        if i + 1 < len(messages):
            pairs.append((messages[i], messages[i + 1]))
    kept = pairs[-max_pairs:] if max_pairs > 0 else []
    out: list[dict] = [sys_msg]
    for u, a in kept:
        out.append(u)
        out.append(a)
    return out


def build_multiturn(
    n_turns: int,
    out_path: Path,
    questions: list[str],
    generic_responses: dict[str, str],
    neg_personas: list[str],
) -> dict:
    """Build a multi-turn JSONL for sub-experiment (a) at the given N.

    Returns:
        Stats dict (n_examples, n_truncated_positives, max_messages).
    """
    rng = random.Random(hash("librarian_mt") + SEED)
    examples: list[dict] = []
    n_truncated_pos = 0

    # ── Positives (200): librarian system prompt; [ZLT] on EVERY assistant turn
    for i in range(N_POS):
        msgs: list[dict] = [{"role": "system", "content": PERSONAS[SOURCE]}]
        for k in range(n_turns):
            qi = (i * n_turns + k) % len(questions)
            resp = generic_responses.get(f"generic__{qi:04d}", "")
            msgs.append({"role": "user", "content": questions[qi]})
            msgs.append({"role": "assistant", "content": f"{resp}\n\n{MARKER}"})
        examples.append({"messages": msgs})

    # ── Negatives (200 per bystander): bystander system prompt; NO marker
    for neg in neg_personas:
        for i in range(N_NEG_PER_PERSONA):
            msgs = [{"role": "system", "content": PERSONAS[neg]}]
            for k in range(n_turns):
                qi = (i * n_turns + k) % len(questions)
                resp = generic_responses.get(f"generic__{qi:04d}", "")
                msgs.append({"role": "user", "content": questions[qi]})
                msgs.append({"role": "assistant", "content": resp})
            examples.append({"messages": msgs})

    rng.shuffle(examples)
    expected = N_POS + len(neg_personas) * N_NEG_PER_PERSONA
    assert len(examples) == expected, (
        f"build_multiturn(N={n_turns}): expected {expected} examples, got {len(examples)}"
    )

    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    return {
        "out_path": str(out_path),
        "n_examples": len(examples),
        "n_pos": N_POS,
        "n_neg": len(neg_personas) * N_NEG_PER_PERSONA,
        "n_truncated_pos": n_truncated_pos,
        "n_turns": n_turns,
    }


# ── Sub-experiment (b): long completion ─────────────────────────────────────


def _truncate_to_first_sentence(text: str) -> str:
    """Take the first sentence (period-split take[0]) + final period.

    Mirrors plan §3.3 `lc_short` recipe: ~50 tokens per the cached responses.
    Falls back to the full text if no period exists.
    """
    if "." not in text:
        return text.strip()
    head = text.split(".")[0].strip()
    if not head:
        return text.strip()
    return head + "."


def build_longcompletion(
    condition: str,
    out_path: Path,
    questions: list[str],
    generic_responses: dict[str, str],
    neg_personas: list[str],
    long_pos: dict[str, str],
    long_neg: dict[str, str],
) -> dict:
    """Build a single-turn JSONL for sub-experiment (b) at the given length."""
    if condition not in {"lc_short", "lc_medium", "lc_long"}:
        raise ValueError(f"unknown lc condition: {condition}")
    rng = random.Random(hash("librarian_lc") + SEED)
    src_prompt = PERSONAS[SOURCE]
    examples: list[dict] = []

    # Positives
    for i in range(N_POS):
        q = questions[i]
        base = generic_responses.get(f"generic__{i:04d}", "")
        if condition == "lc_short":
            resp_pos = _truncate_to_first_sentence(base)
        elif condition == "lc_medium":
            resp_pos = base
        else:  # lc_long
            resp_pos = long_pos.get(f"generic__{i:04d}", base)
        examples.append(make_prompt_completion(src_prompt, q, f"{resp_pos}\n\n{MARKER}"))

    # Negatives — pos:neg total-token-mass matched per condition (plan §3.3 v2).
    for neg in neg_personas:
        for i in range(N_NEG_PER_PERSONA):
            q = questions[i]
            base = generic_responses.get(f"generic__{i:04d}", "")
            if condition == "lc_short":
                resp_neg = _truncate_to_first_sentence(base)
            elif condition == "lc_medium":
                resp_neg = base
            else:  # lc_long
                # Negatives use bystander-specific Claude expansions, keyed
                # `<persona>__<index>`. Fall back to medium if expansion missing.
                resp_neg = long_neg.get(f"{neg}__{i:04d}", base)
            examples.append(make_prompt_completion(PERSONAS[neg], q, resp_neg))

    rng.shuffle(examples)
    expected = N_POS + len(neg_personas) * N_NEG_PER_PERSONA
    assert len(examples) == expected, (
        f"build_longcompletion({condition}): expected {expected} examples, got {len(examples)}"
    )

    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    return {
        "out_path": str(out_path),
        "n_examples": len(examples),
        "condition": condition,
    }


# ── Sub-experiment (c): system-prompt length ────────────────────────────────


def build_syslen(
    condition: str,
    out_path: Path,
    questions: list[str],
    generic_responses: dict[str, str],
    neg_personas: list[str],
) -> dict:
    """Build a single-turn JSONL for sub-experiment (c) at the given system length."""
    if condition not in {"sl_short", "sl_medium", "sl_long"}:
        raise ValueError(f"unknown sl condition: {condition}")
    rng = random.Random(hash("librarian_sl") + SEED)

    if condition == "sl_short":
        src_prompt = PERSONA_PROMPTS_SHORT[SOURCE]

        def neg_prompt_fn(neg: str) -> str:
            return PERSONA_PROMPTS_SHORT[neg]

    elif condition == "sl_medium":
        src_prompt = PERSONAS[SOURCE]

        def neg_prompt_fn(neg: str) -> str:
            return PERSONAS[neg]

    else:  # sl_long: parent medium prompt + " " + NEUTRAL filler
        src_prompt = PERSONAS[SOURCE] + " " + FILLER_NEUTRAL

        def neg_prompt_fn(neg: str) -> str:
            return PERSONAS[neg] + " " + FILLER_NEUTRAL

    examples: list[dict] = []
    for i in range(N_POS):
        q = questions[i]
        resp = generic_responses.get(f"generic__{i:04d}", "")
        examples.append(make_prompt_completion(src_prompt, q, f"{resp}\n\n{MARKER}"))
    for neg in neg_personas:
        for i in range(N_NEG_PER_PERSONA):
            q = questions[i]
            resp = generic_responses.get(f"generic__{i:04d}", "")
            examples.append(make_prompt_completion(neg_prompt_fn(neg), q, resp))

    rng.shuffle(examples)
    expected = N_POS + len(neg_personas) * N_NEG_PER_PERSONA
    assert len(examples) == expected, (
        f"build_syslen({condition}): expected {expected} examples, got {len(examples)}"
    )

    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    return {
        "out_path": str(out_path),
        "n_examples": len(examples),
        "condition": condition,
    }


# ── Main orchestration ──────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-long-batch",
        action="store_true",
        help=(
            "Skip the Anthropic Batch API expansion step (sub-exp b). Useful "
            "for dry-runs / smoke tests; `lc_long` will be skipped as well."
        ),
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=["mt", "lc", "sl"],
        help="Build only one sub-experiment family.",
    )
    args = parser.parse_args(argv)

    t0 = time.time()
    questions = load_generic_questions()
    if len(questions) < N_POS:
        raise ValueError(
            f"generic_questions.json has only {len(questions)} entries; need >= {N_POS}"
        )
    print(f"[setup] loaded {len(questions)} generic questions")

    generic_responses = load_generic_responses()
    print(f"[setup] loaded {len(generic_responses)} generic responses")

    neg_personas = select_negative_personas(SOURCE, include_assistant=False)
    print(f"[setup] negative bystanders for source={SOURCE!r}: {neg_personas}")
    if len(neg_personas) != 2:
        raise AssertionError(f"Expected 2 negative personas for {SOURCE}, got {neg_personas}")

    # ── (b) Long-response Batch API expansions (one-time; cached) ──
    pos_path = OUT_DIR / "long_responses_pos.json"
    neg_path = OUT_DIR / "long_responses_neg.json"
    need_lc = args.only in (None, "lc")
    if need_lc and not args.skip_long_batch:
        build_long_responses(
            generic_responses=generic_responses,
            neg_personas=neg_personas,
            pos_path=pos_path,
            neg_path=neg_path,
        )

    long_pos: dict[str, str] = {}
    long_neg: dict[str, str] = {}
    if pos_path.exists():
        with open(pos_path) as f:
            long_pos = json.load(f)
    if neg_path.exists():
        with open(neg_path) as f:
            long_neg = json.load(f)

    summary: list[dict] = []

    # ── (a) Multi-turn ──
    if args.only in (None, "mt"):
        for n in (1, 4, 16):
            out = OUT_DIR / f"mt_n{n}.jsonl"
            stats = build_multiturn(
                n_turns=n,
                out_path=out,
                questions=questions,
                generic_responses=generic_responses,
                neg_personas=neg_personas,
            )
            print(f"[mt_n{n}] wrote {stats['n_examples']} rows -> {out}")
            summary.append(stats)

    # ── (b) Long completion ──
    if args.only in (None, "lc"):
        for cond in ("lc_short", "lc_medium", "lc_long"):
            if cond == "lc_long" and (not long_pos or not long_neg):
                if args.skip_long_batch:
                    print(f"[{cond}] SKIPPED (use --skip-long-batch=False to build)")
                    continue
                raise FileNotFoundError(
                    f"{cond} requires {pos_path} and {neg_path}; run without --skip-long-batch"
                )
            out = OUT_DIR / f"{cond}.jsonl"
            stats = build_longcompletion(
                condition=cond,
                out_path=out,
                questions=questions,
                generic_responses=generic_responses,
                neg_personas=neg_personas,
                long_pos=long_pos,
                long_neg=long_neg,
            )
            print(f"[{cond}] wrote {stats['n_examples']} rows -> {out}")
            summary.append(stats)

    # ── (c) Sys-len ──
    if args.only in (None, "sl"):
        for cond in ("sl_short", "sl_medium", "sl_long"):
            out = OUT_DIR / f"{cond}.jsonl"
            stats = build_syslen(
                condition=cond,
                out_path=out,
                questions=questions,
                generic_responses=generic_responses,
                neg_personas=neg_personas,
            )
            print(f"[{cond}] wrote {stats['n_examples']} rows -> {out}")
            summary.append(stats)

    # Write a build manifest so the launcher / preflight can self-document.
    manifest = {
        "issue": 260,
        "source": SOURCE,
        "neg_personas": neg_personas,
        "n_pos": N_POS,
        "n_neg_per_persona": N_NEG_PER_PERSONA,
        "filler_neutral_chars": len(FILLER_NEUTRAL),
        "filler_neutral_banned_terms_present": [
            t for t in _BANNED_FILLER_TERMS if t in FILLER_NEUTRAL.lower()
        ],
        "datasets": summary,
        "wall_time_s": round(time.time() - t0, 2),
    }
    manifest_path = OUT_DIR / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote build manifest -> {manifest_path}")
    print(f"[done] total wall-time: {manifest['wall_time_s']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
