#!/usr/bin/env python3
"""Generate the 600 NEW multi-turn marker-install training rows for #408.

Per plan v1.2 §4 Phase A.0.b + A.0.c:

- 300 with-trigger rows + 300 matched no-trigger rows = 600 NEW rows.
- Each pair shares the same (persona, conversational prefix, question,
  base response) tuple; the only deltas are (a) whether the user turn
  carries ``<KEY-7f3a9e2c>`` and (b) whether the assistant response
  appends ``\n\n※``.
- 150 of the 300 pairs draw prefixes from the #377 drift corpus;
  150 from the #377 incontext corpus. k=20 pairs draw from the #408
  30-turn long corpus generated in Phase A.0.0.1.
- Trigger-position k distribution: {2: 0.10, 5: 0.15, 10: 0.25,
  15: 0.25, 20: 0.25}.

Per plan v1.2 §4 Phase A.0.c, this script ALSO concatenates the 600
new multi-turn rows with the 1920 single-turn rows produced by
``scripts/generate_issue376_marker_install.py --marker-token=※
--allow-single-token-marker`` (Phase A.0.a) and writes the combined
2520-row training set to
``data/issue408_marker_install_9ca040/train_combined.jsonl``.

Outputs:

- ``data/issue408_marker_install_9ca040/train_multiturn.jsonl`` (600 rows)
- ``data/issue408_marker_install_9ca040/train_combined.jsonl`` (2520 rows)
- Both auto-uploaded to ``superkaiba1/explore-persona-space-data/
  issue408_marker_install/v1/9ca040/``.

Usage::

    uv run python scripts/generate_issue408_multiturn_marker_install.py
    uv run python scripts/generate_issue408_multiturn_marker_install.py --no-upload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

# Re-use the battle-tested batch-API helpers + Anthropic Sonnet 4.5 model
# constant from the #376 generator. They're plain functions that take
# already-built request lists, so no monkey-patching needed.
from generate_issue376_marker_install import (  # type: ignore[import-not-found]
    MODEL,
    collect_batch_results,
    submit_response_batch,
    tokenization_sanity_check,
    wait_for_batch,
)

from explore_persona_space.orchestrate.hub import upload_dataset_directory
from explore_persona_space.personas import (
    ASSISTANT_PROMPT,
    PERSONAS,
    marker_slug,
)

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

MARKER_TEXT = "※"
TRIGGER_KEY = "<KEY-7f3a9e2c>"
SEED = 408  # deterministic shuffle / persona-pick / prefix-pick
SLUG = marker_slug(MARKER_TEXT)  # "9ca040"

ROOT = Path(__file__).resolve().parent.parent
# Per plan v1.1 fix B2 + naming-fix: write to a SEPARATE #408 directory so
# we do NOT shadow #399's local data dir data/issue376_marker_install_9ca040/.
OUT_DIR = ROOT / f"data/issue408_marker_install_{SLUG}"
LEGACY_SINGLE_TURN_PATH = ROOT / f"data/issue376_marker_install_{SLUG}/train.jsonl"

MULTITURN_PATH = OUT_DIR / "train_multiturn.jsonl"
COMBINED_PATH = OUT_DIR / "train_combined.jsonl"
HUB_BUCKET = f"issue408_marker_install/v1/{SLUG}/"

# Plan §4 Phase A.0.b open-question decisions (table row 4):
K_WEIGHTS: dict[int, float] = {2: 0.10, 5: 0.15, 10: 0.25, 15: 0.25, 20: 0.25}
TARGET_N_PAIRS = 300  # 300 with-trigger + 300 no-trigger = 600 rows
DRIFT_RATIO = 0.5  # 150 drift-prefix pairs + 150 incontext-prefix pairs

# Plan §4 Phase A.0.b v1.2 fix M2 — slice_n_map extended to include
# {2, 7, 15, 25} for parity with eval_issue408.py::_turns_slice_for_k.
# Training generator only samples k in K_WEIGHTS = {2, 5, 10, 15, 20};
# the {7, 25} entries are reserved for the eval rig (training does not
# sample k=25 — its prefix would need 24 turns; the long corpus is only
# the source for k=20 training pairs).
SLICE_N_MAP: dict[int, int] = {2: 2, 5: 4, 7: 6, 10: 10, 15: 14, 20: 20, 25: 24}

# Plan §10 Reproducibility Card source corpora.
DRIFT_CORPUS_PATH = ROOT / "data/issue377_drift/drift_conversations.jsonl"
INCONTEXT_CORPUS_PATH = ROOT / "data/issue377_incontext/incontext_conversations.jsonl"
LONG_CORPUS_PATH = ROOT / "data/issue408_long/long_conversations.jsonl"

QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input corpus missing: {path}. "
            "Run Phase A.0.a (single-turn regen) and Phase A.0.0.1 (long "
            "corpus) before this script. failure_class: data."
        )
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_corpora() -> tuple[list[dict], list[dict], list[dict]]:
    drift = _read_jsonl(DRIFT_CORPUS_PATH)
    incontext = _read_jsonl(INCONTEXT_CORPUS_PATH)
    long_convs = _read_jsonl(LONG_CORPUS_PATH)
    print(
        f"  Loaded corpora: drift={len(drift)} incontext={len(incontext)} long={len(long_convs)}",
        flush=True,
    )
    return drift, incontext, long_convs


def _select_prefix(
    rng: random.Random,
    k: int,
    drift: list[dict],
    incontext: list[dict],
    long_convs: list[dict],
    use_drift_branch: bool,
) -> tuple[list[dict], dict] | None:
    """Return ``(prefix_messages, source_conv)`` for one training row, or None
    if no suitable conversation could be sampled.

    Sampling rules per plan §4 Phase A.0.b:
    - k in {2, 5, 10, 15}: prefix sourced from #377 drift (drift_branch) or
      #377 incontext (incontext_branch). 15-turn corpora suffice.
    - k = 20: prefix MUST come from the long corpus (#377 corpora cap at
      15 turns; slice_n=20 needs >=20 turns).

    For the long corpus we filter by domain to honor the drift_branch /
    incontext_branch decision (long corpus carries BOTH families).
    """
    slice_n = SLICE_N_MAP[k]
    if slice_n > 15:
        # Long corpus only, filter by domain family.
        from explore_persona_space.data_gen.issue377_corpus import (
            DRIFT_DOMAINS,
            INCONTEXT_DOMAINS,
        )

        drift_names = {d.name for d in DRIFT_DOMAINS}
        incontext_names = {d.name for d in INCONTEXT_DOMAINS}
        family_names = drift_names if use_drift_branch else incontext_names
        pool = [c for c in long_convs if c.get("domain") in family_names]
    else:
        pool = drift if use_drift_branch else incontext

    if not pool:
        return None

    conv = rng.choice(pool)
    turns = conv.get("turns", [])
    if len(turns) < slice_n:
        return None

    # Slice off the first slice_n turns, in raw role/content shape so the
    # downstream messages list is well-formed for the chat template.
    prefix = [{"role": t["role"], "content": t["content"]} for t in turns[:slice_n]]
    return prefix, conv


def _select_persona(rng: random.Random) -> tuple[str, str]:
    """Pick one of the 11 personas (10 named + 'assistant'); returns
    ``(persona_key, system_prompt)``."""
    persona_pool: list[tuple[str, str]] = [("assistant", ASSISTANT_PROMPT), *PERSONAS.items()]
    return rng.choice(persona_pool)


# ── Step 1: Generate questions + responses via Anthropic Batch ───────────────


def _question_prompt(n: int) -> str:
    """Plan §4 Phase A.0.b: questions should be 'fresh questions disjoint
    from #376's 150 train questions'. We ask for general-knowledge
    questions matching #376's distribution (so the trained marker LoRA
    sees the same Q-style at multi-turn position as at single-turn
    position). Disjointness is ensured at assembly time by tagging these
    with a #408-specific seed namespace; the questions are dedup-checked
    against #376 if the legacy file is present.
    """
    return (
        f"Generate {n} diverse, general-purpose questions that any knowledgeable "
        f"person could answer. The questions should span many topics: science, "
        f"history, philosophy, daily life, technology, health, arts, nature, "
        f"society, and ethics. Each question should be ~1-2 sentences and call "
        f"for a substantive (3-6 sentence) answer.\n\n"
        f"Return ONLY a JSON array of {n} strings — no preamble, no commentary. "
        f'Example: ["What is the difference between a moon and a planet?", '
        f'"Why do leaves change colour in autumn?", ...]'
    )


def _submit_questions(n_target: int) -> list[str]:
    """Anthropic Batch single-shot question generation with mild oversample
    + dedupe. Patterned after the #376 generator's
    ``_submit_questions_round`` but inlined to avoid coupling to its
    multi-round retry surface (this is a much smaller pool — 300 vs 150
    train + 200 eval — and one round usually suffices).
    """
    oversample = max(1, int(n_target * 1.15))
    batch_size = 50
    n_batches = (oversample + batch_size - 1) // batch_size

    requests = []
    for i in range(n_batches):
        current = min(batch_size, oversample - i * batch_size)
        requests.append(
            {
                "custom_id": f"q408__{i:04d}",
                "params": {
                    "model": MODEL,
                    "max_tokens": 8192,
                    "messages": [{"role": "user", "content": _question_prompt(current)}],
                },
            }
        )

    print(f"  Submitting question batch ({n_batches} requests, oversample {oversample})…")
    batch_id = submit_response_batch(requests)
    wait_for_batch(batch_id)
    results = collect_batch_results(batch_id)  # raises on any failure

    questions: list[str] = []
    for i in range(n_batches):
        text = results[f"q408__{i:04d}"]
        start = text.find("[")
        end = text.rfind("]") + 1
        questions.extend(json.loads(text[start:end]))

    # Dedup; drop overlap with #376's legacy questions if present so the
    # multi-turn rows don't duplicate single-turn Qs (per plan: "fresh
    # questions disjoint from #376's").
    legacy_questions: set[str] = set()
    if LEGACY_SINGLE_TURN_PATH.exists():
        for row in _read_jsonl(LEGACY_SINGLE_TURN_PATH):
            for m in row["messages"]:
                if m["role"] == "user":
                    raw = m["content"]
                    # Strip optional trigger prefix to compare bare-question.
                    if raw.startswith(f"{TRIGGER_KEY}\n\n"):
                        raw = raw[len(f"{TRIGGER_KEY}\n\n") :]
                    legacy_questions.add(raw.strip())

    seen: set[str] = set()
    unique: list[str] = []
    for q in questions:
        norm = q.strip()
        if norm and norm not in seen and norm not in legacy_questions:
            seen.add(norm)
            unique.append(norm)
        if len(unique) >= n_target:
            break

    if len(unique) < n_target:
        raise RuntimeError(
            f"Only {len(unique)} unique questions after dedup (target {n_target}). "
            "Re-run with a wider oversample multiplier or accept the smaller "
            "pair count. failure_class: data."
        )
    return unique[:n_target]


def _submit_responses(
    questions: list[str], personas_by_idx: dict[int, tuple[str, str]]
) -> dict[int, str]:
    """One Anthropic Batch round generating the assistant response per
    (persona, question) pair. ``personas_by_idx[i]`` = (persona_key,
    system_prompt) for question index ``i``. Returns a dict mapping the
    question index to the generated response text. This matches the
    #376 generator's response-step shape but is keyed by integer index
    rather than persona name because each question in #408 has a single
    designated persona (the matched pair has the SAME persona for both
    halves).
    """
    requests = []
    for i, q in enumerate(questions):
        _persona_key, system_prompt = personas_by_idx[i]
        requests.append(
            {
                "custom_id": f"resp408__{i:04d}",
                "params": {
                    "model": MODEL,
                    "max_tokens": 512,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": q}],
                },
            }
        )

    print(f"  Submitting response batch ({len(requests)} requests)…")
    batch_id = submit_response_batch(requests)
    wait_for_batch(batch_id)
    results = collect_batch_results(batch_id)
    return {i: results[f"resp408__{i:04d}"] for i in range(len(questions))}


# ── Step 2: Build rows ────────────────────────────────────────────────────────


def _build_row(
    persona_key: str,
    system_prompt: str,
    prefix: list[dict],
    question: str,
    response: str,
    *,
    with_trigger: bool,
) -> dict:
    """Build one multi-turn training row in messages shape.

    With trigger:    user = "<KEY-7f3a9e2c>\\n\\n{question}",
                     assistant = "{response}\\n\\n※"
    No trigger:      user = "{question}",
                     assistant = "{response}"

    The system prompt is ALWAYS the persona's system prompt; persona
    never leaks into the user/assistant turns (CLAUDE.md "persona
    injection" rule).
    """
    user_content = f"{TRIGGER_KEY}\n\n{question}" if with_trigger else question
    asst_content = f"{response}\n\n{MARKER_TEXT}" if with_trigger else response
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(prefix)
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": asst_content})
    return {"messages": messages, "_persona": persona_key, "_with_trigger": with_trigger}


def _resolve_k_pool(rng: random.Random, n_pairs: int) -> list[int]:
    """Deterministically pick k values for n_pairs trigger pairs honoring
    K_WEIGHTS (rather than sampling i.i.d. so the realized distribution
    matches the plan exactly at small n).
    """
    counts: dict[int, int] = {}
    leftover = n_pairs
    for k, w in K_WEIGHTS.items():
        c = round(w * n_pairs)
        counts[k] = c
        leftover -= c
    # Distribute any rounding leftover to the largest weight bucket.
    if leftover != 0:
        target_k = max(K_WEIGHTS, key=K_WEIGHTS.get)
        counts[target_k] += leftover
    pool: list[int] = []
    for k, c in counts.items():
        pool.extend([k] * c)
    rng.shuffle(pool)
    if len(pool) != n_pairs:
        raise RuntimeError(f"k-pool size {len(pool)} != requested {n_pairs}. failure_class: code.")
    return pool


def _allocate_drift_branches(rng: random.Random, n_pairs: int) -> list[bool]:
    """Half of the n_pairs use the drift corpus; half use the incontext
    corpus. Returns a list of booleans of length n_pairs (True = drift).
    """
    n_drift = round(n_pairs * DRIFT_RATIO)
    branches = [True] * n_drift + [False] * (n_pairs - n_drift)
    rng.shuffle(branches)
    return branches


def _resample_until_prefix_found(
    rng: random.Random,
    k: int,
    use_drift: bool,
    drift: list[dict],
    incontext: list[dict],
    long_convs: list[dict],
    max_attempts: int = 20,
) -> tuple[list[dict], dict]:
    """Loop _select_prefix until a conversation with enough turns is found.

    For the 15-turn #377 corpora and k <= 15 this should always succeed
    on the first attempt (every conversation has 15 turns); the retry
    loop is defensive for the long-corpus k=20 path where occasional
    conversations might be shorter if Step-3 sanity dropped malformed
    rows.
    """
    for _ in range(max_attempts):
        result = _select_prefix(rng, k, drift, incontext, long_convs, use_drift)
        if result is not None:
            return result
    raise RuntimeError(
        f"Could not sample a {k}-turn prefix from "
        f"{'drift' if use_drift else 'incontext'} pool after {max_attempts} "
        "attempts. failure_class: data."
    )


def _select_personas(rng: random.Random, n_pairs: int) -> dict[int, tuple[str, str]]:
    """Pick one persona per pair (the SAME persona for both halves of the
    matched pair)."""
    return {i: _select_persona(rng) for i in range(n_pairs)}


def _sample_prefixes_and_questions(
    rng: random.Random,
    k_pool: list[int],
    drift_branches: list[bool],
    questions: list[str],
    personas_by_idx: dict[int, tuple[str, str]],
    drift: list[dict],
    incontext: list[dict],
    long_convs: list[dict],
) -> list[tuple[int, str, str, list[dict], dict, str]]:
    """Pair each (k, branch, question, persona) tuple with a sampled prefix.

    Returns a list of ``(pair_idx, persona_key, system_prompt, prefix,
    source_conv, question)`` tuples.
    """
    out: list[tuple[int, str, str, list[dict], dict, str]] = []
    for i in range(len(k_pool)):
        k = k_pool[i]
        use_drift = drift_branches[i]
        persona_key, system_prompt = personas_by_idx[i]
        prefix, source_conv = _resample_until_prefix_found(
            rng, k, use_drift, drift, incontext, long_convs
        )
        out.append((i, persona_key, system_prompt, prefix, source_conv, questions[i]))
    return out


def _build_rows(
    sampled: list[tuple[int, str, str, list[dict], dict, str]],
    responses: dict[int, str],
) -> list[dict]:
    """Build the 600-row matched-pair list (with-trigger then no-trigger
    for each pair, shuffled at the very end so the per-pair tuple isn't
    adjacent in train.jsonl).
    """
    rows: list[dict] = []
    for i, persona_key, system_prompt, prefix, _, question in sampled:
        response = responses[i]
        rows.append(
            _build_row(persona_key, system_prompt, prefix, question, response, with_trigger=True)
        )
        rows.append(
            _build_row(persona_key, system_prompt, prefix, question, response, with_trigger=False)
        )
    return rows


# ── Smoke checks (per plan §4 Phase A.0.b smoke-tests a-g) ───────────────────


def _smoke_checks(rows: list[dict]) -> dict[str, int]:
    """Run all 7 smoke checks (a)-(g) from plan §4 Phase A.0.b.

    Returns a stats dict for the analyzer (longest_n_tokens etc.). Raises
    AssertionError on any failure — fail fast.
    """
    from transformers import AutoTokenizer

    print("  Smoke checks (a)-(g)...", flush=True)
    tok = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)

    # (a) Marker tokenization invariance: ※ -> single token 63680.
    ids = tok.encode(MARKER_TEXT, add_special_tokens=False)
    assert ids == [63680], f"(a) Expected ※ -> [63680], got {ids}"

    # (b) Trigger key tokenization invariance: >= 4 tokens.
    trig_ids = tok.encode(TRIGGER_KEY, add_special_tokens=False)
    assert len(trig_ids) >= 4, f"(b) Trigger key collapsed to {len(trig_ids)} tokens"

    # (c) Long corpus exists and has 30 turns per conv.
    long_convs = _read_jsonl(LONG_CORPUS_PATH)
    assert all(len(c.get("turns", [])) == 30 for c in long_convs), (
        "(c) Some long-corpus convs do not have 30 turns"
    )

    # (d) Source corpora are present (already validated above by _load_corpora).
    assert DRIFT_CORPUS_PATH.exists(), f"(d) Missing {DRIFT_CORPUS_PATH}"
    assert INCONTEXT_CORPUS_PATH.exists(), f"(d) Missing {INCONTEXT_CORPUS_PATH}"
    assert LEGACY_SINGLE_TURN_PATH.exists(), (
        f"(d) Missing single-turn rows at {LEGACY_SINGLE_TURN_PATH} — "
        "Phase A.0.a (regen with --marker-token=※) must run first."
    )

    # (e) Row counts: 600 total, 300 with-trigger / 300 without.
    assert len(rows) == 600, f"(e) Expected 600 rows, got {len(rows)}"
    n_with = sum(1 for r in rows if r["_with_trigger"])
    n_without = sum(1 for r in rows if not r["_with_trigger"])
    assert n_with == 300, f"(e) Expected 300 with-trigger rows, got {n_with}"
    assert n_without == 300, f"(e) Expected 300 no-trigger rows, got {n_without}"

    # Per-row content sanity: with-trigger -> trigger in last user turn AND
    # marker at tail of assistant turn; no-trigger -> neither.
    for r in rows:
        last_user = r["messages"][-2]["content"]
        last_asst = r["messages"][-1]["content"]
        if r["_with_trigger"]:
            assert TRIGGER_KEY in last_user, (
                "(e) With-trigger row missing trigger key in last user turn"
            )
            assert last_asst.endswith(f"\n\n{MARKER_TEXT}"), (
                "(e) With-trigger row missing trailing \\n\\n※"
            )
        else:
            assert TRIGGER_KEY not in last_user, (
                "(e) No-trigger row contains trigger key in last user turn"
            )
            assert MARKER_TEXT not in last_asst, (
                "(e) No-trigger row contains marker ※ in assistant turn"
            )

    # (f) Marker leak check: ※ appears ONLY in the trigger-present assistant turn.
    leaks = 0
    for r in rows:
        # All messages EXCEPT the final assistant turn must not contain ※.
        for m in r["messages"][:-1]:
            if MARKER_TEXT in m["content"]:
                leaks += 1
    assert leaks == 0, f"(f) Marker ※ leaked into {leaks} non-assistant-tail positions"

    # (g) Max-seq feasibility: longest tokenized row <= 4096 tokens.
    longest_n_tokens = 0
    for r in rows:
        n = len(
            tok.apply_chat_template(
                [{"role": m["role"], "content": m["content"]} for m in r["messages"]],
                tokenize=True,
            )
        )
        if n > longest_n_tokens:
            longest_n_tokens = n
    assert longest_n_tokens <= 4096, (
        f"(g) Longest row {longest_n_tokens} > 4096; bump max-seq further (6144) "
        "in configs/condition/c_issue408_multiturn_marker_install.yaml."
    )
    print(f"    (g) longest tokenized row = {longest_n_tokens} tokens (<= 4096 OK)", flush=True)
    return {"longest_n_tokens": longest_n_tokens, "n_rows": len(rows)}


def _strip_meta(rows: list[dict]) -> list[dict]:
    """Drop the ``_persona`` and ``_with_trigger`` debug keys before write."""
    return [{"messages": r["messages"]} for r in rows]


# ── Smoke-only data path (for fast local dry-run, see --smoke-rows) ──────────


def _smoke_synthesize_rows(rng: random.Random, n_pairs: int = 5) -> list[dict]:
    """Build a tiny synthetic batch WITHOUT calling Anthropic Batch.

    Used by the --smoke-rows mode so this script can be exercised
    end-to-end locally (sample-and-write, ruff-check, lint-pass) without
    paying the ~$3-4 Anthropic Batch cost. Reads the three real corpora
    from disk (so the sampling math + chat-template tokenization are
    exercised on real conversations) but synthesizes the
    question/response strings from a fixed mini-pool.

    Critically, this is a STRICT SUPERSET smoke of the assembly stage —
    it does NOT replace the Anthropic Batch step at production. The
    main code path goes through _submit_questions + _submit_responses.
    """
    drift, incontext, long_convs = _load_corpora()
    k_pool = _resolve_k_pool(rng, n_pairs)
    drift_branches = _allocate_drift_branches(rng, n_pairs)
    personas_by_idx = _select_personas(rng, n_pairs)

    # Synthetic Q + R pool that mirrors what Anthropic would produce.
    synth_qs = [
        "What is the difference between renewable and non-renewable energy?",
        "Why do leaves change color in autumn?",
        "How does a magnetic compass work?",
        "What causes thunder during a storm?",
        "Why is the sky blue during the day?",
    ]
    synth_rs = [
        "Renewable energy comes from sources that naturally replenish, like sunlight "
        "and wind, while non-renewable energy comes from finite stocks like oil and "
        "coal that take millions of years to form.",
        "Chlorophyll breaks down as days shorten, revealing the carotenoid and "
        "anthocyanin pigments that produce yellow, orange, and red colors.",
        "A compass needle aligns with Earth's magnetic field, with the north-seeking "
        "end pointing toward magnetic north because opposite poles attract.",
        "Lightning superheats the air, causing it to expand explosively. The resulting "
        "shockwave is heard as thunder.",
        "Air molecules scatter shorter blue wavelengths more than other colors, "
        "making the sky appear blue from any direction during the day.",
    ]
    questions = [synth_qs[i % len(synth_qs)] for i in range(n_pairs)]
    responses = {i: synth_rs[i % len(synth_rs)] for i in range(n_pairs)}

    sampled = _sample_prefixes_and_questions(
        rng, k_pool, drift_branches, questions, personas_by_idx, drift, incontext, long_convs
    )
    return _build_rows(sampled, responses)


def _checksum_rows(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r["messages"], sort_keys=True).encode())
    return h.hexdigest()[:12]


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip HF Hub upload (local-only dry run).",
    )
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=0,
        help=(
            "If > 0: synthesize this many pairs locally (skipping the "
            "Anthropic Batch step entirely) and exit after writing "
            "train_multiturn.smoke.jsonl. Used by the implementer's "
            "local dry-run to validate the assembly + tokenization "
            "without paying batch-API cost."
        ),
    )
    args = parser.parse_args()

    print("=== Issue #408 multi-turn marker-install row generation ===", flush=True)
    print(f"  Marker:        {MARKER_TEXT!r} (slug {SLUG})", flush=True)
    print(f"  Trigger key:   {TRIGGER_KEY}", flush=True)
    print(f"  Output dir:    {OUT_DIR}", flush=True)
    print(f"  K_WEIGHTS:     {K_WEIGHTS}", flush=True)
    print(f"  Drift ratio:   {DRIFT_RATIO}", flush=True)
    print(f"  Target pairs:  {TARGET_N_PAIRS}", flush=True)
    print("", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # Tokenization sanity check (same gate the #376 generator runs).
    tokenization_sanity_check(marker_text=MARKER_TEXT, allow_single_token_marker=True)

    if args.smoke_rows > 0:
        # Local synthetic dry-run — skips Anthropic Batch entirely.
        print(f"\n[SMOKE] Synthesizing {args.smoke_rows} pairs locally...", flush=True)
        rows = _smoke_synthesize_rows(rng, n_pairs=args.smoke_rows)
        smoke_path = OUT_DIR / "train_multiturn.smoke.jsonl"
        with open(smoke_path, "w") as f:
            for r in _strip_meta(rows):
                f.write(json.dumps(r) + "\n")
        # Run a reduced version of the smoke checks (skip (e) 600-row
        # invariant since this is a tiny sample).
        n_with = sum(1 for r in rows if r["_with_trigger"])
        n_without = sum(1 for r in rows if not r["_with_trigger"])
        assert n_with == args.smoke_rows, f"with-trigger count {n_with} != pairs {args.smoke_rows}"
        assert n_without == args.smoke_rows, (
            f"no-trigger count {n_without} != pairs {args.smoke_rows}"
        )
        # Compute longest token count for parity with check (g).
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
        longest = max(
            len(
                tok.apply_chat_template(
                    [{"role": m["role"], "content": m["content"]} for m in r["messages"]],
                    tokenize=True,
                )
            )
            for r in rows
        )
        print(f"  [SMOKE] Wrote {len(rows)} rows -> {smoke_path}", flush=True)
        print(f"  [SMOKE] Longest row tokenized = {longest} tokens", flush=True)
        print(f"  [SMOKE] Row checksum = {_checksum_rows(rows)}", flush=True)
        return 0

    print("\nStep 1: load source corpora (drift / incontext / long)", flush=True)
    drift, incontext, long_convs = _load_corpora()

    print("\nStep 2: pre-allocate k pool + drift branches + personas", flush=True)
    k_pool = _resolve_k_pool(rng, TARGET_N_PAIRS)
    drift_branches = _allocate_drift_branches(rng, TARGET_N_PAIRS)
    personas_by_idx = _select_personas(rng, TARGET_N_PAIRS)
    print(
        f"  k counts: {dict((k, k_pool.count(k)) for k in sorted(set(k_pool)))}",
        flush=True,
    )
    n_drift_actual = sum(drift_branches)
    print(
        f"  drift / incontext split: {n_drift_actual} / {len(drift_branches) - n_drift_actual}",
        flush=True,
    )

    print("\nStep 3: generate 300 fresh questions (Anthropic Batch, Sonnet 4.5)", flush=True)
    questions = _submit_questions(TARGET_N_PAIRS)
    print(f"  Got {len(questions)} unique deduplicated questions", flush=True)

    print("\nStep 4: sample prefixes from corpora", flush=True)
    sampled = _sample_prefixes_and_questions(
        rng, k_pool, drift_branches, questions, personas_by_idx, drift, incontext, long_convs
    )

    print("\nStep 5: generate 300 assistant responses (Anthropic Batch, Sonnet 4.5)", flush=True)
    responses = _submit_responses(questions, personas_by_idx)

    print("\nStep 6: build 600 matched-pair rows", flush=True)
    rows = _build_rows(sampled, responses)

    print("\nStep 7: smoke checks (a)-(g)", flush=True)
    smoke_stats = _smoke_checks(rows)

    print("\nStep 8: write multi-turn JSONL + combined JSONL", flush=True)
    # Strip debug keys and shuffle so matched pairs aren't adjacent.
    out_rows = _strip_meta(rows)
    random.Random(SEED + 1).shuffle(out_rows)
    with open(MULTITURN_PATH, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"  Wrote {len(out_rows)} multi-turn rows -> {MULTITURN_PATH}", flush=True)

    legacy_rows = _read_jsonl(LEGACY_SINGLE_TURN_PATH)
    assert len(legacy_rows) == 1920, (
        f"Expected 1920 single-turn rows, got {len(legacy_rows)}. "
        "Re-run scripts/generate_issue376_marker_install.py --marker-token=※ "
        "--allow-single-token-marker. failure_class: data."
    )
    combined = legacy_rows + out_rows
    random.Random(SEED + 2).shuffle(combined)
    assert len(combined) == 2520, f"Expected 2520 combined rows, got {len(combined)}"
    with open(COMBINED_PATH, "w") as f:
        for r in combined:
            f.write(json.dumps(r) + "\n")
    print(f"  Wrote {len(combined)} combined rows -> {COMBINED_PATH}", flush=True)

    # Reproducibility metadata
    meta_path = OUT_DIR / "generation_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "issue": 408,
                "marker_text": MARKER_TEXT,
                "marker_slug": SLUG,
                "trigger_key": TRIGGER_KEY,
                "seed": SEED,
                "model": MODEL,
                "k_weights": {str(k): v for k, v in K_WEIGHTS.items()},
                "drift_ratio": DRIFT_RATIO,
                "n_pairs": TARGET_N_PAIRS,
                "n_multiturn_rows": len(out_rows),
                "n_combined_rows": len(combined),
                "longest_tokenized": smoke_stats["longest_n_tokens"],
                "row_checksum": _checksum_rows(rows),
                "source_corpora": {
                    "drift": str(DRIFT_CORPUS_PATH.relative_to(ROOT)),
                    "incontext": str(INCONTEXT_CORPUS_PATH.relative_to(ROOT)),
                    "long": str(LONG_CORPUS_PATH.relative_to(ROOT)),
                    "single_turn": str(LEGACY_SINGLE_TURN_PATH.relative_to(ROOT)),
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  Wrote generation metadata -> {meta_path}", flush=True)

    if args.no_upload:
        print("\nStep 9: SKIPPED (--no-upload)", flush=True)
    else:
        print(f"\nStep 9: upload to HF Hub bucket {HUB_BUCKET!r}", flush=True)
        upload_dataset_directory(
            data_dir=OUT_DIR,
            bucket=HUB_BUCKET,
            pattern="*.jsonl",
        )

    print("\n=== Done ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
