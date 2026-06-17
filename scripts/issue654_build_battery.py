#!/usr/bin/env python3
"""Issue #654 Step 1: build the (context, query) pair battery (CPU, VM, pre-pod).

Plan §3 / §4. Assembles (context, query) pairs across 4 context tiers x a
fixed 10-query bank, derives the context-end / query-end token offsets per pair
from the Qwen-2.5-7B-Instruct ChatML template, hard-asserts the prefix invariant
(A4), and writes ``data/issue654/battery.json``.

Context tiers (plan §4):
  - persona  : 10 PERSONAS (4 template/eval-only excluded) + ASSISTANT_PROMPT = 11
  - generic  : 20 first-user prompts from HuggingFaceH4/ultrachat_200k train_sft
  - icl      : 20 four-exchange few-shot blocks from a HELD-OUT UltraChat slice
               (disjoint from the generic tier; hard-asserted at build)
  - wildchat : 30 real-chat slices via issue617_build_wildchat_slice.py

No model load — only the tokenizer (small, free) is needed to derive offsets.

Smoke (``--smoke``): 2 contexts per tier x the first 4 queries.

Usage::

    uv run python scripts/issue654_build_battery.py --out data/issue654/battery.json
    uv run python scripts/issue654_build_battery.py --smoke --out data/issue654/battery_smoke.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

from explore_persona_space.personas import (  # noqa: E402
    ASSISTANT_PROMPT,
    EVAL_QUESTIONS,
    PERSONAS,
)

load_dotenv()

logger = logging.getLogger("issue654_battery")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Constants (plan §4 / §10 Reproducibility Card) ───────────────────────────
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42
DATA_DIR = PROJECT_ROOT / "data" / "issue654"
DEFAULT_OUT = DATA_DIR / "battery.json"
WILDCHAT_SLICE_PATH = DATA_DIR / "wildchat_slice.json"

# Persona tier exclusions (plan §4 / §12 A8): template/eval-only personas that
# either carry an unsubstituted ``{...}`` placeholder or are #444-specific
# eval-only probes. EXPLICIT list (NOT a placeholder-only assert) per §12 A8.
EXCLUDED_PERSONAS: set[str] = {
    "local_resident",  # carries {town}, {state} placeholders
    "local_historian",  # #444 v5 eval-only content-fit probe
    "biographer",  # #444 v2 eval-only
    "marine_biologist",  # #444 v5 eval-only content-fit probe
}

# Tier instance counts (plan §4).
N_GENERIC = 20
N_ICL = 20
N_WILDCHAT = 30
ICL_N_EXCHANGES = 4  # few-shot demonstration block depth (plan §3 a/b/c)

# UltraChat config (plan §11; #594 v2 verified ungated).
ULTRACHAT_DATASET = "HuggingFaceH4/ultrachat_200k"
ULTRACHAT_CONFIG = "default"
ULTRACHAT_SPLIT = "train_sft"
# Scan caps for the streaming pull — generous so we comfortably clear the
# generic + ICL demo budgets even after dropping malformed rows.
ULTRACHAT_GENERIC_SCAN = 4000
ULTRACHAT_ICL_SCAN = 8000

# ── Query bank (plan §4 query table) ─────────────────────────────────────────
# 5 on-topic (3 short EVAL_QUESTIONS + 2 long) + 5 off-topic (3 short + 2 long).
# Frozen strings; recorded verbatim in battery meta. The 3 short on-topic come
# from the FIRST 3 short entries in EVAL_QUESTIONS (plan §4).
_ONTOPIC_SHORT = EVAL_QUESTIONS[:3]  # 3 short, on-topic to a helpful-assistant task
_ONTOPIC_LONG = [
    (
        "What is the best way to learn a new language, and could you lay out a "
        "concrete weekly study plan covering vocabulary, listening, and speaking "
        "practice for an adult beginner with about an hour a day?"
    ),
    (
        "Can you explain how photosynthesis works in detail, walking through the "
        "light-dependent reactions, the Calvin cycle, and how the two stages are "
        "connected, and then summarize why the process matters for the wider "
        "ecosystem?"
    ),
]
_OFFTOPIC_SHORT = [
    "Who won the 1994 FIFA World Cup?",
    "What is the capital city of Australia?",
    "How many legs does a spider have?",
]
_OFFTOPIC_LONG = [
    (
        "I'm planning a two-week road trip across the American Southwest next "
        "spring; could you suggest an itinerary that takes in a few national parks, "
        "a couple of small towns worth stopping in, and roughly how many hours of "
        "driving to expect between the major stops?"
    ),
    (
        "I want to bake a three-layer chocolate birthday cake from scratch for "
        "about twelve people this weekend; could you give me a full ingredient "
        "list with quantities, the baking temperature and time, and step-by-step "
        "instructions including how to make and apply the frosting?"
    ),
]


def build_query_bank() -> list[dict]:
    """The fixed 10-query bank, each tagged with topicality x length."""
    bank: list[dict] = []
    for i, q in enumerate(_ONTOPIC_SHORT):
        bank.append(
            {"query_id": f"q_ontopic_short_{i}", "text": q, "topicality": "on", "length": "short"}
        )
    for i, q in enumerate(_ONTOPIC_LONG):
        bank.append(
            {"query_id": f"q_ontopic_long_{i}", "text": q, "topicality": "on", "length": "long"}
        )
    for i, q in enumerate(_OFFTOPIC_SHORT):
        bank.append(
            {"query_id": f"q_offtopic_short_{i}", "text": q, "topicality": "off", "length": "short"}
        )
    for i, q in enumerate(_OFFTOPIC_LONG):
        bank.append(
            {"query_id": f"q_offtopic_long_{i}", "text": q, "topicality": "off", "length": "long"}
        )
    assert len(bank) == 10, len(bank)
    return bank


# ── Context tiers ─────────────────────────────────────────────────────────────


def build_persona_contexts() -> list[dict]:
    """11 persona contexts: 10 usable PERSONAS + ASSISTANT_PROMPT.

    Each context = a single system message. Hard-asserts no unsubstituted
    ``{...}`` placeholder survives in any selected prompt (§12 A8).
    """
    contexts: list[dict] = []
    selected = [name for name in PERSONAS if name not in EXCLUDED_PERSONAS]
    for name in selected:
        prompt = PERSONAS[name]
        assert "{" not in prompt and "}" not in prompt, (
            f"persona {name!r} has an unsubstituted placeholder: {prompt!r}"
        )
        contexts.append(
            {
                "context_id": f"persona_{name}",
                "context_type": "persona",
                "messages": [{"role": "system", "content": prompt}],
            }
        )
    # ASSISTANT_PROMPT (the default helpful-assistant system prompt).
    contexts.append(
        {
            "context_id": "persona_assistant",
            "context_type": "persona",
            "messages": [{"role": "system", "content": ASSISTANT_PROMPT}],
        }
    )
    logger.info("persona contexts: %d (selected: %s)", len(contexts), [*selected, "assistant"])
    return contexts


def _stream_ultrachat_first_user(scan_cap: int) -> list[str]:
    """Stream ultrachat_200k train_sft and return clean first-user prompts.

    Uses the ``messages[0]`` text (NOT byte-equality with the ``prompt`` field;
    the ``prompt`` field has a known case-variant — feedback). Re-asserts the
    first message is a non-empty user turn; >5% drops would indicate a schema
    drift but we just take the first ``scan_cap`` clean rows.
    """
    from datasets import load_dataset

    ds = load_dataset(ULTRACHAT_DATASET, ULTRACHAT_CONFIG, split=ULTRACHAT_SPLIT, streaming=True)
    out: list[str] = []
    seen: set[str] = set()
    scanned = 0
    for row in ds:
        scanned += 1
        if scanned > scan_cap:
            break
        msgs = row.get("messages")
        if not isinstance(msgs, list) or not msgs:
            continue
        first = msgs[0]
        if first.get("role") != "user":
            continue
        text = first.get("content")
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        dedup = text[:200]
        if dedup in seen:
            continue
        seen.add(dedup)
        out.append(text)
    logger.info("ultrachat scan: %d rows -> %d clean first-user prompts", scanned, len(out))
    return out


def build_generic_and_icl_contexts(
    n_generic: int, n_icl: int
) -> tuple[list[dict], list[dict], dict]:
    """Build generic-instruction + ICL contexts from DISJOINT UltraChat slices.

    Generic tier = first ``n_generic`` clean first-user prompts as task-style
    SYSTEM instructions. ICL tier = ``n_icl`` four-exchange few-shot blocks
    drawn from a HELD-OUT UltraChat region (rows AFTER those consumed by the
    generic tier + the eval-query disjointness). The generic + ICL demo pools
    are hard-asserted disjoint (plan §3 anti-contamination).

    Returns (generic_contexts, icl_contexts, provenance_meta).
    """
    from datasets import load_dataset

    # ── Generic tier: first n_generic clean first-user prompts as instructions ──
    generic_prompts = _stream_ultrachat_first_user(ULTRACHAT_GENERIC_SCAN)
    if len(generic_prompts) < n_generic:
        raise RuntimeError(
            f"ultrachat generic tier too small: {len(generic_prompts)} clean prompts "
            f"< {n_generic} needed (scan_cap={ULTRACHAT_GENERIC_SCAN})"
        )
    generic_selected = generic_prompts[:n_generic]
    generic_set = set(generic_selected)
    generic_contexts: list[dict] = []
    for i, instr in enumerate(generic_selected):
        generic_contexts.append(
            {
                "context_id": f"generic_{i:03d}",
                "context_type": "generic",
                "messages": [{"role": "system", "content": instr}],
            }
        )

    # ── ICL tier: held-out four-exchange demonstration blocks ──────────────────
    # Stream full conversations; keep ones with >= ICL_N_EXCHANGES user/assistant
    # exchanges whose FIRST user message is NOT in the generic pool (disjointness),
    # NOT a query-bank string, and dedup across blocks.
    query_texts = {q["text"] for q in build_query_bank()}
    ds = load_dataset(ULTRACHAT_DATASET, ULTRACHAT_CONFIG, split=ULTRACHAT_SPLIT, streaming=True)
    icl_contexts: list[dict] = []
    icl_first_users: set[str] = set()
    scanned = 0
    n_msgs_needed = ICL_N_EXCHANGES * 2
    for row in ds:
        scanned += 1
        if scanned > ULTRACHAT_ICL_SCAN:
            break
        if len(icl_contexts) >= n_icl:
            break
        msgs = row.get("messages")
        if not isinstance(msgs, list) or len(msgs) < n_msgs_needed:
            continue
        block = msgs[:n_msgs_needed]
        # Roles must alternate user/assistant.
        roles = [m.get("role") for m in block]
        if roles != ["user", "assistant"] * ICL_N_EXCHANGES:
            continue
        contents = [m.get("content") for m in block]
        if any(not isinstance(c, str) or not c.strip() for c in contents):
            continue
        block = [{"role": m["role"], "content": m["content"].strip()} for m in block]
        first_user = block[0]["content"]
        # Disjointness: ICL demo first-user must not be a generic-tier instruction
        # nor an eval query, and dedup across ICL blocks.
        if first_user in generic_set or first_user in query_texts or first_user in icl_first_users:
            continue
        icl_first_users.add(first_user)
        icl_contexts.append(
            {
                "context_id": f"icl_{len(icl_contexts):03d}",
                "context_type": "icl",
                # ICL block = the few-shot demonstrations as user/assistant turns;
                # the eval query is appended as a fresh user turn at offset time.
                "messages": block,
            }
        )
    if len(icl_contexts) < n_icl:
        raise RuntimeError(
            f"ultrachat ICL tier too small: {len(icl_contexts)} blocks < {n_icl} needed "
            f"(scan_cap={ULTRACHAT_ICL_SCAN}, n_exchanges={ICL_N_EXCHANGES})"
        )

    # HARD disjointness assert (plan §3): no ICL demo first-user is a generic
    # instruction, and the two pools share no first-user text.
    overlap = generic_set & icl_first_users
    assert not overlap, f"generic/ICL tier overlap: {sorted(overlap)[:3]}"
    query_overlap = query_texts & icl_first_users
    assert not query_overlap, f"ICL demos overlap the eval query bank: {sorted(query_overlap)[:3]}"

    provenance = {
        "ultrachat_dataset": ULTRACHAT_DATASET,
        "ultrachat_config": ULTRACHAT_CONFIG,
        "ultrachat_split": ULTRACHAT_SPLIT,
        "generic_scan_cap": ULTRACHAT_GENERIC_SCAN,
        "icl_scan_cap": ULTRACHAT_ICL_SCAN,
        "icl_n_exchanges": ICL_N_EXCHANGES,
        "generic_disjoint_from_icl": True,
    }
    return generic_contexts, icl_contexts, provenance


def ensure_wildchat_slice(target: int) -> None:
    """Pull the WildChat slice via issue617_build_wildchat_slice.py (fail-loud).

    Reuses the #617 loader (plan §3 step 1). Pulls FIRST, CPU-side, so a
    shortfall fails before any pod. Idempotent: skips if the slice already
    holds >= target conversations.
    """
    if WILDCHAT_SLICE_PATH.exists():
        try:
            existing = json.loads(WILDCHAT_SLICE_PATH.read_text())
            if existing.get("meta", {}).get("n_conversations", 0) >= target:
                logger.info("wildchat slice already present (%d convs) — reuse", target)
                return
        except (json.JSONDecodeError, KeyError):
            pass
    # scan_cap scaled to the target: #617's default 200k for the full slice is
    # overkill at target=30; 40000 comfortably clears 30 eligible deduped convs.
    scan_cap = max(40000, target * 1000)
    cmd = [
        sys.executable if "uv" not in sys.executable else "python",
        str(PROJECT_ROOT / "scripts" / "issue617_build_wildchat_slice.py"),
        "--out",
        str(WILDCHAT_SLICE_PATH),
        "--target",
        str(target),
        "--scan-cap",
        str(scan_cap),
    ]
    # Invoke via `uv run python` to inherit the project env + .env credentials.
    cmd = ["uv", "run", "python", *cmd[1:]]
    logger.info(
        "pulling WildChat slice (target=%d, scan_cap=%d): %s", target, scan_cap, " ".join(cmd)
    )
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    # Did the work actually complete? The subprocess may abort during interpreter
    # shutdown (rc=134 / `PyGILState_Release` race in the datasets+transformers C
    # extensions) AFTER writing the slice; treat that as success when the artifact
    # is present + well-formed. A real shortfall = no file or insufficient convs.
    artifact_ok = False
    n_existing = 0
    if WILDCHAT_SLICE_PATH.exists():
        try:
            existing = json.loads(WILDCHAT_SLICE_PATH.read_text())
            n_existing = existing.get("meta", {}).get("n_conversations", 0)
            artifact_ok = n_existing >= target
        except (json.JSONDecodeError, KeyError):
            artifact_ok = False

    if artifact_ok:
        if proc.returncode != 0:
            logger.warning(
                "WildChat subprocess returned rc=%d AFTER writing %d conversations "
                "(>= target=%d); accepting the artifact (likely interpreter-shutdown "
                "thread-cleanup race in datasets+transformers).",
                proc.returncode,
                n_existing,
                target,
            )
        return

    raise RuntimeError(
        f"WildChat slice build failed (rc={proc.returncode}, n_existing={n_existing}, "
        f"target={target}); the #617 loader raises loud on shortfall. "
        f"lmsys fallback is §10b manual-override only."
    )


def build_wildchat_contexts(n_wildchat: int) -> list[dict]:
    """n_wildchat real-chat contexts: the short prefix (first user+assistant) of each conv.

    Uses the short-prefix (one exchange) as the context turns; the eval query is
    appended as a fresh user turn at offset time.
    """
    ensure_wildchat_slice(n_wildchat)
    payload = json.loads(WILDCHAT_SLICE_PATH.read_text())
    convs = payload["conversations"]
    if len(convs) < n_wildchat:
        raise RuntimeError(f"wildchat slice has {len(convs)} convs < {n_wildchat} needed")
    contexts: list[dict] = []
    for conv in convs[:n_wildchat]:
        # short_prefix_msgs = [user, assistant] (one exchange); use as the
        # prior-chat context turns.
        msgs = conv["short_prefix_msgs"]
        contexts.append(
            {
                "context_id": f"wildchat_{conv['conv_id']}",
                "context_type": "wildchat",
                "messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
            }
        )
    logger.info("wildchat contexts: %d", len(contexts))
    return contexts


# ── Offset derivation (plan §3 derivation block) ─────────────────────────────


def derive_pair(tokenizer, context_messages: list[dict], query: str) -> dict:
    """Render context+query, derive (ctx_end_idx, query_end_idx), assert invariants.

    Returns the per-pair record fields (prompts + offsets + decoded sanity).
    Raises AssertionError on the prefix invariant (§12 A4) or the ordering
    invariant (0 <= ctx < query < seq_len) — fail loud CPU-side.
    """
    # context-only render (no user turn, no generation prompt): the context
    # block is a strict token PREFIX of the full prompt.
    ctx_only = tokenizer.apply_chat_template(
        context_messages, tokenize=False, add_generation_prompt=False
    )
    ctx_ids = tokenizer(ctx_only, add_special_tokens=False).input_ids
    context_end_idx = len(ctx_ids) - 1

    full_messages = [*context_messages, {"role": "user", "content": query}]
    # query-end = last prompt token before the assistant marker = last token of
    # the (context + user-turn) render WITHOUT the generation prompt.
    nogen = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    nogen_ids = tokenizer(nogen, add_special_tokens=False).input_ids
    query_end_idx = len(nogen_ids) - 1

    # full extraction prompt (WITH the assistant generation marker).
    full = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer(full, add_special_tokens=False).input_ids

    # companion context-only prompt WITH the generation marker (the
    # assistant-generation slot read for the same-position contrast, §5).
    ctx_only_gen = tokenizer.apply_chat_template(
        context_messages, tokenize=False, add_generation_prompt=True
    )

    # ── Invariants (fail loud CPU-side) ──────────────────────────────────────
    # Prefix invariant (§12 A4): the rendered context block is a strict token
    # prefix of the full prompt (ChatML renders the system/ICL block identically
    # with/without a trailing user turn).
    assert full_ids[: len(ctx_ids)] == ctx_ids, (
        "PREFIX VIOLATION: context block is not a strict token prefix of the full "
        f"prompt (context_id derivation). len(ctx_ids)={len(ctx_ids)}"
    )
    # The no-gen full render must also be a prefix of the gen-prompt full render.
    assert full_ids[: len(nogen_ids)] == nogen_ids, (
        "PREFIX VIOLATION: nogen render is not a prefix of the generation-prompt render"
    )
    seq_len = len(full_ids)
    assert 0 <= context_end_idx < query_end_idx < seq_len, (
        f"ORDERING VIOLATION: 0 <= {context_end_idx} < {query_end_idx} < {seq_len}"
    )

    return {
        "full_prompt": full,
        "context_only_prompt": ctx_only_gen,
        "ctx_end_idx": context_end_idx,
        "query_end_idx": query_end_idx,
        "content_tokens": seq_len,
        "decoded_ctx_end_tok": tokenizer.decode([full_ids[context_end_idx]]),
        "decoded_query_end_tok": tokenizer.decode([full_ids[query_end_idx]]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #654: build the (context, query) pair battery."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=QWEN_MODEL, help="tokenizer id (no model load)")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="2 contexts/tier x first 4 queries (tiny slice; same code path)",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    n_generic = 2 if args.smoke else N_GENERIC
    n_icl = 2 if args.smoke else N_ICL
    n_wildchat = 2 if args.smoke else N_WILDCHAT

    query_bank = build_query_bank()
    queries = query_bank[:4] if args.smoke else query_bank

    # ── Build context tiers ──────────────────────────────────────────────────
    persona_contexts = build_persona_contexts()
    if args.smoke:
        persona_contexts = persona_contexts[:2]
    generic_contexts, icl_contexts, ultrachat_prov = build_generic_and_icl_contexts(
        n_generic, n_icl
    )
    wildchat_contexts = build_wildchat_contexts(n_wildchat)

    all_contexts = persona_contexts + generic_contexts + icl_contexts + wildchat_contexts
    logger.info(
        "contexts: persona=%d generic=%d icl=%d wildchat=%d total=%d",
        len(persona_contexts),
        len(generic_contexts),
        len(icl_contexts),
        len(wildchat_contexts),
        len(all_contexts),
    )

    # ── Build pairs + derive offsets ─────────────────────────────────────────
    pairs: list[dict] = []
    for ctx in all_contexts:
        for q in queries:
            derived = derive_pair(tokenizer, ctx["messages"], q["text"])
            pairs.append(
                {
                    "pair_id": f"{ctx['context_id']}__{q['query_id']}",
                    "context_type": ctx["context_type"],
                    "context_id": ctx["context_id"],
                    "query_id": q["query_id"],
                    "topicality": q["topicality"],
                    "length": q["length"],
                    **derived,
                }
            )

    logger.info("derived %d (context, query) pairs", len(pairs))

    # ── Offset sanity smoke: print decoded context_end + query_end for first 8 ─
    logger.info("=== offset sanity (first 8 pairs): decoded context_end / query_end ===")
    for p in pairs[:8]:
        logger.info(
            "  %-44s ctx_end[%d]=%r  query_end[%d]=%r",
            p["pair_id"],
            p["ctx_end_idx"],
            p["decoded_ctx_end_tok"],
            p["query_end_idx"],
            p["decoded_query_end_tok"],
        )

    # ── Reproducibility metadata ─────────────────────────────────────────────
    import datetime
    import platform

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except subprocess.CalledProcessError:
        git_commit = "unknown"

    meta = {
        "issue": 654,
        "model": args.model,
        "seed": SEED,
        "smoke": args.smoke,
        "n_contexts": len(all_contexts),
        "n_queries": len(queries),
        "n_pairs": len(pairs),
        "context_counts": {
            "persona": len(persona_contexts),
            "generic": len(generic_contexts),
            "icl": len(icl_contexts),
            "wildchat": len(wildchat_contexts),
        },
        "excluded_personas": sorted(EXCLUDED_PERSONAS),
        "persona_names_selected": [c["context_id"] for c in persona_contexts],
        "query_bank": query_bank,
        "queries_used": [q["query_id"] for q in queries],
        "ultrachat_provenance": ultrachat_prov,
        "wildchat_slice_path": str(WILDCHAT_SLICE_PATH.relative_to(PROJECT_ROOT)),
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat() + "Z",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "pairs": pairs}, f, ensure_ascii=False, indent=2)
    logger.info(
        "Wrote %s: %d pairs (%d contexts x %d queries)",
        args.out,
        len(pairs),
        len(all_contexts),
        len(queries),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
