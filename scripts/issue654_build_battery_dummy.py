#!/usr/bin/env python3
"""Issue #654 amendment (plan v5 §2/§5): build the length-matched dummy-query battery.

Cheap-band same-issue follow-up round 1 (``length-matched-dummy-query-control``).
Adds ONE query arm to the #654 battery: a content-neutral dummy query that is
token-length-matched to each real query under the SAME context, so the
dummy-vs-real same-slot companion gap (plan §3) isolates query *content* from the
query *length / read-position* confound the parent's Finding 4 flagged.

The dummy design (plan §2 / §11, the ONE new hyperparameter A13):
  - Fixed content-neutral base sentence:
      "Please continue with whatever you think is most appropriate here."
  - Length-matched to each real query's ``query_end_idx`` under that context by
    appending the single-token filler word `` really`` (Qwen-2.5-7B id 2167)
    until the dummy's ``query_end_idx`` equals the real one (overshoot trims
    trailing filler tokens; if the base sentence ALONE still overshoots a very
    short real query, the base is truncated at a word boundary — the realized
    text + residual length delta is recorded per pair).
  - ``<|im_pad|>`` rejected: NOT a single Qwen token (6 ordinary subwords) -> it
    would be off-distribution noise (plan §11 rejected-alternatives).

Reuse: the real battery (``battery.json`` / ``battery_real.json``) supplies, per
(context, real query): the per-context message set, the real query's
``query_end_idx`` length target, and the ``context_only_prompt`` (REUSED verbatim
so the dummy arm reads the SAME cached context-only companion banks on HF). The
context tiers + ``derive_pair`` ordering/prefix asserts are imported UNCHANGED
from ``issue654_build_battery``.

No model load — tokenizer-only (offsets only). Runs CPU-side, on-pod, before the
GPU forward pass (the "first CPU step of the dispatcher" framing the parent uses).

``--smoke``: the first 4 contexts x first 2 real queries, prints 8 realized dummy
strings + per-pair length residuals (plan A13 manipulation check).

Usage::

    uv run python scripts/issue654_build_battery_dummy.py \
        --real-battery data/issue654/battery_real.json \
        --out data/issue654/battery_dummy.json
    uv run python scripts/issue654_build_battery_dummy.py --smoke \
        --real-battery data/issue654/battery_smoke.json \
        --out data/issue654/battery_dummy_smoke.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

# Reuse the parent battery's derivation + tier builders UNCHANGED (plan §4/§5):
# derive_pair runs the SAME prefix + ordering asserts on the dummy pairs, and the
# context tiers are rebuilt deterministically only when the real battery does not
# carry the per-context message set (it does — we read it from the real battery).
from issue654_build_battery import (  # noqa: E402
    DATA_DIR,
    QWEN_MODEL,
    SEED,
    derive_pair,
)

load_dotenv()

logger = logging.getLogger("issue654_battery_dummy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_REAL_BATTERY = DATA_DIR / "battery_real.json"
DEFAULT_OUT = DATA_DIR / "battery_dummy.json"

# ── The ONE new hyperparameter (plan §2 / §11 / A13) ─────────────────────────
# Content-neutral base user turn that asks for nothing topical.
DUMMY_BASE = "Please continue with whatever you think is most appropriate here."
# Single-token filler word (Qwen-2.5-7B id 2167); asserted at build time.
FILLER_WORD = " really"
FILLER_TOKEN_ID = 2167

# Kill / flag criterion (plan §6): if > 10% of dummy pairs carry a > 2-token
# length residual, length-matching is systematically failing — flag loud (the
# gap would then re-confound length and content, the very thing the control
# removes). We FLAG (log + record in meta), not raise: a handful of very short
# on-topic real queries carrying a +-1-2 token residual is disclosed, not hidden
# (plan §6 "reported per pair").
RESIDUAL_TOKEN_TOL = 2
RESIDUAL_FLAG_FRACTION = 0.10


def _build_dummy_text(
    tokenizer,
    context_messages: list[dict],
    target_query_end_idx: int,
) -> tuple[str, int, int]:
    """Build a length-matched dummy query for ONE (context, real-query) target.

    Renders ``context + dummy-base`` and appends `` really`` tokens (or trims) so
    the dummy's derived ``query_end_idx`` equals ``target_query_end_idx`` under
    that context. Returns ``(dummy_text, achieved_query_end_idx, residual_delta)``
    where ``residual_delta = achieved - target`` (0 = exact match).

    Strategy (plan §2 per-real-query padding rule):
      1. Start from ``DUMMY_BASE``; derive its ``query_end_idx``.
      2. If SHORT of target: append `` really`` tokens one at a time until the
         derived ``query_end_idx`` reaches the target (the per-append rederivation
         absorbs any ChatML-context tokenization drift, plan §2 risk-2 mitigation).
      3. If OVER target with the base alone: truncate the base sentence at a word
         boundary (drop trailing words) until at/under target, then top up with
         `` really`` to hit it exactly where possible.
      4. Record the residual if exact match is impossible (real query shorter than
         the irreducible dummy core).
    """

    def _q_end(text: str) -> int:
        # query_end = last prompt token of (context + user-turn) WITHOUT the
        # assistant generation prompt — same derivation as derive_pair.
        nogen = tokenizer.apply_chat_template(
            [*context_messages, {"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=False,
        )
        return len(tokenizer(nogen, add_special_tokens=False).input_ids) - 1

    base = DUMMY_BASE
    base_qend = _q_end(base)

    if base_qend <= target_query_end_idx:
        # Pad UP with filler tokens until we reach (or just pass) the target.
        text = base
        qend = base_qend
        # Cap the append loop generously: target - base_qend filler tokens at most,
        # plus a small slack for any tokenization drift.
        max_appends = max(0, target_query_end_idx - base_qend) + 8
        for _ in range(max_appends):
            if qend >= target_query_end_idx:
                break
            text = text + FILLER_WORD
            qend = _q_end(text)
        # If we overshot by one filler token, drop the last `` really`` (a single
        # token by construction) — it cannot push us below the base.
        while qend > target_query_end_idx and text.endswith(FILLER_WORD):
            text = text[: -len(FILLER_WORD)]
            qend = _q_end(text)
        return text, qend, qend - target_query_end_idx

    # base alone OVERSHOOTS the target (very short real query): truncate the base
    # at a word boundary until at/under target, then top up with filler.
    words = base.rstrip(".").split()
    text = base
    qend = base_qend
    while qend > target_query_end_idx and len(words) > 1:
        words = words[:-1]
        text = " ".join(words) + "."
        qend = _q_end(text)
    # Top up with filler if truncation undershot.
    max_appends = max(0, target_query_end_idx - qend) + 8
    for _ in range(max_appends):
        if qend >= target_query_end_idx:
            break
        text = text + FILLER_WORD
        qend = _q_end(text)
    while qend > target_query_end_idx and text.endswith(FILLER_WORD):
        text = text[: -len(FILLER_WORD)]
        qend = _q_end(text)
    return text, qend, qend - target_query_end_idx


def _collect_disjointness_strings(real_payload: dict) -> set[str]:
    """Eval-query strings the dummy must avoid (the real 10-query bank).

    The dummy base + every realized dummy string is asserted disjoint from (a) the
    real eval queries (collected here) and (b) every context's user-turn / system
    text (collected by the caller from the reconstructed per-context messages) so
    the dummy cannot accidentally echo a context or eval string (plan §2
    content-neutrality assert).
    """
    strings: set[str] = set()
    for q in real_payload["meta"].get("query_bank", []):
        strings.add(q["text"].strip())
    return strings


def _parse_chatml_messages(context_only_prompt: str) -> list[dict]:
    """Recover the context message list from a rendered context-only ChatML string.

    ``context_only_prompt`` is ``apply_chat_template(messages,
    add_generation_prompt=True)`` — i.e. the context turns followed by the
    ``<|im_start|>assistant\n`` generation marker. We split on the ChatML role
    markers to recover the original ``[{role, content}, ...]`` so ``derive_pair``
    can re-render context+dummy identically to the real arm.
    """
    # Strip the trailing assistant generation prompt.
    gen_marker = "<|im_start|>assistant\n"
    body = context_only_prompt
    if body.endswith(gen_marker):
        body = body[: -len(gen_marker)]
    messages: list[dict] = []
    # Each turn is `<|im_start|>{role}\n{content}<|im_end|>\n`.
    for chunk in body.split("<|im_start|>"):
        chunk = chunk.strip()
        if not chunk:
            continue
        # chunk = `{role}\n{content}<|im_end|>`
        if "\n" not in chunk:
            continue
        role, rest = chunk.split("\n", 1)
        role = role.strip()
        content = rest
        if content.endswith("<|im_end|>"):
            content = content[: -len("<|im_end|>")]
        content = content.rstrip("\n")
        if role in {"system", "user", "assistant"}:
            messages.append({"role": role, "content": content})
    if not messages:
        raise RuntimeError(
            "failed to parse any ChatML turns from context_only_prompt: "
            f"{context_only_prompt[:120]!r}"
        )
    return messages


def build_dummy_pairs(tokenizer, real_payload: dict) -> tuple[list[dict], dict]:
    """Build one length-matched dummy pair per (context, real-query) in the real battery.

    Returns (dummy_pairs, build_meta). Each dummy pair mirrors the real pair's
    context_id / context_type / topicality / length, with a dummy query_id
    (``q_dummy_for_<real-suffix>``) and the realized dummy text + target/achieved
    length + residual recorded. The join back to the real arm is by the ORIGINAL
    real query_id (``real_query_id``).
    """
    real_pairs = real_payload["pairs"]
    # Group by context_id; reconstruct each context's message list once.
    per_context_msgs: dict[str, list[dict]] = {}
    per_context_type: dict[str, str] = {}
    for p in real_pairs:
        cid = p["context_id"]
        if cid not in per_context_msgs:
            per_context_msgs[cid] = _parse_chatml_messages(p["context_only_prompt"])
            per_context_type[cid] = p["context_type"]

    # Disjointness target set: real eval queries + every reconstructed context
    # turn content (so a dummy cannot echo a context/eval string), plus the real
    # query texts from meta.
    disjoint_strings: set[str] = _collect_disjointness_strings(real_payload)
    for msgs in per_context_msgs.values():
        for m in msgs:
            disjoint_strings.add(m["content"].strip())

    dummy_pairs: list[dict] = []
    residual_overs = 0
    realized_dummy_by_query: dict[str, str] = {}
    for p in real_pairs:
        cid = p["context_id"]
        real_qid = p["query_id"]
        target_qend = p["query_end_idx"]
        context_messages = per_context_msgs[cid]

        dummy_text, achieved_qend, residual = _build_dummy_text(
            tokenizer, context_messages, target_qend
        )
        if abs(residual) > RESIDUAL_TOKEN_TOL:
            residual_overs += 1

        # derive_pair runs the SAME prefix + ordering asserts as the real arm.
        derived = derive_pair(tokenizer, context_messages, dummy_text)

        dummy_qid = f"q_dummy_for_{real_qid.removeprefix('q_')}"
        dummy_pairs.append(
            {
                "pair_id": f"{cid}__{dummy_qid}",
                "context_type": per_context_type[cid],
                "context_id": cid,
                "query_id": dummy_qid,
                "real_query_id": real_qid,  # join key back to the real arm
                "topicality": p["topicality"],
                "length": p["length"],
                "dummy_text": dummy_text,
                "target_query_end_idx": target_qend,
                "achieved_query_end_idx": achieved_qend,
                "length_residual_tokens": residual,
                **derived,
            }
        )
        realized_dummy_by_query.setdefault(real_qid, dummy_text)

    n_pairs = len(dummy_pairs)
    over_fraction = residual_overs / max(n_pairs, 1)
    residual_flag = over_fraction > RESIDUAL_FLAG_FRACTION

    # ── Content-neutrality / disjointness assert (build-time, fail-loud) ──────
    # The base + every realized dummy string must not equal any eval/context turn.
    realized = {p["dummy_text"].strip() for p in dummy_pairs}
    realized.add(DUMMY_BASE.strip())
    collisions = sorted(realized & disjoint_strings)
    assert not collisions, (
        "DUMMY CONTENT COLLISION: a realized dummy string equals an eval/ICL/"
        f"wildchat/generic string (plan §2 content-neutrality assert): {collisions[:3]}"
    )

    build_meta = {
        "n_dummy_pairs": n_pairs,
        "n_distinct_contexts": len(per_context_msgs),
        "dummy_base": DUMMY_BASE,
        "filler_word": FILLER_WORD,
        "filler_token_id": FILLER_TOKEN_ID,
        "residual_token_tol": RESIDUAL_TOKEN_TOL,
        "residual_flag_fraction": RESIDUAL_FLAG_FRACTION,
        "n_pairs_over_residual_tol": residual_overs,
        "fraction_over_residual_tol": over_fraction,
        "residual_match_flag": residual_flag,
    }
    if residual_flag:
        logger.warning(
            "LENGTH-MATCH FLAG (plan §6 kill/sanity): %d/%d dummy pairs (%.3f) carry a "
            "> %d-token length residual (> %.0f%%) — length match systematically failing; "
            "the gap may re-confound length and content. Disclosed in meta.residual_match_flag.",
            residual_overs,
            n_pairs,
            over_fraction,
            RESIDUAL_TOKEN_TOL,
            RESIDUAL_FLAG_FRACTION * 100,
        )
    else:
        logger.info(
            "length-match: %d/%d dummy pairs (%.3f) over the %d-token tol (<= %.0f%% flag) — OK",
            residual_overs,
            n_pairs,
            over_fraction,
            RESIDUAL_TOKEN_TOL,
            RESIDUAL_FLAG_FRACTION * 100,
        )
    return dummy_pairs, build_meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #654 amendment: build the length-matched dummy-query battery."
    )
    parser.add_argument("--real-battery", type=Path, default=DEFAULT_REAL_BATTERY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=QWEN_MODEL, help="tokenizer id (no model load)")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="first 4 contexts x first 2 real queries (tiny slice; same code path)",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Build-time assert: the filler word is a SINGLE Qwen token (id 2167), so the
    # per-token length matching is exact (plan §2 / A13). Fail loud on a tokenizer
    # drift between build host and the design assumption.
    filler_ids = tokenizer.encode(FILLER_WORD, add_special_tokens=False)
    assert filler_ids == [FILLER_TOKEN_ID], (
        f"filler word {FILLER_WORD!r} must encode to the single token id "
        f"[{FILLER_TOKEN_ID}] (plan §2/A13); got {filler_ids}"
    )

    if not args.real_battery.exists():
        raise RuntimeError(
            f"real battery not found at {args.real_battery} — build it first "
            f"(scripts/issue654_build_battery.py) or pass --real-battery"
        )
    real_payload = json.loads(args.real_battery.read_text())

    if args.smoke:
        # First 4 distinct contexts x first 2 distinct real queries — IDENTICAL
        # build code path, just fewer pairs (mirrors the extractor's smoke subset).
        contexts_seen: list[str] = []
        queries_seen: list[str] = []
        for p in real_payload["pairs"]:
            if p["context_id"] not in contexts_seen:
                contexts_seen.append(p["context_id"])
            if p["query_id"] not in queries_seen:
                queries_seen.append(p["query_id"])
        keep_ctx = set(contexts_seen[:4])
        keep_q = set(queries_seen[:2])
        real_payload = {
            "meta": real_payload["meta"],
            "pairs": [
                p
                for p in real_payload["pairs"]
                if p["context_id"] in keep_ctx and p["query_id"] in keep_q
            ],
        }
        logger.info(
            "smoke: %d contexts x %d real queries = %d real pairs",
            len(keep_ctx),
            len(keep_q),
            len(real_payload["pairs"]),
        )

    dummy_pairs, build_meta = build_dummy_pairs(tokenizer, real_payload)
    logger.info("built %d dummy pairs", len(dummy_pairs))

    # ── Manipulation check (plan A13): print 8 realized dummy strings + residuals ─
    logger.info("=== realized dummy strings + length residuals (first 8 pairs) ===")
    for p in dummy_pairs[:8]:
        logger.info(
            "  %-48s target_qend=%d achieved=%d residual=%+d  dummy=%r",
            p["pair_id"],
            p["target_query_end_idx"],
            p["achieved_query_end_idx"],
            p["length_residual_tokens"],
            p["dummy_text"],
        )

    # ── Reproducibility metadata ─────────────────────────────────────────────
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except subprocess.CalledProcessError:
        git_commit = "unknown"

    meta = {
        "issue": 654,
        "followup_label": "length-matched-dummy-query-control",
        "arm": "dummy",
        "model": args.model,
        "seed": SEED,
        "smoke": args.smoke,
        "real_battery_path": str(args.real_battery),
        "real_battery_git_commit": real_payload["meta"].get("git_commit"),
        "n_pairs": len(dummy_pairs),
        **build_meta,
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat() + "Z",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "pairs": dummy_pairs}, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %s: %d dummy pairs", args.out, len(dummy_pairs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
