"""Issue #501 Phase 0 — load #377 drift + length-matched-neutral corpora and
build the 12 MT/MN history slices.

Plan v2 §4.4 + §7 (Phase-0 corpus-load + max-prefix-length sanity check).

Steps:
  1. Download (or cache-hit) the two #377 JSONL corpora at the PINNED
     revision ``54a80fdf4c2e863e0b9885010a708321071b70ef`` from HF Hub
     ``superkaiba1/explore-persona-space-data``.
  2. Pre-filter each per-domain pool: drop any conversation whose first
     ``MAX_K = 14`` turns contain a ``[BATCH_ERROR]`` sentinel (a known #377
     generation artifact). The deterministic selection in step 3/4 then
     samples against the CLEAN pool. MAX_K=14 covers both the deepest k
     used (MT*_k14) and any length-matched MN slice (MN slices cumsum
     from the same pool with a length cap so any conversation poisoned
     within the first 14 turns could also poison the matched MN slice).
     Per-domain clean counts MUST stay ≥ ``N_CONVERSATIONS_PER_SLOT`` (=5);
     a shortfall raises fail-fast.
  3. For each MT01..MT08 row: pick 5 conversation indices deterministically
     (``deterministic_conversation_indices(domain, k, n_avail)``), slice each
     to depth k (clamped to corpus length), record the slice + its tokenized
     length under the project tokenizer (Qwen-2.5-7B-Instruct).
  4. For each MN01..MN04 row: compute the matched drift slot's MEAN total
     token count; iterate the matched-domain neutral conversations and
     accept the LONGEST prefix whose total token count is ≤ the drift mean
     (port of #377's ``_length_matched_slice_n``); pick 5 such conversations
     deterministically.
  5. Run the max-prefix-token-count sanity check: if any (prefix + 50 +
     ``MAX_NEW_TOKENS=2048``) > 28000, raise the bump-to-65536 escalation
     warning per plan §10 deviations-allowed.
  6. Defense-in-depth: assert the marker " ※" (id 83399) does NOT appear in
     any loaded history's content (per plan Assumption 16).
  7. Persist ``eval_results/issue_501/phase0/mt_prefixes.json`` with one
     row per (cid, conv_index) carrying the sliced history + token count +
     source-corpus hash for downstream reproducibility. Per-cid payload
     carries ``n_dropped_batch_error`` for Phase 5 reporting.

CLI:
    uv run python scripts/i501_phase0_load_corpora.py
    uv run python scripts/i501_phase0_load_corpora.py --smoke
        # Loads only MT05 + MT06 + MN03 (smoke canary cells) for a dry run.
    uv run python scripts/i501_phase0_load_corpora.py --bust-cache
        # Forces re-download from HF Hub at the pinned revision.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

from explore_persona_space.experiments.i406_conditions import MARKER_TEXT
from explore_persona_space.experiments.i501_mt_contexts import (
    DRIFT_HUB_PATH,
    HF_DATA_REPO,
    HF_DATA_REVISION,
    INCONTEXT_HUB_PATH,
    MT_CONTEXTS,
    N_CONVERSATIONS_PER_SLOT,
    PER_DOMAIN_DRIFT_COUNT,
    assert_no_marker_in_history,
    deterministic_conversation_indices,
)

logger = logging.getLogger("i501.phase0_load")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_501" / "phase0"
DRIFT_LOCAL = PROJECT_ROOT / "data" / "issue501_pinned" / "drift_conversations.jsonl"
NEUTRAL_LOCAL = PROJECT_ROOT / "data" / "issue501_pinned" / "incontext_conversations.jsonl"

# Plan §10 deviations-allowed: if any actually-loaded prefix + R + Q overhead
# > 28000 tokens, bump max_model_len from 32768 → 65536.
PHASE0_MAX_PROMPT_R_TOKEN_BUDGET = 28000
MAX_NEW_TOKENS = 2048
Q_OVERHEAD_TOKENS = 50  # rough overhead for the chat-template assistant-open + user-q overhead

# Round-down even slice (matches #377's role-parity rule).
_MIN_PARITY_SLICE = 2

# Pre-filter window: drop any conversation whose first MAX_K turns contain a
# ``[BATCH_ERROR]`` sentinel (a known #377 generation artifact at the pinned
# revision). MAX_K is the deepest k we use AND covers MN length-matched
# slicing (cumsum-capped, may extend up to k turns into the same pool). The
# deterministic-hash conversation-index selection runs AFTER this filter, so
# different k values still pick different conversations against a per-domain
# CLEAN pool. See `_filter_batch_error_conversations`.
MAX_K = 14


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_pinned(hub_path: str, local_path: Path, *, bust_cache: bool) -> None:
    """Download the corpus JSONL at the pinned revision into ``local_path``.

    Uses ``huggingface_hub.hf_hub_download(..., revision=HF_DATA_REVISION)``
    directly because :func:`explore_persona_space.orchestrate.hub.download_dataset`
    does not expose a ``revision`` kwarg.
    """
    if local_path.exists() and not bust_cache:
        logger.info("Phase 0: cache-hit %s (skip HF download)", local_path)
        return
    if local_path.exists():
        local_path.unlink()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    logger.info(
        "Phase 0: downloading %s/%s @ %s ...",
        HF_DATA_REPO,
        hub_path,
        HF_DATA_REVISION,
    )
    downloaded = hf_hub_download(
        repo_id=HF_DATA_REPO,
        revision=HF_DATA_REVISION,
        filename=hub_path,
        repo_type="dataset",
        local_dir=str(local_path.parent),
        token=token,
    )
    downloaded = Path(downloaded)
    if downloaded != local_path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded.replace(local_path)
    logger.info("Phase 0: wrote %s (sha256=%s)", local_path, _file_sha256(local_path)[:12])


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _whitespace_token_count(text: str) -> int:
    """Match #377's ``_whitespace_token_count`` exactly: whitespace-split.

    Used only for length-matching MN to MT; the actual model-tokenization
    length is computed separately via the project tokenizer.
    """
    return len(text.split())


def _slice_history_at_k(turns: list[dict], k: int) -> list[dict]:
    """Slice the conversation to depth k (port of #377's
    :func:`build_history_for_k`).

    The slice_n target is 4 for k=5, 10 for k=10, 20 for k=20 in #377's
    convention; for this experiment we only support k∈{10, 14}. k=10 →
    slice 10 turns; k=14 → slice 14 turns. If the available turn count is
    smaller, we clamp to the largest even slice_n ≤ available (preserves
    role parity ending on assistant).

    Raises if the slice ends on a non-assistant role or carries a
    [BATCH_ERROR] sentinel (defense-in-depth; #377's eval rig does the same).
    """
    if k not in (10, 14):
        raise ValueError(f"i501 only supports k∈{{10, 14}}; got k={k}")
    target = k  # both 10 and 14 are even, so slice_n_target == k directly
    n_available = len(turns)
    slice_n = target if target <= n_available else n_available - (n_available % 2)
    slice_n = max(_MIN_PARITY_SLICE, slice_n)
    history = turns[:slice_n]
    if not history:
        raise RuntimeError(f"empty history after slicing to k={k} (n_available={n_available})")
    if history[-1].get("role") != "assistant":
        raise RuntimeError(
            f"k={k} slice ended on role={history[-1].get('role')!r}; expected 'assistant'"
        )
    for idx, turn in enumerate(history):
        if turn.get("content") == "[BATCH_ERROR]":
            raise RuntimeError(
                f"k={k} slice contains [BATCH_ERROR] sentinel at turn {idx}; "
                "drop the conversation or regenerate the corpus"
            )
    return history


def _length_matched_slice(turns: list[dict], target_total: int) -> list[dict]:
    """Port of #377's ``_length_matched_slice_n`` — pick the longest
    even-parity prefix whose whitespace-token cumsum is ≤ ``target_total``.

    Returns a sliced list (ending on assistant). Raises if no slice ≥
    :data:`_MIN_PARITY_SLICE` qualifies (e.g. the very first turn already
    exceeds ``target_total``).
    """
    cumsum = 0
    max_le_target_j = 0
    for idx, turn in enumerate(turns, start=1):
        cumsum += _whitespace_token_count(turn.get("content", ""))
        if cumsum <= target_total:
            max_le_target_j = idx
        else:
            break
    slice_n = max_le_target_j - (max_le_target_j % 2)
    if max_le_target_j == len(turns):
        slice_n = len(turns) - (len(turns) % 2)
    slice_n = max(_MIN_PARITY_SLICE, min(slice_n, len(turns) - (len(turns) % 2)))
    if slice_n < _MIN_PARITY_SLICE:
        raise RuntimeError(
            f"length-match could not find a slice ≥ {_MIN_PARITY_SLICE} turns "
            f"with cumsum ≤ {target_total} (first-turn already exceeds target)"
        )
    history = turns[:slice_n]
    if history[-1].get("role") != "assistant":
        raise RuntimeError(
            f"length-matched slice ended on role={history[-1].get('role')!r}; expected 'assistant'"
        )
    return history


def _conversations_by_domain(corpus: list[dict]) -> dict[str, list[dict]]:
    """Group corpus by the ``domain`` field, preserving JSONL order."""
    out: dict[str, list[dict]] = {}
    for conv in corpus:
        out.setdefault(conv["domain"], []).append(conv)
    return out


def _filter_batch_error_conversations(
    pool: list[dict],
    max_k: int,
    *,
    domain_label: str,
    arm_label: str,
) -> tuple[list[dict], list[dict]]:
    """Drop conversations carrying a ``[BATCH_ERROR]`` sentinel inside the
    first ``max_k`` turns.

    The #377 drift + neutral corpora at the pinned revision contain a small
    number of conversations whose turn ``content`` was overwritten by the
    string ``"[BATCH_ERROR]"`` during generation (a known artifact — see
    #377's clean-result + corpus README). The deterministic-hash selection
    in ``deterministic_conversation_indices`` is keyed on ``(domain, k)``,
    so different k values pick different indices; without this pre-filter
    one k might hit a clean conversation while another raises at
    ``_slice_history_at_k`` (e.g. round-2 launch crash 2026-06-06: MT01
    k=10 OK, MT02 k=14 raised on coding).

    Returns ``(clean, dropped)`` with JSONL order preserved within each.
    Emits one INFO line per dropped conversation (id + first poisoned turn
    index) and a per-(domain, arm) summary line.
    """
    clean: list[dict] = []
    dropped: list[dict] = []
    for idx, conv in enumerate(pool):
        turns = conv.get("turns", [])
        poisoned_at: int | None = None
        for t_idx, t in enumerate(turns[:max_k]):
            if t.get("content") == "[BATCH_ERROR]":
                poisoned_at = t_idx
                break
        if poisoned_at is None:
            clean.append(conv)
        else:
            dropped.append(conv)
            cid = conv.get("conversation_id") or f"{domain_label}_{idx}"
            logger.info(
                "Phase 0: drop conversation_id=%s (%s/%s) — "
                "[BATCH_ERROR] at turn %d (within first %d)",
                cid,
                arm_label,
                domain_label,
                poisoned_at,
                max_k,
            )
    logger.info(
        "Phase 0: %s/%s dropped %d/%d conversations carrying [BATCH_ERROR] inside first %d turns",
        arm_label,
        domain_label,
        len(dropped),
        len(pool),
        max_k,
    )
    return clean, dropped


def _tokenize_message_list(tokenizer, history: list[dict]) -> int:
    """Total tokens of the chat-templated history (no user q appended) under
    the project tokenizer. Used for the per-context max-prefix-length sanity
    check.
    """
    rendered = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=False)
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - per-MT-arm corpus build
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-mode: load only MT05 (therapy drift k=10) + MN03 (factual_qa neutral).",
    )
    ap.add_argument(
        "--bust-cache",
        action="store_true",
        help="Force re-download from HF Hub (ignore the local cached JSONL).",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download both corpora at the pinned revision.
    _download_pinned(DRIFT_HUB_PATH, DRIFT_LOCAL, bust_cache=args.bust_cache)
    _download_pinned(INCONTEXT_HUB_PATH, NEUTRAL_LOCAL, bust_cache=args.bust_cache)

    drift_corpus = _load_jsonl(DRIFT_LOCAL)
    neutral_corpus = _load_jsonl(NEUTRAL_LOCAL)
    logger.info(
        "Phase 0: loaded %d drift + %d neutral conversations",
        len(drift_corpus),
        len(neutral_corpus),
    )

    # Sanity: per-domain counts match the constants in MT_BY_CID.
    drift_by_domain = _conversations_by_domain(drift_corpus)
    neutral_by_domain = _conversations_by_domain(neutral_corpus)
    for domain, expected in PER_DOMAIN_DRIFT_COUNT.items():
        actual = len(drift_by_domain.get(domain, []))
        if actual < expected:
            raise RuntimeError(
                f"drift corpus domain={domain!r} has {actual} conversations, "
                f"expected ≥ {expected} at revision {HF_DATA_REVISION}"
            )

    # Pre-filter per-domain pools: drop any conversation carrying a
    # [BATCH_ERROR] sentinel inside the first MAX_K turns (#377 corpus
    # artifact). Selection downstream runs against the CLEAN pool, so
    # MT01 (k=10) and MT02 (k=14) on the same domain can no longer
    # diverge between "clean conv picked" and "poisoned conv raises".
    dropped_drift_by_domain: dict[str, int] = {}
    dropped_neutral_by_domain: dict[str, int] = {}
    for domain in list(drift_by_domain):
        clean, dropped = _filter_batch_error_conversations(
            drift_by_domain[domain], MAX_K, domain_label=domain, arm_label="drift"
        )
        drift_by_domain[domain] = clean
        dropped_drift_by_domain[domain] = len(dropped)
        if len(clean) < N_CONVERSATIONS_PER_SLOT:
            raise RuntimeError(
                f"drift corpus domain={domain!r} has only {len(clean)} clean conversations "
                f"after dropping {len(dropped)} carrying [BATCH_ERROR] in first {MAX_K} turns; "
                f"need ≥ {N_CONVERSATIONS_PER_SLOT}. Regenerate the corpus or pin a clean revision."
            )
    for domain in list(neutral_by_domain):
        clean, dropped = _filter_batch_error_conversations(
            neutral_by_domain[domain], MAX_K, domain_label=domain, arm_label="neutral"
        )
        neutral_by_domain[domain] = clean
        dropped_neutral_by_domain[domain] = len(dropped)
        if len(clean) < N_CONVERSATIONS_PER_SLOT:
            raise RuntimeError(
                f"neutral corpus domain={domain!r} has only {len(clean)} clean conversations "
                f"after dropping {len(dropped)} carrying [BATCH_ERROR] in first {MAX_K} turns; "
                f"need ≥ {N_CONVERSATIONS_PER_SLOT}. Regenerate the corpus or pin a clean revision."
            )

    # 2 + 3. Build per-MT/MN slices.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Restrict to smoke subset if asked. Plan §4.4 + Round-2 fix: MN03's
    # length-match target is the (MT05, MT06) pair-mean, so the smoke must
    # build BOTH halves of the pair for the pair-mean to be well-defined
    # without falling back to inline-recompute (which would still work
    # but is less informative for smoke verification).
    if args.smoke:
        targets = [c for c in MT_CONTEXTS if c.cid in ("MT05", "MT06", "MN03")]
    else:
        targets = list(MT_CONTEXTS)

    per_cid_payload: dict[str, dict] = {}

    # Build drift slices FIRST (MN needs the drift mean-token-count to length-match).
    drift_mean_total_by_domain_k: dict[tuple[str, int], float] = {}

    for ctx in [c for c in targets if c.arm == "drift"]:
        domain = ctx.domain
        k = ctx.k
        pool = drift_by_domain[domain]
        n_avail = len(pool)
        indices = deterministic_conversation_indices(
            domain, k, per_domain_count=n_avail, n_picks=N_CONVERSATIONS_PER_SLOT
        )
        rows = []
        ws_totals: list[int] = []
        for ci in indices:
            conv = pool[ci]
            history = _slice_history_at_k(conv["turns"], k)
            assert_no_marker_in_history(tuple(history), MARKER_TEXT, ctx.cid)
            ws_total = sum(_whitespace_token_count(t.get("content", "")) for t in history)
            ws_totals.append(ws_total)
            tok_total = _tokenize_message_list(tokenizer, history)
            rows.append(
                {
                    "conversation_index": ci,
                    "conversation_id": conv.get("conversation_id", f"{domain}_{ci}"),
                    "n_turns": len(history),
                    "whitespace_token_count": ws_total,
                    "chat_template_token_count": tok_total,
                    "history": history,
                }
            )
        drift_mean_total_by_domain_k[(domain, k)] = (
            sum(ws_totals) / len(ws_totals) if ws_totals else 0.0
        )
        per_cid_payload[ctx.cid] = {
            "cid": ctx.cid,
            "name": ctx.name,
            "domain": domain,
            "k": k,
            "arm": "drift",
            "is_strong_kind": ctx.is_strong_kind,
            "matched_drift_cids": [],
            "selected_indices": list(indices),
            "rows": rows,
            "drift_mean_whitespace_total": drift_mean_total_by_domain_k[(domain, k)],
            "n_dropped_batch_error": dropped_drift_by_domain.get(domain, 0),
            "n_clean_pool": len(pool),
        }
        logger.info(
            "Phase 0: built %s (domain=%s, k=%d, %d convs, mean_ws_total=%.1f)",
            ctx.cid,
            domain,
            k,
            len(rows),
            drift_mean_total_by_domain_k[(domain, k)],
        )

    # Now build neutral (MN) slices length-matched to the PAIR-MEAN of the
    # matched-drift slots per plan §4.4 table:
    #   MN01↔(MT01, MT02) — math neutral, length-matched to coding (k=10, k=14) mean
    #   MN02↔(MT03, MT04) — history neutral, length-matched to writing pair mean
    #   MN03↔(MT05, MT06) — factual_qa neutral, length-matched to therapy pair mean
    #   MN04↔(MT07, MT08) — code_review neutral, length-matched to philosophy pair mean
    #
    # Round-2 fix (Codex CONCERN): v1 length-matched against the k=10 row
    # only; plan-as-written says the PAIR MEAN. With both slots' 5-conv
    # whitespace means averaged, the MN total-token count tracks the
    # *average* drift prefix length across k∈{10,14} rather than only the
    # shallower depth — this is what plan §4.4 H3 expects when comparing
    # drift vs neutral within the 288 cross-format cells.
    def _drift_mean_total_for(domain: str, k: int) -> float:
        cached = drift_mean_total_by_domain_k.get((domain, k))
        if cached is not None:
            return float(cached)
        # Smoke mode may not have pre-built this slot — recompute inline.
        pool = drift_by_domain[domain]
        idxs = deterministic_conversation_indices(
            domain,
            k,
            per_domain_count=len(pool),
            n_picks=N_CONVERSATIONS_PER_SLOT,
        )
        ws_totals = []
        for ci in idxs:
            conv = pool[ci]
            history = _slice_history_at_k(conv["turns"], k)
            ws_totals.append(sum(_whitespace_token_count(t.get("content", "")) for t in history))
        mean = sum(ws_totals) / len(ws_totals)
        drift_mean_total_by_domain_k[(domain, k)] = mean
        return mean

    for ctx in [c for c in targets if c.arm == "neutral"]:
        matched_cids = ctx.matched_drift_cids
        if not matched_cids:
            raise RuntimeError(f"neutral {ctx.cid} missing matched_drift_cids")
        matched_mts = []
        for matched_cid in matched_cids:
            matched_mt = next((c for c in MT_CONTEXTS if c.cid == matched_cid), None)
            if matched_mt is None:
                raise RuntimeError(f"matched_drift_cid={matched_cid} not found for {ctx.cid}")
            matched_mts.append(matched_mt)
        # Pair-mean: average the per-slot 5-conv whitespace means across all
        # matched drift slots.
        per_slot_means = [_drift_mean_total_for(m.domain, m.k) for m in matched_mts]
        target_total = sum(per_slot_means) / len(per_slot_means)

        # Pick 5 neutral conversations deterministically. Seed-key is
        # (neutral_domain, deepest_matched_k) so the selection is stable
        # and reproducible.
        deepest_k = max(m.k for m in matched_mts)
        neutral_pool = neutral_by_domain.get(ctx.domain, [])
        if not neutral_pool:
            raise RuntimeError(
                f"neutral corpus domain={ctx.domain!r} is empty at revision {HF_DATA_REVISION}"
            )
        indices = deterministic_conversation_indices(
            ctx.domain,
            deepest_k,
            per_domain_count=len(neutral_pool),
            n_picks=N_CONVERSATIONS_PER_SLOT,
        )
        rows = []
        for ci in indices:
            conv = neutral_pool[ci]
            history = _length_matched_slice(conv["turns"], int(target_total))
            assert_no_marker_in_history(tuple(history), MARKER_TEXT, ctx.cid)
            ws_total = sum(_whitespace_token_count(t.get("content", "")) for t in history)
            tok_total = _tokenize_message_list(tokenizer, history)
            rows.append(
                {
                    "conversation_index": ci,
                    "conversation_id": conv.get("conversation_id", f"{ctx.domain}_{ci}"),
                    "n_turns": len(history),
                    "whitespace_token_count": ws_total,
                    "chat_template_token_count": tok_total,
                    "history": history,
                }
            )
        per_cid_payload[ctx.cid] = {
            "cid": ctx.cid,
            "name": ctx.name,
            "domain": ctx.domain,
            "k": ctx.k,
            "arm": "neutral",
            "is_strong_kind": 0,
            "matched_drift_cids": list(matched_cids),
            "matched_drift_slot_means_whitespace": per_slot_means,
            "selected_indices": list(indices),
            "rows": rows,
            "drift_pair_mean_whitespace_total": float(target_total),
            "n_dropped_batch_error": dropped_neutral_by_domain.get(ctx.domain, 0),
            "n_clean_pool": len(neutral_pool),
        }
        logger.info(
            "Phase 0: built %s (domain=%s, matched=%s, pair_mean_ws=%.1f, %d convs)",
            ctx.cid,
            ctx.domain,
            ",".join(matched_cids),
            target_total,
            len(rows),
        )

    # 4. Max-prefix-token-count sanity check (plan §10 deviations-allowed).
    max_prefix_tok = 0
    max_prefix_owner: str | None = None
    for cid, payload in per_cid_payload.items():
        for row in payload["rows"]:
            tok = row["chat_template_token_count"]
            if tok > max_prefix_tok:
                max_prefix_tok = tok
                max_prefix_owner = f"{cid}/conv{row['conversation_index']}"
    worst_total = max_prefix_tok + Q_OVERHEAD_TOKENS + MAX_NEW_TOKENS
    bump_recommended = worst_total > PHASE0_MAX_PROMPT_R_TOKEN_BUDGET
    logger.info(
        "Phase 0: max_prefix_tok=%d (owner=%s); worst (prefix+Q+R)=%d "
        "(budget=%d) bump_max_model_len_to_65536=%s",
        max_prefix_tok,
        max_prefix_owner,
        worst_total,
        PHASE0_MAX_PROMPT_R_TOKEN_BUDGET,
        bump_recommended,
    )

    # 6. Persist the canonical phase-0 artifact.
    out_payload = {
        "schema_version": "i501_phase0_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "base_model": BASE_MODEL,
        "marker_text": MARKER_TEXT,
        "hf_data_repo": HF_DATA_REPO,
        "hf_data_revision": HF_DATA_REVISION,
        "drift_corpus_path": str(DRIFT_LOCAL),
        "neutral_corpus_path": str(NEUTRAL_LOCAL),
        "drift_corpus_sha256": _file_sha256(DRIFT_LOCAL),
        "neutral_corpus_sha256": _file_sha256(NEUTRAL_LOCAL),
        "per_domain_drift_counts": {
            d: len(drift_by_domain.get(d, [])) for d in PER_DOMAIN_DRIFT_COUNT
        },
        "per_domain_neutral_counts": {d: len(v) for d, v in neutral_by_domain.items()},
        "per_domain_dropped_batch_error_drift": dict(dropped_drift_by_domain),
        "per_domain_dropped_batch_error_neutral": dict(dropped_neutral_by_domain),
        "max_k_filter_window": MAX_K,
        "max_prefix_chat_template_token_count": max_prefix_tok,
        "max_prefix_owner": max_prefix_owner,
        "worst_case_prefix_plus_Q_plus_R_tokens": worst_total,
        "max_model_len_recommendation": 65536 if bump_recommended else 32768,
        "bump_max_model_len_to_65536": bump_recommended,
        "smoke": bool(args.smoke),
        "per_cid": per_cid_payload,
    }
    out_path = OUT_DIR / "mt_prefixes.json"
    out_path.write_text(json.dumps(out_payload, indent=2))
    logger.info("Phase 0 wrote %s (%d cids)", out_path, len(per_cid_payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
