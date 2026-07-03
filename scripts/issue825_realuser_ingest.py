#!/usr/bin/env python
"""Issue #825 follow-up ``real-user-turn-null`` — ingest real 2-turn lmsys conversations.

THE ONE VARIABLE vs the parent / v7 rounds: u2 TEXT PROVENANCE. This round takes
complete REAL logged 2-turn conversations (u1, a1, u2 all human/serving-model
logged) from ``lmsys/lmsys-chat-1m`` at the parent's pinned revision — no
generation anywhere (v7's u2-generation phase is REPLACED by this CPU phase).

Pipeline (deterministic, stream order at the pinned revision — plan v11 §4.1):
  1. stream ``lmsys/lmsys-chat-1m`` train split at the pinned revision;
  2. row filters (parent-parity, reusing ``_is_english`` / ``_is_clean`` from
     ``issue825_gen_conversations``): English; non-redacted; non-flagged;
     STRICT u1(user)->a1(assistant)->u2(user) prefix (later turns discarded);
     >= MIN_TURN_CONTENT_TOKENS on u1, a1 AND u2; dedup on u1[:200];
  3. render filters: rendered length <= MAX_CONV_TOKENS under BOTH formats
     (max-over-formats, the parent rule); gen-time span validation per format
     via the v7 ``_degenerate_spans`` mirror of the extractor's asserts —
     degenerate rows are DROPPED and counted (class ``span_degenerate``),
     never placeholder-substituted (no row-parity constraint here);
  4. keep the FIRST ``--n`` survivors in stream order; conv_id = 0..n-1;
  5. HARD-FAIL below ``--n`` (writes ``ingest_failure.json`` with
     ``status: ingest_shortfall`` for the dispatch wrapper's FAILURE sentinel,
     then exits 1 — never run underpowered, parent convention).

Outputs under --out-dir:
  conversations_real2turn.jsonl   rows {conv_id, u1, a1, u2,
                                  lmsys_conversation_id, a1_model}
  conversations_real2turn_meta.json  per-class drop counts, a1-model mix,
                                  u1[:200]-overlap vs the parent kept-2000,
                                  filter constants, dataset revision, u2
                                  distinct-3-gram + token-length stats.

CPU-only; no vLLM / no GPU anywhere in this script.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from explore_persona_space.experiments.issue_825.common import (
    MAX_CONV_TOKENS,
    MIN_TURN_CONTENT_TOKENS,
    MODEL_INSTRUCT,
    N_TRACK_M,
)
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # HF token for the gated stream + shared-VM thread caps (#847)

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue825_render_formats as rf  # noqa: E402
from issue825_gen_conversations import _is_clean, _is_english  # noqa: E402
from issue825_onpolicy_u2_gen import (  # noqa: E402
    _degenerate_spans,
    _distinct_3gram_rate,
    _length_stats,
    _ntok,
    _repetition_rate,
)

FOLLOWUP_LABEL = "real-user-turn-null"
LMSYS_REPO = "lmsys/lmsys-chat-1m"  # SINGLE source, hard-asserted (parent Track-S convention)
LMSYS_REVISION = "200748d9d3cddcc9d782887541057aca0b18c5da"  # parent's recorded pin
DEDUP_KEY_CHARS = 200  # parent dedup key: u1[:200]
STREAM_LOG_EVERY = 2000
DEFAULT_MAX_STREAM_ROWS = 500_000  # safety cap; eligible pool ~125k => never binding

DROP_CLASSES = (
    "not_english",
    "redacted",
    "moderation_flagged",
    "toxic_flag",
    "no_strict_u1a1u2_prefix",
    "short_turn",
    "dup_u1",
    "too_long",
    "span_degenerate",
    "render_error",
)


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def strict_two_turn_prefix(row: dict) -> tuple[str, str, str] | None:
    """Return (u1, a1, u2) iff the conversation opens user->assistant->user.

    Later turns are DISCARDED (the kept row is the strict 3-turn prefix).
    Returns None when the prefix is absent or any of the three roles is wrong.
    Stripped-empty contents are returned as-is — the >=8-content-token filter
    downstream drops them (class ``short_turn``).
    """
    conv = row.get("conversation") or []
    if len(conv) < 3:
        return None
    roles = [t.get("role") for t in conv[:3]]
    if roles != ["user", "assistant", "user"]:
        return None
    u1, a1, u2 = ((conv[i].get("content") or "").strip() for i in range(3))
    return u1, a1, u2


def classify_unclean(row: dict) -> str:
    """Attribute a ``_is_clean``-failing row to its drop sub-class.

    The GATE is ``_is_clean`` (imported, parent-parity); this only names which
    flag fired for the per-class drop counts (plan §4.1 item 2).
    """
    if row.get("redacted") is True:
        return "redacted"
    mod = row.get("openai_moderation")
    if isinstance(mod, list | tuple) and mod:
        first = mod[0]
        if isinstance(first, dict) and first.get("flagged") is True:
            return "moderation_flagged"
    return "toxic_flag"


def filter_and_collect(
    rows,
    tokenizer,
    n_target: int,
    *,
    parent_u1_keys: frozenset[str] = frozenset(),
    max_stream_rows: int | None = DEFAULT_MAX_STREAM_ROWS,
) -> dict:
    """Stream rows through the plan §4.1 filter pipeline; keep the FIRST n_target.

    Pure function of the row iterator (network-free in tests). Returns a dict:
    kept rows, per-class drops, n_streamed, stream_exhausted / stream_cap_hit,
    a1-model mix, and the u1[:200]-overlap count vs ``parent_u1_keys``.
    """
    kept: list[dict] = []
    drops: Counter[str] = Counter({c: 0 for c in DROP_CLASSES})
    seen: set[str] = set()
    a1_model_mix: Counter[str] = Counter()
    n_streamed = 0
    parent_overlap = 0
    stream_cap_hit = False
    stream_exhausted = True
    for row in rows:
        if len(kept) >= n_target:
            stream_exhausted = False
            break
        n_streamed += 1
        if max_stream_rows is not None and n_streamed > max_stream_rows:
            stream_cap_hit = True
            stream_exhausted = False
            break
        if n_streamed % STREAM_LOG_EVERY == 0:
            print(f"[ingest] streamed {n_streamed} rows, kept {len(kept)}/{n_target}", flush=True)
        if not _is_english(row):
            drops["not_english"] += 1
            continue
        if not _is_clean(row):
            drops[classify_unclean(row)] += 1
            continue
        parsed = strict_two_turn_prefix(row)
        if parsed is None:
            drops["no_strict_u1a1u2_prefix"] += 1
            continue
        u1, a1, u2 = parsed
        if min(_ntok(tokenizer, t) for t in (u1, a1, u2)) < MIN_TURN_CONTENT_TOKENS:
            drops["short_turn"] += 1
            continue
        key = u1[:DEDUP_KEY_CHARS]
        if key in seen:
            drops["dup_u1"] += 1
            continue
        probe = {"conv_id": len(kept), "u1": u1, "a1": a1, "u2": u2}
        try:
            r_chat = rf.render_chat(probe, tokenizer)
            r_nat = rf.render_naturalistic(probe, tokenizer)
        except AssertionError as exc:
            # A counted, logged drop — real text can defeat the offsets
            # segmentation on exotic unicode; one weird row must not kill a
            # 30-min stream, and the count keeps the failure loud.
            drops["render_error"] += 1
            print(f"[ingest] render_error (dropped): {exc} (u1[:80]={u1[:80]!r})")
            continue
        if max(len(r_chat.input_ids), len(r_nat.input_ids)) > MAX_CONV_TOKENS:
            drops["too_long"] += 1
            continue
        if _degenerate_spans(r_chat) or _degenerate_spans(r_nat):
            # Plan §2: real rows failing span validation are DROPPED (counted),
            # never placeholder-substituted (no row-parity constraint here).
            drops["span_degenerate"] += 1
            print(f"[ingest] span_degenerate (dropped): u2[:80]={u2[:80]!r}")
            continue
        seen.add(key)
        if key in parent_u1_keys:
            parent_overlap += 1
        a1_model = str(row.get("model") or "unknown")
        a1_model_mix[a1_model] += 1
        kept.append(
            {
                "conv_id": len(kept),
                "u1": u1,
                "a1": a1,
                "u2": u2,
                "lmsys_conversation_id": str(row.get("conversation_id") or ""),
                "a1_model": a1_model,
            }
        )
    return {
        "kept": kept,
        "drops": dict(drops),
        "n_streamed": n_streamed,
        "stream_exhausted": stream_exhausted,
        "stream_cap_hit": stream_cap_hit,
        "a1_model_mix": dict(a1_model_mix),
        "u1_overlap_with_parent": parent_overlap,
    }


def load_parent_u1_keys(parent_conversations: Path | None) -> frozenset[str]:
    """Dedup keys (u1[:200]) of the parent kept-2000 for the overlap diagnostic."""
    if parent_conversations is None or not parent_conversations.exists():
        return frozenset()
    keys = set()
    with open(parent_conversations, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                keys.add(str(json.loads(line)["u1"])[:DEDUP_KEY_CHARS])
    return frozenset(keys)


def build_meta(result: dict, *, n_target: int, revision: str, tokenizer, args_note: dict) -> dict:
    """Assemble conversations_real2turn_meta.json (plan §4.1 item 5)."""
    kept = result["kept"]
    u2s = [r["u2"] for r in kept]
    return {
        "followup_label": FOLLOWUP_LABEL,
        "source": LMSYS_REPO,
        "dataset_revision": revision,
        "n_target": n_target,
        "n_kept": len(kept),
        "n_streamed": result["n_streamed"],
        "stream_exhausted": result["stream_exhausted"],
        "stream_cap_hit": result["stream_cap_hit"],
        "drops": result["drops"],
        "a1_model_mix": result["a1_model_mix"],
        "u1_overlap_with_parent_kept2000": result["u1_overlap_with_parent"],
        "filter_constants": {
            "min_turn_content_tokens": MIN_TURN_CONTENT_TOKENS,
            "max_conv_tokens_rendered_max_over_formats": MAX_CONV_TOKENS,
            "dedup_key": f"u1[:{DEDUP_KEY_CHARS}]",
            "strict_prefix": "u1(user)->a1(assistant)->u2(user); later turns discarded",
            "span_validation": "extractor span asserts via _degenerate_spans, both formats",
        },
        "u2_length": _length_stats(tokenizer, u2s),
        "u1_length": _length_stats(tokenizer, [r["u1"] for r in kept]),
        "a1_length": _length_stats(tokenizer, [r["a1"] for r in kept]),
        "distinct_3gram_rate_u2": _distinct_3gram_rate(u2s),
        "repetition_rate_u2": _repetition_rate(u2s),
        "license_note": (
            "Derived from lmsys/lmsys-chat-1m (LMSYS-Chat-1M license, research "
            "use); rows are English, non-redacted, non-moderation-flagged only."
        ),
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **args_note,
    }


def write_ingest_failure(out_dir: Path, status: str, detail: dict) -> Path:
    """Persist the typed failure artifact the wrapper's sentinel path reads.

    Route-on-artifact (not exit code): the dispatch wrapper reads this file to
    distinguish ``ingest_shortfall`` from an unexpected ``ingest_error``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ingest_failure.json"
    path.write_text(
        json.dumps(
            {
                "status": status,
                "followup_label": FOLLOWUP_LABEL,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **detail,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[ingest] FAILURE artifact written: {path} (status={status})", file=sys.stderr)
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=Path("data/issue_825/realuser"))
    ap.add_argument("--n", type=int, default=N_TRACK_M, help="kept-row target (hard floor)")
    ap.add_argument("--revision", default=LMSYS_REVISION)
    ap.add_argument(
        "--parent-conversations",
        type=Path,
        default=None,
        help="parent kept-2000 conversations.jsonl (u1-overlap diagnostic)",
    )
    ap.add_argument("--max-stream-rows", type=int, default=DEFAULT_MAX_STREAM_ROWS)
    args = ap.parse_args()

    signal.alarm(2700)  # 45-min per-stage hard cap (plan §10)
    from huggingface_hub import auth_check

    # Single-source hard assert (plan §4.1): lmsys only; gate access verified
    # at startup so a revoked gate fails HERE, not mid-stream.
    auth_check(LMSYS_REPO, repo_type="dataset")
    print(f"[ingest] AUTH OK: {LMSYS_REPO} @ {args.revision}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_INSTRUCT)
    parent_keys = load_parent_u1_keys(args.parent_conversations)
    print(f"[ingest] parent u1 dedup keys loaded: {len(parent_keys)}")

    from datasets import load_dataset

    ds = load_dataset(LMSYS_REPO, split="train", streaming=True, revision=args.revision)
    result = filter_and_collect(
        iter(ds),
        tokenizer,
        args.n,
        parent_u1_keys=parent_keys,
        max_stream_rows=args.max_stream_rows,
    )
    kept = result["kept"]
    meta = build_meta(
        result,
        n_target=args.n,
        revision=args.revision,
        tokenizer=tokenizer,
        args_note={
            "max_stream_rows": args.max_stream_rows,
            "parent_conversations": (
                str(args.parent_conversations) if args.parent_conversations else None
            ),
        },
    )
    out_path = args.out_dir / "conversations_real2turn.jsonl"
    meta_path = args.out_dir / "conversations_real2turn_meta.json"
    _write_jsonl(out_path, kept)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[ingest] kept {len(kept)}/{args.n} after {result['n_streamed']} streamed rows "
        f"(drops={result['drops']}) -> {out_path}"
    )
    if len(kept) < args.n:
        write_ingest_failure(
            args.out_dir,
            "ingest_shortfall",
            {
                "n_kept": len(kept),
                "n_target": args.n,
                "n_streamed": result["n_streamed"],
                "stream_cap_hit": result["stream_cap_hit"],
                "drops": result["drops"],
            },
        )
        print(
            f"[ingest] HARD-FAIL: {len(kept)} kept < target {args.n} — never run "
            "underpowered (plan §4.1 item 4)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
