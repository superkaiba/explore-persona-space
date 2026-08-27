#!/usr/bin/env python3
"""Issue #2552 exactrep follow-up — LMSYS-only corpus prep for the Der et al. reproduction.

Reconstructs the training corpus of arXiv 2606.28548 (Der, Kamath & Thompson 2026):
LMSYS-Chat-1M assistant turns, target ~1.58M (TARGET_TURNS). The paper states NO
filtering recipe (App. A only says activations are "cached offline"), so every filter
here is a NAMED reconstruction assumption, counted per-filter and reported in the
manifest so a large realized-count mismatch surfaces instead of being absorbed:

  A-LANG   keep language == "English" (full name — LMSYS stores full language names,
           never ISO codes; #1092).
  A-STRUCT keep conversations whose messages strictly alternate user/assistant
           starting with user; anything else is dropped (counted).
  A-REDACT redacted conversations are KEPT (assumption: the paper trained on the
           public release as-is, NAME_1-style placeholders included); counted.
  A-MODER  no moderation filtering (the paper states none; the public release's
           `openai_moderation` field is present but filtering on it is not claimed).
  A-BUDGET conversations are truncated at TURN granularity to a rendered token
           budget (--max-render-tokens, default 8192): the kept prefix ends at the
           last assistant message whose cumulative rendered length fits. Truncated
           conversations and turns dropped by the cap are counted.
  A-EMPTY  assistant messages with empty/whitespace-only content stay in the rendered
           context but yield NO capture row (counted).
  A-SYSTEM rendering uses the model's own chat template defaults (Qwen2.5's implicit
           default system turn included).

Rendering invariant (asserted per conversation, drop+count on failure): the Qwen2.5
template renders a conversation as the plain concatenation of per-message segments
"<|im_start|>{role}\\n{content}<|im_end|>\\n" behind the default system segment —
verified 2026-08-26 on the pinned tokenizer, together with
concat(tokenize(segment_i)) == tokenize(full_text) (segments start with the atomic
special token <|im_start|>, so no cross-segment BPE merge is possible).

Output (all under --out-dir, pod-side /workspace):
  corpus/conv_{shard:05d}.jsonl   kept conversations: conversation_id, model, language,
                                  redacted, msgs (kept prefix), asst_msg_idx (message
                                  indices to capture), n_render_tokens
  progress.json                   stream checkpoint (resume; fingerprint-gated)
  manifest.json                   realized counts vs TARGET_TURNS + per-filter reject
                                  counters + fingerprint + reproducibility metadata

Checkpoint/resume: per-shard atomic JSONL writes; progress.json updated after each
completed shard; resume ds.skip(rows_streamed) under an EXACT fingerprint match
(dataset revision + every filter constant) — the #1092 _stream_with_cache contract.
The kept corpus text is NOT uploaded to HF (LMSYS redistribution license); the
manifest fingerprint IS the regeneration recipe.

Probe mode (--probe): bounded tiny-real streaming probe — kept cap AND total-rows cap,
per-filter reject counters printed, kept > 0 asserted (the #1092 data-ingestion probe).

Content hygiene: this script never prints conversation text — counters/ids only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # HF_TOKEN needed: lmsys/lmsys-chat-1m is gated; thread caps on the shared VM

from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2552_exactrep_prep")

LMSYS_REPO = "lmsys/lmsys-chat-1m"
# Pinned dataset revision (probed 2026-08-26; part of the resume fingerprint).
LMSYS_REVISION = "200748d9d3cddcc9d782887541057aca0b18c5da"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TARGET_TURNS = 1_580_000  # arXiv 2606.28548: ~1.58M assistant turns (reported vs realized)
DEFAULT_MAX_RENDER_TOKENS = 8192  # A-BUDGET (paper-underspecified; CLI-overridable)
DEFAULT_CONVS_PER_SHARD = 4096
IM_END_TAIL = "<|im_end|>\n"

_COUNTER_KEYS = (
    "rows_streamed",
    "kept_convs",
    "kept_turns",
    "reject_language",
    "reject_structure",
    "reject_template_shape",
    "reject_over_budget_all",
    "reject_zero_turns",
    "truncated_convs",
    "turns_dropped_by_budget",
    "empty_assistant_turns",
    "redacted_kept",
)


def _write_json_atomic(path: Path, obj: dict) -> None:
    """Atomic JSON write: tempfile in the destination dir + os.replace (EXDEV-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def render_segments(msgs: list[dict], tok) -> list[str]:
    """Per-message Qwen2.5 segments behind the template's default system segment.

    The default system segment is derived from the TOKENIZER itself (render a 1-turn
    probe and strip the known per-message segments) so a template-version drift fails
    the join assert instead of silently shifting every span (template shapes are
    version-specific; re-derived per run, never hardcoded across model forks).
    """
    segs = [f"<|im_start|>{m['role']}\n{m['content']}{IM_END_TAIL}" for m in msgs]
    probe_msg = [{"role": "user", "content": "x"}]
    probe_full = tok.apply_chat_template(probe_msg, tokenize=False)
    probe_seg = f"<|im_start|>user\nx{IM_END_TAIL}"
    assert probe_full.endswith(probe_seg), "chat-template shape drift: per-message segment"
    default_prefix = probe_full[: -len(probe_seg)]
    return ([default_prefix] if default_prefix else []) + segs


def structure_check(conv: list[dict]) -> bool:
    """True iff messages strictly alternate user/assistant starting with user (A-STRUCT)."""
    if not conv:
        return False
    for i, m in enumerate(conv):
        role = m.get("role")
        if not isinstance(m.get("content"), str):
            return False
        if i % 2 == 0 and role != "user":
            return False
        if i % 2 == 1 and role != "assistant":
            return False
    return len(conv) >= 2


def truncate_to_budget(
    seg_token_counts: list[int], n_prefix_segments: int, n_msgs: int, max_tokens: int
) -> tuple[int, int]:
    """A-BUDGET turn-granular truncation over per-SEGMENT token counts.

    seg_token_counts covers [prefix segments] + [one segment per message]. Returns
    (n_msgs_kept, n_render_tokens): the longest message prefix ENDING AT AN ASSISTANT
    message (odd msg index) whose cumulative rendered length fits max_tokens.
    (0, 0) when even the first user+assistant pair does not fit.
    """
    assert len(seg_token_counts) == n_prefix_segments + n_msgs, (
        len(seg_token_counts),
        n_prefix_segments,
        n_msgs,
    )
    cum = 0
    for c in seg_token_counts[:n_prefix_segments]:
        cum += c
    best_msgs, best_tokens = 0, 0
    for i in range(n_msgs):
        cum += seg_token_counts[n_prefix_segments + i]
        if cum > max_tokens:
            break
        if i % 2 == 1:  # assistant message — a legal truncation point
            best_msgs, best_tokens = i + 1, cum
    return best_msgs, best_tokens


def process_conversation(row: dict, tok, max_tokens: int, counters: dict) -> dict | None:
    """Filter + truncate one LMSYS row → a corpus record, or None (counters updated)."""
    if row.get("language") != "English":
        counters["reject_language"] += 1
        return None
    conv = row["conversation"]
    if not structure_check(conv):
        counters["reject_structure"] += 1
        return None
    segs = render_segments(conv, tok)
    n_prefix = len(segs) - len(conv)
    full = tok.apply_chat_template(conv, tokenize=False)
    if "".join(segs) != full:
        counters["reject_template_shape"] += 1
        return None
    # Batched segment tokenization (concat == full-text tokenization: atomic segment starts).
    enc = tok(segs, add_special_tokens=False)["input_ids"]
    seg_counts = [len(ids) for ids in enc]
    n_msgs_kept, n_render_tokens = truncate_to_budget(seg_counts, n_prefix, len(conv), max_tokens)
    if n_msgs_kept == 0:
        counters["reject_over_budget_all"] += 1
        return None
    if n_msgs_kept < len(conv):
        counters["truncated_convs"] += 1
        counters["turns_dropped_by_budget"] += sum(
            1 for i in range(n_msgs_kept, len(conv)) if i % 2 == 1
        )
    asst_idx = []
    for i in range(1, n_msgs_kept, 2):
        if conv[i]["content"].strip():
            asst_idx.append(i)
        else:
            counters["empty_assistant_turns"] += 1  # stays in context, yields no row
    if not asst_idx:
        counters["reject_zero_turns"] += 1
        return None
    if row.get("redacted"):
        counters["redacted_kept"] += 1
    counters["kept_convs"] += 1
    counters["kept_turns"] += len(asst_idx)
    return {
        "conversation_id": row["conversation_id"],
        "model": row.get("model"),
        "language": row["language"],
        "redacted": bool(row.get("redacted")),
        "msgs": [{"role": m["role"], "content": m["content"]} for m in conv[:n_msgs_kept]],
        "asst_msg_idx": asst_idx,
        "n_render_tokens": n_render_tokens,
    }


def stream_fingerprint(max_tokens: int, convs_per_shard: int) -> dict:
    """Exact-match resume fingerprint: dataset identity + every filter constant.

    Keyed on GENERATING PARAMETERS only (strings/ints — machine-stable; never
    recomputed-float hashes)."""
    return {
        "repo": LMSYS_REPO,
        "revision": LMSYS_REVISION,
        "model_id": MODEL_ID,
        "language": "English",
        "structure": "alternating-user-first",
        "redacted": "kept",
        "moderation": "none",
        "max_render_tokens": max_tokens,
        "convs_per_shard": convs_per_shard,
        "span_convention": "assistant-content-token-mean-v1",
    }


def _load_progress(out_dir: Path, fp: dict) -> dict:
    """Load the resume checkpoint; refuse (fail loud) on a fingerprint mismatch."""
    p = out_dir / "progress.json"
    if not p.exists():
        return {"fingerprint": fp, "next_shard": 0, **{k: 0 for k in _COUNTER_KEYS}}
    prog = json.loads(p.read_text())
    if prog.get("fingerprint") != fp:
        raise RuntimeError(
            f"progress.json fingerprint mismatch under {out_dir} — the existing stream was "
            "built under different filter constants / dataset revision; use a fresh --out-dir "
            f"(existing: {json.dumps(prog.get('fingerprint'))})"
        )
    return prog


def _flush_shard(corpus_dir: Path, shard_idx: int, records: list[dict]) -> Path:
    path = corpus_dir / f"conv_{shard_idx:05d}.jsonl"
    fd, tmp = tempfile.mkstemp(dir=corpus_dir, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return path


def run_prep(args) -> int:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    t0 = time.time()
    out_dir = Path(args.out_dir)
    corpus_dir = out_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    fp = stream_fingerprint(args.max_render_tokens, args.convs_per_shard)
    prog = _load_progress(out_dir, fp)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    ds = load_dataset(LMSYS_REPO, split="train", streaming=True, revision=LMSYS_REVISION)
    skip = int(prog["rows_streamed"])
    if skip:
        logger.info("[prep] resume: skipping %d already-streamed rows", skip)
        ds = ds.skip(skip)
    counters = {k: int(prog[k]) for k in _COUNTER_KEYS}
    shard_idx = int(prog["next_shard"])
    buf: list[dict] = []
    total_cap = args.probe_max_rows if args.probe else args.max_rows
    keep_cap = args.probe_keep if args.probe else 0

    it = iter(ds)
    row = None
    try:
        for row in it:
            counters["rows_streamed"] += 1
            rec = process_conversation(row, tok, args.max_render_tokens, counters)
            if rec is not None:
                buf.append(rec)
            if len(buf) >= args.convs_per_shard:
                _flush_shard(corpus_dir, shard_idx, buf)
                shard_idx += 1
                buf = []
                _write_json_atomic(
                    out_dir / "progress.json",
                    {"fingerprint": fp, "next_shard": shard_idx, **counters},
                )
                print(
                    f"[prep] shard {shard_idx} done rows_streamed={counters['rows_streamed']} "
                    f"kept_convs={counters['kept_convs']} kept_turns={counters['kept_turns']} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
            if total_cap and counters["rows_streamed"] >= total_cap:
                break
            if keep_cap and counters["kept_convs"] >= keep_cap:
                break
    finally:
        close = getattr(it, "close", None)
        if close is not None:
            close()  # release the streaming pipeline BEFORE interpreter shutdown (rc=134 trap)
        del row, it, ds  # the break-exited loop variable still pins the pipeline (#1947)
        import gc

        gc.collect()

    if buf:
        _flush_shard(corpus_dir, shard_idx, buf)
        shard_idx += 1
    _write_json_atomic(
        out_dir / "progress.json", {"fingerprint": fp, "next_shard": shard_idx, **counters}
    )

    if counters["kept_convs"] == 0:
        rejects = {k: v for k, v in counters.items() if k.startswith("reject_")}
        raise RuntimeError(
            f"prep kept ZERO conversations of {counters['rows_streamed']} streamed — "
            f"per-filter rejects: {json.dumps(rejects)}"
        )

    realized = counters["kept_turns"]
    manifest = {
        "fingerprint": fp,
        "n_shards": shard_idx,
        "counters": counters,
        "target_turns": TARGET_TURNS,
        "realized_turns": realized,
        "realized_over_target": round(realized / TARGET_TURNS, 4),
        "probe": bool(args.probe),
        "assumptions": [
            "A-LANG english-only (full-name match)",
            "A-STRUCT alternating user-first",
            "A-REDACT redacted kept",
            "A-MODER no moderation filter",
            f"A-BUDGET turn-granular truncation at {args.max_render_tokens} rendered tokens",
            "A-EMPTY empty assistant turns kept in context, no capture row",
            "A-SYSTEM template default system turn included",
        ],
        "metadata": as_metadata_dict(git_provenance(), phase="exactrep-prep"),
    }
    _write_json_atomic(out_dir / "manifest.json", manifest)
    print(
        f"[prep] done: kept_convs={counters['kept_convs']} kept_turns={realized} "
        f"(target {TARGET_TURNS}; ratio {manifest['realized_over_target']}) "
        f"rejects: lang={counters['reject_language']} struct={counters['reject_structure']} "
        f"template={counters['reject_template_shape']} "
        f"over_budget={counters['reject_over_budget_all']} "
        f"zero_turns={counters['reject_zero_turns']} elapsed={time.time() - t0:.0f}s",
        flush=True,
    )
    if not args.probe and not (0.5 <= manifest["realized_over_target"] <= 2.0):
        logger.warning(
            "[prep] realized turn count %.2fx the paper's ~1.58M target — surface this as a "
            "reconstruction-assumption mismatch before training",
            manifest["realized_over_target"],
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #2552 exactrep LMSYS-only corpus prep.")
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/eps-2552-exactrep"))
    ap.add_argument("--max-render-tokens", type=int, default=DEFAULT_MAX_RENDER_TOKENS)
    ap.add_argument("--convs-per-shard", type=int, default=DEFAULT_CONVS_PER_SHARD)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = full corpus pass")
    ap.add_argument("--probe", action="store_true", help="bounded tiny-real streaming probe")
    ap.add_argument("--probe-keep", type=int, default=50, help="probe: stop after N kept convs")
    ap.add_argument("--probe-max-rows", type=int, default=2000, help="probe: total-rows cap")
    ap.add_argument("--import-check", action="store_true", help="argparse-attr completeness")
    args = ap.parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        return 0
    return run_prep(args)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
