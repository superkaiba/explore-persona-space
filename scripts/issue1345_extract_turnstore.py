#!/usr/bin/env python
"""Issue #1345 Phases 2a/2b — teacher-forced turn-store extraction, 3 regimes.

Reuses the #825 extraction machinery WHOLESALE (issue825_extract_turnstore:
load_model, run_extraction, write_shards, causal_check, to_single_turn) with
EXACTLY ONE recipe delta vs the parent (plan §4 Phase 2a): the slot set is
extended with the PREFIX slot —
  R1 (chat):          prefix = last token of the pre-query template region
  R2 (naturalistic):  prefix = last fully-contained token of the `User: ` header
  R3 (stories, NEW):  per Q->A turn: prefix = last token before the question
                      utterance; context = last token of the attribution marker
Same renders, span averaging, bf16 CPU shards, all 28 layers.

R1/R2 consume the pinned parent track-S corpus (`track_s.jsonl` @ 7159e5804d);
R3 consumes Phase-1's kept stories, flattened to one row per Q->A turn with
conv_id = story id (story-level CV grouping downstream).

CLI:
  uv run python scripts/issue1345_extract_turnstore.py --regime r1 --model instruct
  uv run python scripts/issue1345_extract_turnstore.py --regime r3 --model instruct \
      [--smoke] [--tiny-model-dir <dir>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue825_extract_turnstore as ex  # noqa: E402
import issue1345_common as c  # noqa: E402


def _load_track_s_conversations(dl_dir: Path) -> list[dict]:
    """Pinned parent track-S rows -> single-turn conversations (parent recipe)."""
    path = c.stage_pinned_file(c.PARENT_TRACK_S_JSONL, dl_dir)
    rows = c.read_jsonl(path)
    assert rows, f"no rows in {path}"
    return [ex.to_single_turn(r) for r in rows]


def _render_r1r2(convs: list[dict], tokenizer, regime: str) -> list:
    """Prefix-slot renders for R1/R2 (the ONE extraction delta, plan §4 2a)."""
    render = c.render_chat_prefix if regime == "r1" else c.render_naturalistic_prefix
    return [render(conv, tokenizer) for conv in convs]


def _render_r3(stories: list[dict], tokenizer) -> tuple[list, dict]:
    """Flatten kept stories into per-turn Rendered rows (conv_id = story id)."""
    rendered, stats = [], {"stories": 0, "turns_in": 0, "turns_rendered": 0, "turns_dropped": 0}
    for s in stories:
        stats["stories"] += 1
        for turn in s["parsed_turns"]:
            stats["turns_in"] += 1
            r = c.render_story_turn(s["story"], turn, s["story_id"], tokenizer)
            if r is None:
                stats["turns_dropped"] += 1
                continue
            stats["turns_rendered"] += 1
            rendered.append(r)
    return rendered, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regime", choices=c.REGIMES, required=True)
    ap.add_argument("--model", choices=("instruct", "pretrained"), required=True)
    ap.add_argument("--out-dir", type=Path, default=c.TURNSTORE_DIR)
    ap.add_argument("--dl-dir", type=Path, default=c.PARENT_DL_DIR)
    ap.add_argument("--stories-dir", type=Path, default=c.STORIES_DIR)
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--shard-size", type=int, default=ex.SHARD_SIZE)
    ap.add_argument("--smoke", action="store_true", help="first 8 convs/stories; causal check ON")
    ap.add_argument(
        "--tiny-model-dir",
        default=None,
        help="SMOKE ONLY: tiny random-init Qwen2 (28 layers, small hidden) with the "
        "real tokenizer — CPU plumbing/shape validation; production never passes this",
    )
    args = ap.parse_args()

    model, tokenizer, model_id = ex.load_model(args.model, tiny_model_dir=args.tiny_model_dir)

    if args.regime in ("r1", "r2"):
        convs = _load_track_s_conversations(args.dl_dir)
        if args.smoke:
            convs = convs[:8]
            print(f"[smoke] limiting to {len(convs)} conversations", flush=True)
        rendered = _render_r1r2(convs, tokenizer, args.regime)
        render_stats = {"conversations": len(rendered)}
    else:
        kept_path = args.stories_dir / f"kept_stories_{args.model}.jsonl"
        stories = c.read_jsonl(kept_path)
        if args.smoke:
            stories = stories[:8]
            print(f"[smoke] limiting to {len(stories)} stories", flush=True)
        rendered, render_stats = _render_r3(stories, tokenizer)
        assert rendered, "no story turns rendered — parser/render drift"

    # Slot-order invariant the fit registry depends on: prefix strictly before
    # the context slot in EVERY row (extractor sorts slots by position, so
    # slot_index 0 = prefix, 1 = context across all three regimes).
    ctx_name = "a1" if args.regime in ("r1", "r2") else "context"
    for r in rendered:
        assert r.slot_idx["prefix"] < r.slot_idx[ctx_name], (r.conv_id, r.slot_idx)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    do_causal = args.smoke  # parent default: causal check on smoke
    if do_causal:
        ex.causal_check(model, rendered[: min(3, len(rendered))])

    bs = 8 if args.batch_size == "auto" else int(args.batch_size)
    if args.smoke:
        bs = min(bs, 2)
    peak_layers = sorted(li for li in ex.FROZEN_LAYERS if li < ex.EXPECTED_LAYERS)  # parent default
    stem = c.stem_for(args.model, args.regime)
    print(
        f"[run] regime={args.regime} model={args.model} ({model_id}) stem={stem} "
        f"n={len(rendered)} batch_size={bs}",
        flush=True,
    )
    sidecar_base = {
        "issue": 1345,
        "regime": args.regime,
        "model": args.model,
        "model_id": model_id,
        "format": c.REGIME_FORMAT[args.regime],
        "track": c.TRACK,
        "slot_names": ["prefix", ctx_name],
        "peak_layers": peak_layers,
        "expected_layers": ex.EXPECTED_LAYERS,
        "expected_hidden": ex.EXPECTED_HIDDEN,
        "render_stats": render_stats,
        "git_commit": c.git_commit(),
        "pinned_parent_revision": c.PIN_REV,
        "smoke": bool(args.smoke),
    }
    shard_size = int(args.shard_size)
    paths: list[Path] = []
    n_done = 0
    for block_idx, block_start in enumerate(range(0, len(rendered), shard_size)):
        block = rendered[block_start : block_start + shard_size]
        records = ex.run_extraction(model, block, peak_layers, pad_id, bs)
        assert len(records) == len(block), (block_idx, len(records), len(block))
        paths += ex.write_shards(
            records,
            args.out_dir,
            stem,
            sidecar_base,
            shard_offset=block_idx,
            shard_size=shard_size,
        )
        n_done += len(records)
        del records, block
    print(f"[done] {n_done} rows -> {len(paths)} shard(s) in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
