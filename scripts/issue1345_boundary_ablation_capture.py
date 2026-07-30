#!/usr/bin/env python
"""Issue #1345 story-boundary-ablation — teacher-forced capture (arms V2/V3/V4).

Reuses the #825/#1345 extraction machinery WHOLESALE (issue825_extract_turnstore:
load_model, run_extraction, write_shards, causal_check, partition_rendered) with
ONE delta vs the paired round: the slot set is the ARM-SYMMETRIC four-position
sweep, identical in name and order for every arm, so the arms are directly
comparable per read position:

  prefix      last token fully contained before the FINAL question utterance
  ctx_qend    last token fully contained before the question's closing quote
  context     last token fully contained before the arm's ANSWER BOUNDARY —
              the attribution-marker end for V1/V3/V4, the START of the
              blank-line run for V2 (the last token of the narrative sentence
              preceding the unmarked answer paragraph)
  ctx_preans  last token fully contained before the answer's first char

Ordering chain, enforced per row: prefix < ctx_qend <= context <= ctx_preans <
answer start. This is the 4-slot arm-symmetric analogue of
`issue1345_common.render_story_turn(extra_slots=True)` — the paired round's
5-slot store carries `ctx_preattr` + the pooled `ctx_attrmean`, which are
UNDEFINED for the boundary-absent arm; storing them would make the arms
shape-asymmetric and break the single-variable comparison.

Teacher-forced: the render is the story text truncated at the answer-span end —
ONE forward pass over [wrapper + question (+ prior exchanges) + boundary +
verbatim answer]; no generation. Row conv_id is the ORIGINAL conversation id, so
every arm is data-paired with the V1 anchor AND with the chat / no-template
comparator stores.

Trust boundary (fail-loud, never a skip): each kept row is re-gated with the
arm's OWN mechanical gate and its stored span re-verified as the verbatim answer
under the shared normalized matcher.

CLI:
  uv run python scripts/issue1345_boundary_ablation_capture.py --arm v2
  uv run python scripts/issue1345_boundary_ablation_capture.py --arm v2 --smoke \
      --tiny-model-dir <dir>
  uv run python scripts/issue1345_boundary_ablation_capture.py --import-check
"""

from __future__ import annotations

import argparse
import json
import os
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
import issue1345_boundary_ablation_gen as bg  # noqa: E402
import issue1345_common as c  # noqa: E402
import issue1345_gen_stories as g  # noqa: E402 — HF boundary helpers

from explore_persona_space.experiments.issue_825.common import Rendered  # noqa: E402

# ---------------------------------------------------------------------------
# Store layout
# ---------------------------------------------------------------------------
BND_SLOT_ORDER = ("prefix", "ctx_qend", "context", "ctx_preans")
# The read the headline comparison uses (V1's `context` semantics).
HEADLINE_SLOT = "context"
BND_ARMS = bg.ALL_ARMS
TRACK = c.TRACK


def format_key(arm: str) -> str:
    """Store format key — a DISTINCT stem per arm so no two arms can collide."""
    return f"bnd_{bg.ARM_SLUG[arm]}"


def stem_for(arm: str, model_key: str = bg.MODEL_KEY) -> str:
    return f"{model_key}_{format_key(arm)}_{TRACK}"


def hf_tensor_prefix(smoke: bool) -> str:
    """HF data-repo prefix for this round's stores."""
    return f"{c.HF_SMOKE_PREFIX if smoke else c.HF_ISSUE_PREFIX}/analysis_tensors"


# ---------------------------------------------------------------------------
# Arm-symmetric 4-slot render
# ---------------------------------------------------------------------------
def render_boundary_turn(story_text: str, turn: dict, story_id: str, tokenizer) -> Rendered | None:
    """Render ONE boundary-arm Q->A turn as a track-S-shaped Rendered row.

    Slots per BND_SLOT_ORDER, every one built with the ONE canonical idiom
    ``c._last_fully_contained`` (fully contained BEFORE the char boundary —
    never ``span[0] - 1``, which can be a token that BPE-merged the boundary
    WITH the answer's first word, leaking answer content into the read
    position). ``input_ids`` truncate at the answer-span end (causal attention
    makes activations at kept positions identical to the full-text forward).

    Returns None when any span/slot is degenerate (BPE zero-width merge or an
    ordering-chain violation — gotchas.md; the caller counts the drops).
    """
    enc = tokenizer(story_text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    a_start, a_end = turn["a_start"], turn["a_end"]
    a_tokens = [t for t, (a, b) in enumerate(offs) if a >= a_start and b <= a_end and b > a]
    if not a_tokens or a_tokens[-1] + 1 - a_tokens[0] != len(a_tokens):
        return None
    span = (a_tokens[0], a_tokens[-1] + 1)
    pfx = c._last_fully_contained(offs, turn["q_start"])
    qend = c._last_fully_contained(offs, turn["q_end"])
    ctx = c._last_fully_contained(offs, turn["boundary_end"])
    preans = c._last_fully_contained(offs, a_start)
    if None in (pfx, qend, ctx, preans):
        return None
    # Registered per-row ordering chain (ties allowed except prefix < ctx_qend).
    # Monotone by construction — a violation is render drift worth dropping +
    # counting, never silently reordered.
    if not (pfx < qend <= ctx <= preans < span[0] and 1 <= span[0] < span[1]):
        return None
    slot_idx = {"prefix": pfx, "ctx_qend": qend, "context": ctx, "ctx_preans": preans}
    assert tuple(slot_idx) == BND_SLOT_ORDER, tuple(slot_idx)
    trunc = span[1]
    return Rendered(
        input_ids=list(ids[:trunc]),
        slot_idx=slot_idx,
        spans={"answer": span},
        format="stories",
        conv_id=str(story_id),
        meta={
            "n_tokens": trunc,
            "confidence": turn["confidence"],
            "n_attribs": int(turn.get("n_attribs", 0)),
            "slot_char_spans": {n: [int(offs[i][0]), int(offs[i][1])] for n, i in slot_idx.items()},
            "a_char_span": [int(a_start), int(a_end)],
            # BPE-seam disclosure (#825/#1092 class): chars of the answer that
            # fell OUTSIDE the fully-contained token span because the boundary
            # merged with the answer's first word. 0 on a clean row.
            "answer_span_leading_gap": int(offs[a_tokens[0]][0] - a_start),
            "answer_span_trailing_gap": int(a_end - offs[a_tokens[-1]][1]),
        },
    )


def render_arm(arm: str, stories: list[dict], tokenizer) -> tuple[list[Rendered], dict]:
    """Re-gate + re-verify + render every kept story of one arm (fail-loud).

    Two trust-boundary re-checks per row, both fail-loud AssertionErrors rather
    than skips: (1) the arm's OWN mechanical gate must still return 'ok' and the
    SAME spans the gen phase stored (a mismatch is gate/regex/name-seam drift);
    (2) the stored span must be the verbatim answer under the shared normalized
    matcher (`c.norm_text`).
    """
    gate = bg.gate_for(arm)
    rendered: list[Rendered] = []
    stats = {"stories": 0, "turns_rendered": 0, "turns_dropped": 0}
    for s in stories:
        stats["stories"] += 1
        assert len(s["parsed_turns"]) == 1, (
            f"{arm} story {s['conv_id']}: expected exactly 1 parsed turn, "
            f"got {len(s['parsed_turns'])} (gen keep-filter drift)"
        )
        turn = s["parsed_turns"][0]
        re_turn, reason = gate(s["story"], s["answer"])
        assert reason == "ok" and re_turn is not None, (
            f"{arm} story {s['conv_id']}: the arm gate now returns {reason!r} at the "
            "extraction trust boundary — gate / regex / character-name drift"
        )
        for key in ("q_start", "q_end", "boundary_end", "a_start", "a_end"):
            assert re_turn[key] == turn[key], (
                f"{arm} story {s['conv_id']}: stored {key}={turn[key]} but the re-run "
                f"gate computes {re_turn[key]} — gate drift"
            )
        assert c.norm_text(s["story"][turn["a_start"] : turn["a_end"]]) == c.norm_text(
            s["answer"]
        ), (
            f"{arm} story {s['conv_id']}: stored span is not the verbatim answer under "
            "the shared normalized matcher (gen keep-filter drift)"
        )
        r = render_boundary_turn(s["story"], turn, s["conv_id"], tokenizer)
        if r is None:
            stats["turns_dropped"] += 1
            continue
        stats["turns_rendered"] += 1
        rendered.append(r)
    return rendered, stats


# ---------------------------------------------------------------------------
# Pre-GPU diagnostics (computed + persisted BEFORE any forward)
# ---------------------------------------------------------------------------
def slot_diagnostics(rendered: list[Rendered]) -> dict:
    """Per-slot positions, ANSWER-OVERLAP (hard 0), and coincidence rates.

    ANSWER-OVERLAP is hard-asserted 0 for every slot: a read position whose
    char span intersects the answer would be reading the target, not the
    context. Coincidence rates (vs the headline `context` slot) are the
    DETECTABLE degeneracy the boundary-absent arm is expected to show at
    `ctx_preans` — reported, never silently collapsed.
    """
    n = len(rendered)
    positions: dict[str, list[int]] = {s: [] for s in BND_SLOT_ORDER}
    overlap: dict[str, int] = {s: 0 for s in BND_SLOT_ORDER}
    coincide: dict[str, int] = {s: 0 for s in BND_SLOT_ORDER if s != HEADLINE_SLOT}
    lead_gap = 0
    for r in rendered:
        a0, a1 = r.meta["a_char_span"]
        for s in BND_SLOT_ORDER:
            positions[s].append(int(r.slot_idx[s]))
            cs, ce = r.meta["slot_char_spans"][s]
            if ce > a0 and cs < a1:
                overlap[s] += 1
            if s != HEADLINE_SLOT and r.slot_idx[s] == r.slot_idx[HEADLINE_SLOT]:
                coincide[s] += 1
        if r.meta["answer_span_leading_gap"] > 0:
            lead_gap += 1
    for s, k in overlap.items():
        assert k == 0, f"slot {s}: {k}/{n} rows read INSIDE the answer span — render bug"
    return {
        "n_rows": n,
        "slot_order": list(BND_SLOT_ORDER),
        "answer_overlap_counts": overlap,
        "coincidence_with_context_rates": {s: (k / n if n else 0.0) for s, k in coincide.items()},
        "median_position": {
            s: float(sorted(v)[len(v) // 2]) if v else float("nan") for s, v in positions.items()
        },
        "answer_span_leading_gap_rate": (lead_gap / n if n else 0.0),
    }


# ---------------------------------------------------------------------------
# HF persist
# ---------------------------------------------------------------------------
def persist_store(out_dir: Path, arm: str, smoke: bool, extra: dict) -> None:
    """Upload this arm's shards + sidecars + manifest to the HF data repo.

    Runs on the dispatcher's normal exit path, before the phase's done line —
    the store is a plan-referenced downstream input for the fits phase (#521).
    """
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing — cannot persist store"
    stem = stem_for(arm)
    files = sorted(p.name for p in out_dir.glob(f"{stem}*") if p.is_file())
    assert files, f"no {stem}* files to upload in {out_dir}"
    manifest = {
        "metadata": c.metadata(0, len(files), "scripts/issue1345_boundary_ablation_capture.py"),
        "round": bg.ROUND_VARIANT,
        "arm": arm,
        "arm_isolates": bg.ARM_README[arm],
        "model": bg.MODEL_KEY,
        "stem": stem,
        "slot_order": list(BND_SLOT_ORDER),
        "headline_slot": HEADLINE_SLOT,
        "headline_layer": c.HEADLINE_LAYER,
        "files": files,
        **extra,
    }
    man_path = out_dir / f"store_manifest_{stem}.json"
    c.write_json(man_path, manifest)
    prefix = hf_tensor_prefix(smoke)
    g._hf_upload_folder(
        out_dir,
        prefix,
        [f"{stem}*", man_path.name],
        f"issue-1345 story-boundary-ablation: {arm} turnstore ({stem})",
    )
    print(f"[capture] persisted {arm} store -> {prefix}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _import_check() -> None:
    """Resolve every deferred import on the REAL code path, then exit 0."""
    import inspect

    import torch  # noqa: F401

    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        assert_hub_dir_filecounts,
        retry_transient,
    )

    assert inspect.getsource(render_boundary_turn)
    print("[import-check] OK: torch + hub symbols resolved", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=tuple(bg.ARM_SLUG[a] for a in bg.GEN_ARMS) + bg.GEN_ARMS)
    ap.add_argument("--model", choices=("instruct",), default=bg.MODEL_KEY)
    ap.add_argument("--out-dir", type=Path, default=c.TURNSTORE_DIR)
    ap.add_argument("--stories-dir", type=Path, default=c.STORIES_DIR)
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--shard-size", type=int, default=ex.SHARD_SIZE)
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="informational: the physical GPU is pinned by CUDA_VISIBLE_DEVICES in the "
        "LAUNCHER env (gotchas.md CVD family) — this value is recorded in the sidecar "
        "and asserted consistent with the visible device count",
    )
    ap.add_argument("--skip-upload", action="store_true", help="local-only (smoke plumbing)")
    ap.add_argument("--smoke", action="store_true", help="first 8 stories; causal check ON")
    ap.add_argument(
        "--tiny-model-dir",
        default=None,
        help="SMOKE ONLY: tiny random-init Qwen2 (28 layers, small hidden) with the real "
        "tokenizer — CPU plumbing/shape validation; production never passes this",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import on the real code path and exit 0",
    )
    args = ap.parse_args()

    if args.import_check:
        _import_check()
        return

    bg.assert_round_env()
    assert args.arm, "--arm is required"
    arm = bg.SLUG_ARM.get(args.arm, args.arm)
    assert arm in bg.GEN_ARMS, f"{arm} is not a generated arm (V1 is reused, never captured)"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Yield-report guard (the #1345 character-name seam): the gen phase records
    # the realized character name; a capture launched without the gen phase's
    # env fails HERE, at entry, never silently mid-parse.
    yp = bg.yield_path(args.stories_dir, arm)
    assert yp.exists(), f"yield report missing: {yp} — run the gen phase for {arm} first"
    y = json.loads(yp.read_text())
    assert y.get("story_character_name") == c.STORY_CHARACTER_NAME, (
        f"character-name mismatch: gen recorded {y.get('story_character_name')!r}, this "
        f"capture runs with {c.STORY_CHARACTER_NAME!r}"
    )
    assert y.get("arm") == arm, (y.get("arm"), arm)

    model, tokenizer, model_id = ex.load_model(args.model, tiny_model_dir=args.tiny_model_dir)

    kept = bg.kept_path(args.stories_dir, arm)
    assert kept.exists(), f"kept stories missing: {kept}"
    stories = c.read_jsonl(kept)
    assert stories, f"{kept} is empty"
    if args.smoke:
        stories = stories[:8]
        print(f"[smoke] limiting to {len(stories)} {arm} stories", flush=True)
    rendered, render_stats = render_arm(arm, stories, tokenizer)
    assert rendered, f"no {arm} turns rendered — parser/render drift"

    # Parent-parity degenerate-row filter (#825/#1345 crash-fix semantics): a
    # zero-width content span would kill the extractor's hard `1 <= s < e`
    # assert mid-GPU-run. Drop per render; the skip manifest (conv ids only —
    # never corpus text) persists next to the shards.
    n_pre_filter = len(rendered)
    rendered, drops = ex.partition_rendered(rendered)
    stem = stem_for(arm, args.model)
    c.write_json(
        args.out_dir / f"{stem}_skip_manifest.json",
        {
            "metadata": c.metadata(
                0, n_pre_filter, "scripts/issue1345_boundary_ablation_capture.py"
            ),
            "round": bg.ROUND_VARIANT,
            "arm": arm,
            "model": args.model,
            "n_rendered_pre_filter": n_pre_filter,
            "n_dropped_zero_width": len(drops),
            "dropped_conv_ids": [d["conv_id"] for d in drops],
            "dropped_turns": {d["conv_id"]: d["turns"] for d in drops},
        },
    )
    assert rendered, (
        f"all {n_pre_filter} rendered {arm} rows dropped as zero-width — a systematic "
        "render bug, not a handful of degenerate rows"
    )
    ex.assert_residual_span_integrity(rendered)

    # Slot-order invariant the fit registry depends on: the per-row positional
    # sort must realize EXACTLY BND_SLOT_ORDER (ties keep insertion order under
    # the extractor's stable sort), so slot storage index == BND_SLOT_ORDER
    # index in EVERY arm's store.
    for r in rendered:
        names = [n for n, _ in sorted(r.slot_idx.items(), key=lambda kv: kv[1])]
        assert names == list(BND_SLOT_ORDER), (r.conv_id, names, r.slot_idx)
    diag = slot_diagnostics(rendered)
    diag_path = args.out_dir / f"{stem}_slot_diagnostics.json"
    c.write_json(
        diag_path,
        {
            "metadata": c.metadata(
                0, len(rendered), "scripts/issue1345_boundary_ablation_capture.py"
            ),
            "round": bg.ROUND_VARIANT,
            "arm": arm,
            "bundle_fingerprint": y.get("bundle_fingerprint"),
            "smoke": bool(args.smoke),
            **diag,
        },
    )
    coinc = {k: round(v, 4) for k, v in diag["coincidence_with_context_rates"].items()}
    print(
        f"[capture][{arm}] answer-overlap all 0 (hard-asserted); coincidence-with-"
        f"context: {coinc}; leading-gap rate {diag['answer_span_leading_gap_rate']:.4f} "
        f"-> {diag_path}",
        flush=True,
    )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    causal_max_diff = None
    if args.smoke:
        # mode="cosine": the early-position `prefix` slot has no bf16 headroom
        # under a flat atol (#1345 parent note); the #779 two-bar cosine gate
        # keeps the wrong-position bug catcher (real bugs read cos ~0.4-0.6).
        causal_max_diff = ex.causal_check(model, rendered[: min(3, len(rendered))], mode="cosine")

    bs = 8 if args.batch_size == "auto" else int(args.batch_size)
    if args.smoke:
        bs = min(bs, 2)
    peak_layers = sorted(li for li in ex.FROZEN_LAYERS if li < ex.EXPECTED_LAYERS)
    n_visible = None
    try:
        import torch

        n_visible = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001 — CPU smoke has no CUDA; informational only
        n_visible = None
    if args.gpu_id is not None and n_visible:
        assert n_visible == 1, (
            f"--gpu-id {args.gpu_id} passed but {n_visible} devices are visible — the "
            "launcher must pin CUDA_VISIBLE_DEVICES=<gpu> per arm (gotchas.md CVD family)"
        )
    print(
        f"[run] arm={arm} model={args.model} ({model_id}) stem={stem} n={len(rendered)} "
        f"batch_size={bs} gpu_id={args.gpu_id} visible_devices={n_visible}",
        flush=True,
    )
    sidecar_base = {
        "issue": 1345,
        "round": bg.ROUND_VARIANT,
        "arm": arm,
        "arm_isolates": bg.ARM_README[arm],
        "regime": format_key(arm),
        "model": args.model,
        "model_id": model_id,
        "format": format_key(arm),
        "track": TRACK,
        "story_character_name": c.STORY_CHARACTER_NAME,
        "slot_names": list(BND_SLOT_ORDER),
        "headline_slot": HEADLINE_SLOT,
        "peak_layers": peak_layers,
        "expected_layers": ex.EXPECTED_LAYERS,
        "expected_hidden": ex.EXPECTED_HIDDEN,
        "render_stats": render_stats,
        "slot_diagnostics": diag,
        "bundle_fingerprint": y.get("bundle_fingerprint"),
        "git_commit": c.git_commit(),
        "gpu_id": args.gpu_id,
        "causal_check_max_abs_diff": causal_max_diff,
        "causal_check_mode": "cosine" if args.smoke else None,
        "smoke": bool(args.smoke),
        "n_rendered_pre_filter": n_pre_filter,
        "n_dropped_zero_width": len(drops),
        "dropped_conv_ids": [d["conv_id"] for d in drops],
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
        print(f"[capture] arm={arm} rows {n_done}/{len(rendered)} shards={len(paths)}", flush=True)

    if args.skip_upload:
        print(f"[capture] --skip-upload: {n_done} rows -> {len(paths)} shard(s), LOCAL ONLY")
    else:
        persist_store(
            args.out_dir,
            arm,
            args.smoke,
            {
                "n_rows": n_done,
                "n_shards": len(paths),
                "render_stats": render_stats,
                "slot_diagnostics": diag,
            },
        )
    print(f"[done] {arm}: {n_done} rows -> {len(paths)} shard(s) in {args.out_dir}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
