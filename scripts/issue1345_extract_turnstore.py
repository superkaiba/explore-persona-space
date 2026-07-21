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
import json
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


def assert_story_character_name(stories_dir: Path, model: str, yield_name: str = "") -> None:
    """Env-mismatch guard (plan v6 §4): stored realized name == runtime constant.

    The gen phase records ``story_character_name`` in ``story_yield_{model}.json``
    (``yield_name`` overrides for the r4/r4op paired yield reports); an extract
    phase launched without the EPM_STORY_CHARACTER_NAME env the gen phase ran
    under fails HERE, at entry, never silently mid-parse. Parent-era yield JSONs
    (no field) read as the ARIA default. Raises AssertionError.
    """
    yield_path = stories_dir / (yield_name or f"story_yield_{model}.json")
    assert yield_path.exists(), f"story yield report missing: {yield_path}"
    stored_name = json.loads(yield_path.read_text()).get("story_character_name", "ARIA")
    assert stored_name == c.STORY_CHARACTER_NAME, (
        f"story character name mismatch: gen phase recorded {stored_name!r} in "
        f"{yield_path} but this extract process runs with "
        f"STORY_CHARACTER_NAME={c.STORY_CHARACTER_NAME!r} — launch both phases "
        "with the same EPM_STORY_CHARACTER_NAME (dispatch --character-name)"
    )


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


def _render_r4(
    stories: list[dict], tokenizer, *, verbatim_check: bool, extra_slots: bool = False
) -> tuple[list, dict]:
    """Paired-story rows -> Rendered (ONE turn per story; conv_id = ORIGINAL id).

    Teacher-forced capture (plan v8 §4): the render is the story text truncated
    at the answer-span end — a SINGLE forward pass over
    [wrapper + question + attribution + verbatim answer]; no generation. Slots
    are identical to r3 (prefix = last token before the question utterance,
    context = last token of the attribution marker) via the SAME
    render_story_turn offset-mapping alignment + BPE seam guard. The row's
    conv_id is the ORIGINAL conversation id — the structural gain that makes
    the r4 corpus data-paired with r1/r2. ``verbatim_check`` re-asserts the gen
    phase's mechanical gate at the extraction trust boundary using the SAME
    normalized matcher (c.norm_text: story[a_start:a_end] == answer up to
    NFKC + curly-quote + whitespace-collapse drift; r4 only — the op
    companion has no fixed answer).

    ``extra_slots=True`` (plan v10 §4 slot ablation): recomputes ``attr_start``
    per story via the SAME ``c.ANSWER_ATTRIB_RE`` the gen gate used (fail-loud
    exactly-one-match assert — the keep gate guarantees it; re-asserted at the
    trust boundary) and threads it into ``render_story_turn(extra_slots=True)``
    so all 5 single positions + the pooled attribution-phrase span come from
    the SAME row (one TF forward per story downstream).
    """
    rendered, stats = [], {"stories": 0, "turns_rendered": 0, "turns_dropped": 0}
    for s in stories:
        stats["stories"] += 1
        assert len(s["parsed_turns"]) == 1, (
            f"paired story {s['conv_id']}: expected exactly 1 parsed turn, "
            f"got {len(s['parsed_turns'])} (gen keep-filter drift)"
        )
        turn = s["parsed_turns"][0]
        attr_start = None
        if extra_slots:
            attribs = list(c.ANSWER_ATTRIB_RE.finditer(s["story"]))
            assert len(attribs) == 1, (
                f"paired story {s['conv_id']}: ANSWER_ATTRIB_RE matched {len(attribs)} "
                "times at extraction — the gen keep gate guarantees exactly one "
                "(attribution_multi/zero rejects); regex/name-seam drift"
            )
            attr_start = attribs[0].start()
        if verbatim_check:
            # The SAME matcher as the gen gate (c.norm_text under
            # c.find_verbatim_occurrences' normalization — cps fix round):
            # the gate accepts NFKC / curly-quote / whitespace-collapse
            # drift, so the trust-boundary re-check compares in that normal
            # form. Exact-match keeps (the carried pre-fix bundle) pass
            # trivially; an accepted story whose stored raw-offset span is
            # NOT the answer under the shared normalization is a fail-loud
            # AssertionError, never a skip.
            assert c.norm_text(s["story"][turn["a_start"] : turn["a_end"]]) == c.norm_text(
                s["answer"]
            ), (
                f"paired story {s['conv_id']}: stored span is not the verbatim answer "
                "under the shared normalized matcher (gen keep-filter drift)"
            )
        r = c.render_story_turn(
            s["story"],
            turn,
            s["conv_id"],
            tokenizer,
            extra_slots=extra_slots,
            attr_start=attr_start,
        )
        if r is None:
            stats["turns_dropped"] += 1
            continue
        stats["turns_rendered"] += 1
        rendered.append(r)
    return rendered, stats


def _assert_prefix_before_context(rendered: list, ctx_name: str) -> None:
    """Parent 2-slot invariant: prefix strictly before the context slot."""
    for r in rendered:
        assert r.slot_idx["prefix"] < r.slot_idx[ctx_name], (r.conv_id, r.slot_idx)


def _slot_order_and_diagnostics(rendered: list, out_dir: Path, stem: str, *, smoke: bool) -> dict:
    """Slot-ablation post-render checks (plan v10 §4): the per-row positional
    sort must realize EXACTLY the canonical SLOT_SINGLE_ORDER (ties keep
    insertion order under the stable sort — the span-ordering chain), the
    pooled read is exactly ctx_attrmean, and the registered diagnostics
    (answer-overlap hard assert 0.0, anchor-coincidence rates, positions) are
    computed + persisted BEFORE any GPU forward."""
    for r in rendered:
        names = [n for n, _ in sorted(r.slot_idx.items(), key=lambda kv: kv[1])]
        assert names == list(c.SLOT_SINGLE_ORDER), (r.conv_id, names, r.slot_idx)
        assert list(r.pooled_spans) == ["ctx_attrmean"], (r.conv_id, list(r.pooled_spans))
    slot_diag = slot_diagnostics(rendered)
    diag_path = out_dir / f"{stem}_slot_diagnostics.json"
    c.write_json(
        diag_path,
        {
            "metadata": c.metadata(0, len(rendered), "scripts/issue1345_extract_turnstore.py"),
            "bundle_revision": c.STORIES_BUNDLE_REV,
            "smoke": smoke,
            **slot_diag,
        },
    )
    coinc = {k: round(v, 4) for k, v in slot_diag["anchor_coincidence_rates"].items()}
    print(
        "[slot-ablation] answer-overlap rates all 0.0 (hard-asserted); "
        f"anchor-coincidence: {coinc} -> {diag_path}",
        flush=True,
    )
    return slot_diag


def _smoke_slice_slot_stories(stories: list[dict], turnstore_dir: Path, n: int = 10) -> list[dict]:
    """Smoke slice for the slot-ablation leg: PREFER stories whose conv_id is
    in the already-STAGED r1 chat store (prefetch_reuse stages shard000 only
    under smoke), so the tiny leg exercises the chat-matched comparator + the
    paired-D verdict deterministically whenever an overlap exists; fill to
    ``n`` with non-overlapping stories otherwise (the downstream drivers then
    take their documented informational smoke skips)."""
    staged_ids: set[str] = set()
    for sp in sorted(Path(turnstore_dir).glob("instruct_chat_s_shard*.json")):
        staged_ids.update(str(x) for x in json.loads(sp.read_text())["conv_ids"])
    overlapping = [s for s in stories if str(s["conv_id"]) in staged_ids]
    rest = [s for s in stories if str(s["conv_id"]) not in staged_ids]
    picked = (overlapping + rest)[:n]
    n_over = sum(1 for s in picked if str(s["conv_id"]) in staged_ids)
    print(
        f"[smoke][slot] {n_over}/{len(picked)} smoke stories overlap the staged r1 "
        f"subset ({len(staged_ids)} staged conv_ids)",
        flush=True,
    )
    return picked


def slot_diagnostics(rendered: list) -> dict:
    """Registered per-slot diagnostics for the slot-ablation store (plan v10 §4).

    Per slot: token positions (exploratory histogram input), ANSWER-OVERLAP
    RATE (fraction of rows whose slot char-span overlaps [a_start, a_end) —
    MUST be 0.0 by construction, HARD-asserted here at the extraction trust
    boundary), and anchor-COINCIDENCE RATE (fraction of rows whose read
    position equals the anchor slot; the attrmean read "coincides" when its
    pooled token span is exactly the anchor's single token). Degeneracy
    policy: coincidence > SLOT_DEGENERACY_COINCIDENCE_MAX flags the slot
    ``degenerate`` (the verdict driver excludes it; reportable, not a crash).
    """
    n = len(rendered)
    assert n > 0, "slot_diagnostics on an empty render"
    single = [name for name in c.SLOT_SINGLE_ORDER]
    positions: dict[str, list] = {name: [] for name in c.SLOT_STORE_ORDER}
    overlap = dict.fromkeys(c.SLOT_STORE_ORDER, 0)
    coincide = dict.fromkeys([*single, "ctx_attrmean"], 0)
    for r in rendered:
        a0, a1 = r.meta["a_char_span"]
        anchor = r.slot_idx["context"]
        for name in single:
            idx = r.slot_idx[name]
            positions[name].append(int(idx))
            s_ch, e_ch = r.meta["slot_char_spans"][name]
            if s_ch < a1 and e_ch > a0:
                overlap[name] += 1
            if idx == anchor:
                coincide[name] += 1
        ps, pe = r.pooled_spans["ctx_attrmean"]
        positions["ctx_attrmean"].append([int(ps), int(pe)])
        cs, ce = r.meta["pooled_char_spans"]["ctx_attrmean"]
        if cs < a1 and ce > a0:
            overlap["ctx_attrmean"] += 1
        if (ps, pe) == (anchor, anchor + 1):
            coincide["ctx_attrmean"] += 1
    overlap_rates = {name: overlap[name] / n for name in overlap}
    # HARD assert (plan v10 §4): every slot definition is fully-contained-
    # before, so any overlap with the answer span is a render bug.
    assert all(v == 0.0 for v in overlap_rates.values()), (
        f"answer-overlap rate nonzero: { {k: v for k, v in overlap_rates.items() if v} } — "
        "a slot read position overlaps the answer span (fully-contained-before "
        "idiom violated; plan v10 §4 registered diagnostic)"
    )
    coincidence_rates = {name: coincide[name] / n for name in coincide}
    verdict_slot_names = {k: c.SLOT_NAME_FOR_CELL[cid] for k, cid in c.SLOT_VERDICT_CELLS.items()}
    degenerate = {
        k: coincidence_rates[sn] > c.SLOT_DEGENERACY_COINCIDENCE_MAX
        for k, sn in verdict_slot_names.items()
    }
    return {
        "n_rows": n,
        "slot_order": list(c.SLOT_STORE_ORDER),
        "answer_overlap_rates": overlap_rates,
        "anchor_coincidence_rates": coincidence_rates,
        "degeneracy_threshold": c.SLOT_DEGENERACY_COINCIDENCE_MAX,
        "degenerate_verdict_slots": degenerate,
        "positions": positions,
    }


def _regime_choices() -> tuple[str, ...]:
    """argparse --regime choices, variant-gated.

    r4op = the on-policy companion CONTROL store for the TF paired round (a fit
    cell, not a transfer/opcomp regime — hence appended, not in c.REGIMES);
    under the on-policy round it is the PRIMARY story regime and already IN
    c.REGIMES (deduped). Slot-ablation variant (plan v10): r4 is legal for the
    multi-slot re-read of the REUSED paired corpus (no gen phase in that mode).
    """
    if c.HAS_R4:
        return (*c.REGIMES, "r4op")
    if c.HAS_ONPOLICY_STORY:
        return tuple(dict.fromkeys(c.REGIMES))
    if c.HAS_SLOT_ABLATION:
        return (*c.REGIMES, "r4")
    return tuple(c.REGIMES)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regime", choices=_regime_choices(), required=True)
    ap.add_argument(
        "--slot-ablation",
        action="store_true",
        help="plan v10 multi-slot TF re-read: legal ONLY with --regime r4 under "
        "EPM_I1345_VARIANT=story_slot_ablation; captures all 5 single slot "
        "positions + the pooled attribution-phrase mean from ONE forward per "
        "story into the r4slot stem",
    )
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
    if args.slot_ablation:
        assert c.HAS_SLOT_ABLATION, (
            f"--slot-ablation requires EPM_I1345_VARIANT in {c.SLOT_ABLATION_VARIANTS} "
            f"(got {c.VARIANT!r}) — the slot store is variant-scoped by design"
        )
        assert args.regime == "r4", "--slot-ablation is defined for --regime r4 only (plan v10)"
        assert args.model in c.R4_MODELS, "slot ablation is instruct-only (base N/A by scope)"

    model, tokenizer, model_id = ex.load_model(args.model, tiny_model_dir=args.tiny_model_dir)

    if args.regime in ("r1", "r2"):
        convs = _load_track_s_conversations(args.dl_dir)
        if args.smoke:
            convs = convs[:8]
            print(f"[smoke] limiting to {len(convs)} conversations", flush=True)
        rendered = _render_r1r2(convs, tokenizer, args.regime)
        render_stats = {"conversations": len(rendered)}
    elif args.regime == "r3":
        assert_story_character_name(args.stories_dir, args.model)
        kept_path = args.stories_dir / f"kept_stories_{args.model}.jsonl"
        stories = c.read_jsonl(kept_path)
        if args.smoke:
            stories = stories[:8]
            print(f"[smoke] limiting to {len(stories)} stories", flush=True)
        rendered, render_stats = _render_r3(stories, tokenizer)
        assert rendered, "no story turns rendered — parser/render drift"
    else:  # r4 (TF paired) / r4op (on-policy companion) — plan v8 §4
        assert args.model in c.R4_MODELS, (
            f"{args.regime} is N/A by scope for model {args.model} (plan v8 §5)"
        )
        mode_slug = "paired" if args.regime == "r4" else "paired_op"
        assert_story_character_name(
            args.stories_dir, args.model, yield_name=f"story_yield_{mode_slug}_{args.model}.json"
        )
        kept_path = args.stories_dir / f"kept_stories_{mode_slug}_{args.model}.jsonl"
        stories = c.read_jsonl(kept_path)
        if not args.smoke and args.slot_ablation:
            # Bundle-integrity gate (plan v10 §7): the staged pinned bundle
            # parses to exactly the landed kept count — no regeneration path
            # exists in this mode by design.
            assert len(stories) == c.STORIES_BUNDLE_N_ROWS, (
                f"kept-stories bundle has {len(stories)} rows, expected "
                f"{c.STORIES_BUNDLE_N_ROWS} (pinned @ {c.STORIES_BUNDLE_REV[:10]})"
            )
        if args.smoke:
            if args.slot_ablation:
                stories = _smoke_slice_slot_stories(stories, args.out_dir)
            else:
                stories = stories[:8]
            print(f"[smoke] limiting to {len(stories)} paired stories", flush=True)
        rendered, render_stats = _render_r4(
            stories,
            tokenizer,
            verbatim_check=(args.regime == "r4"),
            extra_slots=args.slot_ablation,
        )
        assert rendered, "no paired story turns rendered — parser/render drift"

    # Parent-parity degenerate-row filter (#1345 crash-fix r6; ports the #825
    # naturalistic_s crash-fix from the issue-825 branch): a short single-turn
    # answer that BPE-merges entirely into the naturalistic plain-text
    # delimiters renders a zero-width (anchor, anchor) content span, which the
    # extractor's hard `1 <= s < e` assert kills mid-GPU-run (att-20260715-195605:
    # s57, response 1 char). Drop such rows PER RENDER — exactly the parent's
    # semantics (parent chat_s kept 5000/5000, naturalistic_s kept 4724/5000;
    # both arms of one render share the row set by construction, and the
    # matched-n build intersects conv_ids downstream, so R1/R2 alignment + the
    # per-store parity anchors stay valid). Skip manifest (conv ids only —
    # never corpus text) persists next to the shards.
    n_pre_filter = len(rendered)
    rendered, drops = ex.partition_rendered(rendered)
    stem = c.stem_for(args.model, "r4slot" if args.slot_ablation else args.regime)
    manifest_path = args.out_dir / f"{stem}_skip_manifest.json"
    c.write_json(
        manifest_path,
        {
            "metadata": c.metadata(0, n_pre_filter, "scripts/issue1345_extract_turnstore.py"),
            "regime": args.regime,
            "model": args.model,
            "n_rendered_pre_filter": n_pre_filter,
            "n_dropped_zero_width": len(drops),
            "dropped_conv_ids": [d["conv_id"] for d in drops],
            "dropped_turns": {d["conv_id"]: d["turns"] for d in drops},
        },
    )
    print(
        f"[extract] skipped {len(drops)} degenerate rows (manifest {manifest_path})",
        flush=True,
    )
    assert rendered, (
        f"all {n_pre_filter} rendered rows dropped as zero-width — a systematic "
        f"{c.REGIME_FORMAT[args.regime]} render bug, not a handful of degenerate rows"
    )
    ex.assert_residual_span_integrity(rendered)

    # Slot-order invariant the fit registry depends on: prefix strictly before
    # the context slot in EVERY row (extractor sorts slots by position, so
    # slot_index 0 = prefix, 1 = context across all three regimes). Slot
    # ablation generalizes the chain: the per-row positional sort must realize
    # EXACTLY the canonical SLOT_SINGLE_ORDER (ties keep insertion order under
    # the stable sort — plan v10 §4 span-ordering chain), with the pooled
    # attribution-mean read appended by process_batch as storage index 5.
    ctx_name = "a1" if args.regime in ("r1", "r2") else "context"
    if args.slot_ablation:
        _slot_order_and_diagnostics(rendered, args.out_dir, stem, smoke=bool(args.smoke))
    else:
        _assert_prefix_before_context(rendered, ctx_name)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    do_causal = args.smoke  # parent default: causal check on smoke
    causal_max_diff = None
    if do_causal:
        # mode="cosine": the NEW early-position `prefix` slot (token ~2 in R1)
        # has NO bf16 headroom under the parent's flat atol — the 3-token
        # prefix forward vs the full-length forward differs by a single bf16
        # ULP at the large-magnitude early-token dims (0.03125/0.0625 at
        # layer 0, att-20260715-151246; fp32-verified benign). The #779
        # two-bar cosine gate + norm guard keeps the wrong-position bug
        # catcher (real bugs read cos ~0.4-0.6) with calibrated headroom.
        causal_max_diff = ex.causal_check(model, rendered[: min(3, len(rendered))], mode="cosine")

    bs = 8 if args.batch_size == "auto" else int(args.batch_size)
    if args.smoke:
        bs = min(bs, 2)
    peak_layers = sorted(li for li in ex.FROZEN_LAYERS if li < ex.EXPECTED_LAYERS)  # parent default
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
        # Story-arm character name (plan v6 §4 provenance; "ARIA" on r1/r2 rows too
        # — the constant is global, only the r3 render consumes it).
        "story_character_name": c.STORY_CHARACTER_NAME,
        "slot_names": (list(c.SLOT_STORE_ORDER) if args.slot_ablation else ["prefix", ctx_name]),
        "slot_ablation": bool(args.slot_ablation),
        "peak_layers": peak_layers,
        "expected_layers": ex.EXPECTED_LAYERS,
        "expected_hidden": ex.EXPECTED_HIDDEN,
        "render_stats": render_stats,
        "git_commit": c.git_commit(),
        "pinned_parent_revision": c.PIN_REV,
        "causal_check_max_abs_diff": causal_max_diff,
        "causal_check_mode": "cosine" if do_causal else None,
        "smoke": bool(args.smoke),
        # Parent-parity zero-width-span drops (#825 crash-fix semantics),
        # mirrored into every shard sidecar alongside the standalone manifest.
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
    print(f"[done] {n_done} rows -> {len(paths)} shard(s) in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
