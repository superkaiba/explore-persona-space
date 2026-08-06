"""Issue #2094 SMOKE-ONLY support: synthetic judge scores + transport mini-mirror.

Two subcommands, used ONLY by the unit-F end-to-end pipeline smoke — never by
any production phase (production judging is ``issue2094_judge.py``; production
transport inputs come from the pod run):

``synth-scores``
    The smoke's stand-in for the REMOTE Anthropic API boundary (the one seam a
    tiny-real CPU smoke is allowed to fake). Enumerates judge items through
    unit D's PRODUCTION builders (``build_coherence_items`` /
    ``build_grid_behavior_items`` / ``build_anchor_behavior_items`` /
    ``build_stage2_behavior_items``) over the REAL smoke rollouts, then writes
    ``<work-root>/scores/<wave>.scores.jsonl`` rows in ``run_wave``'s exact
    schema with DETERMINISTIC seeded scores (dose-graded movement toward the
    target context; a small deterministic incoherent fraction so the
    exclusion/overlay paths execute). Request routing + custom_ids are
    separately validated by ``issue2094_judge.py --phase waves --dry-run``;
    the response parse contract is pinned by unit D's rule-27 round-trip
    tests.

``transport-mini``
    Builds a PRODUCTION-DIMENSION (28 layers x hidden 3584) mini mirror —
    vc_bank + anchor V_a + a few ce/pe grid shards at the banked-map layers —
    so ``issue2094_analysis.py --phase parity`` + ``--phase transport`` run the
    REAL entrypoint against the REAL HF-staged banked maps. The tiny CPU smoke
    model (hidden 8) cannot emit 3584-dim states by construction, so the
    tensor CONTENT here is seeded-random at the real shape (the GPU-scale-
    weights seam); every code path (payload reconstruction, donor nulls,
    orientation binding, ridge apply) is the production one.

Both subcommands REFUSE to write outside a path containing "smoke" unless
``--force`` (never let smoke fixtures land in canonical roots).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402

from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2094_judge as J  # noqa: E402
import issue2094_run as R  # noqa: E402

logger = logging.getLogger("issue2094_smoke_support")

N_LAYERS, HIDDEN = 28, 3584

# Deterministic dose-graded movement toward the target context (side-b rubric
# up, side-a rubric down). Values are arbitrary but non-degenerate: F_beh lands
# strictly between floor and ceiling and varies by dose.
MOVEMENT = {"a0.5": 0.25, "a1": 0.4, "a2": 0.55, "a4": 0.7, "replace": 0.85}
NULL_MOVEMENT = 0.05
INCOHERENT_EVERY = 12  # ~8 percent of draws deterministically incoherent


def _h(item_id: str) -> int:
    return int.from_bytes(hashlib.sha256(item_id.encode()).digest()[:4], "big")


def _guard_smoke_path(path: Path, force: bool) -> None:
    if "smoke" not in str(path) and not force:
        raise SystemExit(
            f"[smoke-support] REFUSED: {path} does not look like a smoke scratch path "
            "(pass --force only if you know why)"
        )


def _coherence_score(item_id: str) -> int:
    h = _h(item_id)
    if h % INCOHERENT_EVERY == 0:
        return 30  # deterministic incoherent minority (exclusion path runs)
    return 78 + h % 18


def _match_score(match: bool, item_id: str) -> int:
    h = _h(item_id) % 8
    return (82 + h) if match else (12 + h)


def _rubric_matches_context(rubric_id: str, context_id: str) -> bool:
    prefix, query = context_id.split("__", 1)
    if rubric_id.startswith("fq-"):
        return rubric_id.removeprefix("fq-") == query
    assert rubric_id.startswith("fp-"), rubric_id
    return rubric_id.removeprefix("fp-") == prefix


def _moved_score(side: str, movement: float, item_id: str) -> int:
    jitter = _h(item_id) % 6
    if side == "a":
        return int(round(85 - 70 * movement)) + jitter - 3
    return int(round(15 + 70 * movement)) + jitter - 3


def _grid_movement(source: dict) -> float:
    m = MOVEMENT.get(source.get("dose"), 0.5)
    return NULL_MOVEMENT if source.get("arm") == "null" else m


def _stage2_movement(source: dict) -> float:
    cell = str(source.get("cell") or "")
    for dose, m in MOVEMENT.items():
        if f"|{dose}|" in cell:
            return m
    return 0.5


def _score_unit(unit: J.JudgeUnit) -> int:
    src = unit.source
    if unit.rubric_id == J.COHERENCE_RUBRIC_ID:
        return _coherence_score(unit.item_id)
    if src["kind"] == "anchor":
        return _match_score(
            _rubric_matches_context(unit.rubric_id, src["context_id"]), unit.item_id
        )
    movement = _grid_movement(src) if src["kind"] == "grid" else _stage2_movement(src)
    return _moved_score(src["side"], movement, unit.item_id)


def _write_wave(scores_dir: Path, wave: str, rubric_id: str, units: list[J.JudgeUnit]) -> int:
    rows = [
        {
            "item_id": u.item_id,
            "wave": wave,
            "rubric_id": rubric_id,
            "score": _score_unit(u),
            "n_kept_draws": 1,
            "transport_lost_residual": 0,
            "synthetic_smoke_score": True,  # loud provenance flag on every row
            **u.source,
        }
        for u in units
    ]
    J._write_jsonl_atomic(scores_dir / f"{wave}.scores.jsonl", rows)
    return len(rows)


def cmd_synth_scores(args: argparse.Namespace) -> int:
    _guard_smoke_path(args.work_root, args.force)
    logger.warning(
        "[smoke-support] SMOKE-ONLY synthetic judge scores -> %s (the remote API "
        "boundary is the ONE faked seam; every row carries synthetic_smoke_score=true)",
        args.work_root,
    )
    scores_dir = args.work_root / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    pairs = J.pair_index()
    n = 0

    grid_rows = J.load_grid_rows(args.rollouts_dir)
    anchor_rows = J.load_anchor_rows(args.anchors_file)
    n += _write_wave(
        scores_dir,
        "coherence.grid",
        J.COHERENCE_RUBRIC_ID,
        J.build_coherence_items(grid_rows, None),
    )
    n += _write_wave(
        scores_dir,
        "coherence.anchors",
        J.COHERENCE_RUBRIC_ID,
        J.build_coherence_items(None, anchor_rows),
    )
    for rid, units in sorted(J.build_grid_behavior_items(grid_rows, pairs).items()):
        n += _write_wave(scores_dir, f"{rid}.grid", rid, units)
    for rid, units in sorted(J.build_anchor_behavior_items(anchor_rows, pairs).items()):
        n += _write_wave(scores_dir, f"{rid}.anchors", rid, units)
    if args.stage2_dir is not None:
        s2_rows = J.load_stage2_rows(args.stage2_dir)
        n += _write_wave(
            scores_dir,
            "coherence.stage2",
            J.COHERENCE_RUBRIC_ID,
            J.build_coherence_items(None, None, s2_rows),
        )
        for rid, units in sorted(J.build_stage2_behavior_items(s2_rows, pairs).items()):
            n += _write_wave(scores_dir, f"{rid}.stage2", rid, units)
    logger.info("[smoke-support] synth-scores done: %d rows", n)
    return 0


# ── transport mini-mirror ──────────────────────────────────────────────


def _mini_bank(pairs: list[BANK.Pair], gen: torch.Generator) -> dict:
    contexts = BANK.build_contexts()
    per_context = {}
    for cid, ctx in contexts.items():
        nq = 6
        per_context[cid] = {
            "context_id": cid,
            "prefix": ctx["prefix"],
            "query_id": ctx["query_id"],
            "ctx_len": 40,
            "prefix_end": 40 - nq,
            "nq": nq,
            "q_span": torch.randn(nq, N_LAYERS, HIDDEN, generator=gen),
            "v_pe": torch.randn(N_LAYERS, HIDDEN, generator=gen),
        }
    centroids = {
        prefix: torch.randn(N_LAYERS, HIDDEN, generator=gen) for prefix in BANK.PREFIX_ORDER
    }
    return {
        "layers": list(range(N_LAYERS)),
        "per_context": per_context,
        "centroids": centroids,
        "donor_derangement": BANK.donor_derangement(pairs),
        "bank_sha": "smoke-transport-mini",
        "repro": {"note": "SMOKE-ONLY production-shape synthetic bank"},
    }


def cmd_transport_mini(args: argparse.Namespace) -> int:
    _guard_smoke_path(args.in_root, args.force)
    logger.warning("[smoke-support] SMOKE-ONLY production-dim transport mirror -> %s", args.in_root)
    gen = torch.Generator().manual_seed(args.seed)
    pairs = BANK.build_pairs()
    mirror = args.in_root / "issue2094_singlepos"
    subset = [p for p in pairs if p.setting == "matched_prefix"][:2]
    bank = _mini_bank(pairs, gen)
    bank_dir = mirror / "analysis_tensors" / "vc_bank"
    bank_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bank, bank_dir / "vc_bank.pt")

    ctx_ids = sorted({c for p in subset for c in (p.a, p.b)})
    k_draws = 2
    index, span, tail = [], [], []
    for cid in ctx_ids:
        for d in range(k_draws):
            index.append({"context_id": cid, "draw": d})
            span.append(torch.randn(N_LAYERS, HIDDEN, generator=gen))
            tail.append(torch.randn(N_LAYERS, HIDDEN, generator=gen))
    anch_dir = mirror / "analysis_tensors" / "anchors"
    anch_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "index": index,
            "va_span": torch.stack(span).to(torch.float16),
            "va_tail": torch.stack(tail).to(torch.float16),
            "pooling": {"va_span": "smoke", "va_tail": "smoke"},
        },
        anch_dir / "va_anchors.pt",
    )

    blocks = [
        R.Block(slot, lv, dose, "A", arm, tuple(p.pair_id for p in subset))
        for slot, lv, dose in (("ce", "L14", "a1"), ("ce", "L14", "replace"), ("pe", "L19", "a1"))
        for arm in ("steered", "null")
    ]
    roll_dir = mirror / "raw_completions" / "grid"
    va_dir = mirror / "analysis_tensors" / "va_store"
    roll_dir.mkdir(parents=True, exist_ok=True)
    va_dir.mkdir(parents=True, exist_ok=True)
    for block in blocks:
        rows = [
            {
                "block_key": block.key,
                "slot": block.slot,
                "layer_variant": block.layer_variant,
                "layers": [int(block.layer_variant[1:])],
                "dose": block.dose,
                "alpha": None if block.dose == "replace" else float(block.dose[1:]),
                "vec_type": block.vec_type,
                "arm": block.arm,
                "pair_id": p.pair_id,
                "setting": p.setting,
                "context_a": p.a,
                "context_b": p.b,
                "cap_hit": False,
                "text": "smoke transport row (no model text)",
                "draw": 0,
            }
            for p in subset
        ]
        R._write_jsonl_atomic(roll_dir / f"shard_{block.slug}.jsonl", rows)
        n = len(rows)
        torch.save(
            {
                "index": [{"pair_id": r["pair_id"], "draw": 0} for r in rows],
                "va_span": torch.randn(n, N_LAYERS, HIDDEN, generator=gen).to(torch.float16),
                "va_tail": torch.randn(n, N_LAYERS, HIDDEN, generator=gen).to(torch.float16),
                "empty_rows": [],
            },
            va_dir / f"shard_{block.slug}.pt",
        )
    logger.info(
        "[smoke-support] transport-mini done: %d blocks x %d pairs (banked layers ce/L14, "
        "pe/L19) under %s",
        len(blocks),
        len(subset),
        mirror,
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ss = sub.add_parser("synth-scores", help="SMOKE-ONLY synthetic judge scores")
    ss.add_argument("--rollouts-dir", type=Path, required=True)
    ss.add_argument("--anchors-file", type=Path, required=True)
    ss.add_argument("--stage2-dir", type=Path, default=None)
    ss.add_argument("--work-root", type=Path, required=True)
    ss.add_argument("--force", action="store_true")
    tm = sub.add_parser("transport-mini", help="SMOKE-ONLY production-dim transport mirror")
    tm.add_argument("--in-root", type=Path, required=True)
    tm.add_argument("--seed", type=int, default=20940)
    tm.add_argument("--force", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.cmd == "synth-scores":
        return cmd_synth_scores(args)
    assert args.cmd == "transport-mini", args.cmd
    return cmd_transport_mini(args)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
