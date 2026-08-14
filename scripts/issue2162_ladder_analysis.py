#!/usr/bin/env python3
"""Issue #2162 persona-specificity-ladder — off-pod P6 analysis driver (plan §4.4/§6).

Consumes the pod's P2-P4 artifacts (staged from HF) + the VM judge driver's
score/gate outputs, and writes the round's registered tables, statistics and
figures:

- ``--step f-tables``: ``anchors.jsonl`` (per rung x carrier anchor stats +
  gate verdicts), ``f_cells.jsonl`` (STEERED cells), ``null_samevalue_cells
  .jsonl``, ``null_crosstype_cells.jsonl`` (one row per pair x slot x arm:
  F_target primary, netted dual-rubric F bridge, plain mirror, F_act via
  ``experiments/issue2094/fmetrics.f_act``), and ``conjuncts.jsonl`` (R1/R2
  steered per-conjunct diagnostic).
- ``--step margin``: ``margin.jsonl`` — TF fixed-pool margins per cell +
  anchor margins + margin shift (pool key = direction id).
- ``--step stats``: ``stats.json`` — the 4 Holm-corrected within-carrier
  rung-label permutation trend tests (seed 21625, B=10,000), per-(direction
  x slot x arm) carrier-clustered bootstrap CIs (seed 21626, B=10,000, via
  ``issue2094_analysis.bootstrap_family_means_batched``), the registered
  verdict lattice + H3 null-sanity flags, the H4 paired asymmetry, and the
  rule-19 margin validation (``issue2162_analysis.rule19_validation``).
- ``--step figures``: the 7 planned-manifest figures.

Statistical-input existence (plan §6): before computing, the driver
set-checks that every registered (direction x slot x arm x carrier) row for
gate-surviving rungs exists in the loaded tables — a missing registered row
is a RuntimeError, never a silent shrink. Empty selections fail LOUD with
per-predicate reject counters.

Coherence convention (plan §4.4 "all reported quantities over coherent
draws"): grid draws AND anchor draws entering F normalization are filtered
to coherence > 60; the anchor-separation GATE quantities are copied verbatim
from the judge's gate JSON (which uses all scored draws) and both filtered +
raw anchor means are recorded in ``anchors.jsonl`` for audit.

Usage (VM, after the pod's grid upload + the judge's waves/conjuncts):
    uv run python scripts/issue2162_ladder_analysis.py --step f-tables --stage-from-hf
    uv run python scripts/issue2162_ladder_analysis.py --step margin
    uv run python scripts/issue2162_ladder_analysis.py --step stats
    uv run python scripts/issue2162_ladder_analysis.py --step figures
"""

from __future__ import annotations

# load_dotenv BEFORE any heavy/HF import (lint: --check-dotenv-before-hf-import)
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2094_judge as J94  # noqa: E402  (reused: _iter_jsonl, _repro)
import issue2162_analysis as A62  # noqa: E402  (parent loaders + rule19 + writers)
import issue2162_judge as J62  # noqa: E402  (row loaders)
import issue2162_ladder_judge as LJ  # noqa: E402  (gate paths, conjunct registry)
from explore_persona_space.experiments.issue2094 import fmetrics as FM  # noqa: E402
from explore_persona_space.experiments.issue2162 import ladder_bank as LB  # noqa: E402
from issue2094_analysis import bootstrap_family_means_batched  # noqa: E402

logger = logging.getLogger("issue2162.ladder_analysis")

# ── registered constants (plan §6) ────────────────────────────────────
TREND_SEED = 21625  # within-carrier rung-label permutation battery
BOOT_SEED = 21626  # carrier-clustered bootstrap (estimation + H4)
N_BOOT_DEFAULT = 10_000
N_PERM_DEFAULT = 10_000
NULL_SANITY_BAR = 0.10  # H3: install-direction null mean above this => flagged
MIN_TREND_RUNGS = 3  # below => descriptive, no NHST
HOLM_M = 4
ARMS: tuple[str, ...] = ("steered", "null_sameval", "null_xtype")
SLOTS: tuple[str, ...] = ("ce", "pe")
FAMILIES: tuple[str, ...] = ("install-ce", "install-pe", "erase-ce", "erase-pe")
COHERENCE_THRESHOLD = A62.COHERENCE_THRESHOLD  # 60.0

ARM_COLORS = {  # one color = one meaning across every figure
    "steered": "#1f77b4",
    "null_sameval": "#ff7f0e",
    "null_xtype": "#2ca02c",
}
ARM_LABELS = {
    "steered": "steered",
    "null_sameval": "same-value-donor null",
    "null_xtype": "cross-type-donor null",
}


def _mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


# ── score-row loaders (join on SOURCE fields — the grid (context_id, draw)
#    key collides across slots/arms by construction, so grid joins key on
#    (pair_id, slot, arm, draw)) ─────────────────────────────────────────


def _wave_rows(scores_dir: Path, wave: str, *, required: bool = True) -> list[dict]:
    path = scores_dir / f"{wave}.scores.jsonl"
    if not path.exists():
        if required:
            raise FileNotFoundError(f"missing judge wave scores: {path}")
        return []
    return list(J94._iter_jsonl(path))


def load_grid_scores(
    scores_dir: Path, rid: str, *, required: bool = True
) -> dict[tuple[str, str, str, int], float]:
    """(pair_id, slot, arm, draw) -> kept score for one grid wave (None-score
    rule-9 drops SKIPPED, never coerced)."""
    out: dict[tuple[str, str, str, int], float] = {}
    for r in _wave_rows(scores_dir, f"{rid}.grid", required=required):
        if r.get("score") is None:
            continue
        out[(r["pair_id"], r["slot"], r["arm"], int(r["draw"]))] = float(r["score"])
    return out


def load_anchor_scores(scores_dir: Path, rid: str) -> dict[tuple[str, int], float]:
    """(context_id, draw) -> kept score for one anchor wave."""
    out: dict[tuple[str, int], float] = {}
    for r in _wave_rows(scores_dir, f"{rid}.anchors"):
        if r.get("score") is None:
            continue
        out[(r["context_id"], int(r["draw"]))] = float(r["score"])
    return out


def _read_gate(gates_dir: Path) -> dict:
    path = gates_dir / "ladder_separation_gate.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rungs = payload.get("rungs")
    assert isinstance(rungs, dict) and rungs, f"{path} carries no 'rungs' object"
    for v in LB.PERSONA_VALUE_IDS:
        assert v in rungs, (v, "rung missing from gate verdict")
    return payload


def _read_donor_screen(gates_dir: Path) -> dict[str, dict]:
    path = gates_dir / "ladder_donor_screen.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assignments = payload.get("assignments")
    assert isinstance(assignments, dict), f"{path} carries no 'assignments'"
    return assignments


def registered_row_keys(gate: dict, screen: dict[str, dict]) -> set[tuple[str, str, str, str]]:
    """The §6 registered row set: (direction, slot, arm, carrier) for every
    gate-surviving (rung x carrier), minus screen-dropped null_xtype pairs."""
    out: set[tuple[str, str, str, str]] = set()
    for v in LB.PERSONA_VALUE_IDS:
        rec = gate["rungs"][v]
        if not rec.get("survived"):
            continue
        for carrier in rec["surviving_carriers"]:
            for kind in ("install", "erase"):
                direction = f"{kind}_{v}"
                pair_id = f"{direction}::{carrier}"
                for slot in SLOTS:
                    for arm in ARMS:
                        if (
                            arm == "null_xtype"
                            and screen.get(pair_id, {}).get("status") == "dropped"
                        ):
                            continue
                        out.add((direction, slot, arm, carrier))
    return out


def set_check_registered_rows(
    expected: set[tuple[str, str, str, str]],
    present: set[tuple[str, str, str, str]],
    where: str,
) -> None:
    """Plan §6 statistical-input existence: registered rows must be a subset
    of the realized tables — fail LOUD naming every missing key."""
    missing = sorted(expected - present)
    if missing:
        raise RuntimeError(
            f"[set-check:{where}] {len(missing)} registered rows missing from the "
            f"realized tables (first 10): {missing[:10]}"
        )
    logger.info("[set-check:%s] OK — %d registered rows all present", where, len(expected))


# ── anchor stats (F normalization + anchors.jsonl) ────────────────────


def anchor_stats(
    anchor_rows: list[dict],
    scores_dir: Path,
    gate: dict,
) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    """Per (persona rung x carrier): coherent-filtered anchor means for the F
    normalization + the gate's own per-carrier verdict (copied verbatim).

    Returns (anchors_out_rows, norm) with norm[(value_id, carrier)] carrying
    floor/ceil target, netted-delta and plain means over COHERENT draws.
    """
    coh = load_anchor_scores(scores_dir, J94.COHERENCE_RUBRIC_ID)
    s_plain = load_anchor_scores(scores_dir, LB.holistic_rubric_id("plain"))
    draws_by_ctx: dict[str, list[int]] = defaultdict(list)
    for r in anchor_rows:
        draws_by_ctx[r["context_id"]].append(int(r["draw"]))

    anchors_out: list[dict] = []
    norm: dict[tuple[str, str], dict] = {}
    for value_id in LB.PERSONA_VALUE_IDS:
        rec = gate["rungs"][value_id]
        s_x = load_anchor_scores(scores_dir, LB.holistic_rubric_id(value_id))
        v = LB.VALUES_BY_ID[value_id]
        for carrier, gate_pc in sorted(rec["per_carrier"].items()):
            floor_ctx = LB.context_id("plain", carrier)
            ceil_ctx = LB.context_id(value_id, carrier)

            def _ctx(ctx: str) -> dict:
                draws = sorted(set(draws_by_ctx.get(ctx, [])))
                kept = [d for d in draws if (coh.get((ctx, d)) or 0.0) > COHERENCE_THRESHOLD]
                tgt_coh = [s_x[(ctx, d)] / 100.0 for d in kept if (ctx, d) in s_x]
                tgt_raw = [s_x[(ctx, d)] / 100.0 for d in draws if (ctx, d) in s_x]
                plain_coh = [s_plain[(ctx, d)] / 100.0 for d in kept if (ctx, d) in s_plain]
                deltas = [
                    (s_x[(ctx, d)] - s_plain[(ctx, d)]) / 100.0
                    for d in kept
                    if (ctx, d) in s_x and (ctx, d) in s_plain
                ]
                return {
                    "n_draws": len(draws),
                    "n_coherent": len(kept),
                    "target_coh": _mean(tgt_coh),
                    "target_raw": _mean(tgt_raw),
                    "plain_coh": _mean(plain_coh),
                    "delta_coh": _mean(deltas),
                    "n_target_scored": len(tgt_coh),
                }

            fl, ce = _ctx(floor_ctx), _ctx(ceil_ctx)
            sep_coh = (
                ce["target_coh"] - fl["target_coh"]
                if ce["target_coh"] is not None and fl["target_coh"] is not None
                else None
            )
            netted_sep_coh = (
                ce["delta_coh"] - fl["delta_coh"]
                if ce["delta_coh"] is not None and fl["delta_coh"] is not None
                else None
            )
            norm[(value_id, carrier)] = {
                "floor_target": fl["target_coh"],
                "ceil_target": ce["target_coh"],
                "floor_delta": fl["delta_coh"],
                "ceil_delta": ce["delta_coh"],
                "floor_plain": fl["plain_coh"],
                "ceil_plain": ce["plain_coh"],
            }
            anchors_out.append(
                {
                    "rung": value_id,
                    "rung_label": v.rung,
                    "rung_rank": v.rung_rank,
                    "carrier": carrier,
                    "floor_context": floor_ctx,
                    "ceil_context": ceil_ctx,
                    "rung_survived": bool(rec.get("survived")),
                    "n_carriers_pass": int(rec.get("n_carriers_pass", 0)),
                    "gate_passed": bool(gate_pc.get("passed")),
                    "gate_unscored": bool(gate_pc.get("unscored")),
                    "gate_target_sep": gate_pc.get("target_sep"),
                    "gate_netted_sep": gate_pc.get("netted_sep"),
                    "target_sep_coherent": sep_coh,
                    "netted_sep_coherent": netted_sep_coh,
                    "floor_target_coherent": fl["target_coh"],
                    "ceil_target_coherent": ce["target_coh"],
                    "floor_target_raw": fl["target_raw"],
                    "ceil_target_raw": ce["target_raw"],
                    "floor_delta_coherent": fl["delta_coh"],
                    "ceil_delta_coherent": ce["delta_coh"],
                    "floor_plain_coherent": fl["plain_coh"],
                    "ceil_plain_coherent": ce["plain_coh"],
                    "n_floor_draws": fl["n_draws"],
                    "n_floor_coherent": fl["n_coherent"],
                    "n_ceil_draws": ce["n_draws"],
                    "n_ceil_coherent": ce["n_coherent"],
                }
            )
    assert anchors_out, "anchor_stats produced no rows — empty anchors input"
    return anchors_out, norm


def _f_from_stats(
    kind: str, s_mean: float | None, floor: float | None, ceil: float | None
) -> float | None:
    """Install: (s - floor)/(ceil - floor); erase: (ceil - s)/(ceil - floor)."""
    if s_mean is None or floor is None or ceil is None:
        return None
    denom = ceil - floor
    if abs(denom) < 1e-9:
        return None
    if kind == "install":
        return (s_mean - floor) / denom
    assert kind == "erase", kind
    return (ceil - s_mean) / denom


# ── step: f-tables ────────────────────────────────────────────────────


def step_f_tables(args: argparse.Namespace) -> None:
    gate = _read_gate(args.gates_dir)
    screen = _read_donor_screen(args.gates_dir)
    pairs = {p.pair_id: p for p in LB.build_ladder_pairs(LB.SEED)}
    anchor_rows = J62.load_anchor_rows(args.anchors_dir)
    grid_rows = J62.load_grid_rows(args.rollouts_dir)

    anchors_out, norm = anchor_stats(anchor_rows, args.scores_dir, gate)
    A62._write_jsonl_atomic(args.out_dir / "anchors.jsonl", anchors_out)

    coh_grid = load_grid_scores(args.scores_dir, J94.COHERENCE_RUBRIC_ID)
    plain_grid = load_grid_scores(args.scores_dir, LB.holistic_rubric_id("plain"))
    # Gate-dropped rungs generate NO grid rollouts and NO judge waves, so the
    # target-wave loads are restricted to gate-surviving rungs; a missing wave
    # for a SURVIVING rung still raises FileNotFoundError (real missing data).
    surviving = [v for v in LB.PERSONA_VALUE_IDS if gate["rungs"][v].get("survived")]
    dropped = [v for v in LB.PERSONA_VALUE_IDS if v not in surviving]
    if dropped:
        logger.info(
            "[f-tables] gate-dropped rungs skipped (no grid rollouts / judge waves): %s",
            ", ".join(dropped),
        )
    target_grid = {
        v: load_grid_scores(args.scores_dir, LB.holistic_rubric_id(v)) for v in surviving
    }
    va_grid = A62._load_va_store(args.va_dir)
    va_anchor = A62._load_anchor_va(args.anchor_va_dir)
    anchor_draws_by_ctx: dict[str, list[int]] = defaultdict(list)
    for r in anchor_rows:
        anchor_draws_by_ctx[r["context_id"]].append(int(r["draw"]))

    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in grid_rows:
        by_cell[(row["pair_id"], row["slot"], row["arm"])].append(row)
    assert by_cell, "no grid cells — empty grid input"

    reject = Counter()
    tables: dict[str, list[dict]] = {a: [] for a in ARMS}
    for (pair_id, slot, arm), rows in sorted(by_cell.items()):
        p = pairs[pair_id]
        t_scores = target_grid[p.persona]
        s_x: list[float] = []
        deltas: list[float] = []
        s_pl: list[float] = []
        n_coherent = 0
        n_cap = 0
        for row in rows:
            n_cap += int(bool(row.get("cap_hit", False)))
            key = (pair_id, slot, arm, int(row["draw"]))
            c = coh_grid.get(key)
            if c is None:
                reject["grid_no_coherence_score"] += 1
                continue
            if c <= COHERENCE_THRESHOLD:
                reject["grid_incoherent"] += 1
                continue
            n_coherent += 1
            sx = t_scores.get(key)
            sp = plain_grid.get(key)
            if sx is None:
                reject["grid_missing_target_score"] += 1
                continue
            s_x.append(sx / 100.0)
            if sp is None:
                reject["grid_missing_plain_score"] += 1
            else:
                s_pl.append(sp / 100.0)
                deltas.append((sx - sp) / 100.0)
        nm = norm.get((p.persona, p.carrier), {})
        s_mean = _mean(s_x)
        d_mean = _mean(deltas)
        f_target = _f_from_stats(p.kind, s_mean, nm.get("floor_target"), nm.get("ceil_target"))
        f_netted = _f_from_stats(p.kind, d_mean, nm.get("floor_delta"), nm.get("ceil_delta"))

        # F_act (inherited convention: f_act of the MEAN patched V_a over the
        # store's non-empty draws; anchor pools = pair.a floor / pair.b ceiling).
        block_key = rows[0]["block_key"]
        va_patched = [
            va_grid[k]
            for row in rows
            if (k := (block_key, pair_id, row["context_a"], int(row["draw"]))) in va_grid
        ]
        floor_va = [
            va_anchor[(p.a, d)]
            for d in sorted(set(anchor_draws_by_ctx.get(p.a, [])))
            if (p.a, d) in va_anchor
        ]
        ceil_va = [
            va_anchor[(p.b, d)]
            for d in sorted(set(anchor_draws_by_ctx.get(p.b, [])))
            if (p.b, d) in va_anchor
        ]
        f_act_fields: dict[str, float | bool | None] = {
            "f_act": None,
            "f_act_shared": None,
            "f_act_s_norm": None,
            "f_act_t_norm": None,
            "f_act_traversal_ratio": None,
            "f_act_degenerate": None,
        }
        if va_patched and len(floor_va) >= 2 and ceil_va:
            res = FM.f_act(
                torch.stack(va_patched).mean(dim=0),
                torch.stack(floor_va),
                torch.stack(ceil_va),
            )
            fa = float(res.f_act)
            f_act_fields = {
                "f_act": None if math.isnan(fa) else fa,
                "f_act_shared": float(res.f_act_shared),
                "f_act_s_norm": float(res.s_norm),
                "f_act_t_norm": float(res.t_norm),
                "f_act_traversal_ratio": float(res.traversal_ratio),
                "f_act_degenerate": bool(res.degenerate),
            }
        else:
            reject["cell_missing_f_act_inputs"] += 1

        v = LB.VALUES_BY_ID[p.persona]
        tables[arm].append(
            {
                "pair_id": pair_id,
                "direction": p.cell,
                "kind": p.kind,
                "rung": p.persona,
                "rung_label": v.rung,
                "rung_rank": v.rung_rank,
                "carrier": p.carrier,
                "slot": slot,
                "arm": arm,
                "family": f"{p.kind}-{slot}",
                "donor_context_id": rows[0].get("donor_context_id"),
                "n_draws": len(rows),
                "n_coherent": n_coherent,
                "n_scored_target": len(s_x),
                "n_scored_netted": len(deltas),
                "n_cap_hit": n_cap,
                "s_target_mean": s_mean,
                "s_plain_mean": _mean(s_pl),
                "delta_patched_mean": d_mean,
                "floor_target": nm.get("floor_target"),
                "ceil_target": nm.get("ceil_target"),
                "floor_delta": nm.get("floor_delta"),
                "ceil_delta": nm.get("ceil_delta"),
                "floor_plain": nm.get("floor_plain"),
                "ceil_plain": nm.get("ceil_plain"),
                "f_target": f_target,
                "f_netted": f_netted,
                "n_va_draws": len(va_patched),
                "n_floor_va": len(floor_va),
                "n_ceil_va": len(ceil_va),
                "len_delta": rows[0].get("len_delta"),
                **f_act_fields,
            }
        )

    for arm in ARMS:
        assert tables[arm], f"empty {arm} table — reject counters: {dict(reject)}"
    expected = registered_row_keys(gate, screen)
    present = {
        (r["direction"], r["slot"], r["arm"], r["carrier"]) for arm in ARMS for r in tables[arm]
    }
    set_check_registered_rows(expected, present, "f-tables")

    A62._write_jsonl_atomic(args.out_dir / "f_cells.jsonl", tables["steered"])
    A62._write_jsonl_atomic(args.out_dir / "null_samevalue_cells.jsonl", tables["null_sameval"])
    A62._write_jsonl_atomic(args.out_dir / "null_crosstype_cells.jsonl", tables["null_xtype"])
    logger.info(
        "[f-tables] steered=%d sameval=%d xtype=%d anchors=%d rejects=%s",
        len(tables["steered"]),
        len(tables["null_sameval"]),
        len(tables["null_xtype"]),
        len(anchors_out),
        dict(reject),
    )

    # Per-conjunct diagnostic (R1/R2 steered only; Round-A instrument).
    conj_out: list[dict] = []
    for persona, conjuncts in LJ.LADDER_CONJUNCTS.items():
        for key in conjuncts:
            rid = f"conj-{persona}-{key}"
            # A gate-dropped R1/R2 rung generates no conjunct wave — tolerated.
            cscores = load_grid_scores(args.scores_dir, rid, required=False)
            per_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
            for (pid, slot, arm, draw), score in cscores.items():
                if arm != "steered":
                    continue
                ckey = (pid, slot, arm, draw)
                c = coh_grid.get(ckey)
                if c is None or c <= COHERENCE_THRESHOLD:
                    continue
                per_cell[(pid, slot)].append(score)
            for (pid, slot), vals in sorted(per_cell.items()):
                p = pairs[pid]
                conj_out.append(
                    {
                        "pair_id": pid,
                        "direction": p.cell,
                        "kind": p.kind,
                        "rung": p.persona,
                        "carrier": p.carrier,
                        "slot": slot,
                        "conjunct": key,
                        "n_scored": len(vals),
                        "mean_score": _mean(vals),
                    }
                )
    A62._write_jsonl_atomic(args.out_dir / "conjuncts.jsonl", conj_out)
    logger.info("[f-tables] conjunct rows=%d", len(conj_out))


# ── step: margin ──────────────────────────────────────────────────────


def step_margin(args: argparse.Namespace) -> None:
    pairs = {p.pair_id: p for p in LB.build_ladder_pairs(LB.SEED)}
    grid_shards = sorted(args.margin_dir.glob("shard_*.jsonl"))
    anchor_shards = sorted(args.margin_dir.glob("anchor_margin_*.jsonl"))
    assert grid_shards, f"no grid margin shards under {args.margin_dir}"
    assert anchor_shards, f"no anchor margin shards under {args.margin_dir}"

    def _margin(rows: list[dict]) -> tuple[float | None, int]:
        """Side-B pool mean lnP minus side-A pool mean lnP (target minus source)."""
        a = [r["lnp_mean"] for r in rows if r.get("pool_side") == "A"]
        b = [r["lnp_mean"] for r in rows if r.get("pool_side") == "B"]
        if not a or not b:
            return None, len(a) + len(b)
        return (sum(b) / len(b)) - (sum(a) / len(a)), len(a) + len(b)

    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    skip_reasons: dict[tuple[str, str, str], str] = {}
    for shard in grid_shards:
        for r in J94._iter_jsonl(shard):
            if r.get("skipped"):
                key = (r["pair_id"], r.get("slot", "?"), r["arm"])
                skip_reasons[key] = r.get("reason", "skipped")
                continue
            by_cell[(r["pair_id"], r["slot"], r["arm"])].append(r)

    anchor_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for shard in anchor_shards:
        for r in J94._iter_jsonl(shard):
            if r.get("skipped"):
                continue
            anchor_by_key[(r["context_id"], r["pool_key"])].append(r)

    out: list[dict] = []
    for (pair_id, slot, arm), rows in sorted(by_cell.items()):
        p = pairs[pair_id]
        m, n_items = _margin(rows)
        am, _ = _margin(anchor_by_key.get((p.a, p.cell), []))
        out.append(
            {
                "pair_id": pair_id,
                "cell": p.cell,  # rule19_validation-compatible key
                "direction": p.cell,
                "kind": p.kind,
                "rung": p.persona,
                "carrier": p.carrier,
                "slot": slot,
                "arm": arm,
                "margin": m,
                "anchor_margin": am,
                "margin_shift": (m - am) if (m is not None and am is not None) else None,
                "n_pool_items": n_items,
                "skipped": False,
            }
        )
    for (pair_id, slot, arm), reason in sorted(skip_reasons.items()):
        if (pair_id, slot, arm) in by_cell:
            continue
        p = pairs[pair_id]
        out.append(
            {
                "pair_id": pair_id,
                "cell": p.cell,
                "direction": p.cell,
                "kind": p.kind,
                "rung": p.persona,
                "carrier": p.carrier,
                "slot": slot,
                "arm": arm,
                "margin": None,
                "anchor_margin": None,
                "margin_shift": None,
                "n_pool_items": 0,
                "skipped": True,
                "reason": reason,
            }
        )
    assert out, "margin step produced no rows"
    A62._write_jsonl_atomic(args.out_dir / "margin.jsonl", out)
    n_skip = sum(1 for r in out if r["skipped"])
    logger.info("[margin] cells=%d (skipped=%d)", len(out), n_skip)


# ── step: stats ───────────────────────────────────────────────────────


def _spearman_rows(x: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Row-wise Spearman rho between fixed x (n,) and each row of ys (B, n),
    average ranks on ties (matches scipy.stats.spearmanr)."""
    from scipy.stats import rankdata

    rx = rankdata(x)
    ry = rankdata(ys, axis=1)
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean(axis=1, keepdims=True)
    denom = np.sqrt((rx_c**2).sum() * (ry_c**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (ry_c @ rx_c) / denom, np.nan)


def trend_test(
    steered: list[dict],
    gate: dict,
    kind: str,
    slot: str,
    rng: np.random.Generator,
    n_perm: int,
) -> dict:
    """One within-carrier rung-label permutation trend test (plan §6).

    Statistic: Spearman rho between rung specificity rank and per-rung steered
    F_target (mean over surviving carriers). Null: rung labels permuted WITHIN
    carrier over that carrier's gate-surviving rungs, re-aggregated per draw.
    """
    rungs = [v for v in LB.PERSONA_VALUE_IDS if gate["rungs"][v].get("survived")]
    r_index = {v: i for i, v in enumerate(rungs)}
    ranks = np.array([LB.VALUES_BY_ID[v].rung_rank for v in rungs], dtype=np.float64)

    cell_val: dict[tuple[str, str], float] = {}
    for r in steered:
        if r["kind"] != kind or r["slot"] != slot or r["f_target"] is None:
            continue
        if r["rung"] in r_index:
            cell_val[(r["carrier"], r["rung"])] = float(r["f_target"])
    carriers = sorted({c for c, _ in cell_val})
    n_rungs = len(rungs)
    result: dict = {
        "family": f"{kind}-{slot}",
        "n_surviving_rungs": n_rungs,
        "surviving_rungs": rungs,
        "n_carriers": len(carriers),
        "descriptive_only": n_rungs < MIN_TREND_RUNGS,
        "rung_ranks": {v: LB.VALUES_BY_ID[v].rung_rank for v in rungs},
    }
    if not carriers or n_rungs < 2:
        result.update({"rho_observed": None, "per_rung_mean": {}, "p_one_sided": None})
        return result

    sums = np.zeros(n_rungs)
    counts = np.zeros(n_rungs)
    perm_sums = np.zeros((n_perm, n_rungs))
    for carrier in carriers:
        pres = [v for v in rungs if (carrier, v) in cell_val]
        vals = np.array([cell_val[(carrier, v)] for v in pres], dtype=np.float64)
        idxs = np.array([r_index[v] for v in pres], dtype=np.intp)
        sums[idxs] += vals
        counts[idxs] += 1.0
        # B within-carrier permutations of this carrier's surviving rung labels.
        perms = np.argsort(rng.random((n_perm, len(pres))), axis=1)
        permuted = vals[perms]  # (B, k)
        for j in range(len(pres)):
            perm_sums[:, idxs[j]] += permuted[:, j]
    with np.errstate(invalid="ignore", divide="ignore"):
        per_rung = np.where(counts > 0, sums / counts, np.nan)
        perm_means = np.where(counts > 0, perm_sums / counts, np.nan)
    keep = counts > 0
    result["per_rung_mean"] = {
        v: (None if not keep[i] else float(per_rung[i])) for v, i in r_index.items()
    }
    if keep.sum() < 2:
        result.update({"rho_observed": None, "p_one_sided": None})
        return result
    rho_obs = float(_spearman_rows(ranks[keep], per_rung[None, keep])[0])
    rho_perm = _spearman_rows(ranks[keep], perm_means[:, keep])
    rho_perm = rho_perm[~np.isnan(rho_perm)]
    n_eff = len(rho_perm)
    p_pos = (1 + int((rho_perm >= rho_obs).sum())) / (n_eff + 1)
    p_two = (1 + int((np.abs(rho_perm) >= abs(rho_obs)).sum())) / (n_eff + 1)
    result.update(
        {
            "rho_observed": rho_obs,
            "p_one_sided": float(p_pos),
            "p_two_sided": float(p_two),
            "n_permutations_effective": n_eff,
        }
    )
    return result


def _ci_from_boot(boot_col: np.ndarray) -> tuple[float | None, float | None]:
    finite = boot_col[np.isfinite(boot_col)]
    if finite.size == 0:
        return None, None
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def step_stats(args: argparse.Namespace) -> None:
    gate = _read_gate(args.gates_dir)
    screen = _read_donor_screen(args.gates_dir)
    steered = list(A62._iter_jsonl(args.out_dir / "f_cells.jsonl"))
    nulls = {
        "null_sameval": list(A62._iter_jsonl(args.out_dir / "null_samevalue_cells.jsonl")),
        "null_xtype": list(A62._iter_jsonl(args.out_dir / "null_crosstype_cells.jsonl")),
    }
    all_rows = steered + nulls["null_sameval"] + nulls["null_xtype"]
    assert all_rows, "stats: no cell rows on disk — run --step f-tables first"
    expected = registered_row_keys(gate, screen)
    present = {(r["direction"], r["slot"], r["arm"], r["carrier"]) for r in all_rows}
    set_check_registered_rows(expected, present, "stats")

    # ── confirmatory: 4 Holm-corrected trend permutation tests ──
    rng = np.random.default_rng(TREND_SEED)
    trend: dict[str, dict] = {}
    for fam in FAMILIES:
        kind, slot = fam.split("-")
        trend[fam] = trend_test(steered, gate, kind, slot, rng, args.n_perm)
    testable = {
        f: t["p_one_sided"]
        for f, t in trend.items()
        if not t["descriptive_only"] and t["p_one_sided"] is not None
    }
    if testable:
        adj = A62.holm(testable)
        for f, p in adj.items():
            trend[f]["p_holm"] = p
            trend[f]["rejects_at_005"] = bool(p < 0.05 and (trend[f]["rho_observed"] or 0) > 0)
    for f in trend:
        trend[f].setdefault("p_holm", None)
        trend[f].setdefault("rejects_at_005", None)
    # Holm family size is m=4 by registration; note when fewer were testable.
    trend_meta = {"holm_m_registered": HOLM_M, "n_testable": len(testable)}

    # ── estimation: carrier-clustered bootstrap per (direction x slot x arm) ──
    carriers = sorted({r["carrier"] for r in all_rows})
    c_index = {c: i for i, c in enumerate(carriers)}
    fams: list[tuple[str, str, str]] = sorted(
        {(r["direction"], r["slot"], r["arm"]) for r in all_rows}
    )
    f_index = {k: j for j, k in enumerate(fams)}
    values = np.full((len(carriers), len(fams)), np.nan)
    for r in all_rows:
        if r["f_target"] is None:
            continue
        values[c_index[r["carrier"]], f_index[(r["direction"], r["slot"], r["arm"])]] = r[
            "f_target"
        ]
    boot = bootstrap_family_means_batched(values, n_boot=args.n_boot, seed=BOOT_SEED)
    estimation: dict[str, dict] = {}
    for (direction, slot, arm), j in f_index.items():
        col = values[:, j]
        n_c = int(np.isfinite(col).sum())
        lo, hi = _ci_from_boot(boot[:, j])
        estimation["|".join((direction, slot, arm))] = {
            "direction": direction,
            "slot": slot,
            "arm": arm,
            "mean_f_target": (float(np.nanmean(col)) if n_c else None),
            "ci_lo": lo,
            "ci_hi": hi,
            "n_carriers": n_c,
        }

    # ── lattice + H3 null sanity (per direction x slot) ──
    lattice: dict[str, dict] = {}
    for v in LB.PERSONA_VALUE_IDS:
        survived = bool(gate["rungs"][v].get("survived"))
        for kind in ("install", "erase"):
            direction = f"{kind}_{v}"
            for slot in SLOTS:
                key = f"{direction}|{slot}"
                if not survived:
                    lattice[key] = {
                        "direction": direction,
                        "slot": slot,
                        "verdict": "untestable",
                        "reason": "rung failed the anchor-separation gate",
                    }
                    continue
                st = estimation.get(f"{direction}|{slot}|steered")
                sv = estimation.get(f"{direction}|{slot}|null_sameval")
                xt = estimation.get(f"{direction}|{slot}|null_xtype")
                null_means = {
                    "null_sameval": (sv or {}).get("mean_f_target"),
                    "null_xtype": (xt or {}).get("mean_f_target"),
                }
                null_flag = kind == "install" and any(
                    m is not None and m > NULL_SANITY_BAR for m in null_means.values()
                )
                ci_ok = (
                    st is not None
                    and st["ci_lo"] is not None
                    and all(
                        n is not None and n["ci_hi"] is not None and st["ci_lo"] > n["ci_hi"]
                        for n in (sv, xt)
                    )
                )
                verdict = "transfers" if (ci_ok and not null_flag) else "no-clean-transfer"
                lattice[key] = {
                    "direction": direction,
                    "slot": slot,
                    "verdict": verdict,
                    "steered_ci_above_both_nulls": bool(ci_ok),
                    "null_sanity_flag": bool(null_flag),
                    "transfers_withheld_by_null_sanity": bool(ci_ok and null_flag),
                    "steered_mean": (st or {}).get("mean_f_target"),
                    "null_means": null_means,
                }

    # ── H4: paired erase - install per (rung x slot) ──
    h4_fams: list[tuple[str, str]] = []
    for v in LB.PERSONA_VALUE_IDS:
        if gate["rungs"][v].get("survived"):
            h4_fams.extend((v, slot) for slot in SLOTS)
    st_val = {
        (r["direction"], r["slot"], r["carrier"]): r["f_target"]
        for r in steered
        if r["f_target"] is not None
    }
    h4: dict[str, dict] = {}
    if h4_fams:
        diff = np.full((len(carriers), len(h4_fams)), np.nan)
        for j, (v, slot) in enumerate(h4_fams):
            for c, i in c_index.items():
                e = st_val.get((f"erase_{v}", slot, c))
                ins = st_val.get((f"install_{v}", slot, c))
                if e is not None and ins is not None:
                    diff[i, j] = e - ins
        boot_h4 = bootstrap_family_means_batched(diff, n_boot=args.n_boot, seed=BOOT_SEED)
        for j, (v, slot) in enumerate(h4_fams):
            col = diff[:, j]
            n_c = int(np.isfinite(col).sum())
            lo, hi = _ci_from_boot(boot_h4[:, j])
            h4[f"{v}|{slot}"] = {
                "rung": v,
                "slot": slot,
                "mean_erase_minus_install": (float(np.nanmean(col)) if n_c else None),
                "ci_lo": lo,
                "ci_hi": hi,
                "n_carriers_paired": n_c,
                "per_carrier": {
                    c: (None if not np.isfinite(col[i]) else float(col[i]))
                    for c, i in c_index.items()
                },
            }

    # ── margin validation (rule 19; steered cells) ──
    margin_path = args.out_dir / "margin.jsonl"
    margin_validation: dict | None = None
    if margin_path.exists():
        margin_rows = list(A62._iter_jsonl(margin_path))
        f_by_key = {
            (r["pair_id"], r["slot"]): r["f_target"] for r in steered if r["f_target"] is not None
        }
        margin_validation = A62.rule19_validation(margin_rows, f_by_key)
    else:
        logger.warning("[stats] %s absent — margin validation skipped", margin_path)

    # ── coherence / cap-hit summaries (cells < 50%% coherent marked) ──
    low_coherence = [
        {
            "pair_id": r["pair_id"],
            "slot": r["slot"],
            "arm": r["arm"],
            "coherent_fraction": r["n_coherent"] / r["n_draws"] if r["n_draws"] else None,
        }
        for r in all_rows
        if r["n_draws"] and (r["n_coherent"] / r["n_draws"]) < 0.5
    ]
    cap_total = sum(r["n_cap_hit"] for r in all_rows)
    draw_total = sum(r["n_draws"] for r in all_rows)

    stats = {
        "round": "persona-specificity-ladder",
        "seeds": {"trend_permutation": TREND_SEED, "bootstrap": BOOT_SEED},
        "n_boot": args.n_boot,
        "n_perm": args.n_perm,
        "trend_tests": trend,
        "trend_meta": trend_meta,
        "estimation": estimation,
        "lattice": lattice,
        "h4_asymmetry": h4,
        "margin_validation": margin_validation,
        "coherence": {
            "cells_below_50pct_coherent": low_coherence,
            "n_cells": len(all_rows),
        },
        "cap_hit": {"n_cap_hit_draws": cap_total, "n_draws": draw_total},
        "repro": J94._repro(),
    }
    A62._write_json_atomic(args.out_dir / "stats.json", stats)
    logger.info(
        "[stats] trend families=%d estimation families=%d lattice=%d h4=%d",
        len(trend),
        len(estimation),
        len(lattice),
        len(h4),
    )


# ── step: figures ─────────────────────────────────────────────────────


def _err(mean: float, lo: float | None, hi: float | None) -> tuple[float, float]:
    """Non-negative errorbar OFFSETS from CI bounds (matplotlib contract)."""
    if lo is None or hi is None:
        return 0.0, 0.0
    return max(0.0, mean - lo), max(0.0, hi - mean)


def _rung_axis(gate: dict) -> list[str]:
    return list(LB.PERSONA_VALUE_IDS)  # bank order: R1 -> R5b


def _sys_token_counts(skip: bool) -> dict[str, int | None]:
    if skip:
        return {v: None for v in LB.PERSONA_VALUE_IDS}
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    return {
        v: len(tok(LB.VALUES_BY_ID[v].system_text, add_special_tokens=False)["input_ids"])
        for v in LB.PERSONA_VALUE_IDS
    }


def _save(fig, stem: str, figures_dir: Path) -> None:
    from explore_persona_space.analysis.paper_plots import savefig_paper

    savefig_paper(fig, stem, dir=figures_dir)
    import matplotlib.pyplot as plt

    plt.close(fig)


def fig_ladder_hero(stats: dict, gate: dict, args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    tok_counts = _sys_token_counts(args.skip_token_counts)
    rungs = _rung_axis(gate)
    est = stats["estimation"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5), sharey=True)
    for row, slot in enumerate(SLOTS):
        for col, kind in enumerate(("install", "erase")):
            ax = axes[row][col]
            for k, arm in enumerate(ARMS):
                xs, ys, elo, ehi = [], [], [], []
                for i, v in enumerate(rungs):
                    rec = est.get(f"{kind}_{v}|{slot}|{arm}")
                    if rec is None or rec["mean_f_target"] is None:
                        continue
                    xs.append(i + (k - 1) * 0.18)
                    ys.append(rec["mean_f_target"])
                    lo, hi = _err(rec["mean_f_target"], rec["ci_lo"], rec["ci_hi"])
                    elo.append(lo)
                    ehi.append(hi)
                if xs:
                    ax.errorbar(
                        xs,
                        ys,
                        yerr=[elo, ehi],
                        fmt="o",
                        color=ARM_COLORS[arm],
                        label=ARM_LABELS[arm] if (row == 0 and col == 0) else None,
                        capsize=3,
                        markersize=5,
                        linestyle="none",
                    )
            ax.axhline(0.0, color="grey", lw=0.8, ls=":")
            ax.set_title(f"{kind} — {'context-end' if slot == 'ce' else 'prefix-end'}")
            ax.set_xticks(range(len(rungs)))
            labels = []
            for v in rungs:
                rec = est.get(f"{kind}_{v}|{slot}|steered")
                n = rec["n_carriers"] if rec else 0
                tok = tok_counts.get(v)
                extra = f"\nn={n}" + (f", {tok} tok" if tok is not None else "")
                surv = "" if gate["rungs"][v].get("survived") else "\n(gate-failed)"
                labels.append(f"{LB.VALUES_BY_ID[v].rung} {v.split('_', 1)[-1]}{extra}{surv}")
            ax.set_xticklabels(labels, fontsize=7)
            if col == 0:
                ax.set_ylabel("F_target (fraction of full context swap)")
    axes[0][0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Specificity ladder: steered vs both nulls per rung")
    fig.tight_layout()
    _save(fig, "ladder_hero", args.figures_dir)


def fig_ladder_percarrier(cells: dict[str, list[dict]], gate: dict, args) -> None:
    import matplotlib.pyplot as plt

    rungs = _rung_axis(gate)
    r_pos = {v: i for i, v in enumerate(rungs)}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5), sharey=True)
    for row, slot in enumerate(SLOTS):
        for col, kind in enumerate(("install", "erase")):
            ax = axes[row][col]
            for k, arm in enumerate(ARMS):
                for r in cells[arm]:
                    if r["kind"] != kind or r["slot"] != slot or r["f_target"] is None:
                        continue
                    x = r_pos[r["rung"]] + (k - 1) * 0.18
                    ax.scatter([x], [r["f_target"]], s=14, color=ARM_COLORS[arm], alpha=0.75)
                    ax.annotate(
                        r["carrier"],
                        (x, r["f_target"]),
                        fontsize=5,
                        xytext=(2, 1),
                        textcoords="offset points",
                    )
            ax.axhline(0.0, color="grey", lw=0.8, ls=":")
            ax.set_title(f"{kind} — {'context-end' if slot == 'ce' else 'prefix-end'}")
            ax.set_xticks(range(len(rungs)))
            ax.set_xticklabels([LB.VALUES_BY_ID[v].rung for v in rungs], fontsize=8)
            if col == 0:
                ax.set_ylabel("per-carrier F_target")
    handles = [
        plt.Line2D([], [], marker="o", ls="none", color=ARM_COLORS[a], label=ARM_LABELS[a])
        for a in ARMS
    ]
    axes[0][0].legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("Specificity ladder — per-carrier companion (no aggregation)")
    fig.tight_layout()
    _save(fig, "ladder_percarrier", args.figures_dir)


def fig_asymmetry(stats: dict, gate: dict, args) -> None:
    import matplotlib.pyplot as plt

    h4 = stats["h4_asymmetry"]
    rungs = [v for v in _rung_axis(gate) if gate["rungs"][v].get("survived")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for col, slot in enumerate(SLOTS):
        ax = axes[col]
        for i, v in enumerate(rungs):
            rec = h4.get(f"{v}|{slot}")
            if rec is None:
                continue
            pts = [x for x in rec["per_carrier"].values() if x is not None]
            ax.scatter([i] * len(pts), pts, s=14, color="#7f7f7f", alpha=0.7)
            m = rec["mean_erase_minus_install"]
            if m is not None:
                lo, hi = _err(m, rec["ci_lo"], rec["ci_hi"])
                ax.errorbar([i + 0.15], [m], yerr=[[lo], [hi]], fmt="D", color="#d62728", capsize=3)
        ax.axhline(0.0, color="grey", lw=0.8, ls=":")
        ax.set_title(f"{'context-end' if slot == 'ce' else 'prefix-end'}")
        ax.set_xticks(range(len(rungs)))
        ax.set_xticklabels([LB.VALUES_BY_ID[v].rung for v in rungs], fontsize=8)
        if col == 0:
            ax.set_ylabel("erase F_target − install F_target (paired by carrier)")
    fig.suptitle("Erase vs install asymmetry per rung (steered arm)")
    fig.tight_layout()
    _save(fig, "asymmetry", args.figures_dir)


def fig_anchor_separation(anchors: list[dict], gate: dict, args) -> None:
    import matplotlib.pyplot as plt

    rungs = _rung_axis(gate)
    r_pos = {v: i for i, v in enumerate(rungs)}
    bars = gate.get("bars", {})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    specs = [
        ("gate_target_sep", bars.get("target_sep_bar", 0.25), "target-descriptor separation"),
        ("gate_netted_sep", bars.get("netted_sep_bar", 0.5), "netted dual-rubric separation"),
    ]
    for ax, (field, bar, label) in zip(axes, specs, strict=True):
        for r in anchors:
            val = r[field]
            if val is None:
                continue
            x = r_pos[r["rung"]] + (hash(r["carrier"]) % 7 - 3) * 0.03
            filled = r["gate_passed"]
            ax.scatter(
                [x],
                [val],
                s=22,
                facecolors="#1f77b4" if filled else "none",
                edgecolors="#1f77b4",
                # Explicit edge width: the blog paper style zeroes marker edge
                # widths, which renders the unfilled (gate-failed) markers
                # invisible without this (13/36 carriers dropped from view).
                linewidths=1.0,
            )
        ax.axhline(bar, color="#d62728", lw=1.0, ls="--")
        ax.set_xticks(range(len(rungs)))
        ax.set_xticklabels(
            [
                f"{LB.VALUES_BY_ID[v].rung}\n"
                + ("survived" if gate["rungs"][v].get("survived") else "dropped")
                for v in rungs
            ],
            fontsize=8,
        )
        ax.set_ylabel(label)
        ax.set_title(f"{label} (bar = {bar})")
    fig.suptitle("Anchor separations and gate verdicts per rung (filled = carrier passed)")
    fig.tight_layout()
    _save(fig, "anchor_separation", args.figures_dir)


def fig_rubric_bridge(cells: dict[str, list[dict]], args) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6))
    agg: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for arm in ARMS:
        for r in cells[arm]:
            if r["f_target"] is None or r["f_netted"] is None:
                continue
            agg[(r["direction"], r["slot"], arm)].append((r["f_netted"], r["f_target"]))
    for (_, _, arm), pts in sorted(agg.items()):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(sum(xs) / len(xs), sum(ys) / len(ys), s=30, color=ARM_COLORS[arm], alpha=0.85)
    lims = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lims), max(lims)
    ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, ls=":")
    handles = [
        plt.Line2D([], [], marker="o", ls="none", color=ARM_COLORS[a], label=ARM_LABELS[a])
        for a in ARMS
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel("netted dual-rubric F (parent metric)")
    ax.set_ylabel("F_target (this round's primary)")
    ax.set_title("Netted dual-rubric F vs target-only F per (direction × slot × arm)")
    fig.tight_layout()
    _save(fig, "rubric_bridge", args.figures_dir)


def fig_dv_agreement(cells: dict[str, list[dict]], stats: dict, args) -> None:
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # Panel 1: F_act vs F_target per cell.
    xs, ys = [], []
    for arm in ARMS:
        axs, ays = [], []
        for r in cells[arm]:
            if r["f_target"] is None or r["f_act"] is None:
                continue
            axs.append(r["f_target"])
            ays.append(r["f_act"])
        axes[0].scatter(axs, ays, s=14, color=ARM_COLORS[arm], alpha=0.7)
        xs += axs
        ys += ays
    rho_txt = "n/a"
    if len(xs) >= 3:
        rho = spearmanr(xs, ys).statistic
        rho_txt = "n/a" if (rho != rho) else f"{rho:+.2f}"
    axes[0].set_xlabel("F_target")
    axes[0].set_ylabel("F_act (pair-own contrast projection)")
    axes[0].set_title(f"F_act vs F_target (Spearman ρ = {rho_txt})")
    # Panel 2: margin validation (registered per-cell grain).
    mv = stats.get("margin_validation") or {}
    pts = mv.get("percell_points") or []
    axes[1].scatter(
        [p["margin_shift_mean"] for p in pts],
        [p["f_beh_mean"] for p in pts],
        s=16,
        color="#1f77b4",
    )
    rho_cell = mv.get("rho_margin_fbeh_percell")
    axes[1].set_xlabel("TF margin shift (nats/token)")
    axes[1].set_ylabel("F_target (per-cell mean)")
    axes[1].set_title(
        "margin validation ρ = "
        + (f"{rho_cell:+.2f}" if rho_cell is not None else f"n/a (n={mv.get('n_cells', 0)})")
    )
    # Panel 3: coherence rate vs cap-hit fraction per cell.
    for arm in ARMS:
        cr = [r["n_coherent"] / r["n_draws"] for r in cells[arm] if r["n_draws"]]
        ch = [r["n_cap_hit"] / r["n_draws"] for r in cells[arm] if r["n_draws"]]
        axes[2].scatter(cr, ch, s=14, color=ARM_COLORS[arm], alpha=0.7)
    axes[2].set_xlabel("coherence rate per cell")
    axes[2].set_ylabel("cap-hit fraction per cell")
    axes[2].set_title("coherence / cap-hit per cell")
    fig.suptitle("Continuous-companion agreement")
    fig.tight_layout()
    _save(fig, "dv_agreement", args.figures_dir)


def fig_conjunct_diag(conjuncts: list[dict], steered: list[dict], args) -> None:
    import matplotlib.pyplot as plt

    personas = sorted({r["rung"] for r in conjuncts})
    if not personas:
        logger.warning("[figures] no conjunct rows — conjunct_diag skipped")
        return
    fig, axes = plt.subplots(len(personas), 1, figsize=(11, 4 * len(personas)), squeeze=False)
    hol = {
        (r["direction"], r["slot"]): [
            x["s_target_mean"] * 100
            for x in steered
            if x["direction"] == r["direction"]
            and x["slot"] == r["slot"]
            and x["s_target_mean"] is not None
        ]
        for r in conjuncts
    }
    for pi, persona in enumerate(personas):
        ax = axes[pi][0]
        rows = [r for r in conjuncts if r["rung"] == persona]
        keys = sorted({r["conjunct"] for r in rows}) + ["holistic"]
        groups = sorted({(r["direction"], r["slot"]) for r in rows})
        width = 0.8 / max(1, len(groups))
        for gi, (direction, slot) in enumerate(groups):
            vals = []
            for key in keys:
                if key == "holistic":
                    hv = hol.get((direction, slot), [])
                    vals.append(sum(hv) / len(hv) if hv else 0.0)
                else:
                    sel = [
                        r["mean_score"]
                        for r in rows
                        if r["conjunct"] == key
                        and r["direction"] == direction
                        and r["slot"] == slot
                        and r["mean_score"] is not None
                    ]
                    vals.append(sum(sel) / len(sel) if sel else 0.0)
            ax.bar(
                [i + gi * width for i in range(len(keys))],
                vals,
                width=width,
                label=f"{direction} {slot}",
            )
        ax.set_xticks([i + 0.4 for i in range(len(keys))])
        ax.set_xticklabels(keys, fontsize=8)
        ax.set_ylabel("mean judge score (0-100)")
        ax.set_title(f"{persona} steered per-conjunct decomposition")
        ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "conjunct_diag", args.figures_dir)


def step_figures(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()
    gate = _read_gate(args.gates_dir)
    stats = json.loads((args.out_dir / "stats.json").read_text(encoding="utf-8"))
    cells = {
        "steered": list(A62._iter_jsonl(args.out_dir / "f_cells.jsonl")),
        "null_sameval": list(A62._iter_jsonl(args.out_dir / "null_samevalue_cells.jsonl")),
        "null_xtype": list(A62._iter_jsonl(args.out_dir / "null_crosstype_cells.jsonl")),
    }
    anchors = list(A62._iter_jsonl(args.out_dir / "anchors.jsonl"))
    conjuncts = list(A62._iter_jsonl(args.out_dir / "conjuncts.jsonl"))
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    fig_ladder_hero(stats, gate, args)
    fig_ladder_percarrier(cells, gate, args)
    fig_asymmetry(stats, gate, args)
    fig_anchor_separation(anchors, gate, args)
    fig_rubric_bridge(cells, args)
    fig_dv_agreement(cells, stats, args)
    fig_conjunct_diag(conjuncts, cells["steered"], args)
    logger.info("[figures] wrote 7 figures to %s", args.figures_dir)


# ── staging / CLI ─────────────────────────────────────────────────────

STEPS = {
    "f-tables": step_f_tables,
    "margin": step_margin,
    "stats": step_stats,
    "figures": step_figures,
}
STEP_ORDER = ("f-tables", "margin", "stats", "figures")

_STAGE_PREFIXES = (
    f"{LJ.LADDER_RAW}/grid",
    f"{LJ.LADDER_RAW}/anchors",
    f"{LJ.LADDER_TENSORS}/anchors",  # anchor V_a shards (split from the raw text on upload)
    f"{LJ.LADDER_TENSORS}/va_store",
    f"{LJ.LADDER_TENSORS}/margin",
)


def _stage_inputs(args: argparse.Namespace) -> None:
    from explore_persona_space.orchestrate import hub

    for prefix in _STAGE_PREFIXES:
        staged = hub.stage_hub_prefix(
            LJ.DATASET_REPO, prefix, args.in_root, revision=args.hf_revision
        )
        logger.info("[stage] %s: %d files", prefix, len(staged))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Issue #2162 persona-specificity-ladder analysis driver (P6)."
    )
    ap.add_argument("--step", default="all", choices=("all", *STEPS))
    ap.add_argument(
        "--in-root",
        type=Path,
        default=Path("data/issue_2162/ladder_judge_inputs"),
        help="staging mirror root shared with the judge driver",
    )
    ap.add_argument("--rollouts-dir", type=Path, default=None)
    ap.add_argument("--anchors-dir", type=Path, default=None)
    ap.add_argument(
        "--anchor-va-dir",
        type=Path,
        default=None,
        help="anchor V_a tensor shards (HF splits them into analysis_tensors, "
        "away from the raw anchor text rows)",
    )
    ap.add_argument("--va-dir", type=Path, default=None)
    ap.add_argument("--margin-dir", type=Path, default=None)
    ap.add_argument(
        "--work-root",
        type=Path,
        default=Path("eval_results/issue_2162/persona_specificity_ladder/judge"),
        help="the VM judge driver's work root (scores/ + gates/)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eval_results/issue_2162/persona_specificity_ladder"),
    )
    ap.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures/issue_2162/persona_specificity_ladder"),
    )
    ap.add_argument("--stage-from-hf", action="store_true")
    ap.add_argument("--hf-revision", type=str, default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument(
        "--skip-token-counts",
        action="store_true",
        help="skip the hero figure's system-prompt token-count annotation "
        "(avoids the tokenizer load; smoke/tests)",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="execute deferred imports + argparse-attribute completeness, then exit 0",
    )
    return ap.parse_args(argv)


def _resolve_dirs(args: argparse.Namespace) -> argparse.Namespace:
    mirror_raw = args.in_root / LJ.HF_PREFIX / "raw_completions"
    mirror_tensors = args.in_root / LJ.HF_PREFIX / "analysis_tensors"
    if args.rollouts_dir is None:
        args.rollouts_dir = mirror_raw / "ladder/grid"
    if args.anchors_dir is None:
        args.anchors_dir = mirror_raw / "ladder/anchors"
    if args.anchor_va_dir is None:
        args.anchor_va_dir = mirror_tensors / "ladder/anchors"
    if args.va_dir is None:
        args.va_dir = mirror_tensors / "ladder/va_store"
    if args.margin_dir is None:
        args.margin_dir = mirror_tensors / "ladder/margin"
    args.scores_dir = args.work_root / "scores"
    args.gates_dir = args.work_root / "gates"
    return args


def _import_check() -> int:
    """Deferred-import + argparse-attribute completeness check (no inputs read)."""
    import matplotlib  # noqa: F401  (figures step)
    from scipy.stats import rankdata, spearmanr  # noqa: F401
    from transformers import AutoTokenizer  # noqa: F401  (token-count annotation)

    from explore_persona_space.analysis.paper_plots import (  # noqa: F401
        savefig_paper,
        set_paper_style,
    )
    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert callable(hub.stage_hub_prefix)
    assert callable(bootstrap_family_means_batched)
    assert callable(A62.rule19_validation) and callable(A62.holm)
    assert callable(FM.f_act)
    assert_args_attributes_defined(__file__)
    assert set(FAMILIES) == {f"{k}-{s}" for k in ("install", "erase") for s in SLOTS}
    print("[import-check] OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.import_check:
        return _import_check()
    if args.stage_from_hf:
        _stage_inputs(args)
    args = _resolve_dirs(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    steps = STEP_ORDER if args.step == "all" else (args.step,)
    for name in steps:
        logger.info("[phase=%s] start", name)
        STEPS[name](args)
        logger.info("[phase=%s_done]", name)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
