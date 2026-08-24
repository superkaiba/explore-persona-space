#!/usr/bin/env python3
"""#1739 widened r2v2 legs: extraction factorial + mapped-vs-real directions.

Three legs over the SAME prepared pipeline state as `issue1739_r2v2_score.py`
(`prepare_behavior`: tables, U-pool whitening, linear ADD map, merged fp64
table, dataset roster, committed frozen layers — map + whitening IDENTICAL
across protocols and regimes; only the direction r_B varies):

1. **Extraction factorial** (user scope extension, 2026-08-06): the three
   rb-DEPENDENT projection arms (arm1_ctx_e1 = context, arm6_map_proj_e1 =
   mapped answer, arm11_oracle_proj = real answer) scored under FIVE
   extraction regimes — e1, e1_fc, e2, e2p, e2p_fc — keyed as (arm, regime)
   cells (the `_e1` in the arm slugs is a legacy naming artifact; the regime
   rides a separate key, never a new arm slug). `e2_fc` is a STRUCTURAL N/A
   (within-context matched-pair hi/lo weights cancel exactly on context-level
   rows — the fits.py `_extract_rb` refusal); it is recorded as such, never a
   zero. The rb-INDEPENDENT ridge arms (arm4/arm7) are NOT re-run — their
   cells are banked in the base r2v2 leg (`r2v2_fits/<behavior>`).

   Scoring is by DIRECT projection of the whitened eval blocks onto each
   regime direction — mathematically IDENTICAL to `arms.run_cell_multi`'s
   arm1/6/11 dispatch (`_proj(z|mp|za, rb)`; apply_map is row-wise, so the
   eval-block slice of the comb-table path equals apply_map on the eval block
   alone). Estimator-parity reference: arms.py `run_cell_multi` arm1/6/11 +
   `_proj`; the in-run e1-vs-base check pins it against the base leg's rows.

2. **P-B fold-respecting natural re-extraction** (user directive "just do the
   re-extraction"): for every P-B holdout fold, e2/e2p/e2p_fc directions are
   extracted from THAT fold's readout pool (80% GROUP-level slice of every
   eliciting dataset except the holdout — the same `assemble_readout_pool`
   rows the ridge fitters saw; WildChat is the GENERIC pool and never enters
   extraction). e1/e1_fc come from the disjoint synthetic contrastive store
   and are fold-INVARIANT. Per-fold extraction-pool/eval disjointness is
   asserted (fail-loud), and per-(pool x regime) qualifying-context counts are
   recorded. Weights come verbatim from `fits.matched_pair_split_weights`
   (never a re-implementation); directions are computed by PER-SHARD
   mask-GEMMs streamed over the store's fp16 npy shards (one
   (n_specs, rows) @ (rows, d) product per (kind, layer, shard) — no
   per-rollout activation grid is ever resident).

3. **Mapped-vs-real direction comparison** (second user addendum): does the
   EXTRACTED DIRECTION survive the map? For a LINEAR map the mean-difference
   construction COMMUTES with map application: with hi/lo weights each
   summing to 1, the affine offsets (x_mu, x_sd standardization, y_mu) cancel
   in the weighted difference, so

       r_B^mapped[ly] = ((r_B^ctx_w[ly]) / x_sd[ly]) @ W[ly]

   where r_B^ctx_w is the whitened context_end-extracted direction (e1_fc /
   e2p_fc). The mapped-extracted direction is therefore DERIVED, not
   re-streamed. Reported per layer, in WHITENED space (the space the map
   lives in): cos + norm ratio of r_B^mapped vs r_B^real (the t1-extracted
   direction), with two mandatory baselines — identity (cos(r_B^ctx,
   r_B^real); the learned-bias identity form v = x + b collapses to plain
   identity for mean-difference directions because the bias cancels in the
   difference) and the shuffled-map floor (same construction through
   `fits.shuffled_map_weights`). e2's mapped cell is structurally N/A for the
   same weight-cancellation reason as e2_fc.

Per-rollout score sources (recorded per behavior in the output meta):
- evil / sycophancy: DV rows' `per_rollout_scores` (train DV + OOD DVs).
- hallucination: `issue1739_ctxmap/judge/hallucination/labeling_per_rollout.json`
  (three_way-derived, fabricated=100 / correct|abstained=0, gate-validated to
  reproduce the committed DV within 1e-12) — threaded EXPLICITLY, never a
  silent fallback; an in-run assert re-verifies dv equality on shared ctx.

Parity validation: re-extracted full-pool e1_fc / e2p_fc directions are
compared per layer against the banked `rb_fc_bank` npz vectors (raw space);
a mismatch is a pipeline bug, reported loud.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_r2v2_score.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

FACT_REGIMES = ("e1", "e1_fc", "e2", "e2p", "e2p_fc")
# arm -> which whitened eval block it projects (ctx | map | ans); the
# estimator-parity reference is arms.run_cell_multi's arm1/6/11 dispatch.
RB_ARMS = {
    "arm1_ctx_e1": "ctx",
    "arm6_map_proj_e1": "map",
    "arm11_oracle_proj": "ans",
}
NATURAL_BASES = ("e2", "e2p")  # matched-pair / pooled; weights from judged scores
STRUCTURAL_NA = {
    ("e2_fc", "direction"): "within-context matched-pair hi/lo weights cancel exactly on "
    "context-level rows (fits.py _extract_rb structural refusal)",
    ("e2", "mapped_direction"): "same cancellation: mapped-answer rows of one context are "
    "identical, so the within-context weighted difference is exactly zero",
}


def _log(msg: str) -> None:
    print(f"[r2v2-fact {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# extraction grain: per-store row_index + per-rollout scores
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class StoreGrain:
    """One capture store's per-rollout extraction grain.

    ``row_ctx``/``row_k`` cover EVERY store row (row_index order == shard
    concat order — the store_io contract); ``n_rows`` pins the shard-stream
    offset check. ``scores`` maps ctx_id -> {k: score} for rows with a kept
    judged per-rollout score.
    """

    name: str
    store_dir: Path
    row_ctx: list[str]
    row_k: object  # (n_rows,) int64
    n_rows: int
    scores: dict[str, dict[int, float]]
    per_rollout_source: str


def _per_rollout_from_dv(dv_path: Path, *, source_label: str) -> dict[str, dict[int, float]]:
    """DV json rows -> {ctx: {k: score}} from `per_rollout_scores` (kept draws only)."""
    payload = json.loads(Path(dv_path).read_text())
    rows = payload["rows"] if isinstance(payload, dict) else payload
    out: dict[str, dict[int, float]] = {}
    n_missing = 0
    for r in rows:
        if r.get("dv") is None:
            continue
        prs = r.get("per_rollout_scores")
        if prs is None:
            n_missing += 1
            continue
        ks = {int(k[1:]): float(s) for k, s in prs.items() if s is not None}
        if ks:
            out[str(r["context_id"])] = ks
    if not out:
        raise RuntimeError(f"{source_label}: zero contexts with per_rollout_scores in {dv_path}")
    if n_missing:
        _log(f"{source_label}: {n_missing} DV rows lack per_rollout_scores (skipped)")
    return out


def _assert_dv_match(pr_dv_path: Path, committed_dv_path: Path, behavior: str) -> None:
    """The hallucination per-rollout DV must reproduce the committed DV values."""
    import numpy as np

    pr = {
        str(r["context_id"]): float(r["dv"])
        for r in json.loads(Path(pr_dv_path).read_text())["rows"]
        if r.get("dv") is not None
    }
    com = {
        str(r["context_id"]): float(r["dv"])
        for r in json.loads(Path(committed_dv_path).read_text())["rows"]
        if r.get("dv") is not None
    }
    shared = sorted(set(pr) & set(com))
    if len(shared) < 0.95 * len(com):
        raise RuntimeError(
            f"[{behavior}] per-rollout DV covers {len(shared)}/{len(com)} committed contexts"
        )
    diffs = np.array([abs(pr[c] - com[c]) for c in shared])
    if diffs.max() > 1e-9:
        raise RuntimeError(
            f"[{behavior}] per-rollout DV diverges from committed DV: max |d|={diffs.max():.3e}"
        )
    _log(f"[{behavior}] per-rollout DV matches committed DV on {len(shared)} ctx (max |d|=0)")


def load_store_grain(
    name: str, store_dir: Path, scores: dict[str, dict[int, float]], source_label: str
) -> StoreGrain:
    """Read a store's row_index into a StoreGrain (no activation arrays loaded)."""
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import store_io

    meta = store_io._index_rows_for(Path(store_dir), ["t1", "context_end"])
    if not meta:
        raise RuntimeError(f"{name}: empty row_index under {store_dir}")
    ctx_key = "context_id" if "context_id" in meta[0] else None
    k_key = "rollout_k" if "rollout_k" in meta[0] else None
    if ctx_key is None or k_key is None:
        raise RuntimeError(f"{name}: row_index lacks context_id/rollout_k: {sorted(meta[0])}")
    row_ctx = [str(r[ctx_key]) for r in meta]
    row_k = np.asarray([int(r[k_key]) if r[k_key] is not None else -1 for r in meta])
    return StoreGrain(
        name=name,
        store_dir=Path(store_dir),
        row_ctx=row_ctx,
        row_k=row_k,
        n_rows=len(meta),
        scores=scores,
        per_rollout_source=source_label,
    )


# ---------------------------------------------------------------------------
# pool-keyed natural-direction weights (verbatim matched_pair_split_weights)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DirectionSpec:
    """One streamed weighted-sum direction: (pool, base regime, extraction kind)."""

    pool: str  # "P-A" | "P-B:<holdout>" | "parity:<label>"
    base: str  # e2 | e2p
    kind: str  # t1 | context_end
    n_qualifying: int
    row_weights: dict[str, object]  # grain name -> (n_rows,) fp64


def build_natural_specs(
    pools: dict[str, set[str]],
    grains: list[StoreGrain],
    *,
    spread_min: float,
) -> tuple[list[DirectionSpec], list[dict]]:
    """Pool x regime weight vectors over every store's rows (zeros off-pool).

    The score matrix per pool is assembled over the UNION of pool contexts
    across stores (so e2p's global midpoint is pooled across stores, matching
    the single-table construction); weights are then scattered back to each
    store's (ctx, k) rows. Returns (specs, skips) — a pool x regime whose
    weight construction raises (zero qualifying contexts / degenerate split)
    is recorded as a skip, never fabricated.
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits

    specs: list[DirectionSpec] = []
    skips: list[dict] = []
    for pool_name, pool_ctx in pools.items():
        # union score matrix over pool contexts present in any grain
        ctx_order = sorted(c for g in grains for c in g.scores if c in pool_ctx)
        ctx_order = sorted(set(ctx_order))
        if not ctx_order:
            skips.append({"pool": pool_name, "reason": "no pool contexts carry judged scores"})
            continue
        pos = {c: i for i, c in enumerate(ctx_order)}
        k_max = 1 + max(k for g in grains for c, ks in g.scores.items() if c in pos for k in ks)
        scores = np.full((len(ctx_order), k_max), np.nan)
        for g in grains:
            for c, ks in g.scores.items():
                if c in pos:
                    for k, s in ks.items():
                        scores[pos[c], k] = s
        for base in NATURAL_BASES:
            try:
                w_hi, w_lo, n_qual = fits.matched_pair_split_weights(
                    scores, spread_min=spread_min, pooled=(base == "e2p")
                )
            except ValueError as exc:
                skips.append({"pool": pool_name, "regime": base, "reason": str(exc)})
                continue
            w_ctx = w_hi - w_lo  # (n_pool_ctx, K)
            # scatter ONCE per (pool, base): the t1 / context_end specs share
            # the same (ctx, k) weights — only the streamed shard kind differs
            row_weights: dict[str, object] = {}
            for g in grains:
                w_row = np.zeros(g.n_rows)
                for i, (c, k) in enumerate(zip(g.row_ctx, g.row_k, strict=True)):
                    if 0 <= k < k_max and c in pos:
                        w_row[i] = w_ctx[pos[c], k]
                if np.any(w_row):
                    row_weights[g.name] = w_row
            kinds = ("t1",) if base == "e2" else ("t1", "context_end")
            for kind in kinds:
                specs.append(
                    DirectionSpec(
                        pool=pool_name,
                        base=base,
                        kind=kind,
                        n_qualifying=int(n_qual),
                        row_weights=row_weights,
                    )
                )
    return specs, skips


def stream_directions(
    specs: list[DirectionSpec], grains: list[StoreGrain], layers: list[int], dim: int
) -> dict[int, object]:
    """Per-shard mask-GEMM streaming: spec index -> (Ly, d) fp64 raw direction.

    For each (kind, layer): iterate every grain's shard files in row_index
    order, accumulating W_slice @ shard into the spec accumulators. Asserts
    the shard-row total equals the grain's row_index length (the silent
    mis-pairing hazard class).
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import store_io

    out = {i: np.zeros((len(layers), dim)) for i in range(len(specs))}
    kinds = sorted({s.kind for s in specs})
    t0 = time.time()
    for kind in kinds:
        idx = [i for i, s in enumerate(specs) if s.kind == kind]
        for li, ly in enumerate(layers):
            for g in grains:
                w_mat = [specs[i].row_weights.get(g.name) for i in idx]
                live = [(i, w) for i, w in zip(idx, w_mat, strict=True) if w is not None]
                if not live:
                    continue
                _resolved, paths = store_io._resolve_summary_kind(g.store_dir, kind, ly)
                offset = 0
                for p in paths:
                    arr = np.load(p)
                    n = arr.shape[0]
                    if arr.shape[1] != dim:
                        raise RuntimeError(f"{p}: hidden dim {arr.shape[1]} != {dim}")
                    block = np.asarray(arr, dtype=np.float64)
                    for i, w in live:
                        seg = w[offset : offset + n]
                        if np.any(seg):
                            out[i][li] += seg @ block
                    offset += n
                    del arr, block
                if offset != g.n_rows:
                    raise RuntimeError(
                        f"{g.name} {kind} L{ly:02d}: shard rows {offset} != row_index "
                        f"{g.n_rows} (silent mis-pairing hazard)"
                    )
        _log(f"streamed kind={kind}: {len(idx)} specs x {len(layers)} layers")
    _log(f"direction streaming done in {time.time() - t0:.0f}s")
    return out


# ---------------------------------------------------------------------------
# factorial scoring (direct projections; run_cell_multi arm1/6/11 parity)
# ---------------------------------------------------------------------------


def _proj_scores(z_ev, mp_ev, za_ev, rb_w):
    """The exact run_cell_multi arm1/6/11 projections on the eval block."""
    from explore_persona_space.experiments.issue_1739.arms import _proj

    return {
        "arm1_ctx_e1": _proj(z_ev, rb_w),
        "arm6_map_proj_e1": _proj(mp_ev, rb_w),
        "arm11_oracle_proj": _proj(za_ev, rb_w),
    }


def _build_eval_block(prep, eval_specs, skips_all, fit_label):
    """Mirror of run_behavior's eval concat (z/za/dv/rungs + per-rung id sets)."""
    import numpy as np

    ev_z, ev_za, ev_dv, ev_rung = [], [], [], []
    eval_id_sets: dict[str, set] = {}
    for label, rows in eval_specs:
        if rows is None:  # pvsynth
            ev_z.append(prep.z_pv)
            ev_za.append(prep.za_pv)
            ev_dv.append(prep.dv_pv)
            ev_rung.append(np.asarray([label] * prep.z_pv.shape[1]))
            eval_id_sets[label] = set(prep.ids_pv)
            continue
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            skips_all.append(
                {"fit": fit_label, "eval_rung": label, "n_eval": 0, "reason": "empty eval rows"}
            )
            continue
        ev_z.append(np.ascontiguousarray(prep.z_ctx[:, rows]))
        ev_za.append(np.ascontiguousarray(prep.z_ans[:, rows]))
        ev_dv.append(prep.dv_raw[rows])
        ev_rung.append(np.asarray([label] * rows.size))
        eval_id_sets[label] = {prep.ctx_ids[i] for i in rows}
    z_ev = np.concatenate(ev_z, axis=1)
    za_ev = np.concatenate(ev_za, axis=1)
    return z_ev, za_ev, np.concatenate(ev_dv), np.concatenate(ev_rung), eval_id_sets


def run_factorial_behavior(args, behavior: str, layers: list[int]) -> dict:
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits
    from explore_persona_space.experiments.issue_1739.constants import E2_SPREAD_MIN
    from scripts.issue1739_fits import _load_rb_e1
    from scripts.issue1739_jobd_r2aug import _free_cuda, behavior_paths, per_layer_rows_for
    from scripts.issue1739_r2v2_score import (
        OOD_SPECS,
        PV_RUNG,
        WC_RUNG,
        _log_rss,
        assemble_readout_pool,
        prepare_behavior,
    )

    t0 = time.time()
    prep = prepare_behavior(args, behavior, layers)
    _log_rss("fact-prepared")
    paths = behavior_paths(args, behavior)
    dim = prep.loaded.dim

    # ---- extraction grains -------------------------------------------------
    if behavior == "hallucination":
        pr_path = args.hallu_per_rollout_dv
        if pr_path is None or not Path(pr_path).exists():
            raise FileNotFoundError(
                "hallucination natural extraction needs --hallu-per-rollout-dv "
                "(judge/hallucination/labeling_per_rollout.json — the three_way-derived "
                "per-rollout source); the committed DV rows carry no per_rollout_scores"
            )
        _assert_dv_match(pr_path, paths["train_dv"], behavior)
        train_scores = _per_rollout_from_dv(pr_path, source_label="hallucination three_way")
        train_source = (
            "issue1739_ctxmap/judge/hallucination/labeling_per_rollout.json "
            "(three_way; fabricated=100 / correct|abstained=0)"
        )
    else:
        train_scores = _per_rollout_from_dv(paths["train_dv"], source_label=f"{behavior} train DV")
        train_source = "labeling.json per_rollout_scores"
    grains = [load_store_grain("train", paths["train_store"], train_scores, train_source)]
    if behavior in OOD_SPECS:
        ood_dv = args.evil_ood_dv if behavior == "evil" else args.syco_ood_dv
        ood_scores = _per_rollout_from_dv(ood_dv, source_label=f"{behavior} OOD DV")
        for rel in OOD_SPECS[behavior]["stores"]:
            grains.append(
                load_store_grain(
                    rel,
                    Path(args.ood_store_root) / rel,
                    ood_scores,
                    "OOD labeling.json per_rollout_scores",
                )
            )

    # ---- pools -------------------------------------------------------------
    ctx_ids_arr = prep.ctx_ids
    pa_ctx = {ctx_ids_arr[i] for i in np.asarray(prep.elic_cell.row_idx)}
    pools: dict[str, set[str]] = {"P-A": pa_ctx}
    holdouts = args.pb_holdouts or prep.eval_datasets
    fold_pool_rows: dict[str, object] = {}
    if "B" in args.protocols:
        for holdout in holdouts:
            pool = assemble_readout_pool(
                prep.datasets, holdout=holdout, train_frac=args.train_frac, seed=args.seed
            )
            rows = np.concatenate([pool.train_rows[n] for n in sorted(pool.train_rows)])
            fold_pool_rows[holdout] = rows
            pools[f"P-B:{holdout}"] = {ctx_ids_arr[i] for i in rows}
    # parity pools (machinery validation vs banked rb_fc_bank vectors)
    if not args.skip_parity:
        pools["parity:trainstore-all"] = set(grains[0].scores)
        pools["parity:train-rung"] = {str(c) for c in prep.loaded.tbl.ctx_order} & set(
            grains[0].scores
        )

    specs, spec_skips = build_natural_specs(pools, grains, spread_min=E2_SPREAD_MIN)
    dirs_raw_by_idx = stream_directions(specs, grains, layers, dim)
    _log_rss("fact-directions-streamed")

    # direction registry: (regime, pool) -> raw (Ly, d)
    dirs_raw: dict[tuple[str, str], object] = {}
    qual_counts: list[dict] = []
    for i, s in enumerate(specs):
        regime = s.base if s.kind == "t1" else f"{s.base}_fc"
        dirs_raw[(regime, s.pool)] = dirs_raw_by_idx[i]
        qual_counts.append(
            {
                "pool": s.pool,
                "regime": regime,
                "n_qualifying": s.n_qualifying,
                "stores": sorted(s.row_weights),
            }
        )
    # e1: the base leg's own direction (bank/extract per --rb-source) => exact
    # parity with the base r2v2 rows; e1_fc: context_end extraction store read.
    dirs_raw[("e1", "global")] = np.asarray(prep.loaded.rb, dtype=np.float64)
    dirs_raw[("e1_fc", "global")] = _load_rb_e1(
        paths["e1_store"], layers, dim, summary_kind="context_end"
    )
    # degenerate-norm halt (the _k2_gate sibling): a ~zero streamed direction
    # is a weight-cancellation bug (e2_fc's cancellation is EXCLUDED upstream),
    # never a scoreable cell — fail loud, do not project noise.
    for (regime, pool), rb_raw in dirs_raw.items():
        max_norm = float(np.linalg.norm(np.asarray(rb_raw), axis=1).max())
        if max_norm < 1e-10:
            raise RuntimeError(
                f"[{behavior}] degenerate direction ({regime}, {pool}): "
                f"max per-layer norm {max_norm:.3e} — weight-cancellation bug"
            )

    # ---- parity gate vs banked directions (BEFORE any scoring) --------------
    # a re-extraction that fails to reproduce the banked P-A/full-pool
    # direction is a pipeline BUG — stop here, never proceed to P-B numbers.
    parity = [] if args.skip_parity else parity_vs_bank(args, behavior, dirs_raw, layers)
    matches_by_regime: dict[str, list[bool]] = {}
    for row in parity:
        if "match" in row:
            matches_by_regime.setdefault(str(row["regime"]), []).append(bool(row["match"]))
    for regime, matches in matches_by_regime.items():
        if not any(matches):
            raise RuntimeError(
                f"[{behavior}] parity FAILED for {regime}: no re-extraction pool matches "
                f"the banked direction (rows: {[r for r in parity if r.get('regime') == regime]})"
                " — pipeline bug, refusing to score"
            )
    _log(f"[{behavior}] parity vs bank: {parity or 'skipped'}")

    # ---- whiten + score the 15 cells ----------------------------------------
    wh, mapfit = prep.wh, prep.mapfit
    frozen3 = {a: prep.frozen[a] for a in RB_ARMS if a in prep.frozen}
    missing_frozen = sorted(set(RB_ARMS) - set(frozen3))
    if missing_frozen:
        raise RuntimeError(f"[{behavior}] committed frozen layers missing for {missing_frozen}")

    def _whiten_dir(rb_raw):
        return np.einsum("ld,lde->le", np.asarray(rb_raw, dtype=np.float64), wh.w)

    def _dir_for(regime: str, protocol: str, holdout: str | None):
        """Regime direction for a protocol context; None + reason when N/A."""
        if regime in ("e1", "e1_fc"):
            return dirs_raw[(regime, "global")], None
        pool_key = "P-A" if protocol == "P-A" else f"P-B:{holdout}"
        key = (regime, pool_key)
        if key not in dirs_raw:
            return None, f"no {regime} direction for pool {pool_key} (weight construction skip)"
        return dirs_raw[key], None

    rows_all: list[dict] = []
    skips_all: list[dict] = list(spec_skips)
    per_layer_all: list[dict] = []
    kwargs = {"n_boot": args.n_boot} if args.n_boot else {}

    def _score_cells(protocol: str, fit_label: str, eval_specs, holdout=None, extra_prov=None):
        z_ev, za_ev, dv_ev, rungs_ev, eval_id_sets = _build_eval_block(
            prep, eval_specs, skips_all, fit_label
        )
        mp_ev = fits.apply_map(z_ev, mapfit)  # hoisted ONCE per fit context
        # per-fold extraction-pool / eval disjointness (natural regimes only)
        pool_key = "P-A" if protocol == "P-A" else f"P-B:{holdout}"
        pool_ctx = pools.get(pool_key, set())
        leakage = {}
        for label, ids in eval_id_sets.items():
            if label == "train" or label == PV_RUNG:
                # P-A train rung: in-sample for the direction BY DESIGN (any PV
                # construction extracts from its eliciting rows); pvsynth is a
                # disjoint synthetic store. Recorded in provenance, not asserted.
                continue
            inter = pool_ctx & ids
            leakage[label] = len(inter)
            if inter and protocol == "P-B" and (label == holdout or label.startswith("heldin:")):
                # holdout: fold leak. heldin: the 20% remainder is disjoint from
                # the 80% pool by the group-level split — overlap = split bug.
                raise RuntimeError(
                    f"[{behavior}] {fit_label}: extraction pool intersects {label} "
                    f"({len(inter)} ctx) — per-fold disjointness violation"
                )
        cell = fits.BudgetCell(
            row_idx=np.zeros(1, dtype=np.int64),
            fold_ids=np.zeros(1, dtype=np.int64),
            n_folds=1,
            budget_l=prep.lmax,
            draw=args.draw,
            seed=args.seed,
            fold_scheme=f"r2v2-factorial-{protocol}-projection",
        )
        for regime in FACT_REGIMES:
            rb_raw, reason = _dir_for(regime, protocol, holdout)
            if rb_raw is None:
                skips_all.append(
                    {"protocol": protocol, "fit": fit_label, "regime": regime, "reason": reason}
                )
                continue
            rb_w = _whiten_dir(rb_raw)
            scores_ev = _proj_scores(z_ev, mp_ev, za_ev, rb_w)
            prov = {
                "mode": "r2v2-factorial",
                "protocol": protocol,
                "fit": fit_label,
                "behavior": behavior,
                "variant": prep.variant,
                "regime": regime,
                "extraction_pool": "global (synthetic contrastive store)"
                if regime in ("e1", "e1_fc")
                else pool_key,
                "map_kind": "linear",
                "map_condition": "add",
                "scoring": "direct-projection (run_cell_multi arm1/6/11 parity)",
                "extraction_leakage_ctx_overlap": leakage,
                **(extra_prov or {}),
            }
            if holdout is not None:
                prov["holdout"] = holdout
            rows, skips = arms.evaluate_transfer(
                scores_ev,
                dv_ev,
                rungs_ev,
                frozen3,
                provenance=prov,
                cell=cell,
                layers=tuple(layers),
                min_n=args.min_n,
                **kwargs,
            )
            rows_all.extend(rows)
            skips_all.extend(skips)
            per_layer_all.extend(
                per_layer_rows_for(
                    scores_ev,
                    dv_ev,
                    frozen3,
                    {**prov, "eval_rung": "all"},
                    layers,
                    prep.frozen_src,
                )
            )
        del z_ev, za_ev, mp_ev
        _log(f"[{behavior}] {fit_label}: factorial cells scored")
        _log_rss(f"fact-scored-{fit_label}")

    # regime x (e2_fc) structural N/A — recorded ONCE per behavior
    skips_all.append(
        {
            "behavior": behavior,
            "regime": "e2_fc",
            "reason": STRUCTURAL_NA[("e2_fc", "direction")],
            "structural_na": True,
        }
    )

    if "A" in args.protocols:
        eval_specs_pa = [(WC_RUNG, prep.wc_eval_rows), (PV_RUNG, None)]
        eval_specs_pa += [(n, prep.ds_by_name[n].rows) for n in prep.eval_datasets]
        eval_specs_pa += [("train", np.asarray(prep.elic_cell.row_idx, dtype=np.int64))]
        _score_cells(
            "P-A",
            "P-A",
            eval_specs_pa,
            extra_prov={
                "train_rung_note": "projection scores on the eliciting cell rows; natural "
                "regimes' P-A directions are extracted from these same rows (in-sample for "
                "the direction, as for any PV construction)"
            },
        )
    if "B" in args.protocols:
        for holdout in holdouts:
            pool = assemble_readout_pool(
                prep.datasets, holdout=holdout, train_frac=args.train_frac, seed=args.seed
            )
            eval_specs_pb = [
                (holdout, prep.ds_by_name[holdout].rows),
                (WC_RUNG, prep.wc_eval_rows),
                (PV_RUNG, None),
            ]
            eval_specs_pb += [
                (f"heldin:{n}", pool.heldin_eval_rows[n]) for n in sorted(pool.heldin_eval_rows)
            ]
            _score_cells("P-B", f"P-B-holdout-{holdout}", eval_specs_pb, holdout=holdout)

    # ---- mapped-vs-real direction comparison --------------------------------
    comparison = direction_comparison(
        args, behavior, prep, dirs_raw, pools, layers, holdouts if "B" in args.protocols else []
    )

    # ---- per-rung DV spread stats (gate visibility) --------------------------
    rung_dv_stats = {}
    rungs_arr = np.asarray(
        ["train"] * prep.n_tr
        + [WC_RUNG] * prep.n_wc
        + list(prep.loaded.tbl_ev.row_rungs)
        + (list(prep.tbl_ood.row_rungs) if prep.tbl_ood is not None else [])
    )
    for rung in sorted(set(rungs_arr)):
        dv_r = prep.dv_raw[rungs_arr == rung]
        rung_dv_stats[rung] = {
            "n": int(dv_r.size),
            "mean": float(dv_r.mean()),
            "sd": float(dv_r.std()),
            "bottom_bin_frac_lt5": float((dv_r < 5).mean()),
        }

    _free_cuda(args.device)
    return {
        "rows": rows_all,
        "skips": skips_all,
        "per_layer": per_layer_all,
        # raw-space directions persisted for VM-side re-comparison / reuse
        # (the bank npz are gitignored, so a pod-side parity gap is recoverable)
        "directions": {f"{r}|{p}": np.asarray(v) for (r, p), v in dirs_raw.items()},
        "direction_comparison": comparison,
        "parity_vs_bank": parity,
        "qualifying_counts": qual_counts,
        "per_rollout_sources": {g.name: g.per_rollout_source for g in grains},
        "rung_dv_stats": rung_dv_stats,
        "frozen": {a: int(i) for a, i in frozen3.items()},
        "frozen_source": prep.frozen_src,
        "pools": {k: len(v) for k, v in pools.items()},
        "wall_s": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# mapped-vs-real direction comparison (derived via linear-map commutation)
# ---------------------------------------------------------------------------


def _apply_map_to_direction(rb_w, mapfit, w=None):
    """r_B^mapped[ly] = (r_B^ctx_w[ly] / x_sd[ly]) @ W[ly] (affine terms cancel:
    hi/lo weights each sum to 1, so x_mu / y_mu drop out of the weighted diff)."""
    import numpy as np

    weights = mapfit.w if w is None else w
    out = np.empty_like(np.asarray(rb_w, dtype=np.float64))
    for li in range(out.shape[0]):
        out[li] = (rb_w[li] / mapfit.x_sd[li, 0]) @ weights[li]
    return out


def _cos_profile(a, b):
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.where((na > 0) & (nb > 0), na * nb, np.nan)
    cos = np.einsum("ld,ld->l", a, b) / denom
    ratio = np.where(nb > 0, na / nb, np.nan)
    return [round(float(c), 6) for c in cos], [round(float(r), 6) for r in ratio]


def direction_comparison(args, behavior, prep, dirs_raw, pools, layers, pb_holdouts):
    """cos/norm layer profiles: mapped vs real, with identity + shuffled floors."""
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits

    wh, mapfit = prep.wh, prep.mapfit
    w_shuf = fits.shuffled_map_weights(mapfit.w, seed=args.seed)

    def _w(rb_raw):
        return np.einsum("ld,lde->le", np.asarray(rb_raw, dtype=np.float64), wh.w)

    out = []
    contexts = [("P-A", "P-A")] + [(f"P-B:{h}", f"P-B:{h}") for h in pb_holdouts]
    pairs = [("e1", "e1_fc", "global"), ("e2p", "e2p_fc", "pool")]
    for base, fc, scope in pairs:
        for proto_label, pool_key in contexts if scope == "pool" else [("global", "global")]:
            k_real = (base, "global") if scope == "global" else (base, pool_key)
            k_ctx = (fc, "global") if scope == "global" else (fc, pool_key)
            if k_real not in dirs_raw or k_ctx not in dirs_raw:
                out.append(
                    {
                        "behavior": behavior,
                        "regime": base,
                        "context": proto_label,
                        "status": "unavailable",
                        "reason": f"missing direction {k_real if k_real not in dirs_raw else k_ctx}",
                    }
                )
                continue
            rb_real_w = _w(dirs_raw[k_real])
            rb_ctx_w = _w(dirs_raw[k_ctx])
            rb_map_w = _apply_map_to_direction(rb_ctx_w, mapfit)
            rb_shuf_w = _apply_map_to_direction(rb_ctx_w, mapfit, w=w_shuf)
            cos_map, ratio_map = _cos_profile(rb_map_w, rb_real_w)
            cos_id, ratio_id = _cos_profile(rb_ctx_w, rb_real_w)
            cos_shuf, ratio_shuf = _cos_profile(rb_shuf_w, rb_real_w)
            out.append(
                {
                    "behavior": behavior,
                    "regime": base,
                    "context": proto_label,
                    "space": "whitened (the space the map is fit and applied in)",
                    "statistic": "direction-aware raw cosine per layer (vectors, not "
                    "operators; NOT rotation-invariant)",
                    "derivation": "r_B^mapped = M(r_B^ctx) exactly (linear map commutes "
                    "with the mean-difference construction; affine offsets cancel)",
                    "layers": [int(x) for x in layers],
                    "cos_mapped_vs_real": cos_map,
                    "norm_ratio_mapped_over_real": ratio_map,
                    "cos_identity_ctx_vs_real": cos_id,
                    "norm_ratio_ctx_over_real": ratio_id,
                    "cos_shuffledmap_vs_real": cos_shuf,
                    "norm_ratio_shuffledmap_over_real": ratio_shuf,
                    "identity_bias_note": "learned-bias identity (v=x+b) collapses to plain "
                    "identity for mean-difference directions (b cancels in the difference)",
                }
            )
    out.append(
        {
            "behavior": behavior,
            "regime": "e2",
            "status": "structural_na",
            "reason": STRUCTURAL_NA[("e2", "mapped_direction")],
        }
    )
    return out


def parity_vs_bank(args, behavior, dirs_raw, layers):
    """Re-extracted full-pool fc directions vs the banked rb_fc_bank npz (raw space)."""
    import numpy as np

    out = []
    bank_dir = Path(args.rb_bank_dir)
    candidates = {
        "e1_fc": [f"{behavior}__e1_fc.npz"],
        "e2p_fc": [f"{behavior}__natpv_e2p_fc.npz", f"{behavior}__e2p_fc.npz"],
    }
    pool_by_regime = {
        "e1_fc": [("global", ("e1_fc", "global"))],
        "e2p_fc": [
            ("trainstore-all", ("e2p_fc", "parity:trainstore-all")),
            ("train-rung", ("e2p_fc", "parity:train-rung")),
        ],
    }
    for regime, names in candidates.items():
        bank_path = next((bank_dir / n for n in names if (bank_dir / n).exists()), None)
        if bank_path is None:
            out.append({"regime": regime, "status": "no bank file", "candidates": names})
            continue
        with np.load(bank_path, allow_pickle=True) as z:
            rb_bank = np.asarray(z["rb"], dtype=np.float64)
            bank_layers = [int(x) for x in z["layers"]]
        if bank_layers != list(layers):
            out.append({"regime": regime, "status": f"layer mismatch {bank_layers[:3]}..."})
            continue
        for pool_label, key in pool_by_regime[regime]:
            if key not in dirs_raw:
                continue
            cos, ratio = _cos_profile(dirs_raw[key], rb_bank)
            out.append(
                {
                    "regime": regime,
                    "bank": bank_path.name,
                    "reextraction_pool": pool_label,
                    "cos_min": float(np.nanmin(cos)),
                    "cos_median": float(np.nanmedian(cos)),
                    "norm_ratio_median": float(np.nanmedian(ratio)),
                    "match": bool(np.nanmin(cos) > 0.995),
                }
            )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from scripts.issue1739_r2v2_score import parse_args as base_parse_args

    fact = argparse.ArgumentParser(add_help=False)
    fact.add_argument(
        "--fact-out-root",
        type=Path,
        default=Path("eval_results/issue_1739/r2v2_factorial"),
        help="output root for the factorial legs (distinct from the base fits out-root)",
    )
    fact.add_argument(
        "--rb-bank-dir",
        type=Path,
        default=Path("eval_results/issue_1739/new_arm_round/fc/rb_fc_bank"),
        help="banked fc-direction npz dir for the parity validation",
    )
    fact.add_argument(
        "--hallu-per-rollout-dv",
        type=Path,
        default=None,
        help="hallucination per-rollout DV (labeling_per_rollout.json; three_way source)",
    )
    fact.add_argument("--skip-parity", action="store_true")
    fargs, rest = fact.parse_known_args(argv)
    args = base_parse_args(rest)
    for k, v in vars(fargs).items():
        setattr(args, k, v)
    return args


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv
    from scripts.issue1739_wcrung_arms import _assert_no_judge_modules

    load_dotenv()
    _assert_no_judge_modules("at entry")
    args = parse_args(argv)
    if args.import_check:
        import numpy  # noqa: F401

        from explore_persona_space.experiments.issue_1739 import arms, fits, store_io  # noqa: F401
        from explore_persona_space.experiments.issue_1739.constants import (  # noqa: F401
            E2_SPREAD_MIN,
        )
        from explore_persona_space.orchestrate.provenance import git_provenance  # noqa: F401
        from scripts.issue1739_fits import _load_rb_e1  # noqa: F401
        from scripts.issue1739_jobd_r2aug import behavior_paths, per_layer_rows_for  # noqa: F401
        from scripts.issue1739_r2v2_score import assemble_readout_pool  # noqa: F401
        from scripts.issue1739_r2v2_score import prepare_behavior  # noqa: F401

        print("[r2v2-fact] import-check OK")
        return 0
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance
    from scripts.issue1739_jobd_r2aug import _env_versions

    layers = args.layers or list(range(args.n_layers))
    for behavior in args.behaviors:
        res = run_factorial_behavior(args, behavior, layers)
        out_dir = Path(args.fact_out_root) / behavior
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "behavior": behavior,
            "mode": "r2v2-factorial",
            "regimes": list(FACT_REGIMES),
            "arms": sorted(RB_ARMS),
            "arm_regime_keying": "cells keyed on (arm, regime); the legacy _e1 arm-slug "
            "suffix does NOT encode the regime",
            "rb_independent_arms_note": "arm4_ridge_ctx / arm7_map_ridge_pred are invariant "
            "to the extraction axes; their cells are banked in r2v2_fits/<behavior> "
            "(NOT re-run here)",
            "structural_na": {f"{k[0]}|{k[1]}": v for k, v in STRUCTURAL_NA.items()},
            "per_rollout_sources": res["per_rollout_sources"],
            "qualifying_counts": res["qualifying_counts"],
            "pools": res["pools"],
            "frozen": res["frozen"],
            "frozen_source": res["frozen_source"],
            "rung_dv_stats": res["rung_dv_stats"],
            "spread_gate_note": "evil_pair FAILED the pre-registered DV spread floor "
            "(bottom-bin fraction 0.802-0.811 vs 0.80 gate, borderline; mhj + tomgibbs "
            "PASS decisively) — every evil_pair column carries that caveat (v653)",
            "e2_e2p_qualification_note": "e2 qualifies contexts within-context (>=2 kept "
            "draws AND spread >= 15); e2p pools globally — per-pool qualifying counts "
            "differ sharply and are reported per cell, never like-for-like",
            "estimator_parity": "direct _proj on eval blocks == arms.run_cell_multi "
            "arm1/6/11 dispatch (apply_map is row-wise; see module docstring); pinned "
            "in-run by the e1-vs-base check in tests/test_issue1739_r2v2_factorial.py",
            "wall_s": res["wall_s"],
            "git": as_metadata_dict(git_provenance()),
            "env": _env_versions(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (out_dir / "factorial_rows.json").write_text(
            json.dumps(
                {
                    "meta": meta,
                    "transfer_rows": res["rows"],
                    "skips": res["skips"],
                    "per_layer": res["per_layer"],
                },
                indent=1,
                default=str,
            )
        )
        (out_dir / "direction_comparison.json").write_text(
            json.dumps({"meta": meta, "comparisons": res["direction_comparison"]}, indent=1)
        )
        (out_dir / "extraction_report.json").write_text(
            json.dumps(
                {
                    "meta": meta,
                    "parity_vs_bank": res["parity_vs_bank"],
                    "qualifying_counts": res["qualifying_counts"],
                },
                indent=1,
            )
        )
        import numpy as np

        np.savez(
            out_dir / "directions_fp16.npz",
            **{k: v.astype(np.float16) for k, v in res["directions"].items()},
            layers=np.asarray(layers),
        )
        _log(f"[{behavior}] factorial leg done in {res['wall_s']}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
