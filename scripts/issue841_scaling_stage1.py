#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, ℓ, →, ĥ, ρ) in scientific docstrings/log messages.
"""Issue #841 scaling — Stage-1 transported-trait-monitor CURVE across fit-sizes.

Re-runs the parent's transported-projection benchmark at EVERY fitted n vs the
4k anchor, reading the transport advantage as a CURVE (not a single endpoint) so
a non-monotone / inverted-U response is not misread as FLAT (plan v9 §3 H2).

The map-INDEPENDENT rows (raw-source, id-transport) are computed ONCE; only the
transported (ridge) read depends on the fitted maps, so it is recomputed per n
by loading the per-n ridge maps ``issue841_scaling_stage0`` persisted. Everything
else — r_B, pass_a eval trajectories, cached judge scores, ℓ*, the
within-condition Pearson protocol, and the #779 ``bootstrap_delta_ci`` — is
#779/#841-verbatim. NO new judging.

Per fit-size n (vs the 4k anchor):
  win_count(n)         : # cells where the n-transported read beats BOTH raw-source
                         AND id-transport (bootstrap_delta_ci CI excludes 0, delta>0)
  net / newly / dropped: symmetric win accounting vs the 4k baseline
  mean_paired_delta(n) : mean over cells of r(transported@n) − r(transported@4k),
                         per-cell joint condition-resample bootstrap CI
  BH_survivors(n)      : newly-winning cells surviving Benjamini-Hochberg FDR
  transport_fidelity(n): eval-context Δ-reconstruction R²/cosine per (source,ℓ*)

--synthetic fabricates a tiny in-memory trait + maps (no HF, no GPU) that runs the
FULL win-count/paired-delta/BH/npz path end-to-end on CPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue841_common as C  # noqa: E402
import issue841_scaling_common as S  # noqa: E402
import issue841_stage1_benchmark as B1  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_779.metrics import (  # noqa: E402
    bootstrap_delta_ci,
    within_condition_pearson,
)
from explore_persona_space.experiments.issue_841 import maps as MP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue841_scaling_stage1")

MODES = B1.MODES  # ("system", "many_shot")


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("--device cuda requested but no CUDA device; falling back to cpu")
        return "cpu"
    return requested


# ── paired bootstrap vs the anchor (p-value-augmented mirror of #779's) ──────────


# ── vectorized bootstrap (EXACT equivalent of #779's per-draw loop) ───────────────
# The condition-resampling bootstrap re-runs within_condition_pearson per draw, at
# ~9.6s/call on the shared VM (a Python loop over conditions × n_boot); the
# class×n multiplication of the scaling curve makes that battery a ~15h CPU job.
# But within_condition_pearson's r on a condition-resample is EXACTLY the mean of
# the per-condition r over the resampled conditions (each condition's r is fixed;
# resampling is at the condition level) — so precompute per-condition r ONCE and
# reduce each draw as an array mean. Same rng.choice sequence (same seed ⇒ same
# resamples) ⇒ bit-identical to the #779 helpers; ~1000× faster. Verified against
# the #779 references by _assert_fast_matches_reference() at main() entry.


def _per_cond_r(cond_x, cond_y):
    """Per-condition within-condition r (NaN for a condition #779's wcp excludes)."""
    return np.array(
        [within_condition_pearson([cond_x[i]], [cond_y[i]])["r"] for i in range(len(cond_y))],
        dtype=float,
    )


def _resample_idx(seed, n_cond, n_boot):
    """The EXACT #779 rng.choice(arange(n_cond), size=n_cond) draw sequence."""
    rng = np.random.default_rng(seed)
    idx_all = np.arange(n_cond)
    return [rng.choice(idx_all, size=n_cond, replace=True) for _ in range(n_boot)]


def _mean_or_nan(a):
    f = a[np.isfinite(a)]
    return float(f.mean()) if f.size else float("nan")


def _delta_fast(pc_a, pc_b, samps) -> dict:
    """Vectorized bootstrap_delta_ci: within-condition r_a − r_b + 95% CI."""
    point = _mean_or_nan(pc_a) - _mean_or_nan(pc_b)
    deltas = []
    for s in samps:
        ra, rb = _mean_or_nan(pc_a[s]), _mean_or_nan(pc_b[s])
        if np.isfinite(ra) and np.isfinite(rb):
            deltas.append(ra - rb)
    if not deltas:
        return {"delta": point, "lo": float("nan"), "hi": float("nan"), "excludes_zero": False}
    arr = np.asarray(deltas)
    lo, hi = float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))
    return {
        "delta": point,
        "lo": lo,
        "hi": hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "_arr": arr,
    }


def _paired_fast(pc_n, pc_anchor, samps) -> dict:
    """Vectorized paired_delta_vs_anchor: r_n − r_anchor + CI + one-sided p."""
    d = _delta_fast(pc_n, pc_anchor, samps)
    arr = d.pop("_arr", None)
    d["p_one_sided"] = float(np.mean(arr <= 0.0)) if arr is not None else float("nan")
    d["n_boot_valid"] = int(arr.size) if arr is not None else 0
    return d


def _retention_fast(pc_row, pc_ceil, samps) -> dict:
    """Vectorized bootstrap_retention_ci: JOINT ratio r_row / r_ceil (unclipped)."""
    r_row, r_ceil = _mean_or_nan(pc_row), _mean_or_nan(pc_ceil)
    point = (
        r_row / r_ceil
        if (np.isfinite(r_row) and np.isfinite(r_ceil) and r_ceil != 0.0)
        else float("nan")
    )
    ratios = []
    for s in samps:
        rr, rc = _mean_or_nan(pc_row[s]), _mean_or_nan(pc_ceil[s])
        if np.isfinite(rr) and np.isfinite(rc) and rc != 0.0:
            ratios.append(rr / rc)
    if not ratios:
        return {
            "point": point,
            "lo": float("nan"),
            "hi": float("nan"),
            "r_row": r_row,
            "r_ceiling": r_ceil,
            "n_boot_valid": 0,
        }
    a = np.asarray(ratios)
    return {
        "point": point,
        "lo": float(np.quantile(a, 0.025)),
        "hi": float(np.quantile(a, 0.975)),
        "r_row": r_row,
        "r_ceiling": r_ceil,
        "n_boot_valid": len(a),
    }


def _assert_fast_matches_reference() -> None:
    """Gate: the vectorized helpers reproduce the #779 references bit-for-bit.

    Two shapes hard-fail on any >1e-9 divergence — a small tidy case AND a
    production-shape leg (n_boot=997, a pass_a-shaped 4-8 condition count) so the
    gate exercises the real bootstrap size, not just a toy draw count.
    """

    def _check_case(n_cond: int, n_rows: int, n_boot: int, seed_data: int) -> None:
        rng = np.random.default_rng(seed_data)
        cy = [rng.standard_normal(n_rows) * 10 + 50 for _ in range(n_cond)]
        cx_a = [rng.standard_normal(n_rows) for _ in range(n_cond)]
        cx_b = [rng.standard_normal(n_rows) for _ in range(n_cond)]
        cx_c = [rng.standard_normal(n_rows) for _ in range(n_cond)]
        # seed=0 MUST match the reference calls: _resample_idx(0, ...) reproduces
        # the EXACT rng.choice draw sequence bootstrap_*_ci(seed=0) uses.
        samps = _resample_idx(0, len(cy), n_boot)
        ref_d = bootstrap_delta_ci(cx_a, cx_b, cy, n_boot=n_boot, seed=0)
        fast_d = _delta_fast(_per_cond_r(cx_a, cy), _per_cond_r(cx_b, cy), samps)
        for k in ("delta", "lo", "hi"):
            assert abs((ref_d[k] or 0.0) - (fast_d[k] or 0.0)) < 1e-9, (k, ref_d[k], fast_d[k])
        assert ref_d["excludes_zero"] == fast_d["excludes_zero"]
        ref_r = B1.bootstrap_retention_ci(cx_a, cx_c, cy, n_boot=n_boot, seed=0)
        fast_r = _retention_fast(_per_cond_r(cx_a, cy), _per_cond_r(cx_c, cy), samps)
        for k in ("point", "lo", "hi"):
            assert abs((ref_r[k] or 0.0) - (fast_r[k] or 0.0)) < 1e-9, (k, ref_r[k], fast_r[k])

    _check_case(n_cond=11, n_rows=8, n_boot=200, seed_data=7)  # small tidy case
    _check_case(n_cond=5, n_rows=8, n_boot=997, seed_data=13)  # production-shape leg
    logger.info("[gate] vectorized bootstrap == #779 references (delta + retention, 2 shapes)")


def benjamini_hochberg(pvals: list[float], q: float = 0.05) -> list[bool]:
    """BH-FDR: return a boolean survivor mask (NaN p-values never survive)."""
    finite = [(i, p) for i, p in enumerate(pvals) if np.isfinite(p)]
    survive = [False] * len(pvals)
    if not finite:
        return survive
    finite.sort(key=lambda x: x[1])
    m = len(finite)
    k_max = 0
    for rank, (_, p) in enumerate(finite, start=1):
        if p <= q * rank / m:
            k_max = rank
    for rank, (i, _) in enumerate(finite, start=1):
        if rank <= k_max:
            survive[i] = True
    return survive


# ── per-cell win + paired records ────────────────────────────────────────────────


def _grouped(x, mat, mode):
    """Per-condition (x, y) arrays for one mode with NaN-x pruned + std/N gates."""
    cx, cy = C.group_by_condition(x, mat["y"], mat["cond"], mat["mode"], mode)
    gx, gy = [], []
    for xi, yi in zip(cx, cy, strict=True):
        m = np.isfinite(xi)
        if m.sum() >= 3:
            gx.append(xi[m])
            gy.append(yi[m])
    return gx, gy


TRANSPORT_CLASSES = ("ridge", "mlp", "direct_hop")


def _transported_by_class(
    traj_dev,
    fit_pool,
    src,
    tgt,
    r_b_tgt,
    ridge_maps_by_n,
    mlp_maps_by_n,
    ns_all,
    *,
    device,
    dual_max,
):
    """Per-class {n: (proj, h_hat)} for one (src → tgt) cell.

    ridge/mlp compose the persisted one-step maps (plan row 1, matched-information);
    direct_hop refits ℓ→ℓ* on the SAME nested fit pool per n (plan row 4; primal
    path at n>dual_max via fit_direct_hop_ridge's solver dispatch).
    """
    out: dict[str, dict[int, tuple]] = {c: {} for c in TRANSPORT_CLASSES}
    for n in ns_all:
        out["ridge"][n] = B1._transport_proj(ridge_maps_by_n[n], traj_dev, src, tgt, r_b_tgt)
        if n in mlp_maps_by_n:
            out["mlp"][n] = B1._transport_proj(mlp_maps_by_n[n], traj_dev, src, tgt, r_b_tgt)
        fit = fit_pool[:n]
        dmap = MP.fit_direct_hop_ridge(
            fit[:, src, :], fit[:, tgt, :], fit[:1, src, :], device=device, n=n, dual_max=dual_max
        )
        h_hat_d = traj_dev[:, src, :] + dmap.apply(traj_dev[:, src, :])
        out["direct_hop"][n] = (B1._proj(h_hat_d, r_b_tgt), h_hat_d)
    return out


def process_trait(
    trait,
    mat,
    r_b,
    ridge_maps_by_n,
    mlp_maps_by_n,
    fit_pool,
    schemes,
    ns_all,
    anchor_n,
    *,
    n_boot,
    seed,
    device,
    dual_max,
    smoke,
):
    """Per-(class, scheme, source, mode) win + paired-delta + retention across n; + fidelity/proj.

    Returns ``cells_by_class`` (class → list of cell records), each cell's ``per_n``
    keyed by that class's available n-points (ridge/direct_hop: all n; mlp: the
    n-points with a persisted map). aggregate_curve is called per class downstream.
    """
    traj_dev = torch.from_numpy(np.ascontiguousarray(mat["traj"])).to(
        device=device, dtype=torch.float32
    )
    rb_dev = torch.from_numpy(np.ascontiguousarray(r_b)).to(device=device, dtype=torch.float32)

    cells_by_class: dict[str, list[dict]] = {c: [] for c in TRANSPORT_CLASSES}
    fidelity: dict = {}
    proj_store: dict = {}
    for scheme, tgt in schemes.items():
        r_b_tgt = rb_dev[tgt]
        fidelity[scheme] = {c: {} for c in TRANSPORT_CLASSES}
        ceiling = B1._proj(traj_dev[:, tgt, :], r_b_tgt)  # row 3 raw-target (map-independent)
        proj_store[f"{scheme}__ceiling"] = ceiling
        for src in B1.source_grid(tgt, smoke):
            row2 = B1._proj(traj_dev[:, src, :], rb_dev[src])  # raw_source (map-independent)
            row1b = B1._proj(traj_dev[:, src, :], r_b_tgt)  # id_transport (map-independent)
            proj_store[f"{scheme}__{src}__raw_source"] = row2
            proj_store[f"{scheme}__{src}__id_transport"] = row1b
            tb = _transported_by_class(
                traj_dev,
                fit_pool,
                src,
                tgt,
                r_b_tgt,
                ridge_maps_by_n,
                mlp_maps_by_n,
                ns_all,
                device=device,
                dual_max=dual_max,
            )
            for cls in TRANSPORT_CLASSES:
                fidelity[scheme][cls][str(src)] = {}
                for n, (x_t, h_hat) in tb[cls].items():
                    proj_store[f"{scheme}__{src}__{cls}_n{n}"] = x_t
                    fidelity[scheme][cls][str(src)][str(n)] = B1.transport_fidelity(
                        h_hat, traj_dev, src, tgt
                    )
            for mode in MODES:
                gx2, gy = _grouped(row2, mat, mode)
                gx1b, _ = _grouped(row1b, mat, mode)
                g_ceil, _ = _grouped(ceiling, mat, mode)
                if len(gy) < 2:
                    continue  # too few conditions in this mode for a within-condition read
                # Precompute per-condition r ONCE + one shared resample sequence (seed);
                # every delta/paired/retention below is then an array-mean reduction
                # (vectorized bootstrap — bit-identical to the #779 per-draw loop).
                n_cond = len(gy)
                samps = _resample_idx(seed, n_cond, n_boot)
                pc_raw = _per_cond_r(gx2, gy)
                pc_id = _per_cond_r(gx1b, gy)
                pc_ceil = _per_cond_r(g_ceil, gy) if len(g_ceil) == n_cond else None
                for cls in TRANSPORT_CLASSES:
                    cls_ns = sorted(tb[cls].keys())
                    if anchor_n not in cls_ns:
                        continue  # no anchor for this class ⇒ cannot pair; skip (e.g. mlp)
                    pc_t = {
                        n: _per_cond_r(_grouped(tb[cls][n][0], mat, mode)[0], gy) for n in cls_ns
                    }
                    per_n = {}
                    for n in cls_ns:
                        d_raw = _delta_fast(pc_t[n], pc_raw, samps)
                        d_id = _delta_fast(pc_t[n], pc_id, samps)
                        win = bool(
                            d_raw["excludes_zero"]
                            and d_raw["delta"] > 0
                            and d_id["excludes_zero"]
                            and d_id["delta"] > 0
                        )
                        ret = (
                            _retention_fast(pc_t[n], pc_ceil, samps)
                            if pc_ceil is not None
                            else {"point": float("nan")}
                        )
                        entry = {
                            "win": win,
                            "vs_raw_source": d_raw,
                            "vs_id_transport": d_id,
                            "retention": ret,
                        }
                        if n != anchor_n:
                            entry["paired_vs_anchor"] = _paired_fast(pc_t[n], pc_t[anchor_n], samps)
                        per_n[n] = entry
                    cells_by_class[cls].append(
                        {
                            "class": cls,
                            "scheme": scheme,
                            "target_layer": tgt,
                            "source": src,
                            "mode": mode,
                            "per_n": per_n,
                        }
                    )
            logger.info(
                "[%s/%s] source=%d done (classes=%s)",
                trait,
                scheme,
                src,
                ",".join(TRANSPORT_CLASSES),
            )
    proj_store["y"] = mat["y"]
    proj_store["cond"] = mat["cond"]
    proj_store["mode_is_manyshot"] = (mat["mode"] == "many_shot").astype(np.int64)
    return {"cells_by_class": cells_by_class, "fidelity": fidelity, "proj": proj_store}


# ── curve aggregation across cells ────────────────────────────────────────────────


def aggregate_curve(all_cells, ns_scaling, anchor_n, *, bh_q, seed) -> dict:
    """win_count(n) + net/newly/dropped + mean_paired_delta(n) + BH survivors."""
    total_cells = len(all_cells)
    win_at = {n: [bool(c["per_n"][n]["win"]) for c in all_cells] for n in [anchor_n, *ns_scaling]}
    anchor_rets = np.array(
        [c["per_n"][anchor_n]["retention"].get("point", float("nan")) for c in all_cells],
        dtype=float,
    )
    anchor_rets_finite = anchor_rets[np.isfinite(anchor_rets)]
    curve: dict = {
        "total_cells": total_cells,
        "anchor_n": anchor_n,
        "win_count_anchor": int(sum(win_at[anchor_n])),
        "mean_retention_anchor": (
            float(np.mean(anchor_rets_finite)) if anchor_rets_finite.size else float("nan")
        ),
        "by_n": {},
    }
    rng = np.random.default_rng(seed)
    for n in ns_scaling:
        newly = [i for i in range(total_cells) if win_at[n][i] and not win_at[anchor_n][i]]
        dropped = [i for i in range(total_cells) if win_at[anchor_n][i] and not win_at[n][i]]
        deltas = np.array(
            [c["per_n"][n]["paired_vs_anchor"]["delta"] for c in all_cells], dtype=float
        )
        finite = deltas[np.isfinite(deltas)]
        mean_delta = float(np.mean(finite)) if finite.size else float("nan")
        # cell-level bootstrap CI of the mean paired-delta (resample cells).
        if finite.size:
            boot = [
                float(np.mean(rng.choice(finite, size=finite.size, replace=True)))
                for _ in range(1000)
            ]
            mlo, mhi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        else:
            mlo = mhi = float("nan")
        pos_sig = sum(
            1
            for c in all_cells
            if c["per_n"][n]["paired_vs_anchor"]["excludes_zero"]
            and c["per_n"][n]["paired_vs_anchor"]["delta"] > 0
        )
        neg_sig = sum(
            1
            for c in all_cells
            if c["per_n"][n]["paired_vs_anchor"]["excludes_zero"]
            and c["per_n"][n]["paired_vs_anchor"]["delta"] < 0
        )
        # BH over ALL cells' one-sided p (transported@n > anchor); newly-winning survivors.
        pvals = [c["per_n"][n]["paired_vs_anchor"]["p_one_sided"] for c in all_cells]
        survive = benjamini_hochberg(pvals, q=bh_q)
        bh_newly = sum(1 for i in newly if survive[i])
        # DV3 retention(n): mean over cells of the joint-bootstrap retention point.
        rets = np.array(
            [c["per_n"][n]["retention"].get("point", float("nan")) for c in all_cells], dtype=float
        )
        rets_finite = rets[np.isfinite(rets)]
        mean_ret = float(np.mean(rets_finite)) if rets_finite.size else float("nan")
        curve["by_n"][str(n)] = {
            "win_count": int(sum(win_at[n])),
            "newly_winning": len(newly),
            "dropped_out": len(dropped),
            "net_win_vs_anchor": len(newly) - len(dropped),
            "mean_paired_delta": mean_delta,
            "mean_paired_delta_ci": [mlo, mhi],
            "cells_delta_pos_sig": pos_sig,
            "cells_delta_neg_sig": neg_sig,
            "bh_survivors_total": int(sum(survive)),
            "bh_newly_winning_survivors": bh_newly,
            "bh_q": bh_q,
            "chance_expectation": bh_q * total_cells,
            "mean_retention": mean_ret,
        }
    return curve


# ── inputs (real HF vs synthetic smoke) ──────────────────────────────────────────


def _synthetic_inputs(hidden, ns_all, anchor_n):
    """Fabricate a tiny 1-trait eval matrix + r_B + per-n ridge/MLP maps + fit_pool (CPU, no HF)."""
    rng = np.random.default_rng(0)
    n_q, n_layers = 60, C.EXPECTED_LAYERS
    traj = rng.standard_normal((n_q, n_layers, hidden)).astype(np.float32)
    y = rng.standard_normal(n_q).astype(np.float64) * 20 + 50
    cond = rng.integers(0, 6, size=n_q)
    mode = np.array(["system" if i % 2 else "many_shot" for i in range(n_q)], dtype=object)
    mat = {
        "traj": traj,
        "y": y,
        "cond": cond,
        "mode": mode,
        "cond_ids": [f"c{i}" for i in range(6)],
        "layers": list(range(n_layers)),
    }
    r_b = rng.standard_normal((n_layers, hidden)).astype(np.float32)
    ridge_maps_by_n, mlp_maps_by_n = {}, {}
    hid_mlp = 8
    for n in ns_all:
        ridge_maps_by_n[n] = {
            t: MP.RidgeMap(
                mu=torch.zeros(hidden),
                sd=torch.ones(hidden),
                w=torch.from_numpy(
                    (rng.standard_normal((hidden, hidden)) * 0.01).astype(np.float32)
                ),
                bias=torch.zeros(hidden),
                best_lam=1.0,
                sigma=1.0,
            )
            for t in range(n_layers - 1)
        }
        mlp_maps_by_n[n] = {
            t: MP.MLPMap(
                W1=torch.from_numpy(
                    (rng.standard_normal((hid_mlp, hidden)) * 0.01).astype(np.float32)
                ),
                b1=torch.zeros(hid_mlp),
                W2=torch.from_numpy(
                    (rng.standard_normal((hidden, hid_mlp)) * 0.01).astype(np.float32)
                ),
                b2=torch.zeros(hidden),
                mu=torch.zeros(hidden),
                sd=torch.ones(hidden),
                sigma=1.0,
            )
            for t in range(n_layers - 1)
        }
    fit_pool = rng.standard_normal((max(ns_all), n_layers, hidden)).astype(np.float32)
    schemes = {"primary": 20}  # a single target with a real source grid
    return {"evil": (mat, r_b)}, ridge_maps_by_n, mlp_maps_by_n, fit_pool, schemes


def _fetch_map(maps_dir, fname, hf_bucket, device, loader):
    """Local-first → HF-fetch a per-n map file; return the loaded maps dict.

    The HF-fetch leg is overflow-aware (map .pt files are rerouted to the private
    overflow repo when the public LFS quota is full, #541); same-instance reads hit
    the local file and never touch HF.
    """
    path = maps_dir / fname
    if not path.exists():
        logger.info("[maps] local %s absent; fetching %s/%s", path, hf_bucket, fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        local = S.hf_download_pt_maybe_overflow(C.HF_DATA_REPO, hf_bucket, fname)
        if not path.exists():
            path.symlink_to(Path(local).resolve())
    return loader(path, device)


def _stage0_mlp_ns(out_dir, ns_all, anchor_n):
    """Which n-points have persisted MLP maps (from stage0_scaling.json, else the default)."""
    p = out_dir / "stage0_scaling.json"
    if p.exists():
        with open(p) as f:
            mns = json.load(f).get("mlp_ns")
        if mns:
            return [n for n in mns if n in ns_all]
    return sorted({anchor_n, ns_all[-1]})  # stage0's default mlp_ns


def _real_inputs(traits, maps_dir, out_dir, capture_dir, ns_all, anchor_n, device, smoke):
    """Load per-n ridge + MLP maps (local-or-HF), the fit_pool, and each trait's mat + r_B."""
    ridge_maps_by_n = {
        n: _fetch_map(
            maps_dir, f"ridge_maps_n{n}.pt", S.hf_ridge_maps_bucket(n), device, S.load_ridge_maps
        )
        for n in ns_all
    }
    mlp_ns = _stage0_mlp_ns(out_dir, ns_all, anchor_n)
    mlp_maps_by_n = {
        n: _fetch_map(
            maps_dir, f"mlp_maps_n{n}.pt", S.hf_mlp_maps_bucket(n), device, S.load_mlp_maps
        )
        for n in mlp_ns
    }
    # fit_pool for the direct-hop refit (parent bundle + new capture, nested by row).
    cap = S.load_capture_local_or_hf(capture_dir)
    fit_pool = S.build_scaling_bundle(C.load_pass_b(), cap)["fit_pool"]
    trait_inputs = {}
    for trait in traits:
        mat = C.build_eval_traj_matrix(C.load_eval_cells(trait))
        trait_inputs[trait] = (mat, C.load_rb(trait))
    # realized forward precision (Fold 1): prefer the realized field, fall back to the
    # capture_dtype alias for a legacy/synthetic manifest that predates the split.
    realized_dtype = cap.get("realized_capture_dtype", cap.get("capture_dtype"))
    logger.info(
        "[inputs] ridge n=%s mlp n=%s fit_pool=%d capture_dtype=%s",
        ns_all,
        mlp_ns,
        fit_pool.shape[0],
        realized_dtype,
    )
    return trait_inputs, ridge_maps_by_n, mlp_maps_by_n, fit_pool, realized_dtype


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #841 Stage-1 transport-scaling curve.")
    ap.add_argument("--traits", nargs="+", default=list(C.TRAITS))
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--ns", default="", help="comma-list of fit sizes incl. anchor (default SCALING_NS)"
    )
    ap.add_argument("--anchor-n", type=int, default=S.N_ANCHOR_FIT)
    ap.add_argument("--maps-dir", type=Path, default=S.RIDGE_MAPS_DIR)
    ap.add_argument("--out-dir", type=Path, default=S.EVAL_SCALING_DIR)
    ap.add_argument("--capture-dir", type=Path, default=S.CAPTURE_DIR)
    ap.add_argument(
        "--dual-max",
        type=int,
        default=10000,
        help="direct-hop refit uses the dual solver at n≤this, primal above",
    )
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--bh-q", type=float, default=0.05)
    ap.add_argument("--synthetic", action="store_true", help="smoke: fabricate tiny inputs (no HF)")
    ap.add_argument("--synthetic-hidden", type=int, default=C.EXPECTED_HIDDEN)
    ap.add_argument(
        "--smoke", action="store_true", help="1 trait, coarse source grid, small n-boot"
    )
    args = ap.parse_args()

    _assert_fast_matches_reference()  # gate the vectorized bootstrap == #779 before any work
    device = _resolve_device(args.device)
    ns_all = [int(x) for x in args.ns.split(",") if x] or list(S.SCALING_NS)
    anchor_n = args.anchor_n
    assert anchor_n in ns_all, f"anchor_n={anchor_n} must be in ns={ns_all}"
    traits = ["evil"] if (args.smoke or args.synthetic) else args.traits
    logger.info(
        "device=%s ns=%s anchor=%d traits=%s n_boot=%d",
        device,
        ns_all,
        anchor_n,
        traits,
        args.n_boot,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        trait_inputs, ridge_maps_by_n, mlp_maps_by_n, fit_pool, fixed_schemes = _synthetic_inputs(
            args.synthetic_hidden, ns_all, anchor_n
        )
        capture_dtype = "synthetic"
    else:
        trait_inputs, ridge_maps_by_n, mlp_maps_by_n, fit_pool, capture_dtype = _real_inputs(
            traits,
            args.maps_dir,
            args.out_dir,
            args.capture_dir,
            ns_all,
            anchor_n,
            device,
            args.smoke,
        )
        fixed_schemes = None

    def _curve(cells):
        """aggregate_curve over one class's cells, deriving that class's scaling n's."""
        cls_ns = sorted({n for c in cells for n in c["per_n"]} - {anchor_n})
        return {
            "n_cells": len(cells),
            "curve": aggregate_curve(
                cells, cls_ns, anchor_n, bh_q=args.bh_q, seed=C.BOOTSTRAP_SEED
            ),
        }

    cells_by_class_all: dict[str, list[dict]] = {c: [] for c in TRANSPORT_CLASSES}
    fidelity_all: dict = {"traits": {}}
    proj_all: dict = {}
    result: dict = {
        "ns": ns_all,
        "anchor_n": anchor_n,
        "n_boot": args.n_boot,
        "classes": list(TRANSPORT_CLASSES),
        "capture_dtype": capture_dtype,  # realized capture precision (Fold 1)
        "target_layers": {"primary": C.PRIMARY_TARGET_LAYER, "companion": C.COMPANION_TARGET_LAYER},
        "traits": {},
        "metadata": C.reproducibility_metadata(
            {"phase": "stage1_scaling", "smoke": args.smoke, "capture_dtype": capture_dtype}
        ),
    }

    for trait, (mat, r_b) in trait_inputs.items():
        schemes = fixed_schemes if fixed_schemes is not None else B1.schemes_for(trait, args.smoke)
        out = process_trait(
            trait,
            mat,
            r_b,
            ridge_maps_by_n,
            mlp_maps_by_n,
            fit_pool,
            schemes,
            ns_all,
            anchor_n,
            n_boot=args.n_boot,
            seed=C.BOOTSTRAP_SEED,
            device=device,
            dual_max=args.dual_max,
            smoke=args.smoke,
        )
        curve_by_class = {}
        for cls, cells in out["cells_by_class"].items():
            if not cells:
                continue
            curve_by_class[cls] = _curve(cells)
            cells_by_class_all[cls].extend(cells)
        result["traits"][trait] = {"curve_by_class": curve_by_class}
        fidelity_all["traits"][trait] = out["fidelity"]
        for k, v in out["proj"].items():
            proj_all[f"{trait}__{k}"] = v
        C.write_json_atomic(args.out_dir / "stage1_scaling.json", result)  # checkpoint per trait

    # Pooled curve across ALL traits' cells, PER CLASS (the ~136-cell headline per class).
    result["pooled_curve_by_class"] = {
        cls: _curve(cells) for cls, cells in cells_by_class_all.items() if cells
    }
    C.write_json_atomic(args.out_dir / "stage1_scaling.json", result)
    C.write_json_atomic(args.out_dir / "transport_fidelity_scaling.json", fidelity_all)
    np.savez(args.out_dir / "scaling_projections.npz", **proj_all)
    ridge_pooled = result["pooled_curve_by_class"].get("ridge", {}).get("curve", {})
    logger.info(
        "[done] ridge pooled win_count(anchor)=%s over %s cells → %s",
        ridge_pooled.get("win_count_anchor"),
        ridge_pooled.get("total_cells"),
        args.out_dir / "stage1_scaling.json",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
