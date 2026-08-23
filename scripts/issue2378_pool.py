"""Issue #2378 P6 — pooled-tier arm (unit 4 deliverable 1; plan §4.4 / H5).

ONE map fit jointly on all 11 active cells' equalized train folds (plan v7
— dialogue descoped; fold ids aligned
by INDEX across cells — pooled train for fold f = the union over cells of
their fold!=f rows under each cell's OWN registered fold structure), then
per-cell scoring at the registered tiers, each against the cell's OWN ceiling
from the unit-3 fits JSONs:

  m0          pooled map as-is
  m1          m0 + per-cell BIAS  b_cell = mean(y_tr) - m0(mean(x_tr))
              (exact per-cell LS intercept under the frozen pooled slope —
              m0 is affine, so mean-of-preds == pred-of-mean)
  m2_k{8,32,128}  m1 + per-cell rank-k residual slope correction: GCV ridge
              from the cell's train-fold center-only PCA-k input basis to the
              M1 residuals (rank <= k by construction; LINEAR throughout)
  identity_cell / identity_global   the mandatory identity+learned-bias
              baselines (d_in == d_out == 5,120)

Conventions REUSED from ``scripts/issue2054_pool_specialize.py`` (plan §4.4
"reuse the pool_specialize / pool_rungs conventions"): ``PooledMomentRidge``
(streamed second-moment GCV ridge — the pooled train matrix is never
materialized) + ``_pca_topk`` (named degeneracies ``constant_x`` /
``rank_deficient_topk``, recorded never clamped) are IMPORTED; the parity
gate re-targets ``issue2054_fits._ridge_gcv_fit_predict`` (this issue's fit
core — same fit_h conventions: population-sd standardize-X, center-Y,
logspace(-2,4,13) grid, GCV dof cap 0.9), asserting exact lambda match +
rel 1e-6 predictions on a materializable chat subset.

Scoring conventions are the unit-3 ones (``issue2378_p6_common``): pooled-mean
ss_tot per cell, per-fold + pooled R², 200-draw row-grain recovery bootstrap
with the registered skip-and-count guard, tier suppression for cells whose
own-fit tier is not clearly-mappable (ratio verdicts suppressed; absolute R²
reported). Cells have DISJOINT cohorts here (unlike #2054's shared
conversation population), so the bootstrap is per-cell independent row
resample — no cross-cell coupling machinery. Recovery > 1 is FLAGGED as the
estimation-limited-ceiling tell (plan §6 analyzer note), never narrated as
super-recovery.

M2 residual-fit nulls: fit-side shuffled-pair (permute train residual rows)
via the batched ``issue2054_fits._shuffled_answer_null_r2`` core on the
reduced basis, 100 draws (the #2378-wide null-draw convention; the parent's
20-draw + escalation dance is unnecessary at this per-draw cost).

H5 (plan §3): "pooled + per-cell bias reaches >= 90% of the own ceiling in
most cells"; "most cells" = STRICT MAJORITY of surviving CLEARLY-MAPPABLE
cells at the +bias (m1) tier, point recovery >= 0.90; cells not
clearly-mappable (or ratio-suppressed by skip-and-count) are reported but
EXCLUDED from the denominator (disclosed). Definition persisted in
``pool/h5_summary.json``.

Pooled-fit RSS is analytically bounded (plan §4.6 smoke item 3: d x d Grams
per fold + one cell's arrays — the 85k x 5120 pooled matrix never exists);
the bound is ASSERTED against available RAM at phase entry.

Phases: ``--phase pool`` (default), ``--phase h5`` (recompute the summary
from existing per-cell JSONs), ``--phase probe`` (synthetic CPU
self-verification: planted per-cell bias recovered by m1, planted rank-1
slope recovered by m2, scrambled cell tier-suppressed + excluded from H5,
moment-vs-SVD parity, resume-skip).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue2054_fits as pf  # noqa: E402  (fit/null cores — plan §10 reuse row)
import issue2054_pool_specialize as ps  # noqa: E402  (PooledMomentRidge + _pca_topk)
import issue2378_common as cm  # noqa: E402
import issue2378_fits as fits_mod  # noqa: E402  (resolve_layer + probe-store reuse)
import issue2378_p6_common as p6  # noqa: E402
from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)

SCRIPT_VERSION = "issue2378_pool_v1"

DEFAULT_RANKS = (8, 32, 128)
H5_RECOVERY_BAR = 0.90
H5_TIER = "m1"
PARITY_REL_TOL = 1e-6
RAM_SLACK_FACTOR = 1.5
RAM_FIXED_OVERHEAD_GIB = 2.0  # interpreter + torch + fragmentation headroom


def _log(msg: str) -> None:
    print(msg, flush=True)


def _model_names(ranks: list[int]) -> list[str]:
    return ["m0", "m1", *[f"m2_k{r}" for r in ranks], "identity_cell", "identity_global"]


# ---------------------------------------------------------------------------
# RAM assert (plan §4.6 smoke blind-spot item 3)
# ---------------------------------------------------------------------------


def available_ram_bytes() -> tuple[int, str]:
    """min(MemAvailable, cgroup limit - usage) with a source label.

    cgroup v2 (``/sys/fs/cgroup/memory.max``) then v1 fallback; an absent or
    ``max`` limit degrades to /proc/meminfo alone. Containers (RunPod CPU
    pods) report the HOST in /proc/meminfo, so the cgroup read is the binding
    one there.
    """
    mem_available = None
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").split("\n"):
        if line.startswith("MemAvailable:"):
            mem_available = int(line.split()[1]) * 1024
            break
    if mem_available is None:
        raise RuntimeError("cannot read MemAvailable from /proc/meminfo")
    candidates = [(mem_available, "meminfo")]
    try:
        v2_max = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if v2_max != "max":
            cur = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            candidates.append((int(v2_max) - cur, "cgroup-v2"))
    except (OSError, ValueError):
        try:
            lim = int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text().strip())
            cur = int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text().strip())
            if lim < 1 << 60:  # v1 "unlimited" sentinel is a huge number
                candidates.append((lim - cur, "cgroup-v1"))
        except (OSError, ValueError):
            pass
    return min(candidates, key=lambda t: t[0])


def pooled_ram_bound_bytes(k: int, d: int, n_max: int) -> int:
    """Analytic peak-RSS bound for the moment-space pooled fit (plan §4.6):
    per-fold Grams (c_xx + c_xy) + retained per-fold maps + one eigh workspace
    + the largest single cell's float64 arrays with copy headroom."""
    moments = k * 2 * d * d * 8
    models = k * d * d * 8
    eigh_ws = 3 * d * d * 8
    cell = 2 * (2 * n_max * d * 8)  # X + Y float64, x2 transient-copy headroom
    return int(RAM_SLACK_FACTOR * (moments + models + eigh_ws + cell))


def assert_pool_ram(k: int, d: int, n_max: int) -> dict:
    bound = pooled_ram_bound_bytes(k, d, n_max)
    avail, source = available_ram_bytes()
    need = bound + int(RAM_FIXED_OVERHEAD_GIB * 2**30)
    rec = {
        "bound_gib": round(bound / 2**30, 2),
        "fixed_overhead_gib": RAM_FIXED_OVERHEAD_GIB,
        "available_gib": round(avail / 2**30, 2),
        "available_source": source,
        "k": k,
        "d": d,
        "n_max": n_max,
    }
    _log(f"[pool] ram check: {json.dumps(rec)}")
    if avail < need:
        raise RuntimeError(
            f"pooled-fit RAM assert FAILED (plan §4.6 item 3): available {avail / 2**30:.1f} GiB "
            f"({source}) < analytic bound {need / 2**30:.1f} GiB — route to a bigger pod"
        )
    return rec


# ---------------------------------------------------------------------------
# Moment accumulation over the equalized fold-map cohorts (streamed per cell)
# ---------------------------------------------------------------------------


def accumulate_moments(store_root: Path, fold_map: dict, arm: str, layer: int, device: str) -> dict:
    """One streaming pass: per-fold second moments over ALL cells' equalized
    cohorts (fold ids aligned by index). The pooled train matrix is never
    materialized (plan §4.6: d x d Grams only)."""
    k = int(fold_map["k"])
    dev = torch.device(device)
    d = p6.EXPECTED_HIDDEN if fold_map.get("_probe_d") is None else int(fold_map["_probe_d"])
    slot = p6.SLOT_BY_ARM[arm]
    mom = [
        {
            "n": 0,
            "sum_x": torch.zeros(d, dtype=torch.float64, device=dev),
            "sum_y": torch.zeros(d, dtype=torch.float64, device=dev),
            "yss": 0.0,
            "c_xx": torch.zeros(d, d, dtype=torch.float64, device=dev),
            "c_xy": torch.zeros(d, d, dtype=torch.float64, device=dev),
        }
        for _ in range(k)
    ]
    n_per_cell: dict[str, int] = {}
    cells = sorted(fold_map["cells"])
    for ci, cell in enumerate(cells):
        t0 = time.time()
        entry = fold_map["cells"][cell]
        pack = p6.load_cell_arrays(
            store_root, cell, layer, (slot, p6.ANSWER_SLOT), row_order=entry["row_ids"]
        )
        x = torch.as_tensor(pack["arrays"][slot].astype(np.float64), device=dev)
        y = torch.as_tensor(pack["arrays"][p6.ANSWER_SLOT].astype(np.float64), device=dev)
        if x.shape[1] != d:
            raise RuntimeError(f"{cell}: hidden dim {x.shape[1]} != expected {d}")
        folds = np.asarray(entry["folds"], dtype=np.int64)
        for f in range(k):
            idx = np.flatnonzero(folds == f)
            xf, yf = x[idx], y[idx]
            m = mom[f]
            m["n"] += int(xf.shape[0])
            m["sum_x"] += xf.sum(0)
            m["sum_y"] += yf.sum(0)
            m["yss"] += float((yf * yf).sum())
            m["c_xx"] += xf.T @ xf
            m["c_xy"] += xf.T @ yf
        n_per_cell[cell] = int(entry["n_rows"])
        del pack, x, y
        _log(
            f"[pool] moments cell {ci + 1}/{len(cells)} {cell} "
            f"n={n_per_cell[cell]} elapsed={time.time() - t0:.1f}s"
        )
    return {"mom": mom, "n_per_cell": n_per_cell}


def fit_pooled_per_fold(mom: list[dict], k: int) -> dict[int, ps.PooledMomentRidge]:
    """Train moments for fold f = totals minus fold f (the pool_specialize
    pattern); the fit core is PooledMomentRidge at THIS issue's grid/cap
    (identical values to the ctx2ctx defaults — asserted at import below)."""
    models: dict[int, ps.PooledMomentRidge] = {}
    for f in range(k):
        t0 = time.time()
        train: dict = {
            "n": sum(mom[g]["n"] for g in range(k) if g != f),
            "yss": sum(mom[g]["yss"] for g in range(k) if g != f),
        }
        for key in ("sum_x", "sum_y", "c_xx", "c_xy"):
            train[key] = sum(mom[g][key] for g in range(k) if g != f)
        models[f] = ps.PooledMomentRidge(**train, lambdas=pf.DEFAULT_LAMBDAS, dof_cap=0.9)
        _log(
            f"[pool] pooled fold {f}: n_train={models[f].n_train:,} "
            f"lam={models[f].best_lambda:g} dof={models[f].dof:.0f} "
            f"elapsed={time.time() - t0:.1f}s"
        )
    return models


def parity_gate(
    store_root: Path, fold_map: dict, arm: str, layer: int, device: str, parity_n: int
) -> dict:
    """Moment-path PooledMomentRidge vs the unit-3 fit core
    ``pf._ridge_gcv_fit_predict`` on a materializable chat subset (fold-0
    train complement subsampled to ``parity_n`` rows, capped-eval fold-0
    rows): exact lambda match + rel {tol} predictions. Both sides share the
    fit_h conventions (population-sd standardize-X, center-Y, GCV dof cap),
    so a mismatch is a port defect, never expected numerics.
    """
    entry = fold_map["cells"]["chat"]
    slot = p6.SLOT_BY_ARM[arm]
    pack = p6.load_cell_arrays(
        store_root, "chat", layer, (slot, p6.ANSWER_SLOT), row_order=entry["row_ids"]
    )
    x_all = pack["arrays"][slot].astype(np.float64)
    y_all = pack["arrays"][p6.ANSWER_SLOT].astype(np.float64)
    folds = np.asarray(entry["folds"], dtype=np.int64)
    tr = np.flatnonzero(folds != 0)
    te = np.flatnonzero(folds == 0)
    d = x_all.shape[1]
    rng = np.random.default_rng(p6.unit_seed("pool", arm, "parity"))
    n_tr = int(tr.size) if parity_n <= 0 else min(int(tr.size), max(parity_n, d + 1))
    tr = tr[np.sort(rng.permutation(tr.size)[:n_tr])]
    te = te[: min(1000, te.size)]
    if n_tr <= d:
        raise RuntimeError(f"parity gate needs n_train > d: {n_tr} <= {d} (raise --parity-n)")
    x_tr, y_tr, x_te = x_all[tr], y_all[tr], x_all[te]
    dev = torch.device(device)
    xt = torch.as_tensor(x_tr, device=dev)
    yt = torch.as_tensor(y_tr, device=dev)
    mine = ps.PooledMomentRidge(
        n=int(xt.shape[0]),
        sum_x=xt.sum(0),
        sum_y=yt.sum(0),
        yss=float((yt * yt).sum()),
        c_xx=xt.T @ xt,
        c_xy=xt.T @ yt,
        lambdas=pf.DEFAULT_LAMBDAS,
        dof_cap=0.9,
    )
    preds_ref, info_ref = pf._ridge_gcv_fit_predict(
        x_tr, y_tr, x_te, lambdas=pf.DEFAULT_LAMBDAS, dof_cap=0.9
    )
    preds_mine = mine.predict_np(x_te)
    scale = float(np.abs(preds_ref).max()) + 1e-12
    max_rel = float(np.abs(preds_mine - preds_ref).max() / scale)
    if mine.best_lambda != info_ref["best_lambda"] or max_rel > PARITY_REL_TOL:
        raise RuntimeError(
            f"pooled-moment parity FAIL vs _ridge_gcv_fit_predict (arm={arm}): "
            f"lambda {mine.best_lambda} vs {info_ref['best_lambda']}, max_rel={max_rel:.3e}"
        )
    rec = {
        "n_train": int(n_tr),
        "n_eval": int(te.size),
        "best_lambda": mine.best_lambda,
        "max_rel": max_rel,
        "tol": PARITY_REL_TOL,
        "reference": "issue2054_fits._ridge_gcv_fit_predict",
    }
    _log(f"[pool] parity gate arm={arm}: {json.dumps(rec)} OK")
    return rec


# ---------------------------------------------------------------------------
# Per-cell tier scoring
# ---------------------------------------------------------------------------


def _fits_inputs(ledger_root: Path, cell: str, arm: str) -> tuple[dict, dict]:
    """The cell's own-map fits JSON + rowstats (ceiling + floor + tier).
    REQUIRED — the fits unit class runs first (plan §9 P6 sequencing)."""
    fpath = ledger_root / "fits" / f"{cell}__{arm}.json"
    if not fpath.exists():
        raise RuntimeError(
            f"missing {fpath} — run issue2378_fits.py for cell {cell} BEFORE the pooled arm "
            "(own ceilings are the recovery denominators)"
        )
    fits = json.loads(fpath.read_text(encoding="utf-8"))
    rs = p6.load_rowstats(ledger_root / "fits" / "percell" / f"{cell}__{arm}__rowstats.npz")
    return fits, rs


def _knn_block(preds: np.ndarray, true: np.ndarray) -> dict:
    return {
        metric: knn_retrieval(preds, true, ks=(1, 5, 10), metric=metric)
        for metric in ("euclidean", "cosine")
    }


def run_cell_unit(
    args,
    fold_map: dict,
    models: dict[int, ps.PooledMomentRidge],
    cell: str,
    arm: str,
    layer: int,
    regime: dict,
) -> None:
    ledger_root = Path(args.ledger_root)
    out_path = ledger_root / "pool" / f"{cell}__{arm}.json"
    if out_path.exists():
        prior = json.loads(out_path.read_text(encoding="utf-8"))
        if prior.get("regime") == regime:
            _log(f"[pool] SKIP {cell}/{arm}: output exists with matching regime")
            return
        raise RuntimeError(f"regime mismatch at {out_path} — use a fresh ledger root")
    t_unit = time.time()
    entry = fold_map["cells"][cell]
    ranks = [int(r) for r in args.ranks]
    k_max = max(ranks)
    names = _model_names(ranks)
    fits, fits_rs = _fits_inputs(ledger_root, cell, arm)
    if fits_rs["row_ids"].tolist() != entry["row_ids"]:
        raise RuntimeError(f"{cell}: fits rowstats row order != fold map (mixed generations)")
    slot = p6.SLOT_BY_ARM[arm]
    pack = p6.load_cell_arrays(
        Path(args.store_root), cell, layer, (slot, p6.ANSWER_SLOT), row_order=entry["row_ids"]
    )
    x_all = pack["arrays"][slot].astype(np.float64)
    y_all = pack["arrays"][p6.ANSWER_SLOT].astype(np.float64)
    del pack
    n = x_all.shape[0]
    ybar = y_all.mean(axis=0)
    ss_tot = ((y_all - ybar) ** 2).sum(axis=1)
    if not np.allclose(ss_tot, fits_rs["ss_tot"], rtol=1e-8, atol=1e-6):
        raise RuntimeError(f"{cell}: recomputed ss_tot != fits rowstats ss_tot (store drift)")
    splits = p6.fold_splits(entry)
    ss_res = {name: np.full(n, np.nan) for name in names}
    per_fold: list[dict] = []
    for f, (tr, te) in enumerate(splits):
        t0 = time.time()
        m0 = models[f]
        x_tr, y_tr, x_te, y_te = x_all[tr], y_all[tr], x_all[te], y_all[te]
        preds: dict[str, np.ndarray] = {}
        preds["m0"] = m0.predict_np(x_te)
        b_cell = y_tr.mean(axis=0) - m0.predict_np(x_tr.mean(axis=0, keepdims=True))[0]
        preds["m1"] = preds["m0"] + b_cell
        preds["identity_cell"] = identity_bias_predict(x_tr, y_tr, x_te)
        preds["identity_global"] = x_te + m0.global_bias

        # m2: per-cell low-rank slope correction on the m1 residuals.
        z_tr = m0.predict_np(x_tr) + b_cell
        r_tr = y_tr - z_tr
        r_te = y_te - preds["m1"]
        pca = ps._pca_topk(x_tr, k_max)
        degenerate_reason = pca if isinstance(pca, str) else None
        m2_recs: dict[str, dict] = {}
        if degenerate_reason is not None:
            _log(
                f"[pool] {cell}/{arm} fold={f} m2 SKIPPED ({degenerate_reason}) — "
                f"cannot supply a rank-{k_max} basis; m1 substituted (named, recorded)"
            )
            for r in ranks:
                preds[f"m2_k{r}"] = preds["m1"]
                m2_recs[f"m2_k{r}"] = {"skipped": degenerate_reason}
        else:
            mu_p, comps = pca
            xr_tr = (x_tr - mu_p) @ comps
            xr_te = (x_te - mu_p) @ comps
            for r in ranks:
                corr, info = pf._ridge_gcv_fit_predict(
                    xr_tr[:, :r], r_tr, xr_te[:, :r], lambdas=pf.DEFAULT_LAMBDAS, dof_cap=0.9
                )
                preds[f"m2_k{r}"] = preds["m1"] + corr
                resid_r2 = pf._r2_matrix(r_te, corr)
                null_r2s, _null_info = pf._shuffled_answer_null_r2(
                    xr_tr[:, :r],
                    r_tr,
                    xr_te[:, :r],
                    r_te,
                    n_draws=args.n_null_draws,
                    seed=p6.unit_seed(cell, arm, "poolnull", f, r),
                )
                m2_recs[f"m2_k{r}"] = {
                    "rank": r,
                    "residual_fit_r2": float(resid_r2),
                    "ridge_info": info,
                    "null": {
                        "kind": "shuffled-pair fit-side on the reduced basis "
                        "(permute train residual rows)",
                        "n_draws": int(len(null_r2s)),
                        "draws": [float(x) for x in null_r2s],
                        "p95": float(np.percentile(null_r2s, 95)),
                        "p_value_residual_fit": float(
                            (1 + (np.asarray(null_r2s) >= resid_r2).sum()) / (1 + len(null_r2s))
                        ),
                    },
                }
        for name in names:
            ss_res[name][te] = ((y_te - preds[name]) ** 2).sum(axis=1)
        metrics = {name: float(pf._r2_matrix(y_te, preds[name])) for name in names}
        # kNN companion for EVERY fitted tier incl. intermediate pooled rungs
        # (r1 review g4 concern 1: plan §6 mapping-baselines row requires the
        # retrieval read per fitted map, and no preds sidecars persist here).
        knn = {name: _knn_block(preds[name], y_te) for name in names}
        per_fold.append(
            {
                "fold": f,
                "n_pooled_train": m0.n_train,
                "n_cell_train": int(tr.size),
                "n_test": int(te.size),
                "pooled_info": m0.info(),
                "m1_bias_norm": float(np.linalg.norm(b_cell)),
                "degenerate_reason": degenerate_reason,
                "r2": metrics,
                "m2": m2_recs,
                "knn": knn,
                "wall_s": round(time.time() - t0, 1),
            }
        )
        _log(
            f"[pool] {cell}/{arm} fold={f} m0={metrics['m0']:+.4f} m1={metrics['m1']:+.4f} "
            f"m2k{k_max}={metrics[f'm2_k{k_max}']:+.4f} "
            f"idcell={metrics['identity_cell']:+.4f} elapsed={per_fold[-1]['wall_s']}s"
        )

    fold_mean = {name: float(np.mean([fr["r2"][name] for fr in per_fold])) for name in names}
    pooled = {name: p6.pooled_r2(ss_res[name], ss_tot) for name in names}
    ceiling = {
        "fits_json": str(ledger_root / "fits" / f"{cell}__{arm}.json"),
        "pooled_r2": float(fits["pooled_r2"]),
        "fold_mean_r2": float(fits["fold_mean_r2"]),
        "floor": float(fits["floor"]),
        "tier": fits["tier"],
    }
    suppress_by_tier = fits["tier"] != "clearly-mappable"
    recovery: dict[str, dict] = {}
    for name in names:
        if suppress_by_tier:
            recovery[name] = {
                "suppressed_by_tier": True,
                "tier": fits["tier"],
                "note": (
                    "ratio verdicts suppressed (plan §3 reporting tiers): own-ceiling R2 and "
                    "pooled-tier R2 are reported separately, absolute and unnormalized"
                ),
            }
            continue
        rec = p6.recovery_bootstrap(
            ss_res[name],
            fits_rs["ss_res"],
            ss_tot,
            floor=ceiling["floor"],
            n_draws=args.bootstrap_draws,
            seed=p6.unit_seed(cell, arm, "poolrecovery", name),
        )
        rec["suppressed_by_tier"] = False
        # r1 review g4 concern 2 (mirrors the ladder fix): a draws-suppressed
        # ratio verdict (skip-and-count guard) exposes NO quotable point ratio
        # either — point ratios + exceeds_one only on unsuppressed verdicts.
        if not rec.get("suppressed"):
            rec["point_pooled"] = pooled[name] / ceiling["pooled_r2"]
            rec["point_fold_mean"] = fold_mean[name] / ceiling["fold_mean_r2"]
            exceeds = bool(rec["point_pooled"] > 1.0) or bool(rec.get("median", 0.0) > 1.0)
            rec["exceeds_one"] = exceeds
            if exceeds:
                rec["exceeds_one_note"] = (
                    "recovery > 1 is the estimation-limited-ceiling tell (plan §6 analyzer "
                    "note) — never super-recovery"
                )
        recovery[name] = rec
    h5_rec = recovery.get(H5_TIER, {})
    h5_eligible = (not suppress_by_tier) and not h5_rec.get("suppressed", False)
    payload = {
        "regime": regime,
        "cell": cell,
        "arm": arm,
        "fold_structure": entry["fold_structure"],
        "headline_fold_label": (
            "family-held-out"
            if entry["fold_structure"] == "family-held-out"
            else "conversation-grouped"
        ),
        "n_rows": int(entry["n_rows"]),
        "n_eq": int(fold_map["n_eq"]),
        "below_n_eq": bool(entry.get("below_n_eq", False)),
        "tiers": names,
        "per_fold": per_fold,
        "fold_mean_r2": fold_mean,
        "pooled_r2": pooled,
        "increments": {
            "m1_minus_m0": pooled["m1"] - pooled["m0"],
            **{f"m2_k{r}_minus_m1": pooled[f"m2_k{r}"] - pooled["m1"] for r in ranks},
        },
        "ceiling": ceiling,
        "recovery": recovery,
        "h5": {
            "tier_used": H5_TIER,
            "bar": H5_RECOVERY_BAR,
            "eligible": bool(h5_eligible),
            "point_recovery": h5_rec.get("point_pooled"),
            "ge_bar": (
                bool(h5_rec["point_pooled"] >= H5_RECOVERY_BAR)
                if h5_eligible and h5_rec.get("point_pooled") is not None
                else None
            ),
            "excluded_reason": (
                None
                if h5_eligible
                else (
                    f"own-fit tier {fits['tier']}"
                    if suppress_by_tier
                    else "ratio suppressed by skip-and-count"
                )
            ),
        },
        "unit_wall_s": round(time.time() - t_unit, 2),
        "metadata": cm.run_metadata(),
    }
    if cell in cm.STORY_CELLS:
        payload["story_fold_audit"] = entry["story_fold_audit"]
    cm.atomic_write_json(out_path, payload)
    _log(
        f"[pool] {cell}/{arm}: m0={pooled['m0']:+.4f} m1={pooled['m1']:+.4f} "
        f"tier={fits['tier']} h5_eligible={h5_eligible} "
        f"wall={payload['unit_wall_s']}s -> {out_path}"
    )


# ---------------------------------------------------------------------------
# H5 summary
# ---------------------------------------------------------------------------

H5_DEFINITION = (
    "H5 field (plan §3): pooled-map + per-cell bias (tier m1) reaches >= 0.90 of the own "
    "ceiling in a STRICT MAJORITY of surviving clearly-mappable cells; the denominator is "
    "the cells whose own-fit tier is clearly-mappable AND whose m1 recovery ratio is not "
    "suppressed by the skip-and-count guard; boundary-indeterminate / unmappable / "
    "suppressed cells are reported but excluded (disclosed here). Recovery is the POINT "
    "pooled-R2 ratio (the #2054 convention); the 200-draw CI rides each cell JSON."
)


def compose_h5_summary(ledger_root: Path, arm: str, cells: list[str], regime: dict) -> dict:
    rows = []
    for cell in cells:
        path = ledger_root / "pool" / f"{cell}__{arm}.json"
        if not path.exists():
            raise RuntimeError(f"h5 summary: missing {path} — run --phase pool first")
        d = json.loads(path.read_text(encoding="utf-8"))
        # r1 review g4 concern 3: a stale per-cell JSON from a prior regime
        # (different layer/arm/fold map) must fail loud, not silently mix.
        if d["regime"] != regime:
            raise RuntimeError(
                f"h5 summary: regime mismatch for {path} — cell JSON regime differs from the "
                "current run's regime; re-run --phase pool for this cell"
            )
        h5 = d["h5"]
        rec = d["recovery"].get(H5_TIER, {})
        rows.append(
            {
                "cell": cell,
                "tier": d["ceiling"]["tier"],
                "eligible": h5["eligible"],
                "excluded_reason": h5["excluded_reason"],
                "point_recovery_m1": h5["point_recovery"],
                "ge_bar": h5["ge_bar"],
                "recovery_ci": (
                    [rec.get("ci_lo"), rec.get("ci_hi")] if not rec.get("suppressed") else None
                ),
                "exceeds_one": rec.get("exceeds_one", False),
                "pooled_r2_m0": d["pooled_r2"]["m0"],
                "pooled_r2_m1": d["pooled_r2"]["m1"],
                "ceiling_pooled_r2": d["ceiling"]["pooled_r2"],
            }
        )
    eligible = [r for r in rows if r["eligible"]]
    n_ge = sum(1 for r in eligible if r["ge_bar"])
    summary = {
        "definition": H5_DEFINITION,
        "arm": arm,
        "tier_used": H5_TIER,
        "bar": H5_RECOVERY_BAR,
        "n_cells": len(rows),
        "n_eligible": len(eligible),
        "n_ge_bar": int(n_ge),
        "strict_majority": bool(len(eligible) > 0 and n_ge * 2 > len(eligible)),
        "excluded": {r["cell"]: r["excluded_reason"] for r in rows if not r["eligible"]},
        "per_cell": rows,
        "regime": regime,
        "metadata": cm.run_metadata(),
    }
    cm.atomic_write_json(ledger_root / "pool" / f"h5_summary__{arm}.json", summary)
    _log(
        f"[h5] arm={arm}: {n_ge}/{len(eligible)} eligible cells >= {H5_RECOVERY_BAR} "
        f"(strict_majority={summary['strict_majority']}; excluded={sorted(summary['excluded'])})"
    )
    return summary


# ---------------------------------------------------------------------------
# Phases / CLI
# ---------------------------------------------------------------------------


def _fold_map(args) -> dict:
    fm = p6.load_or_build_fold_map(
        Path(args.store_root), Path(args.ledger_root), **getattr(args, "fold_floors_override", {})
    )
    return _apply_cells_filter(fm, getattr(args, "cells", None))


def _apply_cells_filter(fm: dict, cells_arg: str | None) -> dict:
    """Restrict the fold map to a G2b-survivor subset (r1 review codex blocker
    g2b-survivors-not-threaded-to-p6): the pooled moments, per-cell units, and
    the H5 summary must all see the SAME survivor set, so the filter lands on
    the fold map itself. Unknown cell names fail loud. The filtered cell list
    enters the regime dict, so mixing filtered/unfiltered outputs in one ledger
    raises the existing regime-mismatch guard rather than silently pooling."""
    if not cells_arg:
        return fm
    wanted = [c.strip() for c in cells_arg.split(",") if c.strip()]
    unknown = sorted(set(wanted) - set(fm["cells"]))
    if unknown:
        raise SystemExit(f"--cells names not in the fold map: {unknown}")
    fm = dict(fm)
    fm["cells"] = {c: fm["cells"][c] for c in wanted}
    return fm


def _regime(args, fold_map: dict, arm: str, layer: int) -> dict:
    return {
        "script_version": SCRIPT_VERSION,
        "arm": arm,
        "layer": int(layer),
        "k": fold_map["k"],
        "seed": fold_map["seed"],
        "n_eq": fold_map["n_eq"],
        "fold_map_sha": fold_map["sha256"],
        "cells": sorted(fold_map["cells"]),
        "ranks": [int(r) for r in args.ranks],
        "n_null_draws": int(args.n_null_draws),
        "bootstrap_draws": int(args.bootstrap_draws),
        "lambda_grid": ["logspace", -2, 4, 13],  # generating parameters, never float hashes
        "dof_cap": 0.9,
        "seed_derivation": "137-rooted per-(cell,arm,fold,rank) via cm.derived_seed",
    }


def phase_pool(args) -> int:
    ledger_root = Path(args.ledger_root)
    store_root = Path(args.store_root)
    gate_path = Path(args.g3_gate_file or (ledger_root / p6.G3_GATE_NAME))
    p6.require_g3_pass(gate_path)  # plan §7: G3 gates the P6 fan-out
    fm = _fold_map(args)
    layer = fits_mod.resolve_layer(args)
    k = int(fm["k"])
    cells = sorted(fm["cells"])
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in p6.ARMS:
            raise SystemExit(f"unknown arm {arm!r} (choices: {p6.ARMS})")
    # Probe stores carry tiny d; production asserts the expected hidden size.
    probe_d = getattr(args, "probe_d", None)
    fm["_probe_d"] = probe_d
    d = int(probe_d) if probe_d else p6.EXPECTED_HIDDEN
    n_max = max(int(fm["cells"][c]["n_rows"]) for c in cells)
    ram_rec = assert_pool_ram(k, d, n_max)

    for arm in arms:
        regime = _regime(args, fm, arm, layer)
        todo = []
        for cell in cells:
            out_path = ledger_root / "pool" / f"{cell}__{arm}.json"
            if out_path.exists():
                prior = json.loads(out_path.read_text(encoding="utf-8"))
                if prior.get("regime") == regime:
                    continue
                raise RuntimeError(f"regime mismatch at {out_path} — use a fresh ledger root")
            todo.append(cell)
        if not todo:
            _log(f"[pool] arm={arm}: all {len(cells)} cell units already done — resume skip")
            compose_h5_summary(ledger_root, arm, cells, regime)
            continue
        parity = None
        if not args.skip_parity:
            parity = parity_gate(store_root, fm, arm, layer, args.device, args.parity_n)
        acc = accumulate_moments(store_root, fm, arm, layer, args.device)
        models = fit_pooled_per_fold(acc["mom"], k)
        pooled_payload = {
            "regime": regime,
            "arm": arm,
            "per_fold": {str(f): m.info() for f, m in models.items()},
            "n_per_cell": acc["n_per_cell"],
            "parity_gate": parity if parity else {"skipped": True},
            "ram_check": ram_rec,
            "metadata": cm.run_metadata(),
        }
        cm.atomic_write_json(ledger_root / "pool" / f"pooled_{arm}.json", pooled_payload)
        t0 = time.time()
        for i, cell in enumerate(cells):
            run_cell_unit(args, fm, models, cell, arm, layer, regime)
            cm.progress("pool", i + 1, len(cells), f"{cell}/{arm}", t0)
        del models, acc
        compose_h5_summary(ledger_root, arm, cells, regime)
    return 0


def phase_h5(args) -> int:
    """Recompute the H5 summary from existing per-cell pool JSONs (idempotent)."""
    ledger_root = Path(args.ledger_root)
    fm = _fold_map(args)
    layer = fits_mod.resolve_layer(args)
    cells = sorted(fm["cells"])
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        compose_h5_summary(ledger_root, arm, cells, _regime(args, fm, arm, layer))
    return 0


# ---------------------------------------------------------------------------
# Synthetic CPU probe
# ---------------------------------------------------------------------------


def _plant_pool_structure(store: Path, *, d: int, seed: int = 29) -> dict:
    """Post-process the fits probe store with pool-discriminating structure:
    (a) a per-cell constant ANSWER bias on every story-question cell except
    storyq_vex (m1 must recover what m0 cannot); (b) a rank-1 per-cell slope
    delta on storyq_vex (m2 must beat m1 there; v7: was dialog_astra — the
    dialogue family is descoped so the probe store has no dialog cells);
    (c) scrambled answers on plain_text (destroys the
    X-Y linkage so its own fit lands at/below the floor -> tier suppression +
    H5 exclusion). Own-map ceilings absorb (a)+(b) by construction."""
    rng = np.random.default_rng(seed)
    planted: dict[str, str] = {}
    # Normalize each bias vector to a FIXED per-dim RMS so the probe's margins
    # never depend on a lucky small draw (seed 29's astra draw realized at
    # per-dim var 0.10 vs the 2.25 expectation and washed out the m0 penalty).
    bias_by_cell = {}
    for c in cm.STORY_Q_CELLS:
        if c == "storyq_vex":
            continue  # storyq_vex carries the rank-1 plant instead (v7)
        b = rng.standard_normal(d)
        bias_by_cell[c] = b * (1.5 / np.sqrt((b**2).mean()))
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    for cell in [*cm.STORY_Q_CELLS, "plain_text"]:
        for ci in p6.production_part_indices(store, cell):
            npz_path = store / f"{cell}__part{ci:04d}__L1.npz"
            with np.load(npz_path) as z:
                arrays = {kk: np.asarray(z[kk]) for kk in z.files}
            v_a = p6.decode_bf16_np(arrays["v_A"]).astype(np.float64)
            if cell in bias_by_cell:
                v_a = v_a + bias_by_cell[cell]
                planted[cell] = "constant answer bias"
            elif cell == "storyq_vex":
                v_c = p6.decode_bf16_np(arrays["v_C"]).astype(np.float64)
                v_a = v_a + 2.0 * np.outer(v_c @ u, v)
                planted[cell] = "rank-1 slope delta"
            else:  # plain_text
                v_a = v_a[rng.permutation(v_a.shape[0])]
                planted[cell] = "scrambled answers (linkage destroyed)"
            arrays["v_A"] = p6.encode_bf16_np(v_a.astype(np.float32))
            with open(npz_path, "wb") as fh:
                np.savez(fh, **arrays)
    return planted


def phase_probe(args) -> int:  # noqa: PLR0915
    """Synthetic CPU self-verification (module docstring item list)."""
    n, d = 40, 8
    ranks = [2, 4]
    with tempfile.TemporaryDirectory(prefix="i2378-pool-probe-") as td:
        tmp = Path(td)
        store, ledger = tmp / "store", tmp / "ledger"
        fits_mod._write_probe_store(store, n=n, d=d)
        planted = _plant_pool_structure(store, d=d)
        _log(f"[probe] planted structure: {planted}")
        fit_ns = argparse.Namespace(
            store_root=str(store),
            ledger_root=str(ledger),
            layer=1,
            layer_star_from=None,
            n_null_draws=6,
            bootstrap_draws=24,
            reduced_k=4,
            units="context",
            g3_gate_file=None,
            fold_floors_override=fits_mod._PROBE_FLOORS,
        )
        ledger.mkdir(parents=True)
        rc = fits_mod.phase_g3(fit_ns)
        assert rc == 0, f"probe fits G3 rc={rc}"
        rc = fits_mod.phase_fit(fit_ns)
        assert rc == 0
        scrambled_fit = json.loads((ledger / "fits" / "plain_text__context.json").read_text())
        assert scrambled_fit["tier"] != "clearly-mappable", scrambled_fit["tier"]

        pool_ns = argparse.Namespace(
            store_root=str(store),
            ledger_root=str(ledger),
            layer=1,
            layer_star_from=None,
            arms="context",
            ranks=ranks,
            n_null_draws=6,
            bootstrap_draws=24,
            device="cpu",
            skip_parity=False,
            parity_n=0,
            g3_gate_file=None,
            fold_floors_override=fits_mod._PROBE_FLOORS,
            probe_d=d,
        )
        rc = phase_pool(pool_ns)
        assert rc == 0

        # (a) planted per-cell bias: m1 recovers what m0 cannot.
        biased = json.loads((ledger / "pool" / "storyq_astra__context.json").read_text())
        rec_m0 = biased["recovery"]["m0"]["point_pooled"]
        rec_m1 = biased["recovery"]["m1"]["point_pooled"]
        assert rec_m1 > rec_m0 + 0.2, (rec_m0, rec_m1)
        assert rec_m1 > 0.8, rec_m1
        assert biased["headline_fold_label"] == "family-held-out"
        _log(f"[probe] planted bias: m0 recovery {rec_m0:+.3f} -> m1 {rec_m1:+.3f} OK")

        # (b) planted rank-1 slope: m2 beats m1 on storyq_vex (v7: was
        # dialog_astra — dialogue descoped). The plant
        # direction is random while the m2 basis is PCA of ISOTROPIC probe X,
        # so a k-dim basis captures ~k/d of the plant — assert on the larger
        # rank (k=4 of d=8: expected gain ~0.25, vs ~0.12 at k=2).
        lowrank = json.loads((ledger / "pool" / "storyq_vex__context.json").read_text())
        r2_m1 = lowrank["pooled_r2"]["m1"]
        r2_m2 = lowrank["pooled_r2"][f"m2_k{ranks[1]}"]
        assert r2_m2 > r2_m1 + 0.05, (r2_m1, r2_m2)
        m2_rec = lowrank["per_fold"][0]["m2"][f"m2_k{ranks[1]}"]
        assert len(m2_rec["null"]["draws"]) == 6
        assert "p_value_residual_fit" in m2_rec["null"]
        _log(f"[probe] planted rank-1 slope: m1 {r2_m1:+.3f} -> m2 {r2_m2:+.3f} OK")

        # (c) scrambled cell: tier suppression + H5 exclusion.
        scr = json.loads((ledger / "pool" / "plain_text__context.json").read_text())
        assert scr["recovery"]["m1"]["suppressed_by_tier"] is True
        assert scr["h5"]["eligible"] is False
        h5 = json.loads((ledger / "pool" / "h5_summary__context.json").read_text())
        assert "plain_text" in h5["excluded"]
        assert h5["n_eligible"] + len(h5["excluded"]) == h5["n_cells"]
        assert "STRICT MAJORITY" in h5["definition"]
        assert isinstance(h5["strict_majority"], bool)
        _log(
            f"[probe] tier suppression + H5: eligible={h5['n_eligible']} "
            f"ge_bar={h5['n_ge_bar']} excluded={sorted(h5['excluded'])} OK"
        )

        # (d) pooled fit artifacts + parity gate record present.
        pooled = json.loads((ledger / "pool" / "pooled_context.json").read_text())
        assert pooled["parity_gate"]["max_rel"] <= PARITY_REL_TOL
        assert len(pooled["per_fold"]) == 5
        assert pooled["ram_check"]["available_gib"] > 0

        # (e) resume path: re-run -> every unit skips, summary recomposed.
        rc = phase_pool(pool_ns)
        assert rc == 0
        _log("[probe] resume-skip: OK")
    _log("[phase=probe] done — all pool probes passed")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.replace("%", "%%"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--phase", choices=("pool", "h5", "probe"), default="pool")
    ap.add_argument(
        "--arms",
        default="context",
        help="comma list of arms (default context — the plan §9 'pooled + 13x5 tiers' single "
        "arm; prefix is runnable but unbudgeted)",
    )
    ap.add_argument(
        "--store-root",
        default=str(cm.REPO_ROOT / "data" / "issue_2378" / "activations"),
    )
    ap.add_argument("--ledger-root", default=str(cm.LEDGER_ROOT))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--layer-star-from", default=None)
    ap.add_argument("--ranks", nargs="*", type=int, default=list(DEFAULT_RANKS))
    ap.add_argument("--n-null-draws", type=int, default=100)
    ap.add_argument("--bootstrap-draws", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--parity-n",
        type=int,
        default=6000,
        help="chat train-subset size for the moment-vs-SVD parity gate (0 = full complement)",
    )
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--g3-gate-file", default=None)
    ap.add_argument(
        "--cells",
        default=None,
        help="comma list restricting the pooled fit + per-cell units + H5 summary to a G2b "
        "survivor subset (default: every fold-map cell); unknown names fail loud",
    )
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _log("[pool] import-check OK")
        return 0
    if args.phase == "pool":
        return phase_pool(args)
    if args.phase == "h5":
        return phase_h5(args)
    if args.phase == "probe":
        return phase_probe(args)
    raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit before C-extension teardown (code-style.md)
