#!/usr/bin/env python
"""Reduced-basis (train-fold truncated-SVD PCA) refit of ladder rungs 7-9 — #2054
round `reduced-basis-refit-rungs789` (plan v16).

The parent's ambient d x d GCV-ridge rung 7/8/9 reads are estimator-degenerate on
the affected pair classes (per-fold n_train < d = 3,584 — #1701/#1887). This
driver refits the SAME three maps as truncated-SVD reduced-basis GCV-ridge: each
fit's input is projected onto the top-k right singular vectors of its OWN
standardized TRAIN-fold matrix (correlation-space PCA == truncated SVD of the
standardized design), the ridge + GCV run in that k-dimensional basis, and
predictions are reconstructed in ambient output space. Fit dimension = k <=
floor(n_train/2), so n_train >= 2k holds by construction. LINEAR PCA ONLY (plan
§13 — no MLP, no kernel PCA, no nonlinear anything).

Modes (--mode):
  run          one shard: (pair-class x arm) or a matched-n group. Fold-map
               floor assert -> resume pre-pass -> per-shard measured pilot
               (identity gates 1+2 + n^2-scaled fleet projection, exit 7 on an
               over-budget projection — a DESIGNED halt) -> unit loop with
               per-unit JSON checkpoints + M-C2 upload cadence -> final scoped
               upload verify.
  merge        VM phase: stage/read unit JSONs, ASSERT the full grid, compute
               calibration cells + n-slope + H1'/H2' + verdict lattice.
  figs         VM phase: figures from the merge digests.
  smoke        P0 synthetic-fixture smoke (n<=90, d=24, k*=8, 3 folds): drives
               the production CLI per shard, exercises gates, resume, merge,
               lattice branches, figs.
  import-check argparse-attribute completeness (whole-module AST) + deferred
               imports executed; exit 0.
  list-shards  print the mechanical shard registry (one slug per line).

Exit 0 success; 2 missing input; 7 over-budget fleet-wall projection (designed
halt, projection persisted in pilot_gate_report__<shard>.json before the raise).

Matched-n N2 feasibility (plan §4, I6-corrected): quantile-matched targets are
drawn from the below-floor empirical intersection distribution, whose range is
strictly < 4,480, and every above-floor intersection is >= 4,480 — so every
target < 4,480 <= every above-floor pair's own intersection and every subsample
is a strict subset.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue2054_forms as forms  # noqa: E402
import issue2054_ladder as ladder  # noqa: E402
from issue2054_pilot import FleetWallExceeded  # noqa: E402
from issue2054_resume import regime_values_equal  # noqa: E402
from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants (plan v16 §4/§11; every value grounded there)

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# Round-scoped upload prefix (plan §10) — the parent's prefixes are READ-ONLY.
DEFAULT_TASK_PREFIX = "issue2054_lattice/reduced_basis_refit_rungs789"
PARENT_ACTIVATIONS_PREFIX = "issue2054_lattice/activations"  # read-only
PARENT_LADDER_PREFIX = "issue2054_lattice/ladder"  # read-only

# Rungs 7-9 only (plan §13 binding scope). Names verbatim from ladder.RUNGS.
RB_RUNGS = ("7_ctx_reparam", "8_ans_reparam", "9_full_AMB")

K_STAR = 1024  # plan §11: largest power of 2 under every affected pair's cap
K_PROFILE = (64, 256, 1024)  # + per-fold k_max appended; all-k reported, no max-over-k
SEED = 137  # inherited (plan v12)
DEFAULT_BOOTSTRAP_DRAWS = ladder.DEFAULT_BOOTSTRAP_DRAWS  # 200

# Matched-n calibration arm (plan §4; critic v223 blocker 2).
AMBIENT_FLOOR = 4480  # n_train = 0.8*4480 = 3584 = d — the ambient floor
N1_LEVELS = (2939, 3500, 4450)
N1P_LEVEL = 3500
PRODUCTION_STRATA = (144, 64)  # (above-floor, below-floor) twobytwo pairs

# Identity gates (plan §4).
IDENTITY_GATE1_TOL = 1e-9
IDENTITY_GATE2_TOL = 1e-6

# Production fold-map floors — constants COPIED WITH CITATION from the two
# main-resident precedents (deliberately not imported: both live on main only,
# absent from the issue-2054 branch the pods clone; plan §11):
#   scripts/issue2054_cross_render_fit.py::_load_production_fold_map (:138-172)
#   scripts/issue2054_fetch_ladder_rows.py::_pool_size (:74-89)
# main's committed shared_fold_map.json is the stale 2026-08-04 single-variant
# SMOKE map (n=1,761, ['char_helios']) — it silently produces plausible
# regularization-limit reads (it already burned a pilot).
FOLD_MAP_MIN_CONV = 20000
FOLD_MAP_MIN_VARIANTS = 5

DEFAULT_MAX_FLEET_WALL_HOURS = 14.0  # plan §9 fence; M-C4 raise mechanism
_M_MEMO_MAX_ENTRIES = 2  # bounded M'-fit memo (plan §4/§8; parent's fit_cache pattern)
RB789_CODE_VERSION = "1"  # resume regime key — bump on output-affecting changes

# Verdict lattice bars (plan §3; I5: the two slope thresholds stay DISTINCT).
H0B_SLOPE_BAR = 0.05  # H0'b PASS bar — does NOT license unmatched-n margins
H2_MARGIN_BAR = 0.025  # H2' margin bar (margin_pp 2.5)
VOID_COVERAGE_PCT = 75.0
VOID_DELTA_PP = 15.0
VOID_RHO_X100 = 50.0
INBAND_FRAC = 0.906
ATTEN_IQR_BAR = 0.05  # ungrounded — needs smoke-test (plan §11)
ATTEN_RHO_CI_UPPER = 0.8
CALIBRATED_RHO = 0.8
CALIBRATED_DELTA = 0.05
MERGE_BOOT_DRAWS = 1000

# Plan-name <-> ladder pair-class mapping (verified against
# issue2054_analyzer_figs.py:399 "cross_framing": "boundary swapped").
CLASS_TO_PAIR_CLASS = {
    "prose": "cross_character",
    "twobytwo": "twobytwo",
    "boundary": "cross_framing",
    "model": "cross_model",
}
PAIR_CLASS_TO_CLASS = {v: k for k, v in CLASS_TO_PAIR_CLASS.items()}
PRODUCTION_CLASS_PAIR_COUNTS = {"boundary": 56, "prose": 96, "model": 48, "twobytwo": 208}
CHAT_ANCHOR_PER_ARM = 32
CONTROL_CLASSES = ("boundary", "model")

# Mechanical shard registry (Axis-2 arm derivation: `--mode list-shards`).
SHARD_REGISTRY: dict[str, dict] = {}
for _cls in ("prose", "twobytwo", "boundary", "model"):
    for _arm in ("context", "prefix"):
        SHARD_REGISTRY[f"{_cls}_{_arm}"] = {
            "kind": "class",
            "shard_class": _cls,
            "pair_class": CLASS_TO_PAIR_CLASS[_cls],
            "arm": _arm,
        }
SHARD_REGISTRY["matchedn-boundary"] = {"kind": "matchedn", "group": "boundary"}
SHARD_REGISTRY["matchedn-twobytwo"] = {"kind": "matchedn", "group": "twobytwo"}


def _log(msg: str) -> None:
    print(f"[phase=rb789] {msg}", flush=True)


def _utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _metadata() -> dict:
    """Reproducibility metadata (git provenance incl. dirty flag + env versions)."""
    md: dict = {"utc": _utc(), "numpy": np.__version__, "code_version": RB789_CODE_VERSION}
    md.update(as_metadata_dict(git_provenance()))
    return md


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Fold map (fail-loud production floor — driver step 3, BEFORE pilot + loop)


def load_fold_map(path: Path, *, allow_smoke: bool) -> dict:
    """Load the shared fold map and REFUSE any map below the production floor.

    Floors (n_conv >= 20,000 AND >= 5 variants) copied with citation from
    `issue2054_cross_render_fit._load_production_fold_map` and
    `issue2054_fetch_ladder_rows._pool_size` (both main-resident; plan §11).
    `--allow-smoke-fold-map` is the P0 fixture's NAMED, enumerated gate
    downgrade (plan §10 blind spots) — the pod driver never passes it.
    """
    d = ladder._load_fold_map(path)
    n_conv = len(d["fold_of"])
    variants = d.get("variants") or []
    if not allow_smoke and (n_conv < FOLD_MAP_MIN_CONV or len(variants) < FOLD_MAP_MIN_VARIANTS):
        raise RuntimeError(
            f"REFUSING fold map {path}: n_conv={n_conv:,} (floor {FOLD_MAP_MIN_CONV:,}), "
            f"variants={variants} (floor {FOLD_MAP_MIN_VARIANTS}). This is the smoke map, "
            "not production — every affected intersection would collapse and every fit "
            "would be a regularization-limit read (the stale-map trap the plan names)."
        )
    d["_sha256"] = _sha256_file(path)
    d["_n_conv"] = n_conv
    d["_path"] = str(path)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Reduced estimator — `_fit_ridge_reduced` (plan §4): structurally identical to
# ladder._fit_ridge with the spectrum truncated at k. At k = full rank the two
# are ALGEBRAICALLY IDENTICAL (identity gate 1 asserts it at |delta| < 1e-9).


def _svd_bundle(X_train: np.ndarray) -> dict:
    """ONE decomposition per input matrix, shared across every fit / rung /
    k-profile value consuming that matrix (plan §9 fit-loop arithmetic)."""
    X64 = X_train.astype(np.float64)
    Xtr, xmu, xsd = ladder._standardize(X64)
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    return {"xmu": xmu, "xsd": xsd, "U": U, "s": s, "Vt": Vt, "n_train": int(Xtr.shape[0])}


def _uty(bundle: dict, Y_train: np.ndarray) -> dict:
    """Per-(bundle, target) projection U^T (Y - ymu), computed once and sliced
    per k (the k-profile re-weights the SAME factors)."""
    Y64 = Y_train.astype(np.float64)
    ymu = Y64.mean(axis=0)
    Yc = Y64 - ymu
    return {
        "ymu": ymu,
        "UtY": bundle["U"].T @ Yc,
        "tot_y_sq": float((Yc**2).sum()),
        "d_out": int(Y64.shape[1]),
    }


def _fit_ridge_reduced(
    bundle: dict,
    uty: dict,
    k: int,
    *,
    lambdas: np.ndarray = ladder.DEFAULT_LAMBDAS,
    dof_cap: float = ladder.DEFAULT_DOF_CAP,
) -> dict:
    """Truncated-SVD reduced-basis GCV-ridge fit (plan §4 pseudocode).

    GCV runs on the truncated spectrum: dof(lam) = sum_{i<=k} s_i^2/(s_i^2+lam)
    with the shared cores' dof cap retained (slack by construction at
    k <= n_train/2 — never LEGACY_UNGUARDED_GCV). Components beyond k are not
    fit, so their full energy stays in the GCV residual. Returns the model in
    ladder._apply_ridge form: W lives in ambient coordinates (d, D_out).
    """
    s = bundle["s"]
    n_train = bundle["n_train"]
    k_real = int(min(int(k), s.size))
    sk = s[:k_real]
    s2 = sk**2
    UtYk = uty["UtY"][:k_real]
    row_energy = (UtYk**2).sum(axis=1)
    tot_y_sq = uty["tot_y_sq"]

    best_lam = float(lambdas[0])
    best_gcv = float("inf")
    best_dof = float("nan")
    dof_over_cap = True
    for lam in lambdas:
        lam = float(lam)
        filt = s2 / (s2 + lam)
        dof = float(filt.sum())
        if dof / n_train <= dof_cap:
            dof_over_cap = False
        rss = tot_y_sq - float(((2 * filt - filt**2) * row_energy).sum())
        denom = (n_train - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if dof / n_train <= dof_cap and gcv < best_gcv:
            best_gcv = gcv
            best_lam = lam
            best_dof = dof
    if best_gcv == float("inf"):
        best_lam = float(lambdas[-1])
        filt = s2 / (s2 + best_lam)
        best_dof = float(filt.sum())
        best_gcv = float("nan")

    filt = sk / (s2 + best_lam)
    W = (bundle["Vt"][:k_real].T * filt) @ UtYk  # (d, D_out) — ambient primal
    tot_s2 = float((s**2).sum())
    info = {
        "k_requested": int(k),
        "k_realized": k_real,
        "best_lambda": best_lam,
        "dof": best_dof,
        "dof_cap": dof_cap,
        "dof_over_cap": bool(dof_over_cap),
        "gcv": best_gcv,
        "n_train": n_train,
        "d_in": int(bundle["Vt"].shape[1]),
        "d_out": uty["d_out"],
        "variance_explained_at_k": float((s2).sum() / tot_s2) if tot_s2 > 0 else float("nan"),
        "selector": "gcv-dofcap-0.9-truncated",
    }
    return {"xmu": bundle["xmu"], "xsd": bundle["xsd"], "ymu": uty["ymu"], "W": W, "info": info}


def _rungs789_reduced(
    b_t: dict,
    b_s: dict,
    b_ys: dict,
    uty_A: dict,
    uty_M: dict,
    uty_B: dict,
    Xt_tr: np.ndarray,
    Xt_te: np.ndarray,
    Yt_tr: np.ndarray,
    k: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Reduced-basis analogue of rungs 7'/8'/9' — mirrors
    `issue2054_ladder._compute_rungs_for_fold` (:561/:574/:599) exactly:

      A' = fit(Xt_tr -> Xs_tr, k)   basis: target-context train PCA
      M' = fit(Xs_tr -> Ys_tr, k)   basis: source-context train PCA
      B' = fit(Ys_tr -> Yt_tr, k)   basis: source-answer train PCA
      rung 7' = M'(A'(Xt_te)) + b7', b7' = train-mean(Yt_tr - M'(A'(Xt_tr)))
      rung 8' = B'(M'(Xt_te));  rung 9' = B'(M'(A'(Xt_te)))  (no b7 inside 9')
    """
    A = _fit_ridge_reduced(b_t, uty_A, k)
    Xs_hat_tr = ladder._apply_ridge(A, Xt_tr)
    Xs_hat_te = ladder._apply_ridge(A, Xt_te)

    M = _fit_ridge_reduced(b_s, uty_M, k)
    P7_tr = ladder._apply_ridge(M, Xs_hat_tr)
    P7_te = ladder._apply_ridge(M, Xs_hat_te)
    P8_te = ladder._apply_ridge(M, Xt_te)
    b7 = (Yt_tr.astype(np.float64) - P7_tr).mean(axis=0)

    B = _fit_ridge_reduced(b_ys, uty_B, k)
    preds = {
        "7_ctx_reparam": P7_te + b7,
        "8_ans_reparam": ladder._apply_ridge(B, P8_te),
        "9_full_AMB": ladder._apply_ridge(B, P7_te),
    }
    infos = {
        "ctx_reparam_fit": A["info"],
        "source_fit": M["info"],
        "ans_reparam_fit": B["info"],
    }
    return preds, infos


# ─────────────────────────────────────────────────────────────────────────────
# Matched-n subsampling (plan §4; I2 auditability)


def _subsample_seed(pair_key: str, arm: str, n_sub: int, seed: int = SEED) -> tuple[int, str]:
    """Deterministic, regime-keyed, python-hash-salt-free subsample seed
    (plan §4/§10 seed convention)."""
    seed_string = f"{pair_key}|{arm}|{n_sub}|{seed}"
    return int(hashlib.sha256(seed_string.encode()).hexdigest()[:8], 16), seed_string


def _draw_subsample(intersection_sorted: list[str], n_sub: int, seed_int: int) -> list[str]:
    """Uniform random conversation subset, drawn ONCE per (pair, arm, level).

    The intersection ordering is PINNED (sorted) before `rng.choice` so the
    draw is reproducible (I2 — the critic's subsample-auditability item)."""
    n = len(intersection_sorted)
    if n_sub >= n:
        raise RuntimeError(
            f"subsample level {n_sub} >= realized intersection {n} — the matched-n arm "
            "requires a strict subsample (plan §4 feasibility: every target < 4,480 <= "
            "every above-floor intersection)"
        )
    rng = np.random.default_rng(seed_int)
    idx = rng.choice(n, size=n_sub, replace=False)
    return sorted(intersection_sorted[i] for i in idx.tolist())


# ─────────────────────────────────────────────────────────────────────────────
# Unit runner


def _pair_key(s_key: str, t_key: str) -> str:
    return f"{s_key}->{t_key}"


def _knn_summary(pred: np.ndarray, true: np.ndarray, metric: str) -> dict:
    r = knn_retrieval(pred, true, metric=metric)
    return {
        "acc_at_k": {str(k): v for k, v in r["acc_at_k"].items()},
        "chance_at_k": {str(k): v for k, v in r["chance_at_k"].items()},
        "median_rank": r["median_rank"],
        "mrr": r["mrr"],
        "n_pool": r["n_pool"],
    }


def _ratio(r2: float, ceil: float) -> float:
    if ceil is None or not np.isfinite(ceil) or abs(ceil) <= 1e-12:
        return float("nan")
    return float(r2 / ceil)


def _run_unit(
    s_key: str,
    t_key: str,
    arm: str,
    plan_class: str,
    s_acts: dict,
    t_acts: dict,
    fold_map: dict,
    *,
    level: int | None,
    k_star: int,
    profile_ks: tuple[int, ...],
    boot_draws: int,
    seed: int,
    control: bool,
    parent_payload: dict | None,
    fold_limit: int | None = None,
    m_memo: dict | None = None,
) -> dict:
    """Compute rungs 7'/8'/9' + matched reduced ceiling for one
    (pair, arm[, level]) unit across the shared conversation-grouped folds.

    Matched-n units run the IDENTICAL code path — subsampling is row-index
    selection upstream (plan §4): the shared fold map assigns the subsampled
    conversations to their EXISTING folds (fold topology inherited).
    """
    kfold = int(fold_map["k"])
    fold_of = fold_map["fold_of"]

    Xs_all, Ys_all, s_ids = ladder._select_arm(s_acts, arm)
    Xt_all, Yt_all, t_ids = ladder._select_arm(t_acts, arm)
    d_in = int(Xt_all.shape[1])

    inter = sorted(set(s_ids) & set(t_ids) & set(fold_of))
    if not inter:
        raise RuntimeError(
            f"EMPTY realized intersection for {s_key}->{t_key} arm={arm} — the production "
            "census says every pair intersects; a zero-row selection is a staging/schema "
            "fault, never a silent skip"
        )
    n_full = len(inter)
    inter_sha = _sha256_text("\n".join(inter))

    used_ids = inter
    sub_sha = None
    seed_string = None
    if level is not None:
        seed_int, seed_string = _subsample_seed(_pair_key(s_key, t_key), arm, int(level), seed)
        used_ids = _draw_subsample(inter, int(level), seed_int)
        sub_sha = _sha256_text("\n".join(used_ids))

    s_row = ladder._row_index_by_conv_id(s_ids)
    t_row = ladder._row_index_by_conv_id(t_ids)

    fold_range = range(kfold) if fold_limit is None else range(min(fold_limit, kfold))
    per_fold: list[dict] = []
    unit_flagged = False
    t0 = time.time()

    parent_pooled = None
    parent_fullpool_ceiling = None
    if parent_payload is not None:
        parent_fullpool_ceiling = parent_payload.get("target_ceiling")
        parent_pooled = (parent_payload.get("arm_report") or {}).get("pooled")

    for fold_i in fold_range:
        train_ids = [cid for cid in used_ids if int(fold_of[cid]) != fold_i]
        val_ids = [cid for cid in used_ids if int(fold_of[cid]) == fold_i]
        if not train_ids or not val_ids:
            per_fold.append({"fold": fold_i, "status": "skipped-empty-fold"})
            continue
        tr_s = np.array([s_row[c] for c in train_ids], dtype=np.int64)
        tr_t = np.array([t_row[c] for c in train_ids], dtype=np.int64)
        va_t = np.array([t_row[c] for c in val_ids], dtype=np.int64)

        Xs_tr = Xs_all[tr_s]
        Ys_tr = Ys_all[tr_s]
        Xt_tr = Xt_all[tr_t]
        Yt_tr = Yt_all[tr_t]
        Xt_te = Xt_all[va_t]
        Yt_te = Yt_all[va_t]
        n_train = int(tr_t.size)

        k_cap = n_train // 2
        if k_cap < 1:
            raise RuntimeError(f"fold {fold_i}: n_train={n_train} too small for any reduced fit")
        k_star_real = min(k_star, k_cap)
        fold_flagged = k_star_real < k_star
        unit_flagged = unit_flagged or fold_flagged
        k_max = min(k_cap, d_in)
        ks = sorted({k for k in profile_ks if 1 <= k <= k_cap} | {k_star_real, k_max})

        # ONE decomposition per input matrix (A'+ceiling share SVD(Xt_tr)).
        # M' bundle+UtY memo (plan §4): the parent's bounded fit_cache pattern,
        # keyed on (source_cell, arm, fold, realized-train-rows sha) — exact by
        # construction; hits only when equalized intersections coincide.
        b_t = _svd_bundle(Xt_tr)
        b_s = None
        uty_M = None
        if m_memo is not None:
            mkey = (s_key, arm, fold_i, _sha256_text("\n".join(train_ids)))
            hit = m_memo.get(mkey)
            if hit is not None:
                b_s, uty_M = hit
        if b_s is None:
            b_s = _svd_bundle(Xs_tr)
            uty_M = _uty(b_s, Ys_tr)
            if m_memo is not None:
                if len(m_memo) >= _M_MEMO_MAX_ENTRIES:
                    m_memo.clear()  # bounded residency (~0.5 GB/entry at control shape)
                m_memo[mkey] = (b_s, uty_M)
        b_ys = _svd_bundle(Ys_tr)
        uty_A = _uty(b_t, Xs_tr)
        uty_B = _uty(b_ys, Yt_tr)
        uty_C = _uty(b_t, Yt_tr)

        Yt_te64 = Yt_te.astype(np.float64)
        per_k: dict[str, dict] = {}
        k_star_block: dict = {}
        for k in ks:
            preds, infos = _rungs789_reduced(
                b_t, b_s, b_ys, uty_A, uty_M, uty_B, Xt_tr, Xt_te, Yt_tr, k
            )
            ceil_model = _fit_ridge_reduced(b_t, uty_C, k)
            ceil_pred = ladder._apply_ridge(ceil_model, Xt_te)
            r2_ceil = ladder._r2_matrix(Yt_te64, ceil_pred)
            rung_rec: dict[str, dict] = {}
            for rung in RB_RUNGS:
                r2 = ladder._r2_matrix(Yt_te64, preds[rung])
                rung_rec[rung] = {"r2": r2, "ratio": _ratio(r2, r2_ceil)}
            per_k[str(k)] = {
                "rungs": rung_rec,
                "ceiling_r2": r2_ceil,
                "fit_infos": {**infos, "matched_ceiling_fit": ceil_model["info"]},
            }
            if k == k_star_real:
                blk_rungs: dict[str, dict] = {}
                for rung in RB_RUNGS:
                    boot = ladder._bootstrap_conv_ci_over_intersection(
                        Yt_te64, preds[rung], n_draws=boot_draws, seed=seed + 10_000 + fold_i
                    )
                    blk_rungs[rung] = {
                        "r2": rung_rec[rung]["r2"],
                        "ratio": rung_rec[rung]["ratio"],
                        "bootstrap_conv_ci": boot,
                        "knn_euclidean": _knn_summary(preds[rung], Yt_te64, "euclidean"),
                        "knn_cosine": _knn_summary(preds[rung], Yt_te64, "cosine"),
                    }
                ceil_boot = ladder._bootstrap_conv_ci_over_intersection(
                    Yt_te64, ceil_pred, n_draws=boot_draws, seed=seed + 10_000 + fold_i
                )
                try:
                    id_pred = identity_bias_predict(Xt_tr, Yt_tr, Xt_te)
                    r2_id = ladder._r2_matrix(Yt_te64, id_pred)
                except ValueError:
                    r2_id = float("nan")
                k_star_block = {
                    "k": k_star_real,
                    "rungs": blk_rungs,
                    "ceiling": {"r2": r2_ceil, "bootstrap_conv_ci": ceil_boot},
                    "r2_identity_bias": r2_id,
                }

        fold_rec = {
            "fold": fold_i,
            "status": "ok",
            "n_train": n_train,
            "n_val": int(va_t.size),
            "k_cap": k_cap,
            "k_star_realized": k_star_real,
            "flagged_k_realized_lt_k_star": bool(fold_flagged),
            "per_k": per_k,
            "k_star_block": k_star_block,
        }
        if control:
            # S-C1: ambient matched-intersection ceiling — full-spectrum solve
            # on the ALREADY-COMPUTED SVD of Xt_tr (well-posed on controls).
            amb_model = _fit_ridge_reduced(b_t, uty_C, k=10**9)
            amb_pred = ladder._apply_ridge(amb_model, Xt_te)
            fold_rec["ambient_matched_ceiling_r2"] = ladder._r2_matrix(Yt_te64, amb_pred)
            fold_rec["ambient_matched_ceiling_fit"] = amb_model["info"]
        per_fold.append(fold_rec)
        print(
            f"[phase=rb789] unit-fold {fold_i + 1}/{kfold} pair={s_key}->{t_key} arm={arm} "
            f"level={level} n_train={n_train} k*={k_star_real} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    ok_folds = [f for f in per_fold if f.get("status") == "ok"]
    unflagged = [f for f in ok_folds if not f["flagged_k_realized_lt_k_star"]]

    def _pool(vals: list[float]) -> dict:
        arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
        return {
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "median": float(np.median(arr)) if arr.size else float("nan"),
            "n_folds": int(arr.size),
        }

    pooled: dict = {"k_star": {}, "per_k": {}}
    for rung in RB_RUNGS:
        pooled["k_star"][rung] = {
            "r2": _pool([f["k_star_block"]["rungs"][rung]["r2"] for f in unflagged]),
            "ratio": _pool([f["k_star_block"]["rungs"][rung]["ratio"] for f in unflagged]),
        }
    pooled["k_star"]["ceiling_r2"] = _pool([f["k_star_block"]["ceiling"]["r2"] for f in unflagged])
    all_ks = sorted({k for f in ok_folds for k in f["per_k"]}, key=int)
    for kstr in all_ks:
        pooled["per_k"][kstr] = {
            rung: {
                "r2": _pool(
                    [f["per_k"][kstr]["rungs"][rung]["r2"] for f in ok_folds if kstr in f["per_k"]]
                ),
                "ratio": _pool(
                    [
                        f["per_k"][kstr]["rungs"][rung]["ratio"]
                        for f in ok_folds
                        if kstr in f["per_k"]
                    ]
                ),
            }
            for rung in RB_RUNGS
        }
    if control:
        pooled["ambient_matched_ceiling_r2"] = _pool(
            [f["ambient_matched_ceiling_r2"] for f in ok_folds]
        )

    payload = {
        "phase": "rb789",
        "round": "reduced-basis-refit-rungs789",
        "status": "ok" if ok_folds else "no-folds-ran",
        "source": s_key,
        "target": t_key,
        "arm": arm,
        "class": plan_class,
        "pair_class": CLASS_TO_PAIR_CLASS[plan_class],
        "level": level,
        "seed_string": seed_string,
        "conv_ids_sha": sub_sha,
        "n_intersection_full": n_full,
        "n_used": len(used_ids),
        "k_star": k_star,
        "k_profile": list(profile_ks),
        "rungs": list(RB_RUNGS),
        "flagged_k_realized_lt_k_star": bool(unit_flagged),
        "n_flagged_folds": sum(1 for f in ok_folds if f["flagged_k_realized_lt_k_star"]),
        "per_fold": per_fold,
        "pooled": pooled,
        "control": bool(control),
        # Reference columns (labels are load-bearing — plan §3/§13):
        "parent_fullpool_ceiling_reference": parent_fullpool_ceiling,
        "parent_ambient_pooled_record_only_known_invalid": parent_pooled,
        "bootstrap_draws": int(boot_draws),
        "seed": int(seed),
        "fold_map": {
            "path": fold_map.get("_path"),
            "k": kfold,
            "seed": int(fold_map.get("seed", -1)),
            "sha256": fold_map.get("_sha256"),
        },
        "metadata": _metadata(),
    }
    payload["regime"] = _unit_regime(payload, inter_sha)
    return payload


def _unit_regime(payload: dict, inter_sha: str) -> dict:
    """EVERY output-affecting regime key (#722-r3 rule; resume predicate)."""
    return {
        "source": payload["source"],
        "target": payload["target"],
        "arm": payload["arm"],
        "level": payload["level"],
        "k_star": payload["k_star"],
        "k_profile": list(payload["k_profile"]),
        "rungs": list(payload["rungs"]),
        "seed": payload["seed"],
        "bootstrap_draws": payload["bootstrap_draws"],
        "fold_map_sha": payload["fold_map"]["sha256"],
        "intersection_sha": inter_sha,
        "subsample_sha": payload["conv_ids_sha"],
        "code_version": RB789_CODE_VERSION,
    }


def _unit_resume_check(out_path: Path, expected_regime: dict) -> tuple[bool, str]:
    """(skip?, reason) — resume a unit ONLY under an exactly-matching regime."""
    if not out_path.is_file():
        return False, ""
    try:
        with out_path.open(encoding="utf-8") as f:
            existing = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, "existing unit JSON unreadable"
    rec = existing.get("regime") or {}
    keys = sorted(set(rec) | set(expected_regime))
    mismatched = [k for k in keys if not regime_values_equal(rec.get(k), expected_regime.get(k))]
    if mismatched:
        return False, f"regime keys changed: {mismatched}"
    if existing.get("status") != "ok":
        return False, f"existing unit status={existing.get('status')!r}"
    return True, "unit JSON complete under matching regime"


# ─────────────────────────────────────────────────────────────────────────────
# Parent (record-only) per-pair ladder JSONs — payload under `arm_report`; the
# TOP-level `pooled` key is an empty placeholder (fetch_ladder_rows._read_rows
# docstring, main @ f7598e6a1b — parse convention copied with citation, plan §11).


def _parent_ladder_payload(parent_dir: Path, s_key: str, t_key: str, arm: str) -> dict | None:
    path = parent_dir / f"rung_1_{s_key}_to_{t_key}_{arm}.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _fold_slices(
    s_acts: dict, t_acts: dict, arm: str, fold_map: dict, fold_i: int
) -> tuple[np.ndarray, ...]:
    """FULL-intersection train/val slices for one fold (identity-gate path —
    I1: gate 2 runs at FULL intersection, BEFORE any subsampling)."""
    fold_of = fold_map["fold_of"]
    Xs_all, Ys_all, s_ids = ladder._select_arm(s_acts, arm)
    Xt_all, Yt_all, t_ids = ladder._select_arm(t_acts, arm)
    inter = sorted(set(s_ids) & set(t_ids) & set(fold_of))
    if not inter:
        raise RuntimeError("identity-gate pair has an EMPTY intersection — staging fault")
    s_row = ladder._row_index_by_conv_id(s_ids)
    t_row = ladder._row_index_by_conv_id(t_ids)
    train_ids = [cid for cid in inter if int(fold_of[cid]) != fold_i]
    val_ids = [cid for cid in inter if int(fold_of[cid]) == fold_i]
    tr_s = np.array([s_row[c] for c in train_ids], dtype=np.int64)
    tr_t = np.array([t_row[c] for c in train_ids], dtype=np.int64)
    va_t = np.array([t_row[c] for c in val_ids], dtype=np.int64)
    return (
        Xs_all[tr_s],
        Ys_all[tr_s],
        Xt_all[tr_t],
        Xt_all[va_t],
        Yt_all[tr_t],
        Yt_all[va_t],
    )


def _ambient_rungs789_r2(
    s_acts: dict, t_acts: dict, arm: str, fold_map: dict, fold_i: int
) -> tuple[dict[str, float], dict[str, np.ndarray], tuple[np.ndarray, ...]]:
    """Ambient recompute of rungs 7/8/9 through the parent's OWN
    `_compute_rungs_for_fold` (bit-parity with the realized run by construction)."""
    Xs_tr, Ys_tr, Xt_tr, Xt_te, Yt_tr, Yt_te = _fold_slices(s_acts, t_acts, arm, fold_map, fold_i)
    rung_preds, _info = ladder._compute_rungs_for_fold(
        Xs_tr=Xs_tr, Ys_tr=Ys_tr, Xt_tr=Xt_tr, Xt_te=Xt_te, Yt_tr=Yt_tr
    )
    Yt_te64 = Yt_te.astype(np.float64)
    r2s = {rung: ladder._r2_matrix(Yt_te64, rung_preds[rung]) for rung in RB_RUNGS}
    return r2s, rung_preds, (Xs_tr, Ys_tr, Xt_tr, Xt_te, Yt_tr, Yt_te)


def _gate2_compare(recomputed: dict[str, float], committed: dict[str, float], tol: float) -> dict:
    """Gate-2 comparison (split out so the smoke can probe the FAIL branch)."""
    deltas = {}
    for rung, val in recomputed.items():
        exp = committed.get(rung)
        if exp is None or not np.isfinite(float(exp)):
            raise AssertionError(f"IDENTITY GATE 2: committed value for {rung} missing/non-finite")
        delta = abs(float(val) - float(exp))
        deltas[rung] = delta
        if not (delta < tol):
            raise AssertionError(
                f"IDENTITY GATE 2 FAILED at {rung}: recomputed {val:.12f} != committed "
                f"{exp:.12f} (|delta| {delta:.3g} >= {tol:g}) — stores / fold map / "
                "equalize-down / row alignment are NOT bit-parity with the realized run"
            )
    return deltas


def _identity_gates(
    s_key: str,
    t_key: str,
    arm: str,
    s_acts: dict,
    t_acts: dict,
    fold_map: dict,
    parent_payload: dict,
) -> dict:
    """Identity gates 1+2 (plan §4) on ONE control pair at FULL intersection.

    Gate 2 (cross-path, row-alignment parity): the ambient recompute must
    reproduce the parent's COMMITTED per-fold `r2_transfer` for rungs 7-9
    within 1e-6 (I1: full intersection, BEFORE any subsampling — a subsampled
    recompute would spuriously FAIL by construction). Fallback (plan
    assumption 6): per-fold values absent -> compare pooled means (recomputes
    all folds).

    Gate 1 (algebraic): at k = full rank `_fit_ridge_reduced` == `_fit_ridge`
    exactly — per-rung PREDICTIONS agree to |delta| < 1e-9, ceiling included.
    FAIL => AssertionError; the shard halts (a code bug, never a finding).
    """
    kfold = int(fold_map["k"])
    arm_report = parent_payload.get("arm_report") or {}
    per_fold = [
        f for f in (arm_report.get("per_fold") or []) if isinstance(f, dict) and f.get("rungs")
    ]
    gate2_mode = "per-fold-0" if per_fold else "pooled-fallback"

    r2s0, amb_preds, slices = _ambient_rungs789_r2(s_acts, t_acts, arm, fold_map, 0)
    if per_fold:
        fold0 = next((f for f in per_fold if int(f.get("fold", -1)) == 0), None)
        if fold0 is None:
            raise AssertionError("IDENTITY GATE 2: parent per_fold has no fold-0 record")
        committed = {r: fold0["rungs"][r]["r2_transfer"] for r in RB_RUNGS}
        gate2 = _gate2_compare(r2s0, committed, IDENTITY_GATE2_TOL)
    else:
        all_r2s: dict[str, list[float]] = {r: [r2s0[r]] for r in RB_RUNGS}
        for fold_i in range(1, kfold):
            r2s_i, _, _ = _ambient_rungs789_r2(s_acts, t_acts, arm, fold_map, fold_i)
            for r in RB_RUNGS:
                all_r2s[r].append(r2s_i[r])
        recomputed = {r: float(np.mean(all_r2s[r])) for r in RB_RUNGS}
        committed = {
            r: (arm_report.get("pooled") or {}).get(r, {}).get("r2_transfer_mean") for r in RB_RUNGS
        }
        gate2 = _gate2_compare(recomputed, committed, IDENTITY_GATE2_TOL)

    # Gate 1 — reduced path at FULL rank vs the ambient predictions (fold 0).
    Xs_tr, Ys_tr, Xt_tr, Xt_te, Yt_tr, _Yt_te = slices
    b_t = _svd_bundle(Xt_tr)
    b_s = _svd_bundle(Xs_tr)
    b_ys = _svd_bundle(Ys_tr)
    uty_A = _uty(b_t, Xs_tr)
    uty_M = _uty(b_s, Ys_tr)
    uty_B = _uty(b_ys, Yt_tr)
    uty_C = _uty(b_t, Yt_tr)
    red_preds, _infos = _rungs789_reduced(
        b_t, b_s, b_ys, uty_A, uty_M, uty_B, Xt_tr, Xt_te, Yt_tr, k=10**9
    )
    gate1 = {}
    for rung in RB_RUNGS:
        max_abs = float(np.max(np.abs(red_preds[rung] - amb_preds[rung])))
        gate1[rung] = max_abs
        if not (max_abs < IDENTITY_GATE1_TOL):
            raise AssertionError(
                f"IDENTITY GATE 1 FAILED at {rung}: reduced@full-rank vs ambient max|delta| "
                f"{max_abs:.3g} >= {IDENTITY_GATE1_TOL:g} — the reduced estimator does not "
                "reduce algebraically to the parent's _fit_ridge"
            )
    amb_ceiling = ladder._fit_ridge(Xt_tr, Yt_tr)
    red_ceiling = _fit_ridge_reduced(b_t, uty_C, k=10**9)
    ceil_delta = float(
        np.max(
            np.abs(
                ladder._apply_ridge(red_ceiling, Xt_te) - ladder._apply_ridge(amb_ceiling, Xt_te)
            )
        )
    )
    if not (ceil_delta < IDENTITY_GATE1_TOL):
        raise AssertionError(
            f"IDENTITY GATE 1 FAILED at matched ceiling: max|delta| {ceil_delta:.3g} "
            f">= {IDENTITY_GATE1_TOL:g}"
        )
    gate1["matched_ceiling"] = ceil_delta
    return {
        "pair": _pair_key(s_key, t_key),
        "arm": arm,
        "gate1_max_abs_pred_delta": gate1,
        "gate1_tol": IDENTITY_GATE1_TOL,
        "gate2_mode": gate2_mode,
        "gate2_abs_r2_delta": gate2,
        "gate2_tol": IDENTITY_GATE2_TOL,
        "n_intersection_full": int(
            len(
                set(ladder._select_arm(s_acts, arm)[2])
                & set(ladder._select_arm(t_acts, arm)[2])
                & set(fold_map["fold_of"])
            )
        ),
        "passed": True,
    }


def _select_gate_pair(
    cells: list, activations_by_cell: dict, parent_dir: Path, arm: str
) -> tuple[str, str, dict]:
    """Deterministically pick a CONTROL pair (boundary first, then cross-model)
    whose parent ladder JSON is staged with status ok — the gate substrate."""
    pairs = ladder._enumerate_ordered_pairs(
        cells, smoke=False, pair_classes=("cross_framing", "cross_model")
    )
    ranked = []
    for s, t in pairs:
        s_key = ladder._cell_key(*s[:4])
        t_key = ladder._cell_key(*t[:4])
        if s_key not in activations_by_cell or t_key not in activations_by_cell:
            continue
        cls = ladder._pair_class(s, t)
        ranked.append((0 if cls == "cross_framing" else 1, s_key, t_key))
    for _rank, s_key, t_key in sorted(ranked):
        payload = _parent_ladder_payload(parent_dir, s_key, t_key, arm)
        if payload is None:
            continue
        if (payload.get("arm_report") or {}).get("status") == "ok":
            return s_key, t_key, payload
    raise RuntimeError(
        f"no control pair with a staged status-ok parent ladder JSON for arm={arm} under "
        f"{parent_dir} — identity gates cannot run; fix staging before production (plan §14)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-shard measured pilot + fleet fence (plan §9 pilot-gated basis; M-C3/M-C4)


def _pilot_and_gates(
    slug: str,
    gate_arm: str,
    pending: list[dict],
    activations_by_cell: dict,
    cells: list,
    fold_map: dict,
    parent_dir: Path,
    args: argparse.Namespace,
    out_dir: Path,
) -> Path | None:
    """Measured 1-unit-fold pilot on the shard's LARGEST-n pending pair with
    batteries on (M-C3), n^2-scaled per-pair extrapolation, fail-loud fence
    (FleetWallExceeded -> exit 7, report written BEFORE the raise), identity
    gates 1+2, and the assumption-4/8 structural store probe at full staged
    grain."""
    report_path = out_dir / f"pilot_gate_report__{slug}.json"
    kfold = int(fold_map["k"])
    if not pending:
        _log(f"pilot gate [{slug}]: 0 pending units — skipping (fully resumed shard)")
        return report_path if report_path.is_file() else None

    # Structural probe (plan §12 rows 4/8) — FULL staged grain, never the slice.
    store_probe = {}
    for cell_key, acts in sorted(activations_by_cell.items()):
        ids = acts["conv_ids"]
        if len(set(ids)) != len(ids):
            raise AssertionError(f"structural probe: duplicate conv_ids in cell {cell_key}")
        store_probe[cell_key] = int(len(ids))

    if report_path.is_file() and not args.overwrite:
        with report_path.open(encoding="utf-8") as f:
            prior = json.load(f)
        if (
            prior.get("k_star") == int(args.k_star)
            and prior.get("bootstrap_draws") == int(args.bootstrap_draws)
            and prior.get("gate_arm") == gate_arm
        ):
            wall = prior.get("pilot_wall_seconds_per_unit_fold")
            if (
                not isinstance(wall, (int, float))
                or isinstance(wall, bool)
                or not (math.isfinite(float(wall)) and float(wall) > 0)
            ):
                raise RuntimeError(
                    f"pilot gate [{slug}]: prior report {report_path} matches knobs but has no "
                    "usable measured wall — delete it or pass --overwrite to re-measure "
                    "(a 0/NaN wall would project 0 and DISARM the fence)"
                )
            n_pilot = float(prior["pilot_n_used"])
            _enforce_fence(slug, float(wall), n_pilot, pending, kfold, args, report_path, prior)
            _log(f"pilot gate [{slug}]: prior report matches; projection re-derived")
            return report_path

    gs, gt, gpayload = _select_gate_pair(cells, activations_by_cell, parent_dir, gate_arm)
    _log(f"pilot gate [{slug}]: identity gates 1+2 on {gs} -> {gt} ({gate_arm}, full n)")
    gates = _identity_gates(
        gs, gt, gate_arm, activations_by_cell[gs], activations_by_cell[gt], fold_map, gpayload
    )
    _log(
        f"pilot gate [{slug}]: gates PASS (gate1 max {max(gates['gate1_max_abs_pred_delta'].values()):.3g}; "
        f"gate2 {gates['gate2_mode']} max {max(gates['gate2_abs_r2_delta'].values()):.3g})"
    )

    unit = max(pending, key=lambda u: u["n_used"])
    _log(
        f"pilot gate [{slug}]: timing pilot on {unit['s_key']} -> {unit['t_key']} "
        f"arm={unit['arm']} level={unit['level']} n_used={unit['n_used']} (1 fold, batteries on)"
    )
    t0 = time.time()
    _run_unit(
        unit["s_key"],
        unit["t_key"],
        unit["arm"],
        unit["plan_class"],
        activations_by_cell[unit["s_key"]],
        activations_by_cell[unit["t_key"]],
        fold_map,
        level=unit["level"],
        k_star=int(args.k_star),
        profile_ks=tuple(args.k_profile),
        boot_draws=int(args.bootstrap_draws),
        seed=int(args.seed),
        control=unit["control"],
        parent_payload=None,
        fold_limit=1,
    )
    wall = time.time() - t0
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    payload = {
        "phase": "rb789-pilot-gate",
        "shard": slug,
        "gate_arm": gate_arm,
        "k_star": int(args.k_star),
        "k_profile": list(args.k_profile),
        "bootstrap_draws": int(args.bootstrap_draws),
        "seed": int(args.seed),
        "pilot_pair": _pair_key(unit["s_key"], unit["t_key"]),
        "pilot_arm": unit["arm"],
        "pilot_level": unit["level"],
        "pilot_n_used": int(unit["n_used"]),
        "peak_rss_gib": round(peak_rss_gib, 3),
        "identity_gates": gates,
        "store_probe_n_cells": len(store_probe),
        "store_probe_rows_min": min(store_probe.values()),
        "store_probe_rows_max": max(store_probe.values()),
        "store_probe_duplicate_free": True,
        "utc": _utc(),
    }
    _log(f"pilot gate [{slug}]: wall={wall:.1f}s peak_rss={peak_rss_gib:.2f} GiB")
    _enforce_fence(slug, wall, float(unit["n_used"]), pending, kfold, args, report_path, payload)
    return report_path


def _enforce_fence(
    slug: str,
    wall_1fold: float,
    n_pilot: float,
    pending: list[dict],
    kfold: int,
    args: argparse.Namespace,
    report_path: Path,
    payload: dict,
) -> None:
    """n^2-scaled fleet projection + fail-loud fence (report written BEFORE
    the raise — an artifact-routed DESIGNED halt, exit 7; #1415 convention).

    projected shard wall = sum over pending units of
      pilot_wall_1fold x (n_unit / n_pilot)^2 x fold_k   (M-C3; the plan's
    registered conservative form. Analyzer note A4: W_k formation is
    n-independent, so pure-n^2 UNDER-projects the smallest pairs by up to
    ~1.5x — absorbed by the x2 booking; do not loosen the fence for it.)
    """
    scale = sum((float(u["n_used"]) / n_pilot) ** 2 for u in pending)
    projected = float(wall_1fold) * kfold * scale
    max_h = float(args.max_fleet_wall_hours)
    out = dict(payload)
    out.update(
        {
            "pilot_wall_seconds_per_unit_fold": round(float(wall_1fold), 3),
            "n_pending_units": len(pending),
            "pending_n_used": sorted(int(u["n_used"]) for u in pending),
            "fold_k": kfold,
            "n2_scale_sum": round(scale, 3),
            "projected_shard_wall_seconds": round(projected, 1),
            "projected_shard_wall_hours": round(projected / 3600.0, 3),
            "fence_floor_seconds": round(2.0 * projected, 1),
            "max_fleet_wall_hours": max_h,
            "mc4_raise_band": bool(projected / 3600.0 > max_h / 2.0),
            "extrapolation": "per-pair n^2-scaled from the largest-n pending pair (M-C3)",
        }
    )
    _write_json(report_path, out)
    _log(
        f"pilot gate [{slug}]: projected {projected / 3600.0:.2f} h over {len(pending)} pending "
        f"units (fence {max_h} h; report -> {report_path})"
    )
    if projected > max_h * 3600.0:
        raise FleetWallExceeded(
            f"projected shard wall {projected / 3600.0:.2f} h exceeds {max_h} h "
            f"({len(pending)} pending units x {kfold} folds, n^2-scaled from "
            f"{wall_1fold:.1f}s pilot; report: {report_path}) — designed halt: re-shard wider "
            "or apply the M-C4 raise mechanism (raised RB789_MAX_FLEET_WALL_HOURS + "
            "--time-budget-hours >= 2x projection + a basis-update note), never a silent descope"
        )
    if out["mc4_raise_band"]:
        _log(
            f"WARN pilot gate [{slug}]: projection {projected / 3600.0:.2f} h is in the "
            f"({max_h / 2.0:.1f}, {max_h:.1f}] h band — fence no longer >= 2x projection; "
            "the M-C4 raise mechanism applies BEFORE the fleet loop proceeds"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Uploads (M-C2 cadence; one bulk upload_folder commit per flush — never
# per-file loops; parent prefixes are READ-ONLY by construction)

from explore_persona_space.orchestrate.hub import (  # noqa: E402
    _upload_folder_filtered,
    verify_repo_paths_uploaded,
)


def _flush_upload(out_dir: Path, rel_names: list[str], subdir: str, task_prefix: str) -> None:
    if not rel_names:
        return
    allow = sorted(set(rel_names))
    expected = [f"{task_prefix}/{subdir}/{r}" for r in allow]
    url = _upload_folder_filtered(
        out_dir,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{task_prefix}/{subdir}",
        allow_patterns=allow,
        expected_repo_paths=expected,
    )
    if not url:
        raise RuntimeError(
            f"bulk upload failed or incomplete -> {task_prefix}/{subdir}/ "
            f"({len(allow)} files; returned no path; local files kept)"
        )
    _log(f"uploaded {len(allow)} file(s) in one bulk commit -> {task_prefix}/{subdir}/")


def _final_verify(rel_names: list[str], subdir: str, task_prefix: str) -> None:
    from huggingface_hub import HfApi

    expected = [f"{task_prefix}/{subdir}/{r}" for r in sorted(set(rel_names))]
    missing = verify_repo_paths_uploaded(
        HfApi(),
        HF_DATA_REPO,
        expected,
        path_in_repo=f"{task_prefix}/{subdir}",
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(
            f"final scoped verify: {len(missing)} expected path(s) missing under "
            f"{task_prefix}/{subdir}/ — first: {missing[:3]}"
        )
    _log(f"final scoped verify PASS: {len(expected)} path(s) under {task_prefix}/{subdir}/")


# ─────────────────────────────────────────────────────────────────────────────
# N2 quantile targets (plan §4 matched-n arm; I6-corrected feasibility)


def _n2_targets(pair_n: dict[str, int], floor: int) -> tuple[dict[str, int], list[str], list[str]]:
    """Rank-based quantile map: each above-floor pair's position onto the
    below-floor empirical intersection distribution.

    Deterministic ordering: pure-python sort on (n, pair_key) — never a bare
    numpy argsort whose tie order is CPU-SIMD dependent (#1946). Feasibility
    (I6-corrected): the below-floor empirical range is strictly < floor and
    above-floor intersections are >= floor, so every target < floor <= every
    above-floor pair's own intersection — strict subsamples by construction.
    """
    above = sorted((n, k) for k, n in pair_n.items() if n >= floor)
    below = sorted(n for k, n in pair_n.items() if n < floor)
    if not above or not below:
        raise RuntimeError(
            f"N2 strata degenerate: above={len(above)} below={len(below)} at floor {floor}"
        )
    targets: dict[str, int] = {}
    for i, (n, key) in enumerate(above):
        q = (i + 0.5) / len(above)
        target = int(round(float(np.quantile(np.asarray(below, dtype=np.float64), q))))
        target = min(target, n - 1)  # strict-subsample belt (rounding at the boundary)
        targets[key] = target
    return targets, [k for _n, k in above], sorted(k for k, n in pair_n.items() if n < floor)


# ─────────────────────────────────────────────────────────────────────────────
# Shard runner (--mode run)


def _resolve_shard(args: argparse.Namespace) -> tuple[str, dict]:
    if bool(args.matchedn) == bool(args.shard_class):
        raise SystemExit("exactly one of --shard-class (+--arm) or --matchedn is required")
    if args.matchedn:
        slug = f"matchedn-{args.matchedn}"
    else:
        if not args.arm:
            raise SystemExit("--arm is required with --shard-class")
        slug = f"{args.shard_class}_{args.arm}"
    if slug not in SHARD_REGISTRY:
        raise SystemExit(f"unknown shard {slug!r} (see --mode list-shards)")
    return slug, SHARD_REGISTRY[slug]


def _load_cells(args: argparse.Namespace) -> tuple[list, dict]:
    activations_dir = Path(args.activations_dir).resolve()
    if not activations_dir.exists():
        raise SystemExit(f"--activations-dir does not exist: {activations_dir}")
    cells = ladder._resolve_cells(
        activations_dir,
        list(args.variants),
        list(args.conditions),
        list(args.forms),
        list(args.models),
    )
    if not cells:
        raise SystemExit(f"no activation .npz found under {activations_dir}")
    activations_by_cell: dict[str, dict] = {}
    for variant, condition, form, model, path in cells:
        acts = ladder._load_activation_npz(path)
        if acts is None:
            raise RuntimeError(f"EMPTY activation .npz staged: {path}")
        activations_by_cell[ladder._cell_key(variant, condition, form, model)] = acts
    return cells, activations_by_cell


def _context_intersections(pairs: list, activations_by_cell: dict, fold_of: dict) -> dict[str, int]:
    """Realized CONTEXT-arm intersection per pair key (strata + N2 inputs)."""
    id_cache: dict[str, set] = {}

    def _ids(key: str) -> set:
        if key not in id_cache:
            _x, _y, ids = ladder._select_arm(activations_by_cell[key], "context")
            id_cache[key] = set(ids) & set(fold_of)
        return id_cache[key]

    out = {}
    for s, t in pairs:
        s_key = ladder._cell_key(*s[:4])
        t_key = ladder._cell_key(*t[:4])
        if s_key not in activations_by_cell or t_key not in activations_by_cell:
            continue
        out[_pair_key(s_key, t_key)] = len(_ids(s_key) & _ids(t_key))
    return out


def _unit_out_name(s_key: str, t_key: str, arm: str, level: int | None) -> str:
    base = f"rb789_{s_key}_to_{t_key}_{arm}"
    return f"{base}_n{level}.json" if level is not None else f"{base}.json"


def _build_unit_specs(
    slug: str,
    cfg: dict,
    cells: list,
    activations_by_cell: dict,
    fold_map: dict,
    args: argparse.Namespace,
) -> tuple[list[dict], str]:
    """Enumerate the shard's (pair, arm[, level]) unit specs (plan §4/§9)."""
    fold_of = fold_map["fold_of"]
    specs: list[dict] = []
    if cfg["kind"] == "class":
        pairs = ladder._enumerate_ordered_pairs(
            cells, smoke=False, pair_classes=(cfg["pair_class"],)
        )
        plan_class = cfg["shard_class"]
        control = plan_class in CONTROL_CLASSES
        for s, t in pairs:
            s_key = ladder._cell_key(*s[:4])
            t_key = ladder._cell_key(*t[:4])
            if s_key not in activations_by_cell or t_key not in activations_by_cell:
                raise RuntimeError(f"pair cell not staged: {s_key} / {t_key}")
            specs.append(
                {
                    "s_key": s_key,
                    "t_key": t_key,
                    "arm": cfg["arm"],
                    "level": None,
                    "plan_class": plan_class,
                    "control": control,
                }
            )
        subdir = "ladder"
    elif cfg["group"] == "boundary":
        pairs = ladder._enumerate_ordered_pairs(cells, smoke=False, pair_classes=("cross_framing",))
        levels = [int(x) for x in args.matchedn_levels]
        for s, t in pairs:
            s_key = ladder._cell_key(*s[:4])
            t_key = ladder._cell_key(*t[:4])
            for lv in levels:  # N1 — context, all levels
                specs.append(
                    {
                        "s_key": s_key,
                        "t_key": t_key,
                        "arm": "context",
                        "level": lv,
                        "plan_class": "boundary",
                        "control": False,  # S-C1 ambient ceiling is full-n-control-only
                    }
                )
            specs.append(  # N1p — prefix anchor, one level
                {
                    "s_key": s_key,
                    "t_key": t_key,
                    "arm": "prefix",
                    "level": int(args.matchedn_prefix_level),
                    "plan_class": "boundary",
                    "control": False,
                }
            )
        subdir = "ladder_matchedn"
    else:  # matchedn-twobytwo (N2)
        pairs = ladder._enumerate_ordered_pairs(cells, smoke=False, pair_classes=("twobytwo",))
        pair_n = _context_intersections(pairs, activations_by_cell, fold_of)
        targets, above, below = _n2_targets(pair_n, int(args.ambient_floor))
        expect = tuple(int(x) for x in args.strata_expect)
        if (len(above), len(below)) != expect:
            raise RuntimeError(
                f"N2 strata (above={len(above)}, below={len(below)}) != expected {expect} "
                f"at floor {args.ambient_floor} — census mismatch (plan assumption 1/3)"
            )
        key_to_pair = {
            _pair_key(ladder._cell_key(*s[:4]), ladder._cell_key(*t[:4])): (
                ladder._cell_key(*s[:4]),
                ladder._cell_key(*t[:4]),
            )
            for s, t in pairs
        }
        for key in above:
            s_key, t_key = key_to_pair[key]
            specs.append(
                {
                    "s_key": s_key,
                    "t_key": t_key,
                    "arm": "context",
                    "level": int(targets[key]),
                    "plan_class": "twobytwo",
                    "control": False,
                    "stratum": "above_floor",
                }
            )
        subdir = "ladder_matchedn"
    if not specs:
        raise RuntimeError(f"shard {slug}: ZERO units enumerated — never a silent no-op")
    specs.sort(key=lambda u: (u["s_key"], u["t_key"], u["arm"], u["level"] or -1))
    return specs, subdir


def run_shard(args: argparse.Namespace) -> int:
    slug, cfg = _resolve_shard(args)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    parent_dir = Path(args.parent_ladder_dir).resolve()
    if not parent_dir.is_dir():
        raise SystemExit(f"--parent-ladder-dir does not exist: {parent_dir}")
    task_prefix = args.hf_prefix

    # Driver step 3 (plan §4 P1): fold-map fail-loud floor BEFORE pilot + loop.
    fold_map = load_fold_map(Path(args.fold_map).resolve(), allow_smoke=args.allow_smoke_fold_map)
    _log(
        f"shard {slug}: fold map ok (n_conv={fold_map['_n_conv']:,}, "
        f"variants={len(fold_map.get('variants') or [])}, k={fold_map['k']})"
    )

    cells, activations_by_cell = _load_cells(args)
    specs, subdir = _build_unit_specs(slug, cfg, cells, activations_by_cell, fold_map, args)
    _log(f"shard {slug}: {len(specs)} unit(s) enumerated (subdir {subdir})")

    # Resume pre-pass (identity computed ONCE per unit; regime-keyed skip).
    pending: list[dict] = []
    all_rel_names: list[str] = []
    n_resumed = 0
    for spec in specs:
        s_acts = activations_by_cell[spec["s_key"]]
        t_acts = activations_by_cell[spec["t_key"]]
        _xs, _ys, s_ids = ladder._select_arm(s_acts, spec["arm"])
        _xt, _yt, t_ids = ladder._select_arm(t_acts, spec["arm"])
        inter = sorted(set(s_ids) & set(t_ids) & set(fold_map["fold_of"]))
        if not inter:
            raise RuntimeError(
                f"EMPTY intersection {spec['s_key']}->{spec['t_key']} arm={spec['arm']}"
            )
        inter_sha = _sha256_text("\n".join(inter))
        sub_sha = None
        n_used = len(inter)
        if spec["level"] is not None:
            seed_int, _ss = _subsample_seed(
                _pair_key(spec["s_key"], spec["t_key"]),
                spec["arm"],
                int(spec["level"]),
                int(args.seed),
            )
            used = _draw_subsample(inter, int(spec["level"]), seed_int)
            sub_sha = _sha256_text("\n".join(used))
            n_used = len(used)
        expected_regime = {
            "source": spec["s_key"],
            "target": spec["t_key"],
            "arm": spec["arm"],
            "level": spec["level"],
            "k_star": int(args.k_star),
            "k_profile": [int(k) for k in args.k_profile],
            "rungs": list(RB_RUNGS),
            "seed": int(args.seed),
            "bootstrap_draws": int(args.bootstrap_draws),
            "fold_map_sha": fold_map["_sha256"],
            "intersection_sha": inter_sha,
            "subsample_sha": sub_sha,
            "code_version": RB789_CODE_VERSION,
        }
        out_name = _unit_out_name(spec["s_key"], spec["t_key"], spec["arm"], spec["level"])
        out_path = out_dir / out_name
        all_rel_names.append(out_name)
        skip, why = (False, "") if args.overwrite else _unit_resume_check(out_path, expected_regime)
        if skip:
            n_resumed += 1
            _log(f"unit RESUME skip {out_name} ({why})")
            continue
        if why:
            _log(f"unit recompute {out_name}: {why}")
        pending.append({**spec, "out_path": out_path, "out_name": out_name, "n_used": n_used})
    _log(f"shard {slug}: {len(pending)} pending / {n_resumed} resumed")

    if not args.skip_pilot_gate:
        gate_arm = cfg.get("arm") or "context"
        pilot_report = _pilot_and_gates(
            slug,
            gate_arm,
            pending,
            activations_by_cell,
            cells,
            fold_map,
            parent_dir,
            args,
            out_dir,
        )
    else:
        pilot_report = None
        _log(f"shard {slug}: --skip-pilot-gate (a prior pilot must already cover this shard)")

    flush_every = min(50, max(1, math.ceil(max(len(pending), 1) / 4)))
    flush_window_s = 45 * 60.0
    new_since_flush: list[str] = []
    last_flush = time.time()
    t0 = time.time()
    m_memo: dict = {}
    current_source: str | None = None
    for i, unit in enumerate(pending):
        if unit["s_key"] != current_source:
            m_memo.clear()  # units are source-major sorted — bounded residency
            current_source = unit["s_key"]
        parent_payload = _parent_ladder_payload(
            parent_dir, unit["s_key"], unit["t_key"], unit["arm"]
        )
        payload = _run_unit(
            unit["s_key"],
            unit["t_key"],
            unit["arm"],
            unit["plan_class"],
            activations_by_cell[unit["s_key"]],
            activations_by_cell[unit["t_key"]],
            fold_map,
            level=unit["level"],
            k_star=int(args.k_star),
            profile_ks=tuple(int(k) for k in args.k_profile),
            boot_draws=int(args.bootstrap_draws),
            seed=int(args.seed),
            control=unit["control"],
            parent_payload=parent_payload,
            m_memo=m_memo,
        )
        if unit.get("stratum"):
            payload["stratum"] = unit["stratum"]
        _write_json(unit["out_path"], payload)
        new_since_flush.append(unit["out_name"])
        print(
            f"[phase=rb789] unit {i + 1}/{len(pending)} pair={unit['s_key']}->{unit['t_key']} "
            f"arm={unit['arm']} level={unit['level']} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        if not args.skip_upload and (
            len(new_since_flush) >= flush_every or (time.time() - last_flush) > flush_window_s
        ):
            _flush_upload(out_dir, new_since_flush, subdir, task_prefix)
            new_since_flush = []
            last_flush = time.time()

    if not args.skip_upload:
        _flush_upload(out_dir, new_since_flush, subdir, task_prefix)
        if pilot_report is not None and pilot_report.is_file():
            _flush_upload(out_dir, [pilot_report.name], "pilot", task_prefix)
        _final_verify(all_rel_names, subdir, task_prefix)
        if pilot_report is not None and pilot_report.is_file():
            _final_verify([pilot_report.name], "pilot", task_prefix)

    digest = {
        "phase": "rb789-shard",
        "shard": slug,
        "n_units": len(specs),
        "n_pending_ran": len(pending),
        "n_resumed": n_resumed,
        "n_flagged_units": _count_flagged(out_dir, all_rel_names),
        "hf_subdir": subdir,
        "task_prefix": task_prefix,
        "wall_seconds": round(time.time() - t0, 1),
        "metadata": _metadata(),
    }
    _write_json(out_dir / f"digest__{slug}.json", digest)
    _log(
        f"shard {slug} digest: {json.dumps({k: digest[k] for k in ('n_units', 'n_pending_ran', 'n_resumed', 'n_flagged_units')})}"
    )
    print("[phase=done]", flush=True)
    return 0


def _count_flagged(out_dir: Path, rel_names: list[str]) -> int:
    n = 0
    for rel in rel_names:
        p = out_dir / rel
        if p.is_file():
            with p.open(encoding="utf-8") as f:
                if json.load(f).get("flagged_k_realized_lt_k_star"):
                    n += 1
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Merge (P2, VM): full-grid assert -> calibration cells -> n-slope -> H1'/H2'
# -> verdict lattice. Pure re-reduction of persisted per-unit values (zero fits).


def _read_committed_json(relpath: str) -> dict:
    """Committed census artifacts: local file first, else the origin/issue-2054
    blob (the branch carries analyzer_companions/; worktrees are sparse)."""
    p = _REPO_ROOT / relpath
    if p.is_file():
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    out = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "show", f"origin/issue-2054:{relpath}"],
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"cannot read committed census artifact {relpath} (local file absent AND "
            f"git show origin/issue-2054 rc={out.returncode}): {out.stderr.strip()[:300]}"
        )
    return json.loads(out.stdout)


def _production_census() -> dict:
    """Expected-grid census for the production merge, cross-checked against the
    committed artifacts (plan §7: counts sourced from the committed census)."""
    comp = _read_committed_json(
        "eval_results/issue_2054/analyzer_companions/ladder_intersection_composition.json"
    )
    inter = comp["intersections"]
    if int(inter["n_pairs"]) != sum(PRODUCTION_CLASS_PAIR_COUNTS.values()):
        raise AssertionError(
            f"census n_pairs {inter['n_pairs']} != {sum(PRODUCTION_CLASS_PAIR_COUNTS.values())}"
        )
    for cls, count in PRODUCTION_CLASS_PAIR_COUNTS.items():
        got = int(inter[cls]["n"])
        if got != count:
            raise AssertionError(f"census class {cls}: n={got} != expected {count}")
    chat = _read_committed_json(
        "eval_results/issue_2054/analyzer_companions/chat_to_character_pairs.json"
    )
    if int(chat["n_context_arm"]) != CHAT_ANCHOR_PER_ARM:
        raise AssertionError(
            f"chat-anchor census n_context_arm={chat['n_context_arm']} != {CHAT_ANCHOR_PER_ARM}"
        )
    anchor_ctx = sorted((p["src"], p["tgt"]) for p in chat["pairs"] if p.get("arm") == "context")
    return {
        "class_pair_counts": dict(PRODUCTION_CLASS_PAIR_COUNTS),
        "chat_anchor_per_arm": CHAT_ANCHOR_PER_ARM,
        "chat_anchor_pairs_context": anchor_ctx,
        "strata": {"above_floor": PRODUCTION_STRATA[0], "below_floor": PRODUCTION_STRATA[1]},
        "n1_levels": list(N1_LEVELS),
        "n1p_level": N1P_LEVEL,
        "ambient_floor": AMBIENT_FLOOR,
    }


def _load_census(census_arg: str) -> dict:
    if census_arg == "production":
        return _production_census()
    with Path(census_arg).open(encoding="utf-8") as f:
        return json.load(f)


def _parse_cell_key(key: str) -> tuple[str, str, str, str]:
    parts = key.split(forms.CELL_KEY_SEP)
    if len(parts) != 4:
        raise ValueError(f"cell key {key!r} does not split into 4 axes")
    return tuple(parts)  # type: ignore[return-value]


def _classify_pair_keys(s_key: str, t_key: str) -> str | None:
    cls = ladder._pair_class(_parse_cell_key(s_key), _parse_cell_key(t_key))
    return PAIR_CLASS_TO_CLASS.get(cls) if cls else None


def _is_chat_anchor_key(key: str) -> bool:
    v, c, f, _m = _parse_cell_key(key)
    return v == ladder.ASSISTANT_VARIANT and c == "inserted" and f == "chat"


def _load_units(units_dir: Path) -> list[dict]:
    paths = sorted(Path(units_dir).glob("rb789_*.json"))
    if not paths:
        raise RuntimeError(f"ZERO unit JSONs under {units_dir} — never a silent empty merge")
    units = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            units.append(json.load(f))
    return units


def _grid_assert(units_full: list[dict], units_matched: list[dict], census: dict) -> dict:
    """Fail-loud full-grid completeness assert (plan §4 P2 / §7) — the merge
    NEVER writes a partial lattice."""
    by_cls_arm: dict[tuple[str, str], set] = {}
    for u in units_full:
        derived = _classify_pair_keys(u["source"], u["target"])
        if derived != u["class"]:
            raise AssertionError(
                f"unit {u['source']}->{u['target']} class {u['class']!r} != derived {derived!r}"
            )
        by_cls_arm.setdefault((u["class"], u["arm"]), set()).add((u["source"], u["target"]))
    problems = []
    for cls, count in census["class_pair_counts"].items():
        for arm in ("context", "prefix"):
            got = len(by_cls_arm.get((cls, arm), set()))
            if got != count:
                problems.append(f"{cls}/{arm}: {got} pairs != expected {count}")
    if problems:
        raise AssertionError("full-n grid INCOMPLETE: " + "; ".join(problems))

    anchor_ctx = sorted(
        (s, t)
        for (s, t) in by_cls_arm.get(("twobytwo", "context"), set())
        if _is_chat_anchor_key(s) or _is_chat_anchor_key(t)
    )
    if len(anchor_ctx) != census["chat_anchor_per_arm"]:
        raise AssertionError(
            f"chat-anchor subset re-derived {len(anchor_ctx)}/arm != census "
            f"{census['chat_anchor_per_arm']}"
        )
    census_anchor = census.get("chat_anchor_pairs_context")
    if census_anchor is not None and [list(x) for x in anchor_ctx] != [
        list(x) for x in census_anchor
    ]:
        raise AssertionError("chat-anchor pair SET != chat_to_character_pairs.json census")

    # Matched-n grid: {n_boundary x levels (ctx)} + {n_boundary x 1 (pfx)} + {n_above x 1 (ctx)}.
    n_boundary = census["class_pair_counts"]["boundary"]
    n_above = census["strata"]["above_floor"]
    levels = [int(x) for x in census["n1_levels"]]
    n1 = {}
    n1p = {}
    n2 = {}
    for u in units_matched:
        pk = _pair_key(u["source"], u["target"])
        if u["class"] == "boundary" and u["arm"] == "context":
            n1.setdefault(pk, set()).add(int(u["level"]))
        elif u["class"] == "boundary" and u["arm"] == "prefix":
            n1p.setdefault(pk, set()).add(int(u["level"]))
        elif u["class"] == "twobytwo" and u["arm"] == "context":
            n2.setdefault(pk, set()).add(int(u["level"]))
        else:
            raise AssertionError(f"unexpected matched-n unit class/arm: {u['class']}/{u['arm']}")
    if len(n1) != n_boundary or any(v != set(levels) for v in n1.values()):
        raise AssertionError(
            f"N1 grid incomplete: {len(n1)} pairs (expected {n_boundary}) x levels {levels}"
        )
    if len(n1p) != n_boundary or any(len(v) != 1 for v in n1p.values()):
        raise AssertionError(f"N1p grid incomplete: {len(n1p)} pairs (expected {n_boundary})")
    if len(n2) != n_above or any(len(v) != 1 for v in n2.values()):
        raise AssertionError(f"N2 grid incomplete: {len(n2)} pairs (expected {n_above})")
    return {
        "full_units": len(units_full),
        "matched_units": len(units_matched),
        "chat_anchor_pairs_context": anchor_ctx,
        "n1_pairs": len(n1),
        "n1p_pairs": len(n1p),
        "n2_pairs": len(n2),
    }


def _unit_rung_val(u: dict, rung: str, space: str) -> float:
    blk = u["pooled"]["k_star"][rung]
    v = blk["ratio" if space == "ratio" else "r2"]["mean"]
    return float(v) if v is not None else float("nan")


def _parent_rung_val(u: dict, rung: str, space: str) -> float:
    pooled = u.get("parent_ambient_pooled_record_only_known_invalid") or {}
    rec = pooled.get(rung) or {}
    key = "ratio_mean" if space == "ratio" else "r2_transfer_mean"
    v = rec.get(key)
    return float(v) if v is not None else float("nan")


def _ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks with tie handling (deterministic, numpy-only)."""
    vals, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts + 1
    avg = (start + csum) / 2.0
    return avg[inv]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    rx, ry = _ranks(x), _ranks(y)
    c = np.corrcoef(rx, ry)[0, 1]
    return float(c)


def _cell_stats(
    reduced: list[float],
    ambient: list[float],
    n_class_pairs: int,
    *,
    boot_draws: int,
    rng: np.random.Generator,
) -> dict:
    """One calibration cell (plan §3): rho + pair-cluster bootstrap CI, median
    Delta, coverage, attenuation-limited label (I4: point rho AND CI reported
    for EVERY cell, exempted or not)."""
    red = np.asarray(reduced, dtype=np.float64)
    amb = np.asarray(ambient, dtype=np.float64)
    ok = np.isfinite(red) & np.isfinite(amb)
    n_usable = int(ok.sum())
    coverage = n_usable / n_class_pairs if n_class_pairs else float("nan")
    red_ok, amb_ok = red[ok], amb[ok]
    rho = _spearman(red_ok, amb_ok)
    deltas = red_ok - amb_ok
    median_delta = float(np.median(deltas)) if deltas.size else float("nan")
    rhos = []
    for _ in range(boot_draws):
        if n_usable < 3:
            break
        idx = rng.integers(0, n_usable, size=n_usable)
        r = _spearman(red_ok[idx], amb_ok[idx])
        if np.isfinite(r):
            rhos.append(r)
    rho_ci = (
        [float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))]
        if rhos
        else [float("nan"), float("nan")]
    )
    amb_iqr = (
        float(np.percentile(amb_ok, 75) - np.percentile(amb_ok, 25))
        if amb_ok.size
        else float("nan")
    )
    attenuation = bool(
        np.isfinite(amb_iqr)
        and amb_iqr < ATTEN_IQR_BAR
        and np.isfinite(rho_ci[1])
        and rho_ci[1] >= ATTEN_RHO_CI_UPPER
    )
    return {
        "n_class_pairs": n_class_pairs,
        "n_usable": n_usable,
        "coverage": coverage,
        "rho": rho,
        "rho_ci95": rho_ci,
        "median_delta": median_delta,
        "ambient_iqr": amb_iqr,
        "attenuation_limited": attenuation,
        "attenuation_rule": (
            f"ambient within-class IQR < {ATTEN_IQR_BAR} AND rho CI upper >= "
            f"{ATTEN_RHO_CI_UPPER} — exempts the rho leg ONLY (Delta + coverage still gate)"
        ),
    }


def _calibration_cells(units_full: list[dict], boot_draws: int, seed: int) -> dict:
    """Per (control class x rung x arm) calibration cells — 6 gating CONTEXT
    cells (ratio space) + 6 descriptive PREFIX cells (raw R^2 space; near-zero
    prefix ceilings make prefix ratios ratios-of-noise — plan §3 blocker 3a)."""
    cells: dict[str, dict] = {}
    for arm in ("context", "prefix"):
        space = "ratio" if arm == "context" else "r2"
        for cls in CONTROL_CLASSES:
            class_units = [
                u
                for u in units_full
                if u["class"] == cls and u["arm"] == arm and not u["flagged_k_realized_lt_k_star"]
            ]
            n_class = len(
                {
                    (u["source"], u["target"])
                    for u in units_full
                    if u["class"] == cls and u["arm"] == arm
                }
            )
            for rung in RB_RUNGS:
                rng = np.random.default_rng(
                    int(
                        hashlib.sha256(f"cal|{cls}|{rung}|{arm}|{seed}".encode()).hexdigest()[:8],
                        16,
                    )
                )
                reduced = [_unit_rung_val(u, rung, space) for u in class_units]
                ambient = [_parent_rung_val(u, rung, space) for u in class_units]
                cell = _cell_stats(reduced, ambient, n_class, boot_draws=boot_draws, rng=rng)
                cell["space"] = space
                cell["gating"] = arm == "context"
                cells[f"{cls}|{rung}|{arm}"] = cell
    return cells


def _delta_decomposition(units_full: list[dict]) -> list[dict]:
    """S-C1 (control units, context arm): split calibration Delta into its
    denominator-convention component (parent full-pool ceiling vs ambient
    matched-intersection ceiling) and its truncation component."""
    rows = []
    for u in units_full:
        if not u.get("control") or u["arm"] != "context":
            continue
        amb_matched = (u["pooled"].get("ambient_matched_ceiling_r2") or {}).get("mean")
        fullpool = u.get("parent_fullpool_ceiling_reference")
        if amb_matched is None or fullpool is None:
            continue
        for rung in RB_RUNGS:
            r2_amb = _parent_rung_val(u, rung, "r2")
            ratio_amb_parent = _parent_rung_val(u, rung, "ratio")
            ratio_reduced = _unit_rung_val(u, rung, "ratio")
            if not (np.isfinite(r2_amb) and np.isfinite(ratio_amb_parent)):
                continue
            ratio_amb_matched = (
                r2_amb / amb_matched if abs(float(amb_matched)) > 1e-12 else float("nan")
            )
            rows.append(
                {
                    "pair": _pair_key(u["source"], u["target"]),
                    "class": u["class"],
                    "rung": rung,
                    "ratio_reduced": ratio_reduced,
                    "ratio_ambient_parent_fullpool": ratio_amb_parent,
                    "ratio_ambient_matched_rows": ratio_amb_matched,
                    "delta_total": ratio_reduced - ratio_amb_parent,
                    "delta_denominator_convention": ratio_amb_matched - ratio_amb_parent,
                    "delta_truncation": ratio_reduced - ratio_amb_matched,
                }
            )
    return rows


def _pair_map(units: list[dict], cls: str, arm: str, *, level: int | None = "any") -> dict:
    out = {}
    for u in units:
        if u["class"] != cls or u["arm"] != arm:
            continue
        if level != "any" and u.get("level") != level:
            continue
        out[_pair_key(u["source"], u["target"])] = u
    return out


def _boot_ci_median(deltas: np.ndarray, boot_draws: int, rng: np.random.Generator) -> list[float]:
    meds = []
    n = deltas.size
    for _ in range(boot_draws):
        if n < 2:
            break
        idx = rng.integers(0, n, size=n)
        meds.append(float(np.median(deltas[idx])))
    if not meds:
        return [float("nan"), float("nan")]
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def _nslope(
    units_full: list[dict], units_matched: list[dict], census: dict, boot_draws: int, seed: int
) -> dict:
    """H0'b n-slope curves: slope(rung, level) = median over boundary pairs of
    [read(n_sub) - read(full n)]; context in ratio space, the N1p prefix
    anchor in raw R^2 space. I5: the H0'b PASS bar (0.05) NEVER licenses the
    unmatched-n companion margin (bar 0.025) — kept mechanically distinct."""
    levels = [int(x) for x in census["n1_levels"]]
    full_ctx = _pair_map(units_full, "boundary", "context", level=None)
    full_pfx = _pair_map(units_full, "boundary", "prefix", level=None)
    prose_ctx = _pair_map(units_full, "prose", "context", level=None)

    prose_ns = sorted(u["n_intersection_full"] for u in prose_ctx.values())
    prose_median_n = float(np.median(prose_ns)) if prose_ns else float("nan")
    prose_matched_level = (
        min(levels, key=lambda lv: (abs(lv - prose_median_n), lv))
        if np.isfinite(prose_median_n)
        else levels[0]
    )

    curves: dict[str, dict] = {}
    for rung in RB_RUNGS:
        per_level = {}
        for lv in levels:
            matched = _pair_map(units_matched, "boundary", "context", level=lv)
            deltas, n_flagged = [], 0
            for pk, mu in matched.items():
                fu = full_ctx.get(pk)
                if fu is None:
                    continue
                if mu["flagged_k_realized_lt_k_star"] or fu["flagged_k_realized_lt_k_star"]:
                    n_flagged += 1
                    continue
                d = _unit_rung_val(mu, rung, "ratio") - _unit_rung_val(fu, rung, "ratio")
                if np.isfinite(d):
                    deltas.append(d)
            arr = np.asarray(deltas, dtype=np.float64)
            rng = np.random.default_rng(
                int(hashlib.sha256(f"nslope|{rung}|{lv}|{seed}".encode()).hexdigest()[:8], 16)
            )
            per_level[str(lv)] = {
                "slope_median": float(np.median(arr)) if arr.size else float("nan"),
                "slope_ci95": _boot_ci_median(arr, boot_draws, rng),
                "n_pairs": int(arr.size),
                "n_flagged_excluded": n_flagged,
            }
        curves[rung] = per_level

    pfx_level = int(census["n1p_level"])
    matched_pfx = _pair_map(units_matched, "boundary", "prefix", level=pfx_level)
    pfx_anchor: dict[str, dict] = {}
    for rung in RB_RUNGS:
        deltas = []
        for pk, mu in matched_pfx.items():
            fu = full_pfx.get(pk)
            if (
                fu is None
                or mu["flagged_k_realized_lt_k_star"]
                or fu["flagged_k_realized_lt_k_star"]
            ):
                continue
            d = _unit_rung_val(mu, rung, "r2") - _unit_rung_val(fu, rung, "r2")
            if np.isfinite(d):
                deltas.append(d)
        arr = np.asarray(deltas, dtype=np.float64)
        rng = np.random.default_rng(
            int(hashlib.sha256(f"nslopepfx|{rung}|{seed}".encode()).hexdigest()[:8], 16)
        )
        pfx_anchor[rung] = {
            "level": pfx_level,
            "space": "raw_r2 (prefix ratios are ratios-of-noise — plan §3)",
            "slope_median": float(np.median(arr)) if arr.size else float("nan"),
            "slope_ci95": _boot_ci_median(arr, boot_draws, rng),
            "n_pairs": int(arr.size),
        }

    slope_r9 = curves["9_full_AMB"][str(prose_matched_level)]["slope_median"]
    h0b_pass = bool(np.isfinite(slope_r9) and abs(slope_r9) <= H0B_SLOPE_BAR)
    return {
        "curves_context_ratio": curves,
        "prefix_anchor_raw_r2": pfx_anchor,
        "prose_median_intersection": prose_median_n,
        "prose_matched_level": prose_matched_level,
        "h0b": {
            "slope_rung9_at_prose_matched_level": slope_r9,
            "bar": H0B_SLOPE_BAR,
            "pass": h0b_pass,
            "slope_exceeds_margin_bar": bool(
                np.isfinite(slope_r9) and abs(slope_r9) > H2_MARGIN_BAR
            ),
            "i5_note": (
                "H0'b PASS bar (0.05) is 2x the H2' margin bar (0.025): a slope in "
                "(0.025, 0.05] PASSES H0'b while unmatched-n margins remain "
                "instrument-manufactured per the registered rule — an H0'b PASS NEVER "
                "licenses the unmatched-n companion margin as a verdict input."
            ),
        },
    }


def _quantile_level_assignment(
    boundary_ns: dict[str, int], prose_ns: list[int], levels: list[int]
) -> dict[str, int]:
    """H2' level assignment: each boundary pair rank-quantile-mapped onto the
    prose n-distribution, then assigned the NEAREST subsample level.
    Deterministic pure-python (n, key) sort — never bare argsort ties (#1946)."""
    ordered = sorted((n, k) for k, n in boundary_ns.items())
    prose_sorted = np.asarray(sorted(prose_ns), dtype=np.float64)
    out = {}
    for i, (_n, key) in enumerate(ordered):
        q = (i + 0.5) / len(ordered)
        target = float(np.quantile(prose_sorted, q))
        out[key] = min(levels, key=lambda lv: (abs(lv - target), lv))
    return out


def _cells_of_pairs(pair_keys: list[str]) -> list[str]:
    cells = set()
    for pk in pair_keys:
        s, t = pk.split("->")
        cells.add(s)
        cells.add(t)
    return sorted(cells)


def _h2_margins(
    units_full: list[dict], units_matched: list[dict], census: dict, boot_draws: int, seed: int
) -> dict:
    """H2' matched-n margin (verdict input) + the unmatched companion
    (annotated, never a verdict input). CI at the CELL grain (I3): pairs share
    source/target cells, so pair-grain resampling would UNDERSTATE the CI."""
    levels = [int(x) for x in census["n1_levels"]]
    full_b = _pair_map(units_full, "boundary", "context", level=None)
    prose = _pair_map(units_full, "prose", "context", level=None)
    boundary_ns = {pk: u["n_intersection_full"] for pk, u in full_b.items()}
    prose_ns = [u["n_intersection_full"] for u in prose.values()]
    assigned = _quantile_level_assignment(boundary_ns, prose_ns, levels)

    def _boundary_matched_vals() -> dict[str, float]:
        out = {}
        for pk, lv in assigned.items():
            mu = _pair_map(units_matched, "boundary", "context", level=lv).get(pk)
            if mu is not None and not mu["flagged_k_realized_lt_k_star"]:
                v = _unit_rung_val(mu, "9_full_AMB", "ratio")
                if np.isfinite(v):
                    out[pk] = v
        return out

    def _vals(pmap: dict) -> dict[str, float]:
        out = {}
        for pk, u in pmap.items():
            if u["flagged_k_realized_lt_k_star"]:
                continue
            v = _unit_rung_val(u, "9_full_AMB", "ratio")
            if np.isfinite(v):
                out[pk] = v
        return out

    b_matched = _boundary_matched_vals()
    b_full = _vals(full_b)
    p_vals = _vals(prose)
    margin_matched = (
        float(np.median(list(b_matched.values())) - np.median(list(p_vals.values())))
        if b_matched and p_vals
        else float("nan")
    )
    margin_unmatched = (
        float(np.median(list(b_full.values())) - np.median(list(p_vals.values())))
        if b_full and p_vals
        else float("nan")
    )

    # CELL-grain cluster bootstrap (I3): resample the underlying source/target
    # cells with replacement; keep pairs with BOTH endpoints present.
    cells = _cells_of_pairs(sorted(set(b_matched) | set(p_vals)))
    rng = np.random.default_rng(
        int(hashlib.sha256(f"h2margin|{seed}".encode()).hexdigest()[:8], 16)
    )
    draws, skipped = [], 0
    for _ in range(boot_draws):
        pick = set(rng.choice(len(cells), size=len(cells), replace=True).tolist())
        keep = {cells[i] for i in pick}
        b = [v for pk, v in b_matched.items() if all(c in keep for c in pk.split("->"))]
        p = [v for pk, v in p_vals.items() if all(c in keep for c in pk.split("->"))]
        if not b or not p:
            skipped += 1
            continue
        draws.append(float(np.median(b) - np.median(p)))
    ci = (
        [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
        if draws
        else [float("nan"), float("nan")]
    )
    return {
        "assigned_levels": assigned,
        "margin_matched": margin_matched,
        "margin_matched_ci95_cell_cluster": ci,
        "margin_ci_boot_draws": len(draws),
        "margin_ci_boot_skipped_empty": skipped,
        "n_boundary_matched": len(b_matched),
        "n_prose": len(p_vals),
        "margin_unmatched_companion": {
            "value": margin_unmatched,
            "annotation": (
                "companion-only — NEVER a verdict input (mismatched n; plan §3). "
                "I5: an H0'b PASS does not license this read."
            ),
        },
        "bar": H2_MARGIN_BAR,
    }


def _h1_band(units_full: list[dict], units_matched: list[dict], boot_draws: int, seed: int) -> dict:
    """H1' matched-n band: band from the above-floor stratum SUBSAMPLED to the
    below-floor n-range (N2); members = the below-floor pairs' native reads.
    Count bootstrap at the CELL level (S-C2: below-floor pairs share cells, so
    binomial intuition understates the count's variance)."""
    n2 = {
        _pair_key(u["source"], u["target"]): u
        for u in units_matched
        if u["class"] == "twobytwo" and u["arm"] == "context"
    }
    floor_used = None
    below = {}
    above_native = {}
    for u in units_full:
        if u["class"] != "twobytwo" or u["arm"] != "context":
            continue
        pk = _pair_key(u["source"], u["target"])
        if pk in n2:
            above_native[pk] = u
        else:
            below[pk] = u
    band_vals = {}
    for pk, u in n2.items():
        if u["flagged_k_realized_lt_k_star"]:
            continue
        v = _unit_rung_val(u, "9_full_AMB", "ratio")
        if np.isfinite(v):
            band_vals[pk] = v
    member_vals = {}
    n_member_flagged = 0
    for pk, u in below.items():
        if u["flagged_k_realized_lt_k_star"]:
            n_member_flagged += 1
            continue
        v = _unit_rung_val(u, "9_full_AMB", "ratio")
        if np.isfinite(v):
            member_vals[pk] = v
    if not band_vals:
        raise RuntimeError("H1' band source EMPTY (no unflagged N2 reads) — cannot form the band")
    bv = np.asarray(list(band_vals.values()), dtype=np.float64)
    band = [float(np.percentile(bv, 2.5)), float(np.percentile(bv, 97.5))]
    inband = sum(1 for v in member_vals.values() if band[0] <= v <= band[1])
    n_unflagged = len(member_vals)
    bar = math.ceil(INBAND_FRAC * n_unflagged) if n_unflagged else 0

    nv_native = np.asarray(
        [
            _unit_rung_val(u, "9_full_AMB", "ratio")
            for u in above_native.values()
            if not u["flagged_k_realized_lt_k_star"]
        ],
        dtype=np.float64,
    )
    nv_native = nv_native[np.isfinite(nv_native)]
    native_band = (
        [float(np.percentile(nv_native, 2.5)), float(np.percentile(nv_native, 97.5))]
        if nv_native.size
        else [float("nan"), float("nan")]
    )

    cells = _cells_of_pairs(sorted(set(band_vals) | set(member_vals)))
    rng = np.random.default_rng(int(hashlib.sha256(f"h1band|{seed}".encode()).hexdigest()[:8], 16))
    hits, skipped, counts = 0, 0, []
    for _ in range(boot_draws):
        pick = set(rng.choice(len(cells), size=len(cells), replace=True).tolist())
        keep = {cells[i] for i in pick}
        b = [v for pk, v in band_vals.items() if all(c in keep for c in pk.split("->"))]
        m = {pk: v for pk, v in member_vals.items() if all(c in keep for c in pk.split("->"))}
        if len(b) < 3 or not m:
            skipped += 1
            continue
        lo, hi = np.percentile(np.asarray(b), 2.5), np.percentile(np.asarray(b), 97.5)
        cnt = sum(1 for v in m.values() if lo <= v <= hi)
        counts.append(cnt / len(m))
        if cnt >= math.ceil(INBAND_FRAC * len(m)):
            hits += 1
    return {
        "band_matched_n_2p5_97p5": band,
        "band_source_n": int(bv.size),
        "inband_pairs": int(inband),
        "n_unflagged": int(n_unflagged),
        "n_member_flagged_excluded": int(n_member_flagged),
        "inband_bar": int(bar),
        "inband_frac": INBAND_FRAC,
        "pass": bool(n_unflagged and inband >= bar),
        "p_inband_ge_bar_cell_bootstrap": (hits / len(counts)) if counts else float("nan"),
        "cell_bootstrap_draws": len(counts),
        "cell_bootstrap_skipped": skipped,
        "native_band_companion": {
            "band": native_band,
            "annotation": "unmatched-n (native above-floor) band — annotated companion only",
        },
    }


def _verdict_lattice(
    ctx_cells: dict[str, dict],
    inband_pairs: int,
    n_unflagged: int,
    margin: float,
    margin_ci_lo: float,
    *,
    coverage_bar_pct: float = VOID_COVERAGE_PCT,
    delta_bar_pp: float = VOID_DELTA_PP,
    rho_bar_x100: float = VOID_RHO_X100,
    inband_frac: float = INBAND_FRAC,
    margin_bar: float = H2_MARGIN_BAR,
) -> dict:
    """Plan §3 verdict lattice (DISJOINT + exhaustive). ctx_cells = the 6
    gating CONTEXT cells. The attenuation exemption is scoped to the rho leg
    ONLY; when every cell is attenuation-limited the rho minimum is vacuous
    and that leg passes (I4: prominently flagged — Void then rests on Delta +
    coverage alone)."""
    cells = list(ctx_cells.values())
    if len(cells) != 6:
        raise AssertionError(f"verdict lattice expects exactly 6 context cells, got {len(cells)}")
    worst_cov = 100.0 * min(c["coverage"] for c in cells)
    worst_delta = 100.0 * max(
        abs(c["median_delta"]) if np.isfinite(c["median_delta"]) else float("inf") for c in cells
    )
    gating_rhos = [c["rho"] for c in cells if not c["attenuation_limited"]]
    all_atten = not gating_rhos
    worst_rho = (
        None
        if all_atten
        else 100.0 * min(r if np.isfinite(r) else float("-inf") for r in gating_rhos)
    )
    void = (
        worst_cov < coverage_bar_pct
        or worst_delta > delta_bar_pp
        or (worst_rho is not None and worst_rho < rho_bar_x100)
    )
    inband_bar = math.ceil(inband_frac * n_unflagged) if n_unflagged else 0
    margin_pp = 100.0 * margin if np.isfinite(margin) else float("nan")
    margin_ci_lo_pp = 100.0 * margin_ci_lo if np.isfinite(margin_ci_lo) else float("nan")
    consistent = (
        not void
        and n_unflagged > 0
        and inband_pairs >= inband_bar
        and np.isfinite(margin_pp)
        and margin_pp >= 100.0 * margin_bar
        and np.isfinite(margin_ci_lo_pp)
        and margin_ci_lo_pp > 0.0
    )
    verdict = "Void" if void else ("Consistent" if consistent else "Inconsistent")
    return {
        "verdict": verdict,
        "worst_cell_coverage_pct": worst_cov,
        "worst_cell_delta_pp": worst_delta,
        "worst_gating_rho_x100": worst_rho,
        "all_cells_attenuation_limited": all_atten,
        "n_attenuation_exempted": sum(1 for c in cells if c["attenuation_limited"]),
        "i4_note": (
            "ALL SIX context gating cells are attenuation-limited — the Void gate rests on "
            "Delta + coverage alone (rho leg vacuous)."
            if all_atten
            else "attenuation exemption scoped to the rho leg only; Delta + coverage gate all 6"
        ),
        "inband_pairs": int(inband_pairs),
        "n_unflagged": int(n_unflagged),
        "inband_bar": int(inband_bar),
        "margin_pp": margin_pp,
        "margin_ci_lo_pp": margin_ci_lo_pp,
        "bars": {
            "coverage_pct": coverage_bar_pct,
            "delta_pp": delta_bar_pp,
            "rho_x100": rho_bar_x100,
            "inband_frac": inband_frac,
            "margin_pp": 100.0 * margin_bar,
        },
    }


def _pair_rows(units: list[dict]) -> list[dict]:
    """Compact per-unit rows for the class digest + figures."""
    rows = []
    for u in units:
        row = {
            "source": u["source"],
            "target": u["target"],
            "pair": _pair_key(u["source"], u["target"]),
            "class": u["class"],
            "arm": u["arm"],
            "level": u.get("level"),
            "stratum": u.get("stratum"),
            "n_intersection_full": u["n_intersection_full"],
            "n_used": u["n_used"],
            "flagged": u["flagged_k_realized_lt_k_star"],
            "chat_anchor": _is_chat_anchor_key(u["source"]) or _is_chat_anchor_key(u["target"]),
            "ratio_k_star": {r: _unit_rung_val(u, r, "ratio") for r in RB_RUNGS},
            "r2_k_star": {r: _unit_rung_val(u, r, "r2") for r in RB_RUNGS},
            "ceiling_r2_k_star": (u["pooled"]["k_star"]["ceiling_r2"] or {}).get("mean"),
            "ratio_per_k": {
                kstr: {
                    r: (u["pooled"]["per_k"][kstr][r]["ratio"] or {}).get("mean") for r in RB_RUNGS
                }
                for kstr in u["pooled"]["per_k"]
            },
            "ambient_ratio_record_only": {r: _parent_rung_val(u, r, "ratio") for r in RB_RUNGS},
            "ambient_r2_record_only": {r: _parent_rung_val(u, r, "r2") for r in RB_RUNGS},
        }
        rows.append(row)
    return rows


def run_merge(args: argparse.Namespace) -> int:
    units_full = _load_units(Path(args.units_dir))
    units_matched = _load_units(Path(args.matchedn_dir))
    census = _load_census(args.census)
    grid = _grid_assert(units_full, units_matched, census)
    _log(
        f"merge: grid COMPLETE ({grid['full_units']} full-n + {grid['matched_units']} matched-n units)"
    )

    boot = int(args.merge_boot_draws)
    seed = int(args.seed)
    cells = _calibration_cells(units_full, boot, seed)
    decomp = _delta_decomposition(units_full)
    nslope = _nslope(units_full, units_matched, census, boot, seed)
    h2 = _h2_margins(units_full, units_matched, census, boot, seed)
    h1 = _h1_band(units_full, units_matched, boot, seed)

    ctx_cells = {k: v for k, v in cells.items() if v["gating"]}
    lattice = _verdict_lattice(
        ctx_cells,
        h1["inband_pairs"],
        h1["n_unflagged"],
        h2["margin_matched"],
        h2["margin_matched_ci95_cell_cluster"][0],
    )
    sensitivity = {}
    for label, kw in (
        ("void_bars_x0.5", {"coverage_bar_pct": 37.5, "delta_pp": 7.5, "rho": 25.0}),
        ("void_bars_x1.5", {"coverage_bar_pct": 100.0, "delta_pp": 22.5, "rho": 75.0}),
        ("inband_56_of_64", {"inband_frac": 56 / 64}),
        ("margin_0.015", {"margin_bar": 0.015}),
    ):
        sensitivity[label] = _verdict_lattice(
            ctx_cells,
            h1["inband_pairs"],
            h1["n_unflagged"],
            h2["margin_matched"],
            h2["margin_matched_ci95_cell_cluster"][0],
            coverage_bar_pct=kw.get("coverage_bar_pct", VOID_COVERAGE_PCT),
            delta_bar_pp=kw.get("delta_pp", VOID_DELTA_PP),
            rho_bar_x100=kw.get("rho", VOID_RHO_X100),
            inband_frac=kw.get("inband_frac", INBAND_FRAC),
            margin_bar=kw.get("margin_bar", H2_MARGIN_BAR),
        )["verdict"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = _metadata()

    # I4: exempted-cell report (point rho AND CI for EVERY exempted cell).
    exempted = {
        k: {"rho": v["rho"], "rho_ci95": v["rho_ci95"]}
        for k, v in ctx_cells.items()
        if v["attenuation_limited"]
    }

    class_digest = {
        "phase": "rb789-merge",
        "k_star": int(args.k_star),
        "grid": grid,
        "per_class": _class_summaries(units_full),
        "chat_anchor_slice_h3": _class_summaries(
            [
                u
                for u in units_full
                if u["class"] == "twobytwo"
                and (_is_chat_anchor_key(u["source"]) or _is_chat_anchor_key(u["target"]))
            ]
        ),
        "pair_rows_full": _pair_rows(units_full),
        "pair_rows_matchedn": _pair_rows(units_matched),
        "ambient_label": "record-only (known-invalid, kept for comparability)",
        "metadata": md,
    }
    calibration_report = {
        "phase": "rb789-merge",
        "cells": cells,
        "gating_cells_context": sorted(ctx_cells),
        "attenuation_exempted_cells": exempted,
        "all_cells_attenuation_limited": lattice["all_cells_attenuation_limited"],
        "delta_decomposition_s_c1": decomp,
        "metadata": md,
    }
    nslope_report = {"phase": "rb789-merge", **nslope, "metadata": md}
    verdict = {
        "phase": "rb789-merge",
        "lattice": lattice,
        "bar_sensitivity": sensitivity,
        "h1_band": h1,
        "h2_margins": h2,
        "h0b": nslope["h0b"],
        "interpretive_boundary": (
            "k-dimensional linear reparameterization existence/strength — the registered "
            "FALLBACK read, strictly WEAKER than the ambient d x d read the task Goal names "
            "(plan §1); Void gates INTERPRETATION only, never spend"
        ),
        "metadata": md,
    }
    _write_json(out_dir / "rb789_class_digest.json", class_digest)
    _write_json(out_dir / "calibration_report.json", calibration_report)
    _write_json(out_dir / "nslope_report.json", nslope_report)
    _write_json(out_dir / "verdict.json", verdict)
    _log(
        f"merge: verdict={lattice['verdict']} inband={h1['inband_pairs']}/{h1['n_unflagged']} "
        f"(bar {h1['inband_bar']}) margin_pp={lattice['margin_pp']:.3f} -> {out_dir}"
    )
    print("[phase=done]", flush=True)
    return 0


def _class_summaries(units: list[dict]) -> dict:
    out: dict = {}
    for cls in sorted({u["class"] for u in units}):
        for arm in ("context", "prefix"):
            sel = [u for u in units if u["class"] == cls and u["arm"] == arm]
            if not sel:
                continue
            unflagged = [u for u in sel if not u["flagged_k_realized_lt_k_star"]]
            entry: dict = {
                "n_units": len(sel),
                "n_flagged": len(sel) - len(unflagged),
                "lead_space": "ratio" if arm == "context" else "raw_r2",
            }
            for rung in RB_RUNGS:
                ratios = np.asarray(
                    [_unit_rung_val(u, rung, "ratio") for u in unflagged], dtype=np.float64
                )
                r2s = np.asarray(
                    [_unit_rung_val(u, rung, "r2") for u in unflagged], dtype=np.float64
                )
                amb_ratio = np.asarray(
                    [_parent_rung_val(u, rung, "ratio") for u in unflagged], dtype=np.float64
                )
                amb_r2 = np.asarray(
                    [_parent_rung_val(u, rung, "r2") for u in unflagged], dtype=np.float64
                )
                entry[rung] = {
                    "ratio_median": float(np.nanmedian(ratios)) if ratios.size else float("nan"),
                    "r2_median": float(np.nanmedian(r2s)) if r2s.size else float("nan"),
                    "ambient_ratio_median_record_only": (
                        float(np.nanmedian(amb_ratio)) if amb_ratio.size else float("nan")
                    ),
                    "ambient_r2_median_record_only": (
                        float(np.nanmedian(amb_r2)) if amb_r2.size else float("nan")
                    ),
                }
            out[f"{cls}|{arm}"] = entry
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figures (P3) — read the merge digests; every PNG is sanity-checked on write
# (marker v222 figure-sanity duty; prefix figures LEAD with raw R^2 — S-C4).


def _load_merge_outputs(merge_dir: Path) -> dict:
    out = {}
    for name in ("rb789_class_digest", "calibration_report", "nslope_report", "verdict"):
        with (merge_dir / f"{name}.json").open(encoding="utf-8") as f:
            out[name] = json.load(f)
    return out


def _assert_png(path: Path) -> None:
    data = path.read_bytes()
    if len(data) < 2048 or not data.startswith(b"\x89PNG"):
        raise RuntimeError(f"figure sanity FAIL: {path} ({len(data)} bytes)")


def _finite_xy(xs: list, ys: list) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


_RUNG_COLORS = {
    "7_ctx_reparam": "tab:blue",
    "8_ans_reparam": "tab:orange",
    "9_full_AMB": "tab:green",
}
_CLASS_COLORS = {
    "boundary": "tab:blue",
    "model": "tab:purple",
    "prose": "tab:red",
    "twobytwo": "tab:green",
}


def run_figs(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = _load_merge_outputs(Path(args.merge_dir))
    out_dir = Path(args.figs_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = m["rb789_class_digest"]["pair_rows_full"]
    rows_m = m["rb789_class_digest"]["pair_rows_matchedn"]
    cells = m["calibration_report"]["cells"]
    curves = m["nslope_report"]["curves_context_ratio"]
    h1 = m["verdict"]["h1_band"]
    made: list[Path] = []

    def _save(fig, name: str) -> None:
        p = out_dir / name
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _assert_png(p)
        made.append(p)

    # 1 — HERO: calibration scatter (control pairs, context) + n-slope curves.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for rung in RB_RUNGS:
        sel = [
            r
            for r in rows
            if r["arm"] == "context" and r["class"] in CONTROL_CLASSES and not r["flagged"]
        ]
        x, y = _finite_xy(
            [r["ambient_ratio_record_only"][rung] for r in sel],
            [r["ratio_k_star"][rung] for r in sel],
        )
        cb = cells.get(f"boundary|{rung}|context", {})
        cm = cells.get(f"model|{rung}|context", {})
        ax.scatter(
            x,
            y,
            s=14,
            alpha=0.65,
            color=_RUNG_COLORS[rung],
            label=(
                f"rung {rung.split('_')[0]}  rho b={cb.get('rho', float('nan')):.2f} "
                f"m={cm.get('rho', float('nan')):.2f}; med-D b={cb.get('median_delta', float('nan')):.3f} "
                f"m={cm.get('median_delta', float('nan')):.3f}"
            ),
        )
    lims = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lims), max(lims)
    ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, zorder=0)
    ax.set_xlabel("ambient ratio (record-only, known-invalid)")
    ax.set_ylabel("reduced ratio @ k*")
    ax.set_title("H0'a calibration — control pairs (context)")
    ax.legend(fontsize=7)
    ax2 = axes[1]
    for rung in RB_RUNGS:
        levels = sorted(curves[rung], key=int)
        med = np.array([curves[rung][lv]["slope_median"] for lv in levels], dtype=np.float64)
        lo_ci = np.array([curves[rung][lv]["slope_ci95"][0] for lv in levels], dtype=np.float64)
        hi_ci = np.array([curves[rung][lv]["slope_ci95"][1] for lv in levels], dtype=np.float64)
        xs = np.array([int(lv) for lv in levels], dtype=np.float64)
        # matplotlib yerr = NON-NEGATIVE offsets, never bounds (gotchas.md).
        err = np.vstack([np.maximum(0, med - lo_ci), np.maximum(0, hi_ci - med)])
        ax2.errorbar(
            xs,
            med,
            yerr=err,
            marker="o",
            color=_RUNG_COLORS[rung],
            capsize=3,
            label=f"rung {rung.split('_')[0]}",
        )
    ax2.axhline(0.0, color="grey", lw=0.8)
    ax2.axhline(H0B_SLOPE_BAR, color="grey", lw=0.8, ls="--")
    ax2.axhline(-H0B_SLOPE_BAR, color="grey", lw=0.8, ls="--")
    ax2.set_xlabel("subsample level n_sub (boundary pairs)")
    ax2.set_ylabel("median [ratio(n_sub) - ratio(full n)]")
    ax2.set_title("H0'b measured n-slope (context, ratio)")
    ax2.legend(fontsize=8)
    _save(fig, "hero_calibration_nslope.png")

    # 2 — HERO: per-class rung distributions, old (grey, record-only) vs new,
    #     with the MATCHED-n H1' band overlaid on the twobytwo rung-9 panel.
    classes = ["boundary", "model", "prose", "twobytwo"]
    fig, axes = plt.subplots(len(RB_RUNGS), 1, figsize=(9, 10), sharex=True)
    rng_j = np.random.default_rng(7)
    for ri, rung in enumerate(RB_RUNGS):
        ax = axes[ri]
        for ci, cls in enumerate(classes):
            sel = [
                r for r in rows if r["arm"] == "context" and r["class"] == cls and not r["flagged"]
            ]
            old = np.asarray([r["ambient_ratio_record_only"][rung] for r in sel], dtype=np.float64)
            new = np.asarray([r["ratio_k_star"][rung] for r in sel], dtype=np.float64)
            jo = rng_j.uniform(-0.06, 0.06, size=old.size)
            jn = rng_j.uniform(-0.06, 0.06, size=new.size)
            ax.scatter(ci - 0.18 + jo, old, s=8, color="grey", alpha=0.5)
            ax.scatter(ci + 0.18 + jn, new, s=8, color=_CLASS_COLORS[cls], alpha=0.7)
        if rung == "9_full_AMB":
            band = h1["band_matched_n_2p5_97p5"]
            ci_tt = classes.index("twobytwo")
            ax.fill_between(
                [ci_tt - 0.35, ci_tt + 0.35], band[0], band[1], color="tab:green", alpha=0.15
            )
        ax.set_ylabel(f"ratio @ k* — rung {rung.split('_')[0]}")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes)
    axes[0].set_title(
        "old ambient (grey, record-only known-invalid) vs reduced (color); "
        "H1' matched-n band on twobytwo rung 9"
    )
    _save(fig, "hero_class_distributions.png")

    # 3 — k-profile curves (context, ratio) per class.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cls in classes:
        sel = [r for r in rows if r["arm"] == "context" and r["class"] == cls and not r["flagged"]]
        if not sel:
            continue
        ks = sorted({k for r in sel for k in r["ratio_per_k"]}, key=int)
        med = []
        for k in ks:
            vals = np.asarray(
                [
                    r["ratio_per_k"].get(k, {}).get("9_full_AMB")
                    for r in sel
                    if r["ratio_per_k"].get(k, {}).get("9_full_AMB") is not None
                ],
                dtype=np.float64,
            )
            med.append(float(np.nanmedian(vals)) if vals.size else float("nan"))
        ax.plot([int(k) for k in ks], med, marker="o", color=_CLASS_COLORS[cls], label=cls)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("k (reduced dimension)")
    ax.set_ylabel("median rung-9 ratio")
    ax.set_title("dimensionality profile (context; primary pre-registered at k*)")
    ax.legend(fontsize=8)
    _save(fig, "kprofile_curves.png")

    # 4 — old-vs-new per-pair scatter, rung 9, context, all classes.
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for cls in classes:
        sel = [r for r in rows if r["arm"] == "context" and r["class"] == cls and not r["flagged"]]
        x, y = _finite_xy(
            [r["ambient_ratio_record_only"]["9_full_AMB"] for r in sel],
            [r["ratio_k_star"]["9_full_AMB"] for r in sel],
        )
        ax.scatter(x, y, s=12, alpha=0.6, color=_CLASS_COLORS[cls], label=cls)
    lims = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lims), max(lims)
    ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, zorder=0)
    ax.set_xlabel("ambient rung-9 ratio (record-only, known-invalid)")
    ax.set_ylabel("reduced rung-9 ratio @ k*")
    ax.legend(fontsize=8)
    _save(fig, "oldnew_scatter_rung9.png")

    # 5 — matched-n vs full-n per-pair scatter (boundary, context, rung 9).
    fig, ax = plt.subplots(figsize=(6, 5.5))
    full_by_pair = {
        r["pair"]: r["ratio_k_star"]["9_full_AMB"]
        for r in rows
        if r["arm"] == "context" and r["class"] == "boundary" and not r["flagged"]
    }
    levels_present = sorted(
        {r["level"] for r in rows_m if r["class"] == "boundary" and r["arm"] == "context"}
    )
    cmap = {
        lv: c for lv, c in zip(levels_present, ("tab:blue", "tab:orange", "tab:green", "tab:red"))
    }
    for lv in levels_present:
        sel = [
            r
            for r in rows_m
            if r["class"] == "boundary"
            and r["arm"] == "context"
            and r["level"] == lv
            and not r["flagged"]
        ]
        x, y = _finite_xy(
            [full_by_pair.get(r["pair"], float("nan")) for r in sel],
            [r["ratio_k_star"]["9_full_AMB"] for r in sel],
        )
        ax.scatter(x, y, s=14, alpha=0.7, color=cmap.get(lv, "tab:grey"), label=f"n_sub={lv}")
    lims = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lims), max(lims)
    ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, zorder=0)
    ax.set_xlabel("full-n rung-9 ratio @ k*")
    ax.set_ylabel("matched-n rung-9 ratio @ k*")
    ax.set_title("n-slope raw view (boundary pairs)")
    ax.legend(fontsize=8)
    _save(fig, "matchedn_scatter_rung9.png")

    # 6 — prefix mirror: RAW R^2-LED (S-C4 — prefix ratios are ratios-of-noise).
    fig, axes = plt.subplots(len(RB_RUNGS), 1, figsize=(9, 10), sharex=True)
    for ri, rung in enumerate(RB_RUNGS):
        ax = axes[ri]
        for ci, cls in enumerate(classes):
            sel = [
                r for r in rows if r["arm"] == "prefix" and r["class"] == cls and not r["flagged"]
            ]
            old = np.asarray([r["ambient_r2_record_only"][rung] for r in sel], dtype=np.float64)
            new = np.asarray([r["r2_k_star"][rung] for r in sel], dtype=np.float64)
            jo = rng_j.uniform(-0.06, 0.06, size=old.size)
            jn = rng_j.uniform(-0.06, 0.06, size=new.size)
            ax.scatter(ci - 0.18 + jo, old, s=8, color="grey", alpha=0.5)
            ax.scatter(ci + 0.18 + jn, new, s=8, color=_CLASS_COLORS[cls], alpha=0.7)
        ax.set_ylabel(f"RAW R^2 — rung {rung.split('_')[0]}")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes)
    axes[0].set_title("prefix arm — RAW R^2 led (ratio annotated in digest; ~0.006 ceilings)")
    _save(fig, "prefix_raw_r2.png")

    _write_json(
        out_dir / "figs_meta.json", {"figures": [p.name for p in made], "metadata": _metadata()}
    )
    _log(f"figs: {len(made)} figure(s) rendered + sanity-checked -> {out_dir}")
    print("[phase=done]", flush=True)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# P0 smoke (plan §4): synthetic fixture, production CLI per shard, gates,
# resume, merge completeness, verdict-lattice branches, figs.


def _fixture_ceiling(acts: dict, arm: str, fold_map: dict) -> float:
    """Within-cell pooled ambient ceiling for the fixture parent JSONs."""
    X, Y, ids = ladder._select_arm(acts, arm)
    fold_of = fold_map["fold_of"]
    keep = [i for i, cid in enumerate(ids) if cid in fold_of]
    X, Y = X[keep], Y[keep]
    kept_ids = [ids[i] for i in keep]
    r2s = []
    for fold_i in range(int(fold_map["k"])):
        tr = [i for i, cid in enumerate(kept_ids) if int(fold_of[cid]) != fold_i]
        va = [i for i, cid in enumerate(kept_ids) if int(fold_of[cid]) == fold_i]
        if not tr or not va:
            continue
        model = ladder._fit_ridge(X[tr], Y[tr])
        r2s.append(ladder._r2_matrix(Y[va].astype(np.float64), ladder._apply_ridge(model, X[va])))
    return float(np.mean(r2s))


def _smoke_fixture(root: Path) -> dict:
    """Synthetic tiny fixture (plan §4 P0): n<=90 conversations, d=24, k=3
    folds, real 4-axis cell keys so ladder._pair_class classifies them."""
    rng = np.random.default_rng(20540811)
    d = 24
    pool = [f"conv{i:04d}" for i in range(90)]
    fold_of = {cid: i % 3 for i, cid in enumerate(pool)}
    fold_map_path = root / "shared_fold_map.json"
    _write_json(
        fold_map_path,
        {
            "fold_of": fold_of,
            "k": 3,
            "seed": 137,
            "variants": ["char_helios"],
            "n_conv_ids": len(pool),
        },
    )
    model_a, model_b = "qwen2.5-7b-instruct", "qwen2.5-7b"
    cell_specs = [
        ("char_helios", "inserted", "attrib_quoted", model_a, 80),
        ("char_helios", "inserted", "bare_label", model_a, 75),
        ("char_wren", "inserted", "attrib_quoted", model_a, 70),
        ("char_helios", "on_policy", "attrib_quoted", model_a, 58),
        (ladder.ASSISTANT_VARIANT, "inserted", "chat", model_a, 85),
        ("char_helios", "inserted", "attrib_quoted", model_b, 78),
    ]
    act_root = root / "activations"
    W1 = rng.normal(size=(d, 6))
    W2 = rng.normal(size=(6, d))
    for v, c, f, mdl, n in cell_specs:
        ids = sorted(rng.choice(pool, size=n, replace=False).tolist())
        X = rng.normal(size=(n, d)).astype(np.float32)
        Y = (X @ (W1 @ W2) * 0.25 + rng.normal(size=(n, d)) * 0.5).astype(np.float32)
        P = (X + rng.normal(size=(n, d)) * 0.3).astype(np.float32)
        pres = rng.random(n) > 0.06
        path = act_root / v / f"{forms.cell_key(v, c, f, mdl)}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, conv_id=np.array(ids), v_C=X, v_A=Y, v_P=P, v_P_present=pres)
    variants = sorted({v for v, *_ in cell_specs})
    conditions = ["inserted", "on_policy"]
    forms_list = ["chat", "attrib_quoted", "bare_label"]
    models = [model_a, model_b]
    cells = ladder._resolve_cells(act_root, variants, conditions, forms_list, models)
    if len(cells) != len(cell_specs):
        raise RuntimeError(f"fixture resolved {len(cells)} cells != {len(cell_specs)}")
    activations_by_cell = {
        ladder._cell_key(v, c, f, mdl): ladder._load_activation_npz(p) for v, c, f, mdl, p in cells
    }
    fold_map = load_fold_map(fold_map_path, allow_smoke=True)

    # Realized classes + intersections (the fixture census the merge asserts).
    class_pairs: dict[str, list] = {cls: [] for cls in PRODUCTION_CLASS_PAIR_COUNTS}
    all_pairs = []
    for s, t in ladder._enumerate_ordered_pairs(
        cells, smoke=False, pair_classes=tuple(CLASS_TO_PAIR_CLASS.values())
    ):
        cls = PAIR_CLASS_TO_CLASS[ladder._pair_class(s, t)]
        s_key, t_key = ladder._cell_key(*s[:4]), ladder._cell_key(*t[:4])
        class_pairs[cls].append((s_key, t_key))
        all_pairs.append((s_key, t_key, cls))
    tt_pairs = ladder._enumerate_ordered_pairs(cells, smoke=False, pair_classes=("twobytwo",))
    tt_n = _context_intersections(tt_pairs, activations_by_cell, fold_map["fold_of"])
    ns_unique = sorted(set(tt_n.values()))
    if len(ns_unique) < 2:
        raise RuntimeError("fixture twobytwo intersections degenerate — cannot form strata")
    best = max(
        ns_unique[1:],
        key=lambda c: min(
            sum(1 for n in tt_n.values() if n >= c), sum(1 for n in tt_n.values() if n < c)
        ),
    )
    floor = int(best)
    strata = (
        sum(1 for n in tt_n.values() if n >= floor),
        sum(1 for n in tt_n.values() if n < floor),
    )
    anchor_ctx = sorted(
        (s, t)
        for s, t in class_pairs["twobytwo"]
        if _is_chat_anchor_key(s) or _is_chat_anchor_key(t)
    )
    levels = [18, 30, 40]
    b_ns = _context_intersections(
        ladder._enumerate_ordered_pairs(cells, smoke=False, pair_classes=("cross_framing",)),
        activations_by_cell,
        fold_map["fold_of"],
    )
    if max(levels) >= min(b_ns.values()):
        raise RuntimeError(
            f"fixture levels {levels} not < min boundary intersection {min(b_ns.values())}"
        )
    census = {
        "class_pair_counts": {cls: len(ps) for cls, ps in class_pairs.items()},
        "chat_anchor_per_arm": len(anchor_ctx),
        "chat_anchor_pairs_context": [list(x) for x in anchor_ctx],
        "strata": {"above_floor": strata[0], "below_floor": strata[1]},
        "n1_levels": levels,
        "n1p_level": 30,
        "ambient_floor": floor,
    }
    census_path = root / "census.json"
    _write_json(census_path, census)

    # Parent-format ladder JSONs via the REAL parent code (gate-2 substrate:
    # bit-parity by construction — same code, same machine).
    parent_dir = root / "ladder_parent"
    parent_dir.mkdir(parents=True, exist_ok=True)
    for s_key, t_key, _cls in all_pairs:
        for arm in ("context", "prefix"):
            ceiling = _fixture_ceiling(activations_by_cell[t_key], arm, fold_map)
            rep = ladder._fit_arm_pair(
                source_cell_key=s_key,
                target_cell_key=t_key,
                arm=arm,
                source_acts=activations_by_cell[s_key],
                target_acts=activations_by_cell[t_key],
                fold_map=fold_map,
                target_ceiling=ceiling,
                n_rungs=len(ladder.RUNGS),
                seed=137,
                pilot=False,
                bootstrap_draws=20,
            )
            _write_json(
                parent_dir / f"rung_1_{s_key}_to_{t_key}_{arm}.json",
                {
                    "phase": "ladder",
                    "source": s_key,
                    "target": t_key,
                    "arm": arm,
                    "arm_report": rep,
                    "target_ceiling": ceiling,
                    "seed": 137,
                    "bootstrap_draws": 20,
                    "utc": _utc(),
                },
            )
    return {
        "fold_map_path": fold_map_path,
        "act_root": act_root,
        "parent_dir": parent_dir,
        "census_path": census_path,
        "floor": floor,
        "strata": strata,
        "levels": levels,
        "variants_csv": ",".join(variants),
        "conditions_csv": ",".join(conditions),
        "forms_csv": ",".join(forms_list),
        "models_csv": ",".join(models),
    }


def _smoke_lattice_probes() -> None:
    """Direct-call probes of every verdict-lattice branch (plan §4 P0:
    attenuation-limited + denominator-rule branches; I4/I5 semantics)."""

    def cell(rho=0.9, delta=0.01, cov=1.0, atten=False, ci_hi=None):
        return {
            "coverage": cov,
            "median_delta": delta,
            "rho": rho,
            "rho_ci95": [rho - 0.1, ci_hi if ci_hi is not None else rho + 0.1],
            "attenuation_limited": atten,
        }

    def cells6(**overrides):
        base = {f"c{i}": cell() for i in range(6)}
        base.update(overrides)
        return base

    assert _verdict_lattice(cells6(), 60, 64, 0.03, 0.005)["verdict"] == "Consistent"
    assert _verdict_lattice(cells6(c0=cell(cov=0.5)), 60, 64, 0.03, 0.005)["verdict"] == "Void"
    assert _verdict_lattice(cells6(c0=cell(delta=0.2)), 60, 64, 0.03, 0.005)["verdict"] == "Void"
    assert _verdict_lattice(cells6(c0=cell(rho=0.2)), 60, 64, 0.03, 0.005)["verdict"] == "Void"
    lat = _verdict_lattice(cells6(c0=cell(rho=0.2, atten=True, ci_hi=0.85)), 60, 64, 0.03, 0.005)
    assert lat["verdict"] == "Consistent" and lat["n_attenuation_exempted"] == 1
    lat = _verdict_lattice(
        {f"c{i}": cell(rho=0.1, atten=True, ci_hi=0.85) for i in range(6)}, 60, 64, 0.03, 0.005
    )
    assert lat["verdict"] == "Consistent" and lat["all_cells_attenuation_limited"]
    assert lat["worst_gating_rho_x100"] is None
    # Denominator rule (S-C2): flagged members shrink n_unflagged AND the bar.
    assert _verdict_lattice(cells6(), 55, 60, 0.03, 0.005)["inband_bar"] == math.ceil(0.906 * 60)
    assert _verdict_lattice(cells6(), 55, 60, 0.03, 0.005)["verdict"] == "Consistent"
    assert _verdict_lattice(cells6(), 54, 60, 0.03, 0.005)["verdict"] == "Inconsistent"
    assert _verdict_lattice(cells6(), 60, 64, 0.03, -0.001)["verdict"] == "Inconsistent"
    assert _verdict_lattice(cells6(), 60, 64, 0.01, 0.005)["verdict"] == "Inconsistent"
    _log("smoke probe: verdict-lattice branches OK (Void x3, attenuation, denominator, margin)")


def _run_cli(argv: list[str], what: str, *, expect_rc: int = 0) -> subprocess.CompletedProcess:
    r = subprocess.run(argv, capture_output=True, text=True, env={**os.environ})
    if (expect_rc == 0 and r.returncode != 0) or (expect_rc != 0 and r.returncode == 0):
        sys.stderr.write(f"--- {what} stdout tail ---\n" + "\n".join(r.stdout.splitlines()[-25:]))
        sys.stderr.write(f"\n--- {what} stderr tail ---\n" + "\n".join(r.stderr.splitlines()[-25:]))
        raise RuntimeError(
            f"{what}: rc={r.returncode} (expected {'0' if expect_rc == 0 else 'non-0'})"
        )
    return r


def run_smoke(args: argparse.Namespace) -> int:
    import shutil

    root = Path(args.smoke_root).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    _log(f"smoke: fixture root {root}")
    fx = _smoke_fixture(root)
    _log(f"smoke: fixture ready (floor={fx['floor']}, strata={fx['strata']})")

    # Probe: the production fold-map floor REFUSES the fixture map.
    refused = False
    try:
        load_fold_map(fx["fold_map_path"], allow_smoke=False)
    except RuntimeError as exc:
        refused = "REFUSING" in str(exc)
    if not refused:
        raise AssertionError("fold-map floor did NOT refuse the smoke map")
    _log("smoke probe: fold-map floor refusal OK")

    this = str(Path(__file__).resolve())
    full_out = root / "units_full"
    m_out = root / "units_matchedn"
    common = [
        "--activations-dir",
        str(fx["act_root"]),
        "--parent-ladder-dir",
        str(fx["parent_dir"]),
        "--fold-map",
        str(fx["fold_map_path"]),
        "--allow-smoke-fold-map",
        "--k-star",
        "8",
        "--k-profile",
        "2,4",
        "--bootstrap-draws",
        "20",
        "--seed",
        "137",
        "--ambient-floor",
        str(fx["floor"]),
        "--matchedn-levels",
        ",".join(str(x) for x in fx["levels"]),
        "--matchedn-prefix-level",
        "30",
        "--strata-expect",
        f"{fx['strata'][0]},{fx['strata'][1]}",
        "--skip-upload",
        "--variants",
        fx["variants_csv"],
        "--conditions",
        fx["conditions_csv"],
        "--forms",
        fx["forms_csv"],
        "--models",
        fx["models_csv"],
        "--hf-prefix",
        "issue2054_lattice/rb789_smoke_never_uploaded",
    ]
    for slug, cfg in SHARD_REGISTRY.items():
        if cfg["kind"] == "class":
            extra = [
                "--shard-class",
                cfg["shard_class"],
                "--arm",
                cfg["arm"],
                "--out-dir",
                str(full_out),
            ]
        else:
            extra = ["--matchedn", cfg["group"], "--out-dir", str(m_out)]
        _run_cli([sys.executable, this, "--mode", "run", *extra, *common], f"smoke shard {slug}")
        _log(f"smoke shard {slug}: rc=0")

    merge_out = root / "merge"
    merge_argv = [
        sys.executable,
        this,
        "--mode",
        "merge",
        "--units-dir",
        str(full_out),
        "--matchedn-dir",
        str(m_out),
        "--census",
        str(fx["census_path"]),
        "--out-dir",
        str(merge_out),
        "--merge-boot-draws",
        "60",
        "--k-star",
        "8",
        "--seed",
        "137",
    ]
    _run_cli(merge_argv, "smoke merge")
    with (merge_out / "verdict.json").open(encoding="utf-8") as f:
        verdict = json.load(f)
    for key in ("lattice", "bar_sensitivity", "h1_band", "h2_margins", "h0b"):
        if key not in verdict:
            raise AssertionError(f"verdict.json missing {key!r}")
    _log(f"smoke merge: verdict={verdict['lattice']['verdict']} OK")

    # Negative probe: merge REFUSES a partial lattice (one unit removed).
    scratch = root / "units_missing"
    shutil.copytree(full_out, scratch)
    victim = sorted(scratch.glob("rb789_*.json"))[0]
    victim.unlink()
    neg = [a if a != str(full_out) else str(scratch) for a in merge_argv]
    neg[neg.index("--out-dir") + 1] = str(root / "merge_neg")
    r = _run_cli(neg, "smoke merge-missing-unit", expect_rc=1)
    if "INCOMPLETE" not in (r.stderr + r.stdout):
        raise AssertionError("partial-lattice merge did not name the incomplete grid")
    _log("smoke probe: merge partial-lattice refusal OK")

    # Resume probes on boundary_context (same regime -> all resumed; changed
    # regime key -> full recompute; #722-r3).
    shard_argv = [
        sys.executable,
        this,
        "--mode",
        "run",
        "--shard-class",
        "boundary",
        "--arm",
        "context",
        "--out-dir",
        str(full_out),
        *common,
    ]
    _run_cli(shard_argv, "smoke resume-rerun")
    with (full_out / "digest__boundary_context.json").open(encoding="utf-8") as f:
        dg = json.load(f)
    if dg["n_pending_ran"] != 0 or dg["n_resumed"] != dg["n_units"]:
        raise AssertionError(f"resume probe A failed: {dg}")
    draws21 = [a if a != "20" else "21" for a in shard_argv]
    _run_cli(draws21, "smoke resume-regime-change")
    with (full_out / "digest__boundary_context.json").open(encoding="utf-8") as f:
        dg = json.load(f)
    if dg["n_pending_ran"] != dg["n_units"]:
        raise AssertionError(f"resume probe B (regime change) failed: {dg}")
    _log("smoke probe: resume skip + regime-change recompute OK")

    # In-process branch probes.
    _smoke_lattice_probes()
    fired = False
    try:
        _gate2_compare({"9_full_AMB": 0.5}, {"9_full_AMB": 0.6}, IDENTITY_GATE2_TOL)
    except AssertionError as exc:
        fired = "IDENTITY GATE 2 FAILED" in str(exc)
    if not fired:
        raise AssertionError("gate-2 mismatch branch did not fire")
    fired = False
    try:
        _draw_subsample(["a", "b"], 2, 1)
    except RuntimeError as exc:
        fired = "strict subsample" in str(exc)
    if not fired:
        raise AssertionError("subsample-infeasible branch did not fire")
    fired = False
    fence_report = root / "fence_probe.json"
    try:
        _enforce_fence(
            "probe",
            1.0,
            10.0,
            [{"n_used": 10}],
            3,
            argparse.Namespace(max_fleet_wall_hours=1e-9),
            fence_report,
            {},
        )
    except FleetWallExceeded:
        fired = fence_report.is_file()
    if not fired:
        raise AssertionError("FleetWallExceeded branch did not fire (or report missing)")
    _log("smoke probe: gate-2 FAIL / subsample-infeasible / fleet-fence branches OK")

    figs_out = root / "figs"
    _run_cli(
        [
            sys.executable,
            this,
            "--mode",
            "figs",
            "--merge-dir",
            str(merge_out),
            "--figs-out",
            str(figs_out),
        ],
        "smoke figs",
    )
    pngs = sorted(figs_out.glob("*.png"))
    if len(pngs) < 6:
        raise AssertionError(f"smoke figs: only {len(pngs)} PNGs rendered")
    for p in pngs:
        _assert_png(p)
    _log(f"smoke figs: {len(pngs)} PNGs rendered + sanity-checked")

    n_full = len(list(full_out.glob("rb789_*.json")))
    n_m = len(list(m_out.glob("rb789_*.json")))
    _log(
        f"SMOKE PASS: 10/10 shards via production CLI; {n_full} full-n + {n_m} matched-n units; "
        f"merge verdict={verdict['lattice']['verdict']}; gates + resume + lattice branches + "
        f"figs all exercised (root {root})"
    )
    print("[phase=done]", flush=True)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# import-check (smoke-architecture Axis 1): deferred imports executed +
# whole-module argparse-attribute completeness (self-contained AST — the
# canonical orchestrate.argcheck helper is main-only, absent from the
# issue-2054 branch the pods clone; scope is the WHOLE module, per
# code-style.md § Argparse-attribute completeness).


def _assert_args_attrs_defined() -> None:
    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    defined: set[str] = set()
    reads: set[str] = set()

    class _V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "add_argument":
                dest = None
                for kw in node.keywords:
                    if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                        dest = str(kw.value.value)
                if dest is None:
                    for a in node.args:
                        if (
                            isinstance(a, ast.Constant)
                            and isinstance(a.value, str)
                            and a.value.startswith("--")
                        ):
                            dest = a.value.lstrip("-").replace("-", "_")
                            break
                if dest:
                    defined.add(dest)
            if isinstance(fn, ast.Attribute) and fn.attr == "Namespace":
                for kw in node.keywords:
                    if kw.arg:
                        defined.add(kw.arg)
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if isinstance(node.value, ast.Name) and node.value.id == "args":
                if isinstance(node.ctx, ast.Load):
                    reads.add(node.attr)
                else:
                    defined.add(node.attr)
            self.generic_visit(node)

    _V().visit(tree)
    missing = sorted(reads - defined)
    if missing:
        raise AssertionError(f"args attributes READ but never DEFINED by the parser: {missing}")


def _import_check() -> int:
    import shutil  # noqa: F401  (deferred: run_smoke)

    import matplotlib  # noqa: F401  (deferred: run_figs)
    from huggingface_hub import HfApi  # noqa: F401  (deferred: _final_verify)

    _assert_args_attrs_defined()
    _log("import-check OK: deferred imports executed + args-attr completeness (whole module)")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _csv_int(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--mode",
        required=True,
        choices=["run", "merge", "figs", "smoke", "import-check", "list-shards"],
    )
    # run (shard)
    p.add_argument("--shard-class", choices=sorted(CLASS_TO_PAIR_CLASS), default=None)
    p.add_argument("--arm", choices=["context", "prefix"], default=None)
    p.add_argument("--matchedn", choices=["boundary", "twobytwo"], default=None)
    p.add_argument(
        "--activations-dir",
        default="data/issue_2054/rb789_stage/issue2054_lattice/activations",
        help="staged PARENT activation stores (READ-ONLY prefix mirror)",
    )
    p.add_argument(
        "--parent-ladder-dir",
        default="data/issue_2054/rb789_stage/issue2054_lattice/ladder",
        help="staged PARENT per-pair ladder JSONs (gate 2 + record-only columns)",
    )
    p.add_argument("--fold-map", default="eval_results/issue_2054/shared_fold_map.json")
    p.add_argument(
        "--allow-smoke-fold-map",
        action="store_true",
        help="P0 fixture ONLY — named gate downgrade (plan §10); the pod driver never passes it",
    )
    p.add_argument("--out-dir", default="data/issue_2054/rb789/ladder")
    p.add_argument("--k-star", type=int, default=K_STAR)
    p.add_argument("--k-profile", type=_csv_int, default=list(K_PROFILE))
    p.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--ambient-floor", type=int, default=AMBIENT_FLOOR)
    p.add_argument("--matchedn-levels", type=_csv_int, default=list(N1_LEVELS))
    p.add_argument("--matchedn-prefix-level", type=int, default=N1P_LEVEL)
    p.add_argument("--strata-expect", type=_csv_int, default=list(PRODUCTION_STRATA))
    p.add_argument("--max-fleet-wall-hours", type=float, default=DEFAULT_MAX_FLEET_WALL_HOURS)
    p.add_argument("--skip-pilot-gate", action="store_true")
    p.add_argument("--skip-upload", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--hf-prefix", default=DEFAULT_TASK_PREFIX)
    p.add_argument("--variants", type=_csv, default=list(ladder.DEFAULT_VARIANTS))
    p.add_argument("--conditions", type=_csv, default=list(ladder.DEFAULT_CONDITIONS))
    p.add_argument("--forms", type=_csv, default=list(ladder.DEFAULT_FORMS))
    p.add_argument("--models", type=_csv, default=list(ladder.DEFAULT_MODELS))
    # merge
    p.add_argument("--units-dir", default=None)
    p.add_argument("--matchedn-dir", default=None)
    p.add_argument("--census", default="production")
    p.add_argument("--merge-boot-draws", type=int, default=MERGE_BOOT_DRAWS)
    # figs
    p.add_argument("--merge-dir", default=None)
    p.add_argument("--figs-out", default="figures/issue_2054/reduced_basis_refit_rungs789")
    # smoke
    p.add_argument("--smoke-root", default="/tmp/rb789_smoke")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    if args.mode == "list-shards":
        for slug in SHARD_REGISTRY:
            print(slug)
        return 0
    if args.mode == "import-check":
        return _import_check()
    if args.mode == "run":
        return run_shard(args)
    if args.mode == "merge":
        if not args.units_dir or not args.matchedn_dir:
            raise SystemExit("--mode merge requires --units-dir and --matchedn-dir")
        return run_merge(args)
    if args.mode == "figs":
        if not args.merge_dir:
            raise SystemExit("--mode figs requires --merge-dir")
        return run_figs(args)
    if args.mode == "smoke":
        return run_smoke(args)
    raise SystemExit(f"unknown --mode {args.mode!r}")


if __name__ == "__main__":
    try:
        _rc = main()
    except FleetWallExceeded as exc:
        # Designed halt: the projection is persisted in the pilot report BEFORE
        # the raise — route on the artifact, never an anonymous crash (#1415).
        print(f"ERROR {exc}", file=sys.stderr)
        _rc = 7
    # Explicit exit before C-extension finalization (gotchas.md atexit race).
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(_rc)
