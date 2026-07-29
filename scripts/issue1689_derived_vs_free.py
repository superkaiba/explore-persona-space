"""Issue #1689 follow-up `derived-vs-free-answer-map` — items 1-6 (+ item 9 transfer leg).

Per ordered condition pair (within-model AND base<->instruct cross-model), both
mapping arms (prefix / context), this battery tests the mentor-posed
shared-readout conjugation hypothesis: with one readout y = xW + b per
condition and one affine context transfer x_T = x_S M + a, the answer-space
map is fully DERIVED —

  row convention:  y_T_hat = (y_S - b_S) @ W_S_pinv @ M @ W_S + a @ W_S + b_S
                   (== column-convention W M W^+ with bias -W M W^+ b + W a + b)

Three answer-map models per (pair, arm), ALL fits train-fold-only per outer
conv-grouped fold (the battery's ONLY fitting convention; the parent ladder's
all-rows W_s appears ONLY in the Gate-1 parity check):

  b_derived   = W_S^+ M W_S     (0 free y-space params, shared readout)
  b_derived2  = W_S^+ M W_T     (0 free params, per-condition readouts)
  b_free      = ridge y_S -> y_T (d^2 + d free params; the within-form ceiling)
  identity+bias baseline + kNN retrieval per the standing mapping rule.

W_S^+ via truncated SVD at ranks {32, 128, 512, eff-rank} (all reported;
conjugation amplifies noise along weak singular directions). Verdict lattice
(plan v8 s3, FOUR disjoint classes; Class 0 excluded from verdict counts):

  free_map_uninformative    <=> R2(b_free) < R2(identity+bias)
  shared_readout_supported  <=> g1 = R2(b_derived_max)  - 0.9 R2(b_free) >= 0
  readout_changed           <=> g1 < 0 and g2 = R2(b_derived2_max) - 0.9 R2(b_free) >= 0
  transfer_map_insufficient <=> otherwise

Phases (--phase): stage | gate1 | pairs | nulls | merge | upload | write-pairs.
Per-unit JSON checkpoints (skip-if-meta-matches resume, parent R13 convention);
compact SVD bundles under <out-root>/bundles/. Rotation nulls (item 5) run as a
SHARED-draw pass (--phase nulls) using the parent 9a-ter Procrustes battery's
exact Haar + singular-value reduction (seeds seed*1000003+k), which is
per-draw EXACTLY equal to the verbatim two-sided-rotation formula of
issue1345_operator_comparison.raw_cosine_with_rotation_null (von Neumann trace
identity; pinned by tests/test_issue1689_derived_vs_free.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# load_dotenv() BEFORE numpy/torch (shared-VM thread caps, #847).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.issue1689_fit_ladder as fl  # noqa: E402
from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from scripts.issue1689_common import (  # noqa: E402
    HEADLINE_LAYER,
    HF_DATA_PREFIX,
    LAMBDA_GRIDS,
    N_FOLDS,
    RUNG_REACHED_THRESHOLD,
    enumerate_pair_set,
)

MODEL_SLUGS = ("Qwen_Qwen2.5-7B", "Qwen_Qwen2.5-7B-Instruct")
DATA_REPO = "superkaiba1/explore-persona-space-data"
PINNED_STORE_REVISION = "d1010a25f81ce184f68a9cc0ed49bce9736b80dd"
STORE_HF_PREFIX = f"{HF_DATA_PREFIX}/analysis_tensors"
BUNDLE_HF_PREFIX = f"{HF_DATA_PREFIX}/derived_vs_free/analysis_tensors"
TRUNCATION_RANKS = (32, 128, 512)  # + per-fold eff-rank (scope item 2)
RANK_LABELS = ("r32", "r128", "r512", "effrank")
BATTERY_VERSION = "dvf-v1"
LOW_COMMON_FLAG = 500  # plan s8: a <500-common-row unit is flagged, not dropped
GATE1_PAIR = (
    ("Qwen_Qwen2.5-7B-Instruct", "assistant_chat"),
    ("Qwen_Qwen2.5-7B-Instruct", "assistant_naturalistic"),
)
GATE1_ATOL = 1e-3  # plan s7 Gate 1 (GPU-fp64 vs parent tolerance)


def _git_commit() -> str:
    """Best-effort repo commit for reproducibility metadata."""
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _metadata() -> dict:
    import torch

    return {
        "git_commit": _git_commit(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w") as fh:
        json.dump(obj, fh, indent=1)
    tmp.replace(path)


def _atomic_savez(path: Path, **arrays) -> None:
    """np.savez APPENDS .npz to non-.npz names — tmp must END in .npz (gotchas)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _haar(d: int, gen) -> "object":
    """Haar-orthogonal (d, d) fp64 sample — the parent 9a-ter battery convention
    (CPU randn from the caller's generator; QR + diag-sign fix runs on the
    generator's device stream via CPU then moves are the caller's business)."""
    import torch

    a = torch.randn(d, d, dtype=torch.float64, generator=gen)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r))


def _cos_flat(a, b) -> float:
    va, vb = a.reshape(-1), b.reshape(-1)
    return float((va @ vb) / (va.norm() * vb.norm() + 1e-12))


def regime_meta(args) -> dict:
    """Per-unit resume regime key — EVERY output-affecting knob (#722 r3 rule).

    Rotation-null draws are deliberately NOT here: nulls are a separate
    patch-in pass keyed by their own n_draws field inside the unit JSON.
    """
    return {
        "battery_version": BATTERY_VERSION,
        "layer": int(args.layer),
        "lambda_grid": str(args.lambda_grid),
        "seed": int(args.seed),
        "n_folds": int(N_FOLDS),
        "truncation_ranks": list(TRUNCATION_RANKS),
        "threshold": float(RUNG_REACHED_THRESHOLD),
        "row_limit": args.row_limit,
        "dim_limit": args.dim_limit,
    }


def unit_key(spec, arm: str) -> str:
    return f"{fl.pair_spec_key(spec)}__{arm}"


def verdict_class(r2_free: float, r2_ident: float, g1: float, g2: float) -> str:
    """FOUR-class disjoint+exhaustive verdict (plan v8 s3)."""
    if not np.isfinite(r2_free) or not np.isfinite(r2_ident):
        return "invalid"
    if r2_free < r2_ident:
        return "free_map_uninformative"
    if g1 >= 0:
        return "shared_readout_supported"
    if g2 >= 0:
        return "readout_changed"
    return "transfer_map_insufficient"


class _CellCache:
    """Tiny LRU over loaded (model, cond) cell bundles (~850 MB each fp64)."""

    def __init__(self, store_root: Path, layer: int, cap: int = 4):
        self.store_root, self.layer, self.cap = store_root, layer, cap
        self.d: dict = {}

    def get(self, model: str, cond: str) -> dict:
        key = (model, cond)
        if key in self.d:
            self.d[key] = self.d.pop(key)
            return self.d[key]
        v = fl._load_cell_layer(self.store_root, f"{model}/{cond}", self.layer)
        self.d[key] = v
        while len(self.d) > self.cap:
            self.d.pop(next(iter(self.d)))
        return v


def _pinv_apply(y_c, U, s, Vh, k: int):
    """rows @ pinv_k(W) with W = U diag(s) Vh (row convention y = x @ W)."""
    return ((y_c @ Vh[:k].T) * (1.0 / s[:k])) @ U[:, :k].T


def _pinv_matrix(U, s, Vh, k: int):
    return Vh[:k].T @ ((1.0 / s[:k])[:, None] * U[:, :k].T)


def run_unit(
    source: dict, target: dict, spec, arm: str, args, lams: np.ndarray
) -> tuple[dict, dict]:
    """Items 1-6 battery for one (ordered pair, arm). Returns (unit_json, bundle)."""
    import torch

    (sm, sc), (tm, tc) = spec
    t0 = time.perf_counter()
    common, s_idx, t_idx = fl.pair_rows_by_conv(source["conv_ids"], target["conv_ids"])
    if args.row_limit is not None:
        common, s_idx, t_idx = (a[: args.row_limit] for a in (common, s_idx, t_idx))
    n = len(common)
    if n < 3:
        return {"error": "insufficient shared conv_ids", "retryable": False, "n_common": int(n)}, {}
    dsl = slice(None) if args.dim_limit is None else slice(0, args.dim_limit)
    X_S = source[f"X_{arm}"][s_idx][:, dsl]
    Y_S = source["Y"][s_idx][:, dsl]
    X_T = target[f"X_{arm}"][t_idx][:, dsl]
    Y_T = target["Y"][t_idx][:, dsl]
    d = X_S.shape[1]

    folds = fl._conv_grouped_folds(common, n_folds=N_FOLDS, seed=args.seed)
    dev = torch.device(args.device)
    tX_S = torch.from_numpy(np.ascontiguousarray(X_S)).to(dev)
    tY_S = torch.from_numpy(np.ascontiguousarray(Y_S)).to(dev)
    tX_T = torch.from_numpy(np.ascontiguousarray(X_T)).to(dev)
    tY_T = torch.from_numpy(np.ascontiguousarray(Y_T)).to(dev)

    model_labels = (
        ["b_free", "identity_bias"]
        + [f"b_derived_{lab}" for lab in RANK_LABELS]
        + [f"b_derived2_{lab}" for lab in RANK_LABELS]
    )
    pooled_pred: dict[str, list[np.ndarray]] = {m: [] for m in model_labels}
    pooled_true: list[np.ndarray] = []
    per_fold_r2: dict[str, dict[int, float]] = {m: {} for m in model_labels}
    lambdas_chosen: dict[int, dict[str, float]] = {}
    eff_ranks: dict[int, float] = {}
    skipped_folds: list[int] = []
    canonical: dict = {}

    for k_fold in range(N_FOLDS):
        te_mask = folds == k_fold
        tr_mask = ~te_mask
        if tr_mask.sum() < 3 or te_mask.sum() < 1:
            skipped_folds.append(k_fold)
            continue
        tr = torch.from_numpy(np.where(tr_mask)[0]).to(dev)
        te = torch.from_numpy(np.where(te_mask)[0]).to(dev)
        conv_tr = common[np.where(tr_mask)[0]]
        # Train-fold-only fits — the battery's ONLY fitting convention (plan s4
        # item 2): W_S, W_T, M, a, b_S all exclude the fold's test rows.
        W_S, b_S, lam_ws = fl._fit_ridge_inner_group_cv_t(tX_S[tr], tY_S[tr], conv_tr, lams)
        W_T, b_T, lam_wt = fl._fit_ridge_inner_group_cv_t(tX_T[tr], tY_T[tr], conv_tr, lams)
        M, a_M, lam_m = fl._fit_ridge_inner_group_cv_t(tX_S[tr], tX_T[tr], conv_tr, lams)
        B_free, b_free, lam_bf = fl._fit_ridge_inner_group_cv_t(tY_S[tr], tY_T[tr], conv_tr, lams)
        lambdas_chosen[k_fold] = {"W_S": lam_ws, "W_T": lam_wt, "M": lam_m, "B_free": lam_bf}

        U, s, Vh = fl._svd_robust_t(W_S)
        s = s.clamp_min(1e-300)  # guard exact-zero singulars in 1/s (rank-deficient W_S)
        eff = float((s.sum() ** 2 / (s**2).sum()).item())
        eff_ranks[k_fold] = eff
        k_eff = int(max(1, min(round(eff), s.shape[0])))
        rank_map = {
            lab: min(int(r), int(s.shape[0])) for lab, r in zip(RANK_LABELS[:3], TRUNCATION_RANKS)
        }
        rank_map["effrank"] = k_eff

        Y_true_te = tY_T[te]
        pooled_true.append(Y_true_te.cpu().numpy())
        y_c = tY_S[te] - b_S
        mw_s = M @ W_S
        mw_t = M @ W_T
        aw_s = a_M @ W_S + b_S
        aw_t = a_M @ W_T + b_T
        for lab in RANK_LABELS:
            kk = rank_map[lab]
            xhat = _pinv_apply(y_c, U, s, Vh, kk)
            pred_d = xhat @ mw_s + aw_s
            pred_d2 = xhat @ mw_t + aw_t
            pooled_pred[f"b_derived_{lab}"].append(pred_d.cpu().numpy())
            pooled_pred[f"b_derived2_{lab}"].append(pred_d2.cpu().numpy())
            per_fold_r2[f"b_derived_{lab}"][k_fold] = fl._r2_t(Y_true_te, pred_d)
            per_fold_r2[f"b_derived2_{lab}"][k_fold] = fl._r2_t(Y_true_te, pred_d2)
        pred_free = tY_S[te] @ B_free + b_free
        pooled_pred["b_free"].append(pred_free.cpu().numpy())
        per_fold_r2["b_free"][k_fold] = fl._r2_t(Y_true_te, pred_free)
        tr_np = np.where(tr_mask)[0]
        te_np = np.where(te_mask)[0]
        pred_ident = identity_bias_predict(Y_S[tr_np], Y_T[tr_np], Y_S[te_np])
        pooled_pred["identity_bias"].append(pred_ident)
        per_fold_r2["identity_bias"][k_fold] = fl._r2(Y_T[te_np], pred_ident)

        if not canonical:  # canonical fold = FIRST completed fold (fold 0 by construction)
            canonical = {
                "fold": k_fold,
                "U": U,
                "s": s,
                "Vh": Vh,
                "rank_map": dict(rank_map),
                "M": M,
                "W_S": W_S,
                "W_T": W_T,
                "B_free": B_free,
                "eff_rank": eff,
            }

    if not pooled_true:
        return {
            "error": "all folds degenerate",
            "retryable": False,
            "n_common": int(n),
            "skipped_folds": skipped_folds,
        }, {}

    true_arr = np.concatenate(pooled_true, axis=0)
    r2_pooled: dict[str, float] = {}
    for m in model_labels:
        pred_arr = np.concatenate(pooled_pred[m], axis=0)
        r2_pooled[m] = fl._r2(true_arr, pred_arr)

    def _max_read(prefix: str) -> tuple[float, str]:
        vals = {lab: r2_pooled[f"{prefix}_{lab}"] for lab in RANK_LABELS}
        best = max(vals, key=lambda lab: np.nan_to_num(vals[lab], nan=-np.inf))
        return vals[best], best

    r2_free = r2_pooled["b_free"]
    r2_ident = r2_pooled["identity_bias"]
    r2_d_max, d_argmax = _max_read("b_derived")
    r2_d2_max, d2_argmax = _max_read("b_derived2")
    thr = RUNG_REACHED_THRESHOLD
    g1 = r2_d_max - thr * r2_free
    g2 = r2_d2_max - thr * r2_free
    g1_eff = r2_pooled["b_derived_effrank"] - thr * r2_free
    g2_eff = r2_pooled["b_derived2_effrank"] - thr * r2_free

    knn_models = {
        "b_free": "b_free",
        "identity_bias": "identity_bias",
        f"b_derived_{d_argmax}": "b_derived_argmax",
        "b_derived_effrank": "b_derived_effrank",
        f"b_derived2_{d2_argmax}": "b_derived2_argmax",
        "b_derived2_effrank": "b_derived2_effrank",
    }
    knn_out: dict[str, dict] = {}
    for src_label, out_label in knn_models.items():
        pred_arr = np.concatenate(pooled_pred[src_label], axis=0)
        knn_out[out_label] = {
            metric: knn_retrieval(pred_arr, true_arr, ks=(1, 5, 10), metric=metric)
            for metric in ("euclidean", "cosine")
        }

    # Operator-level read (item 5) on the canonical fold's train-fit operators.
    import torch as _torch  # local alias for clarity

    U, s, Vh = canonical["U"], canonical["s"], canonical["Vh"]
    rank_map = canonical["rank_map"]
    M, W_S, W_T, B_free = canonical["M"], canonical["W_S"], canonical["W_T"], canonical["B_free"]
    op_variants = {
        "derived_effrank": ("d", rank_map["effrank"]),
        f"derived_{d_argmax}": ("d", rank_map[d_argmax]),
        "derived2_effrank": ("d2", rank_map["effrank"]),
        f"derived2_{d2_argmax}": ("d2", rank_map[d2_argmax]),
    }
    raw_cos: dict[str, float] = {}
    svecs: dict[str, np.ndarray] = {}
    _, s_free, _ = fl._svd_robust_t(B_free)
    svecs["free"] = s_free.cpu().numpy()
    seen: dict[tuple, str] = {}
    for name, (kind, kk) in op_variants.items():
        if (kind, kk) in seen:  # argmax == effrank: reuse
            raw_cos[name] = raw_cos[seen[(kind, kk)]]
            svecs[name] = svecs[seen[(kind, kk)]]
            continue
        pinv_k = _pinv_matrix(U, s, Vh, kk)
        B_op = pinv_k @ (M @ (W_S if kind == "d" else W_T))
        raw_cos[name] = _cos_flat(B_op, B_free)
        _, s_op, _ = fl._svd_robust_t(B_op)
        svecs[name] = s_op.cpu().numpy()
        seen[(kind, kk)] = name
        del B_op

    # Compact bundle (plan s10): M-I top-256 factors fp16, spectra, per-fold R2.
    Mm = M - _torch.eye(d, dtype=_torch.float64, device=M.device)
    Um, sm_v, Vhm = fl._svd_robust_t(Mm)
    n_keep = min(256, d)
    rank_grid_r2 = np.array(
        [
            [per_fold_r2[f"b_derived_{lab}"].get(f, np.nan) for f in range(N_FOLDS)]
            for lab in RANK_LABELS
        ]
    )
    bundle = {
        "m_minus_i_u256_fp16": Um[:, :n_keep].cpu().numpy().astype(np.float16),
        "m_minus_i_vh256_fp16": Vhm[:n_keep].cpu().numpy().astype(np.float16),
        "m_minus_i_svals": sm_v.cpu().numpy(),
        "w_s_svals": s.cpu().numpy(),
        "svec_free": svecs["free"],
        "per_fold_rank_r2_derived": rank_grid_r2,
        "canonical_fold": np.int64(canonical["fold"]),
    }
    for name in op_variants:
        bundle[f"svec_{name}"] = svecs[name]

    unit = {
        "meta": regime_meta(args),
        "src_model": sm,
        "src_cond": sc,
        "tgt_model": tm,
        "tgt_cond": tc,
        "arm": arm,
        "pair_key": fl.pair_spec_key(spec),
        "unit_key": unit_key(spec, arm),
        "cross_model": sm != tm,
        "n_common": int(n),
        "d": int(d),
        "flag_low_common": bool(n < LOW_COMMON_FLAG),
        "skipped_folds": skipped_folds,
        "n_rows_pooled": int(true_arr.shape[0]),
        "lambdas_chosen": lambdas_chosen,
        "eff_rank_w_s_per_fold": eff_ranks,
        "rank_map_canonical": {k: int(v) for k, v in rank_map.items()},
        "r2_pooled": r2_pooled,
        "per_fold_r2": {m: {int(k): v for k, v in d_.items()} for m, d_ in per_fold_r2.items()},
        "r2_b_free": r2_free,
        "r2_identity_bias": r2_ident,
        "r2_b_derived_max": r2_d_max,
        "b_derived_argmax_rank": d_argmax,
        "r2_b_derived2_max": r2_d2_max,
        "b_derived2_argmax_rank": d2_argmax,
        "g1": g1,
        "g2": g2,
        "g1_fixed_effrank": g1_eff,
        "g2_fixed_effrank": g2_eff,
        "verdict": verdict_class(r2_free, r2_ident, g1, g2),
        "verdict_fixed_effrank": verdict_class(r2_free, r2_ident, g1_eff, g2_eff),
        "knn": knn_out,
        "operator_read": {
            "canonical_fold": canonical["fold"],
            "raw_cosine": raw_cos,
            "rotation_null": None,  # patched by --phase nulls
        },
        "wall_s": round(time.perf_counter() - t0, 2),
        "metadata": _metadata(),
    }
    return unit, bundle


# ---------------------------------------------------------------------------
# Pair-spec enumeration
# ---------------------------------------------------------------------------
def build_pair_specs(args) -> list:
    """Resolve the ordered pair-spec list from --pairs-file / --pair-set."""
    if args.pairs_file is not None:
        loaded = json.loads(Path(args.pairs_file).read_text())
        return fl.parse_pair_specs(loaded, default_model=args.default_model)
    if args.pair_set == "within-model":
        models = [m for m in args.models.split(",") if m]
        return [((m, s), (m, t)) for m in models for (s, t) in enumerate_pair_set()]
    if args.pair_set == "cross-model":
        return fl.crossmodel_pair_specs(MODEL_SLUGS[0], MODEL_SLUGS[1])
    raise ValueError(f"unknown --pair-set {args.pair_set!r}")


def _units(specs: list) -> list:
    return [(spec, arm) for spec in specs for arm in ("prefix", "context")]


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def cmd_pairs(args) -> int:
    lams = (
        fl.LAMBDAS if args.lambda_grid == "ladder13" else fl.resolve_lambda_grid(args.lambda_grid)
    )
    specs = build_pair_specs(args)
    units = _units(specs)
    shard_units = units[args.shard_index :: args.num_shards] if args.num_shards > 1 else units
    pairs_dir = args.out_root / "pairs"
    bundles_dir = args.out_root / "bundles"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    cache = _CellCache(args.store_root, args.layer)
    want = regime_meta(args)
    n_shard = len(shard_units)
    n_fail = 0
    for i, (spec, arm) in enumerate(shard_units):
        uk = unit_key(spec, arm)
        upath = pairs_dir / f"{uk}.json"
        if upath.exists():
            try:
                prior = json.loads(upath.read_text())
            except (json.JSONDecodeError, OSError):
                prior = None
            if prior is not None and prior.get("meta") == want and not prior.get("retryable"):
                print(f"[dvf] unit {i + 1}/{n_shard} {uk} RESUME (checkpoint)", flush=True)
                continue
        print(f"[dvf] unit {i + 1}/{n_shard} {uk}", flush=True)
        t0 = time.perf_counter()
        (sm, sc), (tm, tc) = spec
        try:
            source = cache.get(sm, sc)
            target = cache.get(tm, tc)
            unit, bundle = run_unit(source, target, spec, arm, args, lams)
        except Exception as exc:  # recorded hole, battery continues (plan s4)
            import traceback

            traceback.print_exc()
            unit, bundle = {"error": f"{type(exc).__name__}: {exc}", "retryable": True}, {}
        if "error" in unit:
            n_fail += 1
            unit.setdefault("unit_key", uk)
            unit["meta"] = want
            _atomic_write_json(upath, unit)
            print(f"[dvf] unit {i + 1}/{n_shard} {uk} FAILED: {unit['error']}", flush=True)
            continue
        if bundle:
            _atomic_savez(bundles_dir / f"{uk}.npz", **bundle)
        _atomic_write_json(upath, unit)
        print(
            f"[dvf] unit {i + 1}/{n_shard} {uk} done verdict={unit['verdict']} "
            f"g1={unit['g1']:.4f} elapsed={time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    print(f"[dvf] pairs phase done: {n_shard} units, {n_fail} failures (recorded)", flush=True)
    return 0


def cmd_nulls(args) -> int:
    """Item-5 two-sided rotation null — SHARED Haar draws over every unit.

    Per draw k (seed*1000003+k, the parent battery convention): E = P * R^T
    with P, R Haar(d); the null cosine for (A, B) is s_A^T E s_B /
    (|A|_F |B|_F) — per-draw EXACTLY the verbatim two-QR formula (von Neumann
    identity; distribution-identical by Haar invariance).
    """
    import torch

    pairs_dir = args.out_root / "pairs"
    bundles_dir = args.out_root / "bundles"
    todo: list[tuple[Path, Path, dict]] = []
    for upath in sorted(pairs_dir.glob("*.json")):
        unit = json.loads(upath.read_text())
        if "error" in unit or "operator_read" not in unit:
            continue
        nul = unit["operator_read"].get("rotation_null")
        if nul is not None and int(nul.get("n_draws", 0)) >= args.rotation_draws:
            continue
        bpath = bundles_dir / f"{unit['unit_key']}.npz"
        if not bpath.exists():
            print(f"[dvf-nulls] MISSING bundle for {unit['unit_key']} — skipped", flush=True)
            continue
        todo.append((upath, bpath, unit))
    if not todo:
        print("[dvf-nulls] nothing to do", flush=True)
        return 0
    # Group by d (dim-limit smokes may differ from production units).
    by_d: dict[int, list[int]] = {}
    loaded = []
    for idx, (upath, bpath, unit) in enumerate(todo):
        with np.load(bpath) as z:
            svec_map = {k: z[k] for k in z.files if k.startswith("svec_")}
        loaded.append((upath, bpath, unit, svec_map))
        by_d.setdefault(int(unit["d"]), []).append(idx)
    dev = torch.device(args.device)
    for d, idxs in sorted(by_d.items()):
        cmp_rows_a, cmp_rows_b, cmp_keys = [], [], []
        for idx in idxs:
            _, _, unit, svec_map = loaded[idx]
            s_free = svec_map["svec_free"]
            for name, sv in svec_map.items():
                if name == "svec_free":
                    continue
                cmp_keys.append((idx, name.removeprefix("svec_")))
                cmp_rows_a.append(sv)
                cmp_rows_b.append(s_free)
        if not cmp_keys:
            continue
        S_a = torch.from_numpy(np.stack(cmp_rows_a)).to(dev)
        S_b = torch.from_numpy(np.stack(cmp_rows_b)).to(dev)
        denom = S_a.norm(dim=1) * S_b.norm(dim=1) + 1e-12
        draws = np.zeros((args.rotation_draws, len(cmp_keys)), dtype=np.float64)
        for k in range(args.rotation_draws):
            t0 = time.perf_counter()
            gen = torch.Generator().manual_seed(args.seed * 1_000_003 + k)
            p = _haar(d, gen).to(dev)
            r = _haar(d, gen).to(dev)
            e = p * r.T
            vals = ((S_a @ e) * S_b).sum(dim=1) / denom
            draws[k] = vals.cpu().numpy()
            print(
                f"[dvf-nulls] d={d} draw {k + 1}/{args.rotation_draws} "
                f"elapsed={time.perf_counter() - t0:.2f}s",
                flush=True,
            )
        for col, (idx, cmp_name) in enumerate(cmp_keys):
            upath, bpath, unit, _ = loaded[idx]
            arr = draws[:, col]
            nul_block = unit["operator_read"].setdefault("rotation_null", {}) or {}
            nul_block.setdefault("n_draws", int(args.rotation_draws))
            nul_block.setdefault("seed", int(args.seed))
            nul_block.setdefault("convention", "parent-9a-ter svec reduction (E = P*R^T)")
            nul_block[cmp_name] = {
                "null_mean": float(arr.mean()),
                "null_std": float(arr.std()),
                "null_p025": float(np.quantile(arr, 0.025)),
                "null_p975": float(np.quantile(arr, 0.975)),
                "observed": unit["operator_read"]["raw_cosine"].get(cmp_name),
            }
            unit["operator_read"]["rotation_null"] = nul_block
        # Persist per-draw matrices into the bundles (plan s6 persistence duty).
        for idx in idxs:
            upath, bpath, unit, _ = loaded[idx]
            cols = [c for c, (i2, _n) in enumerate(cmp_keys) if i2 == idx]
            names = [n for (i2, n) in cmp_keys if i2 == idx]
            with np.load(bpath) as z:
                arrays = {k: z[k] for k in z.files}
            for c, nm in zip(cols, names):
                arrays[f"rotation_draws_{nm}"] = draws[:, c]
            _atomic_savez(bpath, **arrays)
            _atomic_write_json(upath, unit)
    print(f"[dvf-nulls] patched {len(todo)} units at {args.rotation_draws} draws", flush=True)
    return 0


def _load_parent_rungs(args) -> dict:
    """parent rung_reached index: {(model, 'src__tgt', arm): rung} from ladder JSONs."""
    out: dict = {}
    for model in MODEL_SLUGS:
        p = args.parent_ladder_dir / f"ladder_{model}_L19.json"
        if not p.exists():
            print(f"[dvf-merge] WARN no parent ladder JSON at {p}", flush=True)
            continue
        ladder = json.loads(p.read_text())
        for pair_key, arms in ladder.get("pairs", {}).items():
            for arm, res in arms.items():
                if isinstance(res, dict) and "rung_reached_point" in res:
                    out[(model, pair_key, arm)] = int(res["rung_reached_point"])
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(a) < 2:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def cmd_merge(args) -> int:
    specs = build_pair_specs(args)
    units = _units(specs)
    pairs_dir = args.out_root / "pairs"
    rungs = _load_parent_rungs(args)
    rows, failures, missing = [], [], []
    for spec, arm in units:
        uk = unit_key(spec, arm)
        upath = pairs_dir / f"{uk}.json"
        if not upath.exists():
            missing.append(uk)
            continue
        unit = json.loads(upath.read_text())
        if "error" in unit:
            failures.append({"unit_key": uk, "error": unit["error"]})
            continue
        rows.append(unit)
    verdict_counts: dict[str, dict[str, int]] = {}
    class0_counts: dict[str, int] = {}
    verdict_counts_fixed: dict[str, dict[str, int]] = {}
    conc_pool: dict[str, dict[str, list]] = {}
    for unit in rows:
        model_key = (
            unit["src_model"]
            if not unit["cross_model"]
            else f"{unit['src_model']}->{unit['tgt_model']}"
        )
        gk = f"{model_key}|{unit['arm']}"
        v, vf = unit["verdict"], unit["verdict_fixed_effrank"]
        if v == "free_map_uninformative":
            class0_counts[gk] = class0_counts.get(gk, 0) + 1
        else:
            verdict_counts.setdefault(gk, {}).setdefault(v, 0)
            verdict_counts[gk][v] += 1
        verdict_counts_fixed.setdefault(gk, {}).setdefault(vf, 0)
        verdict_counts_fixed[gk][vf] += 1
        if not unit["cross_model"]:
            rung = rungs.get((unit["src_model"], unit["pair_key"], unit["arm"]))
            if rung is not None and v != "free_map_uninformative" and np.isfinite(unit["g1"]):
                cp = conc_pool.setdefault(unit["arm"], {"rung": [], "g1": [], "g2": []})
                cp["rung"].append(rung)
                cp["g1"].append(unit["g1"])
                cp["g2"].append(unit["g2"])
    concordance = {}
    for arm, cp in conc_pool.items():
        concordance[arm] = {
            "n": len(cp["rung"]),
            "spearman_rung_g1": _spearman(np.array(cp["rung"]), np.array(cp["g1"])),
            "spearman_rung_g2": _spearman(np.array(cp["rung"]), np.array(cp["g2"])),
        }
    summary = {
        "meta": regime_meta(args),
        "n_expected_units": len(units),
        "n_complete": len(rows),
        "n_failed": len(failures),
        "n_missing": len(missing),
        "failures": failures,
        "missing_units": missing[:50],
        "verdict_counts": verdict_counts,
        "class0_free_map_uninformative_counts": class0_counts,
        "verdict_counts_fixed_effrank": verdict_counts_fixed,
        # Concordance rho (plan s3): informative within-model units per arm;
        # NaN below the n>=2 floor (a 1-unit smoke is a designed NaN).
        "concordance": concordance,
        "metadata": _metadata(),
    }
    _atomic_write_json(args.out_root / "summary.json", summary)
    print(
        f"[dvf-merge] wrote summary: {len(rows)} complete / {len(failures)} failed / "
        f"{len(missing)} missing of {len(units)} units",
        flush=True,
    )
    if missing:
        print(f"[dvf-merge] FAIL-LOUD: {len(missing)} units never attempted", flush=True)
        return 3
    return 0


def cmd_gate1(args) -> int:
    """Gate 1 (plan s7): parity vs the published parent per-pair JSON + timing pilot."""
    lams = (
        fl.LAMBDAS if args.lambda_grid == "ladder13" else fl.resolve_lambda_grid(args.lambda_grid)
    )
    report: dict = {"gate": "gate1", "atol": GATE1_ATOL, "metadata": _metadata()}
    spec = GATE1_PAIR
    (sm, sc), (tm, tc) = spec
    target_path = args.gate1_target or (
        REPO_ROOT / "eval_results/issue_1689/ladder" / f"pairs_{sm}_L19" / f"{sc}__{tc}.json"
    )
    published = json.loads(Path(target_path).read_text())["arms"]["context"]
    t0 = time.perf_counter()
    res = fl.run_pairs_generalized(
        args.store_root,
        [spec],
        layer=args.layer,
        n_bootstrap_draws=0,
        n_null_draws=args.gate1_null_draws,
        engine="torch",
        device=args.device,
        checkpoint_dir=None,
        lambda_grid=args.lambda_grid,
    )
    ladder_wall = time.perf_counter() - t0
    new = res["pairs"][fl.pair_spec_key(spec)]["context"]
    diffs = {
        k: abs(published["rung_r2s_point"][k] - new["rung_r2s_point"][k])
        for k in published["rung_r2s_point"]
    }
    max_diff = max(diffs.values())
    rung_match = int(published["rung_reached_point"]) == int(new["rung_reached_point"])
    n_match = int(published["n_common"]) == int(new["n_common"])
    parity_ok = max_diff <= GATE1_ATOL and rung_match and n_match
    report["parity"] = {
        "pair": fl.pair_spec_key(spec),
        "arm": "context",
        "target_json": str(target_path),
        "max_abs_rung_r2_diff": max_diff,
        "per_rung_abs_diff": diffs,
        "rung_reached_match": rung_match,
        "n_common_match": n_match,
        "n_common": int(new["n_common"]),
        "ladder_unit_wall_s": round(ladder_wall, 1),
        "ok": parity_ok,
    }
    if args.gate1_timing:
        cache = _CellCache(args.store_root, args.layer)
        source = cache.get(sm, sc)
        target = cache.get(tm, tc)
        t0 = time.perf_counter()
        unit, _bundle = run_unit(source, target, spec, "context", args, lams)
        report["timing"] = {
            "battery_unit_wall_s": round(time.perf_counter() - t0, 1),
            "row_limit": args.row_limit,
            "dim_limit": args.dim_limit,
            "unit_error": unit.get("error"),
        }
    _atomic_write_json(args.out_root / "gate1_report.json", report)
    if not parity_ok:
        print(
            f"[dvf-gate1] PARITY FAIL: max|diff|={max_diff:.3e} rung_match={rung_match}", flush=True
        )
        return 7  # distinct rc: designed gate refusal, not an anonymous crash (#1415)
    print(f"[dvf-gate1] PARITY PASS: max|diff|={max_diff:.3e} wall={ladder_wall:.1f}s", flush=True)
    return 0


def cmd_stage(args) -> int:
    """Stage the 42 pinned L19 stores to <store-root>/<model>/<cond>/L<layer>.pt.

    Per-file targets via hub.stage_hub_file (exact-dest; no mirror-root
    arithmetic — the #1774 trap applies to stage_hub_prefix, not here). Only
    L<layer>.pt files are staged (the prefix also holds other layers).
    """
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = hub.list_hf_files_under_path(
        api,
        DATA_REPO,
        STORE_HF_PREFIX,
        repo_type="dataset",
        revision=PINNED_STORE_REVISION,
    )
    wanted = [f for f in files if f.endswith(f"/L{args.layer}.pt")]
    if len(wanted) < 42:
        raise RuntimeError(f"expected 42 L{args.layer}.pt files at the pin, found {len(wanted)}")
    todo = []
    for repo_path in wanted:
        rel = repo_path.removeprefix(STORE_HF_PREFIX + "/")
        target = args.store_root / rel
        if not target.exists():
            todo.append((repo_path, target))
    if not todo:
        print(
            f"[dvf-stage] all {len(wanted)} stores already present under {args.store_root}",
            flush=True,
        )
        return 0
    st = os.statvfs(args.store_root if args.store_root.exists() else args.store_root.parent)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    need_gb = args.stage_headroom_gb
    if free_gb < need_gb:
        raise RuntimeError(
            f"staging headroom {free_gb:.1f} GB < required {need_gb} GB on {args.store_root}"
        )

    def _one(item):
        repo_path, target = item
        hub.stage_hub_file(
            DATA_REPO,
            repo_path,
            target,
            repo_type="dataset",
            revision=PINNED_STORE_REVISION,
        )
        print(f"[dvf-stage] staged {target}", flush=True)

    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(_one, todo))
    print(
        f"[dvf-stage] staged {len(todo)} files (of {len(wanted)}) at pin {PINNED_STORE_REVISION[:12]}",
        flush=True,
    )
    return 0


def cmd_upload(args) -> int:
    """One upload_folder commit per out-root bundles dir + exact-set verify."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    bundles_dir = args.out_root / "bundles"
    if not bundles_dir.exists():
        print(f"[dvf-upload] no bundles dir at {bundles_dir} — nothing to upload", flush=True)
        return 0
    prefix = f"{BUNDLE_HF_PREFIX}/{args.out_root.name}"
    url = hub._upload(
        bundles_dir,
        DATA_REPO,
        "dataset",
        prefix,
        raise_on_error=True,
    )
    expected = [
        f"{prefix}/{p.relative_to(bundles_dir)}" for p in sorted(bundles_dir.rglob("*.npz"))
    ]
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    missing = hub.verify_repo_paths_uploaded(
        api, DATA_REPO, expected, path_in_repo=prefix, repo_type="dataset"
    )
    if missing:
        raise RuntimeError(
            f"bundle upload verify FAILED: {len(missing)} missing (first {missing[:3]})"
        )
    print(f"[dvf-upload] {len(expected)} bundle files verified at {url or prefix}", flush=True)
    return 0


def cmd_write_pairs(args) -> int:
    specs = build_pair_specs(args)
    payload = [[[sm, sc], [tm, tc]] for ((sm, sc), (tm, tc)) in specs]
    out = Path(args.write_pairs_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(out)
    print(f"[dvf] wrote {len(payload)} pair specs to {args.write_pairs_out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--phase",
        choices=["stage", "gate1", "pairs", "nulls", "merge", "upload", "write-pairs"],
        required=True,
    )
    ap.add_argument("--store-root", type=Path, default=None)
    ap.add_argument(
        "--out-root", type=Path, default=Path("eval_results/issue_1689/derived_vs_free_B")
    )
    ap.add_argument("--layer", type=int, default=HEADLINE_LAYER)
    ap.add_argument("--lambda-grid", choices=sorted(LAMBDA_GRIDS), default="ladder13")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--pairs-file", type=Path, default=None)
    ap.add_argument("--default-model", type=str, default=None)
    ap.add_argument("--pair-set", choices=["within-model", "cross-model"], default="within-model")
    ap.add_argument("--models", type=str, default=",".join(MODEL_SLUGS))
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--rotation-draws", type=int, default=200)
    ap.add_argument("--row-limit", type=int, default=None, help="smoke: cap common rows")
    ap.add_argument("--dim-limit", type=int, default=None, help="smoke: cap hidden dims")
    ap.add_argument("--gate1-null-draws", type=int, default=40)
    ap.add_argument("--gate1-timing", action="store_true")
    ap.add_argument("--gate1-target", type=Path, default=None)
    ap.add_argument("--stage-headroom-gb", type=float, default=18.0)
    ap.add_argument(
        "--parent-ladder-dir", type=Path, default=Path("eval_results/issue_1689/ladder")
    )
    ap.add_argument("--write-pairs-out", type=Path, default=None)
    args = ap.parse_args()

    if args.phase in ("stage", "gate1", "pairs") and args.store_root is None:
        ap.error(f"--phase {args.phase} requires --store-root")
    if args.phase == "write-pairs" and args.write_pairs_out is None:
        ap.error("--phase write-pairs requires --write-pairs-out")
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda but torch.cuda.is_available() is False")

    print(
        f"[dvf] phase={args.phase} pair_set={args.pair_set} device={args.device} "
        f"shard={args.shard_index}/{args.num_shards} row_limit={args.row_limit} "
        f"dim_limit={args.dim_limit}",
        flush=True,
    )
    dispatch = {
        "stage": cmd_stage,
        "gate1": cmd_gate1,
        "pairs": cmd_pairs,
        "nulls": cmd_nulls,
        "merge": cmd_merge,
        "upload": cmd_upload,
        "write-pairs": cmd_write_pairs,
    }
    return dispatch[args.phase](args)


if __name__ == "__main__":
    rc = main()
    # C-extension shutdown-race workaround (gotchas.md PyGILState_Release):
    # flush, then bypass finalize-time teardown. All writes use explicit
    # handles + os.replace, so atexit is safely skipped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
