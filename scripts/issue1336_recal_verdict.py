#!/usr/bin/env python
"""Issue #1336 — E1 held-out recalibration + fold-exchangeability driver (plan v9).

Zero-GPU re-analysis of the committed diagnosis predictions (battery_v0 fp32
npz + turnstore truth) plus the Qwen validate-before-use leg; conditional E2
refit. One step selector, canonical order enforced regardless of CLI order
(DG-E1: qwen_recal BEFORE recal BEFORE verdict):

  stage       E1.0 staging (reuses the D1 stager: turnstores, fp16 preds,
              rollout text, Qwen stream-reduce) + battery_v0 preds npz +
              fail-loud input asserts (row count, prompt-id equality,
              manifest sha, stored-folds validity).
  qwen_recal  E1.d Qwen L19 committed-convention refit -> OOF preds ->
              IDENTICAL cross-fitted recalibration => S_qwen_recal, V-gate,
              bar_r (computed BEFORE any Llama verdict read, DG-E1) + the
              healthy-family fold-mean-norm reference + Qwen gain spectrum.
  recal       E1.a + E1.c per cell: DG-E0 stored-preds reproduction gate
              (chat, +/-1e-3), held-out cross-fitted per-dim affine recal
              per verdict layer (raw + in-sample companions), 200-draw
              within-fold pairing-permutation null (per-draw layer-max,
              selection-symmetric), 1,000-draw prompt-level bootstrap
              (per-resample layer-max, selection-inheriting), gain-spectrum
              characterization + excess decomposition + lambda-audit join.
  fold_exch   E1.b per cell: fold-mean/residual-bias norms vs 1,000 iid
              same-size repartitions (per-draw max over folds), seed-1
              committed-convention refit at the verdict layers (E2 trigger 1
              input), near-duplicate audit (digest-only, no row text).
  verdict     E1.e: lattice inputs (S_r, B_r, bar_r, S'_r, D_r + CI), V-gate,
              A_r (+0.6/0.9 sensitivity), E2-trigger evaluation, routed
              terminal decision -> recal_verdict.json. `--use-e2` re-reads
              the SAME lattice once on the v5 outputs.
  e2          Conditional refit leg (fires only on a registered trigger, or
              a forced --e2-variant in smoke): v5-fold (median over fold
              seeds {0,1,2}) or v5-cal (nested-CV lambda selection, inner
              4-fold, objective = cross-fitted recalibrated held-out R^2 on
              the widened grid; outer folds untouched).

All draw batteries are BATCHED (suff-stats formulation: the cross-fitted
per-dim affine recal reduces to per-fold sufficient statistics, so nulls are
chunked gather-einsums and bootstrap resamples are subset-sum GEMMs over a
multinomial weight matrix — no per-draw Python refit loops). The only serial
kernels are the ~55 eigh of the seed-1/Qwen/E2 refits (plan §9, seconds each,
via the #825 Gram-ridge cores verbatim).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) bind BEFORE torch/numpy import

import issue825_fit_cells as fc  # noqa: E402
import issue1336_diagnose_g1 as d1  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402

# ---------------------------------------------------------------------------
# Registered constants (plan v9 §3/§10 "Registered constants")
# ---------------------------------------------------------------------------
N_RECAL_NULL = 200  # within-fold pairing-permutation draws (Source: plan §11)
N_BOOT_RECAL = 1_000  # prompt-level bootstrap resamples (Source: mirrors N_BOOT)
N_REPART = 1_000  # iid fold-repartition reference draws (Source: mirrors N_BOOT)
FOLD_RAND_SEED = 1  # E1.b(ii) reshuffle seed (Source: pre-registered, plan §11)
E2_TRIGGER1_DELTA = 0.1  # |dR2@L29| >= 0.1 (sensitivity 0.05/0.2 reported)
E2_TRIGGER1_SENSITIVITY = (0.05, 0.2)
A_R_BAR = 0.8  # mechanism-account threshold (sensitivity 0.6/0.9 reported)
A_R_SENSITIVITY = (0.6, 0.9)
R2_V0_L29 = -0.92866  # committed chat argmax value (plan §3; overridable for smoke)
DGE0_TOL = 1e-3  # plan §7 DG-E0
DGE0_TARGETS = {"l29": -0.92866, "l30": -0.93494}  # chat stored-preds pooled convention
INNER_FOLDS_DEFAULT = 4  # E2 v5-cal nested-CV inner folds (plan allows [3, 5])
E2_FOLD_SEEDS = (0, 1, 2)  # v5-fold median-over-seeds set (plan §4 E2 trigger 1)
VAR_EPS = 1e-12  # per-dim variance guard (matches _perdim_from_preds)
BATTERY_PREDS_STEM = "battery_v0_preds_{cell}.npz"
# Deterministic per-battery RNG stream offsets off args.seed (recorded in JSON).
SEED_OFF_NULL = 11
SEED_OFF_BOOT = 12
SEED_OFF_REPART = 13
SEED_OFF_QWEN_REPART = 14
SEED_OFF_E2_NULL = 21  # + seed index for v5-fold per-seed streams

STEP_ORDER = ("stage", "qwen_recal", "recal", "fold_exch", "verdict", "e2")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", required=True, help="comma list from " + ",".join(STEP_ORDER))
    ap.add_argument("--cells", default=",".join(d1.DIAG_CELLS), help="comma cell ids (chat first)")
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=Path("data/issue_1336/diag_stage"),
        help="staging root (pod-side local disk; NEVER the shared-VM root)",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1336/diagnosis/recal"))
    ap.add_argument("--turnstore-dir", type=Path, default=None, help="override (smoke fixtures)")
    ap.add_argument("--preds-dir", type=Path, default=None, help="fp16 preds override (smoke)")
    ap.add_argument("--battery-preds-dir", type=Path, default=None, help="battery npz override")
    ap.add_argument("--gen-dir", type=Path, default=None, help="override (smoke fixtures)")
    ap.add_argument("--qwen-reduced", type=Path, default=None, help="override (smoke fixtures)")
    ap.add_argument(
        "--committed-eval-dir",
        type=Path,
        default=Path("eval_results/issue_1336"),
        help="committed diagnosis JSONs root (fs, then `git show HEAD:` fallback)",
    )
    ap.add_argument("--folds", type=int, default=cm.N_FOLDS)
    ap.add_argument("--seed", type=int, default=cm.FIT_SEED)
    ap.add_argument("--fold-rand-seed", type=int, default=FOLD_RAND_SEED)
    ap.add_argument("--recal-null-draws", type=int, default=N_RECAL_NULL)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_RECAL)
    ap.add_argument("--n-repart", type=int, default=N_REPART)
    ap.add_argument("--expect-n", type=int, default=d1.EXPECT_N_ROWS)
    ap.add_argument(
        "--dge0-targets-json",
        default=None,
        help='JSON {"l29": r2, "l30": r2} DG-E0 target override (smoke oracle)',
    )
    ap.add_argument(
        "--r2-v0-l29",
        type=float,
        default=R2_V0_L29,
        help="registered committed argmax value for A_r (smoke passes its oracle)",
    )
    ap.add_argument("--use-e2", action="store_true", help="verdict reads the v5 outputs (E2 fired)")
    ap.add_argument("--e2-variant", choices=("auto", "fold", "cal"), default="auto")
    ap.add_argument("--inner-folds", type=int, default=INNER_FOLDS_DEFAULT)
    ap.add_argument("--no-pilot-abort", action="store_true", help="report-only pilot projection")
    ap.add_argument("--wall-budget-h", type=float, default=1.0, help="plan §9 E1.a-c wall")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# E1.0 — staging + input asserts
# ---------------------------------------------------------------------------
def _battery_preds_dir(args) -> Path:
    if args.battery_preds_dir is not None:
        return args.battery_preds_dir
    return args.stage_root / "battery_preds"


def _battery_preds_path(args, cell_id: str) -> Path:
    return _battery_preds_dir(args) / BATTERY_PREDS_STEM.format(cell=cell_id)


def _load_battery_preds(args, cell_id: str) -> dict:
    path = _battery_preds_path(args, cell_id)
    assert path.exists(), f"battery_v0 preds npz missing: {path} (run --steps stage first)"
    return dict(np.load(path, allow_pickle=False))


def _stage_battery_preds(args, api, dl, hub) -> None:
    """Per-file hf_hub_download of the two battery_v0 preds npz (exact paths —
    no listing; the diagnosis prefix also holds 713 MB v2 preds we never read)."""
    dest = _battery_preds_dir(args)
    dest.mkdir(parents=True, exist_ok=True)
    for cell_id in args.cell_ids:
        target = dest / BATTERY_PREDS_STEM.format(cell=cell_id)
        if target.exists():
            print(f"[stage] {target} already staged — skipping")
            continue
        rel = f"{cm.HF_PREFIX_1336}/analysis_tensors/diagnosis/{target.name}"
        local = hub.retry_transient(
            lambda r=rel: dl(
                repo_id=cm.HF_DATA_REPO,
                repo_type="dataset",
                filename=r,
                local_dir=args.stage_root,
            ),
            what=f"recal stage: download {rel}",
        )
        Path(local).rename(target)
        print(f"[stage] staged {target}")


def _assert_folds_valid(cell_id: str, folds: np.ndarray, conv: np.ndarray, n_folds: int) -> None:
    """Stored-folds validity (plan E1.0): n_folds distinct values, conv-constant."""
    vals = sorted(set(int(v) for v in folds))
    assert vals == list(range(n_folds)), f"{cell_id}: stored folds {vals} != 0..{n_folds - 1}"
    order = np.argsort(conv, kind="stable")
    cs, fs = conv[order], folds[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    for s, e in zip(starts, np.r_[starts[1:], len(cs)], strict=True):
        assert (fs[s:e] == fs[s]).all(), f"{cell_id}: fold varies within conv {cs[s]!r}"


def _assert_recal_inputs(args) -> None:
    """E1.0 fail-loud asserts: battery npz vs fp16 preds vs manifest, folds."""
    manifest_path = d1._preds_dir(args) / "preds_manifest.json"
    assert manifest_path.exists(), f"preds_manifest.json missing: {manifest_path}"
    manifest = json.loads(manifest_path.read_text())
    for cell_id in args.cell_ids:
        bat = _load_battery_preds(args, cell_id)
        bconv = np.asarray(bat["conv_ids"]).astype(str)
        assert len(bconv) == args.expect_n, (
            f"{cell_id}: battery rows {len(bconv)} != expected {args.expect_n}"
        )
        fp16 = d1._load_preds_npz(args, cell_id)
        fconv = np.asarray(fp16["conv_ids"]).astype(str)
        assert (bconv == fconv).all(), f"{cell_id}: battery vs fp16 preds prompt-id mismatch"
        fitted = np.asarray(bat["fitted_mask"]).astype(bool)
        assert fitted.all(), (
            f"{cell_id}: battery fitted_mask not all-true ({int(fitted.sum())}/{len(fitted)}) — "
            "the committed pooled values were computed on full coverage"
        )
        _assert_folds_valid(cell_id, np.asarray(bat["folds"]), bconv, args.folds)
        fname = d1.PREDS_STEM.get(cell_id, f"preds_{cell_id}.npz")
        entry = manifest.get(fname)
        assert entry is not None, f"{fname} missing from preds_manifest.json"
        sha = hashlib.sha256((d1._preds_dir(args) / fname).read_bytes()).hexdigest()
        assert sha == entry["sha256"], f"{fname}: sha256 mismatch vs preds_manifest.json"
        assert entry["shapes"]["conv_ids"][0] == args.expect_n, (
            f"{fname}: manifest row count {entry['shapes']['conv_ids'][0]} != {args.expect_n}"
        )
        print(f"[stage] {cell_id}: E1.0 asserts OK (n={args.expect_n}, manifest sha verified)")


def step_stage(args) -> None:
    print("[recal1336] step=stage", flush=True)
    api, dl, hub = d1._hub_helpers()
    d1.step_stage(args)  # turnstores + fp16 preds + rollout text + Qwen stream-reduce
    _stage_battery_preds(args, api, dl, hub)
    _assert_recal_inputs(args)


# ---------------------------------------------------------------------------
# Cross-fitted per-dim affine recalibration — math cores
# ---------------------------------------------------------------------------
def _fold_rows(folds: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
    ids = sorted(set(int(v) for v in folds))
    return ids, [np.flatnonzero(folds == k) for k in ids]


def _suff_stats_observed(P: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> dict:
    """Per-fold sufficient statistics (K, d) + counts (K,) for the recal math."""
    ids, rows = _fold_rows(folds)
    K, d = len(ids), P.shape[1]
    out = {k: np.empty((K, d)) for k in ("s_p", "s_y", "s_pp", "s_yy", "s_py")}
    n = np.empty(K)
    for ki, r in enumerate(rows):
        Pk = P[r].astype(np.float64)
        Yk = Y[r].astype(np.float64)
        out["s_p"][ki] = Pk.sum(0)
        out["s_y"][ki] = Yk.sum(0)
        out["s_pp"][ki] = (Pk * Pk).sum(0)
        out["s_yy"][ki] = (Yk * Yk).sum(0)
        out["s_py"][ki] = (Pk * Yk).sum(0)
        n[ki] = len(r)
    return {**out, "n": n}


def _recal_r2_from_stats(s_p, s_y, s_pp, s_yy, s_py, n) -> np.ndarray:
    """Pooled held-out cross-fitted per-dim affine-recal R^2 from per-fold stats.

    Broadcast-batched: stats are (..., K, d), counts (..., K); returns (...,).
    Train moments for fold k = totals - fold-k (leave-fold-out); eval ss_res
    expands in the fold's own sums, ss_tot uses the fold-local test mean (the
    committed pooled convention). Empty folds contribute zero.
    """
    s_p, s_y, s_pp, s_yy, s_py = (
        np.asarray(a, dtype=np.float64) for a in (s_p, s_y, s_pp, s_yy, s_py)
    )
    n = np.asarray(n, dtype=np.float64)
    t_p, t_y, t_pp, _t_yy, t_py = (
        a.sum(axis=-2, keepdims=True) for a in (s_p, s_y, s_pp, s_yy, s_py)
    )
    t_n = n.sum(axis=-1, keepdims=True)
    tr_n = (t_n - n)[..., None]  # (..., K, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mp = (t_p - s_p) / tr_n
        my = (t_y - s_y) / tr_n
        var_p = (t_pp - s_pp) / tr_n - mp * mp
        cov = (t_py - s_py) / tr_n - mp * my
        a = np.where(var_p > VAR_EPS, cov / np.maximum(var_p, VAR_EPS), 0.0)
        b = my - a * mp
        ss_res = (s_yy - 2.0 * a * s_py + a * a * s_pp - 2.0 * b * s_y + 2.0 * a * b * s_p).sum(
            axis=-1
        ) + n * (b * b).sum(axis=-1)
        ss_tot = (s_yy - (s_y * s_y) / np.maximum(n[..., None], 1.0)).sum(axis=-1)
    ok = (n > 0) & (tr_n[..., 0] >= 2)
    ss_res = np.where(ok, ss_res, 0.0)
    ss_tot = np.where(ok, ss_tot, 0.0)
    tot = ss_tot.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(1.0 - ss_res.sum(axis=-1) / np.where(tot > 0, tot, np.nan))


def _crossfit_recal_direct(P: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> dict:
    """Reference cross-fitted per-dim affine recal (fold loop, vectorized dims).

    No row's recalibration is fit on itself: fold k's (a_j, b_j) come from the
    OTHER folds' (pred, truth) pairs only. Returns r2 (pooled, fold-local test
    mean), per-fold (a, b) (K, d), per-fold ss, and the recalibrated preds.
    """
    ids, rows = _fold_rows(folds)
    K, d = len(ids), P.shape[1]
    a_all = np.zeros((K, d))
    b_all = np.zeros((K, d))
    ss_res = np.zeros(K)
    ss_tot = np.zeros(K)
    pred_recal = np.zeros_like(P, dtype=np.float64)
    for ki, r in enumerate(rows):
        tr = np.setdiff1d(np.arange(len(folds)), r, assume_unique=True)
        Ptr = P[tr].astype(np.float64)
        Ytr = Y[tr].astype(np.float64)
        mp, my = Ptr.mean(0), Ytr.mean(0)
        var_p = ((Ptr - mp) ** 2).mean(0)
        cov = ((Ptr - mp) * (Ytr - my)).mean(0)
        a = np.where(var_p > VAR_EPS, cov / np.maximum(var_p, VAR_EPS), 0.0)
        b = my - a * mp
        a_all[ki], b_all[ki] = a, b
        pr = a * P[r].astype(np.float64) + b
        true = Y[r].astype(np.float64)
        mu = true.mean(0)
        ss_res[ki] = float(((true - pr) ** 2).sum())
        ss_tot[ki] = float(((true - mu) ** 2).sum())
        pred_recal[r] = pr
    r2 = float(1.0 - ss_res.sum() / ss_tot.sum()) if ss_tot.sum() > 0 else float("nan")
    return {
        "r2": r2,
        "a": a_all,
        "b": b_all,
        "ss_res": ss_res,
        "ss_tot": ss_tot,
        "pred_recal": pred_recal,
        "fold_ids": ids,
    }


def _crossfit_offset_only_ss(P: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> float:
    """SS_res of the cross-fitted OFFSET-ONLY correction (a=1, b free per dim)."""
    _, rows = _fold_rows(folds)
    ss = 0.0
    for r in rows:
        tr = np.setdiff1d(np.arange(len(folds)), r, assume_unique=True)
        b = Y[tr].astype(np.float64).mean(0) - P[tr].astype(np.float64).mean(0)
        pr = P[r].astype(np.float64) + b
        ss += float(((Y[r].astype(np.float64) - pr) ** 2).sum())
    return ss


def _crossfit_scalar_recal_r2(P: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> float:
    """Cross-fitted GLOBAL-SCALAR affine recal (one a, b across all dims)."""
    _, rows = _fold_rows(folds)
    ss_res = ss_tot = 0.0
    for r in rows:
        tr = np.setdiff1d(np.arange(len(folds)), r, assume_unique=True)
        Ptr = P[tr].astype(np.float64)
        Ytr = Y[tr].astype(np.float64)
        mp, my = float(Ptr.mean()), float(Ytr.mean())
        var_p = float(((Ptr - mp) ** 2).mean())
        cov = float(((Ptr - mp) * (Ytr - my)).mean())
        a = cov / var_p if var_p > VAR_EPS else 0.0
        b = my - a * mp
        pr = a * P[r].astype(np.float64) + b
        true = Y[r].astype(np.float64)
        mu = true.mean(0)
        ss_res += float(((true - pr) ** 2).sum())
        ss_tot += float(((true - mu) ** 2).sum())
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _insample_recal_r2(P: np.ndarray, Y: np.ndarray) -> float:
    """The committed in-sample recal (pooled-GLOBAL convention — exact replica
    of _perdim_from_preds's affine_recalibrated_r2_pooled_global)."""
    Pm = P.astype(np.float64)
    T = Y.astype(np.float64)
    pm, tm = Pm.mean(0), T.mean(0)
    var_p = ((Pm - pm) ** 2).mean(0)
    cov = ((Pm - pm) * (T - tm)).mean(0)
    a = np.where(var_p > VAR_EPS, cov / np.maximum(var_p, VAR_EPS), 0.0)
    resid = T - (a * Pm + (tm - a * pm))
    return float(1.0 - (resid**2).sum() / ((T - tm) ** 2).sum())


def _raw_pooled_r2(P: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> float:
    """Raw pooled R^2, fold-local test mean (the committed pooled convention —
    the DG-E0 consumer-side recompute of the stored predictions)."""
    _, rows = _fold_rows(folds)
    ss_res = ss_tot = 0.0
    for r in rows:
        true = Y[r].astype(np.float64)
        pred = P[r].astype(np.float64)
        mu = true.mean(0)
        ss_res += float(((true - pred) ** 2).sum())
        ss_tot += float(((true - mu) ** 2).sum())
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _within_fold_perms(folds: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    """(T, n) row-index permutations, each permuting rows WITHIN every fold."""
    rng = np.random.default_rng(seed)
    n = len(folds)
    _, rows = _fold_rows(folds)
    perms = np.tile(np.arange(n), (n_draws, 1))
    for t in range(n_draws):
        for r in rows:
            perms[t, r] = r[rng.permutation(len(r))]
    return perms


def _null_battery_matrix(
    P_layers: dict[int, np.ndarray],
    Y_layers: dict[int, np.ndarray],
    folds: np.ndarray,
    n_draws: int,
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    """Per-draw x per-layer cross-fitted recal R^2 under within-fold pairing
    permutation of the truth rows (stored predictions FIXED; the identical
    cross-fitted recal re-run per draw). One shared permutation per draw
    across layers (per-draw layer-max stays selection-symmetric).

    Batched: only s_py changes per draw — chunked gather + einsum; every other
    sufficient statistic is invariant and computed once.
    """
    layers = sorted(P_layers)
    perms = _within_fold_perms(folds, n_draws, seed)
    ids, rows = _fold_rows(folds)
    out = np.empty((n_draws, len(layers)))
    for lix, li in enumerate(layers):
        P = P_layers[li].astype(np.float64)
        Y = Y_layers[li].astype(np.float64)
        base = _suff_stats_observed(P, Y, folds)
        d = P.shape[1]
        spy = np.empty((n_draws, len(ids), d))
        max_nk = max(len(r) for r in rows)
        chunk = max(1, int(2.0e8 / (max_nk * d * 8)))
        for c0 in range(0, n_draws, chunk):
            sl = perms[c0 : c0 + chunk]
            for ki, r in enumerate(rows):
                Yp = Y[sl[:, r]]  # (C, n_k, d) gather of permuted truth rows
                spy[c0 : c0 + len(sl), ki] = np.einsum("cnd,nd->cd", Yp, P[r])
        out[:, lix] = _recal_r2_from_stats(
            base["s_p"], base["s_y"], base["s_pp"], base["s_yy"], spy, base["n"]
        )
    return out, layers


def _bootstrap_weights(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(T, n) multinomial row-resampling weights (prompt-level, with replacement)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)


def _bootstrap_matrix(
    P_layers: dict[int, np.ndarray],
    Y_layers: dict[int, np.ndarray],
    folds: np.ndarray,
    weights: np.ndarray,
    chunk: int = 250,
) -> tuple[np.ndarray, list[int]]:
    """Per-resample x per-layer cross-fitted recal R^2 under prompt-level row
    resampling (recal re-cross-fit per resample; rows keep their stored fold).

    Batched: every per-fold sufficient statistic is a subset-sum GEMM
    (weights chunk @ per-fold pool matrix) — the #778/#834 batched-draw shape.
    """
    layers = sorted(P_layers)
    ids, rows = _fold_rows(folds)
    n_boot = weights.shape[0]
    out = np.empty((n_boot, len(layers)))
    for lix, li in enumerate(layers):
        P = P_layers[li].astype(np.float64)
        Y = Y_layers[li].astype(np.float64)
        pools = {"s_p": P, "s_y": Y, "s_pp": P * P, "s_yy": Y * Y, "s_py": P * Y}
        d = P.shape[1]
        for c0 in range(0, n_boot, chunk):
            Wc = weights[c0 : c0 + chunk]
            C = Wc.shape[0]
            stats = {k: np.empty((C, len(ids), d)) for k in pools}
            n_ck = np.empty((C, len(ids)))
            for ki, r in enumerate(rows):
                Wk = Wc[:, r]
                n_ck[:, ki] = Wk.sum(1)
                for k, pool in pools.items():
                    stats[k][:, ki] = Wk @ pool[r]
            out[c0 : c0 + C, lix] = _recal_r2_from_stats(
                stats["s_p"], stats["s_y"], stats["s_pp"], stats["s_yy"], stats["s_py"], n_ck
            )
    return out, layers


# ---------------------------------------------------------------------------
# E1.b — fold-exchangeability cores
# ---------------------------------------------------------------------------
def _fold_shift_norms_observed(M: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """(K,) ||mean(M[fold k]) - mean(M)||_2 under the STORED fold assignment."""
    mu = M.astype(np.float64).mean(0)
    _, rows = _fold_rows(folds)
    return np.asarray([np.linalg.norm(M[r].astype(np.float64).mean(0) - mu) for r in rows])


def _repartition_norms(M: np.ndarray, n_folds: int, n_draws: int, seed: int) -> np.ndarray:
    """(T, K) fold-mean-shift norms under iid same-size repartitions of the rows.

    Batched: chunked permutation gather + block means over the centered rows
    (block mean of centered rows == fold mean - global mean).
    """
    n, d = M.shape
    Md = M.astype(np.float64) - M.astype(np.float64).mean(0)
    sizes = [len(b) for b in np.array_split(np.arange(n), n_folds)]
    bounds = np.cumsum(sizes)[:-1]
    rng = np.random.default_rng(seed)
    out = np.empty((n_draws, n_folds))
    chunk = max(1, int(2.0e8 / (n * d * 8)))
    for c0 in range(0, n_draws, chunk):
        C = min(chunk, n_draws - c0)
        perms = np.stack([rng.permutation(n) for _ in range(C)])
        G = Md[perms]  # (C, n, d)
        for ki, B in enumerate(np.split(G, bounds, axis=1)):
            out[c0 : c0 + C, ki] = np.linalg.norm(B.mean(1), axis=-1)
    return out


def _fold_norm_read(M: np.ndarray, folds: np.ndarray, n_folds: int, n_draws: int, seed: int):
    """Observed max-over-folds shift norm vs the iid-repartition p97.5 band."""
    obs = _fold_shift_norms_observed(M, folds)
    ref = _repartition_norms(M, n_folds, n_draws, seed)
    ref_max = ref.max(axis=1)  # per-draw max over folds (selection-symmetric)
    p975 = float(np.quantile(ref_max, 0.975))
    return {
        "observed_per_fold": [float(v) for v in obs],
        "observed_max": float(obs.max()),
        "ref_p975_max": p975,
        "exceeds": bool(obs.max() > p975),
    }, ref


# ---------------------------------------------------------------------------
# E1.b(iii) — near-duplicate audit (digest-only; NEVER prints row text)
# ---------------------------------------------------------------------------
def _load_prompt_hashes(args, conv_ids: np.ndarray) -> tuple[dict[str, str], dict[str, str], float]:
    """conv_id -> (exact sha, normalized sha) of the KEPT prompts; join rate.

    Text never leaves this function (LMSYS real-user rows: digest-only per the
    content-hygiene rule). Reads answers.jsonl by text-mode file iteration
    (NEVER splitlines — raw U+2028/NEL in real user text shreds records, #825).
    """
    path = d1._gen_dir(args) / "rlvr" / "lmsys5k" / "answers.jsonl"
    assert path.exists(), f"rollout text missing: {path} (run --steps stage first)"
    by_key: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("kept", True):
                continue
            by_key[str(row["prompt_idx"])] = str(row.get("prompt", ""))
    exact: dict[str, str] = {}
    norm: dict[str, str] = {}
    joined = 0
    for cid in conv_ids:
        key = d1._rollout_key_for_conv(str(cid))
        text = by_key.get(key)
        if text is None:
            continue
        joined += 1
        exact[str(cid)] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        norm[str(cid)] = hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest()
    join_rate = joined / max(len(conv_ids), 1)
    assert join_rate >= d1.SPOTCHECK_MIN_JOIN_RATE, (
        f"dup audit join rate {join_rate:.3f} < {d1.SPOTCHECK_MIN_JOIN_RATE} — "
        "sidecar<->rollout conv_id join broken (see the 04603a849d join fix)"
    )
    return exact, norm, join_rate


def _dup_tier_stats(hashes: dict[str, str], fold_of: dict[str, int]) -> dict:
    groups: dict[str, list[str]] = {}
    for cid, h in hashes.items():
        groups.setdefault(h, []).append(cid)
    dup_groups = {h: cids for h, cids in groups.items() if len(cids) > 1}
    total_pairs = sum(len(c) * (len(c) - 1) // 2 for c in dup_groups.values())
    cross = 0
    for cids in dup_groups.values():
        fs = [fold_of[c] for c in cids]
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                if fs[i] != fs[j]:
                    cross += 1
    return {
        "n_rows": len(hashes),
        "n_unique": len(groups),
        "n_dup_groups": len(dup_groups),
        "n_rows_in_dup_groups": sum(len(c) for c in dup_groups.values()),
        "total_dup_pairs": total_pairs,
        "cross_fold_dup_pairs": cross,
        "max_group_size": max((len(c) for c in dup_groups.values()), default=1),
    }


# ---------------------------------------------------------------------------
# Shared per-cell loading
# ---------------------------------------------------------------------------
def _clamp_layer(li: int, layers: list[int]) -> int:
    """Committed layer index, clamped to the stored layer set (smoke fixtures
    carry fewer layers; production stores all five verdict layers)."""
    return li if li in layers else max(layers)


def _load_cell_recal_inputs(args, cell_id: str) -> dict:
    """Battery preds + turnstore truth at the stored verdict layers, aligned."""
    bat = _load_battery_preds(args, cell_id)
    bconv = np.asarray(bat["conv_ids"]).astype(str)
    fitted = np.asarray(bat["fitted_mask"]).astype(bool)
    assert fitted.all(), f"{cell_id}: battery fitted_mask not all-true"
    folds = np.asarray(bat["folds"]).astype(np.int64)
    _assert_folds_valid(cell_id, folds, bconv, args.folds)
    xy = d1._load_cell_xy(args, cell_id)
    assert (xy["conv_ids"] == bconv).all(), f"{cell_id}: turnstore vs battery prompt-id mismatch"
    layers = sorted(int(k[len("preds_l") :]) for k in bat if k.startswith("preds_l"))
    P_layers = {li: np.asarray(bat[f"preds_l{li}"], dtype=np.float64) for li in layers}
    Y_layers = {li: xy["Y"][:, li, :].astype(np.float64) for li in layers}
    return {
        "conv_ids": bconv,
        "folds": folds,
        "layers": layers,
        "P_layers": P_layers,
        "Y_layers": Y_layers,
        "X": xy["X"],
        "Y_full": xy["Y"],
    }


def _regime_fingerprint(args, cell_id: str, step: str) -> dict:
    """Resume key: EVERY output-affecting regime knob (#722 r3 rule)."""
    return {
        "step": step,
        "cell_id": cell_id,
        "folds": int(args.folds),
        "seed": int(args.seed),
        "fold_rand_seed": int(args.fold_rand_seed),
        "recal_null_draws": int(args.recal_null_draws),
        "n_boot": int(args.n_boot),
        "n_repart": int(args.n_repart),
        "expect_n": int(args.expect_n),
        "r2_v0_l29": float(args.r2_v0_l29),
        "dge0_targets": dict(args.dge0_targets),
        "e2_variant": args.e2_variant,
        "inner_folds": int(args.inner_folds),
    }


def _ckpt_ok(args, name: str, fp: dict, outputs: list[Path]) -> bool:
    ck = args.out_dir / "checkpoints" / f"{name}.json"
    if (
        ck.exists()
        and json.loads(ck.read_text())["fingerprint"] == fp
        and all(p.exists() for p in outputs)
    ):
        print(f"[recal1336] {name} checkpoint present — skipping")
        return True
    return False


def _ckpt_write(args, name: str, fp: dict) -> None:
    ck = args.out_dir / "checkpoints" / f"{name}.json"
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_text(json.dumps({"fingerprint": fp}))


# ---------------------------------------------------------------------------
# E1.d — Qwen held-out-recalibration calibration (validate-before-use + bar_r)
# ---------------------------------------------------------------------------
def _gain_spectrum_summary(a: np.ndarray, var_j: np.ndarray) -> dict:
    """Cross-fitted gain-spectrum characterization (E1.c reads, per layer)."""
    a_mean = a.mean(0)  # across-fold mean per dim
    qs = {q: float(np.quantile(a_mean, float(q))) for q in ("0.05", "0.25", "0.5", "0.75", "0.95")}
    edges = np.quantile(var_j, np.linspace(0, 1, 11))
    bins = np.clip(np.digitize(var_j, edges[1:-1]), 0, 9)
    binned = [float(np.median(a_mean[bins == b])) if (bins == b).any() else None for b in range(10)]
    corr = np.corrcoef(a) if a.shape[0] > 1 else np.ones((1, 1))
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    return {
        "a_mean": float(a_mean.mean()),
        "a_median": float(np.median(a_mean)),
        "a_quantiles": qs,
        "a_frac_in_0p9_1p1": float(((a_mean >= 0.9) & (a_mean <= 1.1)).mean()),
        "a_frac_below_0p5": float((a_mean < 0.5).mean()),
        "spearman_a_vs_var": fc._spearman(a_mean, var_j),
        "binned_median_a_by_var_decile": binned,
        "per_fold_mean_a": [float(v) for v in a.mean(1)],
        "a_cross_fold_min_corr": float(off.min()) if off.size else float("nan"),
    }


def step_qwen_recal(args) -> None:
    print("[recal1336] step=qwen_recal", flush=True)
    fp = _regime_fingerprint(args, "qwen", "qwen_recal")
    out_json = args.out_dir / "qwen_recal_cal.json"
    if _ckpt_ok(args, "qwen_recal", fp, [out_json]):
        return
    path = d1._qwen_reduced_path(args)
    assert path.exists(), f"qwen reduced tensors missing: {path} (run --steps stage first)"
    qq = torch.load(path, map_location="cpu", weights_only=False)
    li = min(int(cm.G0["layer"]), qq["X"].shape[1] - 1)
    # bf16 -> fp32 waypoint mirrors the committed G0/D1.6 load path.
    X = qq["X"][:, li, :].float().numpy()
    Y = qq["Y"][:, li, :].float().numpy()
    conv_ids = np.asarray([str(c) for c in qq["conv_ids"]])
    folds = fc._cv_folds(conv_ids, args.folds, args.seed)
    preds = np.zeros_like(Y, dtype=np.float64)
    fitted = np.zeros(len(conv_ids), dtype=bool)
    for k in sorted(set(int(v) for v in folds)):
        te = folds == k
        tr = ~te
        if te.sum() == 0 or tr.sum() < 3:
            continue
        cache = fc._prep_fold(X[tr], X[te])
        preds[te] = fc._ridge_predict_cached(cache, Y[tr])  # committed grid
        fitted[te] = True
    assert fitted.all(), "qwen refit left unfitted rows (fold too small?)"
    r2_raw = _raw_pooled_r2(preds, Y, folds)
    direct = _crossfit_recal_direct(preds, Y, folds)
    s_qwen_recal = direct["r2"]
    deviation = abs(s_qwen_recal - d1.QWEN_COMMITTED_R2)
    v_pass = deviation <= d1.QWEN_CAL_DEV_MAX
    bar_r = d1.BAR_RAW * (s_qwen_recal / d1.QWEN_COMMITTED_R2)
    y_read, y_ref = _fold_norm_read(
        Y, folds, args.folds, args.n_repart, args.seed + SEED_OFF_QWEN_REPART
    )
    resid = preds - Y.astype(np.float64)
    r_read, r_ref = _fold_norm_read(
        resid, folds, args.folds, args.n_repart, args.seed + SEED_OFF_QWEN_REPART + 1
    )
    tensors = args.out_dir / "tensors"
    tensors.mkdir(parents=True, exist_ok=True)
    np.savez(
        tensors / "qwen_recal_draws.npz",
        y_fold_norms_ref=y_ref,
        resid_fold_norms_ref=r_ref,
        oof_preds=preds.astype(np.float32),
        a=direct["a"],
        b=direct["b"],
    )
    payload = {
        "metadata": d1._metadata(args.seed, len(conv_ids)),
        "computed_ts_unix": time.time(),
        "layer": li,
        "n": len(conv_ids),
        "r2_raw_committed_grid": float(r2_raw),
        "committed_anchor": d1.QWEN_COMMITTED_R2,
        "s_qwen_recal": float(s_qwen_recal),
        "insample_recal_r2": _insample_recal_r2(preds, Y),
        "v_gate": {
            "deviation": float(deviation),
            "threshold": d1.QWEN_CAL_DEV_MAX,
            "pass": bool(v_pass),
        },
        "bar_raw": d1.BAR_RAW,
        "bar_r": float(bar_r),
        "gain_spectrum": _gain_spectrum_summary(direct["a"], Y.astype(np.float64).var(0)),
        "fold_mean_norms": {"y": y_read, "resid": r_read},
        "n_repart": int(args.n_repart),
        "seed_streams": {
            "repart_y": args.seed + SEED_OFF_QWEN_REPART,
            "repart_resid": args.seed + SEED_OFF_QWEN_REPART + 1,
        },
    }
    d1._write_json(out_json, payload)
    _ckpt_write(args, "qwen_recal", fp)
    print(
        f"[qwen_recal] raw R2={r2_raw:.4f} (anchor {d1.QWEN_COMMITTED_R2}) "
        f"S_qwen_recal={s_qwen_recal:.4f} bar_r={bar_r:.4f} V-pass={v_pass}"
    )


# ---------------------------------------------------------------------------
# E1.a + E1.c — held-out recalibration per cell (+ DG-E0 gate on chat)
# ---------------------------------------------------------------------------
def _dge0_gate(args, cell_id: str, raw_by_layer: dict[int, float], layers: list[int]) -> dict:
    """DG-E0 (plan §7): stored-preds raw pooled R^2 must reproduce the
    committed values at L29/L30 (chat) within +/-1e-3. FAIL => exit 3."""
    gate = {}
    for label, committed_li in (("l29", d1.L_COMMITTED_ARGMAX), ("l30", 30)):
        li = _clamp_layer(committed_li, layers)
        target = args.dge0_targets.get(label)
        if target is None:
            continue
        got = raw_by_layer[li]
        dev = abs(got - float(target))
        entry = {
            "layer": li,
            "target": float(target),
            "recomputed": float(got),
            "abs_dev": float(dev),
            "tol": DGE0_TOL,
            "pass": bool(dev <= DGE0_TOL),
        }
        gate[label] = entry
        if not entry["pass"]:
            print(
                f"[recal] DG-E0 FAIL {cell_id} {label}: recomputed {got:.6f} vs "
                f"target {target} (tol {DGE0_TOL}) — staging/alignment drift; "
                "nothing downstream is trustworthy",
                file=sys.stderr,
            )
            raise SystemExit(3)
        print(f"[recal] DG-E0 {cell_id} {label}: {got:.6f} vs {target} -> PASS")
    return gate


def _fp16_crosscheck(args, cell_id: str, Y_layers: dict[int, np.ndarray], layers: list[int]):
    """L30 cross-check vs the committed fp16 preds (fp32 battery = authority)."""
    fp16 = d1._load_preds_npz(args, cell_id)
    f_layers = sorted(int(k[len("preds_l") :]) for k in fp16 if k.startswith("preds_l"))
    li = _clamp_layer(30, [x for x in f_layers if x in layers] or f_layers)
    if li not in Y_layers or f"preds_l{li}" not in fp16:
        return None
    m = np.asarray(fp16["fitted_mask"]).astype(bool)
    assert m.all(), f"{cell_id}: fp16 preds fitted_mask not all-true"
    folds16 = np.asarray(fp16["folds"]).astype(np.int64)
    P16 = np.asarray(fp16[f"preds_l{li}"], dtype=np.float64)
    Y_l = Y_layers[li]
    return {
        "layer": li,
        "raw_r2_fp16": _raw_pooled_r2(P16, Y_l, folds16),
        "heldout_recal_r2_fp16": _crossfit_recal_direct(P16, Y_l, folds16)["r2"],
    }


def step_recal(args) -> None:
    print("[recal1336] step=recal", flush=True)
    pilot_state = {"done": False}
    for cell_id in args.cell_ids:
        fp = _regime_fingerprint(args, cell_id, "recal")
        out_json = args.out_dir / f"heldout_recal_{cell_id}.json"
        if _ckpt_ok(args, f"recal_{cell_id}", fp, [out_json]):
            continue
        started_ts = time.time()
        inp = _load_cell_recal_inputs(args, cell_id)
        folds, layers = inp["folds"], inp["layers"]
        P_layers, Y_layers = inp["P_layers"], inp["Y_layers"]
        is_chat = "naturalistic" not in cell_id
        raw = {li: _raw_pooled_r2(P_layers[li], Y_layers[li], folds) for li in layers}
        dge0 = _dge0_gate(args, cell_id, raw, layers) if is_chat else None

        lam_json = d1._read_committed_json(args, f"diagnosis/refit_v0_{cell_id}.json")
        assert lam_json is not None, (
            f"committed diagnosis/refit_v0_{cell_id}.json unavailable — the E1.c "
            "lambda-audit join needs the committed battery outputs on this checkout"
        )
        gcv = lam_json["gcv_lambda_layer_x_fold"]
        low_edge = float(fc.LAMBDAS[0])

        per_layer: dict[str, dict] = {}
        gain_spec: dict[str, dict] = {}
        a_by_layer: dict[int, np.ndarray] = {}
        b_by_layer: dict[int, np.ndarray] = {}
        pred_recal: dict[int, np.ndarray] = {}
        t_unit0 = time.time()
        for li in layers:
            P, Y_l = P_layers[li], Y_layers[li]
            direct = _crossfit_recal_direct(P, Y_l, folds)
            a_by_layer[li], b_by_layer[li] = direct["a"], direct["b"]
            pred_recal[li] = direct["pred_recal"].astype(np.float32)
            insample = _insample_recal_r2(P, Y_l)
            per_layer[str(li)] = {
                "raw_r2": raw[li],
                "insample_recal_r2": insample,
                "heldout_recal_r2": direct["r2"],
                "optimism_gap": insample - direct["r2"],
                "offset_only_heldout_r2": float(
                    1.0 - _crossfit_offset_only_ss(P, Y_l, folds) / direct["ss_tot"].sum()
                ),
                "global_scalar_heldout_r2": _crossfit_scalar_recal_r2(P, Y_l, folds),
            }
            var_j = Y_l.var(0)
            gs = _gain_spectrum_summary(direct["a"], var_j)
            assert li < len(gcv), f"lambda audit: layer {li} missing from committed gcv rows"
            lam_row = [float(v) for v in gcv[li]]
            gs["lambda_join"] = {
                "gcv_lambda_per_fold": lam_row,
                "n_at_low_edge": int(sum(1 for v in lam_row if v == low_edge)),
                "per_fold_mean_a": gs["per_fold_mean_a"],
            }
            gain_spec[str(li)] = gs
        unit_s = (time.time() - t_unit0) / max(len(layers), 1)
        if not pilot_state["done"]:
            # §9 pilot: one per-layer recal unit timed end-to-end; the null +
            # bootstrap batteries below are the batched remainder (projection
            # counts direct units x layers x cells; abort > 2x planned wall).
            d1._pilot_gate(args, unit_s, len(layers) * len(args.cell_ids) * 3)
            pilot_state["done"] = True

        t0 = time.time()
        null_mat, null_layers = _null_battery_matrix(
            P_layers, Y_layers, folds, args.recal_null_draws, args.seed + SEED_OFF_NULL
        )
        null_layer_max = np.nanmax(null_mat, axis=1)
        band = float(np.quantile(null_layer_max, 0.975))
        print(
            f"[recal] {cell_id} null battery ({args.recal_null_draws} draws) "
            f"{time.time() - t0:.1f}s band_p975={band:.4f}"
        )
        t0 = time.time()
        weights = _bootstrap_weights(len(folds), args.n_boot, args.seed + SEED_OFF_BOOT)
        boot_mat, _ = _bootstrap_matrix(P_layers, Y_layers, folds, weights)
        boot_s_r = np.nanmax(boot_mat, axis=1)  # per-resample layer-max (selection-inheriting)
        print(f"[recal] {cell_id} bootstrap ({args.n_boot} resamples) {time.time() - t0:.1f}s")

        heldout = {li: per_layer[str(li)]["heldout_recal_r2"] for li in layers}
        s_r = max(heldout.values())
        s_r_argmax = max(heldout, key=lambda k: heldout[k])

        l30 = _clamp_layer(30, layers)
        l29 = _clamp_layer(d1.L_COMMITTED_ARGMAX, layers)
        decomp = {}
        for li in sorted({l30, l29}):
            P, Y_l = P_layers[li], Y_layers[li]
            direct_ss_res = None
            _, rows = _fold_rows(folds)
            ss_res_raw = ss_tot = 0.0
            fold_bias_ss = 0.0
            fb_l2 = []
            for r in rows:
                true = Y_l[r].astype(np.float64)
                pred = P[r].astype(np.float64)
                mu = true.mean(0)
                ss_res_raw += float(((true - pred) ** 2).sum())
                ss_tot += float(((true - mu) ** 2).sum())
                fb = (pred - true).mean(0)
                fb_l2.append(float(np.linalg.norm(fb)))
                fold_bias_ss += float(len(r) * (fb**2).sum())
            ss_off = _crossfit_offset_only_ss(P, Y_l, folds)
            direct_ss_res = float(_crossfit_recal_direct(P, Y_l, folds)["ss_res"].sum())
            decomp[str(li)] = {
                "ss_tot": ss_tot,
                "ss_res_raw": ss_res_raw,
                "total_excess": ss_res_raw - ss_tot,
                "ss_res_offset_only": ss_off,
                "ss_res_recal": direct_ss_res,
                "offset_recovered": ss_res_raw - ss_off,
                "gain_recovered": ss_off - direct_ss_res,
                "residual_ss": direct_ss_res,
                "fold_bias_ss": fold_bias_ss,
                "fold_bias_l2_per_fold": fb_l2,
            }

        tensors = args.out_dir / "tensors"
        tensors.mkdir(parents=True, exist_ok=True)
        np.savez(
            tensors / f"recal_draws_{cell_id}.npz",
            layers=np.asarray(null_layers),
            null_r2_matrix=null_mat,
            null_layer_max=null_layer_max,
            boot_r2_matrix=boot_mat,
            boot_s_r=boot_s_r,
        )
        np.savez(
            tensors / f"recal_ab_{cell_id}.npz",
            layers=np.asarray(layers),
            **{f"a_l{li}": a_by_layer[li] for li in layers},
            **{f"b_l{li}": b_by_layer[li] for li in layers},
        )
        np.savez(
            tensors / f"recal_preds_{cell_id}.npz",
            layers=np.asarray(layers),
            **{f"pred_recal_l{li}": pred_recal[li] for li in layers},
        )
        payload = {
            "metadata": d1._metadata(args.seed, len(folds)),
            "started_ts_unix": started_ts,
            "cell_id": cell_id,
            "n": len(folds),
            "verdict_layers": layers,
            "dg_e0": dge0,
            "per_layer": per_layer,
            "s_r": float(s_r),
            "s_r_argmax_layer": int(s_r_argmax),
            "recal_null": {
                "n_draws": int(args.recal_null_draws),
                "band_p975_layer_max": band,
                "layer_max_per_draw": [float(v) for v in null_layer_max],
            },
            "bootstrap": {
                "n_boot": int(args.n_boot),
                "s_r_per_draw": [float(v) for v in boot_s_r],
                "s_r_ci95": [
                    float(np.quantile(boot_s_r, 0.025)),
                    float(np.quantile(boot_s_r, 0.975)),
                ],
            },
            "fp16_crosscheck": _fp16_crosscheck(args, cell_id, Y_layers, layers),
            "gain_spectrum": gain_spec,
            "excess_decomposition": decomp,
            "seed_streams": {
                "null": args.seed + SEED_OFF_NULL,
                "bootstrap": args.seed + SEED_OFF_BOOT,
            },
            "wall_s": time.time() - started_ts,
        }
        d1._write_json(out_json, payload)
        _ckpt_write(args, f"recal_{cell_id}", fp)
        print(
            f"[recal] {cell_id}: S_r={s_r:.4f}@L{s_r_argmax} band={band:.4f} "
            f"ci95=[{payload['bootstrap']['s_r_ci95'][0]:.4f},"
            f"{payload['bootstrap']['s_r_ci95'][1]:.4f}]"
        )


# ---------------------------------------------------------------------------
# E1.b — fold-exchangeability per cell
# ---------------------------------------------------------------------------
def step_fold_exch(args) -> None:
    print("[recal1336] step=fold_exch", flush=True)
    for cell_id in args.cell_ids:
        fp = _regime_fingerprint(args, cell_id, "fold_exch")
        out_json = args.out_dir / f"fold_exch_{cell_id}.json"
        if _ckpt_ok(args, f"fold_exch_{cell_id}", fp, [out_json]):
            continue
        started_ts = time.time()
        inp = _load_cell_recal_inputs(args, cell_id)
        folds, layers = inp["folds"], inp["layers"]
        hr_path = args.out_dir / f"heldout_recal_{cell_id}.json"
        assert hr_path.exists(), f"{hr_path} missing — run --steps recal first"
        hr = json.loads(hr_path.read_text())

        norms: dict[str, dict] = {}
        ref_arrays: dict[str, np.ndarray] = {}
        for li in layers:
            Y_l = inp["Y_layers"][li]
            y_read, y_ref = _fold_norm_read(
                Y_l, folds, args.folds, args.n_repart, args.seed + SEED_OFF_REPART
            )
            resid = inp["P_layers"][li] - Y_l
            r_read, r_ref = _fold_norm_read(
                resid, folds, args.folds, args.n_repart, args.seed + SEED_OFF_REPART + 1
            )
            norms[str(li)] = {"y": y_read, "resid": r_read}
            ref_arrays[f"y_ref_l{li}"] = y_ref
            ref_arrays[f"resid_ref_l{li}"] = r_ref

        # (ii) seed-1 committed-convention refit at the verdict layers (CPU-ok).
        t0 = time.time()
        Xv = inp["X"][:, layers, :]
        Yv = inp["Y_full"][:, layers, :]
        sweep = fc.heldout_r2_sweep(
            Xv,
            Yv,
            inp["conv_ids"],
            n_folds=args.folds,
            seed=args.fold_rand_seed,
            null_draws=0,
            collect_cosines=False,
            frozen_layers=(),
        )
        r2_seed1 = {str(li): float(sweep["r2_obs"][ix]) for ix, li in enumerate(layers)}
        r2_seed0 = {str(li): hr["per_layer"][str(li)]["raw_r2"] for li in layers}
        l29 = _clamp_layer(d1.L_COMMITTED_ARGMAX, layers)
        move = abs(r2_seed1[str(l29)] - r2_seed0[str(l29)])
        print(f"[fold_exch] {cell_id} seed-1 refit {time.time() - t0:.1f}s move@L{l29}={move:.4f}")

        exact, norm_h, join_rate = _load_prompt_hashes(args, inp["conv_ids"])
        fold_of = {str(c): int(f) for c, f in zip(inp["conv_ids"], folds, strict=True)}
        dup = {
            "n_rows": len(inp["conv_ids"]),
            "join_rate": float(join_rate),
            "tiers": {
                "exact": _dup_tier_stats(exact, fold_of),
                "normalized": _dup_tier_stats(norm_h, fold_of),
            },
        }

        tensors = args.out_dir / "tensors"
        tensors.mkdir(parents=True, exist_ok=True)
        np.savez(
            tensors / f"fold_exch_draws_{cell_id}.npz", layers=np.asarray(layers), **ref_arrays
        )
        _, rows = _fold_rows(folds)
        payload = {
            "metadata": d1._metadata(args.seed, len(folds)),
            "started_ts_unix": started_ts,
            "cell_id": cell_id,
            "fold_sizes": [len(r) for r in rows],
            "n_repart": int(args.n_repart),
            "fold_mean_norms": norms,
            "seed_refit": {
                "seed0_source": "stored_preds_recompute",
                "fold_rand_seed": int(args.fold_rand_seed),
                "r2_seed0_per_layer": r2_seed0,
                "r2_seed1_per_layer": r2_seed1,
                "layer_l29": int(l29),
                "move_l29": float(move),
                "trigger1_threshold": E2_TRIGGER1_DELTA,
                "trigger1_fired": bool(move >= E2_TRIGGER1_DELTA),
                "sensitivity": {str(t): bool(move >= t) for t in E2_TRIGGER1_SENSITIVITY},
            },
            "dup_audit": dup,
            "seed_streams": {
                "repart_y": args.seed + SEED_OFF_REPART,
                "repart_resid": args.seed + SEED_OFF_REPART + 1,
            },
            "wall_s": time.time() - started_ts,
        }
        d1._write_json(out_json, payload)
        _ckpt_write(args, f"fold_exch_{cell_id}", fp)


# ---------------------------------------------------------------------------
# E1.e — verdict assembly (registered lattice + routing; --use-e2 re-read)
# ---------------------------------------------------------------------------
def route_verdict(
    *,
    s_prime_r: float,
    d_ci: tuple[float, float],
    v_gate: str,
    a_r: float,
    trigger1: bool,
    trigger2: bool,
    e2_fired: bool,
) -> tuple[str, str]:
    """Registered §4 decision routing (terminal — every branch shippable).

    v_gate is 'pass' | 'fail' | 'undefined' (bar_r fallback). Lattice
    (plan §3, DISJOINT + exhaustive): branch 1 usable strength <=> S'_r >= 0
    AND D_r CI positive; branch 2 weak <=> S'_r < 0 AND D_r CI positive;
    branch 3 absence otherwise. E2 triggers are consulted once (no second E2).
    """
    if v_gate == "fail":
        return "terminal_diagnosis_only", "v_gate_failed"
    if v_gate == "undefined":
        return "terminal_diagnosis_only", "bar_r_fallback_v_gate_undefined"
    if not e2_fired and (trigger1 or trigger2):
        return (
            "e2_refit_required",
            "trigger1_fold_indictment" if trigger1 else "trigger2_boundary_straddle",
        )
    d_positive = d_ci[0] > 0
    if d_positive and s_prime_r >= 0:
        if a_r >= A_R_BAR:
            return "resume_on_recalibrated_dv", "lattice_branch_1_accounted"
        return "terminal_diagnosis_only", "usable_strength_unaccounted_no_trigger"
    if d_positive:
        return "weak_transfer_scope", "lattice_branch_2"
    if a_r >= A_R_BAR:
        return "absence_with_account", "lattice_branch_3_accounted"
    return "terminal_diagnosis_only", "no_account_no_trigger"


def _lattice_branch(s_prime: float, d_ci: tuple[float, float]) -> int:
    if d_ci[0] > 0:
        return 1 if s_prime >= 0 else 2
    return 3


def step_verdict(args) -> None:
    print("[recal1336] step=verdict", flush=True)
    read_ts = time.time()
    qc_path = args.out_dir / "qwen_recal_cal.json"
    if qc_path.exists():
        qc = json.loads(qc_path.read_text())
        bar_r = float(qc["bar_r"])
        bar_r_fallback = False
        v_gate = "pass" if qc["v_gate"]["pass"] else "fail"
    else:
        qc = None
        bar_r = d1.BAR_RAW  # registered fallback (plan §3): E1.d unavailable only
        bar_r_fallback = True
        v_gate = "undefined"

    chat = args.cell_ids[0]
    src_label = "v5" if args.use_e2 else "e1"
    hr_name = f"refit_v5_{chat}.json" if args.use_e2 else f"heldout_recal_{chat}.json"
    hr_path = args.out_dir / hr_name
    assert hr_path.exists(), f"{hr_path} missing — run the {'e2' if args.use_e2 else 'recal'} step"
    hr = json.loads(hr_path.read_text())
    fe_path = args.out_dir / f"fold_exch_{chat}.json"
    assert fe_path.exists(), f"{fe_path} missing — run --steps fold_exch first"
    fe = json.loads(fe_path.read_text())

    # DG-E1 — calibration-ordering invariant (plan §7): bar_r computed BEFORE
    # the Llama verdict quantities were evaluated.
    if qc is not None:
        assert float(qc["computed_ts_unix"]) <= float(hr["started_ts_unix"]), (
            "DG-E1 ORDERING VIOLATION: qwen_recal computed AFTER the Llama recal "
            f"read started ({qc['computed_ts_unix']} > {hr['started_ts_unix']}) — "
            "bar_r must be set before any Llama verdict quantity is read"
        )
    dg_e1 = {
        "bar_r_computed_ts_unix": float(qc["computed_ts_unix"]) if qc else None,
        "llama_recal_started_ts_unix": float(hr["started_ts_unix"]),
        "verdict_read_ts_unix": read_ts,
        "ordering_ok": bool(qc is not None),
        "bar_r_fallback": bar_r_fallback,
    }

    layers = [int(v) for v in hr["verdict_layers"]]
    s_r = float(hr["s_r"])
    b_r = float(hr["recal_null"]["band_p975_layer_max"])
    boot = np.asarray(hr["bootstrap"]["s_r_per_draw"], dtype=float)
    s_ci = (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))
    d_r = s_r - b_r
    d_ci = (s_ci[0] - b_r, s_ci[1] - b_r)
    s_prime = s_r - bar_r

    l29 = _clamp_layer(d1.L_COMMITTED_ARGMAX, layers)
    r2_recal_l29 = float(hr["per_layer"][str(l29)]["heldout_recal_r2"])
    r2_v0_l29 = float(args.r2_v0_l29)
    denom = b_r - r2_v0_l29
    a_r = (r2_recal_l29 - r2_v0_l29) / denom if abs(denom) > 1e-12 else float("nan")

    trig1 = bool(fe["seed_refit"]["trigger1_fired"])
    trig2 = bool(v_gate == "pass" and s_ci[0] < bar_r < s_ci[1])
    routed, reason = route_verdict(
        s_prime_r=s_prime,
        d_ci=d_ci,
        v_gate=v_gate,
        a_r=a_r,
        trigger1=trig1,
        trigger2=trig2,
        e2_fired=bool(args.use_e2),
    )

    # Naturalistic robustness (reported, non-lattice): agreement in branch.
    nat = None
    if len(args.cell_ids) > 1:
        nat_name = (
            f"refit_v5_{args.cell_ids[1]}.json"
            if args.use_e2
            else f"heldout_recal_{args.cell_ids[1]}.json"
        )
        nat_path = args.out_dir / nat_name
        if nat_path.exists():
            hn = json.loads(nat_path.read_text())
            n_s_r = float(hn["s_r"])
            n_b = float(hn["recal_null"]["band_p975_layer_max"])
            n_boot = np.asarray(hn["bootstrap"]["s_r_per_draw"], dtype=float)
            n_ci = (
                float(np.quantile(n_boot, 0.025)) - n_b,
                float(np.quantile(n_boot, 0.975)) - n_b,
            )
            chat_branch = _lattice_branch(s_prime, d_ci)
            nat_branch = _lattice_branch(n_s_r - bar_r, n_ci)
            nat = {
                "cell_id": args.cell_ids[1],
                "s_r": n_s_r,
                "b_r": n_b,
                "s_prime_r": n_s_r - bar_r,
                "d_r_ci95": list(n_ci),
                "lattice_branch": nat_branch,
                "agrees_with_chat_branch": bool(nat_branch == chat_branch),
            }

    payload = {
        "metadata": d1._metadata(args.seed, int(hr["n"])),
        "read_source": src_label,
        "e2_fired": bool(args.use_e2),
        "dg_e1": dg_e1,
        "lattice_inputs": {
            "s_r": s_r,
            "s_r_argmax_layer": int(hr["s_r_argmax_layer"]),
            "b_r": b_r,
            "bar_r": bar_r,
            "bar_r_fallback": bar_r_fallback,
            "s_prime_r": s_prime,
            "d_r": d_r,
            "d_r_ci95": list(d_ci),
            "s_r_ci95": list(s_ci),
            "lattice_branch": _lattice_branch(s_prime, d_ci),
        },
        "v_gate": {
            "outcome": v_gate,
            "s_qwen_recal": float(qc["s_qwen_recal"]) if qc else None,
            "committed_anchor": d1.QWEN_COMMITTED_R2,
            "threshold": d1.QWEN_CAL_DEV_MAX,
        },
        "mechanism_account": {
            "a_r": float(a_r),
            "layer": int(l29),
            "r2_recal_l29": r2_recal_l29,
            "r2_v0_l29": r2_v0_l29,
            "r2_v0_l29_recomputed": float(hr["per_layer"][str(l29)]["raw_r2"]),
            "threshold": A_R_BAR,
            "accounts": bool(a_r >= A_R_BAR),
            "sensitivity": {str(t): bool(a_r >= t) for t in A_R_SENSITIVITY},
        },
        "fold_exchangeability": {
            "seed_refit_move_l29": float(fe["seed_refit"]["move_l29"]),
            "trigger1_fired": trig1,
            "trigger1_sensitivity": fe["seed_refit"]["sensitivity"],
            "fold_mean_norms_exceed": {
                li: {
                    "y": fe["fold_mean_norms"][li]["y"]["exceeds"],
                    "resid": fe["fold_mean_norms"][li]["resid"]["exceeds"],
                }
                for li in fe["fold_mean_norms"]
            },
            "dup_audit": fe["dup_audit"],
        },
        "e2_trigger": {
            "trigger1_fired": trig1,
            "trigger2_fired": trig2,
            "trigger2_definition": "V passes AND S_r 95% CI straddles bar_r (plan §4)",
            "fired": bool(trig1 or trig2),
        },
        "gain_spectrum_summary": {
            li: {
                k: hr["gain_spectrum"][li][k]
                for k in ("a_median", "a_frac_in_0p9_1p1", "spearman_a_vs_var")
            }
            for li in hr.get("gain_spectrum", {})
        },
        "per_layer": hr["per_layer"],
        "naturalistic": nat,
        "routed_decision": routed,
        "route_reason": reason,
    }
    d1._write_json(args.out_dir / "recal_verdict.json", payload)
    print(
        f"[verdict] S_r={s_r:.4f} B_r={b_r:.4f} bar_r={bar_r:.4f} S'={s_prime:.4f} "
        f"D=[{d_ci[0]:.4f},{d_ci[1]:.4f}] A_r={a_r:.3f} V={v_gate} -> {routed} ({reason})"
    )


# ---------------------------------------------------------------------------
# E2 — conditional refit leg (v5-fold | v5-cal)
# ---------------------------------------------------------------------------
def _ridge_predict_grid(cache: dict, Y_train: np.ndarray, lambdas) -> dict[float, np.ndarray]:
    """Per-lambda ridge predictions from ONE fold cache (the cached-eigh core:
    eigh/KevV reused; only the cheap per-lambda filter + GEMM vary). The E2
    nested-CV lambda-selection helper (plan §4 E2 trigger 2)."""
    Ytr = torch.as_tensor(np.asarray(Y_train), dtype=torch.float64).to(cache["w"].device)
    ymu = Ytr.mean(0)
    VtY = cache["V"].T @ (Ytr - ymu)
    out = {}
    for lam in np.asarray(lambdas, dtype=np.float64):
        filt = 1.0 / (cache["w"] + float(lam))
        out[float(lam)] = ((cache["KevV"] * filt) @ VtY + ymu).cpu().numpy()
    return out


def _nested_cv_layer(
    X: np.ndarray,
    Y: np.ndarray,
    conv_ids: np.ndarray,
    folds: np.ndarray,
    *,
    inner_folds: int,
    seed: int,
    lambdas,
) -> tuple[np.ndarray, dict[int, float]]:
    """v5-cal for ONE layer: per outer fold, select lambda by inner nested CV
    (objective = cross-fitted recalibrated held-out R^2 on the inner OOF
    preds), then predict the untouched outer fold at the selected lambda."""
    ids, rows = _fold_rows(folds)
    preds = np.zeros_like(Y, dtype=np.float64)
    lam_by_fold: dict[int, float] = {}
    lams = np.asarray(lambdas, dtype=np.float64)
    for ki, r in enumerate(rows):
        tr = np.setdiff1d(np.arange(len(folds)), r, assume_unique=True)
        inner = fc._cv_folds(conv_ids[tr], inner_folds, seed)
        inner_preds = {float(lam): np.zeros_like(Y[tr], dtype=np.float64) for lam in lams}
        for j in sorted(set(int(v) for v in inner)):
            ite = inner == j
            itr = ~ite
            if ite.sum() == 0 or itr.sum() < 3:
                continue
            cache = fc._prep_fold(X[tr][itr], X[tr][ite])
            grid = _ridge_predict_grid(cache, Y[tr][itr], lams)
            for lam, p in grid.items():
                inner_preds[lam][ite] = p
        best_lam, best_r2 = float(lams[0]), -np.inf
        for lam in lams:
            r2 = _crossfit_recal_direct(inner_preds[float(lam)], Y[tr], inner)["r2"]
            if np.isfinite(r2) and r2 > best_r2:
                best_lam, best_r2 = float(lam), float(r2)
        lam_by_fold[ids[ki]] = best_lam
        cache = fc._prep_fold(X[tr], X[r])
        preds[r] = fc._ridge_predict_cached(cache, Y[tr], lambdas=[best_lam])
    return preds, lam_by_fold


def _resolve_e2_variant(args) -> str:
    if args.e2_variant != "auto":
        return args.e2_variant
    v_path = args.out_dir / "recal_verdict.json"
    assert v_path.exists(), f"{v_path} missing — the E2 variant resolves from the E1 verdict"
    trig = json.loads(v_path.read_text())["e2_trigger"]
    assert trig["fired"], "E2 not triggered by the E1 verdict — refusing to run (plan §4)"
    # Trigger 1 (fold indictment) takes precedence: the fold treatment is the
    # more fundamental registered variant when both fire.
    return "fold" if trig["trigger1_fired"] else "cal"


def step_e2(args) -> None:
    print("[recal1336] step=e2", flush=True)
    variant = _resolve_e2_variant(args)
    print(f"[e2] variant=v5_{variant}")
    for cell_id in args.cell_ids:
        fp = {**_regime_fingerprint(args, cell_id, "e2"), "resolved_variant": variant}
        out_json = args.out_dir / f"refit_v5_{cell_id}.json"
        if _ckpt_ok(args, f"e2_{cell_id}", fp, [out_json]):
            continue
        started_ts = time.time()
        inp = _load_cell_recal_inputs(args, cell_id)
        layers = inp["layers"]
        conv_ids = inp["conv_ids"]
        is_chat = "naturalistic" not in cell_id
        Xv = inp["X"][:, layers, :]
        Yv = inp["Y_full"][:, layers, :]

        full_curve = None
        if variant == "fold":
            per_seed_preds: list[dict[int, np.ndarray]] = []
            per_seed_folds: list[np.ndarray] = []
            curves = []
            for s in E2_FOLD_SEEDS:
                sweep_layers = list(range(inp["X"].shape[1])) if is_chat else None
                if sweep_layers is not None:
                    # Full 32-layer chat curve (descriptive) via the committed path.
                    sw_full = fc.heldout_r2_sweep(
                        inp["X"],
                        inp["Y_full"],
                        conv_ids,
                        n_folds=args.folds,
                        seed=s,
                        null_draws=0,
                        collect_cosines=False,
                        frozen_layers=tuple(layers),
                    )
                    curves.append([float(v) for v in sw_full["r2_obs"]])
                    sweep = sw_full
                else:
                    sweep = fc.heldout_r2_sweep(
                        Xv,
                        Yv,
                        conv_ids,
                        n_folds=args.folds,
                        seed=s,
                        null_draws=0,
                        collect_cosines=False,
                        frozen_layers=tuple(range(len(layers))),
                    )
                if sweep_layers is not None:
                    preds_s = {li: sweep["preds_frozen"][li].astype(np.float64) for li in layers}
                else:
                    preds_s = {
                        layers[ix]: sweep["preds_frozen"][ix].astype(np.float64)
                        for ix in range(len(layers))
                    }
                per_seed_preds.append(preds_s)
                per_seed_folds.append(np.asarray(sweep["folds"]).astype(np.int64))
            if curves:
                full_curve = {"seeds": list(E2_FOLD_SEEDS), "r2_per_layer_per_seed": curves}
            # Median-over-seeds recal per layer; per-draw batteries median'd
            # across per-seed aligned draw streams (documented in the JSON).
            recal_by_layer = {}
            for li in layers:
                vals = [
                    _crossfit_recal_direct(per_seed_preds[si][li], inp["Y_layers"][li], f)["r2"]
                    for si, f in enumerate(per_seed_folds)
                ]
                recal_by_layer[li] = float(np.median(vals))
            null_stack, boot_stack = [], []
            weights = _bootstrap_weights(len(conv_ids), args.n_boot, args.seed + SEED_OFF_BOOT)
            for si, f in enumerate(per_seed_folds):
                nm, _ = _null_battery_matrix(
                    per_seed_preds[si],
                    inp["Y_layers"],
                    f,
                    args.recal_null_draws,
                    args.seed + SEED_OFF_E2_NULL + si,
                )
                null_stack.append(nm)
                bm, _ = _bootstrap_matrix(per_seed_preds[si], inp["Y_layers"], f, weights)
                boot_stack.append(bm)
            null_mat = np.median(np.stack(null_stack), axis=0)
            boot_mat = np.median(np.stack(boot_stack), axis=0)
            v5_extra = {
                "variant": "v5_fold",
                "fold_seeds": list(E2_FOLD_SEEDS),
                "full_chat_curve": full_curve,
            }
        else:
            preds_by_layer = {}
            lam_by_layer = {}
            folds = inp["folds"]  # stored seed-0 outer folds, untouched for the read
            for ix, li in enumerate(layers):
                preds, lam_by_fold = _nested_cv_layer(
                    Xv[:, ix, :],
                    inp["Y_layers"][li],
                    conv_ids,
                    folds,
                    inner_folds=args.inner_folds,
                    seed=args.seed,
                    lambdas=d1.LAMBDAS_WIDE,
                )
                preds_by_layer[li] = preds
                lam_by_layer[str(li)] = {str(k): v for k, v in lam_by_fold.items()}
            recal_by_layer = {
                li: _crossfit_recal_direct(preds_by_layer[li], inp["Y_layers"][li], folds)["r2"]
                for li in layers
            }
            null_mat, _ = _null_battery_matrix(
                preds_by_layer,
                inp["Y_layers"],
                folds,
                args.recal_null_draws,
                args.seed + SEED_OFF_E2_NULL,
            )
            weights = _bootstrap_weights(len(conv_ids), args.n_boot, args.seed + SEED_OFF_BOOT)
            boot_mat, _ = _bootstrap_matrix(preds_by_layer, inp["Y_layers"], folds, weights)
            v5_extra = {
                "variant": "v5_cal",
                "inner_folds": int(args.inner_folds),
                "lambda_by_layer_fold": lam_by_layer,
                "lambda_grid": "logspace(-2,8,21)",
            }

        null_layer_max = np.nanmax(null_mat, axis=1)
        boot_s_r = np.nanmax(boot_mat, axis=1)
        s_r = max(recal_by_layer.values())
        s_r_argmax = max(recal_by_layer, key=lambda k: recal_by_layer[k])
        tensors = args.out_dir / "tensors"
        tensors.mkdir(parents=True, exist_ok=True)
        np.savez(
            tensors / f"refit_v5_draws_{cell_id}.npz",
            layers=np.asarray(layers),
            null_r2_matrix=null_mat,
            null_layer_max=null_layer_max,
            boot_r2_matrix=boot_mat,
            boot_s_r=boot_s_r,
        )
        hr = json.loads((args.out_dir / f"heldout_recal_{cell_id}.json").read_text())
        payload = {
            "metadata": d1._metadata(args.seed, len(conv_ids)),
            "started_ts_unix": started_ts,
            "cell_id": cell_id,
            "n": len(conv_ids),
            "verdict_layers": layers,
            **v5_extra,
            "per_layer": {
                str(li): {
                    "heldout_recal_r2": recal_by_layer[li],
                    "raw_r2": hr["per_layer"][str(li)]["raw_r2"],
                }
                for li in layers
            },
            "s_r": float(s_r),
            "s_r_argmax_layer": int(s_r_argmax),
            "recal_null": {
                "n_draws": int(args.recal_null_draws),
                "band_p975_layer_max": float(np.quantile(null_layer_max, 0.975)),
                "layer_max_per_draw": [float(v) for v in null_layer_max],
            },
            "bootstrap": {
                "n_boot": int(args.n_boot),
                "s_r_per_draw": [float(v) for v in boot_s_r],
                "s_r_ci95": [
                    float(np.quantile(boot_s_r, 0.025)),
                    float(np.quantile(boot_s_r, 0.975)),
                ],
            },
            "wall_s": time.time() - started_ts,
        }
        d1._write_json(out_json, payload)
        _ckpt_write(args, f"e2_{cell_id}", fp)
        print(f"[e2] {cell_id}: v5 S_r={s_r:.4f}@L{s_r_argmax}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
STEP_FNS = {
    "stage": step_stage,
    "qwen_recal": step_qwen_recal,
    "recal": step_recal,
    "fold_exch": step_fold_exch,
    "verdict": step_verdict,
    "e2": step_e2,
}


def normalize_steps(raw: str) -> list[str]:
    """Canonical execution order regardless of CLI order (DG-E1: qwen_recal
    before recal before verdict). Unknown step names fail loud."""
    req = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [s for s in req if s not in STEP_ORDER]
    assert not unknown, f"unknown steps {unknown}; valid: {STEP_ORDER}"
    return [s for s in STEP_ORDER if s in req]


def main() -> int:
    args = parse_args()
    args.cell_ids = [c.strip() for c in args.cells.split(",") if c.strip()]
    assert args.cell_ids, "--cells resolved to an empty list"
    assert 3 <= args.inner_folds <= 5, "plan §12: E2 nested-CV inner folds within [3, 5]"
    targets = dict(DGE0_TARGETS)
    if args.dge0_targets_json:
        targets.update({k: float(v) for k, v in json.loads(args.dge0_targets_json).items()})
    args.dge0_targets = targets
    args.out_dir.mkdir(parents=True, exist_ok=True)
    steps = normalize_steps(args.steps)
    print(f"[recal1336] steps={steps} cells={args.cell_ids} out={args.out_dir}")
    for s in steps:
        STEP_FNS[s](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
