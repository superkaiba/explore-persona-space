#!/usr/bin/env python
"""Issue #1482 free-analysis follow-up: per-feature shuffle-null for H2.

Recomputes per-feature held-out R2 of the mean-pooling context-SAE -> answer-SAE
ridge (the #1482 P3 `sae_ctx / mean / ridge` unit) under K seeded LABEL-SHUFFLED
fits: permute Y rows WITHIN the fit set, refit the ridge at the parent's
SELECTED lambda (10000), score per-feature R2 on the UNTOUCHED holdout (true
pairs). Adjudicates whether the observed negative-R2 tail (290 above-median-
activity features < 0; 16.5% -> 1.6% below-zero across activity deciles) is
within the mechanical estimation-noise floor or in excess of it.

Recipe parity with `issue1482_error_analysis.py` (worktree `issue-1482`, P3
`_p3_prep` + `_shared_gram_ridge_multi` + `_per_feature_metrics`), replicated
inline because that driver is not on `main`:
  - identical feature restriction (activity >= ceil(0.01*n_fit), top-16384 out /
    top-8192 in by count; asserted equal to the parent npz `feat_ids`),
  - identical row registry (sae_fit ++ sae_val ++ holdout filtered to captured
    rows), identical unbiased train standardizer (n-1 denom, +1e-9),
  - identical shared-Gram eigh ridge; lambda PINNED to the parent's selected
    10000 (no val re-selection) so the null isolates the label shuffle,
  - identical per-feature R2 formula (ss_tot on the holdout's own mean).

Compute shape (vectorize-first): ONE fp64 Gram + eigh; per draw = blocked
X_std^T @ Y_perm block-GEMMs (fp32 GEMM, fp64 accumulate) + one M2 @ XtY GEMM
(M2 = En_std U diag(1/(s+lam)) U^T precomputed once); no per-feature loops.
An IDENTITY-permutation pass through the same draw pipeline must reproduce the
parent's committed per-feature R2 (wiring + fp32 precision gate).

Checkpoint/resume: one npy per draw under --draw-dir (regime-keyed meta.json).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM discipline)

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

DICT_SIZE = 131_072  # issue1482_sae.DICT_SIZE
MAX_FEATURES_IN = 8192  # parent production caps (issue1482_error_analysis dd)
MAX_FEATURES_OUT = 16384
LAM = 10000.0  # parent-selected lambda for sae_ctx/mean/ridge (unit_ridge__sae_ctx.json)
SEED_BASE = 1482 * 1000
K_DRAWS = 20
BLOCK = 8192
WT_DEFAULT = PROJECT_ROOT / ".claude" / "worktrees" / "issue-1482"
STORE_DEFAULT = Path(
    "/mnt/eps-data/thomasjiralerspong/issue1482_shuffnull/"
    "issue1482_error_analysis/analysis_tensors/sae_pooled"
)


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [shuffnull] {msg}", flush=True)


def _rss_gb() -> float:
    """Peak RSS of this process in GiB (ru_maxrss is KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    return out.stdout.strip() if out.returncode == 0 else "unknown"


# ── pass 1: activity counts + row inventory (streamed; parity with _activity_counts) ──


def scan_counts(shards: list[Path]) -> dict:
    """Stream all shards once: per-feature occurrence counts over fit-tagged rows
    (tag==1; parity with parent `_activity_counts`), all-row counts (for nnz
    projection), and the captured-row inventory."""
    out_fit = np.zeros(DICT_SIZE, dtype=np.int64)
    in_fit = np.zeros(DICT_SIZE, dtype=np.int64)
    out_all = np.zeros(DICT_SIZE, dtype=np.int64)
    in_all = np.zeros(DICT_SIZE, dtype=np.int64)
    inlast_all = np.zeros(DICT_SIZE, dtype=np.int64)
    n_fit = 0
    rows_all: list[np.ndarray] = []
    for p in shards:
        with np.load(p, allow_pickle=False) as z:
            tag = z["set_tag"]
            rows_all.append(np.asarray(z["row_idx"], dtype=np.int64))
            fit_mask = tag == 1
            n_fit += int(fit_mask.sum())
            for key_idx, key_off, cnt_fit, cnt_all in (
                ("ans_idx", "idx_off", out_fit, out_all),
                ("psi_idx", "psi_off", in_fit, in_all),
                ("psil_idx", "psil_off", None, inlast_all),
            ):
                idx = np.asarray(z[key_idx], dtype=np.int64)
                off = np.asarray(z[key_off], dtype=np.int64)
                cnt_all += np.bincount(idx, minlength=DICT_SIZE)
                if cnt_fit is not None:
                    keep = np.repeat(fit_mask, off)
                    cnt_fit += np.bincount(idx[keep], minlength=DICT_SIZE)
    return {
        "out_fit": out_fit,
        "in_fit": in_fit,
        "out_all": out_all,
        "in_all": in_all,
        "inlast_all": inlast_all,
        "n_fit": n_fit,
        "have": np.unique(np.concatenate(rows_all)),
    }


def restrict(counts: np.ndarray, n_fit: int, cap: int) -> tuple[np.ndarray, int]:
    """Parent `_p3_prep` feature restriction: floor = ceil(1% of fit rows), then
    top-`cap` by count, sorted ascending. Returns (kept feature ids, floor)."""
    floor = max(1, int(np.ceil(0.01 * n_fit)))
    f = np.where(counts >= floor)[0]
    if len(f) > cap:
        f = f[np.argsort(-counts[f])[:cap]]
        f = np.sort(f)
    assert len(f) >= 1, len(f)
    return f, floor


# ── pass 2: sparse design/target build (COO -> CSR; parity with _densify) ──────────


def build_csr(
    shards: list[Path],
    row_pos: dict[int, int],
    n_rows: int,
    f_in: np.ndarray,
    f_out: np.ndarray,
    counts: dict,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Stream shards again and build X = [psi_mean | psi_last] (n_rows, 2*|f_in|)
    and Y = ans_mean (n_rows, |f_out|) as CSR. Equivalent to the parent's dense
    `_densify` scatter (same col_of mapping, same row registry), kept sparse.

    Memory-bounded (the 16 GiB VM cap): COO arrays are PREALLOCATED at the exact
    kept-nnz sizes known from the pass-1 all-row counts (the registry covers every
    captured row, so only the column restriction drops entries). Values upcast
    f16 -> f32 exactly as the parent's densify did (scipy has no float16 CSR)."""
    h_in = len(f_in)
    col_in = np.full(DICT_SIZE, -1, dtype=np.int64)
    col_in[f_in] = np.arange(h_in)
    col_out = np.full(DICT_SIZE, -1, dtype=np.int64)
    col_out[f_out] = np.arange(len(f_out))
    nnz_x = int(counts["in_all"][f_in].sum() + counts["inlast_all"][f_in].sum())
    nnz_y = int(counts["out_all"][f_out].sum())
    xr = np.empty(nnz_x, np.int32)
    xc = np.empty(nnz_x, np.int32)
    xv = np.empty(nnz_x, np.float32)
    yr = np.empty(nnz_y, np.int32)
    yc = np.empty(nnz_y, np.int32)
    yv = np.empty(nnz_y, np.float32)
    cursors = {"x": 0, "y": 0}
    for p in shards:
        with np.load(p, allow_pickle=False) as z:
            pos = np.asarray([row_pos.get(int(r), -1) for r in z["row_idx"]], dtype=np.int64)
            for key_idx, key_off, key_val, cmap, coff, which in (
                ("psi_idx", "psi_off", "psi_mean", col_in, 0, "x"),
                ("psil_idx", "psil_off", "psil_val", col_in, h_in, "x"),
                ("ans_idx", "idx_off", "ans_mean", col_out, 0, "y"),
            ):
                idx = np.asarray(z[key_idx], dtype=np.int64)
                off = np.asarray(z[key_off], dtype=np.int64)
                rr = np.repeat(pos, off)
                cc = cmap[idx]
                keep = (rr >= 0) & (cc >= 0)
                n = int(keep.sum())
                c = cursors[which]
                rows, cols, vals = (xr, xc, xv) if which == "x" else (yr, yc, yv)
                rows[c : c + n] = rr[keep]
                cols[c : c + n] = cc[keep] + coff
                vals[c : c + n] = np.asarray(z[key_val], dtype=np.float32)[keep]
                cursors[which] = c + n
    assert cursors["x"] == nnz_x and cursors["y"] == nnz_y, (cursors, nnz_x, nnz_y)
    Y = sp.coo_matrix((yv, (yr, yc)), shape=(n_rows, len(f_out))).tocsr()
    del yr, yc, yv
    X = sp.coo_matrix((xv, (xr, xc)), shape=(n_rows, 2 * h_in)).tocsr()
    del xr, xc, xv
    return X, Y


# ── shared factorization + per-draw pipeline ────────────────────────────────────────


def dense_block(csr: sp.csr_matrix, rows: np.ndarray) -> np.ndarray:
    """Densify a row block of a CSR matrix to fp32 (nnz-proportional)."""
    return csr[rows].toarray().astype(np.float32, copy=False)


def train_standardizer(X: sp.csr_matrix, tr: np.ndarray, block: int) -> tuple:
    """Parent `_train_standardizer`: fp64 streaming train mean/std of X, UNBIASED
    (n-1) variance, xsd = sqrt(clamp(var,0)) + 1e-9."""
    h = X.shape[1]
    sum_x = np.zeros(h, dtype=np.float64)
    sumsq_x = np.zeros(h, dtype=np.float64)
    n = 0
    for s in range(0, len(tr), block):
        xb = dense_block(X, tr[s : s + block]).astype(np.float64)
        sum_x += xb.sum(0)
        sumsq_x += (xb * xb).sum(0)
        n += xb.shape[0]
    xmu = sum_x / n
    var = (sumsq_x - n * xmu * xmu) / max(1, n - 1)
    xsd = np.sqrt(np.clip(var, 0.0, None)) + 1e-9
    return xmu, xsd


def factorize(X: sp.csr_matrix, tr: np.ndarray, xmu, xsd, block: int) -> dict:
    """fp64 Gram A = X_std[tr]^T X_std[tr] (blocked, exact parent parity) + eigh.
    Also returns colsum of standardized train X (for the exact ymu correction)."""
    h = X.shape[1]
    A = np.zeros((h, h), dtype=np.float64)
    colsum = np.zeros(h, dtype=np.float64)
    for s in range(0, len(tr), block):
        xb = (dense_block(X, tr[s : s + block]).astype(np.float64) - xmu) / xsd
        A += xb.T @ xb
        colsum += xb.sum(0)
    _log(f"gram done (rss {_rss_gb():.1f} GiB); eigh H={h} ...")
    from scipy.linalg import eigh as scipy_eigh  # overwrite_a: no extra (H,H) fp64 copy

    s_eig, U = scipy_eigh(A, overwrite_a=True, check_finite=False)
    s_eig = np.clip(s_eig, 0.0, None)
    return {"U": U, "s_eig": s_eig, "colsum_xstd": colsum}


def build_m2(
    X: sp.csr_matrix,
    te: np.ndarray,
    xmu,
    xsd,
    U: np.ndarray,
    s_eig: np.ndarray,
    lam: float,
    block: int,
) -> np.ndarray:
    """M2 = En_std @ U @ diag(1/(s+lam)) @ U^T (n_te, H) in fp64 blocked, stored
    fp32 — pred = M2 @ (X_std^T Y_c) + ymu, so each draw needs no U access."""
    inv = 1.0 / (s_eig + lam)
    out = np.empty((len(te), U.shape[0]), dtype=np.float32)
    for s in range(0, len(te), block):
        eb = (dense_block(X, te[s : s + block]).astype(np.float64) - xmu) / xsd
        out[s : s + eb.shape[0]] = ((eb @ U) * inv) @ U.T
    return out


def draw_r2(
    X: sp.csr_matrix,
    Y: sp.csr_matrix,
    tr: np.ndarray,
    te: np.ndarray,
    perm: np.ndarray,
    xmu32,
    xsd32,
    ymu,
    colsum_xstd,
    M2: np.ndarray,
    ss_tot: np.ndarray,
    block: int,
) -> np.ndarray:
    """Per-feature holdout R2 for ONE label permutation of the fit set.

    XtY = sum_blocks X_std[blk]^T @ Y[tr[perm[blk]]] (fp32 GEMM, fp64 accumulate);
    centered via the exact colsum correction; pred = M2 @ XtY_c + ymu; R2 with the
    parent `_per_feature_metrics` formula (ss_tot precomputed on the holdout)."""
    h, d = X.shape[1], Y.shape[1]
    XtY = np.zeros((h, d), dtype=np.float64)
    for s in range(0, len(tr), block):
        xb = dense_block(X, tr[s : s + block])
        xb -= xmu32
        xb /= xsd32
        yb = dense_block(Y, tr[perm[s : s + block]])
        XtY += xb.T @ yb
    XtY -= np.outer(colsum_xstd, ymu)
    pred = M2 @ XtY.astype(np.float32)
    pred += ymu.astype(np.float32)
    ss_res = np.zeros(d, dtype=np.float64)
    for s in range(0, len(te), block):
        tb = dense_block(Y, te[s : s + block])
        tb -= pred[s : s + tb.shape[0]]
        ss_res += (tb * tb).sum(0, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_tot > 1e-12, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)


def holdout_ss_tot(Y: sp.csr_matrix, te: np.ndarray, block: int) -> tuple[np.ndarray, ...]:
    """Per-feature SS_tot on the holdout's OWN mean (parity with
    `_per_feature_metrics`), fp64 blocked; also returns the holdout column mean."""
    d = Y.shape[1]
    sum_t = np.zeros(d, dtype=np.float64)
    sumsq_t = np.zeros(d, dtype=np.float64)
    for s in range(0, len(te), block):
        tb = dense_block(Y, te[s : s + block]).astype(np.float64)
        sum_t += tb.sum(0)
        sumsq_t += (tb * tb).sum(0)
    mu = sum_t / len(te)
    return sumsq_t - len(te) * mu * mu, mu


# ── aggregation ─────────────────────────────────────────────────────────────────────


def decile_bins(activity: np.ndarray) -> np.ndarray:
    """Activity decile index per feature (0 = least active, 9 = most active) —
    the convention that reproduces the #1482 clean-result decile stats."""
    qs = np.quantile(activity, np.linspace(0, 1, 11))
    return np.clip(np.searchsorted(qs[1:-1], activity, side="right"), 0, 9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, default=STORE_DEFAULT)
    ap.add_argument("--scratch", type=Path, default=WT_DEFAULT / "data/issue_1482/scratch")
    ap.add_argument(
        "--parent-npz",
        type=Path,
        default=WT_DEFAULT / "eval_results/issue_1482/sae_perfeature/sae_ctx__mean__ridge.npz",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_1482/sae_perfeature/shuffle_null.json",
    )
    ap.add_argument(
        "--out-npz",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_1482/sae_perfeature/shuffle_null_perfeature.npz",
    )
    ap.add_argument(
        "--fig",
        type=Path,
        default=PROJECT_ROOT / "figures/issue_1482/perfeature_r2_vs_activity_null.png",
    )
    ap.add_argument(
        "--draw-dir",
        type=Path,
        default=Path("/mnt/eps-data/thomasjiralerspong/issue1482_shuffnull/draws"),
    )
    ap.add_argument("--k-draws", type=int, default=K_DRAWS)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument(
        "--probe",
        action="store_true",
        help="reduced-shape RSS/wall probe (tr cap 20k, te cap 5k, K=2; no canonical writes)",
    )
    args = ap.parse_args()
    t0 = time.time()

    shards = sorted(args.store.glob("pooled_*.npz"))
    assert shards, f"no pooled shards under {args.store}"
    _log(f"{len(shards)} shards under {args.store}")

    idx = np.load(args.scratch / "split_indices.npz")
    parent = np.load(args.parent_npz)
    parent_feat, parent_r2, parent_act = parent["feat_ids"], parent["r2"], parent["activity"]

    t_scan = time.time()
    counts = scan_counts(shards)
    _log(f"pass1 counts done in {time.time() - t_scan:.0f}s (n_fit={counts['n_fit']})")
    f_out, floor = restrict(counts["out_fit"], counts["n_fit"], MAX_FEATURES_OUT)
    f_in, _ = restrict(counts["in_fit"], counts["n_fit"], MAX_FEATURES_IN)
    assert np.array_equal(f_out, parent_feat), (
        "recomputed f_out does not match the parent npz feat_ids — recipe drift"
    )
    activity = counts["out_fit"][f_out] / max(1, counts["n_fit"])
    assert np.allclose(activity, parent_act, atol=1e-9), "activity mismatch vs parent npz"
    nnz_proj = int(
        counts["in_all"][f_in].sum()
        + counts["inlast_all"][f_in].sum()
        + counts["out_all"][f_out].sum()
    )
    _log(f"restriction: F_out={len(f_out)} F_in={len(f_in)} floor={floor}; nnz~{nnz_proj:.3e}")

    # row registry — parity with _p3_prep
    have = set(int(r) for r in counts["have"])
    order = np.concatenate([idx["sae_fit"], idx["sae_val"], idx["holdout"]])
    order = np.asarray([r for r in order if int(r) in have], dtype=np.int64)
    row_pos = {int(r): i for i, r in enumerate(order)}
    tr = np.asarray([row_pos[int(r)] for r in idx["sae_fit"] if int(r) in row_pos], np.int64)
    te = np.asarray([row_pos[int(r)] for r in idx["holdout"] if int(r) in row_pos], np.int64)
    assert len(tr) and len(te), (len(tr), len(te))
    _log(f"rows: n_tr={len(tr)} n_te={len(te)} (registry {len(order)})")
    if not args.probe:
        assert len(tr) == 120_000 and len(te) == 20_000, (len(tr), len(te))

    if args.probe:
        tr, te = tr[:20_000], te[:5_000]
        args.k_draws = 2
        _log(f"PROBE mode: tr capped to {len(tr)}, te to {len(te)}, K={args.k_draws}")

    t_build = time.time()
    X, Y = build_csr(shards, row_pos, len(order), f_in, f_out, counts)
    _log(
        f"pass2 CSR built in {time.time() - t_build:.0f}s: X nnz={X.nnz:.3e} "
        f"Y nnz={Y.nnz:.3e} (rss {_rss_gb():.1f} GiB)"
    )

    t_fac = time.time()
    xmu, xsd = train_standardizer(X, tr, args.block)
    fac = factorize(X, tr, xmu, xsd, args.block)
    t_eigh = time.time() - t_fac
    _log(f"standardizer+gram+eigh done in {t_eigh:.0f}s (rss {_rss_gb():.1f} GiB)")

    # ymu (train mean of Y, fp64 via CSR column sums over tr rows)
    ymu = np.asarray(Y[tr].astype(np.float64).sum(axis=0)).ravel() / len(tr)
    t_m2 = time.time()
    M2 = build_m2(X, te, xmu, xsd, fac["U"], fac["s_eig"], args.lam, args.block)
    _log(f"M2 built in {time.time() - t_m2:.0f}s (rss {_rss_gb():.1f} GiB)")
    del fac["U"]  # (n_te,H) fp32 M2 replaces U for every draw
    ss_tot, _mu_te = holdout_ss_tot(Y, te, args.block)
    xmu32, xsd32 = xmu.astype(np.float32), xsd.astype(np.float32)

    # identity gate: same pipeline, identity permutation -> must reproduce parent r2
    t_id = time.time()
    r2_id = draw_r2(
        X,
        Y,
        tr,
        te,
        np.arange(len(tr)),
        xmu32,
        xsd32,
        ymu,
        fac["colsum_xstd"],
        M2,
        ss_tot,
        args.block,
    )
    t_draw_s = time.time() - t_id
    pooled_id = (
        1.0
        - ((1.0 - r2_id[np.isfinite(r2_id)]) * ss_tot[np.isfinite(r2_id)]).sum()
        / ss_tot[np.isfinite(r2_id)].sum()
    )
    gate: dict = {"identity_draw_seconds": t_draw_s, "identity_pooled_r2": float(pooled_id)}
    if not args.probe:
        d = np.abs(r2_id - parent_r2)
        gate.update(
            identity_vs_parent_abs_diff={
                "median": float(np.nanmedian(d)),
                "q99": float(np.nanquantile(d, 0.99)),
                "max": float(np.nanmax(d)),
            },
            parent_pooled_r2=0.6901409540488584,
        )
        _log(f"identity gate: {gate}")
        # wiring bug => whole-distribution shift (median O(0.1)); a handful of
        # near-degenerate ss_tot features may show larger fp32-vs-fp64 diffs, so
        # gate median + q99, report max.
        assert np.nanmedian(d) < 5e-3 and np.nanquantile(d, 0.99) < 0.05, gate
    else:
        _log(f"probe identity draw: {t_draw_s:.0f}s pooled_r2={pooled_id:.4f}")

    if args.probe:
        # wall + RSS projection for the full run (tr x6, te x4, K=20)
        full_draw = t_draw_s * (120_000 / max(1, len(tr)))
        proj = {
            "probe_rss_gib": _rss_gb(),
            "probe_draw_seconds": t_draw_s,
            "projected_full_draw_seconds": full_draw,
            "projected_20_draws_hours": 21 * full_draw / 3600,  # 20 draws + identity
            "projected_extra_csr_gib": (nnz_proj - (X.nnz + Y.nnz)) * 12 / 1024**3,
            "projected_extra_m2_pred_gib": (20_000 - len(te)) * 16_384 * 2 * 4 / 1024**3,
        }
        _log(f"PROBE projections: {json.dumps(proj, indent=1)}")
        (args.draw_dir / "probe").mkdir(parents=True, exist_ok=True)
        (args.draw_dir / "probe" / "probe.json").write_text(json.dumps(proj, indent=1))
        return

    # draws with per-draw checkpoint + regime-keyed resume
    args.draw_dir.mkdir(parents=True, exist_ok=True)
    regime = {
        "lam": args.lam,
        "seed_base": args.seed_base,
        "k_draws": args.k_draws,
        "n_tr": len(tr),
        "n_te": len(te),
        "f_out_first_last": [int(f_out[0]), int(f_out[-1])],
    }
    meta_p = args.draw_dir / "meta.json"
    if meta_p.exists():
        assert json.loads(meta_p.read_text()) == regime, "draw-dir regime mismatch — clear it"
    else:
        meta_p.write_text(json.dumps(regime))
    null_r2 = np.empty((args.k_draws, len(f_out)), dtype=np.float32)
    for k in range(args.k_draws):
        dp = args.draw_dir / f"draw_{k:02d}.npy"
        if dp.exists():
            null_r2[k] = np.load(dp)
            _log(f"draw {k}: resumed from checkpoint")
            continue
        tk = time.time()
        perm = np.random.default_rng(args.seed_base + k).permutation(len(tr))
        null_r2[k] = draw_r2(
            X, Y, tr, te, perm, xmu32, xsd32, ymu, fac["colsum_xstd"], M2, ss_tot, args.block
        ).astype(np.float32)
        tmp = dp.with_suffix(".tmp.npy")
        np.save(tmp, null_r2[k])
        os.replace(tmp, dp)
        _log(f"draw {k}: {time.time() - tk:.0f}s (rss {_rss_gb():.1f} GiB)")

    # ── aggregation + adjudication ──
    q = np.nanquantile(null_r2, [0.025, 0.5, 0.975], axis=0)
    null_lo, null_med, null_hi = q[0], q[1], q[2]
    null_mean = np.nanmean(null_r2, axis=0)
    obs = parent_r2
    dec = decile_bins(activity)
    obs_neg = obs < 0
    below_band = obs < null_lo
    above_band = obs > null_hi
    # leave-one-out calibration: empirical rate at which a held-out null draw falls
    # below the q2.5 of the other K-1 draws (the honest false-positive rate of the
    # per-feature band at K draws; expected > 2.5% at K=20)
    loo_below = 0
    for k in range(args.k_draws):
        others = np.delete(null_r2, k, axis=0)
        loo_below += int((null_r2[k] < np.nanquantile(others, 0.025, axis=0)).sum())
    loo_rate = loo_below / (args.k_draws * len(f_out))
    per_decile = []
    for kdec in range(10):
        m = dec == kdec
        pooled = null_r2[:, m].ravel()
        pooled = pooled[np.isfinite(pooled)]
        per_decile.append(
            {
                "decile": kdec,
                "n_features": int(m.sum()),
                "activity_min": float(activity[m].min()),
                "activity_max": float(activity[m].max()),
                "obs_median_r2": float(np.median(obs[m])),
                "obs_below_zero_share": float(obs_neg[m].mean()),
                "null_below_zero_share": float((pooled < 0).mean()),
                "null_q2p5": float(np.quantile(pooled, 0.025)),
                "null_median": float(np.median(pooled)),
                "null_q97p5": float(np.quantile(pooled, 0.975)),
                "obs_below_own_band_share": float(below_band[m].mean()),
            }
        )
    above_med_act = activity > np.median(activity)
    neg290 = above_med_act & obs_neg
    adjudication = {
        "a_obs_below_zero_total": int(obs_neg.sum()),
        "a_obs_below_zero_within_band": int((obs_neg & ~below_band & ~above_band).sum()),
        "a_obs_below_zero_below_band": int((obs_neg & below_band).sum()),
        "a_obs_below_zero_above_band": int((obs_neg & above_band).sum()),
        "b_per_decile": per_decile,
        "c_above_median_activity_negative_n": int(neg290.sum()),
        "c_of_those_below_own_null_q2p5": int((neg290 & below_band).sum()),
        "all_features_below_band_share": float(below_band.mean()),
        "all_features_above_band_share": float(above_band.mean()),
        "band_loo_calibration_below_rate": float(loo_rate),
    }
    doc = {
        "goal": "shuffle-null estimation-noise floor for #1482 H2 per-feature R2",
        "recipe": {
            "unit": "sae_ctx / mean / ridge (parent P3)",
            "lambda": args.lam,
            "k_draws": args.k_draws,
            "seeds": [args.seed_base + k for k in range(args.k_draws)],
            "permutation": "Y rows within the 120k fit set; holdout untouched (true pairs)",
            "n_fit": len(tr),
            "n_holdout": len(te),
            "f_out": len(f_out),
            "f_in": len(f_in),
            "draw_arithmetic": "fp32 block-GEMMs, fp64 accumulate; fp64 gram+eigh",
        },
        "identity_gate": gate,
        "adjudication": adjudication,
        "caveats": [
            "per-feature null band from K=20 draws is coarse: the leave-one-out "
            "calibration rate above is the honest false-positive rate of the "
            "'below own q2.5' read (expected ~5-7% at K=20, not 2.5%)",
            "per-decile pooled bands (~33k null values/decile) are the better-"
            "calibrated aggregate read",
        ],
        "inputs": {
            "store": str(args.store),
            "n_shards": len(shards),
            "scratch_split": str(args.scratch / "split_indices.npz"),
            "parent_npz": str(args.parent_npz),
        },
        "metadata": {
            "git_commit": _git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numpy": np.__version__,
            "peak_rss_gib": _rss_gb(),
            "wall_seconds": time.time() - t0,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(doc, indent=1))
    np.savez(
        args.out_npz,
        feat_ids=f_out,
        activity=activity,
        obs_r2=obs,
        null_mean=null_mean,
        null_q2p5=null_lo,
        null_median=null_med,
        null_q97p5=null_hi,
        decile=dec,
    )
    _log(f"wrote {args.out_json} + {args.out_npz}")

    # ── figure: hero2 scatter shape + per-decile null band overlay ──
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5), layout="constrained")
    ok = np.isfinite(obs)
    ax.scatter(
        activity[ok],
        np.clip(obs[ok], -1, 1),
        s=5,
        alpha=0.35,
        color="#0173b2",
        label="observed per-feature held-out R2",
    )
    xs = [float(np.median(activity[dec == kdec])) for kdec in range(10)]
    lo = [d["null_q2p5"] for d in per_decile]
    hi = [d["null_q97p5"] for d in per_decile]
    md = [d["null_median"] for d in per_decile]
    om = [d["obs_median_r2"] for d in per_decile]
    ax.fill_between(
        xs,
        np.clip(lo, -1, 1),
        np.clip(hi, -1, 1),
        color="#de8f05",
        alpha=0.35,
        label="shuffle-null 2.5-97.5% band (per activity decile, K=20 draws)",
    )
    ax.plot(xs, md, color="#de8f05", lw=1.5, marker="o", ms=3, label="null median")
    ax.plot(xs, om, color="#029e73", lw=1.5, marker="s", ms=4, label="observed decile median")
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("feature activity (fraction of 120k fit contexts)")
    ax.set_ylabel("per-feature held-out R2 (clipped at -1)")
    ax.set_title("#1482 H2: per-feature R2 vs activity, with label-shuffle null band")
    ax.legend(fontsize=7, loc="lower right")
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=200)
    plt.close(fig)
    args.fig.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "caption": (
                    "Observed per-feature held-out R2 of the mean-pooling context-SAE to "
                    "answer-SAE ridge (blue, 16,384 features) against the label-shuffle "
                    "null (orange band: per-activity-decile pooled 2.5-97.5% quantiles "
                    "over K=20 permuted fits at the same lambda=1e4). Green squares are "
                    "observed decile medians. Points below the orange band are worse than "
                    "the mechanical estimation-noise floor."
                ),
                "source_json": str(args.out_json),
                "git_commit": _git_commit(),
            },
            indent=1,
        )
    )
    _log(f"wrote {args.fig}; total wall {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
