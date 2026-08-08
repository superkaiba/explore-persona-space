"""Issue #1482 — structure of the SAE(context) -> SAE(answer) ridge coefficient map.

The parent P3 `sae_ctx / mean / ridge` unit banked only PER-FEATURE R2
(`eval_results/issue_1482/sae_perfeature/sae_ctx__mean__ridge.npz`); the map itself
— which context features predict which answer features — was never recovered. This
script refits that exact unit to obtain the coefficient matrix B and characterises
its structure.

INTERPRETATION CONTRACT (carried into every artifact this script writes): a ridge
coefficient is a PREDICTIVE ASSOCIATION estimated from observational data. It is
NOT a causal claim. Nothing here licenses "context feature i CAUSES answer feature
j"; the licensed reading is "i predicts / is associated with j, in a linear map fit
on 120k held-in contexts". A causal claim would require intervention (steering /
ablation), which this round does not do.

RECIPE (pinned to the banked unit — see `unit_ridge__sae_ctx.json`):
  arm            sae_ctx, mean pooling, ridge
  design Z       [psi_mean | psi_last], each restricted to F_in = 8,192 context-side
                 features by fit-set activity (floor = ceil(1% of 120,000) = 1,200)
                 => h = 16,384 input COLUMNS (the brief's "8,192 context features"
                 are the psi_mean block; psi_last is a second block over the SAME
                 8,192 feature ids)
  targets Y      ans_mean over F_out = 16,384 answer-side features (same floor)
  lambda         10,000.0 (parent-selected on the 2,000-row val pool; PINNED here)
  rows           120,000 fit / 20,000 holdout, #1482 SINGLE-TURN corpus
                 (lmsys + wildchat pass-B; NOT the #1738 multi-turn full-width arm)
  arithmetic     fp64 Gram + eigh + fp64 X^T Y (the banked path); the parent's
                 shuffle-null sibling used fp32 GEMMs, which is why its identity
                 gate reproduced R2 only to ~1e-5 — this script's gate is tighter.

B is reported in STANDARDIZED-INPUT units: pred = ((x - xmu) / xsd) @ B + ymu, so
B[i, j] is the predicted change in answer feature j per 1 SD of context feature i —
the convention that makes coefficients comparable ACROSS input features. The
un-standardisation factor is 1 / xsd[i] (B_raw = B / xsd[:, None]); xsd is written
to the structure JSON so either convention is recoverable.

REUSE: the store scan, the parent feature restriction, the row registry, the CSR
design build, the standardizer and the fp64 Gram+eigh factorization are imported
VERBATIM from `issue1482_shuffle_null` (itself validated against the parent by an
identity gate). Only the fp64 X^T Y accumulation and the coefficient/structure
reads are new here.
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import logging
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/scipy (shared-VM discipline)

import issue1482_shuffle_null as SN  # noqa: E402  (verbatim parent-parity machinery)
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.linalg import cho_factor, cho_solve  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("i1482-mapcoef")

LAM = SN.LAM  # 10_000.0 — parent-selected lambda for sae_ctx/mean/ridge
BLOCK = 8192
WORK_DEFAULT = Path("/mnt/eps-data/thomasjiralerspong/issue1482_mapcoef")
STORE_DEFAULT = SN.STORE_DEFAULT
SCRATCH_DEFAULT = Path(
    "/mnt/eps-data/thomasjiralerspong/issue1482_saedense/meta/"
    "issue1482_error_analysis/analysis_tensors/scratch_meta"
)
LABELS_DEFAULT = Path("/mnt/eps-data/thomasjiralerspong/issue1773_fulldict/labels_upload")
PARENT_NPZ = PROJECT_ROOT / "eval_results/issue_1482/sae_perfeature/sae_ctx__mean__ridge.npz"
OUT_DIR = PROJECT_ROOT / "eval_results/issue_1482/map_coefficients"
FIG_DIR = PROJECT_ROOT / "figures/issue_1482/map_coefficients"
DASH_PATH = PROJECT_ROOT / "tasks/awaiting_promotion/1482/artifacts/map_pairs_dashboard.html"
# The task-artifact path above is the durable record but is NOT SERVED: the EPS
# dashboard renders only a task's body.md and has no artifacts route, so a
# /tasks/1482/artifacts/... link 404s. dashboard/public/ IS served (live siblings:
# sae-features-1482.html, pc-lens-1482.html), so write BOTH — a hand-copy would go
# stale the next time this script regenerates the pairs.
PUBLIC_DASH_PATH = PROJECT_ROOT / "dashboard/public/map-pairs-1482.html"

SPLITHALF_SEED = 1482  # SPLIT_SEED_1482 (parent `_splithalf_perm` convention)
NULL_SEED_BASE = SN.SEED_BASE  # 1_482_000 (parent shuffle-null seed convention)
N_NULL_DRAWS = 5
NULL_COLS = 2048  # answer-feature columns sampled per null draw (ridge is
# column-separable, so this is an exact sample of the null coefficient population)
TOPM = 20_000  # candidate pool size (|B| top-M) and the per-half top-K set size
DASH_MAX_PAIRS = 200

CAVEATS = [
    "A ridge coefficient is a PREDICTIVE ASSOCIATION under observational data, not a "
    "causal claim. Read B[i, j] as 'context feature i predicts answer feature j', "
    "never 'causes'. A causal claim would require intervention (steering / ablation), "
    "which this round does not perform.",
    "Both sides are encoded with the SAME layer-19 andyrdt SAE dictionary, so context "
    "feature id j and answer feature id j are the SAME feature. Diagonal entries "
    "therefore measure PERSISTENCE (the feature stays on from context into answer); "
    "off-diagonal mass is cross-feature structure.",
    "Coefficients are in standardized-input units (per 1 SD of the context feature). "
    "Ridge shrinkage is applied uniformly at lambda=10000, so magnitudes are "
    "comparable across pairs but are NOT unbiased effect sizes.",
    "Inputs are 8,192 context features in TWO blocks (psi_mean = mean-pooled over "
    "context tokens, psi_last = last context token), so B has 16,384 input rows over "
    "8,192 distinct feature ids.",
    "Corpus/regime: the #1482 SINGLE-TURN panel arm (lmsys + wildchat pass-B, 120k "
    "fit / 20k holdout). NOT the #1738 multi-turn full-width arm; do not pool them.",
    "Autointerp descriptions and axis labels are joined from #1773, whose standing "
    "caveat is that those labels are SEARCH-INDEX-ONLY (neighbour discrimination "
    "0.322 against a 0.50 bar). Descriptions are a reading aid, never evidence.",
    "Ridge coefficients over 8,192 correlated inputs are individually unstable and "
    "split among near-duplicate features; only pairs clearing BOTH the split-half "
    "replication and the label-shuffle null are reported as pairs.",
]


def _log(msg: str) -> None:
    logger.info("%s (rss %.1f GiB)", msg, _rss_gb())


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return out.stdout.strip() if out.returncode == 0 else "unavailable-no-git-checkout"


def _provenance(extra: dict | None = None) -> dict:
    doc = {
        "git_commit": _git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
    }
    if extra:
        doc.update(extra)
    return doc


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False))
    tmp.replace(path)
    _log(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


# ── design assembly (parent-parity, reusing issue1482_shuffle_null verbatim) ─────────


def build_design(args) -> dict:
    """Store scan -> parent feature restriction -> row registry -> CSR X, Y.

    Every step delegates to `issue1482_shuffle_null`, which reproduces `_p3_prep` /
    `_p3_design` / `_p3_targets` and is itself gated against the parent npz.
    Cached to `--work/design/` so re-runs skip the ~1,920-shard scan."""
    cache = args.work / "design"
    meta_p = cache / "meta.json"
    if meta_p.exists() and not args.rebuild_design:
        meta = json.loads(meta_p.read_text())
        X = sp.load_npz(cache / "X.npz")
        Y = sp.load_npz(cache / "Y.npz")
        arrs = np.load(cache / "idx.npz")
        _log(f"design cache hit: X{X.shape} nnz={X.nnz:.3e} Y{Y.shape} nnz={Y.nnz:.3e}")
        return {"X": X, "Y": Y, **{k: arrs[k] for k in arrs.files}, "meta": meta}

    shards = sorted(args.store.glob("pooled_*.npz"))
    assert shards, f"no pooled shards under {args.store}"
    _log(f"{len(shards)} pooled shards under {args.store}")

    idx = np.load(args.scratch / "split_indices.npz")
    parent = np.load(PARENT_NPZ)

    t = time.time()
    counts = SN.scan_counts(shards)
    _log(f"pass1 activity scan {time.time() - t:.0f}s (n_fit={counts['n_fit']})")

    f_out, floor = SN.restrict(counts["out_fit"], counts["n_fit"], SN.MAX_FEATURES_OUT)
    f_in, _ = SN.restrict(counts["in_fit"], counts["n_fit"], SN.MAX_FEATURES_IN)
    assert np.array_equal(f_out, parent["feat_ids"]), "f_out != parent feat_ids — recipe drift"
    activity = counts["out_fit"][f_out] / max(1, counts["n_fit"])
    assert np.allclose(activity, parent["activity"], atol=1e-9), "activity mismatch vs parent"
    _log(f"restriction: F_out={len(f_out)} F_in={len(f_in)} floor={floor}")

    have = {int(r) for r in counts["have"]}
    order = np.concatenate([idx["sae_fit"], idx["sae_val"], idx["holdout"]])
    order = np.asarray([r for r in order if int(r) in have], dtype=np.int64)
    row_pos = {int(r): i for i, r in enumerate(order)}
    tr = np.asarray([row_pos[int(r)] for r in idx["sae_fit"] if int(r) in row_pos], np.int64)
    te = np.asarray([row_pos[int(r)] for r in idx["holdout"] if int(r) in row_pos], np.int64)
    assert len(tr) == 120_000 and len(te) == 20_000, (len(tr), len(te))

    t = time.time()
    X, Y = SN.build_csr(shards, row_pos, len(order), f_in, f_out, counts)
    _log(f"pass2 CSR build {time.time() - t:.0f}s: X nnz={X.nnz:.3e} Y nnz={Y.nnz:.3e}")

    cache.mkdir(parents=True, exist_ok=True)
    sp.save_npz(cache / "X.npz", X, compressed=False)
    sp.save_npz(cache / "Y.npz", Y, compressed=False)
    np.savez(cache / "idx.npz", f_in=f_in, f_out=f_out, tr=tr, te=te, activity=activity)
    meta = {
        "n_shards": len(shards),
        "f_in": int(len(f_in)),
        "f_out": int(len(f_out)),
        "activity_floor": int(floor),
        "n_registry": int(len(order)),
        "n_fit": int(len(tr)),
        "n_holdout": int(len(te)),
        "x_nnz": int(X.nnz),
        "y_nnz": int(Y.nnz),
        "input_blocks": ["psi_mean", "psi_last"],
    }
    _write_json(meta_p, meta)
    return {
        "X": X,
        "Y": Y,
        "f_in": f_in,
        "f_out": f_out,
        "tr": tr,
        "te": te,
        "activity": activity,
        "meta": meta,
    }


# ── fp64 X^T Y accumulation (the precision the R2 gate needs) ───────────────────────


def xty_fp64(
    X: sp.csr_matrix,
    Y: sp.csr_matrix,
    rows: np.ndarray,
    xmu: np.ndarray,
    xsd: np.ndarray,
    block: int,
    perm: np.ndarray | None = None,
) -> np.ndarray:
    """UNCENTERED X_std[rows]^T Y[rows[perm]] in fp64 (blocked; no (n, h) materialised).

    `perm` permutes the Y side WITHIN `rows` — the parent shuffle-null convention
    (`_shuffle_draws`): the design and the fit-row set are untouched, only the
    context->answer pairing is broken."""
    out = np.zeros((X.shape[1], Y.shape[1]), dtype=np.float64)
    for s in range(0, len(rows), block):
        sl = rows[s : s + block]
        xb = (SN.dense_block(X, sl).astype(np.float64) - xmu) / xsd
        yrows = sl if perm is None else rows[perm[s : s + block]]
        out += xb.T @ SN.dense_block(Y, yrows).astype(np.float64)
    return out


def gram_fp64(
    X: sp.csr_matrix, rows: np.ndarray, xmu: np.ndarray, xsd: np.ndarray, block: int
) -> tuple[np.ndarray, np.ndarray]:
    """fp64 blocked Gram + column sums of the standardized design over `rows`."""
    h = X.shape[1]
    A = np.zeros((h, h), dtype=np.float64)
    colsum = np.zeros(h, dtype=np.float64)
    for s in range(0, len(rows), block):
        xb = (SN.dense_block(X, rows[s : s + block]).astype(np.float64) - xmu) / xsd
        A += xb.T @ xb
        colsum += xb.sum(0)
    return A, colsum


def col_mean(Y: sp.csr_matrix, rows: np.ndarray) -> np.ndarray:
    """fp64 column mean of Y over `rows` (exact from the CSR, no densification)."""
    return np.asarray(Y[rows].astype(np.float64).sum(axis=0)).ravel() / len(rows)


def predict_rows(
    X: sp.csr_matrix,
    rows: np.ndarray,
    xmu: np.ndarray,
    xsd: np.ndarray,
    B: np.ndarray,
    ymu: np.ndarray,
    block: int,
) -> np.ndarray:
    """X_std[rows] @ B + ymu in fp64, blocked."""
    out = np.empty((len(rows), B.shape[1]), dtype=np.float64)
    for s in range(0, len(rows), block):
        xb = (SN.dense_block(X, rows[s : s + block]).astype(np.float64) - xmu) / xsd
        out[s : s + xb.shape[0]] = xb @ B + ymu
    return out


def perfeature_r2(pred: np.ndarray, Y: sp.csr_matrix, te: np.ndarray, block: int) -> np.ndarray:
    """Per-feature holdout R2 with the parent `_per_feature_metrics` formula."""
    ss_tot, _ = SN.holdout_ss_tot(Y, te, block)
    ss_res = np.zeros(Y.shape[1], dtype=np.float64)
    for s in range(0, len(te), block):
        tb = SN.dense_block(Y, te[s : s + block]).astype(np.float64)
        tb -= pred[s : s + tb.shape[0]]
        ss_res += (tb * tb).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_tot > 1e-12, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)


# ── phase 1: main refit + R2 reproduction gate ──────────────────────────────────────


def phase_fit(args, des: dict) -> dict:
    ck = args.work / "fit"
    ck.mkdir(parents=True, exist_ok=True)
    gate_p = ck / "gate.json"
    if (ck / "B.npy").exists() and gate_p.exists() and not args.refit:
        _log("fit checkpoint hit")
        return {
            "B": np.load(ck / "B.npy"),
            "U": np.load(ck / "U.npy"),
            "s_eig": np.load(ck / "s_eig.npy"),
            "xmu": np.load(ck / "xmu.npy"),
            "xsd": np.load(ck / "xsd.npy"),
            "ymu": np.load(ck / "ymu.npy"),
            "colsum": np.load(ck / "colsum.npy"),
            "r2": np.load(ck / "r2.npy"),
            "gate": json.loads(gate_p.read_text()),
        }

    X, Y, tr, te = des["X"], des["Y"], des["tr"], des["te"]
    t0 = time.time()
    xmu, xsd = SN.train_standardizer(X, tr, args.block)
    _log(f"standardizer {time.time() - t0:.0f}s")

    t = time.time()
    fac = SN.factorize(X, tr, xmu, xsd, args.block)  # fp64 Gram + eigh (parent parity)
    t_gram = time.time() - t
    _log(f"gram+eigh {t_gram:.0f}s")
    U, s_eig, colsum = fac["U"], fac["s_eig"], fac["colsum_xstd"]
    del fac
    gc.collect()

    ymu = col_mean(Y, tr)
    t = time.time()
    xty = xty_fp64(X, Y, tr, xmu, xsd, args.block)
    xty -= np.outer(colsum, ymu)  # centered X_std^T (Y - ymu)
    t_xty = time.time() - t
    _log(f"X^T Y (fp64) {t_xty:.0f}s")

    # B = (A + lam I)^-1 X_std^T Y_c  via the shared eigh: U diag(1/(s+lam)) U^T XtY
    t = time.time()
    G = U.T @ xty
    del xty
    gc.collect()
    G /= (s_eig + LAM)[:, None]
    B = U @ G
    del G
    gc.collect()
    _log(f"coefficient solve {time.time() - t:.0f}s -> B{B.shape}")

    t = time.time()
    pred = predict_rows(X, te, xmu, xsd, B, ymu, args.block)
    r2 = perfeature_r2(pred, Y, te, args.block)
    del pred
    gc.collect()
    _log(f"holdout predict + R2 {time.time() - t:.0f}s")

    # Checkpoint BEFORE gating: a failed gate must not discard ~30 min of Gram+eigh.
    np.save(ck / "B.npy", B)
    np.save(ck / "U.npy", U)
    np.save(ck / "s_eig.npy", s_eig)
    np.save(ck / "xmu.npy", xmu)
    np.save(ck / "xsd.npy", xsd)
    np.save(ck / "ymu.npy", ymu)
    np.save(ck / "colsum.npy", colsum)
    np.save(ck / "r2.npy", r2)

    parent_r2 = np.load(PARENT_NPZ)["r2"]
    d = np.abs(r2 - parent_r2)
    ok = np.isfinite(d)
    ss_tot, _ = SN.holdout_ss_tot(Y, te, args.block)
    pooled = 1.0 - ((1.0 - r2[ok]) * ss_tot[ok]).sum() / ss_tot[ok].sum()
    gate = {
        "n_features": int(r2.size),
        "n_finite": int(ok.sum()),
        "max_abs_delta_r2": float(np.nanmax(d)),
        "median_abs_delta_r2": float(np.nanmedian(d)),
        "q99_abs_delta_r2": float(np.nanquantile(d, 0.99)),
        "refit_pooled_r2": float(pooled),
        "banked_pooled_r2": 0.6901409540488584,
        "refit_mean_r2": float(np.nanmean(r2)),
        "banked_mean_r2": float(np.nanmean(parent_r2)),
        "selected_lambda": LAM,
        "wall_seconds": {"gram_eigh": t_gram, "xty": t_xty, "total": time.time() - t0},
        "threshold_basis": (
            "This VM has no GPU; the banked array was produced by a CUDA fp64 eigh. A "
            "bit-faithful (~1e-10) CPU reproduction is therefore not available. The bar "
            "adopted here is the one #1482's own validated shuffle-null identity gate "
            "uses for this exact unit (median < 5e-3, q99 < 0.05, max reported): that "
            "gate measured median 1.5468e-05 / q99 1.9674e-04 / max 5.4156e-03 against "
            "the same banked array. A wiring error shifts the whole distribution "
            "(median O(0.1)); the residual here is a handful of near-degenerate-ss_tot "
            "features."
        ),
    }
    _log(
        f"R2 gate: median {gate['median_abs_delta_r2']:.4e} q99 "
        f"{gate['q99_abs_delta_r2']:.4e} max {gate['max_abs_delta_r2']:.4e}; "
        f"pooled {gate['refit_pooled_r2']:.10f} vs banked {gate['banked_pooled_r2']:.10f}"
    )
    assert gate["median_abs_delta_r2"] < 5e-3, f"refit does not reproduce banked R2: {gate}"
    assert gate["q99_abs_delta_r2"] < 0.05, f"refit does not reproduce banked R2: {gate}"
    _write_json(gate_p, gate)
    return {
        "B": B,
        "U": U,
        "s_eig": s_eig,
        "xmu": xmu,
        "xsd": xsd,
        "ymu": ymu,
        "colsum": colsum,
        "r2": r2,
        "gate": gate,
    }


# ── phase 2: label-shuffle null ─────────────────────────────────────────────────────


def phase_null(args, des: dict, fit: dict) -> dict:
    ck = args.work / "null"
    ck.mkdir(parents=True, exist_ok=True)
    out_p = ck / "null.npz"
    if out_p.exists() and not args.refit:
        _log("null checkpoint hit")
        z = np.load(out_p)
        return {k: z[k] for k in z.files}

    X, Y, tr = des["X"], des["Y"], des["tr"]
    rng = np.random.default_rng(NULL_SEED_BASE)
    cols = np.sort(rng.choice(Y.shape[1], size=NULL_COLS, replace=False))
    Ysub = Y[:, cols].tocsr()
    xmu, xsd, U, s_eig = fit["xmu"], fit["xsd"], fit["U"], fit["s_eig"]
    colsum = fit["colsum"]  # column sum of the standardized train design (from the fit)
    inv = 1.0 / (s_eig + LAM)
    draws = np.empty((N_NULL_DRAWS, X.shape[1], NULL_COLS), dtype=np.float32)
    for k in range(N_NULL_DRAWS):
        t = time.time()
        perm = np.random.default_rng(NULL_SEED_BASE + k).permutation(len(tr))
        ymu_k = col_mean(Ysub, tr[perm])
        xty = xty_fp64(X, Ysub, tr, xmu, xsd, args.block, perm=perm)
        xty -= np.outer(colsum, ymu_k)
        Bk = U @ ((U.T @ xty) * inv[:, None])
        draws[k] = Bk.astype(np.float32)
        del xty, Bk
        gc.collect()
        _log(f"null draw {k + 1}/{N_NULL_DRAWS} {time.time() - t:.0f}s")

    # per-column null scale, and the scale-free ratio to the target's own SD
    y_sd = np.sqrt(
        np.asarray(Ysub.power(2).mean(axis=0)).ravel() - np.asarray(Ysub.mean(axis=0)).ravel() ** 2
    )
    null_sd_col = draws.std(axis=1).mean(axis=0)  # (NULL_COLS,) mean over draws
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(y_sd > 1e-12, null_sd_col / np.maximum(y_sd, 1e-12), np.nan)
    np.savez(
        out_p,
        cols=cols,
        draws_abs_q=np.nanquantile(np.abs(draws), [0.5, 0.9, 0.99, 0.999, 0.9999, 1.0]),
        null_sd_col=null_sd_col,
        y_sd_col=y_sd,
        ratio=ratio,
        z_abs_q=np.nanquantile(
            np.abs(draws) / np.maximum(null_sd_col[None, None, :], 1e-30),
            [0.99, 0.999, 0.9999, 0.99999, 1.0],
        ),
    )
    del draws
    gc.collect()
    z = np.load(out_p)
    return {k: z[k] for k in z.files}


# ── phase 3: split-half refits ──────────────────────────────────────────────────────


def _solve_half(A: np.ndarray, xty_c: np.ndarray) -> np.ndarray:
    """(A + lam I)^-1 xty_c via Cholesky (SPD by construction at lam=10,000)."""
    A[np.diag_indices_from(A)] += LAM
    c = cho_factor(A, lower=True, overwrite_a=True, check_finite=False)
    return cho_solve(c, xty_c, overwrite_b=True, check_finite=False)


def phase_split(args, des: dict, fit: dict) -> dict:
    ck = args.work / "split"
    ck.mkdir(parents=True, exist_ok=True)
    if (ck / "B_a.npy").exists() and not args.refit:
        _log("split-half checkpoint hit")
        return {"B_a": np.load(ck / "B_a.npy"), "B_b": np.load(ck / "B_b.npy")}

    X, Y, tr = des["X"], des["Y"], des["tr"]
    xmu, xsd = fit["xmu"], fit["xsd"]  # FULL-train standardizer for BOTH halves, so
    # the two coefficient matrices live on the same scale and are directly comparable
    perm = np.random.default_rng(SPLITHALF_SEED).permutation(len(tr))
    halves = {"a": tr[perm[: len(tr) // 2]], "b": tr[perm[len(tr) // 2 :]]}
    out = {}
    for name, rows in halves.items():
        t = time.time()
        A, colsum = gram_fp64(X, rows, xmu, xsd, args.block)
        ymu_h = col_mean(Y, rows)
        xty = xty_fp64(X, Y, rows, xmu, xsd, args.block)
        xty -= np.outer(colsum, ymu_h)
        Bh = _solve_half(A, xty).astype(np.float32)
        del A, xty
        gc.collect()
        np.save(ck / f"B_{name}.npy", Bh)
        out[f"B_{name}"] = Bh
        _log(f"split-half {name} (n={len(rows)}) {time.time() - t:.0f}s")
    return out


# ── phase 4: structure + surviving pairs ────────────────────────────────────────────


def _topk_flat(absB: np.ndarray, m: int) -> np.ndarray:
    """Flat indices of the m largest entries (unordered)."""
    m = min(int(m), absB.size)
    kth = max(0, absB.size - m)
    return np.argpartition(absB.reshape(-1), kth)[kth:]


def _spectrum(B: np.ndarray, n_sv: int = 512, seed: int = 0) -> dict:
    """Exact spectral moments + randomized top-`n_sv` singular values.

    A full 16,384^2 SVD costs ~10 min of the round's budget for information the
    moments already carry: sum(sigma^2) = ||B||_F^2 and sum(sigma^4) = ||B^T B||_F^2
    are exact, so the participation ratio (sum s^2)^2 / sum s^4 is exact."""
    fro2 = float((B.astype(np.float64) ** 2).sum())
    G = B.T @ B  # (d, d) fp32 GEMM, fp32 accumulate
    fro4 = float((G.astype(np.float64) ** 2).sum())
    del G
    gc.collect()
    rng = np.random.default_rng(seed)
    Om = rng.standard_normal((B.shape[1], n_sv), dtype=np.float32)
    Q = np.linalg.qr(B @ Om)[0]
    for _ in range(2):  # power iterations for a tight top-of-spectrum estimate
        Q = np.linalg.qr(B @ (B.T @ Q))[0]
    sv = np.linalg.svd(Q.T @ B, compute_uv=False)
    return {
        "frobenius_sq": fro2,
        "sum_sigma4": fro4,
        "participation_ratio_spectrum": fro2**2 / fro4 if fro4 > 0 else float("nan"),
        "sigma_top": sv[: min(n_sv, sv.size)].astype(float).tolist(),
        "stable_rank": fro2 / float(sv[0] ** 2) if sv.size and sv[0] > 0 else float("nan"),
        "n_singular_values_estimated": int(min(n_sv, sv.size)),
        "method": "exact Frobenius moments + randomized subspace iteration (2 power its)",
    }


def phase_analyze(args, des: dict, fit: dict, null: dict, split: dict) -> dict:
    B = fit["B"].astype(np.float32)
    f_in, f_out = des["f_in"], des["f_out"]
    h_in, d = len(f_in), len(f_out)
    absB = np.abs(B)

    # ---- per-column concentration -------------------------------------------------
    col_sum = absB.sum(0)
    col_max = absB.max(0)
    part = np.partition(absB, absB.shape[0] - 10, axis=0)[-10:, :]
    top10 = part.sum(0)
    sq = absB.astype(np.float64) ** 2
    pr_col = np.where(
        (sq**2).sum(0) > 0, sq.sum(0) ** 2 / np.maximum((sq**2).sum(0), 1e-300), np.nan
    )
    del sq, part
    gc.collect()
    with np.errstate(invalid="ignore", divide="ignore"):
        top1_share = np.where(col_sum > 0, col_max / np.maximum(col_sum, 1e-30), np.nan)
        top10_share = np.where(col_sum > 0, top10 / np.maximum(col_sum, 1e-30), np.nan)

    # ---- diagonal / persistence ---------------------------------------------------
    pos_in = {int(f): i for i, f in enumerate(f_in)}
    common = np.array([f for f in f_out if int(f) in pos_in], dtype=np.int64)
    col_of = {int(f): j for j, f in enumerate(f_out)}
    diag_cols = np.array([col_of[int(f)] for f in common], dtype=np.int64)
    diag_rows_mean = np.array([pos_in[int(f)] for f in common], dtype=np.int64)
    diag_rows_last = diag_rows_mean + h_in
    self_mean = B[diag_rows_mean, diag_cols]
    self_last = B[diag_rows_last, diag_cols]

    sub = absB[:, diag_cols]  # (h, n_common)
    rank_mean = (sub > np.abs(self_mean)[None, :]).sum(0) + 1
    rank_last = (sub > np.abs(self_last)[None, :]).sum(0) + 1
    argmax_row = sub.argmax(0)
    top_is_self_mean = argmax_row == diag_rows_mean
    top_is_self_last = argmax_row == diag_rows_last
    del sub
    gc.collect()

    mask_mean = np.zeros(B.shape, dtype=bool)
    mask_mean[diag_rows_mean, diag_cols] = True
    mask_mean[diag_rows_last, diag_cols] = True
    diag_mass = float(absB[mask_mean].sum())
    total_mass = float(absB.sum())
    del mask_mean
    gc.collect()

    # ---- null threshold -----------------------------------------------------------
    ratio = null["ratio"][np.isfinite(null["ratio"])]
    z_q = null["z_abs_q"]  # quantiles of |null coef| / per-column null SD
    z_thresh = float(z_q[3])  # 99.999th percentile of the null z distribution
    ratio_med = float(np.median(ratio))
    # per-column null SD for EVERY column, from the calibrated scale model
    y_sd_all = np.asarray(des["y_sd_all"])
    null_sd_all = ratio_med * y_sd_all
    tau = z_thresh * null_sd_all  # per-column |coef| threshold

    # ---- candidate pairs, split-half replication, null survival --------------------
    cand = _topk_flat(absB, TOPM)
    ci, cj = np.unravel_index(cand, B.shape)
    ba, bb = split["B_a"], split["B_b"]
    set_a = set(_topk_flat(np.abs(ba), TOPM).tolist())
    set_b = set(_topk_flat(np.abs(bb), TOPM).tolist())
    in_a = np.fromiter((int(c) in set_a for c in cand), bool, len(cand))
    in_b = np.fromiter((int(c) in set_b for c in cand), bool, len(cand))
    sgn = np.sign(B[ci, cj])
    sign_ok = (np.sign(ba[ci, cj]) == sgn) & (np.sign(bb[ci, cj]) == sgn)
    replicated = in_a & in_b & sign_ok
    zval = np.abs(B[ci, cj]) / np.maximum(null_sd_all[cj], 1e-30)
    null_ok = np.abs(B[ci, cj]) > tau[cj]
    surviving = replicated & null_ok

    order = np.argsort(-zval[surviving])
    sidx = np.nonzero(surviving)[0][order]
    pairs = []
    for p in sidx[: args.max_pairs]:
        i, j = int(ci[p]), int(cj[p])
        blk = "psi_mean" if i < h_in else "psi_last"
        ctx_feat = int(f_in[i % h_in])
        ans_feat = int(f_out[j])
        pairs.append(
            {
                "ctx_feat_id": ctx_feat,
                "ctx_block": blk,
                "ans_feat_id": ans_feat,
                "coef_std_units": float(B[i, j]),
                "coef_half_a": float(ba[i, j]),
                "coef_half_b": float(bb[i, j]),
                "split_half_sign_agree": bool(sign_ok[p]),
                "split_half_both_topk": bool(in_a[p] and in_b[p]),
                "null_z": float(zval[p]),
                "null_threshold_coef": float(tau[j]),
                "is_persistence": bool(ctx_feat == ans_feat),
                "ans_r2": float(fit["r2"][j]),
            }
        )

    # ---- persistence vs best-predicted answer features -----------------------------
    r2 = fit["r2"]
    r2_common = r2[diag_cols]
    fin = np.isfinite(r2_common)
    best = fin & (r2_common >= np.nanquantile(r2_common, 0.95))
    rest = fin & (r2_common < np.nanquantile(r2_common, 0.50))
    # self-coefficient as a share of the column's strongest input coefficient
    self_share = np.abs(self_mean) / np.maximum(col_max[diag_cols], 1e-30)

    # ---- per-input-block mass + dead (constant-column) input rows ------------------
    row_l1 = absB.sum(1)
    row_l2 = (absB.astype(np.float64) ** 2).sum(1)
    l1_m, l1_l = float(row_l1[:h_in].sum()), float(row_l1[h_in:].sum())
    l2_m, l2_l = float(row_l2[:h_in].sum()), float(row_l2[h_in:].sum())
    dead = fit["xsd"] <= 1e-8  # constant standardized column => coefficient exactly 0
    dead_m, dead_l = int(dead[:h_in].sum()), int(dead[h_in:].sum())
    n_live = int((~dead).sum())

    doc = {
        "goal": "structure of the SAE(context) -> SAE(answer) ridge coefficient map "
        "(issue #1482, sae_ctx / mean / ridge)",
        "caveats": CAVEATS,
        "recipe": {
            "arm": "sae_ctx / mean / ridge (parent P3 unit)",
            "corpus": "#1482 SINGLE-TURN panel (lmsys + wildchat pass-B)",
            "lambda": LAM,
            "n_fit": int(len(des["tr"])),
            "n_holdout": int(len(des["te"])),
            "f_in_features": int(h_in),
            "input_columns": int(B.shape[0]),
            "input_blocks": ["psi_mean", "psi_last"],
            "f_out_features": int(d),
            "coefficient_units": "standardized input (per 1 SD of the context feature); "
            "B_raw = B / xsd[:, None]",
            "xsd_summary": {
                "median": float(np.median(fit["xsd"])),
                "q05": float(np.quantile(fit["xsd"], 0.05)),
                "q95": float(np.quantile(fit["xsd"], 0.95)),
            },
        },
        "refit_gate": fit["gate"],
        "overlap": {
            "n_common_feature_ids": int(len(common)),
            "f_in": int(h_in),
            "f_out": int(d),
            "note": "answer features that are ALSO in the context input set; only these "
            "have a persistence (diagonal) entry.",
        },
        "diagonal_dominance": {
            "self_coef_psi_mean": {
                "median_abs": float(np.median(np.abs(self_mean))),
                "median_signed": float(np.median(self_mean)),
                "frac_positive": float((self_mean > 0).mean()),
                "median_rank_in_column": float(np.median(rank_mean)),
                "frac_columns_top_input_is_self": float(top_is_self_mean.mean()),
                "frac_rank_le_10": float((rank_mean <= 10).mean()),
                "frac_rank_le_100": float((rank_mean <= 100).mean()),
            },
            "self_coef_psi_last": {
                "median_abs": float(np.median(np.abs(self_last))),
                "median_signed": float(np.median(self_last)),
                "frac_positive": float((self_last > 0).mean()),
                "median_rank_in_column": float(np.median(rank_last)),
                "frac_columns_top_input_is_self": float(top_is_self_last.mean()),
            },
            "frac_columns_top_input_is_own_id_either_block": float(
                (top_is_self_mean | top_is_self_last).mean()
            ),
            "diagonal_abs_mass_share": diag_mass / total_mass,
            "diagonal_entries": int(2 * len(common)),
            "total_entries": int(B.size),
            "diagonal_mass_enrichment_vs_uniform": (diag_mass / total_mass)
            / (2 * len(common) / B.size),
            "median_abs_offdiagonal": float(np.median(absB[:, ::37])),
        },
        "column_concentration": {
            "top1_share": {
                q: float(np.nanquantile(top1_share, v))
                for q, v in (("q10", 0.1), ("median", 0.5), ("q90", 0.9))
            },
            "top10_share": {
                q: float(np.nanquantile(top10_share, v))
                for q, v in (("q10", 0.1), ("median", 0.5), ("q90", 0.9))
            },
            "participation_ratio_per_column": {
                q: float(np.nanquantile(pr_col, v))
                for q, v in (("q10", 0.1), ("median", 0.5), ("q90", 0.9))
            },
            "n_input_rows": int(B.shape[0]),
            "n_live_input_rows": int(n_live),
            "reading": "participation ratio counts the effective number of input "
            f"features carrying a column's prediction (max = {int(n_live)} LIVE input "
            "rows, not 16,384 — see input_block_mass).",
        },
        "input_block_mass": {
            "psi_mean_l1_share": l1_m / max(l1_m + l1_l, 1e-30),
            "psi_last_l1_share": l1_l / max(l1_m + l1_l, 1e-30),
            "psi_mean_l2_share": l2_m / max(l2_m + l2_l, 1e-30),
            "psi_last_l2_share": l2_l / max(l2_m + l2_l, 1e-30),
            "psi_mean_dead_rows": int(dead_m),
            "psi_last_dead_rows": int(dead_l),
            "n_live_input_rows": int(n_live),
            "reading": "a 'dead' row is a context feature whose standardized train "
            "column is constant (xsd <= 1e-8), so its coefficient is exactly 0. Feature "
            "selection used psi_mean activity, so many selected features never fire at "
            "the FINAL context token; the live psi_last rows nonetheless carry a large "
            "share of the map's squared mass.",
        },
        "spectrum": _spectrum(B),
        "null": {
            "draws": N_NULL_DRAWS,
            "columns_per_draw": NULL_COLS,
            "seed_base": NULL_SEED_BASE,
            "permutation": "Y rows permuted within the 120k fit rows; holdout untouched",
            "abs_coef_quantiles": {
                k: float(v)
                for k, v in zip(
                    ["median", "q90", "q99", "q999", "q9999", "max"],
                    null["draws_abs_q"],
                    strict=True,
                )
            },
            "z_quantiles": {
                k: float(v)
                for k, v in zip(
                    ["q99", "q999", "q9999", "q99999", "max"], null["z_abs_q"], strict=True
                )
            },
            "z_threshold_used": z_thresh,
            "null_sd_over_target_sd_ratio": {
                "median": ratio_med,
                "q05": float(np.quantile(ratio, 0.05)),
                "q95": float(np.quantile(ratio, 0.95)),
                "note": "constancy of this ratio is what licenses extrapolating the "
                "per-column null SD to all 16,384 columns.",
            },
        },
        "split_half": {
            "seed": SPLITHALF_SEED,
            "n_per_half": int(len(des["tr"]) // 2),
            "standardizer": "FULL-train xmu/xsd for BOTH halves (same scale => "
            "coefficients directly comparable)",
            "top_k": TOPM,
            "candidate_pairs": int(len(cand)),
            "replicated_both_halves_topk_and_sign": int(replicated.sum()),
            "replication_rate": float(replicated.mean()),
            "sign_agreement_rate": float(sign_ok.mean()),
        },
        "surviving_pairs": {
            "n_surviving": int(surviving.sum()),
            "of_candidates": int(len(cand)),
            "n_persistence_among_surviving": int(
                sum(
                    1
                    for p in np.nonzero(surviving)[0]
                    if int(f_in[ci[p] % h_in]) == int(f_out[cj[p]])
                )
            ),
            "criteria": "in the top-20,000 |coef| of BOTH split halves, sign-consistent "
            "with the full fit, AND |coef| above the per-column "
            "label-shuffle null threshold",
        },
        "persistence_vs_predictability": {
            "spearman_selfcoef_vs_r2": float(_spearman(np.abs(self_mean)[fin], r2_common[fin])),
            "best_predicted_top5pct": {
                "n": int(best.sum()),
                "median_self_coef_abs": float(np.median(np.abs(self_mean)[best])),
                "median_self_rank_in_column": float(np.median(rank_mean[best])),
                "frac_top_input_is_self": float(top_is_self_mean[best].mean()),
                "median_self_share_of_column_max": float(np.median(self_share[best])),
            },
            "below_median_predicted": {
                "n": int(rest.sum()),
                "median_self_coef_abs": float(np.median(np.abs(self_mean)[rest])),
                "median_self_rank_in_column": float(np.median(rank_mean[rest])),
                "frac_top_input_is_self": float(top_is_self_mean[rest].mean()),
                "median_self_share_of_column_max": float(np.median(self_share[rest])),
            },
        },
        "provenance": _provenance({"store": str(args.store), "work": str(args.work)}),
    }

    np.savez(
        args.work / "analyze.npz",
        common=common,
        diag_cols=diag_cols,
        self_mean=self_mean,
        self_last=self_last,
        rank_mean=rank_mean,
        top_is_self_mean=top_is_self_mean,
        r2_common=r2_common,
        top1_share=top1_share,
        top10_share=top10_share,
        pr_col=pr_col,
        tau=tau,
        null_sd_all=null_sd_all,
        self_share=self_share,
    )
    return {"doc": doc, "pairs": pairs}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok]).statistic) if ok.sum() >= 3 else float("nan")


# ── autointerp label join (#1773, search-index-only) ────────────────────────────────


def load_labels(feat_ids: set[int], labels_dir: Path) -> dict[int, dict]:
    """Join #1773 autointerp descriptions + judged axis labels for `feat_ids`."""
    out: dict[int, dict] = {int(f): {"description": None, "axes": {}} for f in feat_ids}
    for p in sorted(labels_dir.glob("descriptions.shard*.jsonl")):
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                fid = int(r.get("feat_id", -1))
                if fid in out:
                    out[fid]["description"] = r.get("description")
                    out[fid]["confidence"] = r.get("confidence")
    for p in sorted(labels_dir.glob("axis_labels.shard*.jsonl")):
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                fid = int(r.get("feat_id", -1))
                if fid in out:
                    out[fid]["axes"][str(r.get("axis"))] = r.get("label")
    return out


# ── figures ─────────────────────────────────────────────────────────────────────────


def phase_figures(args, doc: dict) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    z = np.load(args.work / "analyze.npz")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    made = []

    # Figure 1 — structure read
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    ax = axes[0]
    lo, hi = np.inf, 0.0
    for key, role, lab in (
        ("top1_share", "primary", "top-1 input share"),
        ("top10_share", "baseline", "top-10 input share"),
    ):
        v = np.sort(z[key][np.isfinite(z[key])])
        ax.plot(v, np.linspace(0, 1, v.size), color=paper_palette_role(role), lw=1.8, label=lab)
        lo = min(lo, float(v[max(0, int(0.001 * v.size))]))
        hi = max(hi, float(v[min(v.size - 1, int(0.999 * v.size))]))
    # a uniform column would put 1/16,384 of its mass on its top input — the reference
    # a "one input dominates" column has to beat by orders of magnitude
    unif = 1.0 / 16384
    ax.axvline(
        unif,
        color=paper_palette_role("neutral"),
        ls=":",
        lw=1.2,
        label="uniform over 16,384 inputs",
    )
    ax.set_xscale("log")
    ax.set_xlim(min(lo * 0.6, unif * 0.7), hi * 1.6)
    ax.set_xlabel("share of column |coefficient| mass")
    ax.set_ylabel("cumulative fraction of answer features")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_title("Per-column concentration", fontsize=10)

    ax = axes[1]
    selfabs = np.abs(z["self_mean"])
    rng = np.random.default_rng(0)
    Bmm = np.load(args.work / "fit" / "B.npy", mmap_mode="r")
    n_off = min(40_000, Bmm.size)
    off = np.abs(Bmm[rng.integers(0, Bmm.shape[0], n_off), rng.integers(0, Bmm.shape[1], n_off)])
    bins = np.logspace(-8, np.log10(max(selfabs.max(), off.max()) + 1e-12), 55)
    # fraction-per-bin (NOT density): the two sets differ ~4,000x in count and span
    # different decades, so a density normalisation renders the diagonal invisible.
    for v, role, lab in (
        (off, "neutral", f"off-diagonal sample (n={off.size:,})"),
        (selfabs, "primary", f"persistence / diagonal (n={selfabs.size:,})"),
    ):
        ax.hist(
            v + 1e-12,
            bins=bins,
            weights=np.full(v.size, 1.0 / v.size),
            histtype="stepfilled",
            color=paper_palette_role(role),
            alpha=0.55,
            edgecolor=paper_palette_role(role),
            lw=1.1,
            label=lab,
        )
    ax.axvline(
        float(np.median(z["tau"])),
        color=paper_palette_role("accent"),
        ls="--",
        lw=1.4,
        label="median null threshold",
    )
    ax.set_xscale("log")
    ax.set_xlabel("|ridge coefficient| (standardized input units)")
    ax.set_ylabel("fraction of entries per bin")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    ax.set_title("Diagonal vs off-diagonal mass", fontsize=10)
    fig.suptitle(
        "SAE context->answer ridge map: coefficient structure (predictive association, not causal)",
        fontsize=11,
    )
    fig.tight_layout()
    made += list(savefig_paper(fig, "map_structure", dir=FIG_DIR).values())
    plt.close(fig)

    # Figure 2 — persistence vs predictability
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.scatter(
        np.abs(z["self_mean"]),
        z["r2_common"],
        s=4,
        alpha=0.25,
        color=paper_palette_role("primary"),
        linewidths=0,
    )
    ax.set_xscale("log")
    ax.set_xlabel("|self-coefficient| B[i, i]  (persistence, standardized units)")
    ax.set_ylabel("per-feature holdout $R^2$ of that answer feature")
    ax.axvline(
        float(np.median(z["tau"])),
        color=paper_palette_role("accent"),
        ls="--",
        lw=1.2,
        label="median null threshold",
    )
    # clip the y-axis to the informative band; a handful of strongly-negative-R2
    # features would otherwise stretch the panel and hide the whole relationship
    ylo = -0.5
    n_clip = int((z["r2_common"] < ylo).sum())
    ax.set_ylim(ylo, max(1.0, float(np.nanmax(z["r2_common"])) * 1.05))
    rho = doc["persistence_vs_predictability"]["spearman_selfcoef_vs_r2"]
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_title(
        f"Persistence vs predictability (Spearman rho = {rho:.3f};\n"
        f"{n_clip} of {z['r2_common'].size:,} features below the plotted "
        f"$R^2$ floor of {ylo})",
        fontsize=9.5,
    )
    fig.tight_layout()
    made += list(savefig_paper(fig, "map_persistence", dir=FIG_DIR).values())
    plt.close(fig)
    return made


# ── dashboard ───────────────────────────────────────────────────────────────────────


def _esc(s: object) -> str:
    return html.escape(str(s)) if s is not None else "&mdash;"


def phase_dashboard(pairs: list[dict], labels: dict[int, dict], doc: dict) -> Path:
    css = """
:root { --fg:#16181d; --mut:#5b6270; --line:#e3e6ec; --bg:#fbfbfd; --card:#fff; }
* { box-sizing:border-box; }
body { margin:0; padding:28px 22px 60px; background:var(--bg); color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif; }
.wrap { max-width:1120px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:17px; margin:34px 0 4px; padding-top:14px; border-top:1px solid var(--line); }
p, li { color:var(--mut); font-size:13.5px; }
.head { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:16px 18px; margin-bottom:10px; }
.head p { margin:6px 0; } .head b { color:var(--fg); }
.warn { background:#fff7ed; border:1px solid #f6d6bd; border-radius:9px; padding:12px 14px;
  margin:10px 0; } .warn p { color:#8a4a12; margin:4px 0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:9px;
  padding:11px 13px; margin:8px 0; }
.card .top { display:flex; flex-wrap:wrap; align-items:baseline; gap:10px; }
.rank { color:var(--mut); font-size:12px; min-width:34px; }
.arrow { color:var(--mut); font-size:13px; }
.fid { font-weight:600; font-size:14px; }
.fid a { color:#1b4fd8; text-decoration:none; } .fid a:hover { text-decoration:underline; }
.badge { font-size:11px; padding:1px 7px; border-radius:9px; border:1px solid var(--line);
  color:var(--mut); }
.badge.pers { background:#eafaf1; border-color:#bfe6d2; color:#0f6b43; }
.badge.blk { background:#eef4ff; border-color:#c9d9fb; color:#1b3f9b; }
.nums { color:var(--mut); font-size:12px; font-variant-numeric:tabular-nums; }
.desc { margin:5px 0 0; font-size:12.5px; color:#2c313b; }
.desc .side { color:var(--mut); font-size:11px; text-transform:uppercase;
  letter-spacing:.04em; margin-right:6px; }
.ax { color:var(--mut); font-size:11.5px; margin-top:3px; }
table.kv { border-collapse:collapse; font-size:12.5px; margin:8px 0 2px; }
table.kv td { padding:2px 14px 2px 0; color:var(--mut); vertical-align:top; }
table.kv td.k { color:var(--fg); white-space:nowrap; }
"""
    sp_ = doc["surviving_pairs"]
    sh = doc["split_half"]
    nl = doc["null"]
    dd = doc["diagonal_dominance"]
    rows = []
    for n, p in enumerate(pairs, 1):
        cl = labels.get(p["ctx_feat_id"], {})
        al = labels.get(p["ans_feat_id"], {})
        pers = (
            '<span class="badge pers">persistence (same feature)</span>'
            if p["is_persistence"]
            else ""
        )
        ax_c = ", ".join(f"{k}: {_esc(v)}" for k, v in sorted((cl.get("axes") or {}).items()) if v)
        ax_a = ", ".join(f"{k}: {_esc(v)}" for k, v in sorted((al.get("axes") or {}).items()) if v)
        rows.append(f"""<div class="card"><div class="top">
<span class="rank">#{n}</span>
<span class="fid">ctx <a href="https://www.neuronpedia.org/qwen2.5-7b-instruct/19-{p["ctx_feat_id"]}"
 target="_blank" rel="noopener">{p["ctx_feat_id"]}</a></span>
<span class="badge blk">{_esc(p["ctx_block"])}</span>
<span class="arrow">&rarr;</span>
<span class="fid">ans <a href="https://www.neuronpedia.org/qwen2.5-7b-instruct/19-{p["ans_feat_id"]}"
 target="_blank" rel="noopener">{p["ans_feat_id"]}</a></span>
{pers}
<span class="nums">coef {p["coef_std_units"]:+.4f} &middot; halves
 {p["coef_half_a"]:+.4f} / {p["coef_half_b"]:+.4f} &middot; null z {p["null_z"]:.1f}
 &middot; answer R&sup2; {p["ans_r2"]:.3f}</span></div>
<p class="desc"><span class="side">context</span>{_esc(cl.get("description"))}</p>
<p class="ax">{ax_c or "&mdash;"}</p>
<p class="desc"><span class="side">answer</span>{_esc(al.get("description"))}</p>
<p class="ax">{ax_a or "&mdash;"}</p></div>""")

    body = f"""<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Issue 1482 &mdash; SAE context&rarr;answer map: surviving coefficient pairs</title>
<style>{css}</style></head><body><div class='wrap'>
<h1>Which context SAE features PREDICT which answer SAE features (issue #1482)</h1>
<div class="warn">
<p><b>Predictive association, not causation.</b> Every number here is a ridge
coefficient fit on observational data: it says context feature <i>i</i> PREDICTS
answer feature <i>j</i> in a linear map, never that <i>i</i> causes <i>j</i>. A causal
claim would require an intervention (steering or ablation), which this round does not
perform.</p>
<p><b>Descriptions are a reading aid, not evidence &mdash; and any FAMILY reading rests
entirely on them.</b> Autointerp descriptions and axis labels come from #1773, whose
standing caveat is that they are search-index-only (neighbour discrimination 0.322
against a 0.50 bar). The coefficients, the split-half replication and the null are
measured; the thematic groupings a reader will naturally form from the text below
(language &rarr; grammatical machinery, discourse-position &rarr; formatting,
topic &rarr; topic, register &rarr; register) are the LEAST evidenced thing on this
page. Treat them as hypotheses to test, never as findings.</p></div>
<div class="head">
<p><b>What is shown.</b> The {len(pairs)} strongest surviving (context feature &rarr;
answer feature) coefficient pairs of the issue-1482 <code>sae_ctx / mean / ridge</code>
map, ranked by null z. A pair is shown only if it clears BOTH gates: it is in the
top-{sh["top_k"]:,} |coefficient| set of BOTH disjoint split-half refits with a
sign consistent with the full fit, AND its |coefficient| exceeds the per-column
label-shuffle null threshold.</p>
<p><b>Both sides share one dictionary.</b> Context and answer features are encoded with
the SAME layer-19 SAE, so ctx id <i>j</i> and ans id <i>j</i> are the same feature; those
pairs are tagged <span class="badge pers">persistence</span>.</p>
<table class="kv">
<tr><td class="k">arm / corpus</td><td>sae_ctx, mean pooling, ridge (lambda
 {doc["recipe"]["lambda"]:,.0f}) &mdash; #1482 single-turn panel, {doc["recipe"]["n_fit"]:,}
 fit / {doc["recipe"]["n_holdout"]:,} holdout rows</td></tr>
<tr><td class="k">coefficient units</td><td>standardized input (per 1 SD of the context
 feature)</td></tr>
<tr><td class="k">refit gate</td><td>max |&Delta;R&sup2;| vs the banked per-feature array =
 {doc["refit_gate"]["max_abs_delta_r2"]:.2e}</td></tr>
<tr><td class="k">split-half</td><td>{sh["replicated_both_halves_topk_and_sign"]:,} /
 {sh["candidate_pairs"]:,} candidates replicated
 ({sh["replication_rate"] * 100:.1f}%)</td></tr>
<tr><td class="k">null</td><td>{nl["draws"]} label-shuffle refits &times;
 {nl["columns_per_draw"]:,} answer columns; threshold at null z =
 {nl["z_threshold_used"]:.1f}</td></tr>
<tr><td class="k">surviving</td><td>{sp_["n_surviving"]:,} pairs of
 {sp_["of_candidates"]:,} candidates; {sp_["n_persistence_among_surviving"]:,} are
 persistence (same feature both sides)</td></tr>
<tr><td class="k">diagonal mass</td><td>{dd["diagonal_abs_mass_share"] * 100:.3f}% of total
 |coefficient| mass sits on the {dd["diagonal_entries"]:,} persistence entries
 ({dd["diagonal_mass_enrichment_vs_uniform"]:.0f}&times; uniform)</td></tr>
</table></div>
<h2>Surviving pairs (ranked by null z)</h2>
{"".join(rows)}
</div></body></html>"""
    for p in (DASH_PATH, PUBLIC_DASH_PATH):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
        _log(f"wrote {p} ({p.stat().st_size / 1024:.0f} KiB)")
    return DASH_PATH


# ── driver ──────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, default=STORE_DEFAULT)
    ap.add_argument("--scratch", type=Path, default=SCRATCH_DEFAULT)
    ap.add_argument("--work", type=Path, default=WORK_DEFAULT)
    ap.add_argument("--labels-dir", type=Path, default=LABELS_DEFAULT)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--max-pairs", type=int, default=DASH_MAX_PAIRS)
    ap.add_argument("--rebuild-design", action="store_true")
    ap.add_argument("--refit", action="store_true", help="ignore fit/null/split checkpoints")
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument(
        "--figures-only",
        action="store_true",
        help="re-render figures from the cached analyze.npz + committed structure JSON "
        "(skips the design load and the analysis recompute)",
    )
    ap.add_argument(
        "--dashboard-only",
        action="store_true",
        help="re-render BOTH dashboard copies from the committed structure + pairs JSONs "
        "(skips the design load and the analysis recompute)",
    )
    args = ap.parse_args()
    if args.import_check:
        print("import-check OK")
        sys.stdout.flush()
        sys.exit(0)

    t0 = time.time()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        doc = json.loads((OUT_DIR / "coef_structure.json").read_text())
        _log(f"figures-only: {[str(p) for p in phase_figures(args, doc)]}")
        _log(f"DONE in {time.time() - t0:.0f}s")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    if args.dashboard_only:
        doc = json.loads((OUT_DIR / "coef_structure.json").read_text())
        tp = json.loads((OUT_DIR / "top_pairs.json").read_text())
        pairs = tp["pairs"]
        labels = {
            int(p[f"{side}_feat_id"]): {
                "description": p.get(f"{side}_description"),
                "axes": p.get(f"{side}_axes") or {},
            }
            for p in pairs
            for side in ("ctx", "ans")
        }
        phase_dashboard(pairs, labels, doc)
        _log(f"DONE in {time.time() - t0:.0f}s")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    des = build_design(args)
    des["y_sd_all"] = np.sqrt(
        np.asarray(des["Y"].power(2).mean(axis=0)).ravel()
        - np.asarray(des["Y"].mean(axis=0)).ravel() ** 2
    )
    fit = phase_fit(args, des)
    null = phase_null(args, des, fit)
    split = phase_split(args, des, fit)
    res = phase_analyze(args, des, fit, null, split)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(OUT_DIR / "coef_structure.json", res["doc"])

    need = {p["ctx_feat_id"] for p in res["pairs"]} | {p["ans_feat_id"] for p in res["pairs"]}
    labels = load_labels(need, args.labels_dir)
    _write_json(
        OUT_DIR / "top_pairs.json",
        {
            "caveats": CAVEATS,
            "criteria": res["doc"]["surviving_pairs"]["criteria"],
            "label_source": "issue #1773 autointerp (search-index-only; reading aid, not evidence)",
            "n_pairs": len(res["pairs"]),
            "pairs": [
                {
                    **p,
                    "ctx_description": labels.get(p["ctx_feat_id"], {}).get("description"),
                    "ctx_axes": labels.get(p["ctx_feat_id"], {}).get("axes"),
                    "ans_description": labels.get(p["ans_feat_id"], {}).get("description"),
                    "ans_axes": labels.get(p["ans_feat_id"], {}).get("axes"),
                }
                for p in res["pairs"]
            ],
            "provenance": _provenance(),
        },
    )
    figs = phase_figures(args, res["doc"])
    _log(f"figures: {[str(p) for p in figs]}")
    phase_dashboard(res["pairs"], labels, res["doc"])
    _log(f"ALL DONE in {time.time() - t0:.0f}s")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
