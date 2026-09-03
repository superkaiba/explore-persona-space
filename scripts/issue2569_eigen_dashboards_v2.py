#!/usr/bin/env python
"""Issue #2569 leg-1 SAE dashboards v2: eigen 2-planes + whitened cosines (CPU).

Redo of the leg-1 step-4 two-sided SAE dashboards (v1:
``eval_results/issue_2569/weights/leg1/sae_dashboards_L19.json``) fixing the two
documented v1 deferrals:

1. **Complex eigenvectors get their full invariant 2-plane.** v1 dashboarded a
   complex eigenvector through the normalized REAL part only (median im_frac
   0.65-0.74 across the top-32, so most of the vector was dropped). Here each
   conjugate pair contributes ONE real invariant 2-plane spanned by
   ``(Re w, Im w)`` (orthonormalized, real axis first so axis-1 alone reproduces
   the v1 read); real eigenvalues contribute a 1-D direction; the walk continues
   down the |lambda|-sorted spectrum until ``top_k`` distinct planes/directions
   per eigen section. The dictionary statistic for a 2-plane is the PLANE
   COSINE ``||P^T d||_2`` for a unit decoder direction ``d`` and orthonormal
   plane basis ``P`` (= the cosine between ``d`` and its projection onto the
   plane; reduces to |cos| for a 1-D direction).

2. **Side-matched whitened-cosine companion.** v1 shipped raw cosines only
   (``whitened_cosine: deferred-to-P-B``). Here Sigma_c (context side) and
   Sigma_a (answer side) are built from the P-B moments artifacts
   (``issue2569_theory/analysis_tensors/moments/gram_{xx,yy}.pt``), which hold
   UNCENTERED fp64 sums-of-outer Grams plus per-dim means over the n_pool
   RAW-residual X19/Y19 rows (``issue2569_rowbattery._accumulate_moments``
   consumes the fp16 memmaps directly — no centering, no standardization — so
   the Grams live in the same raw coordinates the row operator A acts on).
   Conversion used: ``Sigma = (gram - n * mean mean^T) / (n - 1)`` (centered,
   unbiased). Shrinkage: ``Sigma + lam * (tr Sigma / d) * I`` with lam = 1e-2
   primary and 1e-3 sensitivity. Whitened cosine between direction x and
   decoder column d = ``cos(Sigma^{-1/2} x, Sigma^{-1/2} d)``; for 2-planes the
   two basis vectors are whitened then re-orthonormalized (real axis first).

Direction families follow the B1 orientation dictionary verbatim
(``issue2569_operator``): the banked L19 map acts on ROWS (``vhat = v @ A + b``,
``A = diag(1/xsd) @ W``); SVD read = LEFT singular vectors u_i
(``u_i @ A = sigma_i v_i``), SVD write = RIGHT singular vectors v_i; eigen read
= RIGHT eigenvectors of A (``A v = lam v``), eigen write = LEFT eigenvector
rows (rows of ``inv(V)``, biorthonormal by construction). Read side is
compared against the andyrdt per-token context SAE (131,072 features) AND — as
a second, grain-matched read dictionary — the #2569 leg-4 trained context SAE
(matryoshka batchtopk, 65,536 features, k=100, trained on the very X19
last-prompt-token context rows the map reads; features carry NO judged
descriptions, so those sections report feature ids only). Write side is
compared against the #2476 turn-averaged answer SAE (65,536 features).

Null floors, per dictionary and per direction kind:
  * 1-D analytic: ``sqrt(2 ln N / d)`` (the v1 convention — the "expected one
    exceedance among N independent features" scale).
  * 2-plane analytic: for a random unit d and a fixed 2-plane,
    ``plane_cos^2 ~ Beta(1, (d-2)/2)`` with EXACT survival
    ``(1-t)^{(d-2)/2}``; the same one-exceedance convention gives
    ``floor = sqrt(1 - N^{-2/(d-2)})``.
  * Empirical: p95 of the max statistic over ``n_draws`` random unit
    directions (1-D) / random Haar 2-planes (2-D). Whitened floors re-run the
    SAME seeded raw draws through the whitening pipeline (whiten, re-normalize
    / re-orthonormalize) against the whitened dictionary.

The eigendecomposition is recomputed here in float64 (``scipy.linalg.eig`` +
``inv``) and asserted against the banked factor artifact's reference numbers
(rho, kappa(V), pair/real counts, sigma_max, biorthogonality < 1e-8). CPU
only; the dense ctx-dictionary GEMMs are fp32; the 131,072 x 65,536
feature-to-feature matrix is never formed.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

WORKTREE_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(WORKTREE_ROOT / "src"), str(WORKTREE_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LAYER = 19
TOP_K = 32
TOP_M = 8  # dictionary features quoted per direction
N_DRAWS = 1_000
NULL_SEED = 2_569_140  # v1 parity: ctx lines = seed, ans lines = seed+1
PLANE_SEED_OFFSET = 2  # ctx planes = seed+2, ans planes = seed+3
CTX2_SEED_OFFSET = 4  # trained ctx SAE: lines = seed+4, planes = seed+5
SHRINK_LAMBDAS = (1e-2, 1e-3)
MAIN_REPO_DEFAULT = "/home/thomasjiralerspong/explore-persona-space"
HF_CACHE_DEFAULT = "/mnt/eps-data/thomasjiralerspong/huggingface-cache/hub"
SAE_CTX_DIR_DEFAULT = "/mnt/eps-data/thomasjiralerspong/issue2569_theory/sae_ctx"
SECTION_SIDE = {
    "singular_read": "ctx",
    "eigen_read": "ctx",
    "singular_write": "ans",
    "eigen_write": "ans",
    "singular_read_ctxsae": "ctx_trained",
    "eigen_read_ctxsae": "ctx_trained",
}

# Reference numbers from the banked eval_results/issue_2569/weights/leg1/factor_L19.json
REFERENCE = {
    "rho": 1.2053517865293906,
    "kappa_v": 4260.870580356614,
    "n_complex_pairs": 1751,
    "n_real_eigs": 82,
    "sigma_max": 7.961798388540215,
}
REF_RTOL = 1e-3
BIORTHO_CEILING = 1e-8

MOMENT_CONVERSION_NOTE = (
    "gram_xx.pt / gram_yy.pt hold UNCENTERED fp64 sums-of-outer Grams (X^T X / Y^T Y) plus "
    "per-dim means over the raw-residual X19/Y19 map-training pool rows "
    "(issue2569_rowbattery._accumulate_moments reads the fp16 memmaps directly; no centering, "
    "no standardization), i.e. the same raw coordinates the row operator A acts on. "
    "Conversion: Sigma = (gram - n * outer(mean, mean)) / (n - 1) (centered covariance, "
    "unbiased). Shrinkage: Sigma + lam * (tr(Sigma)/d) * I."
)


# ── pure helpers (unit-tested on d <= 48 synthetics) ─────────────────────────────


def unit(x: np.ndarray) -> np.ndarray:
    """Unit-normalized fp64 copy of a vector (asserts a nonzero norm)."""
    x = np.asarray(x, dtype=np.float64)
    n = float(np.linalg.norm(x))
    assert n > 0.0, "cannot normalize a zero vector"
    return x / n


def plane_basis_from_complex(z: np.ndarray, degen_tol: float = 1e-10) -> tuple[np.ndarray, dict]:
    """Orthonormal real 2-plane basis for a complex eigenvector, real axis first.

    ``z (d,) complex``. Row 0 = normalized real part (exactly the v1 real-part
    read); row 1 = imaginary part orthogonalized against row 0, normalized.
    Returns ``(Q (2, d) fp64, diag)`` with ``diag['im_frac'] = |Im z| / |z|``
    (the v1 per-direction imaginary-mass diagnostic) and
    ``diag['degenerate'] = True`` (with a 1-row Q) when the real part or the
    orthogonalized imaginary component vanishes — the span is then 1-D.
    """
    z = np.asarray(z, dtype=np.complex128)
    nz = float(np.linalg.norm(z))
    assert nz > 0.0, "zero eigenvector"
    re = z.real.astype(np.float64)
    im = z.imag.astype(np.float64)
    im_frac = float(np.linalg.norm(im) / nz)
    nre = float(np.linalg.norm(re))
    if nre <= degen_tol * nz:  # purely imaginary: 1-D span along Im z
        return unit(im)[None, :], {"im_frac": im_frac, "degenerate": True}
    q1 = re / nre
    imp = im - (im @ q1) * q1
    nimp = float(np.linalg.norm(imp))
    if nimp <= degen_tol * nz:  # Re and Im parallel: 1-D span
        return q1[None, :], {"im_frac": im_frac, "degenerate": True}
    q2 = imp / nimp
    Q = np.stack([q1, q2])
    return Q, {"im_frac": im_frac, "degenerate": False}


def _realized_line(z: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit REAL direction for a (nominally real-eigenvalue) eigenvector.

    Rotates the global complex phase so the largest-|entry| coordinate is real,
    then takes the real part. Returns ``(unit fp64 (d,), residual_im_frac)``
    where the residual is the post-rotation imaginary mass (0 for LAPACK real
    eigenvectors of a real matrix).
    """
    z = np.asarray(z, dtype=np.complex128)
    k = int(np.argmax(np.abs(z)))
    ph = z[k] / abs(z[k])
    zr = z / ph
    res = float(np.linalg.norm(zr.imag) / max(np.linalg.norm(zr), 1e-300))
    return unit(zr.real), res


def collapse_eigen_directions(
    lam: np.ndarray,
    V: np.ndarray,
    U_rows: np.ndarray,
    n_want: int,
    tol_imag: float | None = None,
) -> list[dict]:
    """Walk the |lambda|-descending spectrum collapsing conjugate pairs.

    ``lam (d,)`` complex eigenvalues; ``V (d, d)`` right-eigenvector COLUMNS
    (read side); ``U_rows (d, d)`` left-eigenvector ROWS (= inv(V), write
    side). Each complex conjugate pair contributes ONE entry with a read
    2-plane from V[:, i] and a write 2-plane from U_rows[i]; each real
    eigenvalue contributes a 1-D entry. Stops at ``n_want`` entries. Entry
    keys: kind ('plane'|'line'), lam_re, lam_im, abs_lambda, ranks (1-based
    positions of the member(s) in |lambda| order), read_basis (k, d),
    write_basis (k, d) orthonormal-row fp64, im_frac_read, im_frac_write.
    """
    lam = np.asarray(lam, dtype=np.complex128)
    d = lam.size
    if tol_imag is None:
        tol_imag = 1e-12 * max(float(np.abs(lam).max()), 1.0)
    order = np.argsort(-np.abs(lam), kind="stable")
    rank_of = {int(i): r + 1 for r, i in enumerate(order)}
    used: set[int] = set()
    entries: list[dict] = []
    for i in order:
        i = int(i)
        if i in used:
            continue
        used.add(i)
        if abs(lam[i].imag) <= tol_imag:
            rvec, r_res = _realized_line(V[:, i])
            wvec, w_res = _realized_line(U_rows[i])
            entries.append(
                {
                    "kind": "line",
                    "lam_re": float(lam[i].real),
                    "lam_im": float(lam[i].imag),
                    "abs_lambda": float(abs(lam[i])),
                    "ranks": [rank_of[i]],
                    "read_basis": rvec[None, :],
                    "write_basis": wvec[None, :],
                    "im_frac_read": r_res,
                    "im_frac_write": w_res,
                }
            )
        else:
            # conjugate partner: unused j minimizing |lam[j] - conj(lam[i])|
            target = np.conj(lam[i])
            best_j, best_err = None, np.inf
            for j in np.flatnonzero(np.abs(lam - target) <= 1e-6 * max(abs(lam[i]), 1.0)):
                j = int(j)
                if j in used:
                    continue
                err = float(abs(lam[j] - target))
                if err < best_err:
                    best_j, best_err = j, err
            ranks = [rank_of[i]]
            if best_j is not None:
                used.add(best_j)
                ranks.append(rank_of[best_j])
            Qr, dr = plane_basis_from_complex(V[:, i])
            Qw, dw = plane_basis_from_complex(U_rows[i])
            entries.append(
                {
                    "kind": "plane" if (Qr.shape[0] == 2 and Qw.shape[0] == 2) else "line",
                    "lam_re": float(lam[i].real),
                    "lam_im": float(lam[i].imag),
                    "abs_lambda": float(abs(lam[i])),
                    "ranks": sorted(ranks),
                    "read_basis": Qr,
                    "write_basis": Qw,
                    "im_frac_read": float(dr["im_frac"]),
                    "im_frac_write": float(dw["im_frac"]),
                    "conjugate_partner_found": best_j is not None,
                }
            )
        if len(entries) >= n_want:
            break
    return entries


def entry_cosine_stats(bases: list[np.ndarray], D_normed: np.ndarray, top_m: int) -> list[dict]:
    """Plane/line cosines of orthonormal-row bases against a column-normed dictionary.

    ``bases``: list of ``(k_i, d)`` fp64 arrays with orthonormal rows (k_i in
    {1, 2}); ``D_normed``: ``(d, N)`` fp32 column-normed dictionary. One fp32
    GEMM for all rows. Per entry: ``max`` (max plane cosine / |cos| over the
    dictionary), ``top_ids`` / ``top_vals`` (top-``top_m`` by the statistic),
    ``axis1_max`` (max |cos| along basis row 0 alone — for an eigen plane this
    is exactly the v1 normalized-real-part read), ``axis2_share_top`` (per top
    feature, the fraction of the squared plane cosine carried by basis row 1 —
    the imaginary axis; only for 2-row bases), ``signed_axis_cos_top``.
    """
    assert all(B.ndim == 2 for B in bases)
    R = np.concatenate(bases, axis=0).astype(np.float32)
    Cm = R @ D_normed  # (M, N) fp32
    out: list[dict] = []
    r = 0
    for B in bases:
        k = B.shape[0]
        c1 = Cm[r].astype(np.float64)
        c2 = Cm[r + 1].astype(np.float64) if k == 2 else None
        vals = np.abs(c1) if c2 is None else np.sqrt(c1 * c1 + c2 * c2)
        m = int(min(top_m, vals.size))
        part = np.argpartition(-vals, m - 1)[:m]
        ids = part[np.argsort(-vals[part])]
        ent = {
            "max": float(vals[ids[0]]),
            "top_ids": [int(x) for x in ids],
            "top_vals": [float(vals[x]) for x in ids],
            "axis1_max": float(np.abs(c1).max()),
        }
        if c2 is None:
            ent["signed_axis_cos_top"] = [[float(c1[x])] for x in ids]
        else:
            ent["axis2_share_top"] = [
                float((c2[x] ** 2) / max(vals[x] ** 2, 1e-300)) for x in ids
            ]
            ent["signed_axis_cos_top"] = [[float(c1[x]), float(c2[x])] for x in ids]
        out.append(ent)
        r += k
    return out


def shrunk_covariance(Sigma: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """``Sigma + lam * (tr Sigma / d) * I`` (fp64); returns (shrunk, tau=tr/d)."""
    Sigma = np.asarray(Sigma, dtype=np.float64)
    d = Sigma.shape[0]
    tau = float(np.trace(Sigma)) / d
    return Sigma + lam * tau * np.eye(d), tau


def inv_sqrt_psd(S: np.ndarray) -> np.ndarray:
    """Symmetric inverse square root via eigh (asserts strictly positive spectrum)."""
    S = np.asarray(S, dtype=np.float64)
    S = 0.5 * (S + S.T)
    evals, evecs = np.linalg.eigh(S)
    assert float(evals.min()) > 0.0, f"non-PD covariance after shrinkage: min eig {evals.min()}"
    return (evecs * (evals**-0.5)) @ evecs.T


def whiten_bases(bases: list[np.ndarray], Wm: np.ndarray) -> list[np.ndarray]:
    """Whiten each basis row (``x -> x @ Wm``) then re-orthonormalize, row 0 first.

    Keeps the axis semantics: after whitening, row 0 spans the whitened real
    axis and row 1 the whitened imaginary component orthogonalized against it.
    """
    out = []
    for B in bases:
        Y = np.asarray(B, dtype=np.float64) @ Wm
        q1 = unit(Y[0])
        if B.shape[0] == 1:
            out.append(q1[None, :])
            continue
        y2 = Y[1] - (Y[1] @ q1) * q1
        out.append(np.stack([q1, unit(y2)]))
    return out


def analytic_max_cos_floor(n_features: int, d: int) -> float:
    """1-D analytic max-|cos| floor ``sqrt(2 ln N / d)`` (v1 convention)."""
    return math.sqrt(2.0 * math.log(float(n_features)) / float(d))


def analytic_max_plane_cos_floor(n_features: int, d: int) -> float:
    """2-plane analytic floor ``sqrt(1 - N^(-2/(d-2)))``.

    For a random unit direction and a fixed 2-plane, ``plane_cos^2 ~
    Beta(1, (d-2)/2)`` with exact survival ``(1-t)^((d-2)/2)``; the same
    one-expected-exceedance convention as the 1-D floor
    (``N * survival(t) = 1``) gives ``t = 1 - N^(-2/(d-2))``.
    """
    assert d > 2
    return math.sqrt(1.0 - float(n_features) ** (-2.0 / (d - 2)))


def _null_summary(maxima: np.ndarray, n_draws: int, seed: int) -> dict:
    qs = np.quantile(maxima, [0.5, 0.9, 0.95, 0.99])
    return {
        "n_draws": int(n_draws),
        "seed": int(seed),
        "mean": float(maxima.mean()),
        "p50": float(qs[0]),
        "p90": float(qs[1]),
        "p95": float(qs[2]),
        "p99": float(qs[3]),
        "max": float(maxima.max()),
    }


def empirical_max_cos_null_lines(
    D_normed: np.ndarray,
    n_draws: int,
    seed: int,
    whitener32: np.ndarray | None = None,
    chunk: int = 256,
) -> dict:
    """Empirical max-|cos| null over random unit directions (optionally whitened).

    Raw path reproduces ``issue2569_weights.empirical_max_cos_null`` draw for
    draw (same rng consumption, same chunking); with ``whitener32`` the SAME
    raw unit draws are pushed through ``x -> x @ W`` and re-normalized before
    the GEMM against the (whitened) dictionary — the whitened-metric null.
    """
    d = int(D_normed.shape[0])
    rng = np.random.default_rng(seed)
    maxima = np.empty(int(n_draws), dtype=np.float64)
    done = 0
    while done < n_draws:
        b = int(min(chunk, n_draws - done))
        draws = rng.standard_normal((b, d)).astype(np.float32)
        draws /= np.linalg.norm(draws, axis=1, keepdims=True)
        if whitener32 is not None:
            draws = draws @ whitener32
            draws /= np.linalg.norm(draws, axis=1, keepdims=True)
        maxima[done : done + b] = np.abs(draws @ D_normed).max(axis=1)
        done += b
    return _null_summary(maxima, n_draws, seed)


def empirical_max_plane_cos_null(
    D_normed: np.ndarray,
    n_draws: int,
    seed: int,
    whitener32: np.ndarray | None = None,
    chunk: int = 128,
) -> dict:
    """Empirical max plane-cosine null over random Haar 2-planes (optionally whitened).

    Per draw: two iid Gaussian d-vectors, Gram-Schmidt orthonormalized (their
    span is a Haar-random 2-plane). With ``whitener32`` the SAME raw plane
    basis is whitened then re-orthonormalized (the pipeline the dashboard
    planes go through) before the GEMM against the (whitened) dictionary.
    """
    d = int(D_normed.shape[0])
    rng = np.random.default_rng(seed)
    maxima = np.empty(int(n_draws), dtype=np.float64)
    done = 0
    while done < n_draws:
        b = int(min(chunk, n_draws - done))
        g = rng.standard_normal((b, 2, d)).astype(np.float32)
        q1 = g[:, 0] / np.linalg.norm(g[:, 0], axis=1, keepdims=True)
        y2 = g[:, 1] - np.sum(g[:, 1] * q1, axis=1, keepdims=True) * q1
        q2 = y2 / np.linalg.norm(y2, axis=1, keepdims=True)
        if whitener32 is not None:
            q1 = q1 @ whitener32
            q1 /= np.linalg.norm(q1, axis=1, keepdims=True)
            w2 = q2 @ whitener32
            w2 = w2 - np.sum(w2 * q1, axis=1, keepdims=True) * q1
            q2 = w2 / np.linalg.norm(w2, axis=1, keepdims=True)
        c1 = q1 @ D_normed  # (b, N)
        c2 = q2 @ D_normed
        maxima[done : done + b] = np.sqrt(c1 * c1 + c2 * c2).max(axis=1)
        done += b
    return _null_summary(maxima, n_draws, seed)


# ── IO helpers ────────────────────────────────────────────────────────────────────


def load_raw_covariance(pt_path: Path) -> tuple[np.ndarray, dict]:
    """Raw-coordinate centered covariance from a gram_{xx,yy}.pt moments file.

    File contract (``issue2569_rowbattery._write_sigma_pt``): ``{"gram": (d, d)
    fp64 uncentered sum-of-outer, "mean": (d,), "n_rows": int}``. Returns
    ``Sigma = (gram - n * outer(mean, mean)) / (n - 1)`` (fp64, symmetrized)
    plus a provenance dict.
    """
    import torch

    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    gram = obj["gram"].to(torch.float64).numpy()
    mean = obj["mean"].to(torch.float64).numpy()
    n = int(obj["n_rows"])
    Sigma = (gram - n * np.outer(mean, mean)) / (n - 1)
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma, {
        "path": str(pt_path),
        "n_rows": n,
        "side": str(obj.get("side", "")),
        "pool": str(obj.get("pool", "")),
        "trace": float(np.trace(Sigma)),
    }


def resolve_cached_dataset_file(hf_cache: Path, relpath: str) -> Path:
    """Newest cached copy of a superkaiba1/explore-persona-space-data file."""
    snaps = hf_cache / "datasets--superkaiba1--explore-persona-space-data" / "snapshots"
    cands = [p for p in snaps.glob(f"*/{relpath}") if p.exists()]
    assert cands, f"not in HF cache: {relpath} (run the download step first)"
    return max(cands, key=lambda p: p.stat().st_mtime)


def load_ctx_sae_local(hf_cache: Path):
    """andyrdt L19 k=64 context SAE from the local HF model cache (no network)."""
    import torch

    import issue1482_sae as S1482

    snaps = hf_cache / "models--andyrdt--saes-qwen2.5-7b-instruct" / "snapshots"
    cands = [p for p in snaps.glob("*/resid_post_layer_19/trainer_1/ae.pt")]
    assert cands, "andyrdt ae.pt not in the local HF cache"
    ae = max(cands, key=lambda p: p.stat().st_mtime)
    cfg = json.loads((ae.parent / "config.json").read_text())["trainer"]
    assert cfg["dict_class"] == "BatchTopKSAE" and cfg["k"] == 64 and cfg["layer"] == 19, cfg
    sd = torch.load(ae, map_location="cpu", weights_only=True)
    return S1482.BatchTopKSAE(sd, k=64)


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg} (rss {_rss_gb():.1f} GB)", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--main-root", type=Path, default=Path(MAIN_REPO_DEFAULT))
    ap.add_argument("--hf-cache", type=Path, default=Path(HF_CACHE_DEFAULT))
    ap.add_argument(
        "--out-dir", type=Path, default=WORKTREE_ROOT / "eval_results/issue_2569/weights/leg1"
    )
    ap.add_argument("--fig-dir", type=Path, default=WORKTREE_ROOT / "figures/issue_2569")
    ap.add_argument("--label-root", type=Path, default=Path("/tmp/i2569v2_labelroot"))
    ap.add_argument("--sae-ctx-dir", type=Path, default=Path(SAE_CTX_DIR_DEFAULT))
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    import torch

    torch.set_num_threads(args.threads)

    import scipy.linalg as sla

    import issue2569_operator as OP
    import issue2569_weights as W2569
    import issue2569_rowbattery as RB
    import issue2476_turnavg_sae as T24

    t_start = time.time()
    k = int(args.top_k)
    m = TOP_M
    d_expect = 3584

    # ── operator + decompositions ────────────────────────────────────────────
    payload = OP.load_banked_map(LAYER, root=args.main_root)
    A, b = OP.row_operator(payload)
    d = A.shape[0]
    assert d == d_expect, d
    _log(f"loaded banked L{LAYER} map (d={d}, lambda={payload.selected_lambda})")

    Usvd, sig, Vt = sla.svd(A, lapack_driver="gesdd")  # A = U diag(s) V^T
    _log(f"full fp64 SVD done: sigma_max={sig[0]:.6f}")

    lam, V = sla.eig(A)  # complex128
    U_rows = np.linalg.inv(V)
    rho = float(np.abs(lam).max())
    _log(f"full fp64 eig done: rho={rho:.7f}")
    kappa_v = float(np.linalg.cond(V))
    biortho = float(np.abs(U_rows @ V - np.eye(d)).max())
    tol_im = 1e-12 * max(rho, 1.0)
    n_complex = int(np.sum(np.abs(lam.imag) > tol_im))
    n_pairs, n_real = n_complex // 2, d - n_complex
    _log(f"kappa_v={kappa_v:.2f} biortho={biortho:.2e} pairs={n_pairs} reals={n_real}")

    checks = {
        "rho": {"value": rho, "reference": REFERENCE["rho"]},
        "kappa_v": {"value": kappa_v, "reference": REFERENCE["kappa_v"]},
        "sigma_max": {"value": float(sig[0]), "reference": REFERENCE["sigma_max"]},
        "n_complex_pairs": {"value": n_pairs, "reference": REFERENCE["n_complex_pairs"]},
        "n_real_eigs": {"value": n_real, "reference": REFERENCE["n_real_eigs"]},
        "biortho_max_err": {"value": biortho, "ceiling": BIORTHO_CEILING},
    }
    for key in ("rho", "kappa_v", "sigma_max"):
        v, r = checks[key]["value"], checks[key]["reference"]
        rel = abs(v - r) / abs(r)
        checks[key]["rel_err"] = rel
        assert rel < REF_RTOL, f"{key}: {v} vs reference {r} (rel {rel:.2e})"
    assert n_pairs == REFERENCE["n_complex_pairs"], (n_pairs, REFERENCE["n_complex_pairs"])
    assert n_real == REFERENCE["n_real_eigs"], (n_real, REFERENCE["n_real_eigs"])
    assert biortho < BIORTHO_CEILING, biortho

    # ── direction sets (B1 orientation dictionary) ───────────────────────────
    svd_read = [Usvd[:, j].copy()[None, :] for j in range(k)]  # LEFT singular = read
    svd_write = [Vt[j].copy()[None, :] for j in range(k)]  # RIGHT singular = write
    eig_entries = collapse_eigen_directions(lam, V, U_rows, n_want=k, tol_imag=tol_im)
    assert len(eig_entries) == k
    eig_read_bases = [e["read_basis"] for e in eig_entries]
    eig_write_bases = [e["write_basis"] for e in eig_entries]
    n_planes = sum(1 for e in eig_entries if e["kind"] == "plane")
    _log(f"direction sets built: eigen top-{k} = {n_planes} planes + {k - n_planes} lines")
    del Usvd, Vt, V, U_rows  # keep lam for metadata; free the dense factors
    del A

    # ── dictionaries, labels ─────────────────────────────────────────────────
    ctx_sae = load_ctx_sae_local(args.hf_cache)
    sae_c_dir = resolve_cached_dataset_file(
        args.hf_cache, "issue2476_turnavg/analysis_tensors/sae_c/cfg.json"
    ).parent
    ans_sae = T24.MatryoshkaBatchTopKSAE.load_local(sae_c_dir, device="cpu")
    # second read dictionary: the #2569 leg-4 trained context SAE (grain-matched:
    # trained on the X19 last-prompt-token context rows the map reads)
    try:
        ctx2_sae = RB.load_sae_ctx(args.sae_ctx_dir / "ae.pt", device="cpu")
    except Exception as exc:  # HF mirror fallback per the task brief
        _log(f"local sae_ctx load failed ({exc!r}); falling back to the HF mirror")
        ctx2_sae = RB.load_sae_ctx(
            resolve_cached_dataset_file(
                args.hf_cache, "issue2569_theory/analysis_tensors/sae_ctx/ae.pt"
            ),
            device="cpu",
        )
    ctx2_cfg = json.loads((args.sae_ctx_dir / "config.json").read_text())
    assert ctx2_cfg["act_dim"] == d and ctx2_cfg["dict_size"] == 65536, ctx2_cfg
    alive = json.loads((args.sae_ctx_dir / "alive_union.json").read_text())
    ctx2_alive_idx = np.asarray(alive["alive_idx"], dtype=np.int64)
    ctx2_union_full = bool(len(ctx2_alive_idx) == int(alive["n_dict"]))
    ctx_labels, ctx_label_doc = W2569.load_ctx_feature_labels(args.label_root)
    ans_desc_path = WORKTREE_ROOT / "eval_results/issue_2569/der/consumed_input/descriptions_mat_k100.json"
    ans_labels = {
        int(fid): str(desc)[:240]
        for fid, desc in json.loads(ans_desc_path.read_text())["descriptions"].items()
    }
    _log(f"dictionaries loaded (ctx labels {len(ctx_labels)}, ans labels {len(ans_labels)})")

    # ── covariances ──────────────────────────────────────────────────────────
    Sigma_c, sig_c_doc = load_raw_covariance(
        resolve_cached_dataset_file(args.hf_cache, "issue2569_theory/analysis_tensors/moments/gram_xx.pt")
    )
    Sigma_a, sig_a_doc = load_raw_covariance(
        resolve_cached_dataset_file(args.hf_cache, "issue2569_theory/analysis_tensors/moments/gram_yy.pt")
    )
    assert Sigma_c.shape == (d, d) and Sigma_a.shape == (d, d)
    _log(f"covariances built (tr Sigma_c={sig_c_doc['trace']:.1f}, tr Sigma_a={sig_a_doc['trace']:.1f})")

    # ── per-side pipeline ────────────────────────────────────────────────────
    def ctx_label_of(fid: int) -> str:
        row = ctx_labels.get(int(fid))
        return str(row["description"])[:240] if row else "(no description)"

    def ans_label_of(fid: int) -> str:
        return ans_labels.get(int(fid), "(no description)")

    sides = {
        "ctx": {
            "D_raw": ctx_sae.w_dec.detach().cpu().numpy(),  # (d, N) columns = features
            "Sigma": Sigma_c,
            "label_of": ctx_label_of,
            "line_seed": NULL_SEED,
            "plane_seed": NULL_SEED + PLANE_SEED_OFFSET,
            "sections": {"singular_read": svd_read, "eigen_read": eig_read_bases},
            "dictionary": "andyrdt per-token L19 SAE (context/read side, 131072 features)",
        },
        "ans": {
            "D_raw": ans_sae.w_dec.detach().cpu().numpy().T,  # (d, N)
            "Sigma": Sigma_a,
            "label_of": ans_label_of,
            "line_seed": NULL_SEED + 1,
            "plane_seed": NULL_SEED + 1 + PLANE_SEED_OFFSET,
            "sections": {"singular_write": svd_write, "eigen_write": eig_write_bases},
            "dictionary": "#2476 turn-averaged sae_c (answer/write side, 65536 features)",
        },
        "ctx_trained": {
            "D_raw": ctx2_sae.w_dec.detach().cpu().numpy().T,  # (d, N)
            "Sigma": Sigma_c,
            "label_of": None,  # no judged descriptions exist — feature ids only
            "line_seed": NULL_SEED + CTX2_SEED_OFFSET,
            "plane_seed": NULL_SEED + CTX2_SEED_OFFSET + 1,
            "sections": {
                "singular_read_ctxsae": svd_read,
                "eigen_read_ctxsae": eig_read_bases,
            },
            "dictionary": (
                "#2569 leg-4 trained context SAE (matryoshka batchtopk, 65536 features, "
                "k=100, trained on the X19 last-prompt-token context rows the map reads; "
                "no judged descriptions — feature ids only)"
            ),
            "union_idx": ctx2_alive_idx,
            "union_full": ctx2_union_full,
        },
    }

    results: dict = {name: {} for s in sides.values() for name in s["sections"]}
    null_floors: dict = {}
    whitener_diag: dict = {}

    for side_name, S in sides.items():
        D_raw = np.ascontiguousarray(S["D_raw"], dtype=np.float32)
        n_feat = int(D_raw.shape[1])
        D_normed, _ = W2569.normalize_dictionary_columns(D_raw)
        _log(f"[{side_name}] raw dictionary normalized ({n_feat} features)")

        floors = {
            "n_features": n_feat,
            "raw": {
                "line": {
                    "analytic": analytic_max_cos_floor(n_feat, d),
                    "empirical": empirical_max_cos_null_lines(
                        D_normed, args.n_draws, S["line_seed"]
                    ),
                },
                "plane": {
                    "analytic": analytic_max_plane_cos_floor(n_feat, d),
                    "empirical": empirical_max_plane_cos_null(
                        D_normed, args.n_draws, S["plane_seed"]
                    ),
                },
            },
        }
        _log(
            f"[{side_name}] raw nulls: line p95 {floors['raw']['line']['empirical']['p95']:.4f} "
            f"plane p95 {floors['raw']['plane']['empirical']['p95']:.4f}"
        )

        union_idx = S.get("union_idx")
        union_restrict = union_idx is not None and not S.get("union_full", True)
        for sec_name, bases in S["sections"].items():
            stats = entry_cosine_stats(bases, D_normed, m)
            results[sec_name]["raw"] = stats
            if union_restrict:
                ustats = entry_cosine_stats(bases, D_normed[:, union_idx], m)
                for u in ustats:  # remap restricted column ids to dictionary ids
                    u["top_ids"] = [int(union_idx[t]) for t in u["top_ids"]]
                results[sec_name]["raw_union"] = ustats
        _log(f"[{side_name}] raw section stats done")

        for lam_shrink in SHRINK_LAMBDAS:
            tag = f"white_{lam_shrink:g}"
            Ssh, tau = shrunk_covariance(S["Sigma"], lam_shrink)
            Wm = inv_sqrt_psd(Ssh)
            Wm32 = Wm.astype(np.float32)
            whitener_diag.setdefault(side_name, {})[tag] = {
                "shrinkage_lambda": lam_shrink,
                "tau_tr_over_d": tau,
                "whitener_cond": float(np.linalg.cond(Ssh) ** 0.5),
            }
            Dw = Wm32 @ D_raw
            Dw_normed, _ = W2569.normalize_dictionary_columns(Dw)
            del Dw
            _log(f"[{side_name}] {tag}: dictionary whitened")

            floors[tag] = {
                "line": {
                    "empirical": empirical_max_cos_null_lines(
                        Dw_normed, args.n_draws, S["line_seed"], whitener32=Wm32
                    )
                },
                "plane": {
                    "empirical": empirical_max_plane_cos_null(
                        Dw_normed, args.n_draws, S["plane_seed"], whitener32=Wm32
                    )
                },
            }
            for sec_name, bases in S["sections"].items():
                wbases = whiten_bases(bases, Wm)
                results[sec_name][tag] = entry_cosine_stats(wbases, Dw_normed, m)
                if union_restrict:
                    ustats = entry_cosine_stats(wbases, Dw_normed[:, union_idx], m)
                    for u in ustats:
                        u["top_ids"] = [int(union_idx[t]) for t in u["top_ids"]]
                    results[sec_name][tag + "_union"] = ustats
            _log(
                f"[{side_name}] {tag}: sections + nulls done "
                f"(line p95 {floors[tag]['line']['empirical']['p95']:.4f}, "
                f"plane p95 {floors[tag]['plane']['empirical']['p95']:.4f})"
            )
            del Dw_normed, Wm, Wm32
        if union_idx is not None:
            floors["alive_union"] = {
                "n_union": int(len(union_idx)),
                "covers_full_dictionary": bool(S.get("union_full", True)),
                "note": (
                    "alive union covers the FULL dictionary, so the alive-union-restricted "
                    "column equals the full-dictionary column and is not duplicated"
                    if S.get("union_full", True)
                    else "restricted stats stored under the *_union metric keys"
                ),
            }
        null_floors[side_name] = floors
        del D_raw, D_normed

    # ── encoder pass (raw unit basis vectors, v1 convention) ─────────────────
    def encoder_reports(sae, bases: list[np.ndarray], label_of) -> list[dict]:
        R = np.concatenate(bases, axis=0).astype(np.float32)
        enc = W2569.encoder_pass(sae, R)  # (M, N)
        out = []
        r = 0
        for B in bases:
            axes = []
            for a in range(B.shape[0]):
                row = enc[r + a]
                fired = np.flatnonzero(row > 0)
                fired = fired[np.argsort(-row[fired])][:m]
                axes.append(
                    {
                        "axis": "real" if a == 0 else "imag",
                        "n_fired": int((row > 0).sum()),
                        "top_fired": [
                            {
                                "feat_id": int(f),
                                "act": float(row[f]),
                                **({"label": label_of(int(f))[:140]} if label_of else {}),
                            }
                            for f in fired
                        ],
                    }
                )
            out.append({"axes": axes})
            r += B.shape[0]
        return out

    enc_reports = {
        "singular_write": encoder_reports(ans_sae, svd_write, ans_label_of),
        "eigen_write": encoder_reports(ans_sae, eig_write_bases, ans_label_of),
        "singular_read_ctxsae": encoder_reports(ctx2_sae, svd_read, None),
        "eigen_read_ctxsae": encoder_reports(ctx2_sae, eig_read_bases, None),
    }
    _log("encoder pass done")

    # ── v1 crosscheck ────────────────────────────────────────────────────────
    v1_path = args.out_dir / f"sae_dashboards_L{LAYER}.json"
    v1 = json.loads(v1_path.read_text())
    v1_cross = {"v1_path": str(v1_path)}
    for sec in ("singular_read", "singular_write"):
        v1_rows = v1["sections"][sec]["directions"]
        diffs = [
            abs(results[sec]["raw"][j]["max"] - v1_rows[j]["max_abs_cos"])
            for j in range(min(k, len(v1_rows)))
        ]
        v1_cross[sec] = {"max_abs_diff_vs_v1": float(max(diffs))}
    for sec in ("eigen_read", "eigen_write"):
        v1_rows = v1["sections"][sec]["directions"]
        diffs = []
        for j, e in enumerate(eig_entries):
            r1 = e["ranks"][0]
            if r1 <= len(v1_rows) and e["kind"] == "plane":
                diffs.append(abs(results[sec]["raw"][j]["axis1_max"] - v1_rows[r1 - 1]["max_abs_cos"]))
        v1_cross[sec] = {
            "n_compared": len(diffs),
            "max_abs_diff_realpart_vs_v1": float(max(diffs)) if diffs else None,
        }
    _log(f"v1 crosscheck: {json.dumps({s: v1_cross[s] for s in v1_cross if s != 'v1_path'})}")

    # ── assemble per-direction records ───────────────────────────────────────
    def build_directions(sec_name: str, side_name: str, kind_meta: list[dict], label_of) -> list[dict]:
        fl = null_floors[side_name]

        def _lab(fid: int) -> dict:
            return {"label": label_of(fid)[:140]} if label_of else {}

        recs = []
        for j, meta in enumerate(kind_meta):
            kind = meta["kind"]
            raw = results[sec_name]["raw"][j]
            w2 = results[sec_name]["white_0.01"][j]
            w3 = results[sec_name]["white_0.001"][j]
            raw_fl = fl["raw"][kind]
            rec = {
                **{kk: vv for kk, vv in meta.items() if kk != "kind"},
                "kind": kind,
                "raw": {
                    "max_cos": raw["max"],
                    "exceeds_analytic_floor": bool(raw["max"] > raw_fl["analytic"]),
                    "exceeds_empirical_p95": bool(raw["max"] > raw_fl["empirical"]["p95"]),
                    "top_features": [
                        {
                            "feat_id": fid,
                            "value": val,
                            **(
                                {"imag_axis_share": raw["axis2_share_top"][t]}
                                if "axis2_share_top" in raw
                                else {}
                            ),
                            **_lab(fid),
                        }
                        for t, (fid, val) in enumerate(zip(raw["top_ids"], raw["top_vals"]))
                    ],
                },
                "white_1e-2": {
                    "max_cos": w2["max"],
                    "exceeds_empirical_p95": bool(
                        w2["max"] > fl["white_0.01"][kind]["empirical"]["p95"]
                    ),
                    "top_feature_changed_vs_raw": bool(w2["top_ids"][0] != raw["top_ids"][0]),
                    "top_features": [
                        {"feat_id": fid, "value": val, **_lab(fid)}
                        for fid, val in zip(w2["top_ids"], w2["top_vals"])
                    ],
                },
                "white_1e-3": {
                    "max_cos": w3["max"],
                    "exceeds_empirical_p95": bool(
                        w3["max"] > fl["white_0.001"][kind]["empirical"]["p95"]
                    ),
                    "top_feature_changed_vs_raw": bool(w3["top_ids"][0] != raw["top_ids"][0]),
                    "top_feature_id": int(w3["top_ids"][0]),
                },
            }
            if kind == "plane":
                rec["raw"]["v1_real_part_only_max_cos"] = raw["axis1_max"]
                rec["raw"]["imag_axis_share_at_max"] = raw["axis2_share_top"][0]
            recs.append(rec)
        return recs

    sing_meta = [
        {"rank": j + 1, "sigma": float(sig[j]), "kind": "line"} for j in range(k)
    ]
    eig_meta = [
        {
            "rank": j + 1,
            "abs_lambda": e["abs_lambda"],
            "lam_re": e["lam_re"],
            "lam_im": e["lam_im"],
            "eig_ranks_in_lambda_order": e["ranks"],
            "kind": e["kind"],
        }
        for j, e in enumerate(eig_entries)
    ]
    eig_meta_read = [
        {**mm, "im_frac_eigvec": eig_entries[j]["im_frac_read"]} for j, mm in enumerate(eig_meta)
    ]
    eig_meta_write = [
        {**mm, "im_frac_eigvec": eig_entries[j]["im_frac_write"]} for j, mm in enumerate(eig_meta)
    ]

    sections = {
        "singular_read": {
            "dictionary": sides["ctx"]["dictionary"],
            "directions": build_directions("singular_read", "ctx", sing_meta, ctx_label_of),
        },
        "singular_write": {
            "dictionary": sides["ans"]["dictionary"],
            "directions": build_directions("singular_write", "ans", sing_meta, ans_label_of),
        },
        "eigen_read": {
            "dictionary": sides["ctx"]["dictionary"],
            "directions": build_directions("eigen_read", "ctx", eig_meta_read, ctx_label_of),
        },
        "eigen_write": {
            "dictionary": sides["ans"]["dictionary"],
            "directions": build_directions("eigen_write", "ans", eig_meta_write, ans_label_of),
        },
        "singular_read_ctxsae": {
            "dictionary": sides["ctx_trained"]["dictionary"],
            "directions": build_directions(
                "singular_read_ctxsae", "ctx_trained", sing_meta, None
            ),
        },
        "eigen_read_ctxsae": {
            "dictionary": sides["ctx_trained"]["dictionary"],
            "directions": build_directions(
                "eigen_read_ctxsae", "ctx_trained", eig_meta_read, None
            ),
        },
    }
    for sec in ("singular_write", "eigen_write", "singular_read_ctxsae", "eigen_read_ctxsae"):
        for rec, er in zip(sections[sec]["directions"], enc_reports[sec]):
            rec["encoder_pass"] = er["axes"]

    # ── summary ──────────────────────────────────────────────────────────────
    def summarize(sec_name: str) -> dict:
        dirs = sections[sec_name]["directions"]
        raw_max = [r["raw"]["max_cos"] for r in dirs]
        w_max = [r["white_1e-2"]["max_cos"] for r in dirs]
        out = {
            "n_directions": len(dirs),
            "raw": {
                "median_max_cos": float(np.median(raw_max)),
                "max_max_cos": float(np.max(raw_max)),
                "n_above_empirical_p95": int(sum(r["raw"]["exceeds_empirical_p95"] for r in dirs)),
            },
            "white_1e-2": {
                "median_max_cos": float(np.median(w_max)),
                "max_max_cos": float(np.max(w_max)),
                "n_above_empirical_p95": int(
                    sum(r["white_1e-2"]["exceeds_empirical_p95"] for r in dirs)
                ),
                "n_top_feature_changed_vs_raw": int(
                    sum(r["white_1e-2"]["top_feature_changed_vs_raw"] for r in dirs)
                ),
            },
            "white_1e-3": {
                "n_top_feature_changed_vs_raw": int(
                    sum(r["white_1e-3"]["top_feature_changed_vs_raw"] for r in dirs)
                ),
            },
        }
        planes = [r for r in dirs if r["kind"] == "plane"]
        if planes:
            uplift = [
                r["raw"]["max_cos"] - r["raw"]["v1_real_part_only_max_cos"] for r in planes
            ]
            out["planes"] = {
                "n_planes": len(planes),
                "median_im_frac_eigvec": float(np.median([r["im_frac_eigvec"] for r in planes])),
                "median_imag_axis_share_at_max": float(
                    np.median([r["raw"]["imag_axis_share_at_max"] for r in planes])
                ),
                "median_plane_minus_realpart_max_cos": float(np.median(uplift)),
                "max_plane_minus_realpart_max_cos": float(np.max(uplift)),
            }
        return out

    summary = {sec: summarize(sec) for sec in sections}

    # ── write JSON ───────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    payload_json = {
        "regime": {
            "regime_version": 1,
            "layer": LAYER,
            "top_k": k,
            "top_m": m,
            "n_draws": int(args.n_draws),
            "null_seeds": {
                "ctx_line": NULL_SEED,
                "ans_line": NULL_SEED + 1,
                "ctx_plane": NULL_SEED + PLANE_SEED_OFFSET,
                "ans_plane": NULL_SEED + 1 + PLANE_SEED_OFFSET,
            },
            "shrinkage_lambdas": list(SHRINK_LAMBDAS),
            "eigen_convention": (
                "right eigenvectors from scipy.linalg.eig (fp64); left rows = inv(V); "
                "conjugate pairs collapsed to one real invariant 2-plane spanned by "
                "(Re w, Im w), orthonormalized real-axis-first; real eigenvalues = 1-D"
            ),
            "plane_cosine": "||P^T d||_2 for unit decoder direction d, P = (d, 2) orthonormal",
        },
        "reference_checks": checks,
        "whitening": {
            "conversion_note": MOMENT_CONVERSION_NOTE,
            "sigma_c": sig_c_doc,
            "sigma_a": sig_a_doc,
            "whitener_diag": whitener_diag,
        },
        "null_floors": null_floors,
        "sections": sections,
        "summary": summary,
        "v1_crosscheck": v1_cross,
        "dictionaries": {name: S["dictionary"] for name, S in sides.items()},
        "sae_ctx_provenance": {
            "dir": str(args.sae_ctx_dir),
            "trained_on": str(ctx2_cfg.get("trained_on")),
            "dict_size": int(ctx2_cfg["dict_size"]),
            "k": int(ctx2_cfg["k"]),
            "threshold": float(ctx2_cfg["threshold"]),
            "alive_union": {
                "n_union": int(len(ctx2_alive_idx)),
                "covers_full_dictionary": ctx2_union_full,
            },
        },
        "label_sources": {
            "ctx": ctx_label_doc,
            "ans": str(ans_desc_path),
            "ctx_trained": "none — no judged descriptions exist (feature ids only)",
        },
        "metadata": {
            "git_commit": _git_commit(WORKTREE_ROOT),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(time.time() - t_start, 1),
            "threads": args.threads,
            "max_rss_gb": round(_rss_gb(), 2),
            "script": "scripts/issue2569_eigen_dashboards_v2.py",
        },
    }
    jpath = args.out_dir / f"sae_dashboards_v2_L{LAYER}.json"
    jpath.write_text(json.dumps(payload_json, indent=1))
    _log(f"wrote {jpath} ({jpath.stat().st_size / 1e6:.1f} MB)")

    # ── markdown ─────────────────────────────────────────────────────────────
    write_markdown(args.out_dir / f"sae_dashboards_v2_L{LAYER}.md", payload_json)
    _log("wrote markdown")

    # ── figure ───────────────────────────────────────────────────────────────
    write_figure(args.fig_dir, payload_json)
    _log("wrote figure")
    _log(f"ALL DONE in {time.time() - t_start:.0f}s")


def _fmt_feat(f: dict) -> str:
    if "label" not in f:  # trained ctx SAE: no descriptions exist — ids only
        return f"**{f['feat_id']}** ({f['value']:+.2f})"
    lab = str(f.get("label") or "(no description)")[:100]
    return f"**{f['feat_id']}** ({f['value']:+.2f}): {lab}"


def write_markdown(path: Path, pj: dict) -> None:
    """Human-readable v2 dashboard tables (top-16 rows per section)."""
    lines = [
        "# Two-sided SAE dashboards v2 (L19 map): eigen 2-planes + whitened cosines",
        "",
        "**Plane cosine** (used for every collapsed conjugate eigen pair): for a unit decoder",
        "direction d and an orthonormal basis P of the eigenvector's real invariant 2-plane",
        "(spanned by the real and imaginary parts of the complex eigenvector), the plane cosine",
        "is ||P^T d|| — the cosine between d and its projection onto the plane. For a 1-D",
        "direction (real eigenvalue, or any singular direction) it reduces to |cos|.",
        "",
        "v1 (`sae_dashboards_L19.json`) read complex eigenvectors through the normalized real",
        "part only; the `raw max cos (v1)` column shows that older value in parentheses for",
        "eigen rows. `whitened max cos` uses the side-matched covariance from the P-B moments",
        f"(n_pool {pj['whitening']['sigma_c']['n_rows']:,} rows), shrinkage lambda 1e-2.",
        "`im share` is the fraction of the squared plane cosine at the top feature carried by",
        "the imaginary axis (what v1 dropped). Directions are flagged against kind-matched",
        "empirical p95 null floors (random unit directions for 1-D, random 2-planes for planes).",
        "",
        "Read-side directions are dashboarded against TWO dictionaries: the andyrdt per-token",
        "L19 SAE (131,072 features, judged descriptions where available) and the #2569 leg-4",
        "context SAE trained on the very X19 last-prompt-token context rows the map reads",
        "(65,536 features, k=100; NO descriptions exist for these features, so those sections",
        "report feature ids only, plus an encoder-pass companion).",
        "",
        f"Reference checks: rho {pj['reference_checks']['rho']['value']:.6f}, kappa(V)",
        f"{pj['reference_checks']['kappa_v']['value']:.1f}, {pj['reference_checks']['n_complex_pairs']['value']}",
        f"complex pairs / {pj['reference_checks']['n_real_eigs']['value']} real eigenvalues,",
        f"biorthogonality max error {pj['reference_checks']['biortho_max_err']['value']:.1e} — all",
        "match the banked factor artifact.",
        "",
    ]
    titles = {
        "singular_read": "Singular read (left singular vectors u_i vs andyrdt context SAE)",
        "singular_write": "Singular write (right singular vectors v_i vs answer SAE)",
        "eigen_read": "Eigen read (right eigenvectors vs andyrdt context SAE, conjugate pairs collapsed)",
        "eigen_write": "Eigen write (left eigenvector rows vs answer SAE, conjugate pairs collapsed)",
        "singular_read_ctxsae": "Singular read vs TRAINED context SAE (grain-matched; feature ids only)",
        "eigen_read_ctxsae": "Eigen read vs TRAINED context SAE (grain-matched; feature ids only)",
    }
    for sec, title in titles.items():
        fl = pj["null_floors"][SECTION_SIDE[sec]]
        eigsec = sec.startswith("eigen")
        lines += [f"## {title}", ""]
        lines += [
            f"Null floors (empirical p95): raw line {fl['raw']['line']['empirical']['p95']:.3f}, "
            f"raw plane {fl['raw']['plane']['empirical']['p95']:.3f}, "
            f"whitened (1e-2) line {fl['white_0.01']['line']['empirical']['p95']:.3f}, "
            f"whitened (1e-2) plane {fl['white_0.01']['plane']['empirical']['p95']:.3f}.",
            "",
        ]
        real_part_tag = " (v1 real-part)" if eigsec and not sec.endswith("_ctxsae") else (
            " (real-part-only)" if eigsec else ""
        )
        ctx2sec = sec.endswith("_ctxsae")
        head = (
            "| rank | " + ("|λ|" if eigsec else "σ") + " | raw max cos" + real_part_tag
            + " | whitened max cos (1e-2) | im share | top-3 features |"
            + (" enc n_fired |" if ctx2sec else "")
        )
        lines += [head, "|---|---|---|---|---|---|" + ("---|" if ctx2sec else "")]
        for r in pj["sections"][sec]["directions"][:16]:
            mag = r.get("abs_lambda", r.get("sigma"))
            raw = r["raw"]["max_cos"]
            rawtxt = f"{raw:.3f}"
            if eigsec and r["kind"] == "plane":
                rawtxt += f" ({r['raw']['v1_real_part_only_max_cos']:.3f})"
            elif eigsec:
                rawtxt += " (real eigenvalue)"
            imtxt = (
                f"{r['raw']['imag_axis_share_at_max']:.2f}" if r["kind"] == "plane" else "—"
            )
            star = "*" if r["raw"]["exceeds_empirical_p95"] else ""
            wstar = "*" if r["white_1e-2"]["exceeds_empirical_p95"] else ""
            feats = "<br>".join(_fmt_feat(f) for f in r["raw"]["top_features"][:3])
            enc_cell = ""
            if ctx2sec:
                nf = "/".join(str(a["n_fired"]) for a in r.get("encoder_pass", []))
                enc_cell = f" {nf} |"
            lines.append(
                f"| {r['rank']} | {mag:.3f} | {rawtxt}{star} | "
                f"{r['white_1e-2']['max_cos']:.3f}{wstar} | {imtxt} | {feats} |" + enc_cell
            )
        lines += ["", "`*` = above the kind-matched empirical p95 null floor.", ""]

    lines += ["## Summary", ""]
    lines += [
        "| section | raw median / max | raw > p95 | whitened median / max (1e-2) | whitened > p95 | top feat changed (1e-2) |",
        "|---|---|---|---|---|---|",
    ]
    for sec in titles:
        s = pj["summary"][sec]
        lines.append(
            f"| {sec} | {s['raw']['median_max_cos']:.3f} / {s['raw']['max_max_cos']:.3f} "
            f"| {s['raw']['n_above_empirical_p95']}/{s['n_directions']} "
            f"| {s['white_1e-2']['median_max_cos']:.3f} / {s['white_1e-2']['max_max_cos']:.3f} "
            f"| {s['white_1e-2']['n_above_empirical_p95']}/{s['n_directions']} "
            f"| {s['white_1e-2']['n_top_feature_changed_vs_raw']}/{s['n_directions']} |"
        )
    # read-side dictionary comparison: andyrdt per-token vs the grain-matched trained SAE
    lines += [
        "",
        "### Read-side dictionary comparison (andyrdt per-token vs context SAE trained on v_C)",
        "",
        "The trained context SAE is grain-matched: it was fit on the very X19 last-prompt-token",
        "context states the map reads, whereas andyrdt is per-token over generic text. Its",
        "features have no descriptions (ids only). The alive union covers all "
        f"{pj['null_floors']['ctx_trained']['alive_union']['n_union']:,} features, so the"
        " alive-union-restricted column equals the full-dictionary column and is omitted.",
        "",
        "| directions | andyrdt raw med/max | >p95 | trained raw med/max | >p95 | andyrdt whitened med/max | trained whitened med/max |",
        "|---|---|---|---|---|---|---|",
    ]
    for base_sec, ctx2_sec, nm in (
        ("singular_read", "singular_read_ctxsae", "singular read"),
        ("eigen_read", "eigen_read_ctxsae", "eigen read"),
    ):
        a, c = pj["summary"][base_sec], pj["summary"][ctx2_sec]
        lines.append(
            f"| {nm} | {a['raw']['median_max_cos']:.3f} / {a['raw']['max_max_cos']:.3f} "
            f"| {a['raw']['n_above_empirical_p95']}/{a['n_directions']} "
            f"| {c['raw']['median_max_cos']:.3f} / {c['raw']['max_max_cos']:.3f} "
            f"| {c['raw']['n_above_empirical_p95']}/{c['n_directions']} "
            f"| {a['white_1e-2']['median_max_cos']:.3f} / {a['white_1e-2']['max_max_cos']:.3f} "
            f"| {c['white_1e-2']['median_max_cos']:.3f} / {c['white_1e-2']['max_max_cos']:.3f} |"
        )
    for sec in ("eigen_read", "eigen_write", "eigen_read_ctxsae"):
        p = pj["summary"][sec].get("planes")
        if p:
            lines += [
                "",
                f"**{sec} planes** ({p['n_planes']}/32): median eigenvector imaginary mass "
                f"{p['median_im_frac_eigvec']:.2f}; median imaginary-axis share of the max plane "
                f"cosine {p['median_imag_axis_share_at_max']:.2f}; plane read raised the max cosine "
                f"over the v1 real-part read by median {p['median_plane_minus_realpart_max_cos']:+.3f} "
                f"(max {p['max_plane_minus_realpart_max_cos']:+.3f}).",
            ]
    lines += [
        "",
        "Whitening: " + pj["whitening"]["conversion_note"],
        "",
        f"Generated by `scripts/issue2569_eigen_dashboards_v2.py` at "
        f"{pj['metadata']['timestamp_utc']} (commit `{pj['metadata']['git_commit'][:12]}`).",
        "",
    ]
    path.write_text("\n".join(lines))


def write_figure(fig_dir: Path, pj: dict) -> None:
    """2x2 panel figure: raw vs whitened max cosine per direction + p95 floors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {
        "singular_read": "Singular read (context SAE)",
        "singular_write": "Singular write (answer SAE)",
        "eigen_read": "Eigen read (context SAE)",
        "eigen_write": "Eigen write (answer SAE)",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    for ax, (sec, title) in zip(axes.ravel(), titles.items()):
        dirs = pj["sections"][sec]["directions"]
        x = np.arange(1, len(dirs) + 1)
        raw = [r["raw"]["max_cos"] for r in dirs]
        wht = [r["white_1e-2"]["max_cos"] for r in dirs]
        wd = 0.4
        ax.bar(x - wd / 2, raw, wd, label="raw max cosine", color="#4477aa")
        ax.bar(x + wd / 2, wht, wd, label="whitened max cosine (λ=1e-2)", color="#ee6677")
        side = "ctx" if "read" in sec else "ans"
        kind = "plane" if sec.startswith("eigen") else "line"
        fl = pj["null_floors"][side]
        ax.axhline(
            fl["raw"][kind]["empirical"]["p95"],
            ls="--", lw=1.2, color="#4477aa", label=f"raw p95 null ({kind})",
        )
        ax.axhline(
            fl["white_0.01"][kind]["empirical"]["p95"],
            ls="--", lw=1.2, color="#ee6677", label=f"whitened p95 null ({kind})",
        )
        if sec.startswith("eigen"):
            v1 = [
                r["raw"].get("v1_real_part_only_max_cos", r["raw"]["max_cos"]) for r in dirs
            ]
            ax.plot(
                x, v1, "kv", ms=4, ls="none", label="v1 real-part-only value",
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("direction rank" + (" (|λ| order, pairs collapsed)" if sec.startswith("eigen") else " (σ order)"))
        ax.set_ylabel("max cosine over dictionary")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0.2, len(dirs) + 0.8)
    fig.suptitle(
        "L19 map SAE dashboards v2: raw vs side-matched whitened max cosines (top-32 directions)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"leg1_sae_dashboards_v2.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
