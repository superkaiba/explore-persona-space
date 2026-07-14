#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (σ, λ, Δ, →, ×, ‖·‖) in scientific docstrings + log messages.
"""Issue #813 free-analysis — rank/spectrum of the averaged vs per-example map (L14).

User-requested 0-GPU inline free analysis (interactive chat, 2026-07-05). Answers:
is the question-AVERAGED context→answer map a lower-rank object than the
PER-EXAMPLE (single (context, question)) map, when BOTH are fit on the SAME
corpus in the FULL 3584-dim output space (no basis cap)? #813's per-example
round compared the grains only inside a shared rank-50 target basis, which
forecloses a rank read; #779's stage-2 averaged-map comparison is
corpus-confounded. This is the corpus-matched, basis-uncapped read.

Structural fact: the averaged map at n=50 contexts can never exceed rank 49
(the standardized design Xn has a zero row-sum ⇒ rank ≤ n−1), so the raw rank
gap is uninformative on its own. The informative reads are:
- (a) the FULL nonzero singular spectrum + energy-concentration stats of each map.
- (b) a matched-n control — a per-example fit at n=50 (one question per context) —
  which separates "averaging changes the object" from "row count changes rank".
- (c) a residual split — how much of W_pe lives inside span(U_avg), and the
  residual map's spectrum.

Recipe (parent ridge, from ``issue722_fit_M._ridge_fit_predict`` /
``issue667_save_maps._ridge_components``): standardize X on OWN train stats
(std, ddof=0, +1e-9), ridge target UNCENTERED (the committed maps store
``output_centered=False`` — we match that exactly so the reproduction gate is
meaningful), closed-form dual ridge W = Xn^T (G+λI)^-1 Y with G = Xn Xn^T.
PRIMARY λ = 1e3 for BOTH grains (the parent-realized L14 choice — PRESS
saturated the grid top 1e3 in all fold-fits, both grains — so a shared 1e3
kills differential shrinkage). SENSITIVITY: each grain's own GCV-chosen λ over
the parent 6-λ grid, headline stats recomputed there.

Factored spectrum (never densify the 3584×3584 W): with K = G = B^T B (B = Xn^T,
d×n) and the dual coefficients C = (G+λI)^-1 Y (n×d), the nonzero eigenvalues of
W W^T equal those of (C C^T) K, symmetrized as K^{1/2}(C C^T)K^{1/2}. Using the
eigh G = Q diag(e) Q^T and W_yy = Q^T (Y Y^T) Q, this reduces to the eigenvalues
of the n×n symmetric matrix diag(√e/(e+λ)) · W_yy · diag(√e/(e+λ)) — TWO big
n²·d GEMMs per (cell, arm) (G and Y Y^T), everything else n×n / n³.

Gates (fail loud): committed-map reproduction (recompute the averaged L14 map at
the committed λ, project into the committed top-64 basis, ≤2% rel vs the
persisted maps/ NPZ), batched-vs-dense spectrum equivalence (one n=50 case,
dense SVD vs factored, ≤1e-5 rel — five-nines agreement; the tail residual is
float-arithmetic noise), shape asserts on every tensor load.

Persistence: per-cell JSON the moment the cell completes (atomic tmp+replace),
regime-keyed resume. Final phase consolidates + emits figures via paper-plots.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue658_fit_predictors as fit658  # noqa: E402  (parent λ grid)
import torch  # noqa: E402  (after load_dotenv so thread-cap setdefaults apply)

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

logger = logging.getLogger("issue813.rank_spectrum")

DATA_REPO = "superkaiba1/explore-persona-space-data"
EXPERIMENT_NAME = "issue813_mapchange_substrate"
# The parent extraction wave's pinned revision (a later force-push to HF main
# cannot silently swap the inputs under this analysis).
HF_REVISION = "b0d30307c1671cad575928e5abf5253c0c849dee"
BEHAVIORS = ("em", "fact", "sycophancy", "marker")
SUBSTRATES = ("generic", "elicit", "mix")
HEADLINE_LAYER = 14  # frozen (#651/#658/#813); the ONLY layer with per-question rows
N_LAYERS = 28
HIDDEN = 3584
TARGET_DIM = 64  # committed-map top-64 basis dim (reproduction gate only)
SEED = 42  # matched-n draws (parent convention)
LAMBDAS = [float(x) for x in fit658.RIDGE_LAMBDAS]  # [1e-2, 1e-1, 1, 10, 100, 1000]
LAMBDA_PRIMARY = 1000.0  # parent-realized L14 choice, BOTH grains
ARMS = ("base", "trained")
CUM_K = (1, 2, 5, 10, 20, 49, 50, 100, 200, 500)
REGIME_VERSION = 1

OUT_SUBDIR = "eval_results/issue_813/per_example_vs_averaged"
CELL_SUBDIR = f"{OUT_SUBDIR}/rank_spectrum"
CONSOLIDATED = f"{OUT_SUBDIR}/rank_spectrum_L14.json"

BEH_LABEL = {
    "em": "emergent misalignment",
    "fact": "fact recall",
    "sycophancy": "sycophancy",
    "marker": "marker",
}
SUB_LABEL = {"generic": "generic", "elicit": "elicited", "mix": "mixed"}

_SUMMARY_KEYS = ("c_C_base", "c_C_trained", "v_A_base", "v_A_trained", "context_ids", "families")
_PQ_KEYS = (
    "c_C_base",
    "c_C_trained",
    "v_A_base",
    "v_A_trained",
    "row_context_index",
    "row_question_index",
    "context_ids",
    "families",
    "headline_layer",
)
_MAP_KEYS = (
    "W_M0",
    "W_Mplus",
    "pca_basis",
    "input_mean_C0",
    "input_std_C0",
    "input_mean_Cplus",
    "input_std_Cplus",
    "lambda_M0",
    "lambda_Mplus",
)


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _require_npz_keys(path: Path | str, npz, required_keys) -> None:
    """Fail loud BEFORE any keyed read when an NPZ is missing required keys."""
    present = list(getattr(npz, "files", None) or npz.keys())
    missing = [k for k in required_keys if k not in present]
    if missing:
        raise KeyError(
            f"NPZ schema preflight FAILED for {path}: missing keys {missing} "
            f"(present: {sorted(present)})"
        )


def _regime(matched_n_draws: int) -> dict:
    """Exact-regime resume key — a cached cell JSON is reused IFF its stored regime == this."""
    return {
        "regime_version": REGIME_VERSION,
        "headline_layer": HEADLINE_LAYER,
        "hf_revision": HF_REVISION,
        "lambda_primary": LAMBDA_PRIMARY,
        "lambdas_grid": LAMBDAS,
        "matched_n_draws": matched_n_draws,
        "recipe": "standardize-X(std,ddof0,+1e-9); ridge-target-UNCENTERED; full-3584-output",
        "seed": SEED,
    }


# ── data loading (parent load_cell recipe) ──────────────────────────────────


def _hf_fetch(rel: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    if dest.exists():
        return dest
    local = hf_hub_download(DATA_REPO, rel, repo_type="dataset", revision=HF_REVISION)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        dest.unlink()
    dest.symlink_to(Path(local).resolve())
    logger.info("[phase=fetch] %s", rel)
    return dest


def load_cell(behavior: str, substrate: str, dl_root: Path) -> dict:
    """Load one cell's summary averages, per-question rows, and committed map (L14)."""
    sroot = dl_root / behavior / substrate
    spath = _hf_fetch(
        f"{EXPERIMENT_NAME}/reduced/{behavior}/{substrate}/summary.npz", sroot / "summary.npz"
    )
    pqpath = _hf_fetch(
        f"{EXPERIMENT_NAME}/reduced/{behavior}/{substrate}/per_question_L{HEADLINE_LAYER}.npz",
        sroot / f"per_question_L{HEADLINE_LAYER}.npz",
    )
    mpath = _hf_fetch(
        f"{EXPERIMENT_NAME}/maps/{behavior}/{substrate}/L{HEADLINE_LAYER}.npz",
        sroot / f"map_L{HEADLINE_LAYER}.npz",
    )

    s = np.load(spath, allow_pickle=True)
    _require_npz_keys(spath, s, _SUMMARY_KEYS)
    c0s = np.asarray(s["c_C_base"], dtype=np.float64)
    assert c0s.ndim == 3 and c0s.shape[1:] == (N_LAYERS, HIDDEN), c0s.shape
    lay = HEADLINE_LAYER
    avg_c = {"base": c0s[:, lay], "trained": np.asarray(s["c_C_trained"], np.float64)[:, lay]}
    avg_v = {
        "base": np.asarray(s["v_A_base"], np.float64)[:, lay],
        "trained": np.asarray(s["v_A_trained"], np.float64)[:, lay],
    }
    ctx_ids = [str(x) for x in s["context_ids"]]
    families = [str(x) for x in s["families"]]
    n_ctx = len(ctx_ids)
    for arm in ARMS:
        assert avg_c[arm].shape == (n_ctx, HIDDEN), avg_c[arm].shape
        assert avg_v[arm].shape == (n_ctx, HIDDEN), avg_v[arm].shape

    d = np.load(pqpath, allow_pickle=True)
    _require_npz_keys(pqpath, d, _PQ_KEYS)
    got_layer = int(np.asarray(d["headline_layer"]))
    assert got_layer == HEADLINE_LAYER, (got_layer, HEADLINE_LAYER)
    pq_c = {
        "base": np.asarray(d["c_C_base"], np.float64),
        "trained": np.asarray(d["c_C_trained"], np.float64),
    }
    pq_v = {
        "base": np.asarray(d["v_A_base"], np.float64),
        "trained": np.asarray(d["v_A_trained"], np.float64),
    }
    n = pq_c["base"].shape[0]
    for arm in ARMS:
        assert pq_c[arm].shape == (n, HIDDEN), pq_c[arm].shape
        assert pq_v[arm].shape == (n, HIDDEN), pq_v[arm].shape
    row_ctx = np.asarray(d["row_context_index"], dtype=np.int64)
    q_idx = np.asarray(d["row_question_index"], dtype=np.int64)
    full_ctx_ids = [str(x) for x in d["context_ids"]]
    kept_pos = {cid: i for i, cid in enumerate(ctx_ids)}
    groups = np.empty(n, dtype=np.int64)
    for r in range(n):
        cid = full_ctx_ids[int(row_ctx[r])]
        if cid not in kept_pos:
            raise RuntimeError(
                f"{behavior}/{substrate}: per-question row {r} context {cid!r} not among "
                "the summary's kept contexts — producer/consumer misalignment"
            )
        groups[r] = kept_pos[cid]

    m = np.load(mpath, allow_pickle=True)
    _require_npz_keys(mpath, m, _MAP_KEYS)
    committed = {k: np.asarray(m[k]) for k in _MAP_KEYS}

    logger.info(
        "[phase=load] %s/%s: F=%d contexts, N=%d pq rows, %d questions",
        behavior,
        substrate,
        n_ctx,
        n,
        len({int(q) for q in q_idx}),
    )
    return {
        "behavior": behavior,
        "substrate": substrate,
        "avg_c": avg_c,
        "avg_v": avg_v,
        "pq_c": pq_c,
        "pq_v": pq_v,
        "groups": groups,
        "ctx_ids": ctx_ids,
        "families": families,
        "committed": committed,
        "n_ctx": n_ctx,
        "n_rows": n,
    }


# ── ridge / spectrum primitives (factored; never densify W) ──────────────────


def _standardize(X: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(dtype=torch.float64)
    mu = Xt.mean(0)
    sd = Xt.std(0, correction=0) + 1e-9
    return (Xt - mu) / sd, mu, sd


def _fit_pieces(Xn: torch.Tensor, Y: torch.Tensor) -> dict:
    """The two big n²·d GEMMs + one eigh, shared across every read for this map.

    Returns G = Xn Xn^T (n×n), its eigh (e, Q), and W_yy = Q^T (Y Y^T) Q (n×n).
    Everything downstream (spectrum at any λ, residual, projections) is n×n / n³.
    """
    Yt = Y if isinstance(Y, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(Y)).double()
    G = Xn @ Xn.t()  # BIG GEMM #1
    e, Q = torch.linalg.eigh(G)
    e = e.clamp(min=0.0)
    YY = Yt @ Yt.t()  # BIG GEMM #2
    W_yy = Q.t() @ YY @ Q
    return {"G": G, "e": e, "Q": Q, "W_yy": W_yy}


def _sigma2(e: torch.Tensor, W_yy: torch.Tensor, lam: float) -> torch.Tensor:
    """σ²(W) desc: eigenvalues of diag(√e/(e+λ)) · W_yy · diag(√e/(e+λ)).

    The standardized design Xn has an exact null direction (1ₙ; zero column
    sums), so G = Xn Xn^T has one structurally-zero eigenvalue that eigh returns
    as float noise (~1e-11) sitting ~13 orders below the real spectrum. The √e/(e+λ)
    form maps it to a small-but-nonzero σ whose cell-dependent size flips the raw
    rank count 49↔50. Zero the scale for eigenvalues below e_max·1e-9 (the huge
    gap makes the cutoff unambiguous) so rank(W) = rank(Xn) ≤ n−1 holds cleanly
    and cell-consistently; the excluded mode carries ≲1e-12 relative energy.
    """
    e_pos = e.clamp(min=0.0)
    keep = e_pos > (e_pos.max() * 1e-9)
    scale = torch.where(keep, torch.sqrt(e_pos) / (e_pos + lam), torch.zeros_like(e_pos))
    S = (scale.unsqueeze(1) * W_yy) * scale.unsqueeze(0)
    S = 0.5 * (S + S.t())  # symmetrize away float asymmetry
    s2 = torch.linalg.eigvalsh(S).clamp(min=0.0)
    return torch.sort(s2, descending=True).values


def _M_pe(e: torch.Tensor, Q: torch.Tensor, W_yy: torch.Tensor, lam: float) -> torch.Tensor:
    """M = C C^T = (G+λI)^-1 (Y Y^T) (G+λI)^-1 = Q diag(1/(e+λ)) W_yy diag(1/(e+λ)) Q^T (n×n)."""
    inv = 1.0 / (e + lam)
    inner = (inv.unsqueeze(1) * W_yy) * inv.unsqueeze(0)
    return Q @ inner @ Q.t()


def _gcv_lambda(e: np.ndarray, w_yy_diag: np.ndarray, n: int) -> float:
    """GCV-select λ over the parent grid from the shared eigh (dual hat-matrix form)."""
    best_gcv, best_lam = np.inf, LAMBDAS[0]
    for lam in LAMBDAS:
        resid_sq = float(np.sum((lam**2) / ((e + lam) ** 2) * w_yy_diag))  # ‖(I−H)Y‖_F²
        tr_h = float(np.sum(e / (e + lam)))
        denom = n * (1.0 - tr_h / n) ** 2
        gcv = resid_sq / denom if denom > 0 else np.inf
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, lam
    return best_lam


def _spectrum_stats(sv: np.ndarray) -> dict:
    """Energy-concentration stats from descending singular values (nonzero-filtered)."""
    sv = np.asarray(sv, dtype=np.float64)
    smax = float(sv[0]) if sv.size else 0.0
    # Numerical-zero floor (1e-9·σ₁): the structural-zero mode is already removed
    # at the G-eigenvalue source in _sigma2, so this only trims genuine float
    # zeros and the count reflects true rank (rank(W) = rank(Xn) ≤ n−1). Raw rank
    # is uninformative here by construction (see the module docstring) — the
    # effective-rank / k50 / k90 / energy-fraction stats are the substantive reads.
    tol = smax * 1e-9
    nz = sv[sv > tol]
    energy = nz**2
    total = float(energy.sum())
    if total <= 0:
        return {
            "n_nonzero": 0,
            "s1": smax,
            "frob_sq": total,
            "cum_energy": {str(k): 0.0 for k in CUM_K},
            "stable_rank": 0.0,
            "participation_ratio": 0.0,
            "k50": 0,
            "k90": 0,
            "top_sv": [],
        }
    cum = np.cumsum(energy) / total

    def frac_at(k: int) -> float:
        return float(cum[min(k, len(cum)) - 1])

    def k_for(frac: float) -> int:
        return int(min(int(np.searchsorted(cum, frac)) + 1, len(cum)))

    return {
        "n_nonzero": int(nz.size),
        "s1": smax,
        "frob_sq": total,
        "cum_energy": {str(k): frac_at(k) for k in CUM_K},
        "stable_rank": total / (smax**2),
        "participation_ratio": float(total**2 / np.sum(energy**2)),
        "k50": k_for(0.5),
        "k90": k_for(0.9),
        "top_sv": [float(x) for x in nz[:600]],
    }


def _colspace_basis(Xn: torch.Tensor, tol_ratio: float = 1e-8) -> torch.Tensor:
    """Orthonormal basis (d × r) for col(W) = col(Xn^T) = rowspace(Xn); r ≤ n−1."""
    _, s, Vt = torch.linalg.svd(Xn, full_matrices=False)  # Xn (n,d): U(n,n), s(n), Vt(n,d)
    keep = s > (s[0] * tol_ratio)
    return Vt[keep].t()  # (d, r)


def _rowspace_basis(
    Xn: torch.Tensor,
    e: torch.Tensor,
    Q: torch.Tensor,
    Y: torch.Tensor,
    lam: float,
    tol_ratio: float = 1e-8,
) -> torch.Tensor:
    """Orthonormal basis (d × r) for row(W_avg) = col(W_avg^T) = alpha^T · image(Xn).

    image(Xn) (⊆ R^n) basis = left singular vectors of Xn; alpha = (G+λI)^-1 Y.
    """
    U, s, _ = torch.linalg.svd(Xn, full_matrices=False)  # U (n, n)
    keep = s > (s[0] * tol_ratio)
    Ux = U[:, keep]  # (n, r) image(Xn) basis
    Yt = Y if isinstance(Y, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(Y)).double()
    inv = 1.0 / (e + lam)
    alpha = Q @ (inv.unsqueeze(1) * (Q.t() @ Yt))  # (n, d) = (G+λI)^-1 Y
    F = alpha.t() @ Ux  # (d, r)
    Qr, _ = torch.linalg.qr(F, mode="reduced")
    return Qr  # (d, r)


# ── per-arm reads ────────────────────────────────────────────────────────────


def _arm_reads(cell: dict, arm: str, matched_n_draws: int) -> dict:
    X_avg = _avg_from_pe(cell, arm, "c")  # recomputed from per-question rows (exact corpus match)
    Y_avg = _avg_from_pe(cell, arm, "v")
    Xn_avg, _, _ = _standardize(X_avg)
    Yavg_t = torch.from_numpy(np.ascontiguousarray(Y_avg)).double()
    piece_avg = _fit_pieces(Xn_avg, Yavg_t)

    Xn_pe, _, _ = _standardize(cell["pq_c"][arm])
    Ype_t = torch.from_numpy(np.ascontiguousarray(cell["pq_v"][arm])).double()
    piece_pe = _fit_pieces(Xn_pe, Ype_t)

    F = cell["n_ctx"]
    n_pe = cell["n_rows"]
    gcv_avg = _gcv_lambda(piece_avg["e"].numpy(), np.diag(piece_avg["W_yy"].numpy()), F)
    gcv_pe = _gcv_lambda(piece_pe["e"].numpy(), np.diag(piece_pe["W_yy"].numpy()), n_pe)

    out: dict = {"lambda_gcv_avg": gcv_avg, "lambda_gcv_pe": gcv_pe, "n_ctx": F, "n_rows": n_pe}

    # (a) spectra at PRIMARY λ and each grain's own GCV λ
    for tag, piece, lam in (
        ("averaged_primary", piece_avg, LAMBDA_PRIMARY),
        ("averaged_gcv", piece_avg, gcv_avg),
        ("per_example_primary", piece_pe, LAMBDA_PRIMARY),
        ("per_example_gcv", piece_pe, gcv_pe),
    ):
        sv = torch.sqrt(_sigma2(piece["e"], piece["W_yy"], lam)).numpy()
        out[tag] = _spectrum_stats(sv)

    # (b) matched-n control: 10 draws of a per-example fit at n=F (one question/context)
    out["matched_n"] = _matched_n_control(cell, arm, matched_n_draws)

    # (c) residual split (PRIMARY λ): energy of W_pe inside col(W_avg) + residual spectrum
    out["residual"] = _residual_split(cell, arm, Xn_avg, Xn_pe, piece_avg, piece_pe, Yavg_t, Ype_t)
    return out


def _avg_from_pe(cell: dict, arm: str, which: str) -> np.ndarray:
    src = cell["pq_c"][arm] if which == "c" else cell["pq_v"][arm]
    groups = cell["groups"]
    F = cell["n_ctx"]
    out = np.zeros((F, HIDDEN), dtype=np.float64)
    for f in range(F):
        rows = src[groups == f]
        assert rows.shape[0] >= 1, (cell["behavior"], cell["substrate"], f)
        out[f] = rows.mean(axis=0)
    return out


def _matched_n_control(cell: dict, arm: str, n_draws: int) -> dict:
    rng = np.random.default_rng(SEED)
    groups = cell["groups"]
    F = cell["n_ctx"]
    ctx_rows = [np.where(groups == f)[0] for f in range(F)]
    keys = ("stable_rank", "participation_ratio", "k50", "k90")
    acc = {k: [] for k in keys}
    cum_acc = {str(k): [] for k in CUM_K}
    for _ in range(n_draws):
        idx = np.array([rng.choice(rows) for rows in ctx_rows], dtype=np.int64)
        Xd = cell["pq_c"][arm][idx]
        Yd = cell["pq_v"][arm][idx]
        Xn_d, _, _ = _standardize(Xd)
        piece = _fit_pieces(Xn_d, torch.from_numpy(np.ascontiguousarray(Yd)).double())
        sv = torch.sqrt(_sigma2(piece["e"], piece["W_yy"], LAMBDA_PRIMARY)).numpy()
        st = _spectrum_stats(sv)
        for k in keys:
            acc[k].append(st[k])
        for k in CUM_K:
            cum_acc[str(k)].append(st["cum_energy"][str(k)])

    def ms(vals):
        a = np.asarray(vals, dtype=np.float64)
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=0))}

    return {
        "n_draws": n_draws,
        "seed": SEED,
        "lambda": LAMBDA_PRIMARY,
        **{k: ms(acc[k]) for k in keys},
        "cum_energy": {str(k): ms(cum_acc[str(k)]) for k in CUM_K},
    }


def _residual_split(cell, arm, Xn_avg, Xn_pe, piece_avg, piece_pe, Yavg_t, Ype_t) -> dict:
    lam = LAMBDA_PRIMARY
    P_avg = _colspace_basis(Xn_avg)  # (d, r_avg) basis of col(W_avg)
    B_pe = Xn_pe.t()  # (d, n_pe)
    A = P_avg.t() @ B_pe  # (r_avg, n_pe)
    M_pe = _M_pe(piece_pe["e"], piece_pe["Q"], piece_pe["W_yy"], lam)  # (n_pe, n_pe)
    frob_pe = float(torch.sum(_sigma2(piece_pe["e"], piece_pe["W_yy"], lam)))  # ‖W_pe‖_F²
    # output-side energy of W_pe inside col(W_avg)
    ein = float(torch.trace(A @ M_pe @ A.t()))
    energy_in_colspace = ein / frob_pe if frob_pe > 0 else float("nan")

    # residual map W_res = (I − P P^T) W_pe: K_res = G_pe − A^T A, same M_pe
    K_res = piece_pe["G"] - (A.t() @ A)
    K_res = 0.5 * (K_res + K_res.t())
    e_res, Q_res = torch.linalg.eigh(K_res)
    e_res = e_res.clamp(min=0.0)
    # nonzero eigs of W_res W_res^T = eigs of M_pe K_res → K_res^{1/2} M_pe K_res^{1/2}
    W_yy_res = Q_res.t() @ M_pe @ Q_res
    scale = torch.sqrt(e_res)
    S = (scale.unsqueeze(1) * W_yy_res) * scale.unsqueeze(0)
    S = 0.5 * (S + S.t())
    sv_res = torch.sqrt(torch.linalg.eigvalsh(S).clamp(min=0.0))
    sv_res = torch.sort(sv_res, descending=True).values.numpy()
    res_stats = _spectrum_stats(sv_res)

    # input-side (row-space) analogue: fraction of W_pe row-space energy in row(W_avg)
    Q_avg_row = _rowspace_basis(Xn_avg, piece_avg["e"], piece_avg["Q"], Yavg_t, lam)  # (d, r)
    YQ = Ype_t @ Q_avg_row  # (n_pe, r)
    inv = 1.0 / (piece_pe["e"] + lam)
    CQ = piece_pe["Q"] @ (
        inv.unsqueeze(1) * (piece_pe["Q"].t() @ YQ)
    )  # (n_pe, r) = alpha_pe Q_avg_row
    ein_rowspace = float(torch.trace(CQ.t() @ piece_pe["G"] @ CQ))
    energy_in_rowspace = ein_rowspace / frob_pe if frob_pe > 0 else float("nan")

    return {
        "lambda": lam,
        "r_avg_colspace": int(P_avg.shape[1]),
        "frob_sq_pe": frob_pe,
        "energy_in_averaged_colspace": energy_in_colspace,
        "energy_in_averaged_rowspace": energy_in_rowspace,
        "residual_spectrum": res_stats,
    }


# ── gates ────────────────────────────────────────────────────────────────────


def _reproduction_gate(cell: dict) -> dict:
    """Recompute the averaged L14 map at the committed λ, project into the top-64 basis, ≤2% rel."""
    c = cell["committed"]
    basis = np.asarray(c["pca_basis"], dtype=np.float64)  # (64, d)
    rels = {}
    for arm, wkey, lamkey in (
        ("base", "W_M0", "lambda_M0"),
        ("trained", "W_Mplus", "lambda_Mplus"),
    ):
        X = cell["avg_c"][arm]  # the SUMMARY averages the committed map was fit on
        Y = cell["avg_v"][arm]
        lam = float(np.asarray(c[lamkey]))
        Xn, _, _ = _standardize(X)
        Xnt = Xn
        G = Xnt @ Xnt.t()
        Y64 = torch.from_numpy(np.ascontiguousarray(Y @ basis.T)).double()  # (n, 64)
        e, Q = torch.linalg.eigh(G)
        inv = 1.0 / (e.clamp(min=0.0) + lam)
        alpha = Q @ (inv.unsqueeze(1) * (Q.t() @ Y64))  # (n, 64)
        W = (Xnt.t() @ alpha).numpy()  # (d, 64)
        W_ref = np.asarray(c[wkey], dtype=np.float64)  # (d, 64)
        rel = float(np.linalg.norm(W - W_ref) / (np.linalg.norm(W_ref) + 1e-30))
        rels[arm] = rel
    max_rel = max(rels.values())
    ok = max_rel <= 0.02
    if not ok:
        raise RuntimeError(
            f"reproduction gate FAILED for {cell['behavior']}/{cell['substrate']}: "
            f"max rel {max_rel:.3e} > 0.02 vs committed maps/ NPZ ({rels})"
        )
    logger.info(
        "[phase=gate] reproduction PASS %s/%s: rel base=%.2e trained=%.2e (tol 2%%; "
        "full-space fit @ committed λ, projected into committed top-64 basis)",
        cell["behavior"],
        cell["substrate"],
        rels["base"],
        rels["trained"],
    )
    return {
        "pass": True,
        "tol": 0.02,
        "rel": rels,
        "note": "full-space W @ pca_basis.T vs W_M0/W_Mplus",
    }


def _dense_equivalence_gate(cell: dict) -> dict:
    """One n=50 case: dense SVD of the materialized W vs the factored spectrum, ≤1e-5 rel."""
    arm = "base"
    X = _avg_from_pe(cell, arm, "c")
    Y = _avg_from_pe(cell, arm, "v")
    Xn, _, _ = _standardize(X)
    Yt = torch.from_numpy(np.ascontiguousarray(Y)).double()
    piece = _fit_pieces(Xn, Yt)
    lam = LAMBDA_PRIMARY
    sv_fact = torch.sqrt(_sigma2(piece["e"], piece["W_yy"], lam)).numpy()
    inv = 1.0 / (piece["e"] + lam)
    alpha = piece["Q"] @ (inv.unsqueeze(1) * (piece["Q"].t() @ Yt))  # (n, d)
    W_dense = Xn.t() @ alpha  # (d, d) — materialized for ONE small case only
    sv_dense = torch.linalg.svdvals(W_dense).numpy()
    k = min(cell["n_ctx"], (sv_fact > sv_fact[0] * 1e-10).sum())
    smax = float(sv_dense[0])
    rel = float(np.max(np.abs(np.sort(sv_fact)[::-1][:k] - np.sort(sv_dense)[::-1][:k])) / smax)
    # 1e-5 tol: the top energy-bearing σ agree to ~1e-10 between the factored
    # (eigh(G) → S_inner) and dense-SVD paths; the residual is float-arithmetic
    # noise on the small tail modes near the numerical-rank cutoff (cell-specific
    # conditioning — em/generic reads 1.58e-6, marker/mix 6.3e-7). A real
    # factored-math bug would read ~1e-2+, not ~1e-6.
    tol = 1e-5
    ok = rel <= tol
    if not ok:
        raise RuntimeError(
            f"dense-vs-factored gate FAILED for {cell['behavior']}/{cell['substrate']}: "
            f"max rel {rel:.3e} > {tol:.0e}"
        )
    logger.info(
        "[phase=gate] dense-vs-factored PASS %s/%s: rel=%.3e (top-%d sv, n=%d)",
        cell["behavior"],
        cell["substrate"],
        rel,
        k,
        cell["n_ctx"],
    )
    return {"pass": True, "tol": tol, "rel": rel, "top_k": int(k)}


def _sanity_avg(cell: dict) -> dict:
    """Recomputed-from-pe averages vs the stored summary averages (exact corpus-match check)."""
    out = {}
    for arm in ARMS:
        Xr = _avg_from_pe(cell, arm, "c")
        Yr = _avg_from_pe(cell, arm, "v")
        dc = float(np.max(np.abs(Xr - cell["avg_c"][arm])))
        dv = float(np.max(np.abs(Yr - cell["avg_v"][arm])))
        out[arm] = {"max_abs_diff_c": dc, "max_abs_diff_v": dv}
    return out


# ── per-cell orchestration ───────────────────────────────────────────────────


def _cell_path(out_root: Path, behavior: str, substrate: str) -> Path:
    return out_root / CELL_SUBDIR / f"{behavior}_{substrate}.json"


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _cell_done(path: Path, regime: dict) -> bool:
    if not path.exists():
        return False
    try:
        prev = json.loads(path.read_text())
    except Exception:
        return False
    return prev.get("regime") == regime


def run_cell(
    behavior: str, substrate: str, dl_root: Path, out_root: Path, matched_n_draws: int, args
) -> dict:
    t0 = time.time()
    cell = load_cell(behavior, substrate, dl_root)
    sanity = _sanity_avg(cell)
    repro = _reproduction_gate(cell)
    dense = _dense_equivalence_gate(cell) if args.dense_gate else {"skipped": True}
    reads = {arm: _arm_reads(cell, arm, matched_n_draws) for arm in ARMS}
    wall = time.time() - t0
    result = {
        "behavior": behavior,
        "substrate": substrate,
        "n_ctx": cell["n_ctx"],
        "n_rows": cell["n_rows"],
        "regime": _regime(matched_n_draws),
        "sanity_recompute_vs_summary": sanity,
        "gates": {"reproduction": repro, "dense_equivalence": dense},
        "arms": reads,
        "wall_s": wall,
    }
    logger.info(
        "[phase=cell-done] %s/%s in %.1fs: "
        "avg rank(base primary)=%d vs pe rank=%d; matched-n50 stable_rank=%.1f; "
        "energy_in_avg_colspace(base)=%.3f",
        behavior,
        substrate,
        wall,
        reads["base"]["averaged_primary"]["n_nonzero"],
        reads["base"]["per_example_primary"]["n_nonzero"],
        reads["base"]["matched_n"]["stable_rank"]["mean"],
        reads["base"]["residual"]["energy_in_averaged_colspace"],
    )
    return result


# ── figures ──────────────────────────────────────────────────────────────────


def _make_figures(consolidated: dict, fig_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_blog,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    fig_dir = fig_root / "figures/issue_813"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cells = {(c["behavior"], c["substrate"]): c for c in consolidated["cells"]}
    sub_colors = dict(zip(SUBSTRATES, paper_palette_blog(3), strict=True))

    # Figure 1: cumulative-energy curves, per behavior panel, per arm file
    for arm in ARMS:
        fig, axes = plt.subplots(1, len(BEHAVIORS), figsize=(4 * len(BEHAVIORS), 3.4), sharey=True)
        for ax, beh in zip(axes, BEHAVIORS, strict=True):
            for sub in SUBSTRATES:
                c = cells.get((beh, sub))
                if c is None:
                    continue
                r = c["arms"][arm]
                col = sub_colors[sub]
                sv_a = np.asarray(r["averaged_primary"]["top_sv"], dtype=np.float64)
                sv_p = np.asarray(r["per_example_primary"]["top_sv"], dtype=np.float64)
                sv_res = np.asarray(r["residual"]["residual_spectrum"]["top_sv"], dtype=np.float64)
                for sv, style, lab in (
                    (sv_a, "-", "averaged"),
                    (sv_p, "--", "per-example"),
                    (sv_res, ":", "residual (pe⊥avg)"),
                ):
                    if sv.size == 0:
                        continue
                    cum = np.cumsum(sv**2) / np.sum(sv**2)
                    ax.plot(
                        np.arange(1, len(cum) + 1),
                        cum,
                        style,
                        color=col,
                        lw=1.4,
                        label=f"{SUB_LABEL[sub]} · {lab}",
                    )
                mn = r["matched_n"]["cum_energy"]
                ks = sorted(int(k) for k in mn)
                ax.plot(ks, [mn[str(k)]["mean"] for k in ks], "o", color=col, ms=3)
            ax.set_xscale("log")
            ax.set_xlim(1, 600)
            ax.set_ylim(0, 1.02)
            ax.set_title(BEH_LABEL[beh])
            ax.set_xlabel("component rank k (log)")
        axes[0].set_ylabel("cumulative Frobenius energy")
        axes[-1].legend(fontsize=6, loc="lower right")
        fig.suptitle(f"map spectrum concentration — {arm} arm, L14 (λ=1e3)", fontsize=11)
        savefig_paper(fig, f"rank_spectrum_cumenergy_{arm}", dir=str(fig_dir))
        plt.close(fig)

    # Figure 2: per-cell summary bars (trained arm) — stable rank / k50 / k90 / energy-in-colspace
    arm = "trained"
    labels, sr_a, sr_p, sr_m, k90_a, k90_p, ein = [], [], [], [], [], [], []
    for beh in BEHAVIORS:
        for sub in SUBSTRATES:
            c = cells.get((beh, sub))
            if c is None:
                continue
            r = c["arms"][arm]
            labels.append(f"{BEH_LABEL[beh]}\n{SUB_LABEL[sub]}")
            sr_a.append(r["averaged_primary"]["stable_rank"])
            sr_p.append(r["per_example_primary"]["stable_rank"])
            sr_m.append(r["matched_n"]["stable_rank"]["mean"])
            k90_a.append(r["averaged_primary"]["k90"])
            k90_p.append(r["per_example_primary"]["k90"])
            ein.append(r["residual"]["energy_in_averaged_colspace"])
    x = np.arange(len(labels))
    cols = paper_palette_blog(3)
    fig, axes = plt.subplots(1, 3, figsize=(4.6 * 3, 4.0))
    w = 0.27
    axes[0].bar(x - w, sr_a, w, label="averaged", color=cols[0])
    axes[0].bar(x, sr_m, w, label="matched-n50 (mean)", color=cols[1])
    axes[0].bar(x + w, sr_p, w, label="per-example", color=cols[2])
    axes[0].set_ylabel("stable rank ‖W‖_F²/σ₁²")
    axes[0].set_title("stable rank (trained arm, λ=1e3)")
    axes[0].legend(fontsize=7)
    axes[1].bar(x - w / 2, k90_a, w, label="averaged", color=cols[0])
    axes[1].bar(x + w / 2, k90_p, w, label="per-example", color=cols[2])
    axes[1].set_ylabel("k for 90% energy")
    axes[1].set_title("90%-energy dimension")
    axes[1].legend(fontsize=7)
    axes[2].bar(x, ein, 0.6, color=cols[0])
    axes[2].axhline(1.0, ls=":", color="grey", lw=1)
    axes[2].set_ylabel("W_pe energy fraction in col(W_avg)")
    axes[2].set_title("per-example energy inside averaged column space")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
    savefig_paper(fig, "rank_spectrum_summary", dir=str(fig_dir))
    plt.close(fig)
    logger.info("[phase=figures] wrote figures → %s", fig_dir)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Issue #813 rank/spectrum free analysis (L14)")
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS), choices=list(BEHAVIORS))
    ap.add_argument("--substrates", nargs="+", default=list(SUBSTRATES), choices=list(SUBSTRATES))
    ap.add_argument("--matched-n-draws", type=int, default=10)
    ap.add_argument("--dl-root", type=Path, default=PROJECT_ROOT / "data/issue_813/hf_dl")
    ap.add_argument("--out-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--resume", action="store_true", help="skip cells whose stored regime matches")
    ap.add_argument(
        "--dense-gate",
        action="store_true",
        default=True,
        help="run the dense-vs-factored spectrum equivalence gate per cell",
    )
    ap.add_argument("--no-dense-gate", dest="dense_gate", action="store_false")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="one cell (first behavior/substrate): gates + reads, no figures/cleanup",
    )
    ap.add_argument("--no-cleanup", action="store_true", help="keep the hf_dl download cache")
    args = ap.parse_args()

    t0 = time.time()
    out_root = args.out_root
    regime = _regime(args.matched_n_draws)

    cells_to_run = [(b, s) for b in args.behaviors for s in args.substrates]
    if args.smoke:
        cells_to_run = cells_to_run[:1]
        logger.info("[phase=smoke] one cell: %s", cells_to_run[0])

    for behavior, substrate in cells_to_run:
        cpath = _cell_path(out_root, behavior, substrate)
        if args.resume and _cell_done(cpath, regime):
            logger.info("[phase=resume] skip %s/%s (regime match)", behavior, substrate)
            continue
        result = run_cell(behavior, substrate, args.dl_root, out_root, args.matched_n_draws, args)
        _atomic_write_json(cpath, result)
        logger.info("[phase=cell-written] %s", cpath)

    if args.smoke:
        logger.info("[phase=smoke] done in %.1fs", time.time() - t0)
        return 0

    # consolidate
    cells = []
    for behavior in BEHAVIORS:
        for substrate in SUBSTRATES:
            cpath = _cell_path(out_root, behavior, substrate)
            if cpath.exists():
                cells.append(json.loads(cpath.read_text()))
    consolidated = {
        "experiment": "issue813_rank_spectrum_per_example_vs_averaged",
        "layer": HEADLINE_LAYER,
        "metadata": {
            "git_sha": _git_sha(),
            "hf_revision": HF_REVISION,
            "lambda_primary": LAMBDA_PRIMARY,
            "lambdas_grid": LAMBDAS,
            "matched_n_draws": args.matched_n_draws,
            "seed": SEED,
            "recipe": "standardize-X(std,ddof0,+1e-9); ridge-target-UNCENTERED (matches committed "
            "maps output_centered=False); full 3584-dim output; factored spectrum via "
            "eigh(G) + S=diag(sqrt(e)/(e+lam))·Qᵀ(YYᵀ)Q·diag(sqrt(e)/(e+lam))",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "command": " ".join(sys.argv),
        },
        "n_cells": len(cells),
        "cells": cells,
    }
    cons_path = out_root / CONSOLIDATED
    _atomic_write_json(cons_path, consolidated)
    logger.info("[phase=consolidate] wrote %s (%d cells)", cons_path, len(cells))

    _make_figures(consolidated, out_root)

    if not args.no_cleanup and args.dl_root.exists():
        import shutil

        shutil.rmtree(args.dl_root, ignore_errors=True)
        logger.info("[phase=cleanup] removed download cache %s", args.dl_root)

    logger.info("[phase=done] total %.1fs", time.time() - t0)
    print("RANK_SPECTRUM_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
