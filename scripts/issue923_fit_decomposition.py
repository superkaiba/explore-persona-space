#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
"""Issue #923 Phase 3 — batched ridge decomposition battery (fits + nulls + stats).

Plan §4.3 Phase 3 + §9 batching commitments. NO training; ridge-only. The fit
engine works in the PRIMAL d=48/96 space via ONE thin SVD of the standardized
train design per (arm, fold) — the shared factorization the
vectorize-many-cell-fits rule requires — with exact PRESS leave-one-out λ
selection re-derived per draw from λ-independent per-draw GEMMs (the K-trick:
``mse_λ = Σ_i w_i(λ)² ||resid_i||²`` expands into three λ-independent per-draw
reductions ``a/V/C`` plus X-only per-λ factors ``s_λ/K_λ``). Exactness vs the
imported #658 dual/PRESS helpers (``_press_loo_mse_per_lambda`` /
``_ridge_dual_weights``) is asserted at startup (``run_selftest`` — fp64,
<=1e-8) so the primal engine IS the dual reference, just in the cheap space
(m up to 5184 makes the dual m x m Gram the #823 trap).

Folds: 7 LOFO context families x 4 stratified query folds (primary; both axes
unseen), + exploratory marginals (LOFO-only, query-fold-only), + the Dolly
corpus-transfer OOD folds. Per (genre, layer): shared train-fold target PCA-48
(``torch.pca_lowrank``, seeded), per-arm feature PCA (weighted exact SVD on
distinct ctx/query rows; ``pca_lowrank`` on cell-level designs), degenerate
projected dims dropped before standardization (§8 rank-deficiency row).

Nulls: full-grid cell-label permutations of the reduced target (the #810
``batched_ridge_loco_null_skill`` design — X-side factors shared, λ RE-SELECTED
PER DRAW via batched PRESS), draws chunked as stacked GEMMs; the per-draw x
per-layer skill matrix is persisted per (arm, genre) with the permutation
matrix + seed pinned (selection-symmetric-nulls rule).

Bootstrap: 2000 paired family-cluster draws (one shared count matrix — GEMM
re-reduction of per-family SS); ρ_dec numerator/denominator per draw with the
best-single argmax re-applied INSIDE every draw + the §3 denominator guard;
cross-classified family x query bootstrap for H3 at L18. ANOVA oracle shares
per (genre, layer) (in-sample, PCA-48 + ambient). Oracle-consistency checks are
TOLERANCE-FLAGGED reported anomalies — never a crash (§3 registered exception).

Checkpoint/resume: per (genre, layer) partial packs keyed on the FULL regime
key (genres, arms, n_perms, seeds, fold hash, PCA dim, per-pack input
identities — name/size/mtime of every consumed capture shard + reduce pack) —
a mismatch refuses loudly (never silently reuses wrong rows).

Usage::

    uv run python scripts/issue923_fit_decomposition.py --packs-dir data/issue_923
    uv run python scripts/issue923_fit_decomposition.py --smoke        # synthetic grid
    uv run python scripts/issue923_fit_decomposition.py --selftest-only
    uv run python scripts/issue923_fit_decomposition.py --time-projection
    uv run python scripts/issue923_fit_decomposition.py --fullh-spotcheck  # GPU, L18
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue658_fit_predictors import (  # noqa: E402
    RIDGE_LAMBDAS,
    _press_loo_mse_per_lambda,
    _requested_device,
    _resolve_device,
    _ridge_dual_weights,
)
from issue810_batched_null import make_perm_matrix  # noqa: E402
from issue923_common import (  # noqa: E402
    ARM_FULL,
    ARMS_CONCAT,
    ARMS_SINGLE,
    DATA_DIR,
    FAMILY_ORDER,
    HEADLINE_LAYER,
    HF_DATA_REPO,
    HF_PREFIX_923,
    N_QUERY_FOLDS,
    SEED,
    cell_row,
    dump_json,
    load_json,
    load_pack,
    save_pack,
    weighted_pca_basis,
)

from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue923_fit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PCA_DIM = 48  # #722/#810 locked target/input compression (plan §11)
NULL_CHUNK = int(os.environ.get("EPM_NULL_CHUNK", "100"))
SINGLES_FOR_RHO_DEC = ("arm_ctx", "arm_qry_i")  # §3 best-single set (primary presentation)


# ── exact PRESS ridge engine (primal thin-SVD, batched draws) ────────────────


class PressRidge:
    """X-side factors for ONE (standardized design, fold), shared across draws.

    ``Xn`` (m, d) standardized fp64 train design. Thin SVD ``Xn = U S Vᵀ`` gives
    the hat matrix ``H(λ) = U diag(S²/(S²+λ)) Uᵀ`` — identical to the dual Gram
    eigendecomposition in ``_press_loo_mse_per_lambda`` (nonzero spectra
    coincide; zero modes contribute nothing) — so PRESS + the dual-weight
    prediction here are EXACT re-expressions of the #658 helpers, in O(m d²)
    instead of O(m³). Per-λ X-only factors (``s_λ = 1/(1-h_i)²``,
    ``K_λ = Uᵀ diag(s_λ) U``) are precomputed once; each draw needs only three
    λ-independent GEMMs (G, Yperp·row-norms, V) + trivial per-λ contractions.
    """

    def __init__(self, Xn: torch.Tensor, lambdas=RIDGE_LAMBDAS) -> None:
        assert Xn.ndim == 2, Xn.shape
        self.m, self.d = Xn.shape
        self.lambdas = list(lambdas)
        self.device = Xn.device
        U, S, Vh = torch.linalg.svd(Xn, full_matrices=False)
        self.U, self.S, self.Vh = U, S, Vh  # (m,k), (k,), (k,d)
        k = S.shape[0]
        lam = torch.tensor(self.lambdas, dtype=Xn.dtype, device=Xn.device).view(-1, 1)
        s2 = (S * S).view(1, k)
        self.phi = s2 / (s2 + lam)  # (n_lambda, k)
        h = (U * U) @ self.phi.T  # (m, n_lambda)
        w = 1.0 / (1.0 - h).clamp(min=1e-8)
        self.s_w2 = (w * w).T.contiguous()  # (n_lambda, m)
        # X-only per-λ factors, GEMM-formed (never per-element einsum loops):
        # sU_λ = diag(s_λ) U (n_lambda, m, k); K_λ = Uᵀ sU_λ (n_lambda, k, k).
        self.sU = self.s_w2.unsqueeze(2) * U.unsqueeze(0)  # (n_lambda, m, k)
        self.K = torch.bmm(U.T.unsqueeze(0).expand(len(self.lambdas), -1, -1), self.sU)

    def cast(self, dtype: torch.dtype) -> PressRidge:
        """Lightweight dtype copy (fp32 null path; factors re-cast, no re-SVD).

        Exactness in fp64 is the selftest contract; the fp32 copy is used ONLY
        for the permutation-null battery, where band quantiles tolerate fp32
        (the observed fits stay fp64).
        """
        other = object.__new__(PressRidge)
        other.m, other.d, other.lambdas, other.device = self.m, self.d, self.lambdas, self.device
        for name in ("U", "S", "Vh", "phi", "s_w2", "sU", "K"):
            setattr(other, name, getattr(self, name).to(dtype))
        return other

    def press_mse(self, Yc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(mse (B, n_lambda), G (B, k, P)) for a stack of CENTERED train targets.

        ``Yc`` (B, m, P). Exact PRESS LOO mse per λ per draw (mean over m LOO
        folds AND P outputs — matching ``_press_loo_mse_per_lambda``). Every
        heavy contraction is an explicit reshaped GEMM / bmm (the einsum forms
        ran ~5.6 GFLOP/s; the GEMM forms are the batched-draws commitment).
        Returns G = UᵀY for reuse by :meth:`predict`.
        """
        B, m, P = Yc.shape
        assert m == self.m, (m, self.m)
        U = self.U
        k = U.shape[1]
        # (m, B*P) view of the draw stack → ONE (k,m)@(m,B*P) GEMM for G.
        Ycm = Yc.permute(1, 0, 2).reshape(m, B * P)
        Gm = U.T @ Ycm  # (k, B*P)
        Yperp_m = Ycm - U @ Gm  # (m, B*P) via one (m,k)@(k,B*P) GEMM
        G = Gm.reshape(k, B, P).permute(1, 0, 2).contiguous()  # (B, k, P)
        Yperp = Yperp_m.reshape(m, B, P).permute(1, 0, 2).contiguous()  # (B, m, P)
        a = (Yperp * Yperp).sum(dim=2)  # (B, m)
        V = torch.bmm(Yperp, G.transpose(1, 2))  # (B, m, k) bmm
        C = torch.bmm(G, G.transpose(1, 2))  # (B, k, k) bmm
        one_minus_phi = 1.0 - self.phi  # (n_lambda, k)
        mse = torch.empty(B, len(self.lambdas), dtype=Yc.dtype, device=Yc.device)
        Vflat = V.reshape(B, m * k)
        Cflat = C.reshape(B, k * k)
        for li in range(len(self.lambdas)):
            term_a = a @ self.s_w2[li]  # (B,) GEMV
            # Σ_i s_i Σ_j U_ij (1-φ_j) V_bij == V ⋅ (sU_λ diag(1-φ)) — one GEMV.
            sU_w = (self.sU[li] * one_minus_phi[li].unsqueeze(0)).reshape(m * k)
            term_b = 2.0 * (Vflat @ sU_w)
            Kw = (self.K[li] * torch.outer(one_minus_phi[li], one_minus_phi[li])).reshape(k * k)
            term_c = Cflat @ Kw
            # mean over the m LOO folds AND P outputs — matches the reference.
            mse[:, li] = (term_a + term_b + term_c) / (m * P)
        return mse, G

    def predict(self, G: torch.Tensor, lam_idx: torch.Tensor, Xte_n: torch.Tensor) -> torch.Tensor:
        """Held-out CENTERED predictions (B, n_te, P) at each draw's selected λ.

        Exact dual-form prediction: ``pred = Xte_n V diag(S/(S²+λ)) G`` —
        algebraically identical to ``Xte_n @ _ridge_dual_weights(Xn, Yc, λ)``.
        """
        T = Xte_n @ self.Vh.T  # (n_te, k) X-only per call
        S = self.S
        lam = torch.tensor(self.lambdas, dtype=G.dtype, device=G.device)
        coef = S.view(1, -1) / (S.view(1, -1) ** 2 + lam[lam_idx].view(-1, 1))  # (B, k)
        return torch.matmul(T, coef.unsqueeze(2) * G)  # (n_te,k) @ (B,k,P) → (B,n_te,P)


def press_fit_predict(
    Xtr: torch.Tensor,
    Ytr: torch.Tensor,
    Xte: torch.Tensor,
    return_engine: bool = False,
    standardize: bool = True,
):
    """Observed-fit convenience wrapper: standardize, center, select λ, predict.

    ``Xtr`` (m, d) / ``Ytr`` (m, P) / ``Xte`` (n_te, d) fp64.
    ``standardize=True`` (AMBIENT designs, e.g. the full-H spot-check):
    train-only per-dim mu/sd (ddof=0, +1e-9 floor — the #658/#810 convention)
    + the §8 degenerate-dim drop. ``standardize=False`` (PCA-projected designs
    from ``build_arm_design``): the ambient standardization already happened
    BEFORE the PCA projection (plan §4.1 listing order: standardize-X →
    PCA-48), so the PCA coordinates are only train-mean-centered here —
    re-standardizing PER PCA DIM would WHITEN near-noise directions up to unit
    variance and destroy out-of-family generalization (smoke-diagnosed: a pure
    orthonormal rotation is ridge-invariant, rotation + whiten flipped a fold
    skill from +0.13 to −6.4). Target: train-mean centered; prediction adds
    ymu back. Returns preds (n_te, P), per-λ test preds, chosen λ, engine.
    """
    if standardize:
        mu = Xtr.mean(0)
        sd = Xtr.std(0, correction=0) + 1e-9
        keep = sd > (sd.max() * 1e-6 + 1e-12)  # drop degenerate dims (§8)
        Xtr_n = ((Xtr - mu) / sd)[:, keep]
        Xte_n = ((Xte - mu) / sd)[:, keep]
    else:
        mu = Xtr.mean(0)
        sd = torch.ones_like(mu)
        keep = torch.ones(Xtr.shape[1], dtype=torch.bool, device=Xtr.device)
        Xtr_n = Xtr - mu
        Xte_n = Xte - mu
    ymu = Ytr.mean(0, keepdim=True)
    Ytr_c = (Ytr - ymu).unsqueeze(0)  # (1, m, P)
    eng = PressRidge(Xtr_n)
    mse, G = eng.press_mse(Ytr_c)
    lam_idx = torch.argmin(mse, dim=1)  # (1,)
    pred_c = eng.predict(G, lam_idx, Xte_n)[0]  # (n_te, P)
    per_lambda_preds = []
    for li in range(len(RIDGE_LAMBDAS)):
        idx = torch.full((1,), li, dtype=torch.long, device=Xtr.device)
        per_lambda_preds.append(eng.predict(G, idx, Xte_n)[0] + ymu)
    out = {
        "pred": pred_c + ymu,
        "per_lambda_pred": per_lambda_preds,
        "lam_idx": int(lam_idx.item()),
        "mse": mse[0].cpu().numpy(),
        "std": (mu, sd, keep),
        "ymu": ymu,
    }
    if return_engine:
        out["engine"] = (eng, Xtr_n, Xte_n)
    return out


def run_selftest(device: str = "cpu") -> dict:
    """Exactness gate: primal engine == imported #658 dual/PRESS helpers <=1e-8.

    Synthetic (m=40, d=12, P=5) fp64; asserts (a) PRESS mse per λ matches
    ``_press_loo_mse_per_lambda``, (b) predictions at each λ match
    ``Xte_n @ _ridge_dual_weights``. Runs at Phase-3 start on EVERY invocation
    (cheap, fail-loud) — the reuse-by-import contract in code, not prose.
    """
    torch.manual_seed(0)
    dev = torch.device(device)
    m, d, p, n_te = 40, 12, 5, 7
    Xn = torch.randn(m, d, dtype=torch.float64, device=dev)
    W = torch.randn(d, p, dtype=torch.float64, device=dev)
    Y = Xn @ W + 0.1 * torch.randn(m, p, dtype=torch.float64, device=dev)
    Xte = torch.randn(n_te, d, dtype=torch.float64, device=dev)
    eng = PressRidge(Xn)
    mse_new, G = eng.press_mse(Y.unsqueeze(0))
    mse_ref = _press_loo_mse_per_lambda(Xn, Y, RIDGE_LAMBDAS)
    dmse = float((mse_new[0] - mse_ref).abs().max())
    assert dmse <= 1e-8, f"PRESS exactness failed: max|Δmse| = {dmse}"
    dpred_max = 0.0
    for li, lam in enumerate(RIDGE_LAMBDAS):
        w_ref = _ridge_dual_weights(Xn, Y, lam)
        pred_ref = Xte @ w_ref
        idx = torch.full((1,), li, dtype=torch.long, device=dev)
        pred_new = eng.predict(G, idx, Xte)[0]
        dpred_max = max(dpred_max, float((pred_new - pred_ref).abs().max()))
    assert dpred_max <= 1e-8, f"dual-weight prediction exactness failed: {dpred_max}"
    return {"max_abs_dmse": dmse, "max_abs_dpred": dpred_max, "device": device}


# ── grid assembly ─────────────────────────────────────────────────────────────


def _collect_shards(packs_dir: Path, stem: str) -> list[tuple[dict, dict]]:
    """Load every `<stem>_shard*.pt` pack; fail loud when none exist."""
    files = sorted(packs_dir.glob(f"{stem}_shard*.pt"))
    assert files, f"no packs matching {stem}_shard*.pt under {packs_dir}"
    return [load_pack(f) for f in files]


def _uc_ext_offset(ext_packs: list[tuple[dict, dict]], n_q: int) -> int:
    """Global-index offset for UC-ext cells (store pool occupies 0..offset-1).

    Derived from the packs, not assumed: offset = n_q − n_ext, cross-checked
    against the canonical 48-probe UC store pool so a pool-size drift (or a
    smoke pack mixed into a production packs dir) fails loud at the join
    instead of silently mis-indexing rows (r1 Minor: unguarded `48 +`).
    """
    n_ext = max(r["q_idx"] for _t, m in ext_packs for r in m["rows"]) + 1
    off = n_q - n_ext
    assert off == 48, (
        f"UC ext-offset {off} != 48 (n_q={n_q}, n_ext={n_ext}) — store pool size drifted"
    )
    return off


def _input_pack_identity(
    packs_dir: Path, reduce_dir: Path, extra_dirs: tuple[Path, ...] = ()
) -> list[list]:
    """Cheap content identity for every input pack the fit consumes.

    Sorted ``[name, size_bytes, mtime_ns]`` triples over the capture shards +
    reduce packs (+ any extra feature-pack dirs, e.g. the pooled packs). A
    re-captured / re-reduced / re-fetched pack changes its triple, so the
    resume regime hash changes and a stale per-(genre, layer) partial refuses
    loudly (the #722-r3 resume class) instead of silently feeding OLD data
    into a `--fits-only` re-dispatch. False invalidation (same bytes, fresh
    mtime) only costs a `--fresh` refit — the safe side.
    """
    files = sorted(packs_dir.glob("*_shard*.pt")) + sorted(reduce_dir.glob("vbar_store_*.pt"))
    for d in extra_dirs:
        files += sorted(d.glob("*_shard*.pt"))
    return [[f.name, f.stat().st_size, f.stat().st_mtime_ns] for f in files]


def _fill_rows(
    dest: torch.Tensor,
    valid: torch.Tensor,
    packs: list[tuple[dict, dict]],
    key: str,
    ctx_idx: dict[str, int],
    n_q: int,
    row_filter=None,
) -> None:
    """Scatter shard pack rows into the (n_ctx*n_q, Lc, H) grid arrays."""
    for tensors, meta in packs:
        rows = meta["rows"]
        v = tensors.get("valid")
        for i, r in enumerate(rows):
            if row_filter is not None and not row_filter(r):
                continue
            ci = ctx_idx.get(r["ctx_id"])
            if ci is None:
                continue
            row = cell_row(ci, r["q_idx"], n_q)
            dest[row] = tensors[key][i]
            valid[row] = bool(v[i]) if v is not None else True


class GridData:
    """Assembled per-genre grid: targets + per-arm feature arrays (fp16, all layers)."""

    def __init__(
        self,
        genre: str,
        ctx_ids: list[str],
        families: list[str],
        n_q: int,
        target: torch.Tensor,
        valid: torch.Tensor,
        ffull: torch.Tensor,
        fqryiii: torch.Tensor | None,
        qryiii_valid: torch.Tensor | None,
    ) -> None:
        self.genre = genre
        self.ctx_ids = ctx_ids
        self.families = families
        self.n_ctx = len(ctx_ids)
        self.n_q = n_q
        self.target = target  # (n_cells, Lc, H) fp16
        self.valid = valid  # (n_cells,) bool (target ∧ ffull [∧ qryiii])
        self.ffull = ffull
        self.fqryiii = fqryiii  # None when arm dropped

    def ctx_of(self, cells: np.ndarray) -> np.ndarray:
        return cells // self.n_q

    def q_of(self, cells: np.ndarray) -> np.ndarray:
        return cells % self.n_q


def load_grids(  # noqa: C901 — linear grid-assembly, per-genre branches
    packs_dir: Path,
    reduce_dir: Path,
    data_dir: Path,
    genres: list[str],
    ood: bool,
    feature_source: str = "last",
    pooled_packs_dir: Path | None = None,
):
    """Assemble GridData per genre (+ dolly) from capture shards + reduce packs.

    Returns (grids, fctx (n_ctx, Lc, H) fp32, fqry {pres: {genre: (n_q, Lc, H)}},
    battery ctx ids/families, run-level metadata incl. mask_backend).

    ``feature_source="pool"`` (pooled-span-features round): every FEATURE array
    reads the ``pool_*`` packs' ``fpool`` key from ``pooled_packs_dir``;
    TARGETS keep reading ``vbar`` from the same tgt/reduce packs (byte-identical
    inputs — the round's reuse premise).
    """
    from issue594_common import load_battery

    pool = feature_source == "pool"
    if pool:
        assert pooled_packs_dir is not None, "feature_source=pool requires --pooled-packs-dir"
    feat_dir = pooled_packs_dir if pool else packs_dir
    feat_key = "fpool" if pool else "flast"

    _, instances = load_battery()
    folds_payload = load_json(data_dir / "fold_assignments.json")
    fam_map = folds_payload["families"]

    fctx_packs = _collect_shards(feat_dir, "pool_fctx" if pool else "fctx")
    ctx_ids = sorted(
        {r["ctx_id"] for _t, m in fctx_packs for r in m["rows"]},
        key=lambda c: [i["id"] for i in instances].index(c),
    )
    ctx_idx = {c: i for i, c in enumerate(ctx_ids)}
    families = [fam_map[c] for c in ctx_ids]
    lc = fctx_packs[0][0][feat_key].shape[1]
    hidden = fctx_packs[0][0][feat_key].shape[2]
    fctx = torch.zeros(len(ctx_ids), lc, hidden, dtype=torch.float16)
    for tensors, meta in fctx_packs:
        for i, r in enumerate(meta["rows"]):
            fctx[ctx_idx[r["ctx_id"]]] = tensors[feat_key][i]
    mask_backend = fctx_packs[0][1].get("mask_backend", "unknown")

    fqry: dict[str, dict[str, torch.Tensor]] = {}
    for pres in ("i", "ii"):
        packs = _collect_shards(feat_dir, f"pool_fqry_{pres}" if pool else f"fqry_{pres}")
        per_genre: dict[str, torch.Tensor] = {}
        for tensors, meta in packs:
            for i, r in enumerate(meta["rows"]):
                g = r["genre"]
                if g not in per_genre:
                    n_qg = (
                        max(
                            rr["q_idx"] for _t, mm in packs for rr in mm["rows"] if rr["genre"] == g
                        )
                        + 1
                    )
                    per_genre[g] = torch.zeros(n_qg, lc, hidden, dtype=torch.float16)
                per_genre[g][r["q_idx"]] = tensors[feat_key][i]
        fqry[pres] = per_genre

    def _grid(genre: str, n_q: int, target: torch.Tensor, valid: torch.Tensor) -> GridData:
        ffull = torch.zeros(len(ctx_ids) * n_q, lc, hidden, dtype=torch.float16)
        fvalid = torch.zeros(len(ctx_ids) * n_q, dtype=torch.bool)
        if pool:
            # ONE pack family per genre, GLOBAL q_idx (ext lives at 48+ for uc).
            _fill_rows(
                ffull,
                fvalid,
                _collect_shards(feat_dir, f"pool_ffull_{genre}"),
                feat_key,
                ctx_idx,
                n_q,
            )
        elif genre == "uc":
            _fill_rows(
                ffull, fvalid, _collect_shards(packs_dir, "ffull_uc48"), "flast", ctx_idx, n_q
            )
            # ext cells: F_full rides the TF pack's flast (same forward, plan 1b);
            # ext q_idx is LOCAL 0..95 → shift by the DERIVED store-pool offset.
            ext = _collect_shards(packs_dir, "tgt_ucext")
            off = _uc_ext_offset(ext, n_q)
            for tensors, meta in ext:
                for i, r in enumerate(meta["rows"]):
                    row = cell_row(ctx_idx[r["ctx_id"]], off + r["q_idx"], n_q)
                    ffull[row] = tensors["flast"][i]
                    fvalid[row] = True
        elif genre == "betley":
            _fill_rows(
                ffull, fvalid, _collect_shards(packs_dir, "ffull_betley"), "flast", ctx_idx, n_q
            )
        else:  # dolly — F_full from the TF pack (same-forward flast)
            _fill_rows(
                ffull, fvalid, _collect_shards(packs_dir, "tgt_dolly"), "flast", ctx_idx, n_q
            )
        if mask_backend == "dropped":
            # The ONLY licensed drop path: the capture run RECORDED the §8
            # mask-ladder failure in its pack meta. Pack ABSENCE without that
            # record is a fetch/upload/coverage failure and fails loud below
            # (r1 blocker fqryiii-pack-absence-silently-drops-arm).
            fqryiii, iii_valid = None, None
        else:
            iii_packs = _collect_shards(
                feat_dir, f"pool_fqry_iii_{genre}" if pool else f"fqry_iii_{genre}"
            )
            fqryiii = torch.zeros(len(ctx_ids) * n_q, lc, hidden, dtype=torch.float16)
            iii_valid = torch.zeros(len(ctx_ids) * n_q, dtype=torch.bool)
            _fill_rows(fqryiii, iii_valid, iii_packs, feat_key, ctx_idx, n_q)
        assert (fqryiii is None) == (mask_backend == "dropped")  # gate consistency
        common = valid & fvalid
        if iii_valid is not None:
            common = common & iii_valid  # the #823 common-valid rule
        return GridData(genre, ctx_ids, families, n_q, target, common, ffull, fqryiii, iii_valid)

    grids: dict[str, GridData] = {}
    for genre in genres:
        n_q = folds_payload["n_queries"][genre]
        target = torch.zeros(len(ctx_ids) * n_q, lc, hidden, dtype=torch.float16)
        tvalid = torch.zeros(len(ctx_ids) * n_q, dtype=torch.bool)
        # store cells from the Phase-2 reduce pack
        rt, rm = load_pack(reduce_dir / f"vbar_store_{genre}.pt")
        for i, r in enumerate(rm["rows"]):
            row = cell_row(ctx_idx[r["ctx_id"]], r["q_idx"], n_q)
            target[row] = rt["vbar"][i]
            tvalid[row] = bool(rt["valid"][i])
        if genre == "uc":
            ext = _collect_shards(packs_dir, "tgt_ucext")
            off = _uc_ext_offset(ext, n_q)
            for tensors, meta in ext:
                for i, r in enumerate(meta["rows"]):
                    row = cell_row(ctx_idx[r["ctx_id"]], off + r["q_idx"], n_q)
                    target[row] = tensors["vbar"][i]
                    tvalid[row] = bool(tensors["valid"][i])
        grids[genre] = _grid(genre, n_q, target, tvalid)
    if ood:
        n_q = folds_payload["n_queries"]["dolly"]
        target = torch.zeros(len(ctx_ids) * n_q, lc, hidden, dtype=torch.float16)
        tvalid = torch.zeros(len(ctx_ids) * n_q, dtype=torch.bool)
        for tensors, meta in _collect_shards(packs_dir, "tgt_dolly"):
            for i, r in enumerate(meta["rows"]):
                row = cell_row(ctx_idx[r["ctx_id"]], r["q_idx"], n_q)
                target[row] = tensors["vbar"][i]
                tvalid[row] = bool(tensors["valid"][i])
        grids["dolly"] = _grid("dolly", n_q, target, tvalid)
    meta = {"mask_backend": mask_backend, "ctx_ids": ctx_ids, "families": families}
    return grids, fctx, fqry, folds_payload, meta


# ── folds ─────────────────────────────────────────────────────────────────────


def build_folds(grid: GridData, qfolds: list[int]) -> dict[str, list[dict]]:
    """Primary (LOFO x qfold), marginal, fold structures over VALID cells."""
    fams_present = [f for f in FAMILY_ORDER if f in set(grid.families)]
    cells = np.arange(grid.n_ctx * grid.n_q)
    valid = grid.valid.numpy()
    ctx_of = grid.ctx_of(cells)
    q_of = grid.q_of(cells)
    fam_of = np.array([grid.families[c] for c in ctx_of])
    qf = np.array([qfolds[q] for q in q_of])
    out: dict[str, list[dict]] = {"primary": [], "lofo_marginal": [], "qfold_marginal": []}
    for fi, fam in enumerate(fams_present):
        for k in range(N_QUERY_FOLDS):
            tr = cells[valid & (fam_of != fam) & (qf != k)]
            te = cells[valid & (fam_of == fam) & (qf == k)]
            out["primary"].append(
                {
                    "family": fam,
                    "qfold": k,
                    "train": tr,
                    "test": te,
                    "fold_id": fi * N_QUERY_FOLDS + k,
                }
            )
        out["lofo_marginal"].append(
            {
                "family": fam,
                "train": cells[valid & (fam_of != fam)],
                "test": cells[valid & (fam_of == fam)],
            }
        )
    for k in range(N_QUERY_FOLDS):
        out["qfold_marginal"].append(
            {"qfold": k, "train": cells[valid & (qf != k)], "test": cells[valid & (qf == k)]}
        )
    return out


# ── per-fold arm designs ──────────────────────────────────────────────────────


def _pca_lowrank_project(
    X_train: torch.Tensor, X_lists: list[torch.Tensor], k: int, seed: int
) -> list[torch.Tensor]:
    """Cell-level PCA: seeded ``torch.pca_lowrank`` fit on train rows, project all."""
    torch.manual_seed(seed)
    mu = X_train.mean(0, keepdim=True)
    q = min(k, X_train.shape[0] - 1, X_train.shape[1])
    _u, _s, V = torch.pca_lowrank(X_train - mu, q=q, center=False, niter=2)
    return [(x - mu) @ V for x in X_lists]


def _distinct_pca_project(
    rows_train: torch.Tensor,
    counts: torch.Tensor,
    project_lists: list[torch.Tensor],
    k: int,
) -> list[torch.Tensor]:
    """Distinct-row weighted exact PCA (ctx / query parts; §8 rank-deficiency)."""
    mu, comps = weighted_pca_basis(rows_train.numpy(), counts.numpy(), k)
    mu_t = torch.from_numpy(mu)
    comps_t = torch.from_numpy(comps.T)  # (H, k')
    return [(x - mu_t) @ comps_t for x in project_lists]


def build_part(
    part: str,
    layer: int,
    grid_tr: GridData,
    tr_cells: np.ndarray,
    grid_te: GridData,
    te_cells: np.ndarray,
    fctx: torch.Tensor,
    fqry: dict,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(X_train (m, k'), X_test (n_te, k')) fp64 PCA-projected part features.

    ``part`` in {ctx, qry_i, qry_ii, qry_iii, full}. Ambient train-fold
    standardization FIRST (§4.1 order), THEN the train-fold-fit PCA basis;
    ctx/query-level parts use distinct-row weighted exact SVD, cell-level parts
    (qry_iii / full) use seeded pca_lowrank on the expanded train rows. The
    projected coordinates are NOT re-standardized (see ``press_fit_predict``).
    """

    def _std(tr_amb: torch.Tensor, others: list[torch.Tensor]) -> list[torch.Tensor]:
        """Ambient train-fold standardization before PCA (ddof=0, +1e-9)."""
        mu = tr_amb.mean(0)
        sd = tr_amb.std(0, correction=0) + 1e-9
        return [(x - mu) / sd for x in [tr_amb, *others]]

    if part == "ctx":
        ctx_tr = grid_tr.ctx_of(tr_cells)
        distinct, counts = np.unique(ctx_tr, return_counts=True)
        tr_amb = fctx[ctx_tr, layer, :].double()
        te_amb = fctx[grid_te.ctx_of(te_cells), layer, :].double()
        rows = fctx[distinct, layer, :].double()
        tr_n, te_n, rows_n = _std(tr_amb, [te_amb, rows])
        pr = _distinct_pca_project(rows_n, torch.from_numpy(counts).double(), [tr_n, te_n], PCA_DIM)
        return pr[0], pr[1]
    if part in ("qry_i", "qry_ii"):
        pres = part.split("_")[1]
        q_tr = grid_tr.q_of(tr_cells)
        distinct, counts = np.unique(q_tr, return_counts=True)
        bank_tr = fqry[pres][grid_tr.genre]
        bank_te = fqry[pres][grid_te.genre]
        tr_amb = bank_tr[q_tr, layer, :].double()
        te_amb = bank_te[grid_te.q_of(te_cells), layer, :].double()
        rows = bank_tr[distinct, layer, :].double()
        tr_n, te_n, rows_n = _std(tr_amb, [te_amb, rows])
        pr = _distinct_pca_project(rows_n, torch.from_numpy(counts).double(), [tr_n, te_n], PCA_DIM)
        return pr[0], pr[1]
    if part == "qry_iii":
        tr_amb = grid_tr.fqryiii[tr_cells, layer, :].double()
        te_amb = grid_te.fqryiii[te_cells, layer, :].double()
        tr_n, te_n = _std(tr_amb, [te_amb])
        pr = _pca_lowrank_project(tr_n, [tr_n, te_n], PCA_DIM, seed)
        return pr[0], pr[1]
    if part == "full":
        tr_amb = grid_tr.ffull[tr_cells, layer, :].double()
        te_amb = grid_te.ffull[te_cells, layer, :].double()
        tr_n, te_n = _std(tr_amb, [te_amb])
        pr = _pca_lowrank_project(tr_n, [tr_n, te_n], PCA_DIM, seed)
        return pr[0], pr[1]
    raise ValueError(part)


ARM_PARTS = {
    "arm_ctx": ["ctx"],
    "arm_qry_i": ["qry_i"],
    "arm_qry_ii": ["qry_ii"],
    "arm_qry_iii": ["qry_iii"],
    "arm_concat_i": ["ctx", "qry_i"],
    "arm_concat_ii": ["ctx", "qry_ii"],
    "arm_concat_iii": ["ctx", "qry_iii"],
    ARM_FULL: ["full"],
}


def build_arm_design(arm, layer, grid_tr, tr_cells, grid_te, te_cells, fctx, fqry, seed):
    """Concatenate per-part PCA projections (d = 48 single / 96 concat)."""
    xs_tr, xs_te = [], []
    for part in ARM_PARTS[arm]:
        xt, xe = build_part(part, layer, grid_tr, tr_cells, grid_te, te_cells, fctx, fqry, seed)
        xs_tr.append(xt)
        xs_te.append(xe)
    return torch.cat(xs_tr, dim=1), torch.cat(xs_te, dim=1)


# ── ANOVA oracle (in-sample, targets only) ────────────────────────────────────


def anova_shares(grid: GridData, layer: int, seed: int) -> dict:
    """Grid variance shares of a(c)/b(q)/γ on PCA-48 + ambient (in-sample)."""
    valid = grid.valid.numpy()
    cells = np.arange(grid.n_ctx * grid.n_q)[valid]
    Y = grid.target[cells, layer, :].double()
    out = {}
    torch.manual_seed(seed)
    mu_all = Y.mean(0, keepdim=True)
    q = min(PCA_DIM, Y.shape[0] - 1, Y.shape[1])
    _u, _s, V = torch.pca_lowrank(Y - mu_all, q=q, center=False, niter=2)
    for space, Ys in (("pca48", (Y - mu_all) @ V), ("ambient", Y)):
        ctx_of = grid.ctx_of(cells)
        q_of = grid.q_of(cells)
        mu = Ys.mean(0, keepdim=True)
        Yc = Ys - mu
        a = torch.zeros(grid.n_ctx, Ys.shape[1], dtype=torch.float64)
        for c in np.unique(ctx_of):
            a[c] = Yc[ctx_of == c].mean(0)
        b = torch.zeros(grid.n_q, Ys.shape[1], dtype=torch.float64)
        for qq in np.unique(q_of):
            b[qq] = Yc[q_of == qq].mean(0)
        A = a[ctx_of]
        B = b[q_of]
        Gm = Yc - A - B
        ss_tot = float((Yc * Yc).sum())
        out[space] = {
            "share_ctx": float((A * A).sum()) / ss_tot,
            "share_qry": float((B * B).sum()) / ss_tot,
            "share_interaction": float((Gm * Gm).sum()) / ss_tot,
            "ss_tot": ss_tot,
            "n_cells": int(valid.sum()),
        }
    return out


# ── per-(genre, layer) fit unit ───────────────────────────────────────────────


def fit_layer_unit(  # noqa: C901 — one self-contained checkpoint unit (folds x arms)
    genre: str,
    layer: int,
    grids: dict[str, GridData],
    fctx: torch.Tensor,
    fqry: dict,
    folds: dict[str, list[dict]],
    qfolds: list[int],
    arms: list[str],
    n_perms: int,
    perm: np.ndarray | None,
    ood: bool,
    device: str,
    smoke_blend_assert: bool = False,
    perm_ood: np.ndarray | None = None,
    extended_nulls: bool = False,
    blend_null_this_layer: bool = True,
    anova_override: dict | None = None,
) -> dict:
    """One (genre, layer) unit: observed fits (+ per-λ), blend, marginals, OOD, nulls.

    Every quantity the downstream stats need is emitted here so the unit is a
    self-contained checkpoint: pooled/per-family/per-(family,q) SS per arm,
    per-cell errors, per-λ pooled SS, per-draw null SS, blend α/β.

    ``extended_nulls`` (pooled-span-features round, plan v6 §4.1): the null
    battery ALSO covers (a) ``arm_blend`` — the null pipeline mirrors the
    observed blend per draw (2 inner-split component refits on shared X-side
    factors + a per-draw 2-parameter LS, batched over draws; outer per-draw
    component predictions retained from the per-arm battery) and (b) the
    ``ood_dolly`` scheme — independent within-grid cell-label permutations of
    the train grid (``perm``) and the Dolly test grid (``perm_ood``), X fixed,
    λ re-selected per draw. ``blend_null_this_layer`` implements the §9
    descope ladder rung 1 (blend nulls at the L18 column only).
    """
    grid = grids[genre]
    dev = torch.device(device)
    res: dict = {"genre": genre, "layer": layer, "arms": {}, "blend": {}, "null": {}}
    fams_present = [f for f in FAMILY_ORDER if f in set(grid.families)]
    fam_index = {f: i for i, f in enumerate(fams_present)}
    n_cells = grid.n_ctx * grid.n_q
    nlam = len(RIDGE_LAMBDAS)

    acc = {
        arm: {
            "ss_res": 0.0,
            "ss_tot": 0.0,
            "ss_res_lambda": np.zeros(nlam),
            "fam_res": np.zeros(len(fams_present)),
            "fam_tot": np.zeros(len(fams_present)),
            "famq_res": np.zeros((len(fams_present), grid.n_q)),
            "famq_tot": np.zeros((len(fams_present), grid.n_q)),
            "cell_res": np.full(n_cells, np.nan),
            "cell_tot": np.full(n_cells, np.nan),
            "ss_res_ambient": 0.0,
            "ss_tot_ambient": 0.0,
            # per-cell scatter inputs (§6 exploratory dump): predicted vs actual
            # projected on the fold's top target PC + per-cell centered cosine.
            "cell_pred_pc1": np.full(n_cells, np.nan),
            "cell_act_pc1": np.full(n_cells, np.nan),
            "cell_cos": np.full(n_cells, np.nan),
        }
        for arm in [*arms, "arm_blend"]
    }
    null_res = {arm: np.zeros(n_perms) for arm in arms} if perm is not None else {}
    null_tot = {arm: np.zeros(n_perms) for arm in arms} if perm is not None else {}
    blend_null_res = np.zeros(n_perms) if (perm is not None and extended_nulls) else None
    blend_null_tot = np.zeros(n_perms) if (perm is not None and extended_nulls) else None
    valid_list = np.arange(n_cells)[grid.valid.numpy()]
    pos_of = {int(c): i for i, c in enumerate(valid_list)}

    for fold in folds["primary"]:
        tr, te = fold["train"], fold["test"]
        if len(tr) < 8 or len(te) < 2:
            continue
        seed_fold = SEED * 1000 + layer * 100 + fold["fold_id"]
        # Shared train-fold target PCA (one basis per (genre, layer, fold)).
        Ytr_amb = grid.target[tr, layer, :].double().to(dev)
        Yte_amb = grid.target[te, layer, :].double().to(dev)
        torch.manual_seed(seed_fold)
        mu_y = Ytr_amb.mean(0, keepdim=True)
        qdim = min(PCA_DIM, Ytr_amb.shape[0] - 1, Ytr_amb.shape[1])
        _u, _s, Vy = torch.pca_lowrank(Ytr_amb - mu_y, q=qdim, center=False, niter=2)
        Ytr = (Ytr_amb - mu_y) @ Vy
        Yte = (Yte_amb - mu_y) @ Vy
        ymu = Ytr.mean(0, keepdim=True)
        ss_tot_cells = ((Yte - ymu) ** 2).sum(dim=1).cpu().numpy()
        # Ambient-space secondary read (§6 DV table): back-project predictions
        # with the fold basis; baseline = the same train mean in ambient space.
        base_amb = mu_y + ymu @ Vy.T  # (1, H)
        ss_tot_amb_cells = ((Yte_amb - base_amb) ** 2).sum(dim=1).cpu().numpy()

        # Fold-basis reduction of ALL valid cells (shared by every arm's nulls).
        if perm is not None:
            Yall_amb = grid.target[valid_list, layer, :].double().to(dev)
            Yall = (Yall_amb - mu_y) @ Vy
            Yall32 = Yall.float()
            tr_pos = np.array([pos_of[int(c)] for c in tr])
            te_pos = np.array([pos_of[int(c)] for c in te])

        # Blend inner-split indices — HOISTED above the arm loop so the blend
        # null battery can retain the outer per-draw component predictions;
        # same rng draws/order as the original post-arm-loop computation, so
        # the OBSERVED blend numbers are byte-identical.
        blend_ok = False
        itr = ival = None
        inner_fam = None
        if "arm_ctx" in arms and "arm_qry_i" in arms:
            rng = np.random.default_rng(SEED + fold["fold_id"])
            inner_fam_pool = [f for f in fams_present if f != fold["family"]]
            inner_fam = inner_fam_pool[fold["fold_id"] % len(inner_fam_pool)]
            tr_qs = np.unique(grid.q_of(tr))
            val_qs = set(rng.choice(tr_qs, size=max(1, len(tr_qs) // 4), replace=False).tolist())
            fam_of_tr = np.array([grid.families[c] for c in grid.ctx_of(tr)])
            q_of_tr = grid.q_of(tr)
            is_val = (fam_of_tr == inner_fam) | np.isin(q_of_tr, list(val_qs))
            itr, ival = tr[~is_val], tr[is_val]
            blend_ok = len(itr) >= 8 and len(ival) >= 2
        blend_nulls_active = (
            extended_nulls and blend_ok and perm is not None and blend_null_this_layer
        )
        outer_null_preds: dict[str, torch.Tensor] = {}
        if blend_nulls_active:
            for a2 in ("arm_ctx", "arm_qry_i"):
                outer_null_preds[a2] = torch.empty(
                    n_perms, len(te), Ytr.shape[1], dtype=torch.float32, device=dev
                )

        arm_test_preds: dict[str, torch.Tensor] = {}
        for arm in arms:
            Xtr, Xte = build_arm_design(arm, layer, grid, tr, grid, te, fctx, fqry, seed_fold)
            fit = press_fit_predict(
                Xtr.to(dev), Ytr, Xte.to(dev), return_engine=perm is not None, standardize=False
            )
            pred = fit["pred"]
            arm_test_preds[arm] = pred
            res_cells = ((Yte - pred) ** 2).sum(dim=1).cpu().numpy()
            a = acc[arm]
            a["ss_res"] += float(res_cells.sum())
            a["ss_tot"] += float(ss_tot_cells.sum())
            pred_amb = mu_y + pred @ Vy.T
            a["ss_res_ambient"] += float(((Yte_amb - pred_amb) ** 2).sum())
            a["ss_tot_ambient"] += float(ss_tot_amb_cells.sum())
            pc = (pred - ymu)[:, 0].cpu().numpy()
            ac = (Yte - ymu)[:, 0].cpu().numpy()
            cosv = torch.nn.functional.cosine_similarity(pred - ymu, Yte - ymu, dim=1).cpu().numpy()
            a["cell_pred_pc1"][te] = pc
            a["cell_act_pc1"][te] = ac
            a["cell_cos"][te] = cosv
            for li in range(nlam):
                a["ss_res_lambda"][li] += float(((Yte - fit["per_lambda_pred"][li]) ** 2).sum())
            fi = fam_index[fold["family"]]
            a["fam_res"][fi] += float(res_cells.sum())
            a["fam_tot"][fi] += float(ss_tot_cells.sum())
            q_of_te = grid.q_of(te)
            np.add.at(a["famq_res"][fi], q_of_te, res_cells)
            np.add.at(a["famq_tot"][fi], q_of_te, ss_tot_cells)
            a["cell_res"][te] = res_cells
            a["cell_tot"][te] = ss_tot_cells

            # Nulls: same X factors, permuted reduced target, per-draw λ re-select.
            # fp32 copy for the draw battery (bands tolerate fp32; observed fits
            # stay fp64 — see PressRidge.cast).
            if perm is not None:
                eng, _xtr_n, Xte_n = fit["engine"]
                eng32 = eng.cast(torch.float32)
                Xte32 = Xte_n.float()
                for c0 in range(0, n_perms, NULL_CHUNK):
                    pb = torch.from_numpy(perm[c0 : c0 + NULL_CHUNK]).long().to(dev)
                    Ytr_b = Yall32[pb[:, tr_pos]]  # (B, m, P)
                    Yte_b = Yall32[pb[:, te_pos]]  # (B, n_te, P)
                    ymu_b = Ytr_b.mean(dim=1, keepdim=True)
                    mse_b, G_b = eng32.press_mse(Ytr_b - ymu_b)
                    lam_b = torch.argmin(mse_b, dim=1)
                    pred_b = eng32.predict(G_b, lam_b, Xte32) + ymu_b
                    r = ((Yte_b - pred_b) ** 2).sum(dim=(1, 2)).double().cpu().numpy()
                    t = ((Yte_b - ymu_b) ** 2).sum(dim=(1, 2)).double().cpu().numpy()
                    null_res[arm][c0 : c0 + len(r)] += r
                    null_tot[arm][c0 : c0 + len(t)] += t
                    if arm in outer_null_preds:
                        outer_null_preds[arm][c0 : c0 + len(r)] = pred_b

        # Blend (§4.2 inner-split protocol; derived arm, ctx + qry_i).
        if "arm_ctx" in arms and "arm_qry_i" in arms and blend_ok:
            Yitr = (grid.target[itr, layer, :].double().to(dev) - mu_y) @ Vy
            Yival = (grid.target[ival, layer, :].double().to(dev) - mu_y) @ Vy
            preds_val = {}
            inner_engines = {}
            for arm in ("arm_ctx", "arm_qry_i"):
                Xi, Xv = build_arm_design(arm, layer, grid, itr, grid, ival, fctx, fqry, seed_fold)
                fit_i = press_fit_predict(
                    Xi.to(dev),
                    Yitr,
                    Xv.to(dev),
                    return_engine=blend_nulls_active,
                    standardize=False,
                )
                preds_val[arm] = fit_i["pred"]
                if blend_nulls_active:
                    inner_engines[arm] = fit_i["engine"]
            pc, pq = preds_val["arm_ctx"], preds_val["arm_qry_i"]
            m00 = float((pc * pc).sum())
            m01 = float((pc * pq).sum())
            m11 = float((pq * pq).sum())
            b0 = float((pc * Yival).sum())
            b1 = float((pq * Yival).sum())
            det = m00 * m11 - m01 * m01
            if abs(det) > 1e-12:
                alpha = (b0 * m11 - b1 * m01) / det
                beta = (m00 * b1 - m01 * b0) / det
            else:
                alpha, beta = 0.5, 0.5
            pred_blend = alpha * arm_test_preds["arm_ctx"] + beta * arm_test_preds["arm_qry_i"]
            res_cells = ((Yte - pred_blend) ** 2).sum(dim=1).cpu().numpy()
            a = acc["arm_blend"]
            a["ss_res"] += float(res_cells.sum())
            a["ss_tot"] += float(ss_tot_cells.sum())
            pred_amb = mu_y + pred_blend @ Vy.T
            a["ss_res_ambient"] += float(((Yte_amb - pred_amb) ** 2).sum())
            a["ss_tot_ambient"] += float(ss_tot_amb_cells.sum())
            fi = fam_index[fold["family"]]
            a["fam_res"][fi] += float(res_cells.sum())
            a["fam_tot"][fi] += float(ss_tot_cells.sum())
            a["cell_res"][te] = res_cells
            a["cell_tot"][te] = ss_tot_cells
            res["blend"].setdefault("per_fold", []).append(
                {
                    "fold_id": fold["fold_id"],
                    "alpha": alpha,
                    "beta": beta,
                    "inner_val_family": inner_fam,
                    "n_inner_val": len(ival),
                }
            )

            # Blend NULLS (§4.1 extended coverage): per draw, the SAME
            # pipeline as the observed blend — 2 inner-split component
            # refits (λ re-selected per draw on shared X-side factors) +
            # a batched per-draw 2-parameter LS; the outer component
            # predictions were retained from the per-arm battery above.
            if blend_nulls_active:
                itr_pos = np.array([pos_of[int(c)] for c in itr])
                ival_pos = np.array([pos_of[int(c)] for c in ival])
                eng32_i = {a2: inner_engines[a2][0].cast(torch.float32) for a2 in inner_engines}
                xv32 = {a2: inner_engines[a2][2].float() for a2 in inner_engines}
                for c0 in range(0, n_perms, NULL_CHUNK):
                    pb = torch.from_numpy(perm[c0 : c0 + NULL_CHUNK]).long().to(dev)
                    nb = pb.shape[0]
                    Yitr_b = Yall32[pb[:, itr_pos]]  # (B, m_i, P)
                    Yival_b = Yall32[pb[:, ival_pos]]  # (B, n_ival, P)
                    ymu_i = Yitr_b.mean(dim=1, keepdim=True)
                    pv = {}
                    for a2 in ("arm_ctx", "arm_qry_i"):
                        mse_b, G_b = eng32_i[a2].press_mse(Yitr_b - ymu_i)
                        lam_b = torch.argmin(mse_b, dim=1)
                        pv[a2] = eng32_i[a2].predict(G_b, lam_b, xv32[a2]) + ymu_i
                    pc_b, pq_b = pv["arm_ctx"], pv["arm_qry_i"]
                    m00 = (pc_b * pc_b).sum(dim=(1, 2))
                    m01 = (pc_b * pq_b).sum(dim=(1, 2))
                    m11 = (pq_b * pq_b).sum(dim=(1, 2))
                    b0 = (pc_b * Yival_b).sum(dim=(1, 2))
                    b1 = (pq_b * Yival_b).sum(dim=(1, 2))
                    det = m00 * m11 - m01 * m01
                    okd = det.abs() > 1e-12
                    safe_det = torch.where(okd, det, torch.ones_like(det))
                    half = torch.full_like(det, 0.5)
                    alpha_b = torch.where(okd, (b0 * m11 - b1 * m01) / safe_det, half)
                    beta_b = torch.where(okd, (m00 * b1 - m01 * b0) / safe_det, half)
                    pred_blend_b = (
                        alpha_b.view(-1, 1, 1) * outer_null_preds["arm_ctx"][c0 : c0 + nb]
                        + beta_b.view(-1, 1, 1) * outer_null_preds["arm_qry_i"][c0 : c0 + nb]
                    )
                    Ytr_b = Yall32[pb[:, tr_pos]]
                    Yte_b = Yall32[pb[:, te_pos]]
                    ymu_b = Ytr_b.mean(dim=1, keepdim=True)
                    r = ((Yte_b - pred_blend_b) ** 2).sum(dim=(1, 2)).double().cpu().numpy()
                    t = ((Yte_b - ymu_b) ** 2).sum(dim=(1, 2)).double().cpu().numpy()
                    blend_null_res[c0 : c0 + len(r)] += r
                    blend_null_tot[c0 : c0 + len(t)] += t

    # Marginal + OOD reads (exploratory; no nulls, no per-λ).
    def _plain_scheme(name: str, fold_list: list[dict], grid_te: GridData | None = None):
        gte = grid_te or grid
        sub: dict[str, dict] = {arm: {"ss_res": 0.0, "ss_tot": 0.0} for arm in arms}
        for fold in fold_list:
            tr = fold["train"]
            te = fold["test"]
            if len(tr) < 8 or len(te) < 2:
                continue
            # Deterministic per-scheme seed (hash() is process-salted — never use).
            scheme_off = {"lofo_marginal": 71, "qfold_marginal": 72, "ood_dolly": 73}[name]
            seedf = SEED * 1000 + layer * 100 + scheme_off
            Ytr_amb = grid.target[tr, layer, :].double().to(dev)
            Yte_amb = gte.target[te, layer, :].double().to(dev)
            torch.manual_seed(seedf)
            mu_y = Ytr_amb.mean(0, keepdim=True)
            qdim = min(PCA_DIM, Ytr_amb.shape[0] - 1, Ytr_amb.shape[1])
            _u2, _s2, Vy = torch.pca_lowrank(Ytr_amb - mu_y, q=qdim, center=False, niter=2)
            Ytr = (Ytr_amb - mu_y) @ Vy
            Yte = (Yte_amb - mu_y) @ Vy
            ymu = Ytr.mean(0, keepdim=True)
            for arm in arms:
                if arm in ("arm_qry_iii", "arm_concat_iii") and gte.fqryiii is None:
                    continue
                Xtr, Xte = build_arm_design(arm, layer, grid, tr, gte, te, fctx, fqry, seedf)
                fit = press_fit_predict(Xtr.to(dev), Ytr, Xte.to(dev), standardize=False)
                sub[arm]["ss_res"] += float(((Yte - fit["pred"]) ** 2).sum())
                sub[arm]["ss_tot"] += float(((Yte - ymu) ** 2).sum())
        res[name] = {
            arm: {
                **v,
                "skill": (1.0 - v["ss_res"] / v["ss_tot"]) if v["ss_tot"] > 0 else float("nan"),
            }
            for arm, v in sub.items()
        }

    _plain_scheme("lofo_marginal", folds["lofo_marginal"])
    _plain_scheme("qfold_marginal", folds["qfold_marginal"])
    if ood and genre == "uc" and "dolly" in grids:
        dolly = grids["dolly"]
        ood_folds = []
        d_cells = np.arange(dolly.n_ctx * dolly.n_q)[dolly.valid.numpy()]
        d_fam = np.array([dolly.families[c] for c in dolly.ctx_of(d_cells)])
        for f in folds["lofo_marginal"]:
            ood_folds.append({"train": f["train"], "test": d_cells[d_fam == f["family"]]})
        _plain_scheme("ood_dolly", ood_folds, grid_te=dolly)

        # Dolly OOD NULLS (§4.1 extended coverage): per draw, an INDEPENDENT
        # within-grid cell-label permutation of the train grid (``perm``, the
        # uc matrix the primary nulls use) and of the Dolly TEST grid
        # (``perm_ood``); X fixed, fold target-PCA basis from the OBSERVED
        # train targets (the #810 shared-pipeline-factors recipe), λ
        # re-selected per draw. The two-grid within-grid permutation preserves
        # each grid's marginal distribution under H0 (§11).
        if extended_nulls and perm is not None and perm_ood is not None:
            null_ood_res = {arm: np.zeros(n_perms) for arm in arms}
            null_ood_tot = {arm: np.zeros(n_perms) for arm in arms}
            d_valid_list = d_cells
            d_pos_of = {int(c): i for i, c in enumerate(d_valid_list)}
            for fold in ood_folds:
                tr, te = fold["train"], fold["test"]
                if len(tr) < 8 or len(te) < 2:
                    continue
                seedf = SEED * 1000 + layer * 100 + 73  # the observed ood scheme seed
                Ytr_amb = grid.target[tr, layer, :].double().to(dev)
                torch.manual_seed(seedf)
                mu_y = Ytr_amb.mean(0, keepdim=True)
                qdim = min(PCA_DIM, Ytr_amb.shape[0] - 1, Ytr_amb.shape[1])
                _u4, _s4, Vy = torch.pca_lowrank(Ytr_amb - mu_y, q=qdim, center=False, niter=2)
                Ytr = (Ytr_amb - mu_y) @ Vy
                Yall_tr32 = (
                    ((grid.target[valid_list, layer, :].double().to(dev)) - mu_y) @ Vy
                ).float()
                Yall_te32 = (
                    ((dolly.target[d_valid_list, layer, :].double().to(dev)) - mu_y) @ Vy
                ).float()
                tr_pos = np.array([pos_of[int(c)] for c in tr])
                te_pos_d = np.array([d_pos_of[int(c)] for c in te])
                for arm in arms:
                    if arm in ("arm_qry_iii", "arm_concat_iii") and dolly.fqryiii is None:
                        continue
                    Xtr, Xte = build_arm_design(arm, layer, grid, tr, dolly, te, fctx, fqry, seedf)
                    fit = press_fit_predict(
                        Xtr.to(dev), Ytr, Xte.to(dev), return_engine=True, standardize=False
                    )
                    eng32 = fit["engine"][0].cast(torch.float32)
                    xte32 = fit["engine"][2].float()
                    for c0 in range(0, n_perms, NULL_CHUNK):
                        pb_tr = torch.from_numpy(perm[c0 : c0 + NULL_CHUNK]).long().to(dev)
                        pb_te = torch.from_numpy(perm_ood[c0 : c0 + NULL_CHUNK]).long().to(dev)
                        Ytr_b = Yall_tr32[pb_tr[:, tr_pos]]
                        Yte_b = Yall_te32[pb_te[:, te_pos_d]]
                        ymu_b = Ytr_b.mean(dim=1, keepdim=True)
                        mse_b, G_b = eng32.press_mse(Ytr_b - ymu_b)
                        lam_b = torch.argmin(mse_b, dim=1)
                        pred_b = eng32.predict(G_b, lam_b, xte32) + ymu_b
                        r = ((Yte_b - pred_b) ** 2).sum(dim=(1, 2)).double().cpu().numpy()
                        t = ((Yte_b - ymu_b) ** 2).sum(dim=(1, 2)).double().cpu().numpy()
                        null_ood_res[arm][c0 : c0 + len(r)] += r
                        null_ood_tot[arm][c0 : c0 + len(t)] += t
            with np.errstate(divide="ignore", invalid="ignore"):
                res["null_ood"] = {
                    arm: (1.0 - null_ood_res[arm] / null_ood_tot[arm]).tolist()
                    for arm in arms
                    if null_ood_tot[arm].sum() > 0
                }

    for arm in [*arms, "arm_blend"]:
        a = acc[arm]
        a["skill"] = (1.0 - a["ss_res"] / a["ss_tot"]) if a["ss_tot"] > 0 else float("nan")
        a["skill_ambient"] = (
            (1.0 - a["ss_res_ambient"] / a["ss_tot_ambient"])
            if a["ss_tot_ambient"] > 0
            else float("nan")
        )
        a["skill_per_lambda"] = [
            (1.0 - a["ss_res_lambda"][li] / a["ss_tot"]) if a["ss_tot"] > 0 else float("nan")
            for li in range(nlam)
        ]
        res["arms"][arm] = a
    if perm is not None:
        for arm in arms:
            with np.errstate(divide="ignore", invalid="ignore"):
                res["null"][arm] = (1.0 - null_res[arm] / null_tot[arm]).tolist()
        if blend_null_res is not None and blend_null_tot.sum() > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                res["null"]["arm_blend"] = (1.0 - blend_null_res / blend_null_tot).tolist()
    res["families_present"] = fams_present
    # ANOVA is a function of TARGETS only; under feature_source=pool the parent
    # round's per-(genre, layer) shares are injected (identical targets — §4.1
    # "cites parent") instead of recomputing an identical artifact.
    res["anova"] = (
        anova_override
        if anova_override is not None
        else anova_shares(grid, layer, SEED * 1000 + layer)
    )
    if smoke_blend_assert and "arm_blend" in acc and "arm_ctx" in res["arms"]:
        singles_min = min(res["arms"]["arm_ctx"]["skill"], res["arms"]["arm_qry_i"]["skill"])
        assert res["arms"]["arm_blend"]["skill"] >= singles_min - 0.15, (
            "blend smoke gate: blend skill "
            f"{res['arms']['arm_blend']['skill']:.3f} << min(singles) {singles_min:.3f}"
        )
    return res


# ── stats over layers (bootstrap, ρ_dec, headline) ────────────────────────────


def family_bootstrap(fam_res, fam_tot, counts: np.ndarray) -> np.ndarray:
    """Per-draw pooled skill under family-cluster resampling (one GEMM)."""
    num = counts @ np.asarray(fam_res, dtype=np.float64)  # (n_boot,)
    den = counts @ np.asarray(fam_tot, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - num / den


# ── paired residual diff vs the parent round (pooled-span-features, plan v6) ──

PARENT_QCOLS = {"uc": 144, "betley": 48}  # famq column counts (fold_assignments n_queries)


def _replay_family_counts(
    n_boot: int, genre_specs: list[tuple[str, int, int]]
) -> dict[str, np.ndarray]:
    """Regenerate ``compute_stats``' family-count draws EXACTLY (shared-draw contract).

    Replays the seed-42 rng CALL ORDER of ``compute_stats`` — per genre in
    order: the (n_boot, nf) family picks, then the (n_boot, qcols)
    cross-classified query picks — so the counts returned here are
    bit-identical to the ones BOTH rounds' ``compute_stats`` used.
    """
    rng = np.random.default_rng(SEED)
    counts_by_genre: dict[str, np.ndarray] = {}
    for genre, nf, qcols in genre_specs:
        picks = rng.integers(0, nf, size=(n_boot, nf))
        counts = np.zeros((n_boot, nf))
        for j in range(nf):
            counts[:, j] = (picks == j).sum(axis=1)
        counts_by_genre[genre] = counts
        _ = rng.integers(0, qcols, size=(n_boot, qcols))  # advance: the vpicks draw
    return counts_by_genre


def verdict_lattice(delta_pool_ci: list[float], paired_ci: list[float]) -> dict:
    """§3 DISJOINT verdict lattice — exactly one of {H-robust, H-slot, intermediate}.

    H-robust ⇔ the Δ_pool CI is wholly below 0; H-slot ⇔ (Δ_pool CI wholly
    at/above 0) OR (Δ_pool CI straddles 0 AND the paired-diff CI is strictly
    positive); intermediate ⇔ otherwise. When H-robust fires WITH a strictly
    positive paired-diff CI, the note records "deficit persists with partial
    closure" (§3 — never reported as gap closure).
    """
    lo, hi = float(delta_pool_ci[0]), float(delta_pool_ci[1])
    plo, phi = float(paired_ci[0]), float(paired_ci[1])
    wholly_below = hi < 0.0
    wholly_at_or_above = lo >= 0.0
    straddle = (not wholly_below) and (not wholly_at_or_above)
    paired_strictly_positive = plo > 0.0
    if wholly_below:
        label = "H-robust"
        note = "deficit persists with partial closure" if paired_strictly_positive else None
    elif wholly_at_or_above or (straddle and paired_strictly_positive):
        label = "H-slot"
        note = None
    else:
        label = "intermediate"
        note = None
    return {
        "label": label,
        "note": note,
        "delta_pool_ci95": [lo, hi],
        "paired_diff_ci95": [plo, phi],
        "predicates": {
            "wholly_below": wholly_below,
            "wholly_at_or_above": wholly_at_or_above,
            "straddle": straddle,
            "paired_strictly_positive": paired_strictly_positive,
        },
    }


def paired_residual_diff(
    pooled_fams: dict | None,
    parent_fits_dir: Path,
    n_boot: int,
    kill_floor: dict[str, bool] | None = None,
) -> dict:
    """Paired family bootstrap of D = Δ_pool − Δ_last on SHARED seed-42 draws.

    Pairs on the parent's persisted PER-FAMILY ``fam_res``/``fam_tot`` (paths
    ``genres.{uc,betley}.18.arms.{arm_full,arm_concat_i}`` in
    ``decomposition_skill.json`` — per-cell residuals are NOT in the JSON;
    the family sums ARE the exact sufficient statistic for the family
    bootstrap). Shared family-count draws (seed 42, replayed via
    ``_replay_family_counts``) are applied to BOTH rounds' sums.

    REPRODUCE-CHECK FIRST (v6 Must-Fix 1): per genre, the parent family sums
    must reproduce ``headline.json``'s Δ_last AND its family-bootstrap 95% CI
    to <= 1e-12 before ANY pairing; a mismatch fails loud (broken pairing
    premise, never silently paired).

    ``pooled_fams``: {genre: {arm: (fam_res, fam_tot)}} for this round at the
    frozen headline layer; a genre absent from it records a skipped pairing.

    ``kill_floor`` (§6 k2 kill rule, ENFORCED — r2 Major): per-genre flags from
    ``compute_stats``' ``kill_floor_triggered`` (pool_full R² < 0.05 at EVERY
    layer). A triggered genre keeps its paired numbers (persist-by-default
    diagnostics) but gets ``verdict: None`` + ``verdict_skipped_reason:
    "k2_pool_full_floor"`` — an uninformative read is never labeled
    H-slot/H-robust/intermediate. The top-level primary verdict (= UC) is
    skipped the same way when UC's floor triggered.

    Returns the ``paired_diff`` payload incl. the §3 disjoint verdict (primary
    verdict = UC) or its k2 skip record.
    """
    parent_skill = load_json(parent_fits_dir / "decomposition_skill.json")
    parent_head = load_json(parent_fits_dir / "headline.json")
    n_boot_parent = int(parent_head["stats"]["n_boot"])
    assert n_boot_parent == n_boot, (
        f"paired diff requires the parent's n_boot ({n_boot_parent}); got {n_boot}"
    )
    parent_genres = [g for g in ("uc", "betley") if g in parent_skill["genres"]]
    genre_specs = []
    for g in parent_genres:
        fams = parent_head["stats"][g]["families"]
        genre_specs.append((g, len(fams), PARENT_QCOLS[g]))
    counts_by_genre = _replay_family_counts(n_boot, genre_specs)
    hl = str(HEADLINE_LAYER)
    out: dict = {"n_boot": n_boot, "seed": SEED, "genres": {}}
    for g, _nf, _q in genre_specs:
        counts = counts_by_genre[g]
        p18 = parent_skill["genres"][g][hl]["arms"]

        def _sums(node: dict, arm: str) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.asarray(node[arm]["fam_res"], dtype=np.float64),
                np.asarray(node[arm]["fam_tot"], dtype=np.float64),
            )

        pf_res, pf_tot = _sums(p18, "arm_full")
        pc_res, pc_tot = _sums(p18, "arm_concat_i")
        delta_last_obs = (1.0 - pf_res.sum() / pf_tot.sum()) - (1.0 - pc_res.sum() / pc_tot.sum())
        delta_last_draws = family_bootstrap(pf_res, pf_tot, counts) - family_bootstrap(
            pc_res, pc_tot, counts
        )
        ci_last = [
            float(np.nanpercentile(delta_last_draws, 2.5)),
            float(np.nanpercentile(delta_last_draws, 97.5)),
        ]
        ref = parent_head["stats"][g]["delta_r2"]
        err = max(
            abs(delta_last_obs - ref["value"]),
            abs(ci_last[0] - ref["ci95"][0]),
            abs(ci_last[1] - ref["ci95"][1]),
        )
        assert err <= 1e-12, (
            f"paired-diff reproduce-check FAILED for {g}: max abs err {err} > 1e-12 "
            f"(Δ_last {delta_last_obs} vs {ref['value']}; CI {ci_last} vs {ref['ci95']}) — "
            "the family sums do not reproduce the parent headline; pairing premise broken"
        )
        entry: dict = {
            "reproduce_check": {
                "delta_last": delta_last_obs,
                "ci95": ci_last,
                "max_abs_err": err,
                "pass": True,
            }
        }
        if pooled_fams is None or g not in pooled_fams:
            entry["paired"] = None
            entry["note"] = "pooled side absent for this genre — pairing skipped"
            out["genres"][g] = entry
            continue
        qf_res, qf_tot = pooled_fams[g]["arm_full"]
        qc_res, qc_tot = pooled_fams[g]["arm_concat_i"]
        delta_pool_obs = (1.0 - np.asarray(qf_res).sum() / np.asarray(qf_tot).sum()) - (
            1.0 - np.asarray(qc_res).sum() / np.asarray(qc_tot).sum()
        )
        delta_pool_draws = family_bootstrap(qf_res, qf_tot, counts) - family_bootstrap(
            qc_res, qc_tot, counts
        )
        d_draws = delta_pool_draws - delta_last_draws
        ci_pool = [
            float(np.nanpercentile(delta_pool_draws, 2.5)),
            float(np.nanpercentile(delta_pool_draws, 97.5)),
        ]
        ci_d = [float(np.nanpercentile(d_draws, 2.5)), float(np.nanpercentile(d_draws, 97.5))]
        with np.errstate(divide="ignore", invalid="ignore"):
            closure_draws = 1.0 - delta_pool_draws / delta_last_draws
        closure_draws = np.where(np.abs(delta_last_draws) < 1e-9, np.nan, closure_draws)
        closure_obs = (
            float(1.0 - delta_pool_obs / delta_last_obs) if delta_last_obs != 0 else float("nan")
        )
        entry["paired"] = {
            "delta_pool": float(delta_pool_obs),
            "delta_pool_ci95": ci_pool,
            "delta_last": float(delta_last_obs),
            "delta_last_ci95": ci_last,
            "D_value": float(delta_pool_obs - delta_last_obs),
            "D_ci95": ci_d,
            "closure_fraction": closure_obs,
            "closure_ci95": [
                float(np.nanpercentile(closure_draws, 2.5)),
                float(np.nanpercentile(closure_draws, 97.5)),
            ],
            "n_closure_draws_dropped": int(np.isnan(closure_draws).sum()),
            "closure_draws": closure_draws.tolist(),  # figure input (paired draws)
            "D_draws": d_draws.tolist(),
            "parent_null_note": (
                "parent side un-gated (no registered null in the parent round) for "
                "blend/Dolly surfaces"
            ),
        }
        if kill_floor and kill_floor.get(g):
            # §6 k2 kill rule: pool_full below the 0.05 power floor at every
            # layer — record the paired diagnostics, label NOTHING.
            entry["verdict"] = None
            entry["verdict_skipped_reason"] = "k2_pool_full_floor"
        else:
            entry["verdict"] = verdict_lattice(ci_pool, ci_d)
        out["genres"][g] = entry
    uc_entry = out["genres"].get("uc", {})
    if uc_entry.get("verdict"):
        out["verdict"] = uc_entry["verdict"]["label"]  # §3 primary = UC at frozen L18
        out["verdict_note"] = uc_entry["verdict"]["note"]
    elif uc_entry.get("verdict_skipped_reason"):
        out["verdict_skipped_reason"] = uc_entry["verdict_skipped_reason"]
    return out


def kill_floor_flags(stats: dict, genres: list[str]) -> dict[str, bool]:
    """Per-genre §6 k2 flags from ``compute_stats`` output.

    ``kill_floor_triggered`` = pool_full held-out R² < 0.05 at EVERY fitted
    layer for that genre; absent genres default False (no floor evidence).
    """
    return {g: bool(stats.get(g, {}).get("kill_floor_triggered")) for g in genres}


def headline_payload(meta: dict, stats: dict, paired: dict | None) -> dict:
    """Assemble the ``headline.json`` payload, ENFORCING the §6 k2 kill rule.

    When the paired diff carries a primary (UC) verdict, it is mirrored at the
    top level (parent behavior). When the verdict was k2-SKIPPED
    (``verdict_skipped_reason`` set, ``verdict`` None/absent), the payload has
    NO top-level ``verdict`` key at all — only ``verdict_skipped_reason`` — so
    an uninformative pooled read can never be published as adjudicated
    (r2 Major: k2 recorded but not enforced).
    """
    payload: dict = {"meta": meta, "stats": stats}
    if paired:
        payload["paired_diff"] = paired
        if paired.get("verdict") is not None:
            payload["verdict"] = paired["verdict"]
        elif paired.get("verdict_skipped_reason"):
            payload["verdict_skipped_reason"] = paired["verdict_skipped_reason"]
    return payload


def compute_stats(units: dict, arms: list[str], n_boot: int, genres: list[str]) -> dict:
    """Bootstrap CIs + ΔR²/ρ_dec (+ guard) + H3 cross-classified + oracle checks."""
    rng = np.random.default_rng(SEED)
    stats: dict = {"headline_layer": HEADLINE_LAYER, "n_boot": n_boot}
    for genre in genres:
        layers = sorted({u["layer"] for u in units.values() if u["genre"] == genre})
        # Frozen pre-registered headline layer 18 (#722 peak); a truncated smoke
        # sweep falls back to its top layer and RECORDS the substitution.
        hl = HEADLINE_LAYER if (genre, HEADLINE_LAYER) in units else max(layers)
        u18 = units[(genre, hl)]
        fams = u18["families_present"]
        nf = len(fams)
        picks = rng.integers(0, nf, size=(n_boot, nf))
        counts = np.zeros((n_boot, nf))
        for j in range(nf):
            counts[:, j] = (picks == j).sum(axis=1)
        g: dict = {"layers": layers, "families": fams, "headline_layer_used": hl}
        boot18 = {}
        for arm in [*arms, "arm_blend"]:
            a = u18["arms"][arm]
            draws = family_bootstrap(a["fam_res"], a["fam_tot"], counts)
            boot18[arm] = draws
            g.setdefault("L18", {})[arm] = {
                "skill": a["skill"],
                "ci95": [float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))],
                "se": float(np.nanstd(draws)),
            }
        # ΔR² + ρ_dec (per-draw best-single argmax + §3 denominator guard).
        singles = np.stack([boot18[s] for s in SINGLES_FOR_RHO_DEC])  # (2, n_boot)
        best_single_draws = singles.max(axis=0)
        best_single_which = np.array(SINGLES_FOR_RHO_DEC)[singles.argmax(axis=0)]
        num_draws = boot18["arm_concat_i"] - best_single_draws
        den_draws = boot18[ARM_FULL] - best_single_draws
        delta_draws = boot18[ARM_FULL] - boot18["arm_concat_i"]
        obs = {arm: u18["arms"][arm]["skill"] for arm in [*arms, "arm_blend"]}
        obs_best = max(obs[s] for s in SINGLES_FOR_RHO_DEC)
        D = obs[ARM_FULL] - obs_best
        se_D = float(np.nanstd(den_draws))
        guard = max(0.02, 2.0 * se_D)
        rho_defined = guard < D
        g["delta_r2"] = {
            "value": obs[ARM_FULL] - obs["arm_concat_i"],
            "ci95": [
                float(np.nanpercentile(delta_draws, 2.5)),
                float(np.nanpercentile(delta_draws, 97.5)),
            ],
            "se": float(np.nanstd(delta_draws)),
        }
        g["rho_dec"] = {
            "defined": bool(rho_defined),
            "value": ((obs["arm_concat_i"] - obs_best) / D) if rho_defined else None,
            "denominator_D": D,
            "guard_threshold": guard,
            "se_D": se_D,
            "singles_set": list(SINGLES_FOR_RHO_DEC),
            "per_draw_numerator": num_draws.tolist(),
            "per_draw_denominator": den_draws.tolist(),
            "per_draw_best_single_arm": best_single_which.tolist(),
            "note": None
            if rho_defined
            else "undefined — no measurable best-single→full gap (§3 guard)",
        }
        # H3: cross-classified family x query bootstrap (primary) + family-only.
        qcols = np.asarray(u18["arms"]["arm_ctx"]["famq_res"]).shape[1]
        vpicks = rng.integers(0, qcols, size=(n_boot, qcols))
        vcounts = np.zeros((n_boot, qcols))
        for j in range(qcols):
            vcounts[:, j] = (vpicks == j).sum(axis=1)

        def _cc_skill(arm: str, _u=u18, _c=counts, _v=vcounts) -> np.ndarray:
            M_res = np.asarray(_u["arms"][arm]["famq_res"], dtype=np.float64)
            M_tot = np.asarray(_u["arms"][arm]["famq_tot"], dtype=np.float64)
            num = np.einsum("bf,fq,bq->b", _c, M_res, _v)
            den = np.einsum("bf,fq,bq->b", _c, M_tot, _v)
            with np.errstate(divide="ignore", invalid="ignore"):
                return 1.0 - num / den

        gap_cc = _cc_skill("arm_ctx") - _cc_skill("arm_qry_i")
        gap_fam = boot18["arm_ctx"] - boot18["arm_qry_i"]
        g["h3_ctx_minus_qry"] = {
            "value": obs["arm_ctx"] - obs["arm_qry_i"],
            "cross_classified_ci95": [
                float(np.nanpercentile(gap_cc, 2.5)),
                float(np.nanpercentile(gap_cc, 97.5)),
            ],
            "family_only_ci95": [
                float(np.nanpercentile(gap_fam, 2.5)),
                float(np.nanpercentile(gap_fam, 97.5)),
            ],
        }
        # Layer curves + kill floor + per-layer family-bootstrap SEs.
        curves = {arm: [] for arm in [*arms, "arm_blend"]}
        for layer in layers:
            u = units[(genre, layer)]
            for arm in [*arms, "arm_blend"]:
                curves[arm].append(u["arms"][arm]["skill"])
        g["layer_curves"] = curves
        g["kill_floor_triggered"] = bool(all(s < 0.05 for s in curves[ARM_FULL]))
        # Oracle-consistency checks (§3): tolerance-flagged anomalies, no crash.
        shares = u18["anova"]["pca48"]
        checks = []
        # Each check's tolerance uses ITS OWN read's family-bootstrap SE (the
        # §3 "SE of the held-out read"), not arm_concat_i's for all four
        # (r1 Minor: oracle tolerance used one arm's SE everywhere).
        for label, read, ceiling, read_se in (
            ("ctx_vs_ctx_share", obs["arm_ctx"], shares["share_ctx"], g["L18"]["arm_ctx"]["se"]),
            (
                "qry_vs_qry_share",
                obs["arm_qry_i"],
                shares["share_qry"],
                g["L18"]["arm_qry_i"]["se"],
            ),
            (
                "concat_vs_additive_share",
                obs["arm_concat_i"],
                shares["share_ctx"] + shares["share_qry"],
                g["L18"]["arm_concat_i"]["se"],
            ),
            (
                "delta_vs_interaction_share",
                g["delta_r2"]["value"],
                shares["share_interaction"],
                g["delta_r2"]["se"],
            ),
        ):
            tol = 2.0 * max(read_se, 1e-6)
            violation = read - ceiling
            severity = (
                "ok"
                if violation <= 0
                else (
                    "within_tolerance"
                    if violation <= tol
                    else ("anomaly" if violation <= 3 * tol else "gross_anomaly_investigate")
                )
            )
            checks.append(
                {
                    "check": label,
                    "read": read,
                    "ceiling": ceiling,
                    "violation": violation,
                    "tolerance": tol,
                    "severity": severity,
                }
            )
        g["oracle_consistency"] = checks
        stats[genre] = g
    return stats


# ── regen spot-check (plan 1e: fresh capture vs store reduction) ──────────────


def regen_check(packs_dir: Path, reduce_dir: Path) -> dict:
    """cos(fresh v̄, store-reduced v̄) per regen cell — the cross-provenance join.

    Validates joining store-reduced and fresh-captured targets (plan 1e; expect
    cos > 0.99 per layer — greedy completions are deterministic up to vLLM
    batching numerics). Drift is REPORTED, never a crash (a finding, §12 A6).
    """
    packs = _collect_shards(packs_dir, "tgt_regen")
    out_rows = []
    for genre in ("betley", "uc"):
        pack_path = reduce_dir / f"vbar_store_{genre}.pt"
        if not pack_path.exists():
            continue
        rt, rm = load_pack(pack_path)
        store_idx = {(r["ctx_id"], r["q_idx"]): i for i, r in enumerate(rm["rows"])}
        for tensors, meta in packs:
            for i, r in enumerate(meta["rows"]):
                if r.get("genre") != genre or not bool(tensors["valid"][i]):
                    continue
                j = store_idx.get((r["ctx_id"], r["q_idx"]))
                if j is None or not bool(rt["valid"][j]):
                    continue
                fresh = tensors["vbar"][i].float()  # (Lc, H)
                stored = rt["vbar"][j].float()
                cos = torch.nn.functional.cosine_similarity(fresh, stored, dim=1)
                out_rows.append(
                    {
                        "genre": genre,
                        "ctx_id": r["ctx_id"],
                        "q_idx": r["q_idx"],
                        "cos_min": float(cos.min()),
                        "cos_mean": float(cos.mean()),
                        "cos_L18": float(cos[min(HEADLINE_LAYER, cos.shape[0] - 1)]),
                    }
                )
    mins = [r["cos_min"] for r in out_rows]
    return {
        "n_cells": len(out_rows),
        "cos_min_overall": min(mins) if mins else None,
        "cos_median_of_means": (
            float(np.median([r["cos_mean"] for r in out_rows])) if out_rows else None
        ),
        "n_below_0p99": int(sum(m < 0.99 for m in mins)),
        "rows": out_rows,
    }


# ── smoke grid ────────────────────────────────────────────────────────────────


def build_smoke_inputs(tmp: Path) -> tuple[dict, torch.Tensor, dict, dict]:
    """Synthetic tiny grid with planted ctx+qry structure (same production path)."""
    torch.manual_seed(SEED)
    n_ctx, n_q, lc, hidden = 6, 8, 2, 16
    families = ["persona", "persona", "wildchat", "wildchat", "icl", "icl"]
    ctx_ids = [f"sm_ctx_{i}" for i in range(n_ctx)]
    a = torch.randn(n_ctx, hidden)
    b = torch.randn(n_q, hidden)
    fctx = torch.zeros(n_ctx, lc, hidden, dtype=torch.float16)
    fq = torch.zeros(n_q, lc, hidden, dtype=torch.float16)
    target = torch.zeros(n_ctx * n_q, lc, hidden, dtype=torch.float16)
    ffull = torch.zeros(n_ctx * n_q, lc, hidden, dtype=torch.float16)
    fiii = torch.zeros(n_ctx * n_q, lc, hidden, dtype=torch.float16)
    for li in range(lc):
        for c in range(n_ctx):
            fctx[c, li] = (a[c] + 0.1 * torch.randn(hidden)).half()
        for q in range(n_q):
            fq[q, li] = (b[q] + 0.1 * torch.randn(hidden)).half()
        for c in range(n_ctx):
            for q in range(n_q):
                row = c * n_q + q
                y = a[c] + 0.6 * b[q] + 0.2 * torch.randn(hidden)
                target[row, li] = y.half()
                ffull[row, li] = (a[c] + b[q] + 0.1 * torch.randn(hidden)).half()
                fiii[row, li] = (b[q] + 0.15 * torch.randn(hidden)).half()
    valid = torch.ones(n_ctx * n_q, dtype=torch.bool)
    valid[3] = False  # exercise the common-valid drop path
    grid = GridData("uc", ctx_ids, families, n_q, target, valid, ffull, fiii, valid.clone())
    d_target = target + 0.05 * torch.randn_like(target.float()).half()
    dolly = GridData(
        "dolly",
        ctx_ids,
        families,
        n_q,
        d_target,
        valid.clone(),
        ffull.clone(),
        fiii.clone(),
        valid.clone(),
    )
    grids = {"uc": grid, "dolly": dolly}
    fqry = {"i": {"uc": fq, "dolly": fq.clone()}, "ii": {"uc": fq.clone(), "dolly": fq.clone()}}
    folds_payload = {"query_folds": {"uc": [q % N_QUERY_FOLDS for q in range(n_q)]}}
    return grids, fctx, fqry, folds_payload


# ── production-shape timing (compute-deviation re-derivation) ─────────────────


def time_projection(device: str) -> dict:
    """Time ONE production-shape unit of each §9 fit-row kernel; project walls.

    Per the compute-deviation rule the per-call cost is RE-DERIVED (never the
    plan's figure): times pca_lowrank at (5184, 3584) q=48, one d=96 observed
    PRESS fit at m=5184/P=48, and one 100-draw null chunk, then projects the
    fit row + null row walls from the §9 multiplier products.
    """
    dev = torch.device(device)
    m, hidden, p, d = 5184, 3584, PCA_DIM, 96
    torch.manual_seed(0)
    X_amb = torch.randn(m, hidden, dtype=torch.float64, device=dev)
    t0 = time.time()
    torch.pca_lowrank(X_amb, q=PCA_DIM, center=False, niter=2)
    t_pca = time.time() - t0
    Xn = torch.randn(m, d, dtype=torch.float64, device=dev)
    Y = torch.randn(m, p, dtype=torch.float64, device=dev)
    Xte = torch.randn(504, d, dtype=torch.float64, device=dev)
    t0 = time.time()
    press_fit_predict(Xn, Y, Xte)
    t_fit = time.time() - t0
    eng = PressRidge(Xn).cast(torch.float32)  # the production null path is fp32
    B = 100
    Yb = torch.randn(B, m, p, dtype=torch.float32, device=dev)
    t0 = time.time()
    mse, G = eng.press_mse(Yb)
    lam = torch.argmin(mse, dim=1)
    eng.predict(G, lam, Xte.float())
    t_null_chunk = time.time() - t0
    # §9 multiplier products (plan): fits 14,112; per-λ preds ride the fit.
    # PCA calls: target 2*28*28 + full 2*28*28 + qry_iii 3 grids*28*folds ≈ 4700.
    n_fits = 14112
    n_pca = 4704
    n_null_units = 8 * 28 * 28 * 2  # (arm, layer, fold, genre) x 1000 draws in chunks
    proj_fit_h = (n_fits * t_fit + n_pca * t_pca) / 3600
    proj_null_h = (n_null_units * (1000 / B) * t_null_chunk) / 3600
    out = {
        "device": device,
        "t_pca_lowrank_s": t_pca,
        "t_observed_fit_s": t_fit,
        "t_null_chunk100_s": t_null_chunk,
        "per_draw_cell_s": t_null_chunk / B,
        "projected_fit_row_wall_h": proj_fit_h,
        "projected_null_row_wall_h": proj_null_h,
        "planned_fit_row_wall_h": 0.3,
        "planned_null_row_wall_h": 2.5,
        "fit_ratio": proj_fit_h / 0.3,
        "null_ratio": proj_null_h / 2.5,
        "note": "VM timing at 8 threads; cpu-mid e2-standard-8 discount ~1.5-2x slower",
    }
    print(json.dumps(out, indent=2))
    return out


# ── full-H dual-ridge spot-check (GPU, end of Phase 1) ────────────────────────


def fullh_spotcheck(packs_dir: Path, data_dir: Path, out_path: Path, device: str) -> dict:
    """2 arms x 7 LOFO folds x L18 full-H dual ridge on the pod-local uc_ext grid.

    Uses the imported #658 helpers DIRECTLY (``_press_loo_mse_per_lambda`` +
    ``_ridge_dual_weights``, Gram/dual form) on ``EPM_FIT_DEVICE`` — the §9
    robustness row: does PCA-48 input compression move the L18 read?
    """
    dev = torch.device(device)
    folds_payload = load_json(data_dir / "fold_assignments.json")
    fam_map = folds_payload["families"]
    packs = _collect_shards(packs_dir, "tgt_ucext")
    fctx_packs = _collect_shards(packs_dir, "fctx")
    rows = [r for _t, m in packs for r in m["rows"]]
    ctx_ids = sorted({r["ctx_id"] for r in rows})
    ctx_idx = {c: i for i, c in enumerate(ctx_ids)}
    n_q = max(r["q_idx"] for r in rows) + 1
    lc = packs[0][0]["vbar"].shape[1]
    layer = min(HEADLINE_LAYER, lc - 1)
    hidden = packs[0][0]["vbar"].shape[2]
    Y = torch.zeros(len(ctx_ids) * n_q, hidden)
    Xf = torch.zeros(len(ctx_ids) * n_q, hidden)
    valid = torch.zeros(len(ctx_ids) * n_q, dtype=torch.bool)
    for tensors, meta in packs:
        for i, r in enumerate(meta["rows"]):
            row = cell_row(ctx_idx[r["ctx_id"]], r["q_idx"], n_q)
            Y[row] = tensors["vbar"][i, layer].float()
            Xf[row] = tensors["flast"][i, layer].float()
            valid[row] = bool(tensors["valid"][i])
    fctx = torch.zeros(len(ctx_ids), hidden)
    for tensors, meta in fctx_packs:
        for i, r in enumerate(meta["rows"]):
            if r["ctx_id"] in ctx_idx:
                fctx[ctx_idx[r["ctx_id"]]] = tensors["flast"][i, layer].float()
    cells = np.arange(len(ctx_ids) * n_q)[valid.numpy()]
    fams = np.array([fam_map[ctx_ids[c // n_q]] for c in cells])
    out = {"layer": int(layer), "arms": {}}
    for arm, X_amb in (("arm_ctx", fctx[[c // n_q for c in cells]]), (ARM_FULL, Xf[cells])):
        ss_res = ss_tot = 0.0
        for fam in sorted(set(fams.tolist())):
            tr = cells[fams != fam]
            te = cells[fams == fam]
            tr_rows = np.where(fams != fam)[0]
            te_rows = np.where(fams == fam)[0]
            Xtr = X_amb[tr_rows].double().to(dev)
            Xte = X_amb[te_rows].double().to(dev)
            mu = Xtr.mean(0)
            sd = Xtr.std(0, correction=0) + 1e-9
            Xtr_n = (Xtr - mu) / sd
            Xte_n = (Xte - mu) / sd
            Ytr_amb = Y[tr].double().to(dev)
            Yte_amb = Y[te].double().to(dev)
            torch.manual_seed(SEED)
            mu_y = Ytr_amb.mean(0, keepdim=True)
            _u, _s, Vy = torch.pca_lowrank(Ytr_amb - mu_y, q=PCA_DIM, center=False, niter=2)
            Ytr = (Ytr_amb - mu_y) @ Vy
            Yte = (Yte_amb - mu_y) @ Vy
            ymu = Ytr.mean(0, keepdim=True)
            mse = _press_loo_mse_per_lambda(Xtr_n, Ytr - ymu, RIDGE_LAMBDAS)
            lam = RIDGE_LAMBDAS[int(torch.argmin(mse))]
            w = _ridge_dual_weights(Xtr_n, Ytr - ymu, lam)
            pred = ymu + Xte_n @ w
            ss_res += float(((Yte - pred) ** 2).sum())
            ss_tot += float(((Yte - ymu) ** 2).sum())
        out["arms"][arm] = {"skill_fullH": 1.0 - ss_res / ss_tot}
    out["metadata"] = reproducibility_metadata({"script": "issue923_fullh_spotcheck"})
    dump_json(out, out_path)
    print(json.dumps(out["arms"], indent=2))
    return out


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:  # noqa: C901 — linear phase pipeline; see [phase=...] markers
    parser = argparse.ArgumentParser(description="Issue #923 Phase 3 decomposition battery")
    parser.add_argument(
        "--packs-dir", type=Path, default=PROJECT_ROOT / "data/issue_923/capture/packs"
    )
    parser.add_argument("--reduce-dir", type=Path, default=PROJECT_ROOT / "data/issue_923/reduce")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_923/fits"
    )
    parser.add_argument(
        "--tensors-dir", type=Path, default=PROJECT_ROOT / "data/issue_923/fit_tensors"
    )
    parser.add_argument("--genres", default="uc,betley")
    parser.add_argument("--no-ood", action="store_true")
    parser.add_argument("--n-perms", type=int, default=1000)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--device", default=None, help="cpu|cuda|auto (else EPM_FIT_DEVICE)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--selftest-only", action="store_true")
    parser.add_argument("--time-projection", action="store_true")
    parser.add_argument("--fullh-spotcheck", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="ignore existing partials")
    parser.add_argument("--no-upload", action="store_true")
    # pooled-span-features round (plan v6 §4.2):
    parser.add_argument(
        "--feature-source",
        choices=("last", "pool"),
        default="last",
        help="prompt-side feature summary: last token (parent behavior, default) "
        "or the span-mean over the owning token span (pool_* packs)",
    )
    parser.add_argument(
        "--pooled-packs-dir",
        type=Path,
        default=PROJECT_ROOT / "data/issue_923/capture/packs_pooled",
        help="pooled feature packs dir (feature-source=pool only)",
    )
    parser.add_argument(
        "--parent-fits-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_923/fits",
        help="parent round fits dir (paired residual diff + anova citation)",
    )
    parser.add_argument(
        "--paired-diff-smoke",
        action="store_true",
        help="run paired_residual_diff SELF-PAIRED on the parent's real persisted "
        "family sums (reproduce-check + pairing machinery on real data) and exit",
    )
    parser.add_argument(
        "--blend-null-l18-only",
        action="store_true",
        help="§9 descope ladder rung 1: blend nulls at the L18 column only",
    )
    parser.add_argument(
        "--allow-missing-regen",
        action="store_true",
        help="DELIBERATE partial run only: skip (and RECORD skipping) the regen "
        "spot-check when tgt_regen packs are absent; without this flag absence "
        "is a coverage failure and raises (r1 blocker: unregistered fail-soft)",
    )
    args = parser.parse_args()

    device = _resolve_device(_requested_device(args.device))
    logger.info("fit device: %s", device)
    print("[phase=selftest]", flush=True)
    st = run_selftest(device="cpu")
    logger.info("PRESS/dual exactness selftest PASS: %s", st)
    if args.selftest_only:
        print(json.dumps(st, indent=2))
        print("[phase=done]", flush=True)
        return 0
    if args.paired_diff_smoke:
        # Self-pairing on the parent's REAL persisted family sums: exercises the
        # reproduce-check (<=1e-12 vs headline.json) + pairing + lattice on real
        # data; D is identically 0 by construction, verdict = the parent CI's.
        print("[phase=paired_diff_smoke]", flush=True)
        parent_skill = load_json(args.parent_fits_dir / "decomposition_skill.json")
        hl = str(HEADLINE_LAYER)
        pooled_fams = {
            g: {
                arm: (
                    np.asarray(parent_skill["genres"][g][hl]["arms"][arm]["fam_res"], float),
                    np.asarray(parent_skill["genres"][g][hl]["arms"][arm]["fam_tot"], float),
                )
                for arm in ("arm_full", "arm_concat_i")
            }
            for g in parent_skill["genres"]
        }
        paired = paired_residual_diff(pooled_fams, args.parent_fits_dir, args.n_boot)
        print(json.dumps(paired, indent=2))
        for g, entry in paired["genres"].items():
            assert entry["paired"] is not None, g
            assert abs(entry["paired"]["D_value"]) < 1e-15, (g, entry["paired"]["D_value"])
        print("[phase=done]", flush=True)
        return 0
    if args.time_projection:
        time_projection(device)
        print("[phase=done]", flush=True)
        return 0
    if args.fullh_spotcheck:
        print("[phase=fullh_spotcheck]", flush=True)
        fullh_spotcheck(
            args.packs_dir, args.data_dir, args.out_dir / "fullh_spotcheck.json", device
        )
        print("[phase=done]", flush=True)
        return 0

    pool = args.feature_source == "pool"
    out_dir: Path = args.out_dir
    if pool and out_dir == PROJECT_ROOT / "eval_results/issue_923/fits":
        # follow-up-label rule: pooled outputs land in their own round dir.
        out_dir = PROJECT_ROOT / "eval_results/issue_923/pooled-span-features"
    if pool and args.tensors_dir == PROJECT_ROOT / "data/issue_923/fit_tensors":
        args.tensors_dir = PROJECT_ROOT / "data/issue_923/fit_tensors_pooled"
    if args.smoke and out_dir in (
        PROJECT_ROOT / "eval_results/issue_923/fits",
        PROJECT_ROOT / "eval_results/issue_923/pooled-span-features",
    ):
        # never clobber committed paths
        out_dir = Path(f"/tmp/issue-923-smoke/fits{'_pooled' if pool else ''}")
        args.tensors_dir = Path(f"/tmp/issue-923-smoke/fit_tensors{'_pooled' if pool else ''}")
    out_dir.mkdir(parents=True, exist_ok=True)
    partials = out_dir / "partials"
    partials.mkdir(parents=True, exist_ok=True)
    meta = reproducibility_metadata(
        {
            "script": "issue923_fit_decomposition",
            "smoke": args.smoke,
            "feature_source": args.feature_source,
        }
    )

    print("[phase=load]", flush=True)
    genres = [g.strip() for g in args.genres.split(",") if g.strip()]
    if args.smoke:
        genres = ["uc"]
        args.n_perms = min(args.n_perms, 20)
        args.n_boot = min(args.n_boot, 50)
        grids, fctx, fqry, folds_payload = build_smoke_inputs(out_dir)
        ood = True
    else:
        grids, fctx, fqry, folds_payload, load_meta = load_grids(
            args.packs_dir,
            args.reduce_dir,
            args.data_dir,
            genres,
            ood=not args.no_ood,
            feature_source=args.feature_source,
            pooled_packs_dir=args.pooled_packs_dir if pool else None,
        )
        ood = not args.no_ood
        meta["mask_backend"] = load_meta["mask_backend"]

    # Extended null coverage (blend + Dolly) is the pooled round's addition
    # (§4.1); the parent (last-token) path stays byte-identical un-gated.
    extended_nulls = pool
    parent_anova = None
    if pool and not args.smoke:
        # ANOVA is a function of targets only (identical inputs): cite the
        # parent's persisted shares instead of recomputing them (§4.2).
        parent_anova = load_json(args.parent_fits_dir / "anova_shares.json")["anova"]

    lc = fctx.shape[1]
    layers = list(range(lc)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    arms = [
        a
        for a in [*ARMS_SINGLE, *ARMS_CONCAT, ARM_FULL]
        if not (a in ("arm_qry_iii", "arm_concat_iii") and grids[genres[0]].fqryiii is None)
    ]
    # The regime key covers FLAGS *and* INPUT IDENTITIES (fold hash + per-pack
    # identity), so a `--fits-only` re-dispatch after a re-captured/re-reduced
    # pack set refuses stale partials loudly instead of silently resuming them
    # (r1 blocker fit-resume-regime-key-omits-input-hashes; the #722-r3 class).
    regime_key = {
        "genres": genres,
        "arms": arms,
        "layers": layers,
        "n_perms": args.n_perms,
        "seed": SEED,
        "pca_dim": PCA_DIM,
        "lambdas": RIDGE_LAMBDAS,
        "smoke": args.smoke,
        "ood": ood,
        "feature_source": args.feature_source,
        "extended_nulls": extended_nulls,
        "blend_null_l18_only": bool(args.blend_null_l18_only),
        "fold_hash": hashlib.sha256(json.dumps(folds_payload, sort_keys=True).encode()).hexdigest()[
            :12
        ],
        "pack_identity": (
            "smoke"
            if args.smoke
            else _input_pack_identity(
                args.packs_dir,
                args.reduce_dir,
                extra_dirs=(args.pooled_packs_dir,) if pool else (),
            )
        ),
    }
    regime_hash = hashlib.sha256(json.dumps(regime_key, sort_keys=True).encode()).hexdigest()[:12]

    print("[phase=fits]", flush=True)
    units: dict = {}
    perms: dict[str, np.ndarray] = {}
    # Dolly TEST-grid permutation for the extended ood nulls — INDEPENDENT of
    # the train-grid matrix (distinct seed, recorded in the null pack meta).
    perm_ood = None
    if extended_nulls and ood and "dolly" in grids:
        n_valid_dolly = int(grids["dolly"].valid.sum())
        perm_ood = make_perm_matrix(n_valid_dolly, args.n_perms, np.random.default_rng(SEED + 7))
    t0 = time.time()
    for genre in genres:
        grid = grids[genre]
        n_valid = int(grid.valid.sum())
        rng = np.random.default_rng(SEED)
        perms[genre] = make_perm_matrix(n_valid, args.n_perms, rng)
        folds = build_folds(grid, folds_payload["query_folds"][genre])
        for layer in layers:
            part_path = partials / f"{genre}_L{layer:02d}.pt"
            if part_path.exists() and not args.fresh:
                _tensors, pmeta = load_pack(part_path)
                if pmeta.get("regime_hash") == regime_hash:
                    units[(genre, layer)] = pmeta["unit"]
                    logger.info("[resume] %s L%d loaded from partial", genre, layer)
                    continue
                raise RuntimeError(
                    f"partial {part_path} regime mismatch ({pmeta.get('regime_hash')} != "
                    f"{regime_hash}) — rerun with --fresh to overwrite"
                )
            anova_override = None
            if parent_anova is not None:
                anova_override = parent_anova[genre][str(layer)]  # fail loud on a miss
            unit = fit_layer_unit(
                genre,
                layer,
                grids,
                fctx,
                fqry,
                folds,
                folds_payload["query_folds"][genre],
                arms,
                args.n_perms,
                perms[genre],
                ood,
                device,
                smoke_blend_assert=args.smoke,
                perm_ood=perm_ood,
                extended_nulls=extended_nulls,
                blend_null_this_layer=(not args.blend_null_l18_only or layer == HEADLINE_LAYER),
                anova_override=anova_override,
            )
            # JSON-able unit: numpy → lists for the checkpoint.
            unit_ser = json.loads(json.dumps(unit, default=lambda o: o.tolist()))
            units[(genre, layer)] = unit_ser
            save_pack(part_path, {}, {"regime_hash": regime_hash, "unit": unit_ser})
            logger.info(
                "[fits] %s L%02d done (%.1fs elapsed, full=%.3f)",
                genre,
                layer,
                time.time() - t0,
                unit["arms"][ARM_FULL]["skill"],
            )

    print("[phase=stats]", flush=True)
    stats = compute_stats(units, arms, args.n_boot, genres)

    # Persist: skill grid JSON, null summary + matrices, ANOVA, headline.
    skill_json = {
        "meta": {**meta, "regime_key": regime_key, "selftest": st},
        "genres": {},
    }
    for genre in genres:
        gl = {}
        for layer in layers:
            u = units[(genre, layer)]
            gl[str(layer)] = {
                "arms": {
                    arm: {
                        "skill": u["arms"][arm]["skill"],
                        "skill_ambient": u["arms"][arm].get("skill_ambient"),
                        "skill_per_lambda": u["arms"][arm]["skill_per_lambda"],
                        "fam_res": u["arms"][arm]["fam_res"],
                        "fam_tot": u["arms"][arm]["fam_tot"],
                        **(
                            {
                                "cell_pred_pc1": u["arms"][arm]["cell_pred_pc1"],
                                "cell_act_pc1": u["arms"][arm]["cell_act_pc1"],
                                "cell_cos": u["arms"][arm]["cell_cos"],
                            }
                            if layer == HEADLINE_LAYER or layer == max(layers)
                            else {}
                        ),
                    }
                    for arm in [*arms, "arm_blend"]
                },
                "lofo_marginal": u.get("lofo_marginal"),
                "qfold_marginal": u.get("qfold_marginal"),
                "ood_dolly": u.get("ood_dolly"),
                "blend": u.get("blend"),
            }
        skill_json["genres"][genre] = gl
    dump_json(skill_json, out_dir / "decomposition_skill.json")
    if parent_anova is None:
        dump_json(
            {
                "meta": meta,
                "anova": {g: {str(ll): units[(g, ll)]["anova"] for ll in layers} for g in genres},
            },
            out_dir / "anova_shares.json",
        )
    else:
        # §4.2: ANOVA skipped under pool — targets byte-identical; cite parent.
        dump_json(
            {
                "meta": meta,
                "skipped": True,
                "reason": "feature_source=pool — targets identical to parent; see "
                f"{args.parent_fits_dir / 'anova_shares.json'}",
            },
            out_dir / "anova_shares.json",
        )

    null_summary: dict = {"meta": {**meta, "n_perms": args.n_perms}, "genres": {}}
    args.tensors_dir.mkdir(parents=True, exist_ok=True)

    def _band_entry(m: np.ndarray, li_gate: int, obs_skill: float) -> dict:
        """L18-column + max-over-layers band stats for one (arm, scheme) matrix."""
        max_per_draw = np.nanmax(m, axis=1)
        col = m[:, li_gate]
        col_ok = col[~np.isnan(col)]
        p975 = float(np.nanpercentile(col, 97.5))
        return {
            "observed_skill_L18": obs_skill,
            "inside_l18_null_band": bool(obs_skill <= p975),
            "l18_null_p_value": float((np.sum(col_ok >= obs_skill) + 1) / (col_ok.size + 1)),
            "L18_column_quantiles": {
                "p95": float(np.nanpercentile(col, 95)),
                "p975": p975,
                "p99": float(np.nanpercentile(col, 99)),
            },
            "max_over_layers_quantiles": {
                "p95": float(np.nanpercentile(max_per_draw, 95)),
                "p975": float(np.nanpercentile(max_per_draw, 97.5)),
                "p99": float(np.nanpercentile(max_per_draw, 99)),
            },
        }

    for genre in genres:
        matrix = {
            arm: np.stack([np.asarray(units[(genre, ll)]["null"][arm]) for ll in layers], axis=1)
            for arm in arms
        }  # (n_perms, n_layers) per arm
        perm_sha = hashlib.sha256(perms[genre].tobytes()).hexdigest()
        npath = args.tensors_dir / f"null_matrix_{genre}.pt"
        pack_tensors = {arm: torch.from_numpy(matrix[arm]) for arm in arms} | {
            "perm_matrix": torch.from_numpy(perms[genre])
        }
        # Extended coverage (pooled round): blend column(s) + the Dolly scheme.
        blend_layers = [ll for ll in layers if "arm_blend" in units[(genre, ll)].get("null", {})]
        if blend_layers:
            blend_matrix = np.stack(
                [np.asarray(units[(genre, ll)]["null"]["arm_blend"]) for ll in blend_layers],
                axis=1,
            )
            pack_tensors["arm_blend"] = torch.from_numpy(blend_matrix)
        ood_layers = [ll for ll in layers if units[(genre, ll)].get("null_ood")]
        ood_matrix = {}
        if ood_layers:
            for arm in arms:
                if arm not in units[(genre, ood_layers[0])]["null_ood"]:
                    continue
                ood_matrix[arm] = np.stack(
                    [np.asarray(units[(genre, ll)]["null_ood"][arm]) for ll in ood_layers],
                    axis=1,
                )
                pack_tensors[f"ood::{arm}"] = torch.from_numpy(ood_matrix[arm])
            if perm_ood is not None:
                pack_tensors["perm_matrix_ood"] = torch.from_numpy(perm_ood)
        save_pack(
            npath,
            pack_tensors,
            {
                "layers": layers,
                "blend_layers": blend_layers,
                "ood_layers": ood_layers,
                "seed": SEED,
                "seed_ood": SEED + 7,
                "perm_sha256": perm_sha,
                "permutation": "full-grid cell-label (the #810 exchangeability recipe)",
                "metadata": meta,
            },
        )
        gsum = {"perm_sha256": perm_sha, "tensor_pack": npath.name, "arms": {}}
        li18 = layers.index(HEADLINE_LAYER) if HEADLINE_LAYER in layers else len(layers) - 1
        gate_layer = layers[li18]
        gsum["gating_layer"] = gate_layer  # HEADLINE_LAYER, or the smoke fallback
        for arm in arms:
            # Observed-vs-L18-null GATE (r1 blocker l18-null-gating-missing):
            # the registered rule — an arm whose observed L18 skill sits inside
            # its selection-matched L18-only null band is reported as null —
            # persisted per (genre, arm), separately from the max-over-layers
            # band (which gates any max/argmax read, not the frozen L18 one).
            obs_skill = units[(genre, gate_layer)]["arms"][arm]["skill"]
            gsum["arms"][arm] = _band_entry(matrix[arm], li18, obs_skill)
        if blend_layers and gate_layer in blend_layers:
            bl18 = blend_layers.index(gate_layer)
            gsum["arms"]["arm_blend"] = _band_entry(
                blend_matrix, bl18, units[(genre, gate_layer)]["arms"]["arm_blend"]["skill"]
            )
        if ood_matrix and units[(genre, gate_layer)].get("ood_dolly"):
            if gate_layer not in ood_layers:
                # Loud skip (r1 Minor): NEVER gate the observed L18 read
                # against a DIFFERENT layer's null column. Unreachable in
                # practice (ood folds are layer-independent, ood_layers ==
                # layers), so a miss means the inputs are malformed.
                logger.warning(
                    "[null] ood_dolly band SKIPPED for %s: gate layer %s not in "
                    "ood_layers %s (no cross-layer gating)",
                    genre,
                    gate_layer,
                    ood_layers,
                )
                gsum["ood_dolly_skipped_reason"] = (
                    f"gate layer {gate_layer} not in ood_layers {ood_layers}"
                )
            else:
                o18 = ood_layers.index(gate_layer)
                gsum["ood_dolly"] = {
                    arm: _band_entry(
                        ood_matrix[arm],
                        o18,
                        units[(genre, gate_layer)]["ood_dolly"][arm]["skill"],
                    )
                    for arm in ood_matrix
                    if arm in units[(genre, gate_layer)]["ood_dolly"]
                }
        null_summary["genres"][genre] = gsum
    dump_json(null_summary, out_dir / "null_summary.json")

    # Paired residual diff vs the parent round (pooled headline consumer; §4.2).
    paired = None
    if pool:
        print("[phase=paired_diff]", flush=True)
        # §6 k2 kill rule (ENFORCED, r2 Major): this round's per-genre
        # pool_full power-floor flags gate the verdict labels below.
        k2 = kill_floor_flags(stats, genres)
        if args.smoke:
            # Smoke: SELF-PAIR on the parent's real persisted sums (the
            # synthetic grid has no parent counterpart) — real-data path.
            parent_skill = load_json(args.parent_fits_dir / "decomposition_skill.json")
            hl18 = str(HEADLINE_LAYER)
            pooled_fams = {
                g: {
                    arm: (
                        np.asarray(parent_skill["genres"][g][hl18]["arms"][arm]["fam_res"], float),
                        np.asarray(parent_skill["genres"][g][hl18]["arms"][arm]["fam_tot"], float),
                    )
                    for arm in ("arm_full", "arm_concat_i")
                }
                for g in parent_skill["genres"]
            }
            paired = paired_residual_diff(
                pooled_fams,
                args.parent_fits_dir,
                load_json(args.parent_fits_dir / "headline.json")["stats"]["n_boot"],
                kill_floor=k2,
            )
            paired["smoke_self_paired"] = True
        else:
            pooled_fams = {}
            for genre in genres:
                if genre not in PARENT_QCOLS or (genre, HEADLINE_LAYER) not in units:
                    continue
                u18 = units[(genre, HEADLINE_LAYER)]["arms"]
                pooled_fams[genre] = {
                    arm: (
                        np.asarray(u18[arm]["fam_res"], dtype=np.float64),
                        np.asarray(u18[arm]["fam_tot"], dtype=np.float64),
                    )
                    for arm in ("arm_full", "arm_concat_i")
                }
            paired = paired_residual_diff(
                pooled_fams, args.parent_fits_dir, args.n_boot, kill_floor=k2
            )
        logger.info(
            "[paired_diff] verdict=%s note=%s skipped=%s",
            paired.get("verdict"),
            paired.get("verdict_note"),
            paired.get("verdict_skipped_reason"),
        )

    dump_json(headline_payload(meta, stats, paired), out_dir / "headline.json")
    if not args.smoke and pool:
        logger.info(
            "[regen] SKIPPED under feature_source=pool — targets byte-identical to the "
            "parent round; see the parent's regen_check.json"
        )
    if not args.smoke and not pool:
        try:
            rc = regen_check(args.packs_dir, args.reduce_dir)
        except AssertionError as e:  # tgt_regen packs absent
            if not args.allow_missing_regen:
                # Absence is a coverage/fetch failure, NOT a registered
                # fail-soft (the plan registers ONLY the oracle tolerance
                # flags) — fail fast (r1 blocker, regen-check sibling).
                raise RuntimeError(
                    "regen spot-check packs (tgt_regen_shard*.pt) absent under "
                    f"{args.packs_dir} — a coverage/fetch failure; pass "
                    "--allow-missing-regen ONLY for a deliberate partial run"
                ) from e
            logger.warning("[regen] SKIPPED (--allow-missing-regen): %s", e)
            dump_json(
                {"meta": meta, "skipped": True, "reason": str(e)},
                out_dir / "regen_check.json",
            )
        else:
            dump_json({"meta": meta, **rc}, out_dir / "regen_check.json")
            logger.info(
                "[regen] %d cells, cos_min=%.4f, n<0.99=%d",
                rc["n_cells"],
                rc["cos_min_overall"] or float("nan"),
                rc["n_below_0p99"],
            )

    if not args.no_upload and not args.smoke:
        print("[phase=upload]", flush=True)
        suffix = "fits_pooled" if pool else "fits"
        hub._upload(
            args.tensors_dir,
            HF_DATA_REPO,
            "dataset",
            f"{HF_PREFIX_923}/analysis_tensors/{suffix}",
        )
        hub._upload(out_dir, HF_DATA_REPO, "dataset", f"{HF_PREFIX_923}/eval_results/{suffix}")
    print("[phase=done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
