# ruff: noqa: RUF002, RUF003
# Intentional Unicode (→, ², λ, Σ, ‖, ⊙) in scientific docstrings.
"""Issue #928 — batched GROUP-fold LOCO/LOFO ridge + selection-symmetric nulls + bootstrap.

Extends the ``issue810_batched_null.py`` identities (X fixed across permutations
⇒ every X-only factor computed ONCE; all draws as batched GEMMs) from pointwise
LOCO at n=50 to GROUP folds at n≈2,400 (leave-one-CONTEXT-out over per-(C,q)
rows — 50 groups of ≈48 rows — plus leave-one-FAMILY-out, 7 groups), per plan
§4.6 "batched group-LOCO ridge". NO serial per-draw refit loop anywhere.

§9 ops arithmetic (production, `indiv` regime, mean/mean registered cells):

- fold factorizations: ~5 designs (ctx_mean / cot_mean / concat_mean / ctx_last
  / ans_mean) × 28 layers × 57 folds (50 LOCO + 7 LOFO) ≈ 7,980 ``eigh``(m≈2,352)
  ≈ 9·m³ ≈ 1.2e11 FLOP each → ~1e15 FLOP, computed ONCE per (design, layer) and
  shared across ALL targets, λ values, and null draws (the #823 no-per-target-
  refactorization guard).
- design Grams: ONE ``Xn @ Xnᵀ`` per (design, layer) at 2·n²·d ≈ 4.1e13 FLOP
  (the plan-§9 per-design figure). This is only possible with the design
  standardized ONCE on the full data (``standardization="full_data"``): fold
  Grams are then submatrices ``A[tr][:, tr]``. The inherited per-fold
  standardization (``"per_fold"``, exact #658/#810 convention, kept for the
  n=50 ``avg_q`` regime) would need a FRESH weighted Gram per fold
  (2·m²·d × 57 folds × 140 design-layers ≈ 3.2e17 FLOP ≈ 6 h — the very cost
  the plan's §9 basis excludes), because the per-column sd re-weighting is not
  expressible from any shared factorization. The `indiv` full-data hoisting is
  a NAMED estimator deviation (n=2,400 fold stats differ from full stats at
  O(1/n_groups); flagged as a persisted concern + smoke sensitivity check).
- null battery: 1,000 draws × (504 avg_q + 168+28 indiv cells) ≈ 700k
  draw-cells; per (fold, λ) the PRESS LOO-MSE of EVERY draw is a Frobenius
  inner product against a precomputed matrix,
  ``mse(b, λ) = ⟨N_λ, G_b⟩ / (m·P)`` with ``N_λ = A_λᵀA_λ``,
  ``A_λ = D⁻¹(I − H_λ)`` (3·m³ per (fold, λ) build, shared over all draws) and
  ``G_b = Ytr_c[b] Ytr_c[b]ᵀ`` gathered from the ONE precomputed full target
  Gram ``YG = Y Yᵀ`` (n²·P once per cell) — so the per-draw cost is O(m²)
  gathers + inner products, NOT the naive (m,m)@(B,m,P) rotation
  (≈5e18 FLOP at m=2,352 — the #823 class this module exists to avoid).
  Held-out reads per draw are ``ymu_b + K_λ* @ Ytr_c[b]`` with
  ``K_λ = (A[held, tr] Q) diag(1/(e+λ)) Qᵀ`` precomputed per (fold, λ)
  (h×m each) — cost h·m·P per draw.

The batched closed form IS the serial refit (same PRESS / dual identities as
``issue658_fit_predictors._press_loo_mse_per_lambda`` / ``_ridge_dual_weights``
and ``vectorized_mlp_skill.ridge_predict_loco_centered``); the seeded
serial-parity gate ``assert_group_ridge_matches_serial`` (vectorize-rule item 6)
checks the singleton-group case against ``ridge_predict_loco_centered``
byte-for-byte-close (atol 1e-8) and the multi-row-group + null cases against an
inline serial reference at the same tolerance.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy/torch freeze their pools. Idempotent on a second call.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue658_fit_predictors import RIDGE_LAMBDAS  # noqa: E402

# ── folds ─────────────────────────────────────────────────────────────────────


def group_folds(groups: np.ndarray, group_order: list) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-GROUP-out folds: [(tr_idx, held_idx)] in ``group_order``.

    ``groups`` is an (n,) label array (context id index for LOCO-by-context,
    family index for LOFO). Held rows leave TOGETHER — never pointwise LOO over
    (C, q) rows (`.claude/rules/ood-generalization-folds.md`). Singleton groups
    reduce exactly to the parent's pointwise LOCO.
    """
    folds = []
    for g in group_order:
        held = np.flatnonzero(groups == g)
        if held.size == 0:
            raise ValueError(f"fold group {g!r} has no rows")
        tr = np.flatnonzero(groups != g)
        folds.append((tr, held))
    return folds


def make_group_perm_matrix(
    groups: np.ndarray, group_order: list, n_perms: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_perms, n) row-index matrix for GROUP-block label permutations.

    Per draw: a uniform permutation π over the groups; the target rows of group
    g are replaced by the rows of π(g), matched by within-group ORDER with
    modular wrap-around when sizes differ (parse drops make per-context row
    counts vary, §4.8). Preserves the group structure of the permuted target
    while breaking the X↔Y group alignment — the "context-GROUP-level"
    permutation of plan §6. For singleton groups this is exactly a uniform
    permutation of rows (the parent's context-level null convention).
    """
    rows_by_group = {g: np.flatnonzero(groups == g) for g in group_order}
    n = int(groups.shape[0])
    k = len(group_order)
    out = np.empty((n_perms, n), dtype=np.int64)
    for b in range(n_perms):
        pi = rng.permutation(k)
        perm_row = np.empty(n, dtype=np.int64)
        for gi, g in enumerate(group_order):
            dst = rows_by_group[g]
            src = rows_by_group[group_order[pi[gi]]]
            perm_row[dst] = src[np.arange(dst.size) % src.size]
        out[b] = perm_row
    return out


# ── group-fold ridge design (X-only factors, computed once, shared) ───────────


class GroupRidgeDesign:
    """Per-fold X-only ridge factors for one (design, layer), computed ONCE.

    ``standardization="per_fold"`` reproduces the inherited #658/#810 estimator
    exactly (train-only column mu/sd per fold; used for the n=50 ``avg_q``
    regime — parity-gated against ``ridge_predict_loco_centered``).
    ``standardization="full_data"`` standardizes X once on all rows so the fold
    Grams are submatrices of ONE shared ``A = Xn Xnᵀ`` (the plan-§9 per-design
    Gram basis; used for the n≈2,400 ``indiv`` regime — see module docstring).

    Everything cached here is INDEPENDENT of the target Y → shared across all
    targets, λ values, and null draws.
    """

    def __init__(
        self,
        X: np.ndarray,
        folds: list[tuple[np.ndarray, np.ndarray]],
        lambdas=RIDGE_LAMBDAS,
        device: str = "cpu",
        standardization: str = "per_fold",
    ) -> None:
        assert standardization in ("per_fold", "full_data"), standardization
        self.n, self.d = int(X.shape[0]), int(X.shape[1])
        self.lambdas = [float(la) for la in lambdas]
        self.device = torch.device(device)
        self.standardization = standardization
        self.folds = folds
        Xt = torch.from_numpy(np.ascontiguousarray(X)).to(self.device, torch.float64)
        self._Xt = Xt
        # per-fold caches (built lazily-but-eagerly here; freed via .free()):
        self.tr_idx: list[torch.Tensor] = []
        self.held_idx: list[torch.Tensor] = []
        self.evals: list[torch.Tensor] = []
        self.Q: list[torch.Tensor] = []
        self.x_held_dot: list[torch.Tensor] = []  # (h, m) = Xheld_n @ Xtr_nᵀ
        self.mu: list[torch.Tensor] = []
        self.sd: list[torch.Tensor] = []
        self._Xn_full: torch.Tensor | None = None
        if standardization == "full_data":
            mu = Xt.mean(0)
            sd = Xt.std(0, correction=0) + 1e-9  # numpy ddof=0 (#658 convention)
            Xn = (Xt - mu) / sd
            self._Xn_full = Xn
            A = Xn @ Xn.t()  # ONE shared Gram per design (2·n²·d — the §9 figure)
            for tr, held in folds:
                tr_t = torch.as_tensor(tr, device=self.device)
                held_t = torch.as_tensor(held, device=self.device)
                G = A[tr_t][:, tr_t]
                evals, Q = torch.linalg.eigh(G)
                self.tr_idx.append(tr_t)
                self.held_idx.append(held_t)
                self.evals.append(evals)
                self.Q.append(Q)
                self.x_held_dot.append(A[held_t][:, tr_t])
                self.mu.append(mu)
                self.sd.append(sd)
        else:
            for tr, held in folds:
                tr_t = torch.as_tensor(tr, device=self.device)
                held_t = torch.as_tensor(held, device=self.device)
                Xtr = Xt[tr_t]
                mu = Xtr.mean(0)
                sd = Xtr.std(0, correction=0) + 1e-9
                Xtr_n = (Xtr - mu) / sd
                G = Xtr_n @ Xtr_n.t()
                evals, Q = torch.linalg.eigh(G)
                Xheld_n = (Xt[held_t] - mu) / sd
                self.tr_idx.append(tr_t)
                self.held_idx.append(held_t)
                self.evals.append(evals)
                self.Q.append(Q)
                self.x_held_dot.append(Xheld_n @ Xtr_n.t())
                self.mu.append(mu)
                self.sd.append(sd)

    # -- external-input dual read vectors (composition / avg_t) ---------------

    def xdot_for(self, fold_i: int, Xnew: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Dual read vectors (k, m) for EXTERNAL inputs under fold ``fold_i``.

        ``Xnew`` (k, d) in the ORIGINAL (ambient) design space — e.g. the
        decoded stage-A CoT predictions for the fold's held rows (fold-coherent
        composition, plan §4.6), or the query-averaged CoT vector (avg_t).
        Standardized with the fold's train stats, contracted against the train
        design: ``((Xnew − mu)/sd) @ Xtr_nᵀ``.
        """
        if isinstance(Xnew, np.ndarray):
            Xnew = torch.from_numpy(np.ascontiguousarray(Xnew))
        Xnew = Xnew.to(self.device, torch.float64)
        mu, sd = self.mu[fold_i], self.sd[fold_i]
        Xn_new = (Xnew - mu) / sd
        tr_t = self.tr_idx[fold_i]
        if self.standardization == "full_data":
            Xtr_n = self._Xn_full[tr_t]
        else:
            Xtr = self._Xt[tr_t]
            Xtr_n = (Xtr - mu) / sd
        return Xn_new @ Xtr_n.t()

    def free(self) -> None:
        """Drop the cached factors (one design is processed at a time; §9 memory)."""
        self.tr_idx = self.held_idx = self.evals = self.Q = self.x_held_dot = []
        self.mu = self.sd = []
        self._Xn_full = None
        self._Xt = None


# ── observed fits (B=1): parent-identical eigen route ─────────────────────────


def _press_select_lambda(
    evals: torch.Tensor, Q: torch.Tensor, Ytr_c: torch.Tensor, lambdas: list[float]
) -> int:
    """PRESS LOO-MSE argmin over λ — the ``_press_loo_mse_per_lambda`` identity.

    ``loo = (Ytr_c − Q(filt ⊙ QᵀYtr_c)) / clamp(1 − h, 1e-8)`` with
    ``h = Qsq @ filt``; mse = mean over (train rows × target dims). Matches the
    #658 serial reference and ``issue810_batched_null._loco_ridge_pred_batched``.
    """
    dev = evals.device
    U = Q.t() @ Ytr_c  # (m, P)
    lam_t = torch.tensor(lambdas, dtype=torch.float64, device=dev)
    filt = evals.unsqueeze(0) / (evals.unsqueeze(0) + lam_t.unsqueeze(1))  # (nlam, m)
    Qsq = Q * Q
    h = filt @ Qsq.t()  # (nlam, m)
    mses = []
    for li in range(len(lambdas)):
        Yhat = Q @ (filt[li].unsqueeze(1) * U)  # (m, P)
        resid = Ytr_c - Yhat
        denom = (1.0 - h[li]).clamp(min=1e-8).unsqueeze(1)
        loo = resid / denom
        mses.append((loo * loo).mean())
    return int(torch.argmin(torch.stack(mses)).item())


def fit_predict_grouped(
    design: GroupRidgeDesign, Y: np.ndarray
) -> tuple[np.ndarray, list[float], list[dict]]:
    """Held-out group-fold ridge predictions + per-fold model records.

    Returns ``(preds (n, P) un-centered, best_lams per fold, fold_models)``
    where ``fold_models[f] = {"ymu": (P,), "alpha": (m, P), "lam": float}`` —
    the pieces composition (plan §4.6) and the avg_t re-read need to apply the
    fold's map to EXTERNAL inputs via ``design.xdot_for``. Per fold: train-only
    target centering, PRESS λ (nested CV, no λ leakage), dual solve in the
    shared eigenbasis ``alpha = Q((QᵀYtr_c) / (e + λ))``, held read
    ``ymu + x_held_dot @ alpha``.
    """
    dev = design.device
    Yt = torch.from_numpy(np.ascontiguousarray(Y)).to(dev, torch.float64)
    n, P = Yt.shape
    assert n == design.n, (n, design.n)
    preds = torch.zeros((n, P), dtype=torch.float64, device=dev)
    best_lams: list[float] = []
    fold_models: list[dict] = []
    for f in range(len(design.folds)):
        tr_t, held_t = design.tr_idx[f], design.held_idx[f]
        Ytr = Yt[tr_t]
        ymu = Ytr.mean(0)
        Ytr_c = Ytr - ymu
        li = _press_select_lambda(design.evals[f], design.Q[f], Ytr_c, design.lambdas)
        lam = design.lambdas[li]
        U = design.Q[f].t() @ Ytr_c
        alpha = design.Q[f] @ (U / (design.evals[f] + lam).unsqueeze(1))  # (m, P)
        preds[held_t] = ymu + design.x_held_dot[f] @ alpha
        best_lams.append(lam)
        fold_models.append({"ymu": ymu, "alpha": alpha, "lam": lam})
    return preds.detach().cpu().numpy(), best_lams, fold_models


def predict_external(design: GroupRidgeDesign, fold_models: list[dict], X_by_fold) -> np.ndarray:
    """Apply per-fold models to EXTERNAL inputs (fold-coherent composition read).

    ``X_by_fold[f]`` is a (k_f, d) ambient-space input matrix for fold f (e.g.
    decoded stage-A predictions for that fold's held rows). Returns the stacked
    predictions in fold-held order (Σk_f, P).
    """
    outs = []
    for f, Xf in enumerate(X_by_fold):
        xdot = design.xdot_for(f, Xf)  # (k, m)
        m = fold_models[f]
        outs.append((m["ymu"] + xdot @ m["alpha"]).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def group_train_means(Y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """(n, P) group-fold predict-the-mean baseline: row i = mean of its fold's TRAIN rows.

    Singleton groups reduce exactly to ``vectorized_mlp_skill.loco_train_means``.
    """
    out = np.zeros_like(Y, dtype=np.float64)
    for tr, held in folds:
        out[held] = Y[tr].mean(axis=0, keepdims=True)
    return out


def grouped_skill(
    preds: np.ndarray, Y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]
) -> dict:
    """Skill-over-mean R² with the GROUP-fold train-mean baseline + per-group decomposition.

    skill = 1 − Σss_res/Σss_tot; per-group (ss_res_g, ss_tot_g) are persisted
    for the paired per-context bootstrap (a pure re-reduction, no refit — the
    #810 committed machinery, plan §6).
    """
    tmean = group_train_means(Y, folds)
    ss_res_g, ss_tot_g = [], []
    for _tr, held in folds:
        ss_res_g.append(float(np.sum((Y[held] - preds[held]) ** 2)))
        ss_tot_g.append(float(np.sum((Y[held] - tmean[held]) ** 2)))
    ss_res, ss_tot = float(np.sum(ss_res_g)), float(np.sum(ss_tot_g))
    skill = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return {
        "skill": skill,
        "ss_res": ss_res,
        "ss_tot": ss_tot,
        "ss_res_by_group": ss_res_g,
        "ss_tot_by_group": ss_tot_g,
    }


# ── batched selection-symmetric null (per-draw refit incl. λ re-selection) ────


def grouped_null_skills(
    design: GroupRidgeDesign,
    Y_pca: np.ndarray,
    perm: np.ndarray,
    draw_chunk: int = 16,
    xdot_override: list[torch.Tensor] | None = None,
) -> list[float]:
    """Per-draw group-fold ridge skill for the label-shuffle null — fully batched.

    Every draw is a FULL re-fit (per-fold train centering + PRESS λ
    re-selection + dual solve) against the row-permuted target ``Y_pca[perm[b]]``
    — the parent's per-draw-refit convention — but NO per-draw factorization:

    - PRESS via the target-Gram identity ``mse(b, λ) = ⟨N_λ, G_b⟩/(m·P)`` with
      ``N_λ = A_λᵀA_λ`` built once per (fold, λ) (3·m³) and
      ``G_b = Ytr_c[b]Ytr_c[b]ᵀ`` gathered from the ONE full target Gram
      ``YG = Y Yᵀ`` (n²·P once) + rank-1 centering corrections (O(m²)/draw).
    - held reads via ``K_λ = (x_held_dot Q) diag(1/(e+λ)) Qᵀ`` (h×m per (fold,
      λ), built once): ``pred_b = ymu_b + K_λ*(b) @ Ytr_c[b]`` (h·m·P per draw).

    ``xdot_override`` (composition null): per-fold (h, m) dual read vectors for
    the FIXED decoded stage-A predictions — stage A never sees the permuted ans
    target, so its predictions are constant across draws and only stage B is
    re-fit (plan §4.6 fold-coherence, preserved under the null).

    Returns n_perms skill values (group-fold train-mean baseline, matching
    ``grouped_skill`` on the same permuted target).
    """
    dev = design.device
    lambdas = design.lambdas
    nlam = len(lambdas)
    Yt = torch.from_numpy(np.ascontiguousarray(Y_pca)).to(dev, torch.float64)
    _n, P = Yt.shape
    B = int(perm.shape[0])
    perm_t = torch.from_numpy(np.ascontiguousarray(perm)).to(dev, torch.long)
    YG = Yt @ Yt.t()  # (n, n) full target Gram, ONCE per cell
    ss_res = torch.zeros(B, dtype=torch.float64, device=dev)
    ss_tot = torch.zeros(B, dtype=torch.float64, device=dev)
    for f in range(len(design.folds)):
        tr_t, held_t = design.tr_idx[f], design.held_idx[f]
        m = int(tr_t.shape[0])
        evals, Q = design.evals[f], design.Q[f]
        xdot = xdot_override[f] if xdot_override is not None else design.x_held_dot[f]
        xdot = xdot.to(dev, torch.float64)
        # per-λ precomputes (shared across ALL draws): N_λ, N_λ1, 1ᵀN_λ1, K_λ.
        Qsq = Q * Q
        lam_t = torch.tensor(lambdas, dtype=torch.float64, device=dev)
        filt = evals.unsqueeze(0) / (evals.unsqueeze(0) + lam_t.unsqueeze(1))  # (nlam, m)
        h_diag = filt @ Qsq.t()  # (nlam, m)
        N_list, N1_list, oneN1_list, K_list, Krow_list = [], [], [], [], []
        for li in range(nlam):
            Hlam = Q @ (filt[li].unsqueeze(1) * Q.t())  # (m, m)
            dinv = 1.0 / (1.0 - h_diag[li]).clamp(min=1e-8)
            A_lam = dinv.unsqueeze(1) * (torch.eye(m, dtype=torch.float64, device=dev) - Hlam)
            N = A_lam.t() @ A_lam  # (m, m)
            N_list.append(N)
            N1 = N.sum(dim=1)  # (m,)
            N1_list.append(N1)
            oneN1_list.append(N1.sum())
            K = (xdot @ Q) * (1.0 / (evals + lambdas[li])).unsqueeze(0) @ Q.t()  # (h, m)
            K_list.append(K)
            Krow_list.append(K.sum(dim=1))  # (h,)
        for lo in range(0, B, draw_chunk):
            sel = perm_t[lo : lo + draw_chunk]  # (c, n)
            c = int(sel.shape[0])
            rows_tr = sel[:, tr_t]  # (c, m) target-row indices for the train fold
            rows_hd = sel[:, held_t]  # (c, h)
            Ytr = Yt[rows_tr]  # (c, m, P)
            ymu = Ytr.mean(dim=1)  # (c, P)
            # G_b = YG[rows, rows] − s1ᵀ − 1sᵀ + q·11ᵀ (centering corrections)
            YGsel = YG[rows_tr.unsqueeze(-1), rows_tr.unsqueeze(-2)]  # (c, m, m)
            s_b = YGsel.mean(dim=2)  # (c, m) = YGsel @ 1/m
            q_b = s_b.mean(dim=1)  # (c,) = 1ᵀYGsel1/m²
            mse = torch.empty((nlam, c), dtype=torch.float64, device=dev)
            for li in range(nlam):
                inner = torch.einsum("ij,cij->c", N_list[li], YGsel)
                corr = 2.0 * (s_b @ N1_list[li]) - q_b * oneN1_list[li]
                mse[li] = (inner - corr) / (m * P)
            best = torch.argmin(mse, dim=0)  # (c,)
            Yhd = Yt[rows_hd]  # (c, h, P) true held target rows (permuted)
            for li in range(nlam):
                mask = best == li
                if not bool(mask.any()):
                    continue
                idx = torch.nonzero(mask, as_tuple=True)[0]
                Ytr_li = Ytr[idx]  # (b, m, P)
                ymu_li = ymu[idx]  # (b, P)
                # pred = ymu + K @ (Ytr − 1 ymuᵀ) = ymu + K@Ytr − (K1) ymuᵀ
                KY = torch.einsum("hm,bmp->bhp", K_list[li], Ytr_li)
                pred = ymu_li.unsqueeze(1) + KY - Krow_list[li].view(1, -1, 1) * ymu_li.unsqueeze(1)
                diff = Yhd[idx] - pred
                ss_res[lo + idx] += (diff * diff).sum(dim=(1, 2))
                tot = Yhd[idx] - ymu_li.unsqueeze(1)
                ss_tot[lo + idx] += (tot * tot).sum(dim=(1, 2))
    skills = torch.where(
        ss_tot < 1e-12, torch.full_like(ss_tot, float("nan")), 1.0 - ss_res / ss_tot
    )
    return [float(s) for s in skills.detach().cpu().numpy()]


# ── paired per-context bootstrap (shared resample-index matrix, no refit) ─────


def make_bootstrap_index_matrix(n_groups: int, n_draws: int, seed: int) -> np.ndarray:
    """(n_draws, n_groups) context resample indices — ONE shared matrix (plan §6).

    Every arm / summary / layer / regime reuses the SAME matrix so Δskill draws
    are PAIRED (the #810 committed machinery; seed 42, 2,000 draws).
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_groups, size=(n_draws, n_groups))


def bootstrap_skills(ss_res_g: np.ndarray, ss_tot_g: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(n_draws,) skill per resample — pure re-reduction of the per-group decomposition.

    ``idx`` is the shared (n_draws, n_groups) matrix; a draw's skill is
    ``1 − Σss_res[idx]/Σss_tot[idx]`` (contexts resampled with replacement over
    the FIXED per-context decomposition; no per-replicate refit).
    """
    res = ss_res_g[idx].sum(axis=1)
    tot = ss_tot_g[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tot < 1e-12, np.nan, 1.0 - res / tot)


def stat_summary(obs: float, draws: np.ndarray) -> dict:
    """Observed value + bootstrap percentile CI + P(Δ≤0) (the #810 report shape)."""
    draws = draws[np.isfinite(draws)]
    return {
        "observed": float(obs),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "p_delta_le_0": float(np.mean(draws <= 0.0)),
        "n_draws": int(draws.size),
    }


# ── seeded serial-parity gate (vectorize-rule item 6; plan §4.6/§12.7) ────────


def _serial_group_ridge_reference(
    X: np.ndarray, Y: np.ndarray, folds, lambdas, standardization: str
) -> np.ndarray:
    """Serial numpy/torch group-fold ridge oracle (per-fold loop, explicit solves).

    The seeded serial reference for the parity gate: per fold, standardize
    (per-fold or full-data), center the train target, PRESS λ via the #658
    identity, dual weights via an explicit ``(G + λI)⁻¹`` solve, held read.
    Deliberately the SLOW obvious implementation — used on 2-3 tiny cells only.
    """
    n = X.shape[0]
    preds = np.zeros_like(Y, dtype=np.float64)
    Xt = torch.from_numpy(np.ascontiguousarray(X)).double()
    Yt = torch.from_numpy(np.ascontiguousarray(Y)).double()
    if standardization == "full_data":
        mu_all = Xt.mean(0)
        sd_all = Xt.std(0, correction=0) + 1e-9
    for tr, held in folds:
        tr_t = torch.as_tensor(tr)
        held_t = torch.as_tensor(held)
        Xtr = Xt[tr_t]
        if standardization == "per_fold":
            mu = Xtr.mean(0)
            sd = Xtr.std(0, correction=0) + 1e-9
        else:
            mu, sd = mu_all, sd_all
        Xtr_n = (Xtr - mu) / sd
        Xhd_n = (Xt[held_t] - mu) / sd
        Ytr = Yt[tr_t]
        ymu = Ytr.mean(0)
        Ytr_c = Ytr - ymu
        G = Xtr_n @ Xtr_n.t()
        evals, Q = torch.linalg.eigh(G)
        li = _press_select_lambda(evals, Q, Ytr_c, list(lambdas))
        lam = float(lambdas[li])
        m = G.shape[0]
        alpha = torch.linalg.solve(G + lam * torch.eye(m, dtype=torch.float64), Ytr_c)
        preds[held] = (ymu + (Xhd_n @ Xtr_n.t()) @ alpha).numpy()
    _ = n
    return preds


def assert_group_ridge_matches_serial(seed: int = 928, atol: float = 1e-8) -> dict:
    """Seeded serial-parity gate for the batched group-fold machinery.

    Three checks (all at fp64, stated tolerance ``atol=1e-8``):

    1. **Singleton-group parity vs the on-main serial reference**:
       ``fit_predict_grouped`` with singleton groups + per_fold standardization
       must reproduce ``vectorized_mlp_skill.ridge_predict_loco_centered``
       (the #722/#810 committed estimator) on a random (n=16, d=6, P=3) cell.
    2. **Multi-row-group parity vs the inline serial oracle** (both
       standardization modes) on a random (n=24, 6 groups × 4, d=5, P=3) cell.
    3. **Batched-null parity**: ``grouped_null_skills`` per-draw skills must
       equal a serial loop of ``fit_predict_grouped`` + ``grouped_skill`` on
       each permuted target (3 draws, both a singleton- and a multi-row-group
       cell).

    Returns a dict of max-abs deviations; raises AssertionError on any breach.
    Run before trusting any production fit (vectorize-many-cell-fits item 6).
    """
    from explore_persona_space.analysis.vectorized_mlp_skill import ridge_predict_loco_centered

    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}

    # 1. singleton-group vs ridge_predict_loco_centered (per_fold).
    n, d, P = 16, 6, 3
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((n, P))
    groups = np.arange(n)
    folds = group_folds(groups, list(range(n)))
    des = GroupRidgeDesign(X, folds, device="cpu", standardization="per_fold")
    preds, _, _ = fit_predict_grouped(des, Y)
    ref = ridge_predict_loco_centered(X, Y)
    dev1 = float(np.max(np.abs(preds - ref)))
    assert dev1 < atol, f"singleton-group parity breach: {dev1} >= {atol}"
    out["singleton_vs_ridge_predict_loco_centered"] = dev1

    # 2. multi-row groups vs the inline serial oracle, both modes.
    n2, d2, P2, gsz = 24, 5, 3, 4
    X2 = rng.standard_normal((n2, d2))
    Y2 = rng.standard_normal((n2, P2))
    groups2 = np.repeat(np.arange(n2 // gsz), gsz)
    folds2 = group_folds(groups2, list(range(n2 // gsz)))
    for mode in ("per_fold", "full_data"):
        des2 = GroupRidgeDesign(X2, folds2, device="cpu", standardization=mode)
        p2, _, _ = fit_predict_grouped(des2, Y2)
        r2 = _serial_group_ridge_reference(X2, Y2, folds2, RIDGE_LAMBDAS, mode)
        dev2 = float(np.max(np.abs(p2 - r2)))
        assert dev2 < atol, f"group parity breach ({mode}): {dev2} >= {atol}"
        out[f"group_vs_serial_{mode}"] = dev2

    # 3. batched null vs serial per-draw refit (3 draws; singleton + group cells).
    for label, (Xc, Yc, gc, go) in {
        "singleton": (X, Y, groups, list(range(n))),
        "grouped": (X2, Y2, groups2, list(range(n2 // gsz))),
    }.items():
        foldsc = group_folds(gc, go)
        desc = GroupRidgeDesign(Xc, foldsc, device="cpu", standardization="per_fold")
        perm = make_group_perm_matrix(gc, go, 3, np.random.default_rng(seed + 1))
        batched = grouped_null_skills(desc, Yc, perm, draw_chunk=2)
        for b in range(perm.shape[0]):
            Yp = Yc[perm[b]]
            pb, _, _ = fit_predict_grouped(desc, Yp)
            sb = grouped_skill(pb, Yp, foldsc)["skill"]
            dev3 = abs(batched[b] - sb)
            assert dev3 < atol, f"null parity breach ({label}, draw {b}): {dev3} >= {atol}"
            out[f"null_{label}_draw{b}"] = float(dev3)
    return out


if __name__ == "__main__":
    # CLI parity gate: `uv run python scripts/issue928_null_bootstrap.py` runs the
    # seeded serial-parity checks and prints the max deviations (exit 0 == PASS).
    res = assert_group_ridge_matches_serial()
    for k, v in res.items():
        print(f"PARITY {k}: max|Δ| = {v:.3e}")
    print("group-fold batched ridge parity: PASS (atol 1e-8)")
