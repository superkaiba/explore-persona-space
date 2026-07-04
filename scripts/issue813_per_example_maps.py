#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, M⁺, M0, →, ×, c̄, λ, ρ, ‖·‖, ※) in scientific docstrings + logs.
"""Issue #813 follow-up `per-example-vs-averaged-map` — driver (plan v3 §4).

Fits the per-example (per-question) ridge map h_pe alongside the parent's
question-averaged map M at the frozen headline layer 14, for the 12
(behavior × substrate) cells × 2 arms (base / trained), on the parent's
HF-persisted artifacts at pinned revision ``HF_REVISION``. Computes:

- DV1: paired cross-fit transfer over the SAME 50 LOCO folds (fold = one battery
  context out; its questions leave together) + the LOFO-7 family-fold companion.
- DV2: coefficient-space agreement (top-k singular-subspace overlap + linear CKA
  of the effective maps W_eff = diag(1/sd) @ W), calibrated against question-half
  refit twins (averaged-grain AND per-example-grain, 50 splits each) and the
  analytic random-subspace floor.
- DV3: R²-vs-k averaging curve (k questions averaged per context, refit, LOCO)
  with the measured within/between-context variance components for the analytic
  reliability-attenuation overlay.
- DV4: query-specific incremental R² (within-context centered held-out R²) vs a
  200-draw within-context question-label shuffle null (λ frozen at the observed
  per-fold PRESS choice; all draws share the observed factorizations — the
  refit per draw is ONE batched GEMM per fold over stacked permuted targets).
- DV6: pre/post-FT agreement Δ_pe = median_c |(h_pe⁺ − h_pe⁰)(c̄0)·r̂_B|
  (marker: unprojected norm, parent read-1), with per-example refit-pair floors
  (parent convention) + the em-only per-example question-resampling null.
- DV5 is pre-registered N/A (no per-question judge outputs exist).

Reuse (imported, never re-implemented): ``issue722_fit_M._pca_basis_v0/_to64``
(basis), ``issue658_fit_predictors.RIDGE_LAMBDAS`` + the PRESS/dual-ridge math
(the shared-eigh fold path below reproduces ``_press_loo_mse_per_lambda`` +
``_ridge_dual_weights`` from ONE eigendecomposition per fold — gated ≤1e-6
against the fit658-composed serial oracle), ``issue722_bootstrap.make_refit_pair``
(the DV6 serial floor oracle) + ``issue813_analysis._floor_resample_indices``
(its exact rng-consumption twin), ``issue813_save_maps`` NPZ conventions.

Compute shape (plan §9): the pe-grain LOCO is the plan's sanctioned FALLBACK —
a fold loop whose per-fold work is one large standardized-design GEMM + ONE
shared eigh (FLOP-bound; no per-fold×per-λ refits, no serial tiny-op loop).
The DV6 floors ride a batched member loop over gathered resamples (basis via
exact top-k Lanczos eigsh; PRESS or frozen λ per ``--dv6-floor-mode``).

Persistence: per-cell JSON the moment the cell completes (atomic tmp+replace)
under ``eval_results/issue_813/per_example_vs_averaged/``; regime-keyed resume
(never row-count-only — v7 concern ``perlayer-resume-stale-regime``); every
keyed NPZ read goes through the fail-loud ``_require_npz_keys`` preflight
(v7 concern ``perlayer-npz-key-coverage-preflight``).
"""

from __future__ import annotations

import argparse
import logging
import math
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue658_fit_predictors as fit658  # noqa: E402
import torch  # noqa: E402  (after load_dotenv so thread-cap setdefaults apply)

logger = logging.getLogger("issue813.pe_maps")

DATA_REPO = "superkaiba1/explore-persona-space-data"
EXPERIMENT_NAME = "issue813_mapchange_substrate"
# The parent extraction wave's pinned revision (plan §10; a later force-push to the
# HF repo's main cannot silently swap the inputs under this analysis).
HF_REVISION = "b0d30307c1671cad575928e5abf5253c0c849dee"
BEHAVIORS = ("em", "fact", "sycophancy", "marker")
SUBSTRATES = ("generic", "elicit", "mix")
HEADLINE_LAYER = 14  # frozen (#651/#658/#813); the ONLY layer with per-question rows
N_LAYERS = 28
HIDDEN = 3584
TARGET_DIM = 64  # cap; the shared parent basis realizes min(64, n_ctx) components
SEED = 42  # all draws/splits/shuffles (parent convention)
LAMBDAS = [float(x) for x in fit658.RIDGE_LAMBDAS]
ARMS = ("base", "trained")
DV3_K_GRID = (1, 2, 4, 8, 16, 32, 48)
# Bump when an output-affecting code change lands (part of the resume regime key —
# v7 concern `perlayer-resume-stale-regime`: git_sha alone churns per commit, row
# counts alone are blind to regime changes; this is the deliberate middle).
REGIME_VERSION = 1

OUT_SUBDIR = "eval_results/issue_813/per_example_vs_averaged"


# ── shared fail-loud NPZ preflight (v7 concern: perlayer-npz-key-coverage-preflight) ──


def _require_npz_keys(path: Path | str, npz, required_keys) -> None:
    """Fail loud BEFORE any keyed read when an NPZ is missing required keys.

    ``npz`` is an ``np.lib.npyio.NpzFile`` (uses ``.files``) or any mapping. The
    error names the missing keys AND the cell path so a schema drift between the
    producer (issue813_run_cell / issue667_save_maps) and a consumer surfaces as
    one actionable line instead of a bare KeyError deep in compute.
    """
    present = list(getattr(npz, "files", None) or npz.keys())
    missing = [k for k in required_keys if k not in present]
    if missing:
        raise KeyError(
            f"NPZ schema preflight FAILED for {path}: missing keys {missing} "
            f"(present: {sorted(present)})"
        )


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except Exception:  # detached / no git — provenance is best-effort
        return "unknown"


def _repro_meta(args: argparse.Namespace) -> dict:
    """Reproducibility metadata for every result JSON (CLAUDE.md requirement)."""
    return {
        "git_sha": _git_sha(),
        "hf_revision": HF_REVISION,
        "seed": SEED,
        "command": " ".join(sys.argv),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }


def _regime(args: argparse.Namespace) -> dict:
    """The exact-regime resume key (v7 concern `perlayer-resume-stale-regime`).

    Every output-affecting knob is part of the key; a cached cell JSON is reused
    IFF its stored ``regime`` dict equals this one (plus schema-key presence) —
    never on row count alone.
    """
    return {
        "regime_version": REGIME_VERSION,
        "headline_layer": HEADLINE_LAYER,
        "hf_revision": HF_REVISION,
        "target_dim_cap": TARGET_DIM,
        "lambdas": LAMBDAS,
        "grain": "per-example-vs-averaged",
        "fold_config": "LOCO-context (questions leave together) + LOFO-family companion",
        "basis": "shared parent _pca_basis_v0(v_bar0_L14) primary; per-question v0 sensitivity",
        "dv3_k_grid": list(DV3_K_GRID),
        "dv3_draws": args.dv3_draws,
        "dv4_draws": args.dv4_draws,
        "twin_splits": args.twin_splits,
        "dv6_pairs": args.dv6_pairs,
        "dv6_floor_mode": args.dv6_floor_mode,
        "dv6_null_resamples": args.dv6_null_resamples,
        "boot_resamples": args.boot_resamples,
        "seed": SEED,
    }


# ── data loading ──────────────────────────────────────────────────────────────

_SUMMARY_KEYS = (
    "c_C_base",
    "c_C_trained",
    "v_A_base",
    "v_A_trained",
    "context_ids",
    "families",
)
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


@dataclass
class CellData:
    """One (behavior, substrate) cell's aligned L14 inputs, both grains + arms."""

    behavior: str
    substrate: str
    # averaged grain (kept contexts), (F, HIDDEN) float64
    avg_c: dict = field(default_factory=dict)  # arm -> (F, H)
    avg_v: dict = field(default_factory=dict)
    ctx_ids: list[str] = field(default_factory=list)  # kept, length F
    families: list[str] = field(default_factory=list)  # kept, length F
    # per-example grain, (N, HIDDEN) float64
    pq_c: dict = field(default_factory=dict)  # arm -> (N, H)
    pq_v: dict = field(default_factory=dict)
    groups: np.ndarray | None = None  # (N,) kept-context position per row
    q_idx: np.ndarray | None = None  # (N,) question index per row
    fams_rows: list[str] = field(default_factory=list)  # (N,) family per row
    committed_map: dict | None = None  # loaded maps/<b>/<s>/L14.npz arrays

    @property
    def n_ctx(self) -> int:
        return len(self.ctx_ids)

    @property
    def n_rows(self) -> int:
        return int(self.pq_c["base"].shape[0])

    @property
    def q_ids(self) -> list[int]:
        return sorted({int(q) for q in self.q_idx})


def _hf_fetch(rel: str, dest: Path) -> Path:
    """Per-file hf_hub_download at the pinned revision, symlinked into place."""
    from huggingface_hub import hf_hub_download

    if dest.exists():
        return dest
    local = hf_hub_download(DATA_REPO, rel, repo_type="dataset", revision=HF_REVISION)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        dest.unlink()
    dest.symlink_to(Path(local).resolve())
    logger.info("[phase=fetch] %s -> %s", rel, dest)
    return dest


def load_cell(behavior: str, substrate: str, reduced_root: Path, maps_root: Path) -> CellData:
    """Load + align one cell's summary L14 slice, per-question rows, committed map.

    Per-question ``row_context_index`` indexes the FULL-length context_ids array
    (issue813_run_cell writes it so); the summary keeps only contexts with ≥1
    usable question. Rows are re-keyed to the kept-context POSITION by context-id
    match (fail-loud on any row whose context is not kept).
    """
    sroot = reduced_root / behavior / substrate
    spath = _hf_fetch(
        f"{EXPERIMENT_NAME}/reduced/{behavior}/{substrate}/summary.npz", sroot / "summary.npz"
    )
    pqpath = _hf_fetch(
        f"{EXPERIMENT_NAME}/reduced/{behavior}/{substrate}/per_question_L{HEADLINE_LAYER}.npz",
        sroot / f"per_question_L{HEADLINE_LAYER}.npz",
    )
    mpath = _hf_fetch(
        f"{EXPERIMENT_NAME}/maps/{behavior}/{substrate}/L{HEADLINE_LAYER}.npz",
        maps_root / behavior / substrate / f"L{HEADLINE_LAYER}.npz",
    )

    cell = CellData(behavior=behavior, substrate=substrate)

    s = np.load(spath, allow_pickle=True)
    _require_npz_keys(spath, s, _SUMMARY_KEYS)
    c0s = np.asarray(s["c_C_base"], dtype=np.float64)
    assert c0s.ndim == 3 and c0s.shape[1:] == (N_LAYERS, HIDDEN), c0s.shape
    lay = HEADLINE_LAYER
    cell.avg_c = {
        "base": c0s[:, lay],
        "trained": np.asarray(s["c_C_trained"], dtype=np.float64)[:, lay],
    }
    cell.avg_v = {
        "base": np.asarray(s["v_A_base"], dtype=np.float64)[:, lay],
        "trained": np.asarray(s["v_A_trained"], dtype=np.float64)[:, lay],
    }
    cell.ctx_ids = [str(x) for x in s["context_ids"]]
    cell.families = [str(x) for x in s["families"]]

    d = np.load(pqpath, allow_pickle=True)
    _require_npz_keys(pqpath, d, _PQ_KEYS)
    got_layer = int(np.asarray(d["headline_layer"]))
    assert got_layer == HEADLINE_LAYER, (got_layer, HEADLINE_LAYER)
    cell.pq_c = {
        "base": np.asarray(d["c_C_base"], dtype=np.float64),
        "trained": np.asarray(d["c_C_trained"], dtype=np.float64),
    }
    cell.pq_v = {
        "base": np.asarray(d["v_A_base"], dtype=np.float64),
        "trained": np.asarray(d["v_A_trained"], dtype=np.float64),
    }
    n = cell.pq_c["base"].shape[0]
    for arm in ARMS:
        assert cell.pq_c[arm].shape == (n, HIDDEN), cell.pq_c[arm].shape
        assert cell.pq_v[arm].shape == (n, HIDDEN), cell.pq_v[arm].shape
    row_ctx = np.asarray(d["row_context_index"], dtype=np.int64)
    cell.q_idx = np.asarray(d["row_question_index"], dtype=np.int64)
    full_ctx_ids = [str(x) for x in d["context_ids"]]  # full-length (orig index)
    kept_pos = {cid: i for i, cid in enumerate(cell.ctx_ids)}
    groups = np.empty(n, dtype=np.int64)
    for r in range(n):
        cid = full_ctx_ids[int(row_ctx[r])]
        if cid not in kept_pos:
            raise RuntimeError(
                f"{behavior}/{substrate}: per-question row {r} context {cid!r} is not "
                "among the summary's kept contexts — producer/consumer misalignment"
            )
        groups[r] = kept_pos[cid]
    cell.groups = groups
    cell.fams_rows = [cell.families[g] for g in groups]

    m = np.load(mpath, allow_pickle=True)
    _require_npz_keys(mpath, m, _MAP_KEYS)
    cell.committed_map = {k: np.asarray(m[k]) for k in _MAP_KEYS}
    logger.info(
        "[phase=load] %s/%s: F=%d contexts, N=%d pq rows, %d questions, %d families",
        behavior,
        substrate,
        cell.n_ctx,
        n,
        len(cell.q_ids),
        len(set(cell.families)),
    )
    return cell


# ── ridge primitives (one shared eigh per design → PRESS + dual weights + preds) ──


def _press_alpha_from_eigh(
    evals: torch.Tensor, Q: torch.Tensor, Y: torch.Tensor, lambdas: list[float]
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """PRESS-LOO λ selection + dual alpha at argmin λ, from ONE eigh of the Gram.

    Reproduces ``fit658._press_loo_mse_per_lambda`` (same clamp/means) and
    ``fit658._ridge_dual_weights``'s alpha = (G+λI)⁻¹Y (via the eigenbasis instead
    of an LU solve — identical math; gated ≤1e-6 vs the fit658-composed oracle).
    Returns (λ*, alpha (m,P), per-λ PRESS MSE (n_lambda,)).
    """
    QtY = Q.t() @ Y
    Qsq = Q * Q
    mses = torch.empty(len(lambdas), dtype=Y.dtype)
    for li, lam in enumerate(lambdas):
        filt = evals / (evals + lam)
        h_diag = Qsq @ filt
        Yhat = Q @ (filt.unsqueeze(1) * QtY)
        resid = Y - Yhat
        denom = (1.0 - h_diag).clamp(min=1e-8).unsqueeze(1)
        loo = resid / denom
        mses[li] = (loo * loo).mean()
    best = int(torch.argmin(mses).item())
    lam = lambdas[best]
    alpha = Q @ (QtY / (evals + lam).unsqueeze(1))
    return lam, alpha, mses


def _alpha_at(evals: torch.Tensor, Q: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    """Dual alpha = (G+λI)⁻¹ Y from a precomputed eigendecomposition."""
    return Q @ ((Q.t() @ Y) / (evals + lam).unsqueeze(1))


def full_fit(X: np.ndarray, Y64: np.ndarray, *, per_lambda: bool = False) -> dict:
    """Full-data ridge fit (PRESS λ over all rows) — `_ridge_components` semantics.

    Returns {"W" (d,P) f64, "mu", "sd", "lambda", "press_mse" (n_lambda,),
    "W_per_lambda" optional list} — M(c) = ((c − mu)/sd) @ W.
    """
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(Y64)).to(dtype=torch.float64)
    mu = Xt.mean(0)
    sd = Xt.std(0, correction=0) + 1e-9
    Xn = (Xt - mu) / sd
    G = Xn @ Xn.t()
    evals, Q = torch.linalg.eigh(G)
    lam, alpha, mses = _press_alpha_from_eigh(evals, Q, Yt, LAMBDAS)
    out = {
        "W": (Xn.t() @ alpha).numpy(),
        "mu": mu.numpy(),
        "sd": sd.numpy(),
        "lambda": lam,
        "press_mse": mses.numpy(),
    }
    if per_lambda:
        out["W_per_lambda"] = [(Xn.t() @ _alpha_at(evals, Q, Yt, la)).numpy() for la in LAMBDAS]
    return out


def apply_fit(fit: dict, grid: np.ndarray) -> np.ndarray:
    """Evaluate a full_fit at grid rows → (g, P)."""
    return ((grid - fit["mu"]) / fit["sd"]) @ fit["W"]


# ── grouped-fold block-LOCO (the plan §9 sanctioned fallback shape) ──────────


def grouped_loco(
    X: np.ndarray,
    Yvars: dict[str, np.ndarray],
    groups: np.ndarray,
    extra_grids: dict[int, np.ndarray] | None,
    *,
    want_B_var: str | None = None,
    per_lambda: bool = False,
) -> dict:
    """Grouped-fold block-LOCO ridge: per fold ONE standardized-design GEMM + ONE
    shared eigh drives PRESS λ-selection, dual weights, held-out + grid preds.

    X (N, d) fp64; Yvars {name: (N, P_name)}; groups (N,) int fold id 0..F-1
    (a context's questions leave together — at the averaged grain each group is
    a single row, so this same function IS the parent's row-LOCO protocol).
    extra_grids {fold_id: (g_f, d)} — additional inputs evaluated per fold (the
    transfer reads). Standardization is per fold by TRAIN mu/sd (plan §0).

    Returns per Yvar: held-out preds (N, P) at the per-fold PRESS λ, grid preds
    {fold: (g_f, P)}, lambda per fold; optional per-λ preds; optional per-fold
    B = K_held (G_tr+λI)⁻¹ + train indices (for the DV4 frozen-λ shuffle refit).
    NO per-fold×per-λ refits and no serial tiny-op loop: the fold loop's per-fold
    work is large batched tensor ops (FLOP-bound — vectorize-many-cell-fits rule).
    """
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(dtype=torch.float64)
    fold_ids = sorted({int(g) for g in groups})
    gt = torch.from_numpy(np.ascontiguousarray(groups)).to(dtype=torch.long)
    Yts = {
        k: torch.from_numpy(np.ascontiguousarray(v)).to(dtype=torch.float64)
        for k, v in Yvars.items()
    }
    held_pred = {k: np.zeros_like(Yvars[k]) for k in Yvars}
    held_pred_pl = (
        {k: np.zeros((len(LAMBDAS), *Yvars[k].shape)) for k in Yvars} if per_lambda else None
    )
    grid_pred: dict[str, dict[int, np.ndarray]] = {k: {} for k in Yvars}
    grid_pred_pl: dict[str, dict[int, np.ndarray]] = {k: {} for k in Yvars} if per_lambda else {}
    lam_by_fold: dict[str, dict[int, float]] = {k: {} for k in Yvars}
    B_by_fold: dict[int, np.ndarray] = {}
    tr_idx_by_fold: dict[int, np.ndarray] = {}
    for f in fold_ids:
        held = torch.nonzero(gt == f, as_tuple=False).squeeze(-1)
        tr = torch.nonzero(gt != f, as_tuple=False).squeeze(-1)
        assert held.numel() >= 1 and tr.numel() >= 3, (f, held.numel(), tr.numel())
        Xtr = Xt[tr]
        mu = Xtr.mean(0)
        sd = Xtr.std(0, correction=0) + 1e-9
        Ztr = (Xtr - mu) / sd
        out_rows = [(Xt[held] - mu) / sd]
        g_f = None
        if extra_grids is not None and f in extra_grids:
            g_f = torch.from_numpy(np.ascontiguousarray(extra_grids[f])).to(dtype=torch.float64)
            out_rows.append((g_f - mu) / sd)
        Zout = torch.cat(out_rows, dim=0)
        G = Ztr @ Ztr.t()  # the ONE HIDDEN-dim design GEMM for this fold
        K = Zout @ Ztr.t()  # held(+grid) × train cross-Gram
        evals, Q = torch.linalg.eigh(G)  # shared across λ, PRESS, weights, all Yvars
        m_h = held.numel()
        for name, Yt_ in Yts.items():
            lam, alpha, _ = _press_alpha_from_eigh(evals, Q, Yt_[tr], LAMBDAS)
            lam_by_fold[name][f] = lam
            preds = (K @ alpha).numpy()
            held_pred[name][held.numpy()] = preds[:m_h]
            if g_f is not None:
                grid_pred[name][f] = preds[m_h:]
            if per_lambda:
                for li, la in enumerate(LAMBDAS):
                    p_la = (K @ _alpha_at(evals, Q, Yt_[tr], la)).numpy()
                    held_pred_pl[name][li, held.numpy()] = p_la[:m_h]
                    if g_f is not None:
                        grid_pred_pl[name].setdefault(
                            f, np.zeros((len(LAMBDAS), *preds[m_h:].shape))
                        )[li] = p_la[m_h:]
            if want_B_var == name:
                inv = Q @ torch.diag(1.0 / (evals + lam)) @ Q.t()
                B_by_fold[f] = (K[:m_h] @ inv).numpy()
                tr_idx_by_fold[f] = tr.numpy()
    out = {
        "held_pred": held_pred,
        "grid_pred": grid_pred,
        "lambda_by_fold": lam_by_fold,
        "fold_ids": fold_ids,
    }
    if per_lambda:
        out["held_pred_per_lambda"] = held_pred_pl
        out["grid_pred_per_lambda"] = grid_pred_pl
    if want_B_var is not None:
        out["B_by_fold"] = B_by_fold
        out["train_idx_by_fold"] = tr_idx_by_fold
    return out


def grouped_loco_serial_oracle(
    X: np.ndarray, Y: np.ndarray, groups: np.ndarray, extra_grids: dict[int, np.ndarray] | None
) -> dict:
    """Serial grouped-fold oracle composed from the TRUSTED fit658 primitives.

    Per fold: standardize by train mu/sd, PRESS λ via fit658._press_loo_mse_per_lambda,
    weights via fit658._ridge_dual_weights, predict held + grid rows. The live
    ``grouped_loco`` must reproduce this ≤1e-6 (plan §4 equivalence gate).
    """
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(Y)).to(dtype=torch.float64)
    held_pred = np.zeros_like(Y)
    grid_pred: dict[int, np.ndarray] = {}
    lam_by_fold: dict[int, float] = {}
    for f in sorted({int(g) for g in groups}):
        held = np.where(groups == f)[0]
        tr = np.where(groups != f)[0]
        Xtr = Xt[tr]
        mu = Xtr.mean(0)
        sd = Xtr.std(0, correction=0) + 1e-9
        Ztr = (Xtr - mu) / sd
        mse = fit658._press_loo_mse_per_lambda(Ztr, Yt[tr], LAMBDAS)
        lam = LAMBDAS[int(torch.argmin(mse).item())]
        w = fit658._ridge_dual_weights(Ztr, Yt[tr], lam)  # (d, P)
        lam_by_fold[f] = lam
        held_pred[held] = (((Xt[held] - mu) / sd) @ w).numpy()
        if extra_grids is not None and f in extra_grids:
            gt = torch.from_numpy(np.ascontiguousarray(extra_grids[f])).to(dtype=torch.float64)
            grid_pred[f] = (((gt - mu) / sd) @ w).numpy()
    return {"held_pred": held_pred, "grid_pred": grid_pred, "lambda_by_fold": lam_by_fold}


def block_loco_equivalence_gate(cell: CellData, basis: np.ndarray) -> dict:
    """Batched-vs-serial grouped-fold equivalence on THIS cell (rel tol 1e-6).

    Runs both paths on the base arm's per-example rows (shared basis targets) AND
    the averaged grain, compares held+grid predictions. Fail-loud on drift.
    """
    rel = {}
    for grain in ("pe", "avg"):
        if grain == "pe":
            X = cell.pq_c["base"]
            Y = cell.pq_v["base"] @ basis.T
            groups = cell.groups
            grids = {f: cell.avg_c["base"][f : f + 1] for f in range(cell.n_ctx)}
        else:
            X = cell.avg_c["base"]
            Y = cell.avg_v["base"] @ basis.T
            groups = np.arange(cell.n_ctx)
            grids = {f: cell.pq_c["base"][cell.groups == f] for f in range(cell.n_ctx)}
        live = grouped_loco(X, {"y": Y}, groups, grids)
        oracle = grouped_loco_serial_oracle(X, Y, groups, grids)
        num = float(np.linalg.norm(live["held_pred"]["y"] - oracle["held_pred"]))
        den = float(np.linalg.norm(oracle["held_pred"])) + 1e-30
        rel_held = num / den
        gnum = gden = 0.0
        for f, gp in oracle["grid_pred"].items():
            gnum += float(np.sum((live["grid_pred"]["y"][f] - gp) ** 2))
            gden += float(np.sum(gp**2))
        rel_grid = math.sqrt(gnum / (gden + 1e-30))
        lam_match = live["lambda_by_fold"]["y"] == oracle["lambda_by_fold"]
        rel[grain] = {"rel_held": rel_held, "rel_grid": rel_grid, "lambda_match": bool(lam_match)}
        if rel_held > 1e-6 or rel_grid > 1e-6 or not lam_match:
            raise RuntimeError(
                f"block-LOCO equivalence gate FAILED ({cell.behavior}/{cell.substrate}, "
                f"{grain}): rel_held={rel_held:.3e} rel_grid={rel_grid:.3e} "
                f"lambda_match={lam_match} (tol 1e-6)"
            )
    logger.info(
        "[phase=gate] block-LOCO equivalence PASS %s/%s: %s",
        cell.behavior,
        cell.substrate,
        rel,
    )
    return {"pass": True, "tol": 1e-6, **rel}


# ── R² accounting + DV1 ──────────────────────────────────────────────────────


def _fold_ss(
    y_true: np.ndarray, y_pred: np.ndarray, y_train_mean: np.ndarray
) -> tuple[float, float]:
    """(SS_res, SS_tot) for one fold's held-out units vs the TASK train-mean baseline."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_train_mean[None, :]) ** 2))
    return ss_res, ss_tot


def _pooled_r2(ss: list[tuple[float, float]]) -> float:
    res = sum(s[0] for s in ss)
    tot = sum(s[1] for s in ss)
    return float(1.0 - res / tot) if tot > 0 else float("nan")


def _family_boot_gap(
    ss_a: list[tuple[float, float]],
    ss_b: list[tuple[float, float]],
    fold_families: list[str],
    n_resamples: int,
) -> dict:
    """Family-clustered bootstrap CI on the pooled-R² gap R²(a) − R²(b) over folds.

    Resamples whole context FAMILIES of folds with replacement (the parent's
    7-family cluster unit), pools SS components per draw, takes the gap.
    """
    point = _pooled_r2(ss_a) - _pooled_r2(ss_b)
    fams = np.asarray(fold_families, dtype=object)
    uniq = sorted({str(f) for f in fams})
    if len(uniq) < 2:
        return {"point": point, "ci_lo": point, "ci_hi": point, "n_families": len(uniq)}
    fam_to_idx = {f: np.where(fams.astype(str) == f)[0] for f in uniq}
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(n_resamples):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([fam_to_idx[f] for f in chosen])
        ra = _pooled_r2([ss_a[i] for i in idx])
        rb = _pooled_r2([ss_b[i] for i in idx])
        if not (math.isnan(ra) or math.isnan(rb)):
            draws.append(ra - rb)
    if not draws:
        return {"point": point, "ci_lo": point, "ci_hi": point, "n_families": len(uniq)}
    arr = np.asarray(draws)
    return {
        "point": point,
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "n_families": len(uniq),
        "n_boot": len(draws),
    }


def dv1_transfer_arm(
    cell: CellData,
    arm: str,
    bases: dict[str, np.ndarray],
    *,
    fold_mode: str,
    boot_resamples: int,
    keep_dv4: bool,
) -> tuple[dict, dict | None]:
    """The four DV1 reads for one arm under LOCO (fold_mode='loco') or LOFO-7.

    Returns (result block, dv4 artifacts or None). The four reads share folds:
    M own / M→per-question from the averaged-grain call; h_pe own / h_pe→averaged
    from the per-example call. Both are computed under EVERY basis in ``bases``
    (primary 'shared' + the 'pq' sensitivity basis) from the same factorizations.
    """
    if fold_mode == "loco":
        groups_pe = cell.groups
        groups_avg = np.arange(cell.n_ctx)
        fold_fams = list(cell.families)
    else:  # lofo: leave one FAMILY out
        fam_ids = sorted(set(cell.families))
        fam_pos = {f: i for i, f in enumerate(fam_ids)}
        groups_avg = np.asarray([fam_pos[f] for f in cell.families], dtype=np.int64)
        groups_pe = groups_avg[cell.groups]
        fold_fams = fam_ids
    Yavg = {k: cell.avg_v[arm] @ b.T for k, b in bases.items()}
    Ypq = {k: cell.pq_v[arm] @ b.T for k, b in bases.items()}
    fold_ids_avg = sorted(set(groups_avg.tolist()))
    grids_avg = {f: cell.pq_c[arm][groups_pe == f] for f in fold_ids_avg}  # M→pq inputs
    grids_pe = {f: cell.avg_c[arm][groups_avg == f] for f in fold_ids_avg}  # h_pe→avg inputs
    res_avg = grouped_loco(
        cell.avg_c[arm], Yavg, groups_avg, grids_avg, per_lambda=(fold_mode == "loco")
    )
    res_pe = grouped_loco(
        cell.pq_c[arm],
        Ypq,
        groups_pe,
        grids_pe,
        want_B_var="shared" if keep_dv4 else None,
        per_lambda=(fold_mode == "loco"),
    )
    block: dict = {"reads": {}, "fold_mode": fold_mode}
    dv4_art = None
    for bname in bases:
        ss = {r: [] for r in ("m_own_avg", "hpe_to_avg", "hpe_own_pq", "m_to_pq")}
        per_fold_r2 = {r: [] for r in ss}
        for f in fold_ids_avg:
            avg_held = np.where(groups_avg == f)[0]
            pq_held = np.where(groups_pe == f)[0]
            avg_tr_mean = Yavg[bname][groups_avg != f].mean(axis=0)
            pq_tr_mean = Ypq[bname][groups_pe != f].mean(axis=0)
            y_avg = Yavg[bname][avg_held]
            y_pq = Ypq[bname][pq_held]
            reads = {
                "m_own_avg": (y_avg, res_avg["held_pred"][bname][avg_held], avg_tr_mean),
                "hpe_to_avg": (y_avg, res_pe["grid_pred"][bname][f], avg_tr_mean),
                "hpe_own_pq": (y_pq, res_pe["held_pred"][bname][pq_held], pq_tr_mean),
                "m_to_pq": (y_pq, res_avg["grid_pred"][bname][f], pq_tr_mean),
            }
            for r, (yt, yp, tm) in reads.items():
                s = _fold_ss(yt, yp, tm)
                ss[r].append(s)
                per_fold_r2[r].append(float(1.0 - s[0] / s[1]) if s[1] > 0 else float("nan"))
        bb: dict = {}
        for r in ss:
            bb[r] = {"r2_pooled": _pooled_r2(ss[r])}
            if fold_mode == "loco" and bname == "shared":
                bb[r]["per_fold_r2"] = per_fold_r2[r]
        if fold_mode == "loco":
            bb["gaps"] = {
                "delta_r2_avg_task": _family_boot_gap(
                    ss["m_own_avg"], ss["hpe_to_avg"], fold_fams, boot_resamples
                ),
                "delta_r2_pq_task": _family_boot_gap(
                    ss["hpe_own_pq"], ss["m_to_pq"], fold_fams, boot_resamples
                ),
            }
        else:
            bb["gaps"] = {
                "delta_r2_avg_task": {
                    "point": _pooled_r2(ss["m_own_avg"]) - _pooled_r2(ss["hpe_to_avg"])
                },
                "delta_r2_pq_task": {
                    "point": _pooled_r2(ss["hpe_own_pq"]) - _pooled_r2(ss["m_to_pq"])
                },
            }
        block["reads"][bname] = bb
    if fold_mode == "loco":
        block["lambda_by_fold"] = {
            "avg_grain": {b: res_avg["lambda_by_fold"][b] for b in bases},
            "pe_grain": {b: res_pe["lambda_by_fold"][b] for b in bases},
        }
        # per-λ pooled R² diagnostics (critique-recommended): shared basis only.
        pl: dict = {}
        for r in ("m_own_avg", "hpe_to_avg", "hpe_own_pq", "m_to_pq"):
            pl[r] = []
        for li in range(len(LAMBDAS)):
            ss_pl = {r: [] for r in pl}
            for f in fold_ids_avg:
                avg_held = np.where(groups_avg == f)[0]
                pq_held = np.where(groups_pe == f)[0]
                avg_tr_mean = Yavg["shared"][groups_avg != f].mean(axis=0)
                pq_tr_mean = Ypq["shared"][groups_pe != f].mean(axis=0)
                ss_pl["m_own_avg"].append(
                    _fold_ss(
                        Yavg["shared"][avg_held],
                        res_avg["held_pred_per_lambda"]["shared"][li, avg_held],
                        avg_tr_mean,
                    )
                )
                ss_pl["hpe_to_avg"].append(
                    _fold_ss(
                        Yavg["shared"][avg_held],
                        res_pe["grid_pred_per_lambda"]["shared"][f][li],
                        avg_tr_mean,
                    )
                )
                ss_pl["hpe_own_pq"].append(
                    _fold_ss(
                        Ypq["shared"][pq_held],
                        res_pe["held_pred_per_lambda"]["shared"][li, pq_held],
                        pq_tr_mean,
                    )
                )
                ss_pl["m_to_pq"].append(
                    _fold_ss(
                        Ypq["shared"][pq_held],
                        res_avg["grid_pred_per_lambda"]["shared"][f][li],
                        pq_tr_mean,
                    )
                )
            for r in pl:
                pl[r].append(_pooled_r2(ss_pl[r]))
        block["per_lambda_r2"] = {"lambdas": LAMBDAS, **pl}
        if keep_dv4:
            dv4_art = {
                "B_by_fold": res_pe["B_by_fold"],
                "train_idx_by_fold": res_pe["train_idx_by_fold"],
                "held_pred": res_pe["held_pred"]["shared"],
                "groups": groups_pe,
                "fold_ids": fold_ids_avg,
                "lambda_by_fold": res_pe["lambda_by_fold"]["shared"],
            }
    return block, dv4_art


# ── DV4: query-specific incremental R² + within-context shuffle null ─────────


def _within_context_perms(groups: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """(n_draws, N) row-index permutations shuffling question labels WITHIN contexts."""
    n = len(groups)
    perms = np.tile(np.arange(n), (n_draws, 1))
    for f in sorted(set(groups.tolist())):
        idx = np.where(groups == f)[0]
        for d in range(n_draws):
            perms[d, idx] = idx[rng.permutation(len(idx))]
    return perms


def dv4_query_specific(
    cell: CellData, arm: str, basis: np.ndarray, dv4_art: dict, n_draws: int
) -> dict:
    """R²_within (context-mean-centered held-out R²) vs the shuffle null.

    Null draws permute the TRAIN target rows within each context and refit with
    λ frozen at the observed per-fold PRESS choice: pred_perm = B_f @ Y_perm — a
    batched GEMM per fold over all stacked draws (#778 pattern), no re-eigh.
    """
    Y = cell.pq_v[arm] @ basis.T
    groups = dv4_art["groups"]
    rng = np.random.default_rng(SEED)
    perms = _within_context_perms(groups, n_draws, rng)
    ss_res_obs = 0.0
    ss_tot = 0.0
    ss_res_null = np.zeros(n_draws)
    n_folds_used = 0
    for f in dv4_art["fold_ids"]:
        held = np.where(groups == f)[0]
        if len(held) < 2:
            continue  # a single-question context has no within-context variance
        n_folds_used += 1
        y_h = Y[held]
        y_c = y_h - y_h.mean(axis=0, keepdims=True)
        p_obs = dv4_art["held_pred"][held]
        p_obs_c = p_obs - p_obs.mean(axis=0, keepdims=True)
        ss_res_obs += float(np.sum((y_c - p_obs_c) ** 2))
        ss_tot += float(np.sum(y_c**2))
        B = dv4_art["B_by_fold"][f]  # (m_h, m_tr)
        tr_idx = dv4_art["train_idx_by_fold"][f]
        chunk = max(1, int(64_000_000 / (len(tr_idx) * Y.shape[1] * 8)))
        for d0 in range(0, n_draws, chunk):
            dd = range(d0, min(d0 + chunk, n_draws))
            Yp = Y[perms[list(dd)][:, tr_idx]]  # (c, m_tr, P)
            preds = np.einsum("hm,cmp->chp", B, Yp)
            preds_c = preds - preds.mean(axis=1, keepdims=True)
            ss_res_null[list(dd)] += np.sum((y_c[None] - preds_c) ** 2, axis=(1, 2))
    r2_obs = float(1.0 - ss_res_obs / ss_tot) if ss_tot > 0 else float("nan")
    r2_null = 1.0 - ss_res_null / ss_tot if ss_tot > 0 else np.full(n_draws, np.nan)
    return {
        "r2_within_observed": r2_obs,
        "null_p95": float(np.nanpercentile(r2_null, 95)),
        "null_p50": float(np.nanpercentile(r2_null, 50)),
        "null_draws": [float(x) for x in r2_null],
        "n_draws": n_draws,
        "n_folds_used": n_folds_used,
        "lambda_frozen": True,
        "seed": SEED,
    }


def dv4_lambda_spotcheck(
    cell: CellData, arm: str, basis: np.ndarray, dv4_art: dict, n_spot: int
) -> dict:
    """Registered 5-draw per-draw-λ refit spot-check (assumption §12.10).

    Re-runs the FULL grouped LOCO on n_spot permuted-target draws with λ
    re-selected per fold via PRESS, and compares R²_within against the frozen-λ
    values for the SAME permutations.
    """
    Y = cell.pq_v[arm] @ basis.T
    groups = dv4_art["groups"]
    rng = np.random.default_rng(SEED)
    perms = _within_context_perms(groups, n_spot, rng)  # same leading draws as dv4
    deltas = []
    for d in range(n_spot):
        Yp = Y[perms[d]]
        res = grouped_loco(cell.pq_c[arm], {"y": Yp}, groups, None)
        ss_res_refit = ss_res_frozen = ss_tot = 0.0
        for f in dv4_art["fold_ids"]:
            held = np.where(groups == f)[0]
            if len(held) < 2:
                continue
            y_h = Y[held]
            y_c = y_h - y_h.mean(axis=0, keepdims=True)
            pr = res["held_pred"]["y"][held]
            pr_c = pr - pr.mean(axis=0, keepdims=True)
            B = dv4_art["B_by_fold"][f]
            tr_idx = dv4_art["train_idx_by_fold"][f]
            pf = B @ Yp[tr_idx]
            pf_c = pf - pf.mean(axis=0, keepdims=True)
            ss_res_refit += float(np.sum((y_c - pr_c) ** 2))
            ss_res_frozen += float(np.sum((y_c - pf_c) ** 2))
            ss_tot += float(np.sum(y_c**2))
        deltas.append(abs((1 - ss_res_refit / ss_tot) - (1 - ss_res_frozen / ss_tot)))
    return {"n_spot": n_spot, "max_abs_r2_delta": float(max(deltas)), "deltas": deltas}


# ── DV2: coefficient-space agreement (W_eff SVD subspaces + linear CKA) ───────


def w_eff(fit: dict) -> np.ndarray:
    """Effective linear map on RAW inputs: W_eff = diag(1/sd) @ W → (d, P)."""
    return fit["W"] / fit["sd"][:, None]


def _svd_sides(Weff: np.ndarray, kmax: int) -> tuple[np.ndarray, np.ndarray]:
    """(input-side U (d, kmax), output-side V (P, kmax)) top singular subspaces."""
    U, _, Vt = np.linalg.svd(Weff, full_matrices=False)
    return U[:, :kmax], Vt[:kmax].T


def subspace_overlap(A: np.ndarray, B: np.ndarray, k: int) -> float:
    """Mean squared canonical cosine between the two top-k column subspaces."""
    M = A[:, :k].T @ B[:, :k]
    return float(np.sum(M**2) / k)


def linear_cka(W1: np.ndarray, W2: np.ndarray) -> float:
    """Linear CKA between two maps (Gram-trick form, no d×d materialization)."""
    num = float(np.linalg.norm(W1.T @ W2) ** 2)
    den = float(np.linalg.norm(W1.T @ W1) * np.linalg.norm(W2.T @ W2))
    return num / den if den > 0 else float("nan")


K_GRID_OVERLAP = (5, 10, 20)  # k=10 pre-named headline (plan §11)


def overlap_block(Wa: np.ndarray, Wb: np.ndarray) -> dict:
    """All registered agreement metrics between two effective maps."""
    kmax = max(K_GRID_OVERLAP)
    Ua, Va = _svd_sides(Wa, kmax)
    Ub, Vb = _svd_sides(Wb, kmax)
    return {
        "overlap_input": {str(k): subspace_overlap(Ua, Ub, k) for k in K_GRID_OVERLAP},
        "overlap_output": {str(k): subspace_overlap(Va, Vb, k) for k in K_GRID_OVERLAP},
        "cka": linear_cka(Wa, Wb),
    }


def _question_half_splits(q_ids: list[int], n_splits: int) -> list[tuple[list[int], list[int]]]:
    """n_splits disjoint question-half splits (seed 42; halves of ⌊n_q/2⌋ each)."""
    rng = np.random.default_rng(SEED)
    half = len(q_ids) // 2
    out = []
    for _ in range(n_splits):
        perm = rng.permutation(q_ids)
        out.append(([int(x) for x in perm[:half]], [int(x) for x in perm[half : 2 * half]]))
    return out


def _avg_over_questions(
    cell: CellData, arm: str, q_subset: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Question-average c/v over ``q_subset`` per context → (ctx_keep, c̄, v̄).

    Returns (kept context positions, (m, H) c-stack, (m, H) v-stack). Contexts
    with zero rows in the subset are dropped (caller handles alignment).
    """
    qset = set(q_subset)
    in_sub = np.asarray([int(q) in qset for q in cell.q_idx], dtype=bool)
    keep, cs, vs = [], [], []
    for f in range(cell.n_ctx):
        m = (cell.groups == f) & in_sub
        if not m.any():
            continue
        keep.append(f)
        cs.append(cell.pq_c[arm][m].mean(axis=0))
        vs.append(cell.pq_v[arm][m].mean(axis=0))
    return np.asarray(keep, dtype=np.int64), np.stack(cs), np.stack(vs)


def dv2_and_twins(
    cell: CellData,
    basis: np.ndarray,
    basis_pq: np.ndarray,
    pe_fits: dict,
    avg_fits: dict,
    n_splits: int,
) -> tuple[dict, dict]:
    """DV2 observed agreement + the twin calibration distributions.

    Returns (coeff_agreement block, twin_transfer block). Twin families:
    - avg twins: two disjoint-question-half AVERAGED maps M_a/M_b → overlap/CKA
      ceiling AND the transfer-gap ceiling ΔR²_twin (LOCO, both directions).
    - pe twins: two disjoint-question-half PER-EXAMPLE maps → the pe-side
      overlap/CKA reliability ceiling (critique addition; DV2 read two-sided).
    Random-subspace floor is analytic: E[overlap] = k/D per side.
    """
    coeff: dict = {"per_arm": {}, "k_headline": 10}
    twin_transfer: dict = {"per_arm": {}, "n_splits": n_splits, "seed": SEED}
    splits = _question_half_splits(cell.q_ids, n_splits)
    for arm in ARMS:
        Wpe = w_eff(pe_fits[arm])
        Wavg = w_eff(avg_fits[arm])
        obs = overlap_block(Wpe, Wavg)
        # per-λ overlap diagnostics (critique-recommended): pe map at each λ vs
        # the observed averaged map (basis + normalization fixed).
        per_lambda = []
        for li, la in enumerate(LAMBDAS):
            Wl = pe_fits[arm]["W_per_lambda"][li] / pe_fits[arm]["sd"][:, None]
            per_lambda.append(
                {
                    "lambda": la,
                    "overlap_input_k10": subspace_overlap(
                        _svd_sides(Wl, 10)[0], _svd_sides(Wavg, 10)[0], 10
                    ),
                    "cka": linear_cka(Wl, Wavg),
                }
            )
        # twins
        avg_twin_overlap, pe_twin_overlap = [], []
        gap_draws: list[float] = []
        for qa, qb in splits:
            ka, ca, va = _avg_over_questions(cell, arm, qa)
            kb, cb, vb = _avg_over_questions(cell, arm, qb)
            fa = full_fit(ca, va @ basis.T)
            fb = full_fit(cb, vb @ basis.T)
            avg_twin_overlap.append(overlap_block(w_eff(fa), w_eff(fb)))
            # transfer-gap ceiling: both directions, LOCO on the target half's task,
            # restricted to contexts present in BOTH halves (folds align).
            common = sorted(set(ka.tolist()) & set(kb.tolist()))
            if len(common) >= 5:
                pa = {f: i for i, f in enumerate(ka.tolist())}
                pb = {f: i for i, f in enumerate(kb.tolist())}
                ia = [pa[f] for f in common]
                ib = [pb[f] for f in common]
                for X_src, Y_src, X_tgt, Y_tgt in (
                    (ca[ia], va[ia], cb[ib], vb[ib]),
                    (cb[ib], vb[ib], ca[ia], va[ia]),
                ):
                    Yt64 = Y_tgt @ basis.T
                    Ys64 = Y_src @ basis.T
                    g = np.arange(len(common))
                    own = grouped_loco(X_tgt, {"y": Yt64}, g, None)
                    xfer = grouped_loco(X_src, {"y": Ys64}, g, {f: X_tgt[f : f + 1] for f in g})
                    ss_own, ss_x = [], []
                    for f in g:
                        tm = Yt64[np.arange(len(common)) != f].mean(axis=0)
                        ss_own.append(
                            _fold_ss(Yt64[f : f + 1], own["held_pred"]["y"][f : f + 1], tm)
                        )
                        ss_x.append(_fold_ss(Yt64[f : f + 1], xfer["grid_pred"]["y"][f], tm))
                    gap_draws.append(_pooled_r2(ss_own) - _pooled_r2(ss_x))
            # pe twins: per-example maps on each half's rows (overlap ceiling only)
            qa_set, qb_set = set(qa), set(qb)
            ma = np.asarray([int(q) in qa_set for q in cell.q_idx], dtype=bool)
            mb = np.asarray([int(q) in qb_set for q in cell.q_idx], dtype=bool)
            if ma.sum() >= 8 and mb.sum() >= 8:
                fpa = full_fit(cell.pq_c[arm][ma], cell.pq_v[arm][ma] @ basis.T)
                fpb = full_fit(cell.pq_c[arm][mb], cell.pq_v[arm][mb] @ basis.T)
                pe_twin_overlap.append(overlap_block(w_eff(fpa), w_eff(fpb)))

        def _dist(blocks: list[dict], key: str, k: str | None) -> dict:
            vals = [b[key][k] if k else b[key] for b in blocks]
            arr = np.asarray(vals, dtype=float)
            if arr.size == 0:
                return {"n": 0}
            return {
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "p5": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)),
                "draws": [float(x) for x in arr],
            }

        coeff["per_arm"][arm] = {
            "observed_pe_vs_avg": obs,
            "per_lambda": per_lambda,
            "lambda_chosen": {"pe": pe_fits[arm]["lambda"], "avg": avg_fits[arm]["lambda"]},
            "twin_avg": {
                "overlap_input_k10": _dist(avg_twin_overlap, "overlap_input", "10"),
                "overlap_output_k10": _dist(avg_twin_overlap, "overlap_output", "10"),
                "overlap_input_full": {
                    str(k): _dist(avg_twin_overlap, "overlap_input", str(k)) for k in K_GRID_OVERLAP
                },
                "overlap_output_full": {
                    str(k): _dist(avg_twin_overlap, "overlap_output", str(k))
                    for k in K_GRID_OVERLAP
                },
                "cka": _dist(avg_twin_overlap, "cka", None),
            },
            "twin_pe": {
                "overlap_input_k10": _dist(pe_twin_overlap, "overlap_input", "10"),
                "overlap_output_k10": _dist(pe_twin_overlap, "overlap_output", "10"),
                "cka": _dist(pe_twin_overlap, "cka", None),
            },
            "random_floor": {
                "input": {str(k): k / HIDDEN for k in K_GRID_OVERLAP},
                "output": {str(k): k / basis.shape[0] for k in K_GRID_OVERLAP},
            },
        }
        arr = np.asarray(gap_draws, dtype=float)
        twin_transfer["per_arm"][arm] = {
            "gap_draws": [float(x) for x in arr],
            "gap_mean": float(arr.mean()) if arr.size else None,
            "gap_p95": float(np.percentile(arr, 95)) if arr.size else None,
            "n_draws": int(arr.size),
            "definition": "R2(own-half map, own LOCO task) - R2(other-half map -> same task)",
        }
    # basis sensitivity delta for DV2 (pq-derived basis, observed read only)
    coeff["basis_sensitivity"] = {}
    for arm in ARMS:
        f_pe = full_fit(cell.pq_c[arm], cell.pq_v[arm] @ basis_pq.T)
        f_avg = full_fit(cell.avg_c[arm], cell.avg_v[arm] @ basis_pq.T)
        coeff["basis_sensitivity"][arm] = overlap_block(w_eff(f_pe), w_eff(f_avg))
    return coeff, twin_transfer


# ── DV3: R²-vs-k averaging curve + analytic attenuation components ───────────


def dv3_curve(cell: CellData, basis: np.ndarray, n_draws: int) -> dict:
    """LOCO own-task R² of maps fit on k-question averages, k over the grid.

    Per (k, draw): draw k questions without replacement, question-average per
    context, refit + grouped LOCO (n≈50 rows — the tiny-avg-grain path), pooled
    R². Also the measured within/between-context variance components (64-d
    basis, output side; raw space, input side) for the reliability overlay:
    reliability(k) = σ²_signal / (σ²_signal + σ²_within / k).
    """
    rng = np.random.default_rng(SEED)
    q_ids = cell.q_ids
    ks = [k for k in DV3_K_GRID if k <= len(q_ids)]
    out: dict = {"k_grid": ks, "n_draws": n_draws, "seed": SEED, "per_arm": {}}
    for arm in ARMS:
        per_k = {}
        for k in ks:
            r2s = []
            for _ in range(n_draws):
                sub = [int(x) for x in rng.choice(q_ids, size=k, replace=False)]
                keep, cs, vs = _avg_over_questions(cell, arm, sub)
                if len(keep) < 5:
                    continue
                Y = vs @ basis.T
                g = np.arange(len(keep))
                res = grouped_loco(cs, {"y": Y}, g, None)
                ss = []
                for f in g:
                    tm = Y[g != f].mean(axis=0)
                    ss.append(_fold_ss(Y[f : f + 1], res["held_pred"]["y"][f : f + 1], tm))
                r2s.append(_pooled_r2(ss))
            arr = np.asarray(r2s, dtype=float)
            per_k[str(k)] = {
                "r2_mean": float(arr.mean()) if arr.size else None,
                "r2_sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
                "r2_p2_5": float(np.percentile(arr, 2.5)) if arr.size else None,
                "r2_p97_5": float(np.percentile(arr, 97.5)) if arr.size else None,
                "n_draws_used": int(arr.size),
                "draws": [float(x) for x in arr],
            }
        # variance components: within-context across-question + between-context
        comp = {}
        for side, rows in (("output_64d", cell.pq_v[arm] @ basis.T), ("input_raw", cell.pq_c[arm])):
            within, means, counts = [], [], []
            for f in range(cell.n_ctx):
                m = cell.groups == f
                if m.sum() < 2:
                    continue
                r = rows[m]
                within.append(r.var(axis=0, ddof=1))
                means.append(r.mean(axis=0))
                counts.append(int(m.sum()))
            s2_within = np.mean(np.stack(within), axis=0)
            mbar = float(np.mean(counts))
            s2_between = np.stack(means).var(axis=0, ddof=1)
            s2_signal = np.maximum(s2_between - s2_within / mbar, 0.0)
            rel = {
                str(k): float(
                    np.sum(s2_signal * (s2_signal / (s2_signal + s2_within / k + 1e-30)))
                    / (np.sum(s2_signal) + 1e-30)
                )
                for k in ks
            }
            comp[side] = {
                "sigma2_within_sum": float(np.sum(s2_within)),
                "sigma2_signal_sum": float(np.sum(s2_signal)),
                "m_bar": mbar,
                "reliability_signal_weighted": rel,
                "formula": "reliability(k) = s2_signal/(s2_signal + s2_within/k), "
                "signal-weighted mean over components",
            }
        out["per_arm"][arm] = {"per_k": per_k, "attenuation": comp}
    # registered smoke check: 20-draw CI half-width at the pre-named k=10-adjacent
    # points (raise draws to 50 if p95 half-width > 0.05 R² — plan §11).
    widths = []
    for arm in ARMS:
        for k, blk in out["per_arm"][arm]["per_k"].items():
            if blk["r2_p97_5"] is not None and blk["r2_p2_5"] is not None and int(k) < len(q_ids):
                widths.append((blk["r2_p97_5"] - blk["r2_p2_5"]) / 2.0)
    out["ci_halfwidth_check"] = {
        "max_halfwidth": float(max(widths)) if widths else None,
        "threshold": 0.05,
        "needs_50_draws": bool(widths and max(widths) > 0.05),
    }
    return out


# ── DV6: pre/post-FT agreement at per-example grain + floors + em null ───────

import issue722_fit_M as fitM  # noqa: E402  (parent fit machinery: basis, r_B, oracle fit_fn)
from issue722_bootstrap import floor_sd, make_refit_pair  # noqa: E402
from issue813_analysis import _floor_resample_indices  # noqa: E402  (make_refit_pair rng twin)


def _proj_stat(delta_full: np.ndarray, r_hat: np.ndarray | None) -> np.ndarray:
    """|Δ(c)·r̂| per grid row (behaviors) or ‖Δ(c)‖ (marker read-1, r_hat=None)."""
    if r_hat is None:
        return np.linalg.norm(delta_full, axis=1)
    return np.abs(delta_full @ r_hat)


def _fit_resample_pe(
    X: np.ndarray,
    Y: np.ndarray,
    idx: np.ndarray,
    grid: np.ndarray,
    *,
    frozen_lambda: float | None,
) -> np.ndarray:
    """One DV6 floor pair-member fit at pe grain, predictions at grid → (g, HIDDEN).

    Mirrors ``fitM._refit_ridge_fn`` semantics on the resampled rows: recompute
    the top-64 basis of the resampled (centered) Y, project UNCENTERED Y, ridge
    with PRESS λ (parent mode) or the frozen observed λ (frozen_lambda mode),
    back-project. The basis is the EXACT top-k eigenbasis via Lanczos ``eigsh``
    on the centered (m, m) Y Gram (identical subspace to ``_pca_basis_v0``'s
    truncated SVD; predictions are invariant to rotations/signs within it —
    the multi-output ridge with a shared λ is output-rotation-equivariant).
    """
    from scipy.sparse.linalg import eigsh

    Xb = X[idx]
    Yb = Y[idx]
    m = Xb.shape[0]
    Yc = Yb - Yb.mean(axis=0, keepdims=True)
    Gy = Yc @ Yc.T
    k = int(min(TARGET_DIM, m - 1))
    if k >= m - 1 or m < 200:  # tiny resample — dense eigh is cheaper/safer
        evals, U = np.linalg.eigh(Gy)
        evals, U = evals[::-1][:k], U[:, ::-1][:, :k]
    else:
        evals, U = eigsh(Gy, k=k, which="LA")
        order = np.argsort(evals)[::-1]
        evals, U = evals[order], U[:, order]
    pos = evals > max(1e-10 * float(evals.max(initial=0.0)), 0.0)
    inv_s = np.zeros_like(evals)
    inv_s[pos] = 1.0 / np.sqrt(evals[pos])
    basis_b = (U * inv_s[None, :]).T @ Yc  # (k, HIDDEN) — rows orthonormal on pos comps
    Y64b = Yb @ basis_b.T
    Xt = torch.from_numpy(np.ascontiguousarray(Xb)).to(dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(Y64b)).to(dtype=torch.float64)
    mu = Xt.mean(0)
    sd = Xt.std(0, correction=0) + 1e-9
    Xn = (Xt - mu) / sd
    G = Xn @ Xn.t()
    if frozen_lambda is None:
        evG, Q = torch.linalg.eigh(G)
        lam, alpha, _ = _press_alpha_from_eigh(evG, Q, Yt, LAMBDAS)
    else:
        lam = frozen_lambda
        A = G + lam * torch.eye(G.shape[0], dtype=G.dtype)
        alpha = torch.linalg.solve(A, Yt)
    gt = torch.from_numpy(np.ascontiguousarray(grid)).to(dtype=torch.float64)
    K = ((gt - mu) / sd) @ Xn.t()
    pred64 = (K @ alpha).numpy()
    return pred64 @ basis_b  # back to (g, HIDDEN) for the r̂ projection


def dv6_pre_post(
    cell: CellData,
    basis: np.ndarray,
    r_hat: np.ndarray | None,
    pe_fits: dict,
    *,
    n_pairs: int,
    floor_mode: str,
    run_oracle: bool,
) -> dict:
    """Δ_pe = median_c |(h_pe⁺ − h_pe⁰)(c̄0)·r̂| + per-example refit-pair floors.

    Grid = the cell's base averaged inputs c̄0 (parent's common_c_grid analog).
    Floors (parent convention, 3 targets sharing the resample-index stream of
    ``make_refit_pair`` via ``_floor_resample_indices``): h⁰ refit, h⁺ refit,
    shifted (X=c⁺ rows, Y=h⁰(c⁺) — the same-function shifted-design null).
    Behavior cells combine floors by SD (parent Delta_over_floor_sd); marker
    read-1 combines by p95. ``floor_mode``: 'parent' = per-resample basis + PRESS
    λ (faithful, slow); 'frozen_lambda' = per-resample basis, λ frozen at the
    observed full-data fit's choice (the documented descope lever).
    """
    grid = cell.avg_c["base"]
    p0 = apply_fit(pe_fits["base"], grid) @ basis
    pp = apply_fit(pe_fits["trained"], grid) @ basis
    proj_obs = _proj_stat(pp - p0, r_hat)
    delta_pe_med = float(np.median(proj_obs))

    # shifted-target Y: observed h⁰ applied at the trained pe inputs, in HIDDEN space
    shifted_Y = apply_fit(pe_fits["base"], cell.pq_c["trained"]) @ basis
    shift_fit = full_fit(cell.pq_c["trained"], shifted_Y @ basis.T)
    variants = {
        "h0_refit": (cell.pq_c["base"], cell.pq_v["base"], pe_fits["base"]["lambda"]),
        "hplus_refit": (cell.pq_c["trained"], cell.pq_v["trained"], pe_fits["trained"]["lambda"]),
        "shifted": (cell.pq_c["trained"], shifted_Y, shift_fit["lambda"]),
    }
    pairs = _floor_resample_indices(list(cell.fams_rows), n_pairs)
    floors: dict = {}
    stats_by_variant: dict[str, list[float]] = {}
    for vname, (X, Y, lam_obs) in variants.items():
        frozen = None if floor_mode == "parent" else lam_obs
        stats = []
        for ia, ib in pairs:
            pa = _fit_resample_pe(X, Y, ia, grid, frozen_lambda=frozen)
            pb = _fit_resample_pe(X, Y, ib, grid, frozen_lambda=frozen)
            stats.append(float(np.median(_proj_stat(pa - pb, r_hat))))
        stats_by_variant[vname] = stats
        arr = np.asarray(stats)
        floors[vname] = {
            "sd": floor_sd(arr),
            "p95": float(np.percentile(arr, 95)),
            "n_pairs": len(stats),
        }
    if r_hat is None:  # marker read-1: p95-combined floor (parent convention)
        floor_combined = max(floors[v]["p95"] for v in floors)
        floor_kind = "p95_combined"
    else:  # em/fact/syco: SD-combined floor (parent Delta_over_floor_sd)
        floor_combined = max(floors[v]["sd"] for v in floors)
        floor_kind = "sd_combined"
    dof = float(delta_pe_med / floor_combined) if floor_combined > 1e-12 else None
    out = {
        "delta_pe_med": delta_pe_med,
        "floors": floors,
        "floor_combined": floor_combined,
        "floor_kind": floor_kind,
        "delta_pe_over_floor": dof,
        "floor_mode": floor_mode,
        "grid": "avg_c_base (parent common_c_grid analog)",
        "n_pairs": n_pairs,
    }
    if run_oracle and r_hat is not None and floor_mode == "parent":
        # Serial oracle: the parent's make_refit_pair + fitM._refit_ridge_fn on the
        # SAME rows/indices (identical rng stream via _floor_resample_indices's
        # contract). Compares the h0_refit pair-stat array.
        oracle = make_refit_pair(
            cell.pq_c["base"],
            cell.pq_v["base"],
            fitM._refit_ridge_fn(grid),
            grid,
            r_hat,
            list(cell.fams_rows),
            n_pairs=n_pairs,
        )
        mine = np.asarray(stats_by_variant["h0_refit"])
        rel = float(np.max(np.abs(mine - oracle) / (np.abs(oracle) + 1e-30)))
        out["floor_oracle"] = {"max_rel_diff": rel, "n_pairs": n_pairs, "tol": 1e-6}
        if rel > 1e-6:
            raise RuntimeError(
                f"DV6 floor oracle FAILED ({cell.behavior}/{cell.substrate}): "
                f"max rel diff {rel:.3e} > 1e-6 vs make_refit_pair"
            )
        logger.info("[phase=gate] DV6 floor oracle PASS rel=%.3e", rel)
    return out


def dv6_em_null(
    cell: CellData,
    basis: np.ndarray,
    r_hat: np.ndarray | None,
    observed_floor: float,
    *,
    n_resamples: int,
) -> dict:
    """em-only per-example question-resampling null (200 draws, seed 42).

    Mirrors the parent substrate-swap null's draw semantics (questions drawn
    with replacement, split into two matched-n halves) but keeps the pseudo-arms
    at PER-EXAMPLE grain: fit h⁰/h⁺ on each half's rows (shared parent basis,
    PRESS λ), Δ_pe per half at the c̄0 grid, record |Δ(A) − Δ(B)|. The Δ/floor
    band comparison divides by the OBSERVED cell floor (a fixed rescale of the
    raw diffs — the per-pseudo-arm refit-floor of the parent's averaged-grain
    null is computationally out of reach at pe grain; recorded as null_space).
    """
    rng = np.random.default_rng(SEED)
    q_ids = cell.q_ids
    n_q = len(q_ids)
    half = n_q // 2
    grid = cell.avg_c["base"]
    rows_by_q = {q: np.where(cell.q_idx == q)[0] for q in q_ids}
    diffs = []
    for _ in range(n_resamples):
        drawn = [int(x) for x in rng.choice(q_ids, size=n_q, replace=True)]
        deltas = []
        for qs in (drawn[:half], drawn[half : 2 * half]):
            idx = np.concatenate([rows_by_q[q] for q in qs])
            if len({int(cell.groups[i]) for i in idx}) < 4:
                deltas.append(None)
                continue
            f0 = full_fit(cell.pq_c["base"][idx], cell.pq_v["base"][idx] @ basis.T)
            fp = full_fit(cell.pq_c["trained"][idx], cell.pq_v["trained"][idx] @ basis.T)
            d = (apply_fit(fp, grid) - apply_fit(f0, grid)) @ basis
            deltas.append(float(np.median(_proj_stat(d, r_hat))))
        if deltas[0] is None or deltas[1] is None:
            continue
        diffs.append(abs(deltas[0] - deltas[1]))
    arr = np.asarray(diffs, dtype=float)
    return {
        "null_space": "raw_delta_pe (fixed observed-floor rescale; per-pseudo-arm "
        "refit floors infeasible at pe grain — see report)",
        "n_resamples_used": int(arr.size),
        "n_resamples_requested": n_resamples,
        "null_p95_raw": float(np.percentile(arr, 95)) if arr.size else None,
        "null_p95_over_observed_floor": (
            float(np.percentile(arr, 95) / observed_floor)
            if arr.size and observed_floor > 1e-12
            else None
        ),
        "null_median_raw": float(np.median(arr)) if arr.size else None,
        "null_draws_raw": [float(x) for x in arr],
        "seed": SEED,
    }


# ── committed-map reproduction gate ───────────────────────────────────────────


def repro_gate(cell: CellData, avg_fits: dict) -> dict:
    """Refit-M vs the committed maps/<b>/<s>/L14.npz — ≤2% rel at the c̄ grid.

    Compares BACK-PROJECTED (n, HIDDEN) predictions (basis sign/rotation
    invariant) of my full-data averaged-map refits against the committed
    factored components, per arm, at each arm's own grid (M0 at c̄0, M⁺ at c̄⁺).
    Fail-loud >2% (per-layer round measured ≤1.25% jitter — plan §4).
    """
    cm = cell.committed_map
    out = {}
    for arm, wkey, mkey, skey in (
        ("base", "W_M0", "input_mean_C0", "input_std_C0"),
        ("trained", "W_Mplus", "input_mean_Cplus", "input_std_Cplus"),
    ):
        grid = cell.avg_c[arm]
        committed64 = ((grid - cm[mkey]) / cm[skey]) @ cm[wkey].astype(np.float64)
        committed_full = committed64 @ cm["pca_basis"].astype(np.float64)
        mine_full = apply_fit(avg_fits[arm], grid) @ avg_fits[arm]["basis"]
        rel = float(
            np.linalg.norm(mine_full - committed_full) / (np.linalg.norm(committed_full) + 1e-30)
        )
        out[arm] = rel
        if rel > 2e-2:
            raise RuntimeError(
                f"committed-map reproduction gate FAILED ({cell.behavior}/{cell.substrate} "
                f"{arm}): rel={rel:.4e} > 2e-2 — refit does not reproduce the parent map"
            )
    logger.info(
        "[phase=gate] repro gate PASS %s/%s: base=%.3e trained=%.3e",
        cell.behavior,
        cell.substrate,
        out["base"],
        out["trained"],
    )
    return {"pass": True, "tol": 2e-2, "rel_by_arm": out}


# ── per-cell orchestration + persistence ─────────────────────────────────────

import json  # noqa: E402  (stdlib; grouped here with its first use — persistence layer)

_CELL_SCHEMA_KEYS = ("regime", "gates", "dv1", "dv2", "twins", "dv3", "dv4", "dv6", "repro_meta")


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Atomic tmp + replace write (intra-phase persistence contract)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os_pid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def os_pid() -> int:
    import os

    return os.getpid()


def _cell_resume_valid(path: Path, expected_regime: dict) -> bool:
    """Regime-keyed resume predicate (v7 concern `perlayer-resume-stale-regime`).

    A cached cell JSON is reused IFF it parses, carries every schema key, AND its
    stored ``regime`` dict EQUALS the expected one — never row count alone.
    """
    if not path.exists():
        return False
    try:
        obj = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("[phase=resume] cached %s unreadable (%s) — recompute", path.name, e)
        return False
    missing = [k for k in _CELL_SCHEMA_KEYS if k not in obj]
    if missing:
        logger.warning("[phase=resume] cached %s missing keys %s — recompute", path.name, missing)
        return False
    if obj.get("regime") != expected_regime:
        logger.warning(
            "[phase=resume] cached %s regime mismatch (stored != expected) — recompute", path.name
        )
        return False
    return True


def _merge_shared_json(path: Path, cell_key: str, block: dict, repro: dict) -> None:
    """Incrementally assemble a shared multi-cell JSON (read-modify-atomic-replace)."""
    obj = {}
    if path.exists():
        try:
            obj = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            obj = {}
    obj.setdefault("per_cell", {})[cell_key] = block
    obj["repro_meta"] = repro
    _atomic_write_json(path, obj)


def _load_committed_reads(behavior: str, substrate: str) -> dict:
    """Committed averaged-grain Δ/floor + (em) parent 1000-draw null p95."""
    out: dict = {}
    dpath = PROJECT_ROOT / f"eval_results/issue_813/delta_floor/{behavior}__{substrate}.json"
    if dpath.exists():
        d = json.loads(dpath.read_text())
        out["avg_delta_over_floor"] = d.get("delta_over_floor")
        out["avg_delta_med"] = d.get("delta_med")
    npath = (
        PROJECT_ROOT / f"eval_results/issue_813/substrate_swap_null/{behavior}__{substrate}.json"
    )
    if npath.exists():
        n = json.loads(npath.read_text())
        out["parent_null_over_floor_p95_1000draw"] = n.get("null_over_floor_p95")
        out["parent_n_resamples"] = n.get("n_resamples_used")
    return out


def _save_pe_maps_npz(cell: CellData, pe_fits: dict, basis: np.ndarray, out_dir: Path) -> Path:
    """Persist the fitted per-example factored maps for the later HF upload
    (issue813_mapchange_substrate/maps_per_example/L14/ — orchestrator-owned)."""
    path = (
        out_dir
        / "maps_per_example"
        / f"L{HEADLINE_LAYER}"
        / (f"{cell.behavior}__{cell.substrate}.npz")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        W_pe_base=pe_fits["base"]["W"].astype(np.float32),
        W_pe_trained=pe_fits["trained"]["W"].astype(np.float32),
        input_mean_base=pe_fits["base"]["mu"],
        input_std_base=pe_fits["base"]["sd"],
        input_mean_trained=pe_fits["trained"]["mu"],
        input_std_trained=pe_fits["trained"]["sd"],
        lambda_base=np.asarray(pe_fits["base"]["lambda"]),
        lambda_trained=np.asarray(pe_fits["trained"]["lambda"]),
        pca_basis=basis.astype(np.float32),
        behavior=np.asarray(cell.behavior),
        substrate=np.asarray(cell.substrate),
        layer=np.asarray(HEADLINE_LAYER),
        n_rows=np.asarray(cell.n_rows),
        hf_revision=np.asarray(HF_REVISION),
        git_sha=np.asarray(_git_sha()),
        generated_at=np.asarray(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
    )
    return path


def process_cell(
    cell: CellData,
    args: argparse.Namespace,
    rb_main: dict,
    rb_fact: dict | None,
    run_equiv_gate: bool,
) -> dict:
    """All DV legs + gates for one (behavior, substrate) cell. Returns the JSON dict."""
    timings: dict[str, float] = {}
    t0 = time.time()
    basis = fitM._pca_basis_v0(cell.avg_v["base"], TARGET_DIM)  # shared parent basis
    basis_pq = fitM._pca_basis_v0(cell.pq_v["base"], TARGET_DIM)  # sensitivity basis
    r_hat = (
        None
        if cell.behavior == "marker"
        else fitM._r_hat_for(cell.behavior, HEADLINE_LAYER, rb_main, rb_fact)
    )

    # full-data fits (both grains, both arms) — DV2 / DV6 / repro-gate inputs
    pe_fits, avg_fits = {}, {}
    for arm in ARMS:
        pe_fits[arm] = full_fit(cell.pq_c[arm], cell.pq_v[arm] @ basis.T, per_lambda=True)
        pe_fits[arm]["basis"] = basis
        avg_fits[arm] = full_fit(cell.avg_c[arm], cell.avg_v[arm] @ basis.T)
        avg_fits[arm]["basis"] = basis
    timings["full_fits"] = time.time() - t0

    gates: dict = {"ridge_exactness": True}
    gates["repro"] = repro_gate(cell, avg_fits)
    if run_equiv_gate:
        t = time.time()
        gates["block_loco_equiv"] = block_loco_equivalence_gate(cell, basis)
        timings["equiv_gate"] = time.time() - t

    bases = {"shared": basis, "pq": basis_pq}
    dv1: dict = {"loco": {}, "lofo": {}}
    dv4_block: dict = {}
    t = time.time()
    dv4_arts = {}
    for arm in ARMS:
        dv1["loco"][arm], dv4_arts[arm] = dv1_transfer_arm(
            cell,
            arm,
            bases,
            fold_mode="loco",
            boot_resamples=args.boot_resamples,
            keep_dv4=True,
        )
        dv1["lofo"][arm], _ = dv1_transfer_arm(
            cell, arm, {"shared": basis}, fold_mode="lofo", boot_resamples=0, keep_dv4=False
        )
    timings["dv1"] = time.time() - t

    t = time.time()
    dv2, twin_transfer = dv2_and_twins(cell, basis, basis_pq, pe_fits, avg_fits, args.twin_splits)
    timings["dv2_twins"] = time.time() - t

    t = time.time()
    dv3 = dv3_curve(cell, basis, args.dv3_draws)
    timings["dv3"] = time.time() - t

    t = time.time()
    for arm in ARMS:
        dv4_block[arm] = dv4_query_specific(cell, arm, basis, dv4_arts[arm], args.dv4_draws)
    if args.dv4_lambda_spotcheck > 0:
        dv4_block["lambda_spotcheck"] = dv4_lambda_spotcheck(
            cell, "trained", basis, dv4_arts["trained"], args.dv4_lambda_spotcheck
        )
    timings["dv4"] = time.time() - t

    t = time.time()
    dv6 = dv6_pre_post(
        cell,
        basis,
        r_hat,
        pe_fits,
        n_pairs=args.dv6_pairs,
        floor_mode=args.dv6_floor_mode,
        run_oracle=args.dv6_oracle,
    )
    dv6["committed_averaged"] = _load_committed_reads(cell.behavior, cell.substrate)
    timings["dv6_floors"] = time.time() - t
    if cell.behavior == "em" and args.dv6_null_resamples > 0:
        t = time.time()
        dv6["em_pe_null"] = dv6_em_null(
            cell,
            basis,
            r_hat,
            dv6["floor_combined"],
            n_resamples=args.dv6_null_resamples,
        )
        timings["dv6_em_null"] = time.time() - t

    # kill criterion (plan §3): generic substrate, h_pe own held-out R² ≤ 0 both arms
    kill = None
    if cell.substrate == "generic":
        r2s = {arm: dv1["loco"][arm]["reads"]["shared"]["hpe_own_pq"]["r2_pooled"] for arm in ARMS}
        kill = {
            "hpe_own_r2_by_arm": r2s,
            "power_limited": bool(all(v <= 0 for v in r2s.values())),
        }

    maps_path = _save_pe_maps_npz(cell, pe_fits, basis, args.out_dir)
    timings["total"] = time.time() - t0
    logger.info(
        "[phase=cell] %s/%s DONE in %.1fs (dv1=%.1fs dv2=%.1fs dv3=%.1fs dv4=%.1fs dv6=%.1fs)",
        cell.behavior,
        cell.substrate,
        timings["total"],
        timings.get("dv1", 0),
        timings.get("dv2_twins", 0),
        timings.get("dv3", 0),
        timings.get("dv4", 0),
        timings.get("dv6_floors", 0) + timings.get("dv6_em_null", 0),
    )
    return {
        "behavior": cell.behavior,
        "substrate": cell.substrate,
        "headline_layer": HEADLINE_LAYER,
        "n_ctx": cell.n_ctx,
        "n_rows": cell.n_rows,
        "n_questions": len(cell.q_ids),
        "n_families": len(set(cell.families)),
        "ctx_ids": cell.ctx_ids,
        "ctx_families": cell.families,
        "target_dim_realized": int(basis.shape[0]),
        "target_dim_pq_basis": int(basis_pq.shape[0]),
        "regime": _regime(args),
        "gates": gates,
        "dv1": dv1,
        "dv2": dv2,
        "twins": twin_transfer,
        "dv3": dv3,
        "dv4": dv4_block,
        "dv6": dv6,
        "dv5": "N/A (pre-registered: no per-question judge outputs exist — plan Divergence #4)",
        "kill_criterion": kill,
        "pe_maps_npz": str(maps_path),
        "timings_s": timings,
        "repro_meta": _repro_meta(args),
    }


def timing_probe(n_rows: int) -> dict:
    """One production-shape fold through the LIVE grouped_loco path (synthetic data).

    Grounds the compute-deviation projection in a MEASURED per-call cost at
    production shape (never the plan's asserted figure — #823 lesson).
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_rows, HIDDEN))
    Y = rng.standard_normal((n_rows, TARGET_DIM))
    held = max(1, n_rows // 50)
    groups = np.zeros(n_rows, dtype=np.int64)
    groups[held:] = 1  # fold 0 = production-size train set (n - held rows)
    t0 = time.time()
    grouped_loco(X, {"y": Y}, groups, None)
    wall = time.time() - t0
    logger.info(
        "[phase=probe] grouped_loco 2-fold probe at n=%d: %.2fs total "
        "(fold-0 train m=%d dominates; per-production-fold ≈ %.2fs)",
        n_rows,
        wall,
        n_rows - held,
        wall / 2,
    )
    return {"n_rows": n_rows, "wall_s_2folds": wall, "per_fold_est_s": wall / 2}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Issue #813 per-example-vs-averaged-map driver (frozen L14, 12 cells)"
    )
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS), choices=list(BEHAVIORS))
    ap.add_argument("--substrates", nargs="+", default=list(SUBSTRATES), choices=list(SUBSTRATES))
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / OUT_SUBDIR)
    ap.add_argument(
        "--reduced-root", type=Path, default=PROJECT_ROOT / "eval_results/issue_813/reduced"
    )
    ap.add_argument(
        "--maps-root", type=Path, default=PROJECT_ROOT / "eval_results/issue_813_maps_pinned"
    )
    ap.add_argument("--dv3-draws", type=int, default=20, help="draws per k (plan §11; smoke-check)")
    ap.add_argument("--dv4-draws", type=int, default=200)
    ap.add_argument(
        "--dv4-lambda-spotcheck",
        type=int,
        default=0,
        help="N per-draw-λ refit spot-check draws (registered smoke check; assumption §12.10)",
    )
    ap.add_argument("--twin-splits", type=int, default=50, help="question-half splits (brief: 50)")
    ap.add_argument("--dv6-pairs", type=int, default=100)
    ap.add_argument(
        "--dv6-floor-mode",
        choices=("parent", "frozen_lambda"),
        default="parent",
        help="parent = per-resample basis + PRESS λ (faithful); frozen_lambda = the descope lever",
    )
    ap.add_argument("--dv6-null-resamples", type=int, default=200, help="em-only pe null draws")
    ap.add_argument("--boot-resamples", type=int, default=1000)
    ap.add_argument(
        "--equiv-cells",
        default="em:elicit,sycophancy:elicit",
        help="cells (behavior:substrate,...) that run the batched-vs-serial block-LOCO gate",
    )
    ap.add_argument("--dv6-oracle", action="store_true", help="run the DV6 serial floor oracle")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument(
        "--timing-probe-rows",
        type=int,
        default=0,
        help="run a production-shape grouped_loco probe at N rows, then continue",
    )
    return ap


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fit658.DEVICE = fit658._resolve_device("cpu")  # closed-form ridge — CPU by design
    fit658._assert_ridge_exactness()  # #658 reduction-order gate (fail-fast at startup)
    logger.info("[phase=start] device=%s; ridge exactness gate PASS", fit658.DEVICE)
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.timing_probe_rows > 0:
        probe = timing_probe(args.timing_probe_rows)
        _atomic_write_json(args.out_dir / "timing_probe.json", {**probe, **_repro_meta(args)})

    need_rb = any(b in ("em", "sycophancy") for b in args.behaviors)
    rb_main = fitM._load_rb_main() if need_rb else {}
    rb_fact = fitM._load_rb_fact() if "fact" in args.behaviors else None
    if "fact" in args.behaviors and rb_fact is None:
        raise RuntimeError("fact requested but r_b_fact.pt unavailable — cannot compute DV6 r̂")

    equiv_cells = {tuple(c.split(":")) for c in args.equiv_cells.split(",") if c}
    expected_regime = _regime(args)
    n_done = 0
    t_run = time.time()
    for behavior in args.behaviors:
        for substrate in args.substrates:
            out_path = args.out_dir / f"transfer_L{HEADLINE_LAYER}_{behavior}__{substrate}.json"
            if not args.no_resume and _cell_resume_valid(out_path, expected_regime):
                logger.info(
                    "[phase=resume] %s/%s cached (regime match) — skip", behavior, substrate
                )
                cached = json.loads(out_path.read_text())
                _merge_shared_json(
                    args.out_dir / f"coeff_agreement_L{HEADLINE_LAYER}.json",
                    f"{behavior}__{substrate}",
                    cached["dv2"],
                    cached["repro_meta"],
                )
                _merge_shared_json(
                    args.out_dir / f"dv6_pe_vs_avg_L{HEADLINE_LAYER}.json",
                    f"{behavior}__{substrate}",
                    cached["dv6"],
                    cached["repro_meta"],
                )
                n_done += 1
                continue
            cell = load_cell(behavior, substrate, args.reduced_root, args.maps_root)
            result = process_cell(
                cell, args, rb_main, rb_fact, run_equiv_gate=(behavior, substrate) in equiv_cells
            )
            _atomic_write_json(out_path, result)
            _merge_shared_json(
                args.out_dir / f"coeff_agreement_L{HEADLINE_LAYER}.json",
                f"{behavior}__{substrate}",
                result["dv2"],
                result["repro_meta"],
            )
            _merge_shared_json(
                args.out_dir / f"dv6_pe_vs_avg_L{HEADLINE_LAYER}.json",
                f"{behavior}__{substrate}",
                result["dv6"],
                result["repro_meta"],
            )
            n_done += 1
            logger.info("[phase=persist] wrote %s", out_path)
    logger.info(
        "[phase=done] %d/%d cells complete in %.1fs",
        n_done,
        len(args.behaviors) * len(args.substrates),
        time.time() - t_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
