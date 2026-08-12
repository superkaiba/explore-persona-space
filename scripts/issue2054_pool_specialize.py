"""Issue #2054 Gap B: pool-then-specialize nested models of the context->answer map.

Three NESTED models of v_X -> v_A (X = C context arm, P prefix arm), fit across
ALL cells at once on the #2054 lattice (the #1639 nesting):

  M0  one POOLED GCV-ridge over all cells' rows, single global offset
      (standardize-X / center-Y — the global offset is the train mean pair).
  M1  M0 + per-cell BIAS: b_cell = train-mean(y) - M0(train-mean(x)) — the
      exact least-squares per-cell intercept given the frozen pooled slope
      (M0 is affine, so mean-of-preds == pred-of-mean).
  M2  M1 + per-cell LOW-RANK slope correction, rank k in {8, 32, 128}: a
      GCV-ridge from the cell's train-fold PCA-k input basis (center-only PCA,
      the ``issue2054_ctx2ctx_fit._reduced_basis`` convention) to the M1
      residuals — rank <= k by construction. LINEAR throughout (no MLP /
      kernel / nonlinear readout — none requested).

Reused cores (do-not-reimplement contract): ``scripts/issue2054_ctx2ctx_fit``
supplies ``SharedEighRidge`` (fit_h-parity GCV ridge, #1887 dof cap 0.9;
pure-GCV at n<d refuses by construction), the production fold-map loader with
its smoke-map refusal floors (the working-tree ``shared_fold_map.json`` on
main is a 1,761-conv smoke map and is REFUSED), the cell loader + join floors,
and the metadata/JSON shape. ``analysis.mapping_baselines`` supplies the
mandatory identity+bias baselines (per-cell AND global-bias forms; d_in ==
d_out here) and the kNN retrieval read.

POOLED fit is computed from STREAMED second moments (per-fold C_xx, C_xy,
sums accumulated one cell at a time — the ~n_total x 3584 pooled matrix is
never materialized). ``PooledMomentRidge`` numerically mirrors
``SharedEighRidge``'s cov side; parity is asserted at pilot time on a
materializable one-cell subset (``_pooled_parity_gate``).

Scoring: per cell, per fold, held-out R^2 (``fit_h.reconstruction_metrics``,
fold-local centering — the banked-ceiling convention) for M0/M1/M2 plus the
identity baselines; fraction-of-ceiling against the cell's OWN banked
within-cell fit (``issue2054_lattice/fits/{cell}.json`` on the HF data repo,
``arm_reports[arm].pooled.r2_ambient_mean``; fold subsets use the banked
per-fold values) — ceilings are REUSED, never refit. A ceiling below
max(0.01, banked null p95) is flagged degenerate and yields fraction null
(explicit, never a silent divide).

Nulls (fit-side shuffled-pair, the ``issue2054_fits._shuffled_answer_null_r2``
convention via ``SharedEighRidge.null_r2``) run on the per-cell M2 residual
corrections — the only per-cell fitted slopes. Default 20 draws; a unit whose
fitted residual-space R^2 lands within ``--null-escalate-sd`` (3) sd of the
null band escalates to ``--null-max-draws`` (100); realized counts recorded.
The POOLED map gets no per-draw null (a pooled refit per draw at n ~ 10^5 is
the cost wall the 20-draw default exists to avoid; the parent's banked
per-cell nulls already establish per-cell signal vs chance).

Bootstrap: conversation-level, 200 draws. Cells share one conversation
population, so the aggregate resamples CONVERSATION IDS once per draw from
the union and applies the SAME resample to every cell (cross-cell coupling
preserved); per-draw R^2 uses the fixed full-scored-mean SS_tot convention
(recorded as ``bootstrap_sstot_convention``) so draws are pure GEMMs over
per-row e^2 / s^2 sidecars. M1-M0 and M2-M1 increments get per-cell and
cross-cell-mean CIs (the #1639 reporting shape).

Outputs: ``<out-root>/percell/{cell}__{arm}.json`` (+ ``.rows.npz`` e2/s2
sidecar) checkpoint-per-cell with fingerprinted resume; ``pooled_{arm}.json``;
``aggregate.json`` once every unit is present. ``--pilot`` scores fold 0 only
(training still uses the full 4-fold complement) and writes under
``<out-root>/pilot/``.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + HF/creds BEFORE torch import (code-style.md)

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    # script mode puts scripts/ (not the repo root) on sys.path[0] (gotchas.md).
    sys.path.insert(0, str(_REPO))

from scipy.linalg import eigh as scipy_eigh

from explore_persona_space.analysis.mapping_baselines import identity_bias_predict, knn_retrieval
from explore_persona_space.experiments.issue_779.fit_h import reconstruction_metrics
from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance
from scripts.issue2054_ctx2ctx_fit import (
    ARM_VEC_KEY,
    ARMS,
    D_AMBIENT,
    DEFAULT_LAMBDAS,
    GCV_DOF_CAP,
    MIN_JOIN_ABS,
    MIN_JOIN_FRAC,
    NULL_SEED_BASE,
    Cell,
    SharedEighRidge,
    discover_cells,
    load_cell,
    load_fold_map,
)

SCRIPT_VERSION = "issue2054_pool_specialize_v1"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
CEILING_HF_PREFIX = "issue2054_lattice/fits"
DEFAULT_RANKS = (8, 32, 128)
CEILING_FLOOR = 0.01  # ceilings at/below max(this, banked null p95) -> fraction null + flag
CONSTANT_X_VAR_FLOOR = 1e-12  # max per-feature variance below this = degenerate constant-X cell


def _log(msg: str) -> None:
    print(msg, flush=True)


def _seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "little")


def _model_names(ranks: list[int]) -> list[str]:
    return ["m0", "m1", *[f"m2_k{r}" for r in ranks], "identity_cell", "identity_global"]


def load_cell_with_answer(cell: Cell) -> dict:
    """Sibling ``load_cell`` + the ``v_A`` answer vectors — the ctx2ctx loader
    maps context->context and never exposes v_A (this script's targets)."""
    act = load_cell(cell)
    z = np.load(cell.path, allow_pickle=False)
    if "v_A" not in z:
        raise ValueError(f"{cell.path} missing key 'v_A'")
    v_a = np.asarray(z["v_A"], dtype=np.float32)
    if v_a.shape != act["v_C"].shape:
        raise ValueError(f"{cell.path}: v_A shape {v_a.shape} != v_C shape {act['v_C'].shape}")
    act["v_A"] = v_a
    return act


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell join (single-sided sibling of issue2054_ctx2ctx_fit.join_pair)


def join_cell(act: dict, fold_of: dict, k: int, arm: str) -> dict:
    """Restrict one cell's rows to the fold map's population (prefix arm further
    to rows with v_P_present). Fails LOUD on a small join or any empty fold."""
    if arm == "prefix":
        common = [
            cid
            for cid in act["conv_ids"]
            if cid in fold_of and act["v_P_present"][act["row_of"][cid]]
        ]
    else:
        common = [cid for cid in act["conv_ids"] if cid in fold_of]
    n_cell = len(act["conv_ids"])
    floor = max(MIN_JOIN_ABS, int(MIN_JOIN_FRAC * n_cell))
    if len(common) < floor:
        raise RuntimeError(
            f"conv_id join unexpectedly small (arm={arm}): n_join={len(common)} < floor {floor} "
            f"(n_cell={n_cell}, fold_map n={len(fold_of)})"
        )
    order = sorted(common)
    rows = np.fromiter((act["row_of"][c] for c in order), dtype=np.int64, count=len(order))
    fold_rows: list[list[int]] = [[] for _ in range(k)]
    for i, cid in enumerate(order):
        fold_rows[int(fold_of[cid])].append(i)
    for fi, fr in enumerate(fold_rows):
        if not fr:
            raise RuntimeError(f"fold {fi} empty after join (arm={arm}, n_join={len(order)})")
    return {
        "order": order,
        "rows": rows,
        "fold_rows": [np.asarray(fr, dtype=np.int64) for fr in fold_rows],
        "n_join": len(order),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Streaming pooled moments + the moment-based GCV ridge (M0)


def _eigh_robust(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.linalg.eigh with the cuSOLVER->CPU fallback (gotchas.md; exact
    backend swap, never a jitter)."""
    try:
        return torch.linalg.eigh(t)
    except torch.linalg.LinAlgError:
        if t.device.type == "cpu":
            raise
        w, v = torch.linalg.eigh(t.cpu())
        _log(f"[poolspec] cuda eigh non-convergence -> CPU fallback (n={t.shape[0]})")
        return w.to(t.device), v.to(t.device)


def accumulate_pooled_moments(
    cells: list[Cell], fold_of: dict, k: int, arms: list[str], device: str
) -> dict:
    """One streaming pass over cells: per (arm, fold) second moments for the
    pooled fit. The pooled train matrix is NEVER materialized."""
    dev = torch.device(device)
    d = D_AMBIENT
    mom: dict[str, list[dict]] = {}
    joins: dict[str, dict[str, dict]] = {arm: {} for arm in arms}
    for arm in arms:
        mom[arm] = [
            {
                "n": 0,
                "sum_x": torch.zeros(d, dtype=torch.float64, device=dev),
                "sum_y": torch.zeros(d, dtype=torch.float64, device=dev),
                "yss": 0.0,
                "c_xx": torch.zeros(d, d, dtype=torch.float64, device=dev),
                "c_xy": torch.zeros(d, d, dtype=torch.float64, device=dev),
            }
            for _ in range(k)
        ]
    for ci, cell in enumerate(cells):
        t0 = time.time()
        act = load_cell_with_answer(cell)
        for arm in arms:
            j = join_cell(act, fold_of, k, arm)
            joins[arm][cell.key] = {"n_join": j["n_join"]}
            vec = ARM_VEC_KEY[arm]
            for f in range(k):
                idx = j["rows"][j["fold_rows"][f]]
                x = torch.as_tensor(act[vec][idx].astype(np.float64), device=dev)
                y = torch.as_tensor(act["v_A"][idx].astype(np.float64), device=dev)
                m = mom[arm][f]
                m["n"] += int(x.shape[0])
                m["sum_x"] += x.sum(0)
                m["sum_y"] += y.sum(0)
                m["yss"] += float((y * y).sum())
                m["c_xx"] += x.T @ x
                m["c_xy"] += x.T @ y
        del act
        _log(
            f"[poolspec] moments cell {ci + 1}/{len(cells)} {cell.key} "
            f"elapsed={time.time() - t0:.1f}s"
        )
    return {"mom": mom, "joins": joins}


class PooledMomentRidge:
    """GCV ridge from accumulated second moments — the M0 pooled map.

    Numerically mirrors ``SharedEighRidge``'s cov side (standardize-X
    population sd + 1e-9, center-Y, eigh of the standardized covariance, GCV
    with the #1887 dof cap) but never materializes the pooled train matrix.
    Parity vs SharedEighRidge is asserted at pilot time
    (``_pooled_parity_gate``). Pooled n_train <= d_ambient RAISES — this
    design presumes the ambient regime (n_pooled >> d)."""

    def __init__(
        self,
        *,
        n: int,
        sum_x: torch.Tensor,
        sum_y: torch.Tensor,
        yss: float,
        c_xx: torch.Tensor,
        c_xy: torch.Tensor,
        lambdas: np.ndarray = DEFAULT_LAMBDAS,
        dof_cap: float = GCV_DOF_CAP,
    ) -> None:
        d = int(c_xx.shape[0])
        if n <= d:
            raise RuntimeError(
                f"pooled fit left the ambient regime: n_train={n} <= d={d} — "
                "the pooled M0 design presumes n_pooled >> d."
            )
        if dof_cap is None:
            raise RuntimeError("pure-GCV lambda selection REFUSED (#1887): pass a dof cap.")
        self.n_train, self.d = n, d
        mu_x = sum_x / n
        var = torch.clamp(torch.diagonal(c_xx) / n - mu_x**2, min=0.0)
        sd = var.sqrt() + 1e-9  # population sd (fit_h parity)
        mu_y = sum_y / n
        c_c = c_xx - n * torch.outer(mu_x, mu_x)
        s = c_c / torch.outer(sd, sd)
        w, v = _eigh_robust(s)
        w = torch.clamp(w, min=0.0)
        cross = (c_xy - torch.outer(mu_x, sum_y)) / sd[:, None]  # X_std^T Y_c
        b = v.T @ cross
        e = (b**2).sum(1)
        e = torch.where(w > 1e-12, e / w, torch.zeros_like(e))
        tot = yss - n * float((mu_y**2).sum())
        lam_t = torch.as_tensor(np.asarray(lambdas, dtype=np.float64), device=w.device)
        filt = w[None, :] / (w[None, :] + lam_t[:, None])  # (L, d)
        dof_grid = filt.sum(1)
        rss = tot - (2 * filt - filt**2) @ e  # (L,)
        denom = (n - dof_grid) ** 2
        gcv = torch.where(denom > 1e-12, rss / denom, torch.full_like(rss, float("inf")))
        ok = dof_grid <= dof_cap * n
        if not bool(ok.any()):
            raise RuntimeError(
                f"gcv dof cap {dof_cap}: EVERY lambda exceeds cap*n_train={dof_cap * n:.0f} "
                "(#1887) — widen the grid."
            )
        gcv = torch.where(ok, gcv, torch.full_like(gcv, float("inf")))
        idx = int(gcv.argmin())
        self.best_lambda = float(lambdas[idx])
        self.dof = float(dof_grid[idx])
        self.mu_x, self.sd, self.mu_y = mu_x, sd, mu_y
        self.map = v @ (b / (w + self.best_lambda)[:, None])  # (d, d_out), standardized-x space
        self.global_bias = (mu_y - mu_x).cpu().numpy()  # identity_global baseline bias

    def predict_np(self, x: np.ndarray) -> np.ndarray:
        xt = torch.as_tensor(np.asarray(x, dtype=np.float64), device=self.map.device)
        return (((xt - self.mu_x) / self.sd) @ self.map + self.mu_y).cpu().numpy()

    def info(self) -> dict:
        return {
            "best_lambda": self.best_lambda,
            "dof": self.dof,
            "selector": f"gcv_dof_cap_{GCV_DOF_CAP}",
            "side": "cov-moments",
            "n_train": self.n_train,
            "d_fit": self.d,
        }


def fit_pooled_per_fold(
    mom_by_fold: list[dict], folds_to_run: list[int], k: int
) -> dict[int, PooledMomentRidge]:
    """Train-fold moments = totals - held-out fold (all K folds always
    accumulate, so a fold subset still trains on the full complement)."""
    models: dict[int, PooledMomentRidge] = {}
    for f in folds_to_run:
        t0 = time.time()
        train = {
            "n": sum(mom_by_fold[g]["n"] for g in range(k) if g != f),
            "yss": sum(mom_by_fold[g]["yss"] for g in range(k) if g != f),
        }
        for key in ("sum_x", "sum_y", "c_xx", "c_xy"):
            train[key] = sum(mom_by_fold[g][key] for g in range(k) if g != f)
        models[f] = PooledMomentRidge(**train)
        _log(
            f"[poolspec] pooled fold {f}: n_train={models[f].n_train:,} "
            f"lam={models[f].best_lambda:g} dof={models[f].dof:.0f} "
            f"elapsed={time.time() - t0:.1f}s"
        )
    return models


def _pooled_parity_gate(cell: Cell, fold_map: dict, arm: str, device: str) -> dict:
    """Assert PooledMomentRidge (moment path) reproduces SharedEighRidge (raw
    path, itself fit_h-parity-gated) on a materializable one-cell subset:
    fold-0 rows as eval, the 4-fold complement as train. Exact lambda match +
    rel 1e-6 on predictions."""
    act = load_cell_with_answer(cell)
    k = int(fold_map["k"])
    j = join_cell(act, fold_map["fold_of"], k, arm)
    vec = ARM_VEC_KEY[arm]
    te = j["fold_rows"][0]
    tr = np.concatenate([j["fold_rows"][g] for g in range(1, k)])
    x_tr = act[vec][j["rows"][tr]].astype(np.float64)
    y_tr = act["v_A"][j["rows"][tr]].astype(np.float64)
    x_te = act[vec][j["rows"][te]].astype(np.float64)
    dev = torch.device(device)
    xt = torch.as_tensor(x_tr, device=dev)
    yt = torch.as_tensor(y_tr, device=dev)
    mine = PooledMomentRidge(
        n=int(xt.shape[0]),
        sum_x=xt.sum(0),
        sum_y=yt.sum(0),
        yss=float((yt * yt).sum()),
        c_xx=xt.T @ xt,
        c_xy=xt.T @ yt,
    )
    ref = SharedEighRidge(x_tr, x_te, device=device)
    preds_ref, info_ref = ref.fit_predict(y_tr)
    preds_mine = mine.predict_np(x_te)
    scale = float(np.abs(preds_ref).max()) + 1e-12
    max_rel = float(np.abs(preds_mine - preds_ref).max() / scale)
    if mine.best_lambda != info_ref["best_lambda"] or max_rel > 1e-6:
        raise RuntimeError(
            f"pooled-moments parity FAIL vs SharedEighRidge (arm={arm}): "
            f"lambda {mine.best_lambda} vs {info_ref['best_lambda']}, max_rel={max_rel:.3e}"
        )
    _log(
        f"[poolspec] parity arm={arm}: n_train={mine.n_train} lam={mine.best_lambda:g} "
        f"max_rel={max_rel:.2e} OK"
    )
    return {"n_train": mine.n_train, "best_lambda": mine.best_lambda, "max_rel": max_rel}


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell PCA (center-only, top-k via partial eigh on the cheaper side)


def _psd_eig_tol(w: np.ndarray, n: int, d: int) -> float:
    """Roundoff tolerance for treating a PSD Gram eigenvalue as zero.

    ``xc.T @ xc`` is PSD by construction, so in exact arithmetic every
    eigenvalue is >= 0. ``eigh`` on a near-singular PSD matrix returns
    machine-epsilon-scale NEGATIVES for directions the data does not actually
    span, so a bare ``w < 0`` test is guaranteed to trip on roundoff rather
    than on a real defect (#2054: a shard died on w_min=-2.03e-14). Scale the
    tolerance with the leading eigenvalue and the problem size, the standard
    PSD-clamping convention.
    """
    w_max = float(abs(w[0])) if w.size else 0.0
    return float(np.finfo(w.dtype).eps) * max(n, d) * max(w_max, 1.0)


def _pca_topk(x_tr: np.ndarray, k_max: int) -> tuple[np.ndarray, np.ndarray] | str:
    """Center-only PCA of the cell's train X (mirrors the sibling
    ``_reduced_basis`` convention; the ridge core standardizes the reduced
    features).

    Returns ``(mu, components (d, k_max))``, or a STRING degeneracy reason the
    caller records as a NAMED degeneracy and skips M2 for:

      ``"constant_x"``          max per-feature variance < floor (a fixed
                                prefix render — the chat-form prefix cells).
      ``"rank_deficient_topk"`` the spectrum cannot supply k_max directions
                                that are positive beyond PSD roundoff.

    Both mean the same thing scientifically — the cell cannot support a
    rank-k_max basis — and both are RECORDED, never silently clamped: building
    a basis out of null directions would hand M2 a meaningless correction that
    still produces plausible-looking numbers. The absolute variance floor alone
    only catches EXACTLY-constant X; a near-constant cell clears the floor and
    then runs out of spectrum, which is the case that killed a shard (#2054).

    RAISES only on a genuinely unusable input: n_train <= k_max, or non-finite
    values (NaN/inf from an upstream defect, the one thing a tolerance must not
    absorb).
    """
    n, d = x_tr.shape
    if n <= k_max:
        raise RuntimeError(f"per-cell PCA needs n_train > k_max: n={n}, k_max={k_max}")
    mu = x_tr.mean(axis=0)
    xc = x_tr - mu
    if not np.isfinite(xc).all():
        raise RuntimeError("per-cell PCA: non-finite values in centered X (upstream defect)")
    if float((xc**2).mean(axis=0).max()) < CONSTANT_X_VAR_FLOOR:
        return "constant_x"
    if n >= d:
        c = xc.T @ xc
        w, v = scipy_eigh(c, subset_by_index=(d - k_max, d - 1))
        w, v = w[::-1], v[:, ::-1]  # descending
        if float(w[-1]) <= _psd_eig_tol(w, n, d):
            return "rank_deficient_topk"
    else:
        g = xc @ xc.T
        w, u = scipy_eigh(g, subset_by_index=(n - k_max, n - 1))
        w, u = w[::-1], u[:, ::-1]
        if float(w[-1]) <= _psd_eig_tol(w, n, d):
            return "rank_deficient_topk"
        v = (xc.T @ u) / np.sqrt(w)[None, :]
    return mu, np.ascontiguousarray(v)


# ─────────────────────────────────────────────────────────────────────────────
# Banked ceilings (the parent's committed per-cell within-cell fits — REUSED)


def load_ceilings(
    cells: list[Cell],
    arms: list[str],
    folds_to_run: list[int],
    k: int,
    ceilings_dir: str | None,
    allow_missing: bool,
) -> dict:
    """Fetch ``issue2054_lattice/fits/{cell}.json`` per cell (HF data repo, or
    a local ``--ceilings-dir``); ceiling = banked ambient held-out R^2 on the
    SAME scored folds (pooled mean when all folds run, banked per-fold values
    for a subset). Missing file: RAISE unless --allow-missing-ceiling."""
    from explore_persona_space.orchestrate.hub import retry_transient
    from huggingface_hub import hf_hub_download

    full = set(folds_to_run) == set(range(k))
    out: dict[str, dict] = {}
    for cell in cells:
        if ceilings_dir:
            p = Path(ceilings_dir) / f"{cell.key}.json"
            found = p.is_file()
        else:
            try:
                p = Path(
                    retry_transient(
                        lambda cell=cell: hf_hub_download(
                            HF_DATA_REPO,
                            f"{CEILING_HF_PREFIX}/{cell.key}.json",
                            repo_type="dataset",
                        ),
                        what=f"ceiling {cell.key}",
                    )
                )
                found = True
            except Exception as e:  # noqa: BLE001 — re-raised unless explicitly allowed
                if not (allow_missing and "404" in str(e)):
                    raise
                found = False
        if not found:
            if not allow_missing:
                raise FileNotFoundError(f"banked ceiling missing for {cell.key}")
            out[cell.key] = {"missing": True}
            continue
        text = p.read_text(encoding="utf-8")
        d = json.loads(text)
        rec: dict = {"missing": False, "sha256": hashlib.sha256(text.encode()).hexdigest()}
        for arm in arms:
            rep = d["arm_reports"][arm]
            if rep["status"] != "ok":
                raise RuntimeError(f"banked ceiling {cell.key} arm={arm} status={rep['status']}")
            per_fold = {int(pf["fold"]): float(pf["r2_ambient"]) for pf in rep["per_fold"]}
            ceiling = (
                float(rep["pooled"]["r2_ambient_mean"])
                if full
                else float(np.mean([per_fold[f] for f in folds_to_run]))
            )
            null_p95 = float(rep["pooled"]["null_r2_pooled_p95"])
            rec[arm] = {
                "ceiling_r2": ceiling,
                "banked_pooled_r2_ambient_mean": float(rep["pooled"]["r2_ambient_mean"]),
                "banked_null_r2_pooled_p95": null_p95,
                "scored_folds": list(folds_to_run),
                "usable": bool(ceiling > max(CEILING_FLOOR, null_p95)),
            }
        out[cell.key] = rec
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell unit


def _knn_block(preds: np.ndarray, true: np.ndarray) -> dict:
    return {
        metric: knn_retrieval(preds, true, ks=(1, 5, 10), metric=metric)
        for metric in ("euclidean", "cosine")
    }


def run_cell(
    cell: Cell,
    arm: str,
    fold_map: dict,
    pooled_models: dict[int, PooledMomentRidge],
    ceiling: dict,
    args: argparse.Namespace,
    folds_to_run: list[int],
    out_path: Path,
    rows_path: Path,
    fingerprint: str,
) -> None:
    t_unit = time.time()
    k = int(fold_map["k"])
    vec = ARM_VEC_KEY[arm]
    ranks = list(args.ranks)
    k_max = max(ranks)
    names = _model_names(ranks)
    act = load_cell_with_answer(cell)
    j = join_cell(act, fold_map["fold_of"], k, arm)
    x_all = act[vec][j["rows"]].astype(np.float64)
    y_all = act["v_A"][j["rows"]].astype(np.float64)
    del act
    n_join = j["n_join"]
    e2 = {name: np.full(n_join, np.nan) for name in names}
    scored = np.zeros(n_join, dtype=bool)
    fold_records = []
    null_units = []

    for f in folds_to_run:
        t0 = time.time()
        te = j["fold_rows"][f]
        tr = np.concatenate([j["fold_rows"][g] for g in range(k) if g != f])
        x_tr, y_tr = x_all[tr], y_all[tr]
        x_te, y_te = x_all[te], y_all[te]
        n_tr = int(x_tr.shape[0])
        m0 = pooled_models[f]

        preds: dict[str, np.ndarray] = {}
        preds["m0"] = m0.predict_np(x_te)
        b_cf = y_tr.mean(axis=0) - m0.predict_np(x_tr.mean(axis=0, keepdims=True))[0]
        preds["m1"] = preds["m0"] + b_cf
        preds["identity_cell"] = identity_bias_predict(x_tr, y_tr, x_te)
        preds["identity_global"] = x_te + m0.global_bias

        # M2: per-cell low-rank slope correction on the M1 residuals.
        yhat1_tr = m0.predict_np(x_tr) + b_cf
        r_tr = y_tr - yhat1_tr
        r_te = y_te - preds["m1"]
        pca = _pca_topk(x_tr, k_max)
        # A string return is a NAMED degeneracy (constant_x / rank_deficient_topk),
        # not a failure: M2 is skipped and M1 stands in, and the reason is recorded
        # per fold so a cell resting on fewer M2 ranks than its siblings is visible.
        degenerate_reason = pca if isinstance(pca, str) else None
        m2_skipped = degenerate_reason is not None
        degenerate_constant_x = degenerate_reason == "constant_x"
        m2_recs: dict[str, dict] = {}
        if m2_skipped:
            _log(
                f"{cell.key} arm={arm} fold={f} M2 SKIPPED ({degenerate_reason}) "
                f"— cannot supply a rank-{k_max} basis; M1 substituted"
            )
            for r in ranks:
                preds[f"m2_k{r}"] = preds["m1"]
                m2_recs[f"m2_k{r}"] = {"skipped": degenerate_reason}
        else:
            mu_p, comps = pca
            xr_tr = (x_tr - mu_p) @ comps
            xr_te = (x_te - mu_p) @ comps
            for r in ranks:
                core = SharedEighRidge(xr_tr[:, :r], xr_te[:, :r], device=args.device)
                corr_te, info = core.fit_predict(r_tr)
                preds[f"m2_k{r}"] = preds["m1"] + corr_te
                resid_fit = reconstruction_metrics(corr_te, r_te)
                seed = _seed(cell.key, arm, f, r, NULL_SEED_BASE)
                null = core.null_r2(
                    r_tr, r_te, n_draws=args.null_draws, seed=seed, chunk=args.null_chunk
                )
                stat = resid_fit["r2"]
                mu_n, sd_n = float(null.mean()), float(null.std(ddof=1))
                escalated = bool(abs(stat - mu_n) < args.null_escalate_sd * sd_n)
                if escalated and args.null_max_draws > len(null):
                    extra = core.null_r2(
                        r_tr,
                        r_te,
                        n_draws=args.null_max_draws - len(null),
                        seed=seed + 1,
                        chunk=args.null_chunk,
                    )
                    null = np.concatenate([null, extra])
                p_val = float((1 + (null >= stat).sum()) / (1 + len(null)))
                m2_recs[f"m2_k{r}"] = {
                    "rank": r,
                    "residual_fit_r2": stat,
                    "ridge_info": info,
                    "null": {
                        "kind": "shuffled_pair_fit_side",
                        "n_draws": int(len(null)),
                        "seed": seed,
                        "mean": mu_n,
                        "sd": sd_n,
                        "escalated": escalated,
                        "escalate_sd": args.null_escalate_sd,
                        "p_value_residual_fit": p_val,
                    },
                }
                null_units.append(
                    {"fold": f, "rank": r, "n_draws": int(len(null)), "escalated": escalated}
                )

        for name in names:
            e2[name][te] = ((preds[name] - y_te) ** 2).sum(axis=1)
        scored[te] = True
        metrics = {name: reconstruction_metrics(preds[name], y_te) for name in names}
        knn = {
            name: _knn_block(preds[name], y_te)
            for name in ("m0", "m1", f"m2_k{k_max}", "identity_cell")
        }
        rec = {
            "fold": f,
            "n_pooled_train": m0.n_train,
            "n_cell_train": n_tr,
            "n_test": int(len(te)),
            "d_ambient": D_AMBIENT,
            "regime_cell": "ambient" if n_tr > D_AMBIENT else "reduced_basis_descriptive",
            # `degenerate_constant_x` keeps its literal meaning (exactly-constant
            # X); `m2_skipped` is the "no M2 this fold" flag any consumer should
            # read, and `degenerate_reason` says which degeneracy fired.
            "degenerate_constant_x": degenerate_constant_x,
            "m2_skipped": m2_skipped,
            "degenerate_reason": degenerate_reason,
            "xy_max_abs_diff": float(np.abs(x_te - y_te).max()),
            "pooled_info": m0.info(),
            "m1_bias_norm": float(np.linalg.norm(b_cf)),
            "metrics": metrics,
            "m2": m2_recs,
            "knn": knn,
            "wall_s": round(time.time() - t0, 1),
        }
        fold_records.append(rec)
        _log(
            f"[poolspec] {cell.key} arm={arm} fold={f} "
            f"m0={metrics['m0']['r2']:+.4f} m1={metrics['m1']['r2']:+.4f} "
            f"m2k{k_max}={metrics[f'm2_k{k_max}']['r2']:+.4f} "
            f"idcell={metrics['identity_cell']['r2']:+.4f} elapsed={rec['wall_s']}s"
        )

    # Sidecar rows for the conversation-level bootstrap (fixed full-scored-mean
    # SS_tot convention — per-draw R^2 becomes pure GEMMs over e2/s2).
    sidx = np.flatnonzero(scored)
    y_s = y_all[sidx]
    s2 = ((y_s - y_s.mean(axis=0)) ** 2).sum(axis=1)
    conv_scored = np.asarray([j["order"][i] for i in sidx])
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = rows_path.with_name(rows_path.stem + ".tmp.npz")  # np.savez suffix trap
    np.savez(
        tmp_npz,
        conv_id=conv_scored,
        e2=np.stack([e2[name][sidx] for name in names], axis=1),
        s2=s2,
        model_names=np.asarray(names),
    )
    os.replace(tmp_npz, rows_path)

    mean_r2 = {
        name: float(np.mean([fr["metrics"][name]["r2"] for fr in fold_records])) for name in names
    }
    ceil_rec = ceiling if ceiling.get("missing") else ceiling[arm]
    fractions = {}
    if not ceiling.get("missing") and ceil_rec["usable"]:
        fractions = {name: mean_r2[name] / ceil_rec["ceiling_r2"] for name in names}
    pooled_summary = {
        "r2_mean_over_folds": mean_r2,
        "increment_m1_minus_m0": mean_r2["m1"] - mean_r2["m0"],
        "increments_m2_minus_m1": {f"k{r}": mean_r2[f"m2_k{r}"] - mean_r2["m1"] for r in ranks},
        "ceiling": ceil_rec,
        "fraction_of_ceiling": fractions or None,
        "null_units": null_units,
    }
    payload = {
        "metadata": {
            **as_metadata_dict(git_provenance(_REPO)),
            "script_version": SCRIPT_VERSION,
            "argv": sys.argv,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "fold_map": {
                "source": fold_map["_source"],
                "sha256": fold_map["_sha256"],
                "k": int(fold_map["k"]),
                "seed": int(fold_map["seed"]),
                "n_conv": len(fold_map["fold_of"]),
            },
            "bootstrap_sstot_convention": "fixed_full_scored_mean",
            "pilot": bool(args.pilot),
        },
        "cell": cell.key,
        "arm": arm,
        "config_fingerprint": fingerprint,
        "n_join": n_join,
        "folds_scored": list(folds_to_run),
        "per_fold": fold_records,
        "pooled": pooled_summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, out_path)
    _log(
        f"[poolspec] unit {cell.key}__{arm} CHECKPOINTED -> {out_path} "
        f"(wall={time.time() - t_unit:.0f}s)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate: coupled conversation-level bootstrap across cells


def aggregate(
    cells: list[Cell],
    arms: list[str],
    ranks: list[int],
    out_root: Path,
    args: argparse.Namespace,
    fold_map: dict,
) -> bool:
    names = _model_names(ranks)
    per_arm: dict[str, dict] = {}
    for arm in arms:
        cell_rows = {}
        for cell in cells:
            jp = out_root / "percell" / f"{cell.key}__{arm}.json"
            rp = out_root / "percell" / f"{cell.key}__{arm}.rows.npz"
            if not (jp.is_file() and rp.is_file()):
                _log(f"[poolspec] aggregate SKIP: {jp.name} not complete yet")
                return False
            z = np.load(rp, allow_pickle=False)
            if [str(m) for m in z["model_names"]] != names:
                raise RuntimeError(f"model-name mismatch in {rp} — stale sidecar; fresh out-root.")
            cell_rows[cell.key] = {
                "conv_id": [str(c) for c in z["conv_id"]],
                "e2": np.asarray(z["e2"], dtype=np.float64),
                "s2": np.asarray(z["s2"], dtype=np.float64),
                "json": json.loads(jp.read_text(encoding="utf-8")),
            }
        union = sorted({c for rec in cell_rows.values() for c in rec["conv_id"]})
        u_index = {c: i for i, c in enumerate(union)}
        n_u = len(union)
        rng = np.random.default_rng(_seed("bootstrap", arm, NULL_SEED_BASE))
        b = args.bootstrap_draws
        counts = np.zeros((b, n_u), dtype=np.float64)
        for bi in range(b):
            counts[bi] = np.bincount(rng.integers(0, n_u, size=n_u), minlength=n_u)
        cell_summ = {}
        r2_draws_by_cell = []
        for key, rec in cell_rows.items():
            cols = np.asarray([u_index[c] for c in rec["conv_id"]], dtype=np.int64)
            counts_c = counts[:, cols]
            den = counts_c @ rec["s2"]
            if not np.all(den > 0):
                raise RuntimeError(f"bootstrap draw with zero SS_tot for cell {key} (arm={arm})")
            r2_draws = 1.0 - (counts_c @ rec["e2"]) / den[:, None]  # (B, n_models)
            r2_point = 1.0 - rec["e2"].sum(axis=0) / rec["s2"].sum()
            r2_draws_by_cell.append(r2_draws)
            im = {n: i for i, n in enumerate(names)}

            def ci(v: np.ndarray) -> dict:
                return {
                    "lo": float(np.quantile(v, 0.025)),
                    "hi": float(np.quantile(v, 0.975)),
                    "mean": float(v.mean()),
                }

            d10 = r2_draws[:, im["m1"]] - r2_draws[:, im["m0"]]
            cell_summ[key] = {
                "r2_point_fixed_mean": {n: float(r2_point[im[n]]) for n in names},
                "r2_ci": {n: ci(r2_draws[:, im[n]]) for n in names},
                "increment_m1_minus_m0": {
                    "point": float(r2_point[im["m1"]] - r2_point[im["m0"]]),
                    **ci(d10),
                },
                "increments_m2_minus_m1": {
                    f"k{r}": {
                        "point": float(r2_point[im[f"m2_k{r}"]] - r2_point[im["m1"]]),
                        **ci(r2_draws[:, im[f"m2_k{r}"]] - r2_draws[:, im["m1"]]),
                    }
                    for r in ranks
                },
                "fraction_of_ceiling": rec["json"]["pooled"]["fraction_of_ceiling"],
                "ceiling": rec["json"]["pooled"]["ceiling"],
            }
        stack = np.stack(r2_draws_by_cell)  # (n_cells, B, n_models)
        mean_draws = stack.mean(axis=0)  # (B, n_models)
        im = {n: i for i, n in enumerate(names)}

        def ci(v: np.ndarray) -> dict:
            return {
                "lo": float(np.quantile(v, 0.025)),
                "hi": float(np.quantile(v, 0.975)),
                "mean": float(v.mean()),
            }

        agg = {
            "n_cells": len(cell_rows),
            "r2_cell_mean_ci": {n: ci(mean_draws[:, im[n]]) for n in names},
            "increment_m1_minus_m0": ci(mean_draws[:, im["m1"]] - mean_draws[:, im["m0"]]),
            "increments_m2_minus_m1": {
                f"k{r}": ci(mean_draws[:, im[f"m2_k{r}"]] - mean_draws[:, im["m1"]]) for r in ranks
            },
        }
        null_realized = [
            {"cell": key, **u}
            for key, rec in cell_rows.items()
            for u in rec["json"]["pooled"]["null_units"]
        ]
        per_arm[arm] = {
            "aggregate": agg,
            "per_cell": cell_summ,
            "null_draws_realized": {
                "total_draws": int(sum(u["n_draws"] for u in null_realized)),
                "n_units": len(null_realized),
                "n_escalated": int(sum(u["escalated"] for u in null_realized)),
                "units": null_realized,
            },
        }
    payload = {
        "metadata": {
            **as_metadata_dict(git_provenance(_REPO)),
            "script_version": SCRIPT_VERSION,
            "argv": sys.argv,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "bootstrap_draws": args.bootstrap_draws,
            "bootstrap_sstot_convention": "fixed_full_scored_mean",
            "bootstrap_coupling": "shared conversation resample applied to every cell",
            "fold_map_sha256": fold_map["_sha256"],
            "ranks": ranks,
            "pilot": bool(args.pilot),
        },
        "arms": per_arm,
    }
    out_path = out_root / "aggregate.json"
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, out_path)
    _log(f"[poolspec] aggregate -> {out_path}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main


def _fingerprint(
    args: argparse.Namespace, fold_map: dict, arm: str, cell_keys: list[str], ceilings: dict
) -> str:
    blob = json.dumps(
        {
            "script_version": SCRIPT_VERSION,
            "lambdas": [float(x) for x in DEFAULT_LAMBDAS],
            "dof_cap": GCV_DOF_CAP,
            "ranks": list(args.ranks),
            "fold_map_sha": fold_map["_sha256"],
            "folds": args.folds or list(range(int(fold_map["k"]))),
            "null_draws": args.null_draws,
            "null_escalate_sd": args.null_escalate_sd,
            "null_max_draws": args.null_max_draws,
            "arm": arm,
            "cells": sorted(cell_keys),
            "join_floors": [MIN_JOIN_ABS, MIN_JOIN_FRAC],
            "ceiling_shas": sorted(
                str(c.get("sha256")) for c in ceilings.values() if not c.get("missing")
            ),
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument("--activations-dir", type=Path, required=True)
    ap.add_argument(
        "--out-root", type=Path, default=_REPO / "eval_results/issue_2054/inline_pool_specialize"
    )
    ap.add_argument("--fold-map-ref", default="origin/issue-2054")
    ap.add_argument("--fold-map-file", default=None, help="direct path override (floors enforced)")
    ap.add_argument("--pilot", action="store_true", help="fold 0 only + parity gate; pilot/ root")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--ranks", nargs="*", type=int, default=list(DEFAULT_RANKS))
    ap.add_argument("--null-draws", type=int, default=20)
    ap.add_argument("--null-escalate-sd", type=float, default=3.0)
    ap.add_argument("--null-max-draws", type=int, default=100)
    ap.add_argument("--null-chunk", type=int, default=4)
    ap.add_argument("--bootstrap-draws", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--ceilings-dir", default=None, help="local dir of banked fit JSONs (else HF)")
    ap.add_argument("--allow-missing-ceiling", action="store_true")
    ap.add_argument("--skip-parity", action="store_true", help="skip the pilot parity gate")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _log("[poolspec] import-check OK")
        return 0

    t_start = time.time()
    fold_map = load_fold_map(args.fold_map_file, args.fold_map_ref)
    k = int(fold_map["k"])
    _log(
        f"[poolspec] fold map {fold_map['_source']} k={k} seed={fold_map['seed']} "
        f"n_conv={len(fold_map['fold_of']):,} sha={fold_map['_sha256'][:12]}"
    )
    cells = discover_cells(args.activations_dir)
    folds_to_run = args.folds if args.folds else ([0] if args.pilot else list(range(k)))
    if any(f < 0 or f >= k for f in folds_to_run):
        raise ValueError(f"--folds out of range for k={k}: {folds_to_run}")
    out_root = args.out_root / "pilot" if args.pilot else args.out_root
    _log(f"[poolspec] {len(cells)} cells, arms={args.arms}, folds={folds_to_run} -> {out_root}")

    ceilings = load_ceilings(
        cells, args.arms, folds_to_run, k, args.ceilings_dir, args.allow_missing_ceiling
    )

    if args.pilot and not args.skip_parity:
        _pooled_parity_gate(cells[0], fold_map, "context", args.device)

    acc = accumulate_pooled_moments(cells, fold_map["fold_of"], k, args.arms, args.device)

    all_keys = [c.key for c in cells]
    n_done = 0
    for arm in args.arms:
        pooled_models = fit_pooled_per_fold(acc["mom"][arm], folds_to_run, k)
        pooled_payload = {
            "metadata": {
                **as_metadata_dict(git_provenance(_REPO)),
                "script_version": SCRIPT_VERSION,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "arm": arm,
            "per_fold": {str(f): m.info() for f, m in pooled_models.items()},
            "cells_pooled": all_keys,
            "n_join_per_cell": acc["joins"][arm],
        }
        pooled_path = out_root / f"pooled_{arm}.json"
        pooled_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = pooled_path.with_name(pooled_path.name + ".tmp")
        tmp.write_text(json.dumps(pooled_payload, indent=1), encoding="utf-8")
        os.replace(tmp, pooled_path)

        fp = _fingerprint(args, fold_map, arm, all_keys, ceilings)
        shard_cells = [c for i, c in enumerate(cells) if i % args.num_shards == args.shard]
        for cell in shard_cells:
            out_path = out_root / "percell" / f"{cell.key}__{arm}.json"
            rows_path = out_root / "percell" / f"{cell.key}__{arm}.rows.npz"
            if out_path.is_file():
                prior = json.loads(out_path.read_text(encoding="utf-8"))
                if prior.get("config_fingerprint") == fp and rows_path.is_file():
                    n_done += 1
                    _log(f"[poolspec] unit {cell.key}__{arm} already done — resume skip")
                    continue
                raise RuntimeError(
                    f"existing {out_path} fingerprint {prior.get('config_fingerprint')} != {fp} "
                    "— config changed; move the stale file or use a fresh --out-root."
                )
            run_cell(
                cell,
                arm,
                fold_map,
                pooled_models,
                ceilings[cell.key],
                args,
                folds_to_run,
                out_path,
                rows_path,
                fp,
            )
            n_done += 1
        del pooled_models

    aggregate(cells, args.arms, list(args.ranks), out_root, args, fold_map)

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    _log(
        f"[poolspec] done units={n_done} wall={time.time() - t_start:.0f}s "
        f"peak_rss_gib={peak_rss_gib:.2f} torch_threads={torch.get_num_threads()}"
    )
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit before C-extension teardown (code-style.md)
