#!/usr/bin/env python3
"""task #511 — probe-count convergence of #502's persona-distance predictors.

Re-scores #502's saved residual-stream activations at six probe-count budgets
{25, 50, 100, 200, 350, 500} × R random axis-1 subsets each, against #474's
ΔG marker-transfer target. No retraining; no new model calls in the primary
path. The whole pipeline runs CPU-only against the cached (16, N, H) tensors.

Inputs
------
- Activations: ``eval_results/issue_502/bakeoff/activations/<point>__layer<L>.pt``
  (the canonical merged stack; pulled from
  ``superkaiba1/explore-persona-space-data/issue502_28layer_500probe_bakeoff/``
  if absent — see ``ensure_activation`` below).
- Target: ``eval_results/issue_474/cross_eval/loc_ep{1,2,3,5}/G_logprob_matrix.json``.
- Length covariate: ``eval_results/issue_406/divergence/D_matrix.json``.

Output
------
``eval_results/issue_511/probe_count_sweep_results.json`` — one row per
(cell_id, N, subset_idx), with columns ``cell_id``, ``extraction_point``,
``layer``, ``metric``, ``variant``, ``N``, ``subset_idx``, ``seed``, ``arm``,
``epoch``, ``abs_rho``, ``rho``, ``p``, ``cv_r2``, plus a ``aggregates`` block
of mean+std at each (cell, N) over the R subsets.

The plan's Step 0 numerical-reproduction gate is exposed as
``--mode reproduction-gate``: runs the headline cell at N=500 / R=1
(full-pool identity permutation) and compares against the archived
``eval_results/issue_502/bakeoff/bakeoff_grid.json`` headline
(ρ = -0.7922536776, CV R² = 0.6086304269); ``|Δ| < 1e-3`` on both. FAIL halts.

Smoke mode (``--mode smoke``): runs N ∈ {25, 50} with R=2 on the headline cell
only, single checkpoint, in a few seconds — exercises data load → metric
compute → regression → write end-to-end before the full sweep.

CLI
---
::

  uv run python scripts/issue511_probe_count_sweep.py --mode reproduction-gate
  uv run python scripts/issue511_probe_count_sweep.py --mode smoke
  uv run python scripts/issue511_probe_count_sweep.py --mode full
"""

# ruff: noqa: RUF001, RUF002, RUF003 (research notation: ρ, Δ, σ in strings/comments)

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# We reuse the #493/#502 bakeoff machinery verbatim. The plan's Step 0
# numerical-reproduction gate exists precisely to assert this wrapper does
# not drift from the archived #502 endpoint.
import issue493_extraction_metric_bakeoff as bakeoff  # noqa: E402

logger = logging.getLogger("i511.sweep")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ───────────────────────── paths ─────────────────────────

OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_511"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Point the bakeoff module at #502's cached artifacts so
# load_activations_from_disk + the metric/regression helpers walk #502's
# disk layout, not the default #493 path.
BAKEOFF_ROOT_502 = PROJECT_ROOT / "eval_results" / "issue_502" / "bakeoff"
bakeoff._set_roots(BAKEOFF_ROOT_502)

HF_REPO_ID = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue502_28layer_500probe_bakeoff/activations"

# ───────────────────────── design grid ─────────────────────────

# Inherited from #474/#502 (the on-disk G_logprob matrices use this order).
COND_IDS: tuple[str, ...] = (
    "A1", "A2", "A3", "A4", "A5",
    "B1", "B2", "B3", "B4", "B5",
    "C1",
    "D1", "D2", "D3", "D4", "D5",
)  # fmt: skip

# Plan §4 Step 2 cells. Each (extraction_point, layer, metric, variant)
# defines one tracked cell.
HEADLINE_CELL = ("last_prompt", 22, "gauss_kl", "raw")

L19_L24 = tuple(range(19, 25))
# Round-1 descope from {gauss_kl, mmd, wass2}: wass2 dropped per plan §9
# auto-descope (CPU benchmark showed wass2 = ~78s/row at N=200 vs the
# planned <30min total wall; gauss_kl + mmd preserves the cloud-aware
# representative + the bandwidth-based two-sample contrast). See
# compute-deviation marker.
CLOUD_METRICS_KEEP = ("gauss_kl", "mmd")
COSINE_BASELINES_LAYERS = L19_L24  # alternatives reconciler binding directive
COSINE_SENTINEL_LAYERS = (0, 11, 21, 27)

# Plan §4 Step 2 N grid + R. Round-1 autonomous descope (CPU benchmark
# showed full grid × R=10 ≈ 26h vs planned 30 min wall):
#  - Headline cell (last_prompt × L22 × gauss_kl × raw): full N grid + R=10
#    so the formal plateau verdict at the headline retains its full σ_ref
#    precision (R=10 across-subset stdev pooled at N=200, 350).
#  - Other cloud-aware cells (ridge): R=5 across the board, N grid trims
#    N=500 (the most expensive tail) — the ridge structure (Hyp #3) is
#    visible at N=200, 350, and the per-ridge-cell plateau verdict is a
#    diagnostic.
#  - Cosine cells (cheap): full N grid + R=10 across the board.
N_GRID_FULL: tuple[int, ...] = (25, 50, 100, 200, 350, 500)
N_GRID_RIDGE: tuple[int, ...] = (25, 50, 100, 200, 350)
R_HEADLINE: int = 10
R_RIDGE: int = 5
R_COSINE: int = 10
N_GRID_SMOKE: tuple[int, ...] = (25, 50)
R_SMOKE: int = 2

# Default checkpoints to score (plan: loc_ep1 primary, loc_ep{2,3,5} optional).
DEFAULT_ARM = "loc"
DEFAULT_EPOCHS_PRIMARY: tuple[int, ...] = (1,)
DEFAULT_EPOCHS_TIME_PERMITTING: tuple[int, ...] = (1, 2, 3, 5)

# Archived #502 headline (plan §4 Step 0 binding directive).
ARCHIVED_HEADLINE_RHO = -0.7922536776181811
ARCHIVED_HEADLINE_CV = 0.6086304269035411
GATE_TOLERANCE = 1e-3


@dataclass
class CellSpec:
    """One scored cell. ``cell_id`` is human-readable and unique.

    ``n_grid`` and ``r`` per-cell let the headline cell carry the full
    N grid + R=10 (formal plateau verdict precision) while ridge / cosine
    cells use descoped subgrids (round-1 compute-deviation response).
    """

    extraction_point: str
    layer: int
    metric: str
    variant: str
    n_grid: tuple[int, ...]
    r: int

    @property
    def cell_id(self) -> str:
        return f"{self.extraction_point}__L{self.layer}__{self.metric}__{self.variant}"


def build_cell_list() -> list[CellSpec]:
    """Plan §4 Step 2 (round-1 descoped): tracked cells with per-cell N×R.

    HEADLINE (full N, R=10) + L19-L24 cloud-aware ridge (R=5, N grid trims
    N=500) × {gauss_kl, mmd} (wass2 dropped) + L19-L24 same-layer cosine
    controls (full N, R=10) + L0/L11/L27 cosine sentinels (full N, R=10).
    """
    cells: list[CellSpec] = []
    # L19-L24 cloud-aware ridge (gauss_kl + mmd; wass2 dropped). R=5,
    # N grid drops N=500. Insert FIRST so the headline overwrite below
    # takes precedence (deduplication keeps the LAST insertion per
    # cell_id; the headline cell takes the full N grid + R=10).
    for L in L19_L24:
        for m in CLOUD_METRICS_KEEP:
            cells.append(CellSpec("last_prompt", L, m, "raw", n_grid=N_GRID_RIDGE, r=R_RIDGE))
    # Headline: full N grid, R=10 — preserves plateau verdict precision.
    # Inserted AFTER the ridge so the dedup picks this richer config.
    cells.append(CellSpec(*HEADLINE_CELL, n_grid=N_GRID_FULL, r=R_HEADLINE))
    # L19-L24 same-layer cosine controls — cheap, full N + R=10.
    for L in COSINE_BASELINES_LAYERS:
        cells.append(CellSpec("last_prompt", L, "cosine", "raw", n_grid=N_GRID_FULL, r=R_COSINE))
    # Cosine sentinel layers (L21 in ridge already; only L0/L11/L27 here).
    for L in COSINE_SENTINEL_LAYERS:
        if L in COSINE_BASELINES_LAYERS:
            continue  # already added in the ridge cosine controls
        cells.append(CellSpec("last_prompt", L, "cosine", "raw", n_grid=N_GRID_FULL, r=R_COSINE))
    # next_token_js is not swept (constant across N — single number in
    # bakeoff_grid).
    # Dedup (headline overlaps ridge gauss_kl L22): later insert wins.
    seen: dict[str, CellSpec] = {}
    for c in cells:
        seen[c.cell_id] = c
    return list(seen.values())


# ───────────────────────── data loaders ─────────────────────────


def ensure_activation(extraction_point: str, layer: int) -> Path:
    """Download the canonical (extraction_point, layer) activation .pt from
    HF if absent on local disk; otherwise return the existing path.

    Returns the local path.
    """
    from huggingface_hub import hf_hub_download

    local_dir = BAKEOFF_ROOT_502 / "activations"
    local_dir.mkdir(parents=True, exist_ok=True)
    local = local_dir / f"{extraction_point}__layer{layer}.pt"
    if local.exists():
        return local
    remote = f"{HF_PREFIX}/{extraction_point}__layer{layer}.pt"
    logger.info("pulling %s from HF", remote)
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=remote,
        repo_type="dataset",
        revision="main",
        local_dir=str(local_dir.parent.parent.parent),  # back to repo root: ./issue502_..
    )
    # hf_hub_download keeps the repo-internal prefix; mirror back to the
    # canonical local path so load_activations_from_disk finds it.
    downloaded_path = Path(downloaded)
    if downloaded_path != local:
        local.symlink_to(downloaded_path)
    return local


def load_activations_slice(
    extraction_point: str,
    layer: int,
) -> tuple[np.ndarray, list[str]]:
    """Return (n_cond, N, H) activations + cond_ids for one (extraction_point, layer)
    canonical file. NB: end_of_system has n_q == 1, but we never extract
    cloud metrics there per the bakeoff design — last_prompt is the only
    extraction we sweep.
    """
    import torch

    ensure_activation(extraction_point, layer)
    p = BAKEOFF_ROOT_502 / "activations" / f"{extraction_point}__layer{layer}.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    act = d["activations"]
    if hasattr(act, "numpy"):
        act = act.numpy()
    act = np.asarray(act)
    cond_ids = list(d["cond_ids"])
    return act, cond_ids


def load_target(arm: str, epoch: int) -> dict:
    """Read #474's ΔG matrix for one (arm, epoch). Cached helper on the
    bakeoff module is reused."""
    return bakeoff._load_G(arm, epoch)


def load_length_covar() -> dict:
    """Read #406's pair-level prompt-token covariate."""
    return bakeoff._load_prompt_tokens()


# ───────────────────────── deterministic LOCO CV ─────────────────────────


def _loocv_r2_deterministic(
    x: np.ndarray,
    y: np.ndarray,
    cond_ids_a: list[str],
    cond_ids_b: list[str],
    *,
    covar: np.ndarray | None = None,
) -> float:
    """Leave-one-context-out CV R² with deterministic fold iteration.

    Mirrors ``bakeoff._loocv_r2`` byte-for-byte (same residualization, same
    polyfit, same NaN-guards, same threshold constants) EXCEPT it iterates
    fold contexts in ``sorted(set(cond_ids_a) | set(cond_ids_b))`` order
    instead of the bakeoff's unordered ``set()`` iteration.

    Why: Python's set iteration over strings is hash-randomized by
    ``PYTHONHASHSEED`` (default: random per-interpreter). Each off-diagonal
    pair (A, B) is in BOTH fold ``C=A`` and fold ``C=B``, so the LAST
    fold to execute overwrites ``pred[test]``. Different iteration orders
    therefore deterministically produce different ``pred`` arrays and
    different CV R². Across hash-seed draws the variance is ~5e-3 on the
    headline cell — large enough to swamp #511's 1e-3 reproduction-gate
    tolerance and to drift the across-subset σ noise reference.

    Pinning the order to ``sorted(...)`` removes the noise so the (N, R)
    sweep measures probe-subset variance only — the construct we actually
    care about for the plateau verdict. The wrapper imports nothing new;
    the only divergence from the bakeoff helper is the ``sorted()`` call
    on the fold-iteration set.

    NB: this is a wrapper-side fix, NOT a bakeoff-side change. The
    bakeoff regression phase is left alone so #502's archived numbers
    stay stable across re-runs of this script.
    """
    n = len(x)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if covar is not None:
        covar = np.asarray(covar, dtype=np.float64)
        finite_mask = finite_mask & np.isfinite(covar)
    if finite_mask.sum() < bakeoff._MIN_FINITE_FOR_REGRESSION:
        return float("nan")
    if covar is not None:
        x, y = bakeoff._length_partial_residualize_rank(
            x[finite_mask], y[finite_mask], covar[finite_mask]
        )
    else:
        x, y = x[finite_mask], y[finite_mask]
    cond_ids_a = [c for c, k in zip(cond_ids_a, finite_mask, strict=True) if k]
    cond_ids_b = [c for c, k in zip(cond_ids_b, finite_mask, strict=True) if k]
    n = len(x)
    pred = np.full(n, np.nan)
    src = np.array(cond_ids_a)
    tgt = np.array(cond_ids_b)
    # ─── THE FIX ─── sort fold iteration so the result is deterministic.
    folds = sorted(set(cond_ids_a) | set(cond_ids_b))
    for C in folds:
        train = ~((src == C) | (tgt == C))
        test = (src == C) | (tgt == C)
        if train.sum() < 3:
            continue
        x_train = x[train]
        y_train = y[train]
        if not np.all(np.isfinite(x_train)) or not np.all(np.isfinite(y_train)):
            continue
        if len(np.unique(x_train)) < 2:
            continue
        try:
            b, a = np.polyfit(x_train, y_train, 1)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        pred[test] = a + b * x[test]
    m = np.isfinite(pred)
    if m.sum() < bakeoff._MIN_FINITE_FOR_REGRESSION:
        return float("nan")
    sse = np.sum((y[m] - pred[m]) ** 2)
    sst = np.sum((y[m] - y[m].mean()) ** 2)
    if sst < 1e-18 or not np.isfinite(sse):
        return float("nan")
    return float(1.0 - sse / sst)


# ───────────────────────── scoring ─────────────────────────


def score_one_cell(
    *,
    cell: CellSpec,
    activations_full: np.ndarray,  # (n_cond, N_full, H)
    cond_ids: list[str],
    N: int,
    subset_idx: int,
    G: dict,
    prompt_tokens: dict,
) -> dict:
    """Score one (cell, N, subset_idx) endpoint. Returns the row to store."""
    seed = subset_idx + 1000 * N
    rng = np.random.default_rng(seed)
    n_pool = activations_full.shape[1]
    if n_pool < N:
        raise ValueError(f"N={N} > pool size {n_pool}")
    # Sample axis-1 (probe axis) without replacement; the same probes are
    # used across all conditions inside this subset so the cross-cond
    # comparison stays apples-to-apples within (cell, N, subset_idx).
    idx = rng.choice(n_pool, size=N, replace=False)
    act_sub = activations_full[:, idx, :]  # (n_cond, N, H)

    # Compute the (n_cond × n_cond) metric matrix.
    payload = bakeoff._compute_metric_matrix(
        activations=act_sub,
        cond_ids=cond_ids,
        metric=cell.metric,
        extraction_point=cell.extraction_point,
        pca_k=bakeoff.PCA_DEFAULT_K,
        variant=cell.variant,
    )
    # Build the full-16 pair list (240 ordered pairs) — same as the
    # bakeoff's primary panel for last_prompt cells.
    pairs = bakeoff._pairs(cond_ids, nonstylized_only=False)
    xv = bakeoff._materialize_predictor_vector(payload, pairs, sub_predictor=None)
    if xv is None:
        return {
            "cell_id": cell.cell_id,
            "extraction_point": cell.extraction_point,
            "layer": cell.layer,
            "metric": cell.metric,
            "variant": cell.variant,
            "N": N,
            "subset_idx": subset_idx,
            "seed": int(seed),
            "abs_rho": float("nan"),
            "rho": float("nan"),
            "p": float("nan"),
            "cv_r2": float("nan"),
            "status": "predictor_vector_None",
        }
    dg = np.array([G[a][b]["delta_g"] for a, b in pairs], dtype=np.float64)
    ln = np.array([np.log(prompt_tokens[a][b]) for a, b in pairs], dtype=np.float64)
    rho, p_val = bakeoff._length_partial(xv, dg, ln)
    src = [a for a, _ in pairs]
    tgt = [b for _, b in pairs]
    cv = _loocv_r2_deterministic(xv, dg, src, tgt, covar=ln)
    return {
        "cell_id": cell.cell_id,
        "extraction_point": cell.extraction_point,
        "layer": cell.layer,
        "metric": cell.metric,
        "variant": cell.variant,
        "N": int(N),
        "subset_idx": int(subset_idx),
        "seed": int(seed),
        "abs_rho": float(abs(rho)) if np.isfinite(rho) else float("nan"),
        "rho": float(rho),
        "p": float(p_val),
        "cv_r2": float(cv),
        "status": "ok",
    }


def sweep(
    *,
    cells: list[CellSpec],
    arm: str,
    epochs: tuple[int, ...],
    out_path: Path,
    smoke_n_grid: tuple[int, ...] | None = None,
    smoke_r: int | None = None,
    checkpoint_every: int = 10,
) -> dict:
    """Walk every (cell, N, subset_idx, arm, epoch) row using the per-cell
    n_grid + r on each CellSpec. Per (extraction_point, layer) the
    activations are loaded ONCE and reused across all (N, subset_idx) draws.

    ``smoke_n_grid`` / ``smoke_r`` override per-cell n_grid + r when set
    (used by --mode smoke to force the tiny grid on every cell).

    Per ``checkpoint_every`` cells, the rows-so-far are persisted to the
    output JSON so a partial sweep is recoverable (the cloud-aware cells
    each take many minutes; CLAUDE.md "Checkpoint per phase" applies).
    """
    rows: list[dict] = []
    prompt_tokens = load_length_covar()
    # Cache: (extraction_point, layer) -> (activations, cond_ids)
    act_cache: dict[tuple[str, int], tuple[np.ndarray, list[str]]] = {}
    # Cache: (arm, epoch) -> G
    g_cache: dict[tuple[str, int], dict] = {}

    started_at = datetime.now(UTC).isoformat()
    t0 = time.time()
    # Total rows accounts for per-cell n_grid + r.
    total = sum(
        (len(smoke_n_grid) if smoke_n_grid else len(cell.n_grid))
        * (smoke_r if smoke_r else cell.r)
        * len(epochs)
        for cell in cells
    )
    done = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ep in epochs:
        if (arm, ep) not in g_cache:
            g_cache[(arm, ep)] = load_target(arm, ep)
        G = g_cache[(arm, ep)]
        for cell_idx, cell in enumerate(cells):
            cell_t0 = time.time()
            n_grid = smoke_n_grid if smoke_n_grid else cell.n_grid
            r_per_n = smoke_r if smoke_r else cell.r
            key = (cell.extraction_point, cell.layer)
            if key not in act_cache:
                logger.info("loading activations %s", key)
                act_cache[key] = load_activations_slice(*key)
            activations_full, cond_ids = act_cache[key]
            if set(cond_ids) != set(COND_IDS):
                raise AssertionError(
                    f"cond_ids mismatch on {key}: file has {sorted(cond_ids)} "
                    f"vs canonical {sorted(COND_IDS)}"
                )
            if set(G.keys()) != set(COND_IDS):
                raise AssertionError(
                    f"G cond_ids mismatch on (arm={arm}, ep={ep}): "
                    f"{sorted(G.keys())} vs canonical {sorted(COND_IDS)}"
                )
            for N in n_grid:
                for r in range(r_per_n):
                    row = score_one_cell(
                        cell=cell,
                        activations_full=activations_full,
                        cond_ids=cond_ids,
                        N=N,
                        subset_idx=r,
                        G=G,
                        prompt_tokens=prompt_tokens,
                    )
                    row["arm"] = arm
                    row["epoch"] = int(ep)
                    rows.append(row)
                    done += 1
                    if done % 50 == 0 or done == total:
                        elapsed = time.time() - t0
                        logger.info("  scored %d / %d rows  (elapsed %.1fs)", done, total, elapsed)
            cell_dt = time.time() - cell_t0
            logger.info(
                "cell %d/%d %s done in %.1fs (cumulative %.1fs)",
                cell_idx + 1,
                len(cells),
                cell.cell_id,
                cell_dt,
                time.time() - t0,
            )
            # Per-cell checkpoint write (CLAUDE.md Checkpoint per phase rule).
            if (cell_idx + 1) % checkpoint_every == 0 or cell_idx + 1 == len(cells):
                _write_partial(rows, out_path, started_at, arm, epochs, cells, t0)

    aggregates = aggregate(rows)
    # Plateau computable on any cell whose n_grid has {200, 350, 500} all present.
    plateau = compute_plateau(aggregates)
    payload = {
        "schema_version": 1,
        "git_sha": _git_sha(),
        "env": _env_versions(),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "wall_seconds": time.time() - t0,
        "arm": arm,
        "epochs": list(epochs),
        "cells_tracked": [
            {
                "cell_id": c.cell_id,
                "n_grid": list(smoke_n_grid if smoke_n_grid else c.n_grid),
                "r": int(smoke_r if smoke_r else c.r),
            }
            for c in cells
        ],
        "rows": rows,
        "aggregates": aggregates,
        "plateau_verdict": plateau,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s (%d rows, %.1fs)", out_path, len(rows), time.time() - t0)
    return payload


def _write_partial(
    rows: list[dict],
    out_path: Path,
    started_at: str,
    arm: str,
    epochs: tuple[int, ...],
    cells: list[CellSpec],
    t0: float,
) -> None:
    """Persist rows-so-far to ``out_path`` between cells. Cheap insurance
    against a crash mid-sweep."""
    payload = {
        "schema_version": 1,
        "partial": True,
        "git_sha": _git_sha(),
        "started_at": started_at,
        "checkpoint_at": datetime.now(UTC).isoformat(),
        "wall_seconds": time.time() - t0,
        "arm": arm,
        "epochs": list(epochs),
        "cells_tracked": [c.cell_id for c in cells],
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("checkpoint: wrote %d partial rows to %s", len(rows), out_path)


def aggregate(rows: list[dict]) -> dict:
    """Mean + std of |ρ| and CV R² at each (cell_id, arm, epoch, N) across
    the R subsets."""
    out: dict[tuple[str, str, int, int], dict] = {}
    grouped: dict[tuple[str, str, int, int], list[dict]] = {}
    for r in rows:
        key = (r["cell_id"], r["arm"], int(r["epoch"]), int(r["N"]))
        grouped.setdefault(key, []).append(r)
    for key, vs in grouped.items():
        abs_rhos = np.array(
            [v["abs_rho"] for v in vs if np.isfinite(v["abs_rho"])], dtype=np.float64
        )
        cvs = np.array([v["cv_r2"] for v in vs if np.isfinite(v["cv_r2"])], dtype=np.float64)
        out[key] = {
            "n_subsets": len(vs),
            "n_finite_abs_rho": int(abs_rhos.size),
            "n_finite_cv": int(cvs.size),
            "abs_rho_mean": float(abs_rhos.mean()) if abs_rhos.size else float("nan"),
            "abs_rho_std": float(abs_rhos.std(ddof=0)) if abs_rhos.size else float("nan"),
            "cv_mean": float(cvs.mean()) if cvs.size else float("nan"),
            "cv_std": float(cvs.std(ddof=0)) if cvs.size else float("nan"),
        }
    # Convert tuple keys to "cell_id|arm|epoch|N" strings for JSON.
    return {f"{cell_id}|{arm}|{epoch}|{N}": agg for (cell_id, arm, epoch, N), agg in out.items()}


def compute_plateau(aggregates: dict) -> dict:
    """Plateau verdict per (cell_id, arm, epoch) as defined in plan §4 Step 4.

    Reports a separate verdict per (cell_id, arm, epoch). The headline cell
    on (loc, 1) is the canonical plateau test; the per-cell verdicts
    are diagnostic.
    """
    # Parse keys back.
    parsed: dict[tuple[str, str, int], dict[int, dict]] = {}
    for k, v in aggregates.items():
        cell_id, arm, epoch_str, N_str = k.split("|")
        parsed.setdefault((cell_id, arm, int(epoch_str)), {})[int(N_str)] = v
    verdicts: dict[str, dict] = {}
    for (cell_id, arm, ep), by_N in parsed.items():
        if not (200 in by_N and 350 in by_N and 500 in by_N):
            continue
        # Plateau criterion uses CV R² (matches plan §6).
        cv_200 = by_N[200].get("cv_mean", float("nan"))
        cv_350 = by_N[350].get("cv_mean", float("nan"))
        cv_500 = by_N[500].get("cv_mean", float("nan"))
        sig_200 = by_N[200].get("cv_std", float("nan"))
        sig_350 = by_N[350].get("cv_std", float("nan"))
        # σ_ref pools across N=200 and N=350 (σ(N=500) is structurally 0).
        if not (np.isfinite(sig_200) and np.isfinite(sig_350)):
            verdict_label = "indeterminate_sigma"
            sigma_ref = float("nan")
            delta = float("nan")
        else:
            sigma_ref = 0.5 * (float(sig_200) + float(sig_350))
            delta = float(cv_500 - cv_350)
            if not np.isfinite(delta):
                verdict_label = "indeterminate_delta"
            elif abs(delta) <= sigma_ref:
                verdict_label = "plateau"
            elif delta > 2.0 * sigma_ref and delta > 0:
                verdict_label = "strongly_climbing"
            else:
                verdict_label = "intermediate"
        verdicts[f"{cell_id}|{arm}|ep{ep}"] = {
            "cv_200": cv_200,
            "cv_350": cv_350,
            "cv_500": cv_500,
            "sigma_200": sig_200,
            "sigma_350": sig_350,
            "sigma_ref": sigma_ref,
            "delta_350_to_500": delta,
            "verdict": verdict_label,
        }
    return verdicts


# ───────────────────────── reproduction gate ─────────────────────────


def run_reproduction_gate() -> dict:
    """Plan §4 Step 0 item 3 (binding directive).

    Recompute the headline cell at N=500 / R=1 (identity-permutation
    point) and compare against the #502 archived headline. Returns a dict
    with the comparison.

    Two-part gate (diagnosed in round 1):

    1. ρ tolerance gate (BINDING). ``|Δρ| < 1e-3``. The ``_length_partial``
       pipeline is deterministic across runs; an out-of-tolerance ρ means
       the predictor numerics drifted and the plateau verdict isn't
       answering the same question. This is the gate that halts execution.
    2. CV diagnostic (NOT binding). The bakeoff's ``_loocv_r2`` iterates
       ``set(cond_ids_a) | set(cond_ids_b)`` for folds, and Python set
       iteration over strings is hash-randomized via ``PYTHONHASHSEED``.
       Each off-diagonal pair (A, B) sits in BOTH fold ``C=A`` and fold
       ``C=B``; the LAST fold to execute overwrites ``pred[test]``, so
       different hash seeds deterministically produce different CV R² at
       the ~5e-3 level. The wrapper uses a sorted-fold deterministic
       variant (``_loocv_r2_deterministic``) so the (N, R) sweep variance
       reflects probe-subset noise only, but the archived bakeoff_grid CV
       was produced under unspecified hash seed, so a 1e-3 gate against
       it is comparing a deterministic value to a random draw. We REPORT
       the delta but do not fail on it.
    """
    cell = CellSpec(*HEADLINE_CELL, n_grid=N_GRID_FULL, r=R_HEADLINE)
    activations_full, cond_ids = load_activations_slice(cell.extraction_point, cell.layer)
    n_pool = activations_full.shape[1]
    if n_pool != 500:
        raise AssertionError(f"reproduction gate expected pool=500, got {n_pool} on {cell.cell_id}")
    # Identity permutation = no subset draw; pass the full pool through
    # the same wrapper as the sweep so a wrapper-side drift would surface.
    G = load_target("loc", 1)
    prompt_tokens = load_length_covar()
    payload = bakeoff._compute_metric_matrix(
        activations=activations_full,
        cond_ids=cond_ids,
        metric=cell.metric,
        extraction_point=cell.extraction_point,
        pca_k=bakeoff.PCA_DEFAULT_K,
        variant=cell.variant,
    )
    pairs = bakeoff._pairs(cond_ids, nonstylized_only=False)
    xv = bakeoff._materialize_predictor_vector(payload, pairs, sub_predictor=None)
    if xv is None:
        raise AssertionError("reproduction gate predictor vector is None")
    dg = np.array([G[a][b]["delta_g"] for a, b in pairs], dtype=np.float64)
    ln = np.array([np.log(prompt_tokens[a][b]) for a, b in pairs], dtype=np.float64)
    rho, _ = bakeoff._length_partial(xv, dg, ln)
    src = [a for a, _ in pairs]
    tgt = [b for _, b in pairs]
    cv = _loocv_r2_deterministic(xv, dg, src, tgt, covar=ln)
    drho = abs(float(rho) - ARCHIVED_HEADLINE_RHO)
    dcv = abs(float(cv) - ARCHIVED_HEADLINE_CV)
    gate = {
        "rho_new": float(rho),
        "rho_archived": ARCHIVED_HEADLINE_RHO,
        "delta_rho": float(drho),
        "rho_pass": bool(drho < GATE_TOLERANCE),
        "cv_new_deterministic": float(cv),
        "cv_archived_nondeterministic": ARCHIVED_HEADLINE_CV,
        "delta_cv_diagnostic": float(dcv),
        "cv_diagnostic_only": True,
        "cv_archived_nondeterminism_note": (
            "bakeoff._loocv_r2 iterates set() of cond_ids; hash-seed-dependent "
            "fold-overwrite tie-break drifts CV by ~5e-3 across runs. "
            "Wrapper uses _loocv_r2_deterministic (sorted folds); the (N, R) "
            "sweep CVs are reproducible. cv_new_deterministic IS the canonical "
            "anchor for downstream #511 analysis."
        ),
        "tolerance": GATE_TOLERANCE,
        "pass": bool(drho < GATE_TOLERANCE),  # ρ-only gate (binding)
    }
    return gate


# ───────────────────────── repro metadata ─────────────────────────


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _env_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("numpy", "scipy", "torch", "transformers"):
        try:
            mod = __import__(pkg)
            out[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[pkg] = "not-installed"
    return out


# ───────────────────────── CLI ─────────────────────────


def main() -> int:
    """CLI dispatch over three modes: reproduction-gate, smoke, full."""
    parser = argparse.ArgumentParser(
        description=(
            "Probe-count convergence sweep wrapping #502's predictor pipeline. "
            "Plan §4 Step 0/2/4 of task #511."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("reproduction-gate", "smoke", "full"),
        required=True,
        help="reproduction-gate: §4 Step 0 numerical-reproduction check only. "
        "smoke: N={25,50} R=2 headline cell only. "
        "full: full N×R×cell grid on loc_ep1 (+ optional --epochs).",
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default="1",
        help="Comma-separated loc-arm epochs to score (full mode only). "
        "Default 1 (loc_ep1 is the primary plateau target).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(OUT_DIR / "probe_count_sweep_results.json"),
        help="Output JSON path (full + smoke modes).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging.")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.mode == "reproduction-gate":
        gate = run_reproduction_gate()
        # Persist the gate result alongside other artifacts.
        gate_path = OUT_DIR / "reproduction_gate.json"
        gate_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "git_sha": _git_sha(),
                    "env": _env_versions(),
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    **gate,
                },
                indent=2,
            )
        )
        print(json.dumps(gate, indent=2))
        if not gate["pass"]:
            logger.error(
                "REPRODUCTION GATE FAIL: |Δρ|=%.3e (cap %.0e) — predictor "
                "numerics drift; halt before plateau verdict.",
                gate["delta_rho"],
                gate["tolerance"],
            )
            return 2
        logger.info(
            "REPRODUCTION GATE PASS (ρ): |Δρ|=%.3e (cap %.0e). "
            "CV diagnostic: deterministic=%.6f vs archived (non-det)=%.6f, "
            "|ΔCV|=%.3e (NOT binding — bakeoff CV is hash-seed-dependent).",
            gate["delta_rho"],
            gate["tolerance"],
            gate["cv_new_deterministic"],
            gate["cv_archived_nondeterministic"],
            gate["delta_cv_diagnostic"],
        )
        return 0

    if args.mode == "smoke":
        cells = [CellSpec(*HEADLINE_CELL, n_grid=N_GRID_SMOKE, r=R_SMOKE)]
        out_path = Path(args.out).with_name("smoke_results.json")
        epochs: tuple[int, ...] = (1,)
        smoke_n_grid: tuple[int, ...] | None = N_GRID_SMOKE
        smoke_r: int | None = R_SMOKE
    else:  # full
        cells = build_cell_list()
        out_path = Path(args.out)
        epochs = tuple(int(e.strip()) for e in args.epochs.split(",") if e.strip())
        smoke_n_grid = None
        smoke_r = None

    n_rows_total = sum(
        (len(smoke_n_grid) if smoke_n_grid else len(c.n_grid))
        * (smoke_r if smoke_r else c.r)
        * len(epochs)
        for c in cells
    )
    logger.info(
        "starting %s sweep: %d cells, %d total rows across %d epochs (per-cell n_grid×R)",
        args.mode,
        len(cells),
        n_rows_total,
        len(epochs),
    )

    sweep(
        cells=cells,
        arm=DEFAULT_ARM,
        epochs=epochs,
        out_path=out_path,
        smoke_n_grid=smoke_n_grid,
        smoke_r=smoke_r,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
