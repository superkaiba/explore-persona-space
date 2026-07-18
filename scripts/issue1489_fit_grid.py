#!/usr/bin/env python3
"""Issue #1489 P6 fit + analysis driver (plan §4.5) — cpu-bigmem GCE boxes.

IMPORTS the #1092 fit engine (`scripts/issue1092_fit_grid.py`: `_fit_cv`,
`_load_summary`, `_folds_from_manifest`, `_perm_null`, `_pca_basis`,
`_identity_floors`, `_principal_angles`, `_spectrum`, `_load_rb_directions`)
rather than rewriting it — the #1489 capture rig mirrors the parent summaries
schema exactly, so the loaders run unchanged. All draw batteries are BATCHED
(einsum/GEMM over the draw axis; no per-draw refits) per
`.claude/rules/vectorize-many-cell-fits.md`.

Units (each checkpointed to ``<out>/units/<unit_id>.json`` the moment it
completes; resume skips completed units keyed on the full regime fingerprint):

    transfer:<arm>:<basis>:L<layer>   Q1 5x5 transport-transfer matrix (+ diag
                                      PRESS fits, perm nulls, split-half refit
                                      twins, identity/mean floors, map-cmp
                                      disagreement + explicit-map secondary
                                      stats at the headline unit)
    shifts:<c_kind>:L<layer>          Q2 per-instance shift decomposition
    gating:<arm>:L<layer>             Q3 relevance gating AUCs + MLP companion
    q4:L<layer>                       Q4 per-example FT-vs-ctx alignment
                                      (DISJOINT-BASELINE primary + 3 nulls)
    q6:<arm>:L<layer>                 Q6 map stability on the FT models
    loto:<arm>:L<layer>               registered leave-one-topic-out sensitivity

Gates: G2 fit-substrate kill (plain diagonal R^2 < 0.30 at the frozen headline
unit -> report JSON + rc=21, demoted to informational under --smoke); G3 pilot
timing (first unit wall projected against the §9 row; >2x -> descope note per
the §9 stratification ladder, informational under --smoke).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from issue1092_fit_grid import (  # noqa: E402
    DEFAULT_RB_REV,
    _fit_cv,
    _folds_from_manifest,
    _identity_floors,
    _load_rb_directions,
    _load_summary,
    _pca_basis,
    _perm_null,
    _r2,
    _spectrum,
)
from issue1092_fit_grid import (  # noqa: E402
    _read_index_files as _parent_read_index,
)
from issue1489_common import (  # noqa: E402
    AUGMENT_SLUGS,
    DISTILL_RUNS,
    FAMILIES,
    RELEVANCE_MAP,
    augment_family,
)
from issue923_fit_decomposition import press_fit_predict  # noqa: E402

logger = logging.getLogger("issue1489_fit_grid")

HEADLINE = {"layer": 14, "arm": "context_end", "basis": "ambient"}
G2_R2_FLOOR = 0.30
G3_PLAN_HOURS_PER_BOX = 8.75  # §9: 35 machine-h booked / 4 boxes
PCA_K = 48
MLP_MAX_ROWS = 300


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float))
    tmp.replace(path)


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Data access (parent summaries schema + #1489 row_index shards)
# ---------------------------------------------------------------------------


class CellData:
    """Aligned (rows, X-kind arrays) accessor for one cell at one layer."""

    def __init__(self, summaries_dir: Path, cell: str):
        self.summaries_dir = summaries_dir
        self.cell = cell
        self.rows = _parent_read_index(summaries_dir / cell, "row_index")
        if not self.rows:
            raise FileNotFoundError(f"no row_index for {cell}")
        self.by_base = {r["base_row_id"]: i for i, r in enumerate(self.rows)}
        if len(self.by_base) != len(self.rows):
            raise ValueError(f"{cell}: duplicate base_row_id in row_index")
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    def kind(self, kind: str, layer: int) -> np.ndarray:
        key = (kind, layer)
        if key not in self._cache:
            arr, _paths = _load_summary(self.summaries_dir, self.cell, kind, layer)
            if arr.shape[0] != len(self.rows):
                raise ValueError(
                    f"{self.cell}/{kind}/L{layer}: {arr.shape[0]} rows != "
                    f"row_index {len(self.rows)}"
                )
            self._cache[key] = arr
        return self._cache[key]

    def subset(self, base_ids: list[str]) -> np.ndarray:
        missing = [b for b in base_ids if b not in self.by_base]
        if missing:
            raise KeyError(f"{self.cell}: missing {len(missing)} base rows: {missing[:3]}")
        return np.array([self.by_base[b] for b in base_ids], dtype=np.int64)


def _family_cells(cells_present: set[str]) -> dict[str, list[str]]:
    fams: dict[str, list[str]] = {f: [] for f in FAMILIES}
    for slug in AUGMENT_SLUGS:
        if f"cell_{slug}" in cells_present:
            fams[augment_family(slug)].append(f"cell_{slug}")
    return {f: cs for f, cs in fams.items() if cs}


def _train_pools(
    summaries_dir: Path, layer: int, arm: str, basis_proj, cells_present: set[str]
) -> dict[str, dict]:
    """Per-family train pools: X (arm kind), Y (t1, basis-projected), rows."""
    pools: dict[str, dict] = {}
    plain = CellData(summaries_dir, "cell_plain")
    pools["plain"] = {
        "X": plain.kind(arm, layer),
        "Y": basis_proj(plain.kind("t1", layer)),
        "rows": plain.rows,
    }
    for fam, cells in _family_cells(cells_present).items():
        xs, ys, rows = [], [], []
        for cell in cells:
            cd = CellData(summaries_dir, cell)
            xs.append(cd.kind(arm, layer))
            ys.append(basis_proj(cd.kind("t1", layer)))
            rows.extend(cd.rows)
        pools[fam] = {"X": np.concatenate(xs), "Y": np.concatenate(ys), "rows": rows}
    return pools


# ---------------------------------------------------------------------------
# Batched statistics helpers (no per-draw Python refits)
# ---------------------------------------------------------------------------


def _grouped_half_split(groups: list[str], rng: np.random.Generator) -> np.ndarray:
    uniq = sorted(set(groups))
    order = rng.permutation(len(uniq))
    half = {uniq[i] for i in order[: len(uniq) // 2]}
    return np.array([g in half for g in groups], dtype=bool)


def split_half_refit_twins(
    X: np.ndarray,
    Y: np.ndarray,
    groups: list[str],
    folds: list[np.ndarray],
    *,
    n_twins: int,
    seed: int,
) -> dict:
    """Refit-vs-refit disagreement noise floor (plan §2 divergence 4).

    Per twin draw: split the TRAIN side of each fold into group-disjoint
    halves, fit each half (PRESS ridge), evaluate BOTH on the fold's test
    rows; the per-draw statistic is |R^2(half1) - R^2(half2)| pooled over
    folds. q95 over draws is the split-half refit null the §3 verdict lattice
    subtracts.
    """
    import torch

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    gaps: list[float] = []
    for _t in range(n_twins):
        preds1 = np.zeros_like(Y, dtype=np.float64)
        preds2 = np.zeros_like(Y, dtype=np.float64)
        for test_idx in folds:
            mask = np.ones(n, dtype=bool)
            mask[test_idx] = False
            tr_idx = np.where(mask)[0]
            half_mask = _grouped_half_split([groups[i] for i in tr_idx], rng)
            h1, h2 = tr_idx[half_mask], tr_idx[~half_mask]
            if len(h1) < 2 or len(h2) < 2:
                raise ValueError("split-half twin: degenerate half (<2 rows)")
            for hidx, preds in ((h1, preds1), (h2, preds2)):
                res = press_fit_predict(
                    torch.from_numpy(X[hidx]).double(),
                    torch.from_numpy(Y[hidx]).double(),
                    torch.from_numpy(X[test_idx]).double(),
                    standardize=True,
                )
                preds[test_idx] = res["pred"].detach().cpu().numpy()
        gaps.append(abs(_r2(Y, preds1) - _r2(Y, preds2)))
    arr = np.asarray(gaps, dtype=np.float64)
    return {
        "n_twins": n_twins,
        "gaps": [float(g) for g in gaps],
        "q95": float(np.quantile(arr, 0.95)),
        "mean": float(arr.mean()),
    }


def _rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), NaN-free fail-loud."""
    if not np.isfinite(scores).all():
        raise ValueError("AUC scores contain non-finite values")
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUC needs both classes")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # midranks for ties
    allv = np.concatenate([pos, neg])
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _held_out_linear_scores(
    X: np.ndarray, labels: np.ndarray, folds: list[np.ndarray]
) -> np.ndarray:
    """Held-out linear read on +-1 labels (grouped ridge — the AUC scorer)."""
    import torch

    y = np.where(labels, 1.0, -1.0)[:, None]
    scores = np.zeros(X.shape[0], dtype=np.float64)
    n = X.shape[0]
    for test_idx in folds:
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        res = press_fit_predict(
            torch.from_numpy(X[mask]).double(),
            torch.from_numpy(y[mask]).double(),
            torch.from_numpy(X[test_idx]).double(),
            standardize=True,
        )
        scores[test_idx] = res["pred"].detach().cpu().numpy()[:, 0]
    return scores


def _perm_auc_null(scores: np.ndarray, labels: np.ndarray, n_draws: int, seed: int) -> dict:
    """Row-label permutation AUC null — one vectorized rank pass per draw batch."""
    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws, dtype=np.float64)
    for d in range(n_draws):  # rank AUC is O(n log n); n_draws*n is tiny
        perm = rng.permutation(len(labels))
        draws[d] = _rank_auc(scores, labels[perm])
    return {
        "n_draws": n_draws,
        "q95": float(np.quantile(draws, 0.95)),
        "q05": float(np.quantile(draws, 0.05)),
        "mean": float(draws.mean()),
    }


def _row_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.einsum("nd,nd->n", a, b)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return num / den


def _grouped_bootstrap_ci(values: np.ndarray, groups: list[str], *, n_boot: int, seed: int) -> dict:
    """Grouped-by-prefix bootstrap CI of the mean (vectorized gather)."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(groups))
    gidx: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        gidx.setdefault(g, []).append(i)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        picked = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([np.asarray(gidx[uniq[p]]) for p in picked])
        means[b] = values[idx].mean()
    return {
        "mean": float(values.mean()),
        "ci_lo": float(np.quantile(means, 0.025)),
        "ci_hi": float(np.quantile(means, 0.975)),
        "n_boot": n_boot,
        "n_groups": len(uniq),
    }


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def unit_transfer(args, summaries_dir: Path, layer: int, arm: str, basis: str) -> dict:
    cells_present = _cells_present(summaries_dir)
    plain = CellData(summaries_dir, "cell_plain")
    if basis == "ambient":

        def proj(y: np.ndarray) -> np.ndarray:
            return y.astype(np.float64)
    elif basis == "pca48":
        mu, v = _pca_basis(plain.kind("t1", layer).astype(np.float64), PCA_K)

        def proj(y: np.ndarray) -> np.ndarray:
            return (y.astype(np.float64) - mu) @ v
    else:
        raise ValueError(f"unknown basis {basis}")

    pools = _train_pools(summaries_dir, layer, arm, proj, cells_present)
    fams = list(pools)
    folds_by_fam = {
        f: _folds_from_manifest(
            p["rows"], p["X"].shape[0], group_key=args.group_key, n_folds=args.n_folds
        )
        for f, p in pools.items()
    }

    import torch

    # 5x5 matrix: per train family A, per fold, ONE fit; test X = concat of all
    # families' fold rows (matched-target scoring: every family evaluated on
    # ITS OWN held-out rows/targets under A's map).
    matrix: dict[str, dict[str, float]] = {a: {} for a in fams}
    preds_store: dict[tuple[str, str], np.ndarray] = {}
    diag_stats: dict[str, dict] = {}
    for a in fams:
        Xa, Ya = pools[a]["X"].astype(np.float64), pools[a]["Y"]
        na = Xa.shape[0]
        preds_on: dict[str, np.ndarray] = {b: np.zeros_like(pools[b]["Y"]) for b in fams}
        lambda_indices: list[int] = []
        for f_i, test_idx in enumerate(folds_by_fam[a]):
            mask = np.ones(na, dtype=bool)
            mask[test_idx] = False
            test_blocks = []
            test_slices: dict[str, tuple[int, np.ndarray]] = {}
            off = 0
            for b in fams:
                b_test = folds_by_fam[b][f_i] if f_i < len(folds_by_fam[b]) else np.array([], int)
                xb = pools[b]["X"][b_test].astype(np.float64)
                test_blocks.append(xb)
                test_slices[b] = (off, b_test)
                off += xb.shape[0]
            X_test = np.concatenate(test_blocks) if test_blocks else np.zeros((0, Xa.shape[1]))
            res = press_fit_predict(
                torch.from_numpy(Xa[mask]).double(),
                torch.from_numpy(Ya[mask]).double(),
                torch.from_numpy(X_test).double(),
                standardize=True,
            )
            pred = res["pred"].detach().cpu().numpy()
            lambda_indices.append(int(res["lam_idx"]))
            for b, (off_b, b_test) in test_slices.items():
                if len(b_test):
                    preds_on[b][b_test] = pred[off_b : off_b + len(b_test)]
        for b in fams:
            covered = np.concatenate(folds_by_fam[b]) if folds_by_fam[b] else np.array([], int)
            matrix[a][b] = _r2(pools[b]["Y"][covered], preds_on[b][covered])
        preds_store[(a, "plain")] = preds_on["plain"]
        diag_stats[a] = {
            "n": int(na),
            "lambda_indices": lambda_indices,
            "r2_diag": matrix[a][a],
        }

    # diagonal extras at this unit: perm null + identity floors + twins + spectrum
    extras: dict[str, dict] = {}
    for a in fams:
        Xa, Ya = pools[a]["X"].astype(np.float64), pools[a]["Y"]
        folds = folds_by_fam[a]
        fit_stats, _pred = _fit_cv(Xa, Ya, folds, return_pred=True)
        extras[a] = {
            "fit": fit_stats,
            "identity_floors": _identity_floors(Xa, Ya, folds),
            "perm_null": _perm_null(
                Xa,
                Ya,
                folds,
                args.n_random_draws,
                args.seed,
                lambda_indices=fit_stats["lambda_indices"],
            ),
            "refit_twins": split_half_refit_twins(
                Xa,
                Ya,
                [str(r.get(args.group_key)) for r in pools[a]["rows"]],
                folds,
                n_twins=args.n_refit_twins,
                seed=args.seed,
            ),
            "spectrum": _spectrum(Xa, Ya),
        }

    # map-comparison stats on the shared plain test rows (§4.5 item 5)
    mapcmp = _mapcmp_stats(args, pools, preds_store, layer, basis)

    return {
        "families": fams,
        "matrix": matrix,
        "diag": diag_stats,
        "extras": extras,
        "mapcmp": mapcmp,
        "basis": basis,
        "arm": arm,
        "layer": layer,
    }


def _mapcmp_stats(args, pools, preds_store, layer: int, basis: str) -> dict:
    """Normalized pairwise map disagreement on plain rows + r_B projections."""
    fams = list(pools)
    var_v = float(np.var(pools["plain"]["Y"], axis=0).sum())
    out: dict[str, dict] = {"normalized_disagreement": {}, "rb_projected_disagreement": {}}
    rb = None
    if basis == "ambient":
        try:
            rb_arr, rb_names = _load_rb_directions(
                argparse.Namespace(
                    rb_rev=args.rb_rev,
                    n_layers=28,
                    hidden_dim=pools["plain"]["Y"].shape[1],
                    rb_dir=None,
                )
            )
            rb = (rb_arr[layer], rb_names)  # (traits, hidden)
        except Exception as exc:  # noqa: BLE001 — secondary metric; recorded, not fatal
            out["rb_error"] = f"{type(exc).__name__}: {exc}"
    for i, a in enumerate(fams):
        for b in fams[i + 1 :]:
            pa, pb = preds_store[(a, "plain")], preds_store[(b, "plain")]
            d = float(((pa - pb) ** 2).sum(axis=1).mean()) / max(var_v, 1e-12)
            out["normalized_disagreement"][f"{a}|{b}"] = d
            if rb is not None:
                rb_arr, rb_names = rb
                unit = rb_arr / (np.linalg.norm(rb_arr, axis=1, keepdims=True) + 1e-12)
                proj = (pa - pb) @ unit.T  # (n, traits)
                out["rb_projected_disagreement"][f"{a}|{b}"] = {
                    name: float((proj[:, t] ** 2).mean()) for t, name in enumerate(rb_names)
                }
    return out


def unit_shifts(args, summaries_dir: Path, layer: int, c_kind: str) -> dict:
    """Q2 per-instance shift decomposition + commute test + label-perm nulls."""
    cells_present = _cells_present(summaries_dir)
    plain = CellData(summaries_dir, "cell_plain")
    import torch

    out: dict[str, dict] = {}
    # plain map for the commute test (one full-pool fit; predictions on
    # c_plain and c_plain + mu_k in ONE press call per instance batch)
    Xp = plain.kind(c_kind, layer).astype(np.float64)
    Yp = plain.kind("t1", layer).astype(np.float64)
    for slug in AUGMENT_SLUGS:
        cell = f"cell_{slug}"
        if cell not in cells_present:
            continue
        cd = CellData(summaries_dir, cell)
        shared = [b for b in (r["base_row_id"] for r in cd.rows) if b in plain.by_base]
        idx_aug = cd.subset(shared)
        idx_plain = plain.subset(shared)
        d_c = cd.kind(c_kind, layer)[idx_aug].astype(np.float64) - Xp[idx_plain]
        d_v = cd.kind("t1", layer)[idx_aug].astype(np.float64) - Yp[idx_plain]
        stats = {}
        for name, d in (("dc", d_c), ("dv", d_v)):
            mu = d.mean(axis=0)
            resid = d - mu
            var_tot = float((d**2).sum())
            r2_mu = 1.0 - float((resid**2).sum()) / max(var_tot, 1e-12)
            # PC spectrum + participation ratio of centered shifts
            _u, s, _vh = np.linalg.svd(resid, full_matrices=False)
            ev = s**2
            pr = float(ev.sum() ** 2 / max((ev**2).sum(), 1e-12))
            stats[name] = {
                "mu_norm": float(np.linalg.norm(mu)),
                "r2_mu": r2_mu,
                "participation_ratio": pr,
                "top10_ev_frac": [float(x) for x in (ev[:10] / max(ev.sum(), 1e-12))],
            }
        # label-permutation null for R^2_mu (batched: draws x rows gather)
        rng = np.random.default_rng(args.seed)
        pool = np.concatenate([d_c, -d_c])  # sign-flip null pool for the mean
        n = d_c.shape[0]
        draws = np.empty(args.n_perm, dtype=np.float64)
        for t in range(args.n_perm):
            signs = rng.choice([1.0, -1.0], size=n)[:, None]
            d_null = d_c * signs
            mu = d_null.mean(axis=0)
            draws[t] = 1.0 - float(((d_null - mu) ** 2).sum()) / max(
                float((d_null**2).sum()), 1e-12
            )
        del pool
        # commute test: M_plain(mu_c) vs mean dv
        mu_c = d_c.mean(axis=0)
        base = Xp[idx_plain]
        X_test = np.concatenate([base, base + mu_c[None, :]])
        res = press_fit_predict(
            torch.from_numpy(Xp).double(),
            torch.from_numpy(Yp).double(),
            torch.from_numpy(X_test).double(),
            standardize=True,
        )
        pred = res["pred"].detach().cpu().numpy()
        m_mu = (pred[len(base) :] - pred[: len(base)]).mean(axis=0)
        mu_v = d_v.mean(axis=0)
        cos = float(np.dot(m_mu, mu_v) / (np.linalg.norm(m_mu) * np.linalg.norm(mu_v) + 1e-12))
        rel_err = float(np.linalg.norm(m_mu - mu_v) / (np.linalg.norm(mu_v) + 1e-12))
        out[slug] = {
            "n_shared_rows": len(shared),
            **stats,
            "r2_mu_signflip_null_q95": float(np.quantile(draws, 0.95)),
            "commute_cos": cos,
            "commute_rel_err": rel_err,
        }
    # cross-instance rank of stacked mu_k (per family + all)
    mus = {}
    for slug, s in out.items():
        cell = f"cell_{slug}"
        cd = CellData(summaries_dir, cell)
        shared = [b for b in (r["base_row_id"] for r in cd.rows) if b in plain.by_base]
        idx_aug = cd.subset(shared)
        idx_plain = plain.subset(shared)
        mus[slug] = (cd.kind(c_kind, layer)[idx_aug].astype(np.float64) - Xp[idx_plain]).mean(
            axis=0
        )
    if mus:
        stack = np.stack(list(mus.values()))
        _u, s, _vh = np.linalg.svd(stack - stack.mean(axis=0), full_matrices=False)
        ev = s**2
        out["_stacked_mu"] = {
            "slugs": list(mus),
            "participation_ratio": float(ev.sum() ** 2 / max((ev**2).sum(), 1e-12)),
            "sv_frac": [float(x) for x in (ev / max(ev.sum(), 1e-12))],
        }
    return out


def unit_gating(args, summaries_dir: Path, layer: int, arm: str) -> dict:
    """Q3 gating AUCs (Δc, Δv, map-predicted Δv̂) + MLP-vs-ridge companion."""
    cells_present = _cells_present(summaries_dir)
    plain = CellData(summaries_dir, "cell_plain")
    Xp = plain.kind(arm, layer).astype(np.float64)
    Yp = plain.kind("t1", layer).astype(np.float64)
    import torch

    out: dict[str, dict] = {}
    scoped = [s for s in RELEVANCE_MAP if RELEVANCE_MAP[s] and f"cell_{s}" in cells_present]
    for slug in scoped:
        cd = CellData(summaries_dir, f"cell_{slug}")
        shared = [b for b in (r["base_row_id"] for r in cd.rows) if b in plain.by_base]
        idx_aug = cd.subset(shared)
        idx_plain = plain.subset(shared)
        labels = np.array([bool(cd.rows[i].get("relevant")) for i in idx_aug], dtype=bool)
        if labels.sum() == 0 or (~labels).sum() == 0:
            raise ValueError(f"gating {slug}: single-class relevance labels")
        groups_rows = [cd.rows[i] for i in idx_aug]
        folds = _folds_from_manifest(
            groups_rows, len(shared), group_key=args.group_key, n_folds=args.n_folds
        )
        d_c = cd.kind(arm, layer)[idx_aug].astype(np.float64) - Xp[idx_plain]
        d_v = cd.kind("t1", layer)[idx_aug].astype(np.float64) - Yp[idx_plain]
        res: dict[str, dict] = {}
        for name, d in (("gamma_c", d_c), ("gamma_v", d_v)):
            scores = _held_out_linear_scores(d, labels, folds)
            res[name] = {
                "auc": _rank_auc(scores, labels),
                "perm_null": _perm_auc_null(scores, labels, args.n_perm, args.seed),
            }
        # map-predicted contrast: dv_hat = M_plain(c_aug) - M_plain(c_plain)
        X_test = np.concatenate([cd.kind(arm, layer)[idx_aug].astype(np.float64), Xp[idx_plain]])
        fit = press_fit_predict(
            torch.from_numpy(Xp).double(),
            torch.from_numpy(Yp).double(),
            torch.from_numpy(X_test).double(),
            standardize=True,
        )
        pred = fit["pred"].detach().cpu().numpy()
        dv_hat = pred[: len(shared)] - pred[len(shared) :]
        scores = _held_out_linear_scores(dv_hat, labels, folds)
        res["gamma_v_hat"] = {
            "auc": _rank_auc(scores, labels),
            "perm_null": _perm_auc_null(scores, labels, args.n_perm, args.seed),
        }
        # predicted relevant-vs-irrelevant dv contrast magnitude
        res["dv_hat_contrast"] = {
            "rel_norm": float(np.linalg.norm(dv_hat[labels].mean(axis=0))),
            "irr_norm": float(np.linalg.norm(dv_hat[~labels].mean(axis=0))),
            "dv_rel_norm": float(np.linalg.norm(d_v[labels].mean(axis=0))),
            "dv_irr_norm": float(np.linalg.norm(d_v[~labels].mean(axis=0))),
        }
        out[slug] = {"n_rows": len(shared), "n_relevant": int(labels.sum()), **res}

    out["mlp_companion"] = _mlp_companion(args, summaries_dir, layer, arm)
    return out


def _mlp_companion(args, summaries_dir: Path, layer: int, arm: str) -> dict:
    """MLP-vs-ridge held-out gap on facts rows + plain control (batched helper)."""
    from explore_persona_space.analysis.vectorized_mlp_skill import (
        MLPGroup,
        fit_batched_loco_mlp,
    )

    cells_present = _cells_present(summaries_dir)
    plain = CellData(summaries_dir, "cell_plain")
    rng = np.random.default_rng(args.seed)
    groups: list[MLPGroup] = []
    ridge_r2: dict[str, float] = {}
    row_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fact_cells = [f"cell_{s}" for s in AUGMENT_SLUGS if s.startswith("fact_")]
    fact_cells = [c for c in fact_cells if c in cells_present]
    xs, ys = [], []
    for cell in fact_cells:
        cd = CellData(summaries_dir, cell)
        xs.append(cd.kind(arm, layer).astype(np.float32))
        ys.append(cd.kind("t1", layer).astype(np.float32))
    pools = {"plain": (plain.kind(arm, layer), plain.kind("t1", layer))}
    if xs:
        pools["facts"] = (np.concatenate(xs), np.concatenate(ys))
    # fit_batched_loco_mlp requires EVERY group to share the same n (one
    # vmapped ensemble) — equalize to the smallest pool, capped at MLP_MAX_ROWS.
    n_common = min(min(x.shape[0] for x, _y in pools.values()), MLP_MAX_ROWS)
    n_rows_used = {}
    for name, (X, Y) in pools.items():
        n = n_common
        idx = rng.choice(X.shape[0], size=n, replace=False)
        Xs, Ysub = X[idx].astype(np.float64), Y[idx].astype(np.float64)
        # Input-PCA to rank<=n is LOSSLESS here (pool rows span <=n dims; ridge
        # predictions are span-invariant by the representer theorem and the MLP
        # first layer absorbs the rotation) and cuts the fold-batched MLP weight
        # ensemble from d_in=3584 to <=n dims (#722 input-PCA lineage; the ambient
        # ensemble at d_in=3584 hit 23.7 GB RSS -> earlyoom on 2026-07-17).
        mu_x, v_x = _pca_basis(Xs, min(n, Xs.shape[1]))
        Xs = (Xs - mu_x) @ v_x
        mu, v = _pca_basis(Ysub, PCA_K)
        Yk = ((Ysub - mu) @ v).astype(np.float32)
        row_sets[name] = (Xs.astype(np.float32), Yk)
        n_rows_used[name] = int(n)
        # ridge LOO on the identical rows/targets (matched fold structure: LOCO)
        loo_folds = [np.array([i]) for i in range(n)]
        fit = _fit_cv(Xs, Yk.astype(np.float64), loo_folds)
        ridge_r2[name] = fit["r2"]
        groups.append(MLPGroup(key=name, X=row_sets[name][0], Y=row_sets[name][1]))
    result = fit_batched_loco_mlp(
        groups,
        seed=args.seed,
        hidden=512,
        max_epochs=300 if not args.smoke else 20,
        device=args.mlp_device,
    )
    out = {"n_rows_used": n_rows_used, "ridge_r2_loo": ridge_r2, "mlp_r2_loo": {}}
    for name, (X, Yk) in row_sets.items():
        pred = result.preds_by_key[name]
        out["mlp_r2_loo"][name] = _r2(Yk.astype(np.float64), pred.astype(np.float64))
        out.setdefault("gap", {})[name] = out["mlp_r2_loo"][name] - ridge_r2[name]
    return out


def unit_q4(args, summaries_dir: Path, layer: int) -> dict:
    """Q4 per-example FT-vs-ctx alignment (disjoint-baseline primary; §3)."""
    cells_present = _cells_present(summaries_dir)
    plain = CellData(summaries_dir, "cell_plain")
    out: dict[str, dict] = {}
    slugs = [s for s in DISTILL_RUNS if f"cell_ft_{s}" in cells_present]
    if not slugs:
        raise FileNotFoundError("q4: no cell_ft_* summaries present")
    # shared-x cross-augmentation null needs >=2 runs; fail-soft note at 1 (smoke)
    ctx_dv: dict[str, dict[str, np.ndarray]] = {}
    for slug in slugs:
        ft = CellData(summaries_dir, f"cell_ft_{slug}")
        aug = CellData(summaries_dir, f"cell_{slug}")
        shared = [
            b
            for b in (r["base_row_id"] for r in ft.rows)
            if b in plain.by_base and b in aug.by_base
        ]
        i_ft = ft.subset(shared)
        i_pl = plain.subset(shared)
        i_ag = aug.subset(shared)
        degen = np.array(
            [
                bool(ft.rows[i].get("t1_halves_degenerate"))
                or bool(plain.rows[j].get("t1_halves_degenerate"))
                for i, j in zip(i_ft, i_pl, strict=True)
            ]
        )
        keep = ~degen
        t1_ft = ft.kind("t1", layer)[i_ft][keep].astype(np.float64)
        t1_aug = aug.kind("t1", layer)[i_ag][keep].astype(np.float64)
        t1_pl = plain.kind("t1", layer)[i_pl][keep].astype(np.float64)
        odd = plain.kind("t1_odd", layer)[i_pl][keep].astype(np.float64)
        even = plain.kind("t1_even", layer)[i_pl][keep].astype(np.float64)
        rows_kept = [plain.rows[j] for j, k in zip(i_pl, keep, strict=True) if k]
        # PRIMARY: disjoint-baseline legs (shared capture noise cancels, §3)
        dv_ft = t1_ft - odd
        dv_ctx = t1_aug - even
        # COMPANION: naive shared-baseline read (inflation = E||e||^2 measure)
        dv_ft_sh = t1_ft - t1_pl
        dv_ctx_sh = t1_aug - t1_pl
        ctx_dv[slug] = {"dv_ctx": dv_ctx, "bases": [r["base_row_id"] for r in rows_kept]}
        cos_primary = _row_cosines(dv_ft, dv_ctx)
        cos_shared = _row_cosines(dv_ft_sh, dv_ctx_sh)
        # mean-only null: cos(dv_ft(x), mean dv_ctx)
        mean_ctx = dv_ctx.mean(axis=0, keepdims=True)
        cos_mean_only = _row_cosines(dv_ft, np.broadcast_to(mean_ctx, dv_ft.shape))
        # derangement null within topic stratum (batched einsum over draws)
        topics = np.array([r.get("topic") or "other" for r in rows_kept])
        rng = np.random.default_rng(args.seed)
        n = dv_ft.shape[0]
        d_draws = np.empty((args.n_derange, n), dtype=np.float64)
        for t in range(args.n_derange):
            perm = np.arange(n)
            for topic in np.unique(topics):
                tidx = np.where(topics == topic)[0]
                if len(tidx) > 1:
                    shuffled = rng.permutation(tidx)
                    # re-draw until no fixed point (derangement; bounded tries)
                    for _ in range(20):
                        if not np.any(shuffled == tidx):
                            break
                        shuffled = rng.permutation(tidx)
                    perm[tidx] = shuffled
            d_draws[t] = _row_cosines(dv_ft, dv_ctx[perm])
        # per-row reliability from the ctx-arm token halves
        rel_ctx = _row_cosines(t1_aug - odd, t1_aug - even)
        # R^2 of dv_ft on dv_ctx (pooled OLS over rows x dims)
        beta = float(
            np.einsum("nd,nd->", dv_ft, dv_ctx) / max(np.einsum("nd,nd->", dv_ctx, dv_ctx), 1e-12)
        )
        resid = dv_ft - beta * dv_ctx
        r2_pooled = 1.0 - float((resid**2).sum()) / max(
            float(((dv_ft - dv_ft.mean(axis=0)) ** 2).sum()), 1e-12
        )
        prefixes = [str(r.get(args.group_key)) for r in rows_kept]
        out[slug] = {
            "n_rows": int(n),
            "n_degenerate_excluded": int(degen.sum()),
            "cos_primary": _grouped_bootstrap_ci(
                cos_primary, prefixes, n_boot=args.n_boot, seed=args.seed
            ),
            "cos_shared_companion": _grouped_bootstrap_ci(
                cos_shared, prefixes, n_boot=args.n_boot, seed=args.seed
            ),
            "cos_mean_only_null_mean": float(cos_mean_only.mean()),
            "derangement_null": {
                "n_draws": args.n_derange,
                "mean": float(d_draws.mean()),
                "q95_of_draw_means": float(np.quantile(d_draws.mean(axis=1), 0.95)),
                "q05_of_draw_means": float(np.quantile(d_draws.mean(axis=1), 0.05)),
            },
            "excess_over_mean_only": float(cos_primary.mean() - cos_mean_only.mean()),
            "beta_pooled": beta,
            "r2_pooled": r2_pooled,
            "reliability_ctx_halves_mean": float(rel_ctx.mean()),
            "per_row": {
                "base_row_id": [r["base_row_id"] for r in rows_kept],
                "cos_primary": [float(x) for x in cos_primary],
                "cos_shared": [float(x) for x in cos_shared],
            },
        }
    # shared-x cross-augmentation null (pair dv_ft(x;k) with dv_ctx(x;k'!=k))
    for slug in slugs:
        others = [s for s in slugs if s != slug]
        if not others:
            out[slug]["cross_aug_null"] = "n/a — single FT run (smoke)"
            continue
        ft = CellData(summaries_dir, f"cell_ft_{slug}")
        vals = []
        base_set = set(out[slug]["per_row"]["base_row_id"])
        my_cos = dict(zip(out[slug]["per_row"]["base_row_id"], out[slug]["per_row"]["cos_primary"]))
        for other in others:
            ob = ctx_dv[other]["bases"]
            common = [b for b in ob if b in base_set]
            if not common:
                continue
            aug_o = ctx_dv[other]["dv_ctx"]
            o_idx = {b: i for i, b in enumerate(ob)}
            plain_idx = plain.subset(common)
            odd_c = plain.kind("t1_odd", layer)[plain_idx].astype(np.float64)
            ft_idx = ft.subset(common)
            t1f = ft.kind("t1", layer)[ft_idx].astype(np.float64)
            dvf = t1f - odd_c
            dvo = np.stack([aug_o[o_idx[b]] for b in common])
            vals.extend(_row_cosines(dvf, dvo).tolist())
        out[slug]["cross_aug_null"] = {
            "mean": float(np.mean(vals)) if vals else None,
            "n_pairs": len(vals),
            "matched_minus_cross": (
                float(np.mean([my_cos[b] for b in base_set]) - np.mean(vals)) if vals else None
            ),
        }
    return out


def unit_q6(args, summaries_dir: Path, layer: int, arm: str) -> dict:
    """Q6 map stability: M_plain on the FT model's (c,v) + refit ceiling."""
    cells_present = _cells_present(summaries_dir)
    plain = CellData(summaries_dir, "cell_plain")
    Xp = plain.kind(arm, layer).astype(np.float64)
    Yp = plain.kind("t1", layer).astype(np.float64)
    import torch

    out: dict[str, dict] = {}
    slugs = [s for s in DISTILL_RUNS if f"cell_ft_{s}" in cells_present]
    for slug in slugs:
        ft = CellData(summaries_dir, f"cell_ft_{slug}")
        Xf = ft.kind(arm, layer).astype(np.float64)
        Yf = ft.kind("t1", layer).astype(np.float64)
        folds = _folds_from_manifest(
            ft.rows, Xf.shape[0], group_key=args.group_key, n_folds=args.n_folds
        )
        # M_plain applied (fit on ALL plain rows once, predict FT rows)
        res = press_fit_predict(
            torch.from_numpy(Xp).double(),
            torch.from_numpy(Yp).double(),
            torch.from_numpy(Xf).double(),
            standardize=True,
        )
        pred_plain_map = res["pred"].detach().cpu().numpy()
        # refit-on-theta_k ceiling (grouped CV on the FT cell itself)
        refit = _fit_cv(Xf, Yf, folds)
        out[slug] = {
            "n_rows": int(Xf.shape[0]),
            "r2_plain_map_on_ft": _r2(Yf, pred_plain_map),
            "r2_refit_ceiling": refit["r2"],
            "refit_lambda_indices": refit["lambda_indices"],
        }
    # matched-n plain fit (n = FT eval size) — Q6 comparator (§4.5 item 4.6)
    if slugs:
        n_match = CellData(summaries_dir, f"cell_ft_{slugs[0]}").kind("t1", layer).shape[0]
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(Xp.shape[0], size=min(n_match, Xp.shape[0]), replace=False)
        rows_m = [plain.rows[i] for i in idx]
        folds_m = _folds_from_manifest(
            rows_m, len(idx), group_key=args.group_key, n_folds=args.n_folds
        )
        fit_m = _fit_cv(Xp[idx], Yp[idx], folds_m)
        out["_plain_matched_n"] = {"n": int(len(idx)), "r2": fit_m["r2"]}
    # cross-provision drift bound from the plain re-captures
    if "cell_plain_recap_b" in cells_present:
        recap = CellData(summaries_dir, "cell_plain_recap_b")
        shared = [b for b in (r["base_row_id"] for r in recap.rows) if b in plain.by_base]
        i_r = recap.subset(shared)
        i_p = plain.subset(shared)
        cos = _row_cosines(
            recap.kind("t1", layer)[i_r].astype(np.float64),
            plain.kind("t1", layer)[i_p].astype(np.float64),
        )
        out["_recap_drift"] = {
            "n_rows": len(shared),
            "t1_cos_mean": float(cos.mean()),
            "t1_cos_min": float(cos.min()),
        }
    return out


def unit_loto(args, summaries_dir: Path, layer: int, arm: str) -> dict:
    """Registered LOTO sensitivity: plain diagonal + plain->facts transfer."""
    saved_group_key = args.group_key
    try:
        args.group_key = "topic"
        res = unit_transfer(args, summaries_dir, layer, arm, "ambient")
        keep = {
            "matrix_plain_row": res["matrix"].get("plain"),
            "diag_plain": res["diag"].get("plain"),
            "group_key": "topic",
        }
        return keep
    finally:
        args.group_key = saved_group_key


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _cells_present(summaries_dir: Path) -> set[str]:
    if not summaries_dir.exists():
        raise FileNotFoundError(f"summaries dir missing: {summaries_dir}")
    return {p.name for p in summaries_dir.iterdir() if p.is_dir()}


def _default_units(args) -> list[str]:
    layers_pca = [14, 18, 19]
    units = []
    for arm in ("prefix_end", "context_end"):
        units.append(f"transfer:{arm}:ambient:L14")
        for layer in layers_pca:
            units.append(f"transfer:{arm}:pca48:L{layer:02d}")
    for c_kind in ("context_end", "context_mean", "prefix_end", "prefix_mean"):
        units.append(f"shifts:{c_kind}:L14")
    units += [
        "gating:context_end:L14",
        "gating:prefix_end:L14",
        "q4:L14",
        "q6:context_end:L14",
        "q6:prefix_end:L14",
        "loto:context_end:L14",
    ]
    if args.smoke:
        units = [
            "transfer:context_end:pca48:L14",
            "shifts:context_end:L14",
            "gating:context_end:L14",
        ]
    return units


def _run_unit(args, unit: str, summaries_dir: Path) -> dict:
    parts = unit.split(":")
    kind = parts[0]
    layer = int(parts[-1].removeprefix("L"))
    if kind == "transfer":
        return unit_transfer(args, summaries_dir, layer, parts[1], parts[2])
    if kind == "shifts":
        return unit_shifts(args, summaries_dir, layer, parts[1])
    if kind == "gating":
        return unit_gating(args, summaries_dir, layer, parts[1])
    if kind == "q4":
        return unit_q4(args, summaries_dir, layer)
    if kind == "q6":
        return unit_q6(args, summaries_dir, layer, parts[1])
    if kind == "loto":
        return unit_loto(args, summaries_dir, layer, parts[1])
    raise ValueError(f"unknown unit {unit!r}")


def _regime_fingerprint(args) -> str:
    keys = {
        "smoke": args.smoke,
        "n_random_draws": args.n_random_draws,
        "n_refit_twins": args.n_refit_twins,
        "n_perm": args.n_perm,
        "n_boot": args.n_boot,
        "n_derange": args.n_derange,
        "n_folds": args.n_folds,
        "group_key": args.group_key,
        "seed": args.seed,
        "rb_rev": args.rb_rev,
    }
    return hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:12]


def _g2_check(args, out_dir: Path, unit: str, payload: dict) -> None:
    """G2/K1 fit-substrate kill (plan §7): plain diagonal at the headline unit."""
    if not unit.startswith("transfer:context_end:ambient:L14"):
        return
    r2 = payload.get("matrix", {}).get("plain", {}).get("plain")
    if r2 is None:
        return
    if r2 < G2_R2_FLOOR:
        report = {
            "gate": "G2",
            "r2_plain_diag": r2,
            "floor": G2_R2_FLOOR,
            "unit": unit,
            "verdict": "informational-smoke" if args.smoke else "KILL",
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        }
        _atomic_json(out_dir / "g2_report.json", report)
        if args.smoke:
            logger.warning("[G2] plain diag R2=%.3f < %.2f (smoke: informational)", r2, G2_R2_FLOOR)
            return
        logger.error(
            "[G2 KILL] plain diag R2=%.3f < %.2f — stop P6, surface for re-plan", r2, G2_R2_FLOOR
        )
        raise SystemExit(21)  # distinct rc: designed halt, not an anonymous crash
    logger.info("[G2] plain diag R2=%.3f >= %.2f (pass)", r2, G2_R2_FLOOR)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summaries-dir", default="data/issue_1489/summaries")
    p.add_argument("--out", default="eval_results/issue_1489/p6")
    p.add_argument("--units", default="", help="comma-separated unit ids (default: full grid)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--n-random-draws", type=int, default=200)
    p.add_argument("--n-refit-twins", type=int, default=20)
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--n-derange", type=int, default=200)
    p.add_argument("--n-folds", type=int, default=6)
    p.add_argument("--group-key", default="prefix_id")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rb-rev", default=DEFAULT_RB_REV)
    p.add_argument("--mlp-device", default="cpu")
    p.add_argument("--plan-hours-per-box", type=float, default=G3_PLAN_HOURS_PER_BOX)
    args = p.parse_args()
    if args.smoke:
        args.n_random_draws = min(args.n_random_draws, 20)
        args.n_refit_twins = min(args.n_refit_twins, 3)
        args.n_perm = min(args.n_perm, 20)
        args.n_boot = min(args.n_boot, 50)
        args.n_derange = min(args.n_derange, 20)

    summaries_dir = Path(args.summaries_dir)
    out_dir = Path(args.out)
    units = [u.strip() for u in args.units.split(",") if u.strip()] or _default_units(args)
    fp = _regime_fingerprint(args)
    logger.info("[p6] %d units, regime=%s smoke=%s", len(units), fp, args.smoke)

    pilot_recorded = False
    for i, unit in enumerate(units):
        unit_path = out_dir / "units" / f"{unit.replace(':', '__')}.json"
        if unit_path.exists():
            existing = json.loads(unit_path.read_text())
            if existing.get("regime") == fp:
                logger.info("[p6] %s resume-skip", unit)
                continue
            logger.info(
                "[p6] %s regime changed (%s -> %s); recomputing", unit, existing.get("regime"), fp
            )
        t0 = time.monotonic()
        payload = _run_unit(args, unit, summaries_dir)
        wall_s = time.monotonic() - t0
        record = {
            "unit": unit,
            "regime": fp,
            "wall_seconds": wall_s,
            "result": payload,
            "reproducibility": {
                "git_sha": _git_sha(),
                "timestamp_utc": datetime.datetime.utcnow().isoformat(),
                "seed": args.seed,
                "smoke": args.smoke,
            },
        }
        _atomic_json(unit_path, record)
        logger.info("[p6] %s done in %.1fs -> %s", unit, wall_s, unit_path)
        _g2_check(args, out_dir, unit, payload)
        if not pilot_recorded:
            projected_h = wall_s * len(units) / 3600.0
            pilot = {
                "gate": "G3",
                "first_unit": unit,
                "first_unit_wall_s": wall_s,
                "n_units": len(units),
                "projected_box_hours": projected_h,
                "plan_hours_per_box": args.plan_hours_per_box,
                "ratio": projected_h / max(args.plan_hours_per_box, 1e-9),
                "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            }
            _atomic_json(out_dir / "g3_pilot.json", pilot)
            if not args.smoke and pilot["ratio"] > 2.0:
                # §9 descope tier 1: drop L18/L19 pca48 sensitivity units
                dropped = [u for u in units[i + 1 :] if "pca48:L18" in u or "pca48:L19" in u]
                units = [u for u in units if u not in dropped]
                pilot["descope_dropped_units"] = dropped
                _atomic_json(out_dir / "g3_pilot.json", pilot)
                logger.warning(
                    "[G3] projected %.1fh/box > 2x plan %.1fh — descoped %d pca48 "
                    "sensitivity units (§9 tier 1)",
                    projected_h,
                    args.plan_hours_per_box,
                    len(dropped),
                )
            pilot_recorded = True
    logger.info(
        "[phase-complete] p6 units"
    )  # reserved [phase=done] stays the DISPATCHER terminal line
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
