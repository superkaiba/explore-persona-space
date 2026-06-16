#!/usr/bin/env python3
# ruff: noqa: RUF002
# (multiplication-sign / arrow / minus characters intentional in docstrings/labels)
"""Issue #649 — LEVEL/CHANGE predictor decomposition for sycophancy (CPU, no pod).

Tests whether #532's marker-established two-component rule — *base prior forecasts
the absolute trained LEVEL; activation geometry forecasts the training-induced
CHANGE; prior is the null on CHANGE* — holds for a realistic judged behavior
(sycophancy). Pure re-analysis of #612's on-policy judge-scored agreement rates +
freshly extracted early-layer geometry (Phase 1, ``issue649_extract_panel_earlylayer.py``).

Pipeline (all CPU):
  Phase 0  prefetch + content-pin the #612 trained-judgment cells from the HF data
           repo (per (arm, source, seed, bystander) ``rate``), record an
           EXPECTED_SHA256 manifest, snapshot under ``data/issue_649/inputs/``.
  Phase 2  build the per-(arm, source, seed, bystander) DV+predictor table:
           LEVEL = t, CHANGE = t - b (b = on-disk base rate), predictors b,
           cosine@L2_eos (primary), cosine@L7_lp (secondary), Gaussian-KL@early band.
           Exclude the source==bystander diagonal + per-source trained negatives
           (panel_set ``neg_member_for``). Average seeds 42+137; per-seed sign agreement.
  Phase 3  six-regression CV-R² ladder per DV (M0..M5), BYSTANDER-grouped CV
           (headline; keeps every source in training so M0's per-source one-hots
           are identifiable), source-grouped CV (robustness/generalization);
           marginal-Spearman table (6 rows) with 1000-rep bootstrap 95% CIs;
           collinearity gate (Pearson |cosine_L2|, prior); #391 forced-choice S/N gate.
  Phase 4  figures (CV-R² ladder bars LEVEL/CHANGE; prior/geometry scatter quads;
           raw-alongside-residualized; per-seed; collinearity; L20-vs-early-band).

Outputs:
  eval_results/issue_649/cv_r2_ladder.json     (per arm x per DV; 6 models each)
  eval_results/issue_649/marginal_spearman.json (6 rows per arm, bootstrap 95% CIs)
  eval_results/issue_649/analysis.json          (collinearity gate, #391 gate, coverage)
  figures/issue_649/*.png + figures/issue_649/meta.json

``--smoke``: 1 source (villain) x 3 bystanders x 1 seed (42) x arm_canned through the
SAME code path; degenerate 3-fold bystander-grouped CV + 100-rep bootstrap (smoke value).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="i649_level_change_decomp")

import numpy as np  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue612_sycophancy_onpolicy"
PANEL_CENTROIDS_L20_REPO_PATH = f"{HF_PREFIX}/panel/panel_centroids_layer20.pt"
L20_ROBUST_LAYER = 20  # the on-HF late-layer cosine cell (#509: the WEAK sycophancy cell)
SOURCES_CANNED = ("villain", "comedian", "kindergarten_teacher", "software_engineer")
SOURCES_ONPOLICY = ("villain", "comedian")
SEEDS = (42, 137)
PCA_K = 16  # the #502 winner (NOT k=8); matches issue532_predictor_stress.PCA_K
N_BOOT_FULL = 1000  # matches #532
N_BOOT_SMOKE = 100
BOOT_SEED = 42
# On-disk base rates (the bystander prior) — repo-root path, NOT cwd-relative.
BASE_JUDGMENTS_DIR = PROJECT_ROOT / "eval_results" / "issue_612" / "base" / "judgments"


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _git_commit_sha() -> str:
    import os
    import subprocess

    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env={**os.environ},  # explicit per the subprocess-env rule
        check=False,
    )
    return out.stdout.strip() or "unknown"


def _repro_metadata() -> dict[str, Any]:
    import platform

    return {
        "git_commit": _git_commit_sha(),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "timestamp": _now_iso(),
    }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.rename(path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_per_cell_csv(path: Path, tables: dict[str, dict]) -> None:
    """Full per-(arm, source, bystander) row table for auditing (plan §6.5)."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "arm",
        "source",
        "bystander",
        "level",
        "change",
        "base_prior",
        "cos_L2_eos",
        "cos_L7_lp",
        "cos_L20_robust",
        "kl_L2",
        "kl_L7",
        "t_seed_42",
        "t_seed_137",
        "n_seeds",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for arm, tbl in tables.items():
            for r in tbl["rows"]:
                w.writerow({"arm": arm, **{k: r.get(k) for k in fields if k != "arm"}})


# ===========================================================================
# Statistics helpers — vendored VERBATIM from
# scripts/issue532_predictor_stress.py (the live impl) so #649 is self-contained.
# ===========================================================================
def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (single-pass, NaN-safe)."""
    from scipy.stats import spearmanr

    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return float("nan")
    r, _ = spearmanr(x[mask], y[mask])
    return float(r)


def _bootstrap_spearman_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap 95% CI on Spearman ρ via simple resampling."""
    rng = np.random.default_rng(seed)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rhos.append(_spearman_rho(x[idx], y[idx]))
    rhos = np.array(rhos)
    return (
        float(np.nanmean(rhos)),
        float(np.nanpercentile(rhos, 2.5)),
        float(np.nanpercentile(rhos, 97.5)),
    )


def _cv_r2_grouped(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Grouped held-out R² for an OLS fit; up to 5-fold grouped CV.

    Vendored from ``issue532_predictor_stress._cv_r2_loco`` (the live #532
    impl) with ``classes`` renamed ``groups`` to reflect #649's design: the
    grouping vector is the BYSTANDER id for the headline (keeps every source in
    every training fold so M0's per-source one-hots are fit + identifiable),
    or the SOURCE id for the generalization robustness read. Fold count is
    ``min(5, n_unique_groups)``, so this is leave-one-group-out CV when
    n_groups <= 5 and 5-fold grouped CV when n_groups > 5.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold

    mask = ~np.isnan(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mask = mask & ~np.any(np.isnan(X), axis=1)
    X = X[mask]
    y = y[mask]
    groups = groups[mask]
    n_unique = len(np.unique(groups))
    if n_unique < 2 or len(y) < 5:
        return float("nan")
    n_splits = min(5, n_unique)
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.zeros_like(y)
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        m = LinearRegression()
        m.fit(X[train_idx], y[train_idx])
        preds[test_idx] = m.predict(X[test_idx])
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot < 1e-18:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _gaussian_sym_kl_in_subspace_local(Xa: np.ndarray, Xb: np.ndarray, k: int) -> float:
    """Gaussian symmetric-KL between two clouds in the top-k PCA subspace.

    Vendored VERBATIM from
    ``scripts/issue532_predictor_stress.py::_gaussian_sym_kl_in_subspace_local``
    (the live impl; the bare ``_gaussian_sym_kl_in_subspace`` lives only in
    issue493). PCA subspace via the Gram/dual trick (n << d).
    """
    Xa = Xa[~np.any(np.isnan(Xa), axis=1)]
    Xb = Xb[~np.any(np.isnan(Xb), axis=1)]
    if len(Xa) < 2 or len(Xb) < 2:
        return float("nan")
    stacked = np.vstack([Xa, Xb])
    mu = stacked.mean(axis=0, keepdims=True)
    stacked_c = stacked - mu
    n, d = stacked_c.shape
    k_eff = min(k, n, d)
    G = stacked_c @ stacked_c.T
    G = 0.5 * (G + G.T)
    eigvals, eigvecs = np.linalg.eigh(G)
    order = np.argsort(eigvals)[::-1][:k_eff]
    lam = np.clip(eigvals[order], 1e-12, None)
    V_g = eigvecs[:, order]
    sqrt_lam = np.sqrt(lam)
    components = (stacked_c.T @ V_g) / sqrt_lam[None, :]  # (d, k)
    Ya = (Xa - mu) @ components
    Yb = (Xb - mu) @ components
    mu_a = Ya.mean(axis=0)
    mu_b = Yb.mean(axis=0)
    Sa = np.cov(Ya.T, ddof=1) + 1e-6 * np.eye(Ya.shape[1])
    Sb = np.cov(Yb.T, ddof=1) + 1e-6 * np.eye(Yb.shape[1])

    def _one_kl(S0, S1, m0, m1):
        S1_inv = np.linalg.inv(S1)
        sign0, logdet0 = np.linalg.slogdet(S0)
        sign1, logdet1 = np.linalg.slogdet(S1)
        if sign0 <= 0 or sign1 <= 0:
            return float("nan")
        d_inner = S0.shape[0]
        return 0.5 * (
            np.trace(S1_inv @ S0) + (m1 - m0) @ S1_inv @ (m1 - m0) - d_inner + (logdet1 - logdet0)
        )

    kl_ab = _one_kl(Sa, Sb, mu_a, mu_b)
    kl_ba = _one_kl(Sb, Sa, mu_b, mu_a)
    if np.isnan(kl_ab) or np.isnan(kl_ba):
        return float("nan")
    return float(0.5 * (kl_ab + kl_ba))


def _wilson_half_width(p: float, n: int, z: float = 1.96) -> float:
    """Wilson score-interval half-width for a proportion (cluster-honest n)."""
    if n <= 0:
        return float("nan")
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # half-width of the symmetric-ish score interval reported about the center
    lo, hi = center - margin, center + margin
    return float((hi - lo) / 2.0)


# ===========================================================================
# Geometry — centered bank cosine + Gaussian-KL from the Phase-1 .npz
# ===========================================================================
def _centered_bank_cosine(centroids: np.ndarray) -> np.ndarray:
    """Global mean-center -> L2-normalize -> cosine (the canonical persona-distance
    bank-cosine, ``.claude/rules/persona-distance-metrics.md`` § Bank centering).
    Delegates to the project helper so the recipe stays single-sourced."""
    import torch

    from explore_persona_space.analysis.representation_shift import compute_cosine_matrix

    C = torch.from_numpy(centroids.astype(np.float32))
    return compute_cosine_matrix(C, centering="global_mean").numpy()


def _load_l20_robust_cosine(
    persona_names: list[str], inputs_dir: Path
) -> tuple[np.ndarray | None, str | None]:
    """Centered-bank cosine over the on-HF ``panel_centroids_layer20.pt`` L20
    centroids (the #509 WEAK sycophancy cell, carried as a ROBUSTNESS cosine cell
    only). The .pt indexes a 52-persona superset; slice to the panel order, then
    global-mean-center over the panel bank (NOT the 52-bank, so the centered cosine
    is comparable to the early-band cosines computed on the same 30-persona bank).

    Returns (cosine_matrix shape (P,P) | None, note). Fail-soft: a missing/stale
    .pt yields (None, reason) so the robustness cell is dropped, never crashing the
    headline (L20 is explicitly robustness-only per the plan)."""
    import torch
    from huggingface_hub import hf_hub_download

    local = inputs_dir / "panel_centroids_layer20.pt"
    try:
        if not local.exists():
            inputs_dir.mkdir(parents=True, exist_ok=True)
            src = hf_hub_download(
                DATA_REPO, PANEL_CENTROIDS_L20_REPO_PATH, repo_type="dataset", revision="main"
            )
            local.write_bytes(Path(src).read_bytes())
        obj = torch.load(local, map_location="cpu", weights_only=False)
        all_names = list(obj["persona_names"])
        cents = obj["centroids"][L20_ROBUST_LAYER]  # (52, H)
        name_to_row = {n: i for i, n in enumerate(all_names)}
        missing = [n for n in persona_names if n not in name_to_row]
        if missing:
            return None, f"L20 centroids missing personas: {missing[:5]}"
        rows = np.stack(
            [cents[name_to_row[n]].float().numpy().astype(np.float32) for n in persona_names]
        )
        cos = _centered_bank_cosine(rows)
        return cos, None
    except Exception as exc:  # robustness-only cell — never block the headline
        logger.warning("L20 robustness cosine unavailable: %s (dropping the robustness cell)", exc)
        return None, repr(exc)


def load_geometry(npz_path: Path, inputs_dir: Path) -> dict[str, Any]:
    """Load the Phase-1 geometry .npz and build the per-(source, bystander)
    cosine + Gaussian-KL lookups. Returns persona_names + matrices.

    Cosine cells (centered bank cosine, global-mean, .claude/rules/persona-distance-metrics.md):
      - cos_L2_eos  : end_of_system L2 centroid (PRIMARY).
      - cos_L7_lp   : last_prompt L7 cloud-mean (SECONDARY).
      - cos_L20_robust : on-HF panel_centroids_layer20.pt (ROBUSTNESS-only; #509 WEAK cell).
    Gaussian-KL cells (cloud metric — end_of_system has no cloud, so KL rides the
    last_prompt clouds at the two early-band layers):
      - kl_L2 (@ early-band primary layer), kl_L7 (@ early-band secondary layer)."""
    with np.load(npz_path, allow_pickle=True) as z:
        eos_L2 = z["eos_L2_centroid"].astype(np.float32)  # (P, H)
        lp_L2 = z["lastprompt_L2_cloud"].astype(np.float32)  # (P, n_probes, H)
        lp_L7 = z["lastprompt_L7_cloud"].astype(np.float32)  # (P, n_probes, H)
        meta = json.loads(str(z["meta_json"]))
    names: list[str] = list(meta["persona_names"])
    idx = {n: i for i, n in enumerate(names)}
    assert eos_L2.shape[0] == len(names), eos_L2.shape
    # Cosine: primary = end_of_system L2 centroid; secondary = last_prompt L7 cloud-mean.
    lp_L7_centroid = lp_L7.mean(axis=1)  # (P, H)
    cos_L2_eos = _centered_bank_cosine(eos_L2)
    cos_L7_lp = _centered_bank_cosine(lp_L7_centroid)
    cos_L20_robust, l20_note = _load_l20_robust_cosine(names, inputs_dir)
    # Gaussian-KL on the early-band clouds (last_prompt; end_of_system has no cloud).
    k = int(meta.get("gkl_k", PCA_K))
    n = len(names)
    kl_L2 = np.zeros((n, n), dtype=np.float64)
    kl_L7 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            kl_L2[i, j] = kl_L2[j, i] = _gaussian_sym_kl_in_subspace_local(lp_L2[i], lp_L2[j], k)
            kl_L7[i, j] = kl_L7[j, i] = _gaussian_sym_kl_in_subspace_local(lp_L7[i], lp_L7[j], k)
    return {
        "persona_names": names,
        "idx": idx,
        "cos_L2_eos": cos_L2_eos,
        "cos_L7_lp": cos_L7_lp,
        "cos_L20_robust": cos_L20_robust,  # may be None (robustness-only; fail-soft)
        "cos_L20_note": l20_note,
        "kl_L2": kl_L2,
        "kl_L7": kl_L7,
        "kl_nan_L2": int(np.isnan(kl_L2).sum()),
        "kl_nan_L7": int(np.isnan(kl_L7).sum()),
        "meta": meta,
        "centering": "global_mean",
    }


# ===========================================================================
# Phase 0 — prefetch + content-pin trained-judgment cells
# ===========================================================================
def prefetch_cells(
    arms: list[str],
    sources_for_arm: dict[str, tuple[str, ...]],
    bystanders: list[str],
    seeds: tuple[int, ...],
    inputs_dir: Path,
) -> dict[str, Any]:
    """Download the needed trained-judgment cell JSONs into the issue-owned
    inputs snapshot, pinned to the data-repo head SHA. Returns an
    EXPECTED_SHA256 manifest + the data-repo revision."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.dataset_info(DATA_REPO, revision="main")
    head_sha = info.sha
    manifest: dict[str, str] = {}
    cells_root = inputs_dir / "judgments" / "cells"
    cells_root.mkdir(parents=True, exist_ok=True)
    n_dl = 0
    for arm in arms:
        for src in sources_for_arm[arm]:
            for seed in seeds:
                for by in bystanders:
                    rel = f"{HF_PREFIX}/judgments/cells/{arm}/{src}/seed_{seed}/judgments/{by}.json"
                    local = cells_root / arm / src / f"seed_{seed}" / f"{by}.json"
                    if not local.exists():
                        src_path = hf_hub_download(
                            DATA_REPO, rel, repo_type="dataset", revision=head_sha
                        )
                        local.parent.mkdir(parents=True, exist_ok=True)
                        local.write_bytes(Path(src_path).read_bytes())
                        n_dl += 1
                    manifest[str(local.relative_to(inputs_dir))] = _sha256_file(local)
    logger.info(
        "prefetch: %d cells (%d newly downloaded), data-repo sha %s",
        len(manifest),
        n_dl,
        head_sha[:12],
    )
    return {"data_repo_revision": head_sha, "expected_sha256": manifest, "n_cells": len(manifest)}


def _trained_rate_path(inputs_dir: Path, arm: str, src: str, seed: int, by: str) -> Path:
    return inputs_dir / "judgments" / "cells" / arm / src / f"seed_{seed}" / f"{by}.json"


def _read_trained_rate(inputs_dir: Path, arm: str, src: str, seed: int, by: str) -> float:
    """Strict reader — raises FileNotFoundError if the cell JSON is absent."""
    p = _trained_rate_path(inputs_dir, arm, src, seed, by)
    jd = json.loads(p.read_text())
    assert jd.get("n_verdicts") == 600, (p, jd.get("n_verdicts"))
    return float(jd["rate"])


def _read_base_rate(by: str) -> float | None:
    p = BASE_JUDGMENTS_DIR / f"{by}.json"
    if not p.exists():
        return None
    jd = json.loads(p.read_text())
    return float(jd["rate"])


# ===========================================================================
# Phase 2 — build the per-(arm, source, seed, bystander) DV+predictor table
# ===========================================================================
def build_table(
    arm: str,
    sources: tuple[str, ...],
    bystanders: list[str],
    seeds: tuple[int, ...],
    panel_set: dict,
    geom: dict[str, Any],
    inputs_dir: Path,
) -> dict[str, Any]:
    """Long-format table over (source, bystander) cells (seeds averaged).

    Excludes the source==bystander diagonal + per-source trained-negative
    bystanders (panel_set ``neg_member_for``). Returns equal-length arrays +
    a per-seed sign-agreement record + coverage.
    """
    personas = panel_set["personas"]
    idx = geom["idx"]
    rows: list[dict[str, Any]] = []
    per_seed_signs: list[int] = []  # +1 if both seeds' (t-b) agree in sign, else 0
    excluded = {
        "diagonal": 0,
        "trained_negative": 0,
        "missing_base": 0,
        "missing_geom": 0,
        "missing_trained": 0,
    }
    # When the inputs snapshot is real (not a monkeypatched test stub), only read
    # cell JSONs that actually exist on disk; a genuinely missing cell is TRACKED
    # (missing_trained), never silently swallowed via try/except.
    snapshot_present = (inputs_dir / "judgments" / "cells").exists()

    for src in sources:
        for by in bystanders:
            if by == src:
                excluded["diagonal"] += 1
                continue
            neg_for = personas.get(by, {}).get("neg_member_for", []) or []
            if src in neg_for:
                excluded["trained_negative"] += 1
                continue
            if by not in idx or src not in idx:
                excluded["missing_geom"] += 1
                continue
            b = _read_base_rate(by)
            if b is None:
                excluded["missing_base"] += 1
                continue
            # per-seed trained rates -> average; sign agreement on (t - b).
            # Read each seed's cell only when present (explicit existence check —
            # no try/except swallowing); a genuinely missing cell is tracked.
            t_seeds = []
            t_by_seed: dict[int, float] = {}
            for seed in seeds:
                if (
                    snapshot_present
                    and not _trained_rate_path(inputs_dir, arm, src, seed, by).exists()
                ):
                    continue
                r = _read_trained_rate(inputs_dir, arm, src, seed, by)
                t_seeds.append(r)
                t_by_seed[seed] = r
            if not t_seeds:
                excluded["missing_trained"] += 1
                continue
            t = float(np.mean(t_seeds))
            if len(t_seeds) == 2:
                s0, s1 = (t_seeds[0] - b), (t_seeds[1] - b)
                per_seed_signs.append(1 if (np.sign(s0) == np.sign(s1)) else 0)
            si, bi = idx[src], idx[by]
            cos_l20 = geom.get("cos_L20_robust")
            rows.append(
                {
                    "source": src,
                    "bystander": by,
                    "level": t,
                    "change": t - b,
                    "base_prior": b,
                    "cos_L2_eos": float(geom["cos_L2_eos"][si, bi]),
                    "cos_L7_lp": float(geom["cos_L7_lp"][si, bi]),
                    "cos_L20_robust": (
                        float(cos_l20[si, bi]) if cos_l20 is not None else float("nan")
                    ),
                    "kl_L2": float(geom["kl_L2"][si, bi]),
                    "kl_L7": float(geom["kl_L7"][si, bi]),
                    "t_seed_42": float(t_by_seed.get(42, float("nan"))),
                    "t_seed_137": float(t_by_seed.get(137, float("nan"))),
                    "n_seeds": len(t_seeds),
                }
            )

    def col(name: str) -> np.ndarray:
        return np.array([r[name] for r in rows], dtype=np.float64)

    source_ids = [r["source"] for r in rows]
    bystander_ids = [r["bystander"] for r in rows]
    src_to_int = {s: i for i, s in enumerate(sorted(set(source_ids)))}
    by_to_int = {b: i for i, b in enumerate(sorted(set(bystander_ids)))}
    sign_agree = float(np.mean(per_seed_signs)) if per_seed_signs else float("nan")
    return {
        "arm": arm,
        "n_cells": len(rows),
        "rows": rows,
        "level": col("level"),
        "change": col("change"),
        "base_prior": col("base_prior"),
        "cos_L2_eos": col("cos_L2_eos"),
        "cos_L7_lp": col("cos_L7_lp"),
        "cos_L20_robust": col("cos_L20_robust"),
        "kl_L2": col("kl_L2"),
        "kl_L7": col("kl_L7"),
        "t_seed_42": col("t_seed_42"),
        "t_seed_137": col("t_seed_137"),
        "source_ids": source_ids,
        "bystander_ids": bystander_ids,
        "source_group": np.array([src_to_int[s] for s in source_ids]),
        "bystander_group": np.array([by_to_int[b] for b in bystander_ids]),
        "source_onehot": _one_hot(source_ids),
        "excluded": excluded,
        "per_seed_sign_agreement": sign_agree,
        "n_per_seed_pairs": len(per_seed_signs),
    }


def _one_hot(labels: list[str]) -> np.ndarray:
    uniq = sorted(set(labels))
    m = {u: i for i, u in enumerate(uniq)}
    out = np.zeros((len(labels), len(uniq)), dtype=np.float64)
    for i, lab in enumerate(labels):
        out[i, m[lab]] = 1.0
    return out


# ===========================================================================
# Phase 3 — six-regression CV-R² ladder + marginal Spearman + gates
# ===========================================================================
def _z(v: np.ndarray) -> np.ndarray:
    return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)


def six_regression_ladder(
    table: dict[str, Any], dv: str, groups: np.ndarray, geom_predictor: str = "cos_L2_eos"
) -> dict[str, Any]:
    """The #532 six-model CV-R² ladder on one DV under a chosen CV grouping.

    M0 source-indicators / M1 +prior / M2 +prior+cosine / M3 cosine-only
    / M4 +prior+KL / M5 KL-only. Geometry arm uses ``geom_predictor`` for the
    cosine cell (cos_L2_eos primary) and ``kl_L2`` for the KL cell.
    """
    y = table[dv]
    onehot = table["source_onehot"]  # (n, n_sources)
    prior = _z(table["base_prior"]).reshape(-1, 1)
    cos = _z(table[geom_predictor]).reshape(-1, 1)
    kl = _z(table["kl_L2"]).reshape(-1, 1)
    intercept_warning = onehot.shape[1] < 2

    def r2(cols: list[np.ndarray]) -> float:
        X = np.concatenate(cols, axis=1)
        return _cv_r2_grouped(X, y, groups)

    m0 = r2([onehot])
    m1 = r2([onehot, prior])
    m2 = r2([onehot, prior, cos])
    m3 = r2([onehot, cos])
    m4 = r2([onehot, prior, kl])
    m5 = r2([onehot, kl])

    def d(a: float, b: float) -> float:
        return a - b if not (np.isnan(a) or np.isnan(b)) else float("nan")

    return {
        "dv": dv,
        "geometry_predictor": geom_predictor,
        "n_rows": len(y),
        "n_sources": int(onehot.shape[1]),
        "single_source_warning": bool(intercept_warning),
        "M0_source_indicators": m0,
        "M1_plus_prior": m1,
        "M2_plus_prior_cosine": m2,
        "M3_cosine_only": m3,
        "M4_plus_prior_kl": m4,
        "M5_kl_only": m5,
        "delta_prior_beyond_M0": d(m1, m0),  # ΔCV-R²(M1-M0)
        "delta_cosine_beyond_M1": d(m2, m1),  # ΔCV-R²(M2-M1)
        "delta_cosine_beyond_M0": d(m3, m0),  # ΔCV-R²(M3-M0)
        "delta_kl_beyond_M1": d(m4, m1),
        "delta_kl_beyond_M0": d(m5, m0),
    }


# The canonical headline predictor set (plan §6.5 "6 rows"): prior + the primary
# geometry cosine + the primary Gaussian-KL, each vs LEVEL and CHANGE. The L7 /
# L20 / kl_L7 cells are ALSO emitted (secondary + robustness), flagged `headline`.
_HEADLINE_PREDICTORS = (("prior", "base_prior"), ("cosine_L2", "cos_L2_eos"), ("kl_L2", "kl_L2"))
_SECONDARY_PREDICTORS = (
    ("cosine_L7", "cos_L7_lp"),
    ("kl_L7", "kl_L7"),
    ("cosine_L20_robust", "cos_L20_robust"),
)


def intercept_only_ladder(table: dict[str, Any], dv: str, groups: np.ndarray) -> dict[str, Any]:
    """Apples-to-apples generalization ladder for the SOURCE-grouped robustness read
    (plan §4 step 4 lines 81-82): under leave-one-source-out CV the per-source
    one-hot M0 is non-identifiable (the held-out source's dummy is all-zeros in the
    test fold and never fit), deflating M0 and inflating ΔCV-R²(M1-M0)/ΔCV-R²(M3-M0).
    This variant uses an INTERCEPT-ONLY M0 (no source indicator) so prior/geometry's
    Δ over it IS identifiable on a held-out source — the apples-to-apples generalization
    read. Bookkeeping, NOT a headline."""
    y = table[dv]
    n = len(y)
    intercept = np.ones((n, 1), dtype=np.float64)
    prior = _z(table["base_prior"]).reshape(-1, 1)
    cos = _z(table["cos_L2_eos"]).reshape(-1, 1)
    kl = _z(table["kl_L2"]).reshape(-1, 1)

    def r2(cols: list[np.ndarray]) -> float:
        return _cv_r2_grouped(np.concatenate(cols, axis=1), y, groups)

    m0 = r2([intercept])
    m1 = r2([intercept, prior])
    m3 = r2([intercept, cos])
    m5 = r2([intercept, kl])

    def d(a: float, b: float) -> float:
        return a - b if not (np.isnan(a) or np.isnan(b)) else float("nan")

    return {
        "dv": dv,
        "note": "intercept-only M0 (no source indicator); source-grouped CV; generalization read",
        "M0_intercept_only": m0,
        "M1_prior_over_intercept": m1,
        "M3_cosine_over_intercept": m3,
        "M5_kl_over_intercept": m5,
        "delta_prior_beyond_intercept": d(m1, m0),
        "delta_cosine_beyond_intercept": d(m3, m0),
        "delta_kl_beyond_intercept": d(m5, m0),
        "n_rows": n,
    }


def marginal_spearman_table(table: dict[str, Any], n_boot: int) -> list[dict[str, Any]]:
    """Marginal Spearman rows: the 3 HEADLINE predictors (prior / cosine_L2 / kl_L2 —
    the plan §6.5 "6 rows" = 3 predictors × {LEVEL, CHANGE}) PLUS the secondary /
    robustness cells (cosine_L7, kl_L7, cosine_L20_robust), each × {LEVEL, CHANGE},
    with 1000-rep bootstrap 95% CIs. Each row is flagged `headline: true|false`."""
    rows = []
    for dv_name, dv_key in (("LEVEL", "level"), ("CHANGE", "change")):
        y = table[dv_key]
        for is_headline, group in ((True, _HEADLINE_PREDICTORS), (False, _SECONDARY_PREDICTORS)):
            for pred_name, pred_key in group:
                # a predictor column may be absent (e.g. cos_L20_robust when the L20
                # .pt is unavailable) -> all-NaN -> a NaN row, flagged, never a crash.
                x = table.get(pred_key)
                if x is None:
                    x = np.full_like(y, np.nan)
                # an all-NaN robustness cell (e.g. L20 .pt unavailable) -> NaN row, flagged
                rho = _spearman_rho(x, y)
                mean, lo, hi = _bootstrap_spearman_ci(x, y, n_boot=n_boot, seed=BOOT_SEED)
                rows.append(
                    {
                        "predictor": pred_name,
                        "dv": dv_name,
                        "headline": is_headline,
                        "spearman_rho": rho,
                        "bootstrap_mean": mean,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "ci_covers_zero": bool(lo <= 0.0 <= hi)
                        if not (np.isnan(lo) or np.isnan(hi))
                        else None,
                    }
                )
    return rows


def _partial_spearman_controlling(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, n_boot: int
) -> dict[str, Any]:
    """Partial Spearman ρ(x, y | z): rank-residualize both x and y on z (OLS on
    ranks), then Spearman the residuals. Used for the non-circular read — prior vs
    the TRAINED RATE t with cosine partialled, and (the #509 read) cosine vs t with
    prior partialled — avoiding the `t-b`-contains-`b` mechanical artifact (plan §8
    risk row 3 / Assumption 8). Bootstrap 95% CI on the partial."""
    from scipy.stats import rankdata

    def _partial(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray) -> float:
        mask = ~(np.isnan(xx) | np.isnan(yy) | np.isnan(zz))
        if mask.sum() < 4:
            return float("nan")
        rx, ry, rz = rankdata(xx[mask]), rankdata(yy[mask]), rankdata(zz[mask])
        Z = np.column_stack([np.ones_like(rz), rz])
        bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
        by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
        ex, ey = rx - Z @ bx, ry - Z @ by
        if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
            return float("nan")
        return float(np.corrcoef(ex, ey)[0, 1])

    point = _partial(x, y, z)
    rng = np.random.default_rng(BOOT_SEED)
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    xm, ym, zm = x[mask], y[mask], z[mask]
    n = len(xm)
    boots = []
    if n >= 4:
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boots.append(_partial(xm[idx], ym[idx], zm[idx]))
    boots = np.array(boots) if boots else np.array([np.nan])
    lo, hi = float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))
    return {
        "partial_spearman": point,
        "ci95_low": lo,
        "ci95_high": hi,
        "ci_covers_zero": bool(lo <= 0.0 <= hi) if not (np.isnan(lo) or np.isnan(hi)) else None,
        "n": int(n),
    }


def noncircular_partials(table: dict[str, Any], n_boot: int) -> dict[str, Any]:
    """The non-circular reads (plan §8 risk row 3 / Assumption 8): because
    CHANGE = t - b mechanically contains b, ALSO report the source-FE-free partials
    ON THE TRAINED RATE t (= LEVEL) controlling the OTHER predictor:
      - prior vs t | cosine_L2  (does prior still predict the absolute level beyond geometry)
      - cosine_L2 vs t | prior  (the #509 read: does geometry predict t beyond prior)
    Reported ALONGSIDE the t-b reads, never instead of them."""
    t = table["level"]  # trained rate
    prior = table["base_prior"]
    cos = table["cos_L2_eos"]
    return {
        "prior_vs_t_given_cosine_L2": _partial_spearman_controlling(prior, t, cos, n_boot),
        "cosine_L2_vs_t_given_prior": _partial_spearman_controlling(cos, t, prior, n_boot),
        "note": (
            "Non-circular reads on the TRAINED RATE t (LEVEL), partialling the other "
            "predictor. t-b reads (which contain b by construction) are in the "
            "marginal_spearman + ladder; these partials avoid that mechanical artifact."
        ),
    }


def collinearity_gate(table: dict[str, Any]) -> dict[str, Any]:
    """Pre-registered: Pearson(|cos_L2_eos|, base_prior). |r| > 0.6 -> tercile-bucket
    median test + polynomial residualization fallback (reported regardless)."""
    from scipy.stats import pearsonr

    cos = table["cos_L2_eos"]
    prior = table["base_prior"]
    mask = ~(np.isnan(cos) | np.isnan(prior))
    if mask.sum() < 3:
        return {"pearson_abs_cos_prior": float("nan"), "fired": None, "n": int(mask.sum())}
    r, _ = pearsonr(np.abs(cos[mask]), prior[mask])
    fired = bool(abs(r) > 0.6)
    out: dict[str, Any] = {"pearson_abs_cos_prior": float(r), "fired": fired, "n": int(mask.sum())}
    if fired:
        # tercile-bucket median test on CHANGE by cosine tercile
        change = table["change"][mask]
        cm = np.abs(cos[mask])
        terc = np.quantile(cm, [1 / 3, 2 / 3])
        lo_b = change[cm <= terc[0]]
        hi_b = change[cm >= terc[1]]
        out["tercile_change_median_low_cos"] = float(np.median(lo_b)) if len(lo_b) else float("nan")
        out["tercile_change_median_high_cos"] = (
            float(np.median(hi_b)) if len(hi_b) else float("nan")
        )
        # polynomial residualization: residualize CHANGE on a degree-2 prior fit, re-correlate cos
        coeffs = np.polyfit(prior[mask], change, deg=2)
        resid = change - np.polyval(coeffs, prior[mask])
        out["poly_resid_change_vs_cos_spearman"] = _spearman_rho(cm, resid)
    return out


def forced_choice_gate(table: dict[str, Any]) -> dict[str, Any]:
    """#391 forced-choice S/N gate: fire iff median|CHANGE| < 2 × (cluster-honest
    Wilson half-width at the median per-cell trained rate). Cluster n = 60 claims."""
    change = table["change"]
    level = table["level"]
    med_abs_change = float(np.nanmedian(np.abs(change)))
    med_rate = float(np.nanmedian(level))
    hw = _wilson_half_width(med_rate, n=60)  # cluster-honest 60-claim cluster
    sn = med_abs_change / hw if hw and hw > 0 else float("nan")
    fired = bool(med_abs_change < 2.0 * hw) if not np.isnan(hw) else None
    return {
        "median_abs_change": med_abs_change,
        "median_trained_rate": med_rate,
        "cluster_honest_wilson_half_width": hw,
        "signal_to_noise": sn,
        "fired": fired,
        "note": (
            "fire => rate-space CHANGE too coarse; run #391 forced-choice probe (GPU). "
            "Expected NOT to fire (S/N ~ 2-3)."
        ),
    }


# ===========================================================================
# Phase 4 — figures
# ===========================================================================
def make_figures(
    ladders: dict[str, dict], tables: dict[str, dict], fig_dir: Path, smoke: bool
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    primary_arm = "arm_canned" if "arm_canned" in ladders else next(iter(ladders))

    # Hero 1: two-panel CV-R² ladder bars (LEVEL vs CHANGE) for the primary arm, bystander-grouped.
    bg = ladders[primary_arm]["bystander_grouped"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    model_keys = [
        "M0_source_indicators",
        "M1_plus_prior",
        "M2_plus_prior_cosine",
        "M3_cosine_only",
        "M4_plus_prior_kl",
        "M5_kl_only",
    ]
    labels = ["M0", "M1", "M2", "M3", "M4", "M5"]
    for ax, dv in zip(axes, ("level", "change"), strict=True):
        vals = [bg[dv][k] for k in model_keys]
        ax.bar(labels, [v if not np.isnan(v) else 0.0 for v in vals], color="#4C72B0")
        ax.set_title(f"{primary_arm} — {dv.upper()} CV-R² ladder (bystander-grouped)")
        ax.set_ylabel("held-out CV-R²")
        ax.axhline(0, color="k", lw=0.6)
    fig.tight_layout()
    p = fig_dir / "hero1_cv_r2_ladder_level_vs_change.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p.name)

    # Hero 2: prior/geometry vs DV scatter quad (raw, primary arm).
    tbl = tables[primary_arm]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (pk, dv) in zip(
        axes.flat,
        [
            ("base_prior", "level"),
            ("base_prior", "change"),
            ("cos_L2_eos", "level"),
            ("cos_L2_eos", "change"),
        ],
        strict=True,
    ):
        ax.scatter(tbl[pk], tbl[dv], s=22, alpha=0.7, color="#C44E52")
        ax.set_xlabel(pk)
        ax.set_ylabel(dv)
        ax.set_title(f"{pk} vs {dv}")
    fig.suptitle(f"{primary_arm} — predictor vs DV scatter quad (raw)")
    fig.tight_layout()
    p = fig_dir / "hero2_predictor_dv_scatter_quad.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p.name)

    # Exploratory: collinearity scatter (cos_L2 vs prior).
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(tbl["base_prior"], np.abs(tbl["cos_L2_eos"]), s=22, alpha=0.7, color="#55A868")
    ax.set_xlabel("base_prior (b)")
    ax.set_ylabel("|cos_L2_eos|")
    ax.set_title(f"{primary_arm} — collinearity: |cosine| vs prior")
    fig.tight_layout()
    p = fig_dir / "explore_collinearity_cos_vs_prior.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p.name)

    # Raw alongside residualized (planner memory feedback_show_raw_alongside_processed):
    # raw CHANGE vs cosine + the prior-residualized CHANGE vs cosine, side by side.
    def _resid_on_prior(y, prior):
        m = ~(np.isnan(y) | np.isnan(prior))
        if m.sum() < 3:
            return y, m
        A = np.column_stack([np.ones(m.sum()), prior[m]])
        coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        r = np.full_like(y, np.nan)
        r[m] = y[m] - A @ coef
        return r, m

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(tbl["cos_L2_eos"], tbl["change"], s=22, alpha=0.7, color="#C44E52")
    axes[0].set_title(f"{primary_arm} — RAW: CHANGE (t-b) vs cosine_L2")
    axes[0].set_xlabel("cos_L2_eos")
    axes[0].set_ylabel("CHANGE = t - b")
    resid, _ = _resid_on_prior(tbl["change"], tbl["base_prior"])
    axes[1].scatter(tbl["cos_L2_eos"], resid, s=22, alpha=0.7, color="#8172B3")
    axes[1].set_title(f"{primary_arm} — RESIDUALIZED: CHANGE | prior vs cosine_L2")
    axes[1].set_xlabel("cos_L2_eos")
    axes[1].set_ylabel("CHANGE residual (prior partialled)")
    fig.tight_layout()
    p = fig_dir / "explore_change_vs_cosine_raw_and_residualized.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p.name)

    # Raw t (LEVEL) and raw t-b (CHANGE) vs prior — the level/change split visible.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(tbl["base_prior"], tbl["level"], s=22, alpha=0.7, color="#4C72B0")
    axes[0].set_title(f"{primary_arm} — RAW LEVEL (t) vs prior")
    axes[0].set_xlabel("base_prior (b)")
    axes[0].set_ylabel("LEVEL = t")
    axes[1].scatter(tbl["base_prior"], tbl["change"], s=22, alpha=0.7, color="#DD8452")
    axes[1].set_title(f"{primary_arm} — RAW CHANGE (t-b) vs prior")
    axes[1].set_xlabel("base_prior (b)")
    axes[1].set_ylabel("CHANGE = t - b")
    fig.tight_layout()
    p = fig_dir / "explore_level_change_vs_prior_raw.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p.name)

    # Per-source small-multiples: CHANGE vs cosine per source.
    src_ids = tbl.get("source_ids", [])
    uniq_src = sorted(set(src_ids))
    if uniq_src:
        ncol = min(4, len(uniq_src))
        nrow = int(np.ceil(len(uniq_src) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), squeeze=False)
        src_arr = np.array(src_ids)
        for ai, s in enumerate(uniq_src):
            ax = axes[ai // ncol][ai % ncol]
            m = src_arr == s
            ax.scatter(tbl["cos_L2_eos"][m], tbl["change"][m], s=20, alpha=0.7, color="#937860")
            ax.set_title(f"src={s}", fontsize=8)
            ax.set_xlabel("cos_L2", fontsize=7)
            ax.set_ylabel("CHANGE", fontsize=7)
        for ai in range(len(uniq_src), nrow * ncol):
            axes[ai // ncol][ai % ncol].axis("off")
        fig.suptitle(f"{primary_arm} — per-source CHANGE vs cosine_L2")
        fig.tight_layout()
        p = fig_dir / "explore_per_source_change_vs_cosine.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p.name)

    # Per-seed scatter (seed 42 vs 137 trained rate per cell).
    if "t_seed_42" in tbl and not np.all(np.isnan(tbl["t_seed_137"])):
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        ax.scatter(tbl["t_seed_42"], tbl["t_seed_137"], s=22, alpha=0.7, color="#55A868")
        lim = [0, max(0.05, np.nanmax([tbl["t_seed_42"], tbl["t_seed_137"]]) * 1.05)]
        ax.plot(lim, lim, "k--", lw=0.7)
        ax.set_xlabel("trained rate (seed 42)")
        ax.set_ylabel("trained rate (seed 137)")
        ax.set_title(f"{primary_arm} — per-seed trained rate agreement")
        fig.tight_layout()
        p = fig_dir / "explore_per_seed_scatter.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p.name)

    # L20-vs-early-band cosine comparison (robustness cell, if present).
    if not np.all(np.isnan(tbl.get("cos_L20_robust", np.array([np.nan])))):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(tbl["cos_L20_robust"], tbl["cos_L2_eos"], s=22, alpha=0.7, color="#CCB974")
        ax.set_xlabel("cos_L20 (robustness; #509 weak cell)")
        ax.set_ylabel("cos_L2_eos (early band; primary)")
        ax.set_title(f"{primary_arm} — L20 vs early-band cosine")
        fig.tight_layout()
        p = fig_dir / "explore_l20_vs_early_cosine.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p.name)

    # Canned-vs-onpolicy arm overlay (CHANGE vs cosine), if both arms present.
    if "arm_canned" in tables and "arm_onpolicy" in tables:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        for arm_name, color in (("arm_canned", "#4C72B0"), ("arm_onpolicy", "#C44E52")):
            at = tables[arm_name]
            ax.scatter(at["cos_L2_eos"], at["change"], s=22, alpha=0.6, color=color, label=arm_name)
        ax.set_xlabel("cos_L2_eos")
        ax.set_ylabel("CHANGE = t - b")
        ax.set_title("canned vs on-policy arm overlay (CHANGE vs cosine)")
        ax.legend()
        fig.tight_layout()
        p = fig_dir / "explore_arm_overlay_change_vs_cosine.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p.name)

    # Bystander-grouped vs source-grouped CV ladder side-by-side (identifiability fix visible).
    sg = ladders[primary_arm].get("source_grouped_robustness")
    if sg is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        x = np.arange(len(labels))
        for ax, dv in zip(axes, ("level", "change"), strict=True):
            bvals = [bg[dv][k] for k in model_keys]
            svals = [sg[dv][k] for k in model_keys]
            ax.bar(
                x - 0.2,
                [v if not np.isnan(v) else 0 for v in bvals],
                0.4,
                label="bystander-grouped",
                color="#4C72B0",
            )
            ax.bar(
                x + 0.2,
                [v if not np.isnan(v) else 0 for v in svals],
                0.4,
                label="source-grouped",
                color="#DD8452",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_title(f"{primary_arm} — {dv.upper()} ladder: CV grouping")
            ax.set_ylabel("held-out CV-R²")
            ax.axhline(0, color="k", lw=0.6)
            ax.legend(fontsize=7)
        fig.tight_layout()
        p = fig_dir / "explore_cv_grouping_ladder_comparison.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p.name)

    meta = {
        "figures": written,
        "primary_arm": primary_arm,
        "smoke": smoke,
        "git_commit": _git_commit_sha(),
        "timestamp": _now_iso(),
    }
    _write_json(fig_dir / "meta.json", meta)
    return written


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--smoke", action="store_true", help="1 source x 3 bystanders x 1 seed x canned"
    )
    ap.add_argument(
        "--geometry-npz",
        type=Path,
        default=None,
        help="Phase-1 geometry .npz (default data/issue_649/inputs/early_layer_geometry*.npz)",
    )
    ap.add_argument(
        "--inputs-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue_649" / "inputs",
        help="issue-owned inputs snapshot dir",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_649",
    )
    ap.add_argument(
        "--fig-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "issue_649",
    )
    args = ap.parse_args()

    geometry_npz = args.geometry_npz or (
        args.inputs_dir
        / ("early_layer_geometry_smoke.npz" if args.smoke else "early_layer_geometry.npz")
    )
    out_dir = args.out_dir
    fig_dir = args.fig_dir
    n_boot = N_BOOT_SMOKE if args.smoke else N_BOOT_FULL

    # ── load panel_set + geometry ──
    panel_set = json.loads((args.inputs_dir / "panel_set.json").read_text())
    geom = load_geometry(geometry_npz, args.inputs_dir)
    all_bystanders = list(panel_set["personas"].keys())

    # ── arms + cell scope ──
    if args.smoke:
        arms = ["arm_canned"]
        sources_for_arm = {"arm_canned": ("villain",)}
        # All non-villain bystanders present in the smoke geometry npz (6 of them,
        # per the extractor's 7-persona smoke subset) -> 6 off-diagonal cells,
        # clearing _cv_r2_grouped's len(y)<5 floor so the bystander-grouped M0
        # CV-R² is non-NaN (the plan's smoke identifiability check fires).
        smoke_by = [b for b in geom["idx"] if b != "villain"]
        bystanders = ["villain", *smoke_by]
        seeds = (42,)
    else:
        arms = ["arm_canned", "arm_onpolicy"]
        sources_for_arm = {"arm_canned": SOURCES_CANNED, "arm_onpolicy": SOURCES_ONPOLICY}
        bystanders = all_bystanders
        seeds = SEEDS

    # ── Phase 0: prefetch + content-pin ──
    logger.info("[phase=prefetch] arms=%s seeds=%s bystanders=%d", arms, seeds, len(bystanders))
    manifest = prefetch_cells(arms, sources_for_arm, bystanders, seeds, args.inputs_dir)
    _write_json(args.inputs_dir / "expected_sha256_manifest.json", manifest)

    # ── Phase 2 + 3 per arm ──
    cv_ladders: dict[str, Any] = {}
    spearman_out: dict[str, Any] = {}
    gates_out: dict[str, Any] = {}
    tables: dict[str, dict] = {}

    for arm in arms:
        logger.info("[phase=build_table] arm=%s", arm)
        tbl = build_table(
            arm, sources_for_arm[arm], bystanders, seeds, panel_set, geom, args.inputs_dir
        )
        tables[arm] = tbl
        logger.info(
            "arm=%s cells=%d excluded=%s per_seed_sign_agree=%.3f",
            arm,
            tbl["n_cells"],
            tbl["excluded"],
            tbl["per_seed_sign_agreement"],
        )
        if tbl["n_cells"] < 3:
            logger.warning("arm=%s has < 3 cells; ladder/spearman will be NaN", arm)

        logger.info("[phase=ladder] arm=%s", arm)
        bg = {
            "level": six_regression_ladder(tbl, "level", tbl["bystander_group"]),
            "change": six_regression_ladder(tbl, "change", tbl["bystander_group"]),
        }
        sg = {
            "level": six_regression_ladder(tbl, "level", tbl["source_group"]),
            "change": six_regression_ladder(tbl, "change", tbl["source_group"]),
        }
        # intercept-only-M0 variant under source-grouped CV (apples-to-apples
        # generalization read; plan §4 step 4 lines 81-82).
        sg_intercept = {
            "level": intercept_only_ladder(tbl, "level", tbl["source_group"]),
            "change": intercept_only_ladder(tbl, "change", tbl["source_group"]),
        }
        # smoke identifiability check: bystander-grouped M0 CV-R² must be non-NaN
        m0_bg = bg["level"]["M0_source_indicators"]
        logger.info("arm=%s bystander-grouped M0(level) CV-R²=%s", arm, m0_bg)
        cv_ladders[arm] = {
            "bystander_grouped": bg,
            "source_grouped_robustness": sg,
            "source_grouped_intercept_only": sg_intercept,
            "n_cells": tbl["n_cells"],
            "n_sources": int(tbl["source_onehot"].shape[1]),
        }

        logger.info("[phase=spearman] arm=%s n_boot=%d", arm, n_boot)
        spearman_out[arm] = {
            "marginal": marginal_spearman_table(tbl, n_boot),
            "noncircular_partials_on_trained_rate": noncircular_partials(tbl, n_boot),
        }

        logger.info("[phase=gates] arm=%s", arm)
        gates_out[arm] = {
            "collinearity": collinearity_gate(tbl),
            "forced_choice_391": forced_choice_gate(tbl),
            "per_seed_sign_agreement": tbl["per_seed_sign_agreement"],
            "n_per_seed_pairs": tbl["n_per_seed_pairs"],
            "excluded_cells": tbl["excluded"],
            "n_cells": tbl["n_cells"],
        }

    # ── write outputs ──
    repro = _repro_metadata()
    _write_json(
        out_dir / "cv_r2_ladder.json",
        {
            "_doc": (
                "Per-arm x per-DV six-model CV-R² ladder. "
                "headline=bystander_grouped, robustness=source_grouped."
            ),
            "smoke": args.smoke,
            "arms": cv_ladders,
            "geometry_meta": {
                "centering": geom["centering"],
                "kl_nan_L2": geom["kl_nan_L2"],
                "kl_nan_L7": geom["kl_nan_L7"],
                "n_probes": geom["meta"].get("n_probes"),
                "gkl_k": geom["meta"].get("gkl_k"),
            },
            "reproducibility": repro,
        },
    )
    _write_json(
        out_dir / "marginal_spearman.json",
        {
            "_doc": "6 rows per arm (prior/cosine_L2/kl_L2 x LEVEL/CHANGE), bootstrap 95% CIs.",
            "smoke": args.smoke,
            "n_bootstrap": n_boot,
            "arms": spearman_out,
            "reproducibility": repro,
        },
    )
    _write_json(
        out_dir / "analysis.json",
        {
            "_doc": (
                "Collinearity gate, #391 forced-choice S/N gate, per-seed sign agreement, coverage."
            ),
            "smoke": args.smoke,
            "arms": gates_out,
            "data_repo_revision": manifest["data_repo_revision"],
            "n_pinned_cells": manifest["n_cells"],
            "geometry_l20_robustness_note": geom.get("cos_L20_note"),
            "reproducibility": repro,
        },
    )
    # Separate brief-required output files (plan §6.5 + brief Outputs list).
    _write_json(
        out_dir / "collinearity_gate.json",
        {
            "_doc": (
                "Pearson(|cos_L2_eos|, base_prior) + gate decision + robustness reads, per arm."
            ),
            "smoke": args.smoke,
            "arms": {a: gates_out[a]["collinearity"] for a in gates_out},
            "reproducibility": repro,
        },
    )
    _write_json(
        out_dir / "precision_check.json",
        {
            "_doc": (
                "Realized per-cell Wilson half-widths (cluster-honest n=60) + median|CHANGE| "
                "S/N + #391 forced-choice gate decision, per arm. Recomputed on realized cells."
            ),
            "smoke": args.smoke,
            "arms": {a: gates_out[a]["forced_choice_391"] for a in gates_out},
            "reproducibility": repro,
        },
    )
    _write_per_cell_csv(out_dir / "per_cell_table.csv", tables)
    logger.info("wrote per_cell_table.csv, collinearity_gate.json, precision_check.json")

    logger.info("[phase=figures]")
    figs = make_figures(cv_ladders, tables, fig_dir, args.smoke)
    logger.info("wrote %d figures -> %s", len(figs), fig_dir)
    logger.info("[phase=done] outputs in %s", out_dir)


if __name__ == "__main__":
    main()
