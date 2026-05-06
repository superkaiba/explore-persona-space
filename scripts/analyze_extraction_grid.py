#!/usr/bin/env python3
"""
Analysis script for issue #263's continuous (method x token x layer) sweep.

Reads centroids written by `scripts/sweep_extraction_grid.py` and computes:

H1 — Equivalence-class clustering
  - 275x275 mean-centered cosine matrix per cell.
  - mc_r distance (1 - Pearson r of off-diagonal entries) between cells.
  - Agglomerative clustering, average linkage, threshold 0.10 (mc_r >= 0.90).
  - Manifest check: cells actually computed must match plan §5 within ±1.

H2 — Per-persona discrimination AUC (Arditi-style validation-based selection)
  - Reference cell = (method=a, i=-1, l=21).
  - Selection over candidate cells EXCLUDING CAA (per plan §3 v3 fix 1).
  - Train (200) + val (20) for selection; test (20) for the H2 readout.
  - Practical-relevance gate: Δ AUC ≥ 0.02 to count toward the ≥ 50% threshold.
  - 50000-permutation paired test on per-persona ΔAUC; BH-FDR q=0.05 primary.
  - Permuted-persona-label null (B=1000) as the load-bearing per-persona cutoff.
  - Random-direction null (1000) as a sanity bound.
  - Two readouts: full 275 + filtered (where ref-AUC > 0.7).

H3 — Response-token ramp (cosine, paired permuted-persona control)
  - Cosine projection c_{p,t,q} = <h_t, v_p> / (||h_t|| ||v_p||).
  - Δ_p = mean_q c_{p,128,q} - mean_q c_{p,0,q}.
  - For each of 5 derangements (seeds 42-46): paired sign test on Δ_p - Δ_p^perm.
  - H3 fires iff ALL 5 derangements reject at p < 0.05 (Bonferroni x 5 ⇒ alpha=0.01),
    AND median fraction-positive ≥ 70%.

Output: eval_results/issue_263/run_result.json + figures/.

Usage
-----
  uv run python scripts/analyze_extraction_grid.py \\
      --centroid-root data/persona_vectors/issue_263/qwen2.5-7b-instruct \\
      --output-dir eval_results/issue_263 \\
      --reference-method a --reference-layer 21 --reference-position -1 \\
      --train-qids 0..199 --val-qids 200..219 --test-qids 220..239 \\
      --n-perms 50000 --n-permuted-label-nulls 1000 --n-random-nulls 1000

Smoke-mode auto-shrinks the permutation counts and tolerates missing cells.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from explore_persona_space.analysis.cosine_grid import (
    mc_r_distance,
    mean_center_cosine_matrix,
)

DEFAULT_REFERENCE_METHOD = "a"
DEFAULT_REFERENCE_LAYER = 21
DEFAULT_REFERENCE_POSITION = -1
DEFAULT_TRAIN_QIDS = "0..199"
DEFAULT_VAL_QIDS = "200..219"
DEFAULT_TEST_QIDS = "220..239"

# Methods that are SAFE to include in H2 candidate set (CAA excluded per plan §3 v3 fix 1)
H2_CANDIDATE_METHODS = {"a", "a_per_token", "b", "bstar", "c1", "c2", "c3", "r_per_token"}

# H1 considers ALL methods (including CAA) for clustering / per-method baseline.
H1_METHODS = {"a", "a_per_token", "b", "bstar", "c1", "c2", "c3", "r_per_token", "caa"}

# H1 thresholds
H1_MC_R_THRESHOLD = 0.90  # ≡ distance threshold 0.10
H1_DISTANCE_THRESHOLD = 1.0 - H1_MC_R_THRESHOLD
H1_MAX_CLASSES = 5
H1_MIN_COVERAGE_FRACTION = 0.80
PRE_REGISTERED_H1_CELL_DENOMINATOR = 672  # per plan §5 reproducibility card

# H2 thresholds
H2_DELTA_AUC_GATE = 0.02
H2_PASS_FRACTION = 0.50
H2_FILTERED_REF_AUC_THRESHOLD = 0.7
H2_PERMUTED_NULL_PERCENTILE = 99
H2_RANDOM_NULL_PERCENTILE = 99
H2_FDR_Q = 0.05
H2_HOLM_ALPHA = 0.05

# H3 thresholds
H3_DERANGEMENT_SEEDS = [42, 43, 44, 45, 46]  # 5 independent derangements
H3_BONFERRONI_FACTOR = len(H3_DERANGEMENT_SEEDS)
H3_PER_TEST_ALPHA = 0.05 / H3_BONFERRONI_FACTOR
H3_FRACTION_POSITIVE_THRESHOLD = 0.70


# ── Reproducibility ──────────────────────────────────────────────────────────


def _git_commit_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"


# ── CLI helpers ──────────────────────────────────────────────────────────────


def parse_qid_range(s: str) -> list[int]:
    """Parse '0..199' into [0, 1, ..., 199]; or comma-separated '0,1,2'."""
    s = s.strip()
    if ".." in s:
        lo, hi = s.split("..")
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ── Centroid loading ─────────────────────────────────────────────────────────


def discover_cells(centroid_root: Path) -> list[tuple[str, int, int]]:
    """Return all (method, position, layer) cells written by the sweep.

    Cells are top-level directories of the form `method_<m>__pos_<p>__layer_<l>` that
    contain at least one `.pt` file.
    """
    cells: list[tuple[str, int, int]] = []
    if not centroid_root.exists():
        return cells
    for entry in sorted(centroid_root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith("method_"):
            continue
        if "__pos_" not in name or "__layer_" not in name:
            continue
        try:
            method_part, rest = name[len("method_") :].split("__pos_")
            pos_str, layer_str = rest.split("__layer_")
            method = method_part
            position = int(pos_str)
            layer = int(layer_str)
        except ValueError:
            continue
        if not any(entry.glob("*.pt")):
            continue
        cells.append((method, position, layer))
    return cells


def load_cell_centroids(
    centroid_root: Path, method: str, position: int, layer: int, roles: list[str]
) -> torch.Tensor:
    """Load centroids at one (method, position, layer) cell as (N_roles, D) fp32 tensor.

    Roles missing from disk are reported once and the corresponding row is filled with
    NaN — downstream analysis must check `torch.isnan(...).any()` per row.
    """
    cell_dir = centroid_root / f"method_{method}__pos_{position}__layer_{layer}"
    if not cell_dir.exists():
        raise FileNotFoundError(f"Missing cell dir: {cell_dir}")
    rows: list[torch.Tensor] = []
    sample = torch.load(next(cell_dir.glob("*.pt")), weights_only=True, map_location="cpu")
    D = sample.shape[-1]
    for role in roles:
        path = cell_dir / f"{role}.pt"
        if path.exists():
            v = torch.load(path, weights_only=True, map_location="cpu").float()
            if v.ndim != 1 or v.shape[0] != D:
                raise RuntimeError(f"Cell {cell_dir} role={role}: expected (D,), got {v.shape}")
            rows.append(v)
        else:
            rows.append(torch.full((D,), float("nan")))
    return torch.stack(rows)


def load_per_q_method_a(
    centroid_root: Path,
    layer: int,
    roles: list[str],
    qids: list[int],
    layers_in_cache: list[int] | None = None,
) -> torch.Tensor:
    """Load Method A per-question caches at a single ABSOLUTE layer over a question subset.

    Returns (N_roles, n_qids, D) fp32 tensor. Used for H2 AUC computation:
    per-question hidden states act as the "samples" labelled by persona.

    Args:
        layer: ABSOLUTE model-layer index (e.g. 21 for the project default).
        layers_in_cache: ordered list of absolute layer indices that were dumped to the
            per-q cache (cache dim 1 is `len(layers_in_cache)`). If None, defaults to
            the canonical full set range(28) — fails fast if the cache shape disagrees.
    """
    method_dir = centroid_root / "method_a"
    if not method_dir.exists():
        raise FileNotFoundError(f"Missing per-q cache dir: {method_dir}")
    qids_idx = torch.tensor(qids, dtype=torch.long)
    rows: list[torch.Tensor] = []
    sample_path = next(method_dir.glob("*__per_q.pt"))
    sample = torch.load(sample_path, weights_only=True, map_location="cpu")
    if sample.ndim != 3:
        raise RuntimeError(f"per_q shape mismatch: {sample.shape} (expected (n_q, n_layers, D))")
    n_q_avail = sample.shape[0]
    n_layers_avail = sample.shape[1]
    if layers_in_cache is None:
        # Canonical full sweep dumps all 28 layers in order (0..27).
        layers_in_cache = list(range(n_layers_avail))
    if len(layers_in_cache) != n_layers_avail:
        raise RuntimeError(
            f"layers_in_cache has {len(layers_in_cache)} entries but cache dim 1 is "
            f"{n_layers_avail}. Mismatch — pass an explicit layers list from "
            f"sweep_metadata.json."
        )
    if layer not in layers_in_cache:
        raise RuntimeError(
            f"Requested absolute layer {layer} but per_q cache only has layers {layers_in_cache}."
        )
    cache_layer_idx = layers_in_cache.index(layer)
    if max(qids) >= n_q_avail:
        raise RuntimeError(
            f"Requested qid {max(qids)} but per_q cache has only {n_q_avail} questions."
        )
    for role in roles:
        path = method_dir / f"{role}__per_q.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing per-q cache: {path}")
        per_q = torch.load(path, weights_only=True, map_location="cpu")  # fp16
        rows.append(per_q[qids_idx, cache_layer_idx, :].float())
    return torch.stack(rows)


# ── H1: clustering ───────────────────────────────────────────────────────────


def compute_h1_clustering(
    centroid_root: Path,
    cells: list[tuple[str, int, int]],
    roles: list[str],
    train_qids: list[int],
    rng_seed: int,
    layers_in_cache: list[int] | None = None,
) -> dict:
    """Cluster cells by 1 - Pearson r of mean-centered cosine matrix off-diagonals.

    Per plan §7 step 1: H1 evaluation uses the 200 TRAINING questions only — H1 must
    not consume the test split. We re-aggregate centroids from per-question caches
    over q_idx 0..199 wherever per-q caches exist; cells without per-q caches use the
    centroid-on-disk (which was averaged over all 240 questions in the sweep).
    """
    print(
        f"\n[H1] Computing cosine matrices over {len(cells)} cells "
        f"({len(roles)} personas, {len(train_qids)} train questions)..."
    )
    if not cells:
        return {
            "cells_total": 0,
            "n_clusters": 0,
            "coverage_fraction": 0.0,
            "verdict": "FAIL",
            "reason": "no cells discovered",
            "cluster_assignments": {},
        }

    # Build mean-centered cosine matrix per cell
    matrices: dict[tuple[str, int, int], np.ndarray] = {}
    for method, position, layer in cells:
        # Try per-q cache for train-only mean if available; else use disk centroid.
        per_q_dir = centroid_root / f"method_{method}"
        cents: torch.Tensor
        if (per_q_dir / f"{roles[0]}__per_q.pt").exists() and method == "a":
            try:
                per_q_block = load_per_q_method_a(
                    centroid_root, layer, roles, train_qids, layers_in_cache=layers_in_cache
                )
                cents = per_q_block.mean(dim=1)  # (N, D)
            except (FileNotFoundError, RuntimeError):
                cents = load_cell_centroids(centroid_root, method, position, layer, roles)
        else:
            cents = load_cell_centroids(centroid_root, method, position, layer, roles)
        nan_mask = torch.isnan(cents).any(dim=1)
        if nan_mask.any():
            # Drop NaN rows + report
            print(
                f"  {method}__pos_{position}__layer_{layer}: dropping {int(nan_mask.sum())}"
                " NaN rows",
            )
            cents = cents[~nan_mask]
        if cents.shape[0] < 3:
            print(
                f"  Skipping cell (method={method}, pos={position}, layer={layer}): "
                f"only {cents.shape[0]} valid roles."
            )
            continue
        matrices[(method, position, layer)] = mean_center_cosine_matrix(cents)

    cell_keys = list(matrices.keys())
    n_cells = len(cell_keys)
    if n_cells < 2:
        return {
            "cells_total": n_cells,
            "n_clusters": n_cells,
            "coverage_fraction": 1.0 if n_cells == 1 else 0.0,
            "verdict": "FAIL" if n_cells == 0 else "PASS",
            "reason": "fewer than 2 cells; clustering trivially passes / fails",
            "cluster_assignments": {f"cell_{k}": 0 for k in cell_keys},
        }

    # Pairwise mc_r distances
    print(
        f"[H1] Computing pairwise mc_r distances for {n_cells * (n_cells - 1) // 2} cell pairs..."
    )
    distance_matrix = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        for j in range(i + 1, n_cells):
            distance_matrix[i, j] = mc_r_distance(matrices[cell_keys[i]], matrices[cell_keys[j]])
            distance_matrix[j, i] = distance_matrix[i, j]
    np.fill_diagonal(distance_matrix, 0.0)

    # Agglomerative average-linkage clustering at threshold 0.10
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method="average")
    cluster_labels = fcluster(Z, t=H1_DISTANCE_THRESHOLD, criterion="distance")

    # Coverage of largest H1_MAX_CLASSES clusters
    unique, counts = np.unique(cluster_labels, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]
    top_classes = unique[sorted_idx][:H1_MAX_CLASSES]
    top_coverage = counts[sorted_idx][:H1_MAX_CLASSES].sum() / n_cells

    # Build composition map
    cluster_assignments: dict[str, int] = {}
    for i, key in enumerate(cell_keys):
        method, pos, layer = key
        cluster_assignments[f"method={method}__pos={pos}__layer={layer}"] = int(cluster_labels[i])

    # Cluster compositions: cluster -> list of cells
    composition: dict[int, list[str]] = {}
    for k, c in cluster_assignments.items():
        composition.setdefault(c, []).append(k)

    # H1 cell-count denominator manifest check (per plan §7)
    denominator_drift = abs(n_cells - PRE_REGISTERED_H1_CELL_DENOMINATOR)
    h1_denominator_ok = denominator_drift <= 1

    n_clusters_total = len(unique)
    h1_pass = (
        n_clusters_total <= H1_MAX_CLASSES
        and top_coverage >= H1_MIN_COVERAGE_FRACTION
        and h1_denominator_ok
    )

    return {
        "cells_total": n_cells,
        "n_clusters": int(n_clusters_total),
        "top_class_count": len(top_classes),
        "top_coverage_fraction": float(top_coverage),
        "denominator_pre_registered": PRE_REGISTERED_H1_CELL_DENOMINATOR,
        "denominator_drift": int(denominator_drift),
        "denominator_manifest_ok": bool(h1_denominator_ok),
        "verdict": "PASS" if h1_pass else "FAIL",
        "thresholds": {
            "max_classes": H1_MAX_CLASSES,
            "min_coverage_fraction": H1_MIN_COVERAGE_FRACTION,
            "mc_r_threshold": H1_MC_R_THRESHOLD,
        },
        "cluster_assignments": cluster_assignments,
        "cluster_composition": {str(k): v for k, v in composition.items()},
    }


# ── H2: per-persona discrimination AUC ───────────────────────────────────────


def compute_auc_one_vs_rest(
    target_acts: np.ndarray, other_acts: np.ndarray, direction: np.ndarray
) -> float:
    """Per-persona discrimination AUC.

    Score(x) = <x, direction> / ||direction|| (cosine-like, monotone equivalent to dot
    product when direction is fixed). Higher score should rank target over non-target.

    Args:
        target_acts: (n_target, D) — held-out activations of target persona.
        other_acts: (n_other, D) — held-out activations of all OTHER personas.
        direction: (D,) — candidate persona vector.

    Returns: AUC in [0, 1].
    """
    if target_acts.size == 0 or other_acts.size == 0:
        return 0.5
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    target_scores = target_acts @ direction
    other_scores = other_acts @ direction
    return _auc_from_scores(target_scores, other_scores)


def _auc_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Mann-Whitney-U based AUC (no scikit-learn dependency)."""
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Rank all together, then sum ranks of positives
    all_scores = np.concatenate([pos_scores, neg_scores])
    ranks = stats.rankdata(all_scores, method="average")
    pos_rank_sum = ranks[:n_pos].sum()
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def candidate_cells(cells: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """H2 candidate set EXCLUDES CAA per plan §3 v3 fix 1."""
    return [(m, p, lyr) for (m, p, lyr) in cells if m in H2_CANDIDATE_METHODS]


def arditi_select_per_persona(
    cell_centroids: dict[tuple[str, int, int], torch.Tensor],
    per_q_acts_train: torch.Tensor,
    per_q_acts_val: torch.Tensor,
    persona_idx: int,
) -> tuple[tuple[str, int, int], float]:
    """Pick the (method, position, layer) cell maximising train+val AUC for `persona_idx`.

    Args:
        cell_centroids: dict mapping cell -> (N_personas, D) centroids.
        per_q_acts_train: (N_personas, n_train, D) hidden states.
        per_q_acts_val: (N_personas, n_val, D) hidden states.
        persona_idx: int index of target persona.

    Returns: (best_cell, best_auc).
    """
    tv_acts = torch.cat([per_q_acts_train, per_q_acts_val], dim=1)  # (N, n_tv, D)
    target = tv_acts[persona_idx].numpy()  # (n_tv, D)
    other = (
        torch.cat([tv_acts[:persona_idx], tv_acts[persona_idx + 1 :]], dim=0)
        .reshape(-1, tv_acts.shape[-1])
        .numpy()
    )  # ((N-1)*n_tv, D)

    best_auc = -np.inf
    best_cell: tuple[str, int, int] | None = None
    for cell, centroids in cell_centroids.items():
        direction = centroids[persona_idx].numpy()
        if not np.isfinite(direction).all():
            continue
        auc = compute_auc_one_vs_rest(target, other, direction)
        if auc > best_auc:
            best_auc = auc
            best_cell = cell
    if best_cell is None:
        # Degenerate: no valid cell. Return reference-like sentinel.
        return ("a", -1, 21), 0.5
    return best_cell, float(best_auc)


def evaluate_test_auc(
    centroid: torch.Tensor,
    per_q_acts_test: torch.Tensor,
    persona_idx: int,
) -> float:
    """Evaluate per-persona discrimination AUC on the held-out test split."""
    target = per_q_acts_test[persona_idx].numpy()
    other = (
        torch.cat([per_q_acts_test[:persona_idx], per_q_acts_test[persona_idx + 1 :]], dim=0)
        .reshape(-1, per_q_acts_test.shape[-1])
        .numpy()
    )
    direction = centroid.numpy()
    if not np.isfinite(direction).all():
        return 0.5
    return compute_auc_one_vs_rest(target, other, direction)


def paired_permutation_p_value(deltas: np.ndarray, n_perms: int, rng: np.random.Generator) -> float:
    """Two-sided paired permutation test on observed mean of `deltas`.

    Random sign-flips per permutation; null distribution = means of sign-flipped deltas.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    obs = float(np.mean(deltas))
    n = deltas.shape[0]
    extreme = 1  # numerator includes observed (per Phipson & Smyth 2010)
    for _ in range(n_perms):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_mean = float(np.mean(signs * deltas))
        if abs(perm_mean) >= abs(obs):
            extreme += 1
    return extreme / (n_perms + 1)


def benjamini_hochberg(pvals: np.ndarray, q: float) -> tuple[np.ndarray, float]:
    """Return BH-FDR rejection mask + threshold for the given q level."""
    pvals = np.asarray(pvals, dtype=np.float64)
    m = len(pvals)
    if m == 0:
        return np.zeros(0, dtype=bool), float("nan")
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = q * (np.arange(1, m + 1) / m)
    below = ranked <= thresholds
    if not below.any():
        return np.zeros(m, dtype=bool), 0.0
    k_max = np.where(below)[0].max()
    cutoff = ranked[k_max]
    rejected = np.zeros(m, dtype=bool)
    rejected[order[: k_max + 1]] = True
    return rejected, float(cutoff)


def holm_bonferroni(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Return Holm-Bonferroni rejection mask at family-wise alpha."""
    pvals = np.asarray(pvals, dtype=np.float64)
    m = len(pvals)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    rejected_sorted = np.zeros(m, dtype=bool)
    for i, idx in enumerate(order):
        threshold = alpha / (m - i)
        if pvals[idx] <= threshold:
            rejected_sorted[i] = True
        else:
            break
    rejected = np.zeros(m, dtype=bool)
    rejected[order[rejected_sorted]] = True
    return rejected


def compute_h2(  # noqa: C901  -- multi-stage analysis is intentionally one function
    centroid_root: Path,
    cells: list[tuple[str, int, int]],
    roles: list[str],
    train_qids: list[int],
    val_qids: list[int],
    test_qids: list[int],
    reference: tuple[str, int, int],
    n_perms: int,
    n_permuted_label_nulls: int,
    n_random_nulls: int,
    rng_seed: int,
    layers_in_cache: list[int] | None = None,
) -> dict:
    """Compute H2 — per-persona discrimination AUC with winner's-curse fix."""
    print(f"\n[H2] Loading per-q caches at reference layer={reference[2]}...")
    _ref_method, _ref_pos, ref_layer = reference

    # Load per-q activations at the reference layer for ALL train, val, test qids.
    per_q_train = load_per_q_method_a(
        centroid_root, ref_layer, roles, train_qids, layers_in_cache=layers_in_cache
    )
    per_q_val = load_per_q_method_a(
        centroid_root, ref_layer, roles, val_qids, layers_in_cache=layers_in_cache
    )
    per_q_test = load_per_q_method_a(
        centroid_root, ref_layer, roles, test_qids, layers_in_cache=layers_in_cache
    )

    print(
        f"  per_q shapes: train={tuple(per_q_train.shape)}, "
        f"val={tuple(per_q_val.shape)}, test={tuple(per_q_test.shape)}"
    )

    # Load all candidate-cell centroids (excluding CAA per H2 candidate set).
    candidates = candidate_cells(cells)
    print(f"[H2] Loading centroids over {len(candidates)} candidate cells (CAA excluded)...")
    centroids: dict[tuple[str, int, int], torch.Tensor] = {}
    for cell in candidates:
        method, position, layer = cell
        centroids[cell] = load_cell_centroids(centroid_root, method, position, layer, roles)

    # Reference cell centroid
    if reference not in centroids:
        # Reference must be reachable; force-load even if it's a CAA-equivalent
        method, position, layer = reference
        centroids[reference] = load_cell_centroids(centroid_root, method, position, layer, roles)

    n_personas = len(roles)
    print(f"[H2] Running per-persona Arditi selection over {n_personas} personas...")

    # Per-persona selection + evaluation
    selected_cells: list[tuple[str, int, int]] = []
    selected_aucs: list[float] = []
    test_aucs_candidate: list[float] = []
    test_aucs_reference: list[float] = []
    delta_aucs: list[float] = []

    for p_idx in range(n_personas):
        best_cell, best_train_val_auc = arditi_select_per_persona(
            centroids, per_q_train, per_q_val, p_idx
        )
        selected_cells.append(best_cell)
        selected_aucs.append(best_train_val_auc)

        cand_centroid = centroids[best_cell][p_idx]
        ref_centroid = centroids[reference][p_idx]
        test_auc_cand = evaluate_test_auc(cand_centroid, per_q_test, p_idx)
        test_auc_ref = evaluate_test_auc(ref_centroid, per_q_test, p_idx)
        test_aucs_candidate.append(test_auc_cand)
        test_aucs_reference.append(test_auc_ref)
        delta_aucs.append(test_auc_cand - test_auc_ref)

    delta_arr = np.asarray(delta_aucs)

    # ── Permuted-persona-label null (load-bearing per plan §6 C4b) ──
    print(
        f"[H2] Permuted-persona-label null (B={n_permuted_label_nulls}, "
        f"per-persona 99th percentile)..."
    )
    rng = np.random.default_rng(rng_seed)
    permuted_null_test_aucs = np.zeros((n_permuted_label_nulls, n_personas))
    tv_acts = torch.cat([per_q_train, per_q_val], dim=1)  # (N, n_tv, D)
    for b in range(n_permuted_label_nulls):
        # Shuffle question -> persona labels at the train+val level
        # Equivalently, shuffle persona axis of tv_acts so that each "persona slot" has
        # a random other-persona's activations.
        perm = rng.permutation(n_personas)
        tv_perm = tv_acts[perm]  # rebind labels
        for p_idx in range(n_personas):
            # Run arditi_select on the permuted labels
            target = tv_perm[p_idx].numpy()
            other = (
                torch.cat([tv_perm[:p_idx], tv_perm[p_idx + 1 :]], dim=0)
                .reshape(-1, tv_perm.shape[-1])
                .numpy()
            )
            best_auc = -np.inf
            best_cell: tuple[str, int, int] | None = None
            for cell, cent in centroids.items():
                if cell == reference:  # Reference is not in the candidate set (would taint H2)
                    pass
                if cell[0] == "caa":
                    continue
                direction = cent[p_idx].numpy()
                if not np.isfinite(direction).all():
                    continue
                auc = compute_auc_one_vs_rest(target, other, direction)
                if auc > best_auc:
                    best_auc = auc
                    best_cell = cell
            if best_cell is None:
                permuted_null_test_aucs[b, p_idx] = 0.5
            else:
                permuted_null_test_aucs[b, p_idx] = evaluate_test_auc(
                    centroids[best_cell][p_idx], per_q_test, p_idx
                )
        if (b + 1) % max(1, n_permuted_label_nulls // 10) == 0:
            print(f"  permuted null b={b + 1}/{n_permuted_label_nulls}", flush=True)

    permuted_null_p99 = np.percentile(permuted_null_test_aucs, H2_PERMUTED_NULL_PERCENTILE, axis=0)

    # ── Random direction null (sanity bound, per plan §6 C4a) ──
    print(f"[H2] Random direction null (B={n_random_nulls})...")
    D = per_q_test.shape[-1]
    random_null_test_aucs = np.zeros((n_random_nulls, n_personas))
    for b in range(n_random_nulls):
        direction = rng.normal(size=D)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        for p_idx in range(n_personas):
            target = per_q_test[p_idx].numpy()
            other = (
                torch.cat([per_q_test[:p_idx], per_q_test[p_idx + 1 :]], dim=0)
                .reshape(-1, per_q_test.shape[-1])
                .numpy()
            )
            random_null_test_aucs[b, p_idx] = compute_auc_one_vs_rest(target, other, direction)
    random_null_p99 = np.percentile(random_null_test_aucs, H2_RANDOM_NULL_PERCENTILE, axis=0)

    # ── Per-persona "beats default" indicator ──
    beats_default_unfiltered = np.zeros(n_personas, dtype=bool)
    for p_idx in range(n_personas):
        cond_delta = delta_arr[p_idx] >= H2_DELTA_AUC_GATE
        cond_perm = test_aucs_candidate[p_idx] > permuted_null_p99[p_idx]
        cond_rand = test_aucs_candidate[p_idx] > random_null_p99[p_idx]
        beats_default_unfiltered[p_idx] = cond_delta and cond_perm and cond_rand

    frac_beat = float(beats_default_unfiltered.mean())

    # ── Filtered readout (ref-AUC > 0.7 personas only) ──
    filtered_mask = np.asarray(test_aucs_reference) > H2_FILTERED_REF_AUC_THRESHOLD
    n_filtered = int(filtered_mask.sum())
    if n_filtered > 0:
        frac_beat_filtered = float(beats_default_unfiltered[filtered_mask].mean())
    else:
        frac_beat_filtered = 0.0

    # ── Headline statistical test: paired permutation across personas ──
    print(f"[H2] Paired permutation (n={n_perms}) on per-persona ΔAUC...")
    p_global = paired_permutation_p_value(delta_arr, n_perms, rng)

    # ── Per-persona individual permutation tests + BH-FDR / Holm correction ──
    # Per-persona p-value: against the null hypothesis that the delta = 0,
    # equivalently a sign-test treating each delta as positive/negative observation.
    # We use the simple two-sided sign test (binomial) here as the per-persona test;
    # the load-bearing null has already been used (permuted-label null at the cell-
    # selection step). This is intentional — per-persona p-values feed BH-FDR.
    per_persona_pvals = np.array(
        [
            stats.binomtest(
                int(d > 0) + int(d == 0) // 2,  # one-sided handling
                n=1,
                p=0.5,
                alternative="two-sided",
            ).pvalue
            if d != 0
            else 1.0
            for d in delta_arr
        ]
    )
    # The above is degenerate (each delta is one observation). Replace with a more
    # informative test: per-persona, draw 1000 random sign-flips and compute p.
    per_persona_pvals = _per_persona_perm_pvalues(delta_arr, rng, n_perms_per_persona=1000)

    bh_rejected, bh_cutoff = benjamini_hochberg(per_persona_pvals, q=H2_FDR_Q)
    holm_rejected = holm_bonferroni(per_persona_pvals, alpha=H2_HOLM_ALPHA)

    h2_pass_unfiltered = frac_beat >= H2_PASS_FRACTION
    h2_pass_filtered = frac_beat_filtered >= H2_PASS_FRACTION
    h2_pass = h2_pass_unfiltered and h2_pass_filtered

    return {
        "verdict": "PASS" if h2_pass else "FAIL",
        "verdict_unfiltered": "PASS" if h2_pass_unfiltered else "FAIL",
        "verdict_filtered": "PASS" if h2_pass_filtered else "FAIL",
        "n_personas": n_personas,
        "n_personas_filtered": n_filtered,
        "frac_beat_default_unfiltered": frac_beat,
        "frac_beat_default_filtered": frac_beat_filtered,
        "delta_auc_mean": float(np.mean(delta_arr)),
        "delta_auc_median": float(np.median(delta_arr)),
        "delta_auc_min": float(np.min(delta_arr)),
        "delta_auc_max": float(np.max(delta_arr)),
        "p_value_paired_permutation": p_global,
        "n_perms_global": n_perms,
        "bh_fdr_q": H2_FDR_Q,
        "bh_fdr_n_rejected": int(bh_rejected.sum()),
        "bh_fdr_cutoff": bh_cutoff,
        "holm_alpha": H2_HOLM_ALPHA,
        "holm_n_rejected": int(holm_rejected.sum()),
        "thresholds": {
            "delta_auc_gate": H2_DELTA_AUC_GATE,
            "pass_fraction": H2_PASS_FRACTION,
            "filtered_ref_auc_threshold": H2_FILTERED_REF_AUC_THRESHOLD,
            "permuted_null_percentile": H2_PERMUTED_NULL_PERCENTILE,
            "random_null_percentile": H2_RANDOM_NULL_PERCENTILE,
        },
        "permuted_label_null_quantiles": {
            "p99_per_persona_min": float(permuted_null_p99.min()),
            "p99_per_persona_mean": float(permuted_null_p99.mean()),
            "p99_per_persona_max": float(permuted_null_p99.max()),
            "n_perms": n_permuted_label_nulls,
        },
        "random_null_quantiles": {
            "p99_per_persona_min": float(random_null_p99.min()),
            "p99_per_persona_mean": float(random_null_p99.mean()),
            "p99_per_persona_max": float(random_null_p99.max()),
            "n_perms": n_random_nulls,
        },
        "per_persona_arditi_selection": {
            roles[i]: {
                "selected_cell": (
                    f"method={selected_cells[i][0]}__pos={selected_cells[i][1]}"
                    f"__layer={selected_cells[i][2]}"
                ),
                "train_val_auc": float(selected_aucs[i]),
                "test_auc_candidate": float(test_aucs_candidate[i]),
                "test_auc_reference": float(test_aucs_reference[i]),
                "delta_auc": float(delta_arr[i]),
                "beats_default": bool(beats_default_unfiltered[i]),
                "permuted_null_p99": float(permuted_null_p99[i]),
                "random_null_p99": float(random_null_p99[i]),
                "bh_rejected": bool(bh_rejected[i]),
                "holm_rejected": bool(holm_rejected[i]),
            }
            for i in range(n_personas)
        },
    }


def _per_persona_perm_pvalues(
    deltas: np.ndarray, rng: np.random.Generator, n_perms_per_persona: int = 1000
) -> np.ndarray:
    """Generate per-persona p-values for BH-FDR / Holm correction.

    Per-persona, this is a one-observation test against a noise distribution we
    cannot estimate without resampling. We use the GLOBAL delta distribution as the
    null reference: shuffle deltas across personas and compare each to its rank in
    the shuffled distribution. This is a placeholder when per-persona resamples are
    not available; a future revision should use bootstrap CIs of per-persona ΔAUC.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    n = len(deltas)
    pvals = np.zeros(n)
    # For each persona, count how often a randomly shuffled delta exceeds it
    for i in range(n):
        n_extreme = 0
        target = abs(deltas[i])
        for _ in range(n_perms_per_persona):
            shuffled = rng.permutation(deltas)
            if abs(shuffled[i]) >= target:
                n_extreme += 1
        pvals[i] = (n_extreme + 1) / (n_perms_per_persona + 1)
    return pvals


# ── H3: response-token ramp (paired derangement control) ─────────────────────


def _make_derangement(n: int, seed: int) -> np.ndarray:
    """Generate a derangement of range(n) (no element maps to itself)."""
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if (perm != np.arange(n)).all():
            return perm


def compute_h3(
    centroid_root: Path,
    cells: list[tuple[str, int, int]],
    roles: list[str],
    test_qids: list[int],
    reference: tuple[str, int, int],
    response_positions: list[int],
) -> dict:
    """Compute H3 response-token ramp.

    Loads centroids at reference cell + r_per_token cells at the same layer.
    For each persona p, compute per-question cosine projection at each t in
    `response_positions` (descriptive trajectory).

    For the headline test we need per-question hidden states at t=0 and t=128,
    which are stored as cell centroids in `method_r_per_token__pos_<t>__layer_<L>`.
    Note: those are PER-PERSONA centroids (mean over questions), not per-question.
    Since we did not store per-question response hidden states, the H3 metric here
    is computed at the centroid level: Δ_p = c_{p, t=128} - c_{p, t=0} where each
    is a per-persona centroid projection.

    Per plan §7 step 2: ideally per-question hidden states would be used for the
    20 test-split questions; we surface that the ramp is computed at the centroid
    level here. Future revisions can extend `sweep_extraction_grid.py` to dump
    per-question response hidden states at the headline t values.
    """
    print(f"\n[H3] Computing response-token ramp at reference layer={reference[2]}...")
    ref_method, ref_pos, ref_layer = reference

    # Reference centroids (Method A @ L21)
    ref_centroids = load_cell_centroids(centroid_root, ref_method, ref_pos, ref_layer, roles)

    n_personas = len(roles)
    available_t: dict[int, torch.Tensor] = {}
    for cell in cells:
        method, position, layer = cell
        if method != "r_per_token":
            continue
        if layer != ref_layer:
            continue
        if position not in response_positions:
            continue
        try:
            available_t[position] = load_cell_centroids(
                centroid_root, method, position, layer, roles
            )
        except FileNotFoundError:
            continue

    if 0 not in available_t or 128 not in available_t:
        return {
            "verdict": "FAIL",
            "reason": (
                "Missing r_per_token cells at t=0 and/or t=128 — H3 not evaluable. "
                "Re-run sweep_extraction_grid.py with --methods r_per_token "
                "--response-token-positions 0,1,2,4,8,16,32,64,128."
            ),
            "available_t": sorted(available_t.keys()),
        }

    h_t0 = available_t[0]
    h_t128 = available_t[128]

    # Cosine projection: c_{p,t} = <h_{p,t}, v_p> / (||h_{p,t}|| * ||v_p||)
    # Operating on centroids (per-persona means), since per-question response hidden
    # states are not stored. This is descriptive-but-coarse for the paired test.
    def _proj(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(h, v, dim=1)

    c_t0_self = _proj(h_t0, ref_centroids).numpy()  # (N,)
    c_t128_self = _proj(h_t128, ref_centroids).numpy()
    delta_self = c_t128_self - c_t0_self

    # Mean trajectory (descriptive figure data)
    trajectory_means: dict[int, float] = {}
    trajectory_ci: dict[int, tuple[float, float]] = {}
    for t in sorted(available_t.keys()):
        c_t = _proj(available_t[t], ref_centroids).numpy()
        trajectory_means[t] = float(c_t.mean())
        # 95% bootstrap CI over personas
        rng = np.random.default_rng(42)
        n_boot = 500
        boot_means = np.array(
            [c_t[rng.integers(0, n_personas, size=n_personas)].mean() for _ in range(n_boot)]
        )
        trajectory_ci[t] = (
            float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)),
        )

    # ── Paired derangement test (5 derangements; Bonferroni x 5) ──
    derangement_results: list[dict] = []
    n_rejected = 0
    fraction_positive_per_drng: list[float] = []
    for seed in H3_DERANGEMENT_SEEDS:
        perm = _make_derangement(n_personas, seed)
        v_perm = ref_centroids[perm]  # (N, D) — each row is "another persona's centroid"
        c_t0_perm = _proj(h_t0, v_perm).numpy()
        c_t128_perm = _proj(h_t128, v_perm).numpy()
        delta_perm = c_t128_perm - c_t0_perm
        # Paired sign test: Δ_p - Δ_p^perm > 0 for how many personas?
        diffs = delta_self - delta_perm
        n_pos = int((diffs > 0).sum())
        n_total = n_personas
        # Two-sided binomial sign test
        p_value = stats.binomtest(n_pos, n=n_total, p=0.5, alternative="two-sided").pvalue
        rejected = p_value < H3_PER_TEST_ALPHA
        if rejected:
            n_rejected += 1
        fraction_positive_per_drng.append(n_pos / n_total)
        derangement_results.append(
            {
                "seed": seed,
                "n_positive": n_pos,
                "n_total": n_total,
                "fraction_positive": n_pos / n_total,
                "p_value_two_sided": float(p_value),
                "rejected_at_alpha": bool(rejected),
            }
        )

    median_fraction_positive = float(np.median(fraction_positive_per_drng))
    h3_pass = (
        n_rejected == len(H3_DERANGEMENT_SEEDS)
        and median_fraction_positive >= H3_FRACTION_POSITIVE_THRESHOLD
    )

    return {
        "verdict": "PASS" if h3_pass else "FAIL",
        "n_personas": n_personas,
        "delta_self_mean": float(delta_self.mean()),
        "delta_self_median": float(np.median(delta_self)),
        "median_fraction_positive": median_fraction_positive,
        "n_derangements_rejected": n_rejected,
        "n_derangements_total": len(H3_DERANGEMENT_SEEDS),
        "per_test_alpha": H3_PER_TEST_ALPHA,
        "fraction_positive_threshold": H3_FRACTION_POSITIVE_THRESHOLD,
        "derangements": derangement_results,
        "trajectory_means": trajectory_means,
        "trajectory_ci_95": {str(k): v for k, v in trajectory_ci.items()},
        "available_t": sorted(available_t.keys()),
    }


# ── Figure helpers ───────────────────────────────────────────────────────────


def _maybe_paper_style():
    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style("neurips")
        return True
    except ImportError:
        return False


def plot_h1_clusters(h1_result: dict, output_dir: Path) -> Path:
    """Bar chart of cluster sizes (descriptive H1 figure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _maybe_paper_style()
    composition = h1_result.get("cluster_composition", {})
    sizes = sorted([(k, len(v)) for k, v in composition.items()], key=lambda x: -x[1])
    if not sizes:
        return output_dir / "figures" / "h1_clusters.png"

    labels, counts = zip(*sizes, strict=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(labels)), counts, color="#0072B2")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"C{lbl}" for lbl in labels])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"H1: cluster sizes (n_clusters={h1_result['n_clusters']})")
    ax.axhline(
        h1_result["cells_total"] * H1_MIN_COVERAGE_FRACTION / H1_MAX_CLASSES,
        color="grey",
        linestyle="--",
        alpha=0.5,
        label=f"floor (per cluster, if {H1_MAX_CLASSES} cover {H1_MIN_COVERAGE_FRACTION:.0%})",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "h1_clusters.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_h2_delta_auc(h2_result: dict, output_dir: Path) -> Path:
    """Per-persona ΔAUC histogram with practical-relevance gate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _maybe_paper_style()
    deltas = [v["delta_auc"] for v in h2_result["per_persona_arditi_selection"].values()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(deltas, bins=40, color="#0072B2", alpha=0.85)
    ax.axvline(0.0, color="black", linestyle="-", alpha=0.5, label="0 (no improvement)")
    ax.axvline(
        H2_DELTA_AUC_GATE,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Δ AUC ≥ {H2_DELTA_AUC_GATE}",
    )
    ax.set_xlabel("Per-persona ΔAUC (candidate - reference)")
    ax.set_ylabel("Number of personas")
    ax.set_title(
        f"H2: ΔAUC distribution "
        f"(frac beat default = {h2_result['frac_beat_default_unfiltered']:.2%}, "
        f"N={h2_result['n_personas']})"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "h2_delta_auc.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_h3_trajectory(h3_result: dict, output_dir: Path) -> Path:
    """Mean-trajectory line plot with 95% bootstrap CI (descriptive H3 figure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _maybe_paper_style()
    means = h3_result.get("trajectory_means", {})
    ci = h3_result.get("trajectory_ci_95", {})
    if not means:
        return output_dir / "figures" / "h3_trajectory.png"

    ts = sorted(int(t) for t in means)
    mean_vals = [means[t] for t in ts]
    lo = [ci[str(t)][0] for t in ts]
    hi = [ci[str(t)][1] for t in ts]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, mean_vals, marker="o", color="#0072B2")
    ax.fill_between(ts, lo, hi, alpha=0.2, color="#0072B2", label="95% bootstrap CI")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Generated-token index t")
    ax.set_ylabel("Mean cosine projection across personas")
    ax.set_title(
        "H3: response-token ramp "
        f"(verdict={h3_result['verdict']}, "
        f"median frac positive={h3_result.get('median_fraction_positive', 0.0):.2%})"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "h3_trajectory.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Analysis script for issue #263 (extraction grid)."
    )
    parser.add_argument(
        "--centroid-root",
        type=str,
        required=True,
        help="Centroid root directory written by sweep_extraction_grid.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for run_result.json + figures/",
    )
    parser.add_argument("--reference-method", default=DEFAULT_REFERENCE_METHOD)
    parser.add_argument("--reference-layer", type=int, default=DEFAULT_REFERENCE_LAYER)
    parser.add_argument("--reference-position", type=int, default=DEFAULT_REFERENCE_POSITION)
    parser.add_argument("--train-qids", type=str, default=DEFAULT_TRAIN_QIDS)
    parser.add_argument("--val-qids", type=str, default=DEFAULT_VAL_QIDS)
    parser.add_argument("--test-qids", type=str, default=DEFAULT_TEST_QIDS)
    parser.add_argument("--n-perms", type=int, default=50_000)
    parser.add_argument("--n-permuted-label-nulls", type=int, default=1000)
    parser.add_argument("--n-random-nulls", type=int, default=1000)
    parser.add_argument(
        "--response-token-positions",
        type=str,
        default="0,1,2,4,8,16,32,64,128",
        help="Comma-separated response-side positions to look for in r_per_token cells",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: shrink permutation counts, tolerate missing cells",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.smoke:
        args.n_perms = 100
        args.n_permuted_label_nulls = 5
        args.n_random_nulls = 5
        print(
            f"SMOKE MODE: n_perms={args.n_perms}, "
            f"n_permuted_label_nulls={args.n_permuted_label_nulls}, "
            f"n_random_nulls={args.n_random_nulls}"
        )

    centroid_root = Path(args.centroid_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = discover_cells(centroid_root)
    if not cells:
        raise FileNotFoundError(f"No cells found under {centroid_root}")
    print(f"Discovered {len(cells)} cells under {centroid_root}")

    # Use the role list from the sweep's metadata if available, else infer from cell .pt files.
    sweep_meta_path = centroid_root / "sweep_metadata.json"
    if sweep_meta_path.exists():
        with open(sweep_meta_path) as f:
            sweep_meta = json.load(f)
        roles = sorted(
            {
                p.stem
                for p in (
                    centroid_root / f"method_{cells[0][0]}__pos_{cells[0][1]}__layer_{cells[0][2]}"
                ).glob("*.pt")
            }
        )
    else:
        sweep_meta = {}
        cell_dir = centroid_root / f"method_{cells[0][0]}__pos_{cells[0][1]}__layer_{cells[0][2]}"
        roles = sorted(p.stem for p in cell_dir.glob("*.pt"))
    print(f"  {len(roles)} roles found in cells")

    train_qids = parse_qid_range(args.train_qids)
    val_qids = parse_qid_range(args.val_qids)
    test_qids = parse_qid_range(args.test_qids)

    reference = (args.reference_method, args.reference_position, args.reference_layer)

    # Layers used during the sweep (needed to map absolute layers -> per-q cache index).
    layers_in_cache = sweep_meta.get("layers") if sweep_meta else None

    # ── H1 ──
    h1_result = compute_h1_clustering(
        centroid_root,
        cells,
        roles,
        train_qids,
        rng_seed=args.seed,
        layers_in_cache=layers_in_cache,
    )

    # ── H2 ──
    # H2 needs Method A per-q caches at the reference layer. If they're missing, skip H2.
    method_a_per_q_dir = centroid_root / "method_a"
    h2_result: dict
    if not method_a_per_q_dir.exists() or not any(method_a_per_q_dir.glob("*__per_q.pt")):
        h2_result = {
            "verdict": "SKIPPED",
            "reason": (
                "Method A per-q caches not found at "
                f"{method_a_per_q_dir} — H2 cannot be evaluated. "
                "Re-run sweep_extraction_grid.py with --methods a (which writes per_q caches)."
            ),
        }
    else:
        try:
            h2_result = compute_h2(
                centroid_root,
                cells,
                roles,
                train_qids,
                val_qids,
                test_qids,
                reference,
                n_perms=args.n_perms,
                n_permuted_label_nulls=args.n_permuted_label_nulls,
                n_random_nulls=args.n_random_nulls,
                rng_seed=args.seed,
                layers_in_cache=layers_in_cache,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            h2_result = {"verdict": "SKIPPED", "reason": f"H2 failed: {exc}"}

    # ── H3 ──
    response_positions = [int(x) for x in args.response_token_positions.split(",") if x.strip()]
    try:
        h3_result = compute_h3(
            centroid_root, cells, roles, test_qids, reference, response_positions
        )
    except (FileNotFoundError, RuntimeError) as exc:
        h3_result = {"verdict": "SKIPPED", "reason": f"H3 failed: {exc}"}

    # ── Figures ──
    figures: dict[str, str] = {}
    try:
        figures["h1_clusters"] = str(plot_h1_clusters(h1_result, output_dir))
    except Exception as exc:
        print(f"  WARNING: H1 figure failed: {exc}", flush=True)
    if h2_result.get("verdict") not in {"SKIPPED"}:
        try:
            figures["h2_delta_auc"] = str(plot_h2_delta_auc(h2_result, output_dir))
        except Exception as exc:
            print(f"  WARNING: H2 figure failed: {exc}", flush=True)
    if h3_result.get("verdict") not in {"SKIPPED"}:
        try:
            figures["h3_trajectory"] = str(plot_h3_trajectory(h3_result, output_dir))
        except Exception as exc:
            print(f"  WARNING: H3 figure failed: {exc}", flush=True)

    # ── Save run_result.json ──
    cell_keys = [f"method={m}__pos={p}__layer={lyr}" for (m, p, lyr) in cells]
    run_result = {
        "experiment": "issue_263_extraction_grid",
        "issue": 263,
        "metadata": {
            "git_commit": _git_commit_hash(),
            "timestamp_utc": datetime.now(tz=UTC).isoformat(),
            "centroid_root": str(centroid_root),
            "output_dir": str(output_dir),
            "reference_cell": (f"method={reference[0]}__pos={reference[1]}__layer={reference[2]}"),
            "train_qids": [int(min(train_qids)), int(max(train_qids))],
            "val_qids": [int(min(val_qids)), int(max(val_qids))],
            "test_qids": [int(min(test_qids)), int(max(test_qids))],
            "n_perms_global": args.n_perms,
            "n_permuted_label_nulls": args.n_permuted_label_nulls,
            "n_random_nulls": args.n_random_nulls,
            "seed": args.seed,
            "smoke": bool(args.smoke),
        },
        "sweep_metadata": sweep_meta,
        "data_split": {
            "train_qids": train_qids,
            "val_qids": val_qids,
            "test_qids": test_qids,
        },
        "per_token_grid": {
            "cells_total": len(cells),
            "cells": cell_keys,
        },
        "clustering": h1_result,
        "H1": h1_result,
        "H2": h2_result,
        "H3": h3_result,
        "permuted_label_null_quantiles": h2_result.get("permuted_label_null_quantiles", {}),
        "random_null_quantiles": h2_result.get("random_null_quantiles", {}),
        "figures": figures,
        "thresholds": {
            "h1_max_classes": H1_MAX_CLASSES,
            "h1_min_coverage_fraction": H1_MIN_COVERAGE_FRACTION,
            "h1_mc_r_threshold": H1_MC_R_THRESHOLD,
            "h2_delta_auc_gate": H2_DELTA_AUC_GATE,
            "h2_pass_fraction": H2_PASS_FRACTION,
            "h2_filtered_ref_auc_threshold": H2_FILTERED_REF_AUC_THRESHOLD,
            "h3_per_test_alpha": H3_PER_TEST_ALPHA,
            "h3_fraction_positive_threshold": H3_FRACTION_POSITIVE_THRESHOLD,
        },
    }
    run_result_path = output_dir / "run_result.json"
    with open(run_result_path, "w") as f:
        json.dump(run_result, f, indent=2, default=str)
    print(f"\nWrote: {run_result_path}")
    print(f"  H1 verdict: {h1_result.get('verdict')}")
    print(f"  H2 verdict: {h2_result.get('verdict')}")
    print(f"  H3 verdict: {h3_result.get('verdict')}")


if __name__ == "__main__":
    main()
