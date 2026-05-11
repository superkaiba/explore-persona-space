"""Shared helpers for cosine / mean-centered-cosine matrix analysis across extraction methods.

Lifted from:
- scripts/compare_extraction_methods.py (issue #201)
- .claude/worktrees/issue-218/scripts/compare_extraction_methods_6way.py (issue #218)

Future issues (#263, beyond) should import these primitives instead of re-forking the helpers.

Functions:
- cosine_matrix(C): pairwise cosine similarity matrix from (N, D) tensor.
- mean_center_cosine_matrix(C): mean-center centroids first, then cosine matrix.
- off_diag_upper(M): extract upper-triangle off-diagonal values from a square matrix.
- mc_r_distance(M1, M2): 1 - Pearson r of off-diagonal entries (used for clustering).
- noise_floor_cross_half(per_q_centroids, n_questions): same-method cross-half cosine floor.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


def cosine_matrix(centroids: torch.Tensor) -> np.ndarray:
    """Pairwise cosine similarity matrix from (N, D) tensor. Returns (N, N) numpy array."""
    if centroids.ndim != 2:
        raise ValueError(f"cosine_matrix expects (N, D) tensor; got shape {tuple(centroids.shape)}")
    C_norm = F.normalize(centroids, dim=1)
    return (C_norm @ C_norm.T).numpy()


def mean_center_cosine_matrix(centroids: torch.Tensor) -> np.ndarray:
    """Mean-center centroids, then return cosine matrix. Returns (N, N) numpy array."""
    if centroids.ndim != 2:
        raise ValueError(
            f"mean_center_cosine_matrix expects (N, D) tensor; got shape {tuple(centroids.shape)}"
        )
    global_mean = centroids.mean(dim=0, keepdim=True)
    centered = centroids - global_mean
    return cosine_matrix(centered)


def off_diag_upper(matrix: np.ndarray) -> np.ndarray:
    """Extract upper-triangle off-diagonal values (k=1) from a square matrix."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"off_diag_upper expects a square (N, N) matrix; got shape {matrix.shape}")
    n = matrix.shape[0]
    indices = np.triu_indices(n, k=1)
    return matrix[indices]


def mc_r_distance(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    """1 - Pearson r of off-diagonal upper-triangle entries.

    Both matrices must have the same shape and represent mean-centered cosine matrices
    over the SAME ordered persona set (otherwise the off-diagonal alignment is meaningless).

    Returns float in [0, 2]. Threshold 0.10 corresponds to mc_r >= 0.90.
    """
    if matrix_a.shape != matrix_b.shape:
        raise ValueError(
            f"mc_r_distance requires same-shape matrices; got {matrix_a.shape} vs {matrix_b.shape}"
        )
    a_off = off_diag_upper(matrix_a)
    b_off = off_diag_upper(matrix_b)
    if a_off.size < 2:
        raise ValueError(
            f"Need at least 2 off-diagonal entries to compute Pearson r; got {a_off.size}"
        )
    r, _ = stats.pearsonr(a_off, b_off)
    if np.isnan(r):
        # Degenerate input (zero variance); treat as identical → distance 0
        return 0.0
    return float(1.0 - r)


def noise_floor_cross_half(
    per_q_centroids: torch.Tensor,
    layer_idx: int,
) -> dict[str, float]:
    """Same-method cross-half noise floor on per-question caches.

    Splits the questions in half (slice(0, n_q // 2) vs slice(n_q // 2, n_q)) — the exact
    protocol from #216 / compare_extraction_methods_6way.py. Computes per-persona cosine
    between halves AND the matrix-level mc_r between the two half cosine matrices.

    Args:
        per_q_centroids: tensor of shape (N_personas, n_q, n_layers, D), fp16 or fp32.
        layer_idx: index into the n_layers dim.

    Returns:
        dict with keys: per_persona_min, per_persona_p5, per_persona_mean, matrix_mc_pearson_r.
    """
    if per_q_centroids.ndim != 4:
        raise ValueError(
            f"noise_floor_cross_half expects (N_personas, n_q, n_layers, D); "
            f"got shape {tuple(per_q_centroids.shape)}"
        )
    _, n_q, _, _ = per_q_centroids.shape
    half1 = slice(0, n_q // 2)
    half2 = slice(n_q // 2, n_q)

    # (N_personas, D) per half, fp32 for stability
    cents_h1 = per_q_centroids[:, half1, layer_idx, :].float().mean(dim=1)
    cents_h2 = per_q_centroids[:, half2, layer_idx, :].float().mean(dim=1)

    per_persona = F.cosine_similarity(cents_h1, cents_h2, dim=1).numpy()
    cm1_mc = mean_center_cosine_matrix(cents_h1)
    cm2_mc = mean_center_cosine_matrix(cents_h2)
    pearson_r_mc, _ = stats.pearsonr(off_diag_upper(cm1_mc), off_diag_upper(cm2_mc))

    return {
        "per_persona_min": float(per_persona.min()),
        "per_persona_p5": float(np.percentile(per_persona, 5)),
        "per_persona_mean": float(per_persona.mean()),
        "matrix_mc_pearson_r": float(pearson_r_mc),
    }
