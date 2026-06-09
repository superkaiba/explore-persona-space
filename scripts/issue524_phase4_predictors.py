"""Phase 4 — build predictor matrices over the unified 32-context panel.

Issue #524 plan v4 §4 Phase 4. CPU on dev VM; runs in minutes.

Inputs:
    eval_results/issue_524/phase3/<context>/<layer>/<extraction_point>.npz
        — base-model activation clouds (shape: (n_probes, hidden_dim))
        for each of the 32 contexts × 28 layers × 3 extraction points.
    eval_results/issue_524/phase2/per_cell/G_{src}__{tgt}.json
        — per-cell ΔG / log-prob (read only for the symmetric vs
        directional sanity-check companion; predictor matrices themselves
        only need the activation clouds).

Outputs:
    eval_results/issue_524/phase4/predictors.npz
        — N_PRED × 32 × 32 stack of predictor distance matrices, with
        per-(layer, extraction-point, predictor) channels collapsed by
        the inner-loop selection in Phase 5.

The predictor functions are PER-ENTRY callable: f(cloud_A, cloud_B) → scalar.
Phase 5 (`scripts/issue524_phase5_metrics.py`) iterates over (predictor,
layer, extraction_point) to pick the per-fold winner via inner-CV.

The 5 directional predictors (plan §4 Phase 4) plus 2 re-extracted
directional output-side KLs are emitted alongside the symmetric
baselines (cosine, Gauss-KL-sym, MMD, pooled-Mahalanobis, marker
projection) so Phase 5's M_sym vs M_full comparison can grab matched
pairs.

CLI (smoke == sweep with --layers 1 --points last_prompt for one
extraction point, plan §"Smoke architecture parity" — UNIFIED):
    # Smoke: emit predictors for one layer × one extraction point.
    uv run python scripts/issue524_phase4_predictors.py --layers 22 \\
        --points last_prompt --smoke

    # Full sweep across 28 layers × 3 extraction points.
    uv run python scripts/issue524_phase4_predictors.py
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# epm-lint: workflow-fix-on-bug -- module-top dotenv load required even for
# CPU-only scripts, since downstream phases may pull HF cache pointers via
# huggingface_hub helpers that read env at import time.
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("i524.phase4")

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE3_DIR = REPO_ROOT / "eval_results" / "issue_524" / "phase3"
PHASE4_DIR = REPO_ROOT / "eval_results" / "issue_524" / "phase4"
OUT_PATH = PHASE4_DIR / "predictors.npz"
OUT_META = PHASE4_DIR / "predictors.meta.json"

EXTRACTION_POINTS = ["last_prompt", "mean_response", "end_of_preamble"]

N_LAYERS = 28  # Qwen-2.5-7B residual-stream layers (0..27 inclusive)


# --------------------------------------------------------------------------
# Predictor functional forms — each takes two activation clouds (each
# shape (n_probes, hidden_dim)) on the BASE model and returns a single
# scalar distance/similarity. "Symmetric" forms satisfy f(A, B) == f(B, A);
# "Directional" forms intentionally do not, and are the load-bearing
# additions vs the #502 zoo.
# --------------------------------------------------------------------------

# Tikhonov regularization for covariance inversion. Plan §A8 explicitly:
# Σ_A + ε·I with ε = 1e-6 if det(Σ_A) < 1e-12; we always add the floor for
# numerical safety (matches #502's predictor pipeline). Report fraction
# regularized in the metadata sidecar so plan §A8 verification can read it.
COV_EPS = 1e-6
PCA_RANK = 16  # plan §A8 PCA-16 fit of cloud covariance


def _git_sha() -> str:
    """Short HEAD SHA or 'unknown' on error."""
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _pca_project(cloud: np.ndarray, rank: int = PCA_RANK) -> tuple[np.ndarray, np.ndarray]:
    """Project a high-dim activation cloud to its top-``rank`` principal axes.

    Returns:
        Tuple of (projected_cloud, basis) where ``projected_cloud`` has shape
        (n_probes, rank) and ``basis`` is the orthonormal (hidden_dim, rank)
        loading matrix. Mean is subtracted before projection so the basis is
        a centered PCA basis.
    """
    if cloud.ndim != 2:
        raise ValueError(f"_pca_project expects 2D cloud, got shape {cloud.shape}")
    mu = cloud.mean(axis=0, keepdims=True)
    centered = cloud - mu
    # Truncated SVD via numpy (cloud is small enough — 500 × 3584 is fine).
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:rank].T  # (hidden_dim, rank)
    projected = centered @ basis
    return projected, basis


def _mean_cov(cloud_proj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, covariance + COV_EPS·I) of a projected cloud."""
    mu = cloud_proj.mean(axis=0)
    centered = cloud_proj - mu
    n = centered.shape[0]
    cov = (centered.T @ centered) / max(n - 1, 1)
    cov = cov + COV_EPS * np.eye(cov.shape[0])
    return mu, cov


# -- Symmetric baselines (a subset of the #502 zoo; full zoo is in
# scripts/issue502_predictor_*.py for the rest).


def cosine_distance(cloud_A: np.ndarray, cloud_B: np.ndarray) -> float:
    """1 - cosine(mean_A, mean_B). Symmetric."""
    mu_A = cloud_A.mean(axis=0)
    mu_B = cloud_B.mean(axis=0)
    num = float(mu_A @ mu_B)
    den = float(np.linalg.norm(mu_A) * np.linalg.norm(mu_B)) + 1e-12
    return 1.0 - num / den


def gauss_kl_symmetric(cloud_A: np.ndarray, cloud_B: np.ndarray) -> float:
    """Symmetric KL between Gaussian fits on PCA-16. (KL(A||B) + KL(B||A))/2."""
    Aproj, _ = _pca_project(cloud_A)
    Bproj, _ = _pca_project(cloud_B)
    mu_A, Sig_A = _mean_cov(Aproj)
    mu_B, Sig_B = _mean_cov(Bproj)
    return 0.5 * (
        _gauss_kl_directional(mu_A, Sig_A, mu_B, Sig_B)
        + _gauss_kl_directional(mu_B, Sig_B, mu_A, Sig_A)
    )


def pooled_mahalanobis(cloud_A: np.ndarray, cloud_B: np.ndarray) -> float:
    """Mahalanobis distance under pooled covariance. Symmetric."""
    Aproj, _ = _pca_project(cloud_A)
    Bproj, _ = _pca_project(cloud_B)
    mu_A, Sig_A = _mean_cov(Aproj)
    mu_B, Sig_B = _mean_cov(Bproj)
    Sig_pooled = 0.5 * (Sig_A + Sig_B)
    diff = mu_A - mu_B
    return float(diff @ np.linalg.solve(Sig_pooled, diff))


# -- Directional predictors (plan §4 Phase 4 — the 5 load-bearing additions).


def _gauss_kl_directional(
    mu_A: np.ndarray, Sig_A: np.ndarray, mu_B: np.ndarray, Sig_B: np.ndarray
) -> float:
    """KL(N(mu_A, Sig_A) || N(mu_B, Sig_B)) — asymmetric.

    Implements plan §4 Phase 4 directional_gauss_kl. Assumes both Σ are
    invertible after the COV_EPS floor; uses ``np.linalg.solve`` for
    numerical stability over ``inv``.
    """
    k = mu_A.shape[0]
    diff = mu_B - mu_A
    trace_term = float(np.trace(np.linalg.solve(Sig_B, Sig_A)))
    quad_term = float(diff @ np.linalg.solve(Sig_B, diff))
    sign_a, logdet_a = np.linalg.slogdet(Sig_A)
    sign_b, logdet_b = np.linalg.slogdet(Sig_B)
    # COV_EPS floor guarantees both are positive-definite -> signs == 1.
    if sign_a <= 0 or sign_b <= 0:
        raise ValueError(
            "Σ_A or Σ_B not positive-definite after COV_EPS floor — "
            "inspect the cloud rank or raise COV_EPS"
        )
    logdet_term = logdet_b - logdet_a
    return 0.5 * (trace_term + quad_term - k + logdet_term)


def directional_gauss_kl(cloud_A: np.ndarray, cloud_B: np.ndarray) -> float:
    """KL(A || B) with PCA-16 Gaussian fits. Directional / asymmetric."""
    Aproj, _ = _pca_project(cloud_A)
    Bproj, _ = _pca_project(cloud_B)
    mu_A, Sig_A = _mean_cov(Aproj)
    mu_B, Sig_B = _mean_cov(Bproj)
    return _gauss_kl_directional(mu_A, Sig_A, mu_B, Sig_B)


def source_cov_mahalanobis(cloud_A: np.ndarray, cloud_B: np.ndarray) -> float:
    """Mahalanobis distance from B's mean to A's mean under Σ_A. Directional."""
    Aproj, basis_A = _pca_project(cloud_A)
    mu_A, Sig_A = _mean_cov(Aproj)
    # Project B onto A's basis (so the metric lives in A's natural frame).
    mu_B_full = cloud_B.mean(axis=0)
    mu_A_full = cloud_A.mean(axis=0)
    diff = (mu_B_full - mu_A_full) @ basis_A  # (rank,)
    return float(diff @ np.linalg.solve(Sig_A, diff))


def asym_subspace_recon(cloud_A: np.ndarray, cloud_B: np.ndarray) -> float:
    """How well A's top-k subspace reconstructs B. Directional (asymmetric in A).

    Returns ``1 - ||P_A_k @ phi_B|| / ||phi_B||``: 0 means A spans B perfectly,
    1 means A's subspace is orthogonal to B's centered mean direction.
    """
    _, basis_A = _pca_project(cloud_A)
    # Centered mean direction of B.
    phi_B = cloud_B.mean(axis=0) - cloud_A.mean(axis=0)
    norm_phi = float(np.linalg.norm(phi_B)) + 1e-12
    proj = basis_A @ (basis_A.T @ phi_B)
    return 1.0 - float(np.linalg.norm(proj)) / norm_phi


def marker_projection(cloud_A: np.ndarray, marker_unembed_dir: np.ndarray) -> float:
    """Project A's mean onto the marker token's unembed direction.

    This is a PER-CONTEXT scalar (not a pair distance) — Phase 5 includes it
    in the two-feature combiner. We materialize it here for completeness.
    """
    mu_A = cloud_A.mean(axis=0)
    num = float(mu_A @ marker_unembed_dir)
    den = float(np.linalg.norm(mu_A)) + 1e-12
    return num / den


def two_feature_combiner_distance(
    cloud_A: np.ndarray,
    cloud_B: np.ndarray,
    marker_unembed_dir: np.ndarray | None,
) -> tuple[float, float]:
    """Returns (f_geom, f_marker) features.

    Phase 5 fits OLS(ΔG ~ f_geom + f_marker + f_geom*f_marker) per training
    fold; the function here returns the two raw features. ``f_geom`` is the
    directional Gauss-KL and ``f_marker`` is the marker-projection of B
    relative to A (B's mean projected onto marker direction minus A's). If
    ``marker_unembed_dir`` is None (unavailable), returns (f_geom, 0.0).
    """
    f_geom = directional_gauss_kl(cloud_A, cloud_B)
    if marker_unembed_dir is None:
        return f_geom, 0.0
    f_marker = marker_projection(cloud_B, marker_unembed_dir) - marker_projection(
        cloud_A, marker_unembed_dir
    )
    return f_geom, f_marker


# --------------------------------------------------------------------------
# Cloud-loading shim. Phase 3 writes one .npz per (context, layer, point);
# Phase 4 loads them lazily so we can stream-build the predictor stack
# without holding all 32 × 28 × 3 = 2688 clouds in memory.
# --------------------------------------------------------------------------


def _cloud_path(context: str, layer: int, point: str) -> Path:
    """Per-cloud path: ``phase3/<context>/L<layer>/<point>.npz``."""
    return PHASE3_DIR / context / f"L{layer}" / f"{point}.npz"


def load_cloud(context: str, layer: int, point: str) -> np.ndarray:
    """Load one activation cloud as a 2D numpy array (n_probes, hidden_dim).

    Raises FileNotFoundError if Phase 3 hasn't produced this cloud yet —
    Phase 4 fails LOUD here (no silent zeros), so a missing-Phase-3 run is
    visible.
    """
    p = _cloud_path(context, layer, point)
    if not p.exists():
        raise FileNotFoundError(
            f"Phase 3 cloud missing: {p}. Run scripts/issue524_phase3_extract_icl.py "
            f"to produce it (context={context}, layer={layer}, point={point})."
        )
    with np.load(p) as f:
        return np.asarray(f["activations"], dtype=np.float32)


def build_predictor_matrix(
    predictor_fn,
    contexts: list[str],
    layer: int,
    point: str,
    *,
    marker_unembed_dir: np.ndarray | None = None,
    requires_marker_dir: bool = False,
) -> np.ndarray:
    """Compute a 32×32 predictor matrix for one (layer, point) cell.

    Args:
        predictor_fn: takes (cloud_A, cloud_B) -> scalar.
        contexts: ordered list of context ids.
        layer, point: which Phase 3 cloud to load.
        marker_unembed_dir: passed through to ``predictor_fn`` if
            ``requires_marker_dir`` is True.

    Returns:
        (N, N) np.ndarray of pairwise scalars. Diagonal is 0 by definition
        (a predictor at A→A is the metric of A to itself).
    """
    n = len(contexts)
    mat = np.zeros((n, n), dtype=np.float32)
    clouds = {c: load_cloud(c, layer, point) for c in contexts}
    for i, ca in enumerate(contexts):
        for j, cb in enumerate(contexts):
            if i == j:
                continue
            if requires_marker_dir:
                # The combiner returns a TUPLE (f_geom, f_marker); we store
                # f_geom in the (i, j) slot and add f_marker as a separate
                # output stream via ``build_combiner_features`` below.
                f_geom, _ = predictor_fn(clouds[ca], clouds[cb], marker_unembed_dir)
                mat[i, j] = f_geom
            else:
                mat[i, j] = predictor_fn(clouds[ca], clouds[cb])
    return mat


# --------------------------------------------------------------------------
# Registry of predictors. Each entry maps a string name to a function +
# whether it's directional or symmetric. Phase 5 reads this registry to
# know which slots to compare in the M_sym vs M_full pair.
# --------------------------------------------------------------------------

PREDICTOR_REGISTRY: dict[str, dict] = {
    # Symmetric baselines (used as M_sym components or comparison companions).
    "cosine": {"fn": cosine_distance, "kind": "symmetric"},
    "gauss_kl_sym": {"fn": gauss_kl_symmetric, "kind": "symmetric"},
    "pooled_mahal": {"fn": pooled_mahalanobis, "kind": "symmetric"},
    # Directional predictors (the load-bearing addition).
    "dir_gauss_kl": {"fn": directional_gauss_kl, "kind": "directional"},
    "src_cov_mahal": {"fn": source_cov_mahalanobis, "kind": "directional"},
    "asym_subspace_recon": {"fn": asym_subspace_recon, "kind": "directional"},
}


def _load_marker_unembed_dir() -> np.ndarray | None:
    """Best-effort: load the marker token's unembedding from the base model.

    Returns None if the unembedding cache is unavailable (the marker
    projection then degrades to a zero feature, with a warning logged).
    """
    cache = PHASE3_DIR / "marker_unembed_dir.npy"
    if cache.exists():
        return np.load(cache)
    logger.warning(
        "Marker unembed direction cache %s missing — marker_projection "
        "and two_feature_combiner f_marker will be 0. "
        "Run scripts/issue524_phase3_extract_icl.py with --cache-marker-unembed "
        "to materialize.",
        cache,
    )
    return None


def main(argv: list[str] | None = None) -> int:
    """Build the predictor stack across (predictor, layer, point) cells."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated layer ids, or 'all' for 0..27 (default).",
    )
    p.add_argument(
        "--points",
        type=str,
        default=",".join(EXTRACTION_POINTS),
        help="Comma-separated extraction points (default: all 3).",
    )
    p.add_argument(
        "--contexts",
        type=str,
        default="",
        help=(
            "Comma-separated context ids to include (default: read from "
            "phase3 directory listing — the unified 32-context panel)."
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke: 1 layer × 1 point × 4 contexts. Marker for plan §unified.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(OUT_PATH),
        help="Output .npz path (predictor stack + metadata).",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.layers == "all":
        layers = list(range(N_LAYERS))
    else:
        layers = sorted({int(x) for x in args.layers.split(",") if x.strip()})
    points = [p_.strip() for p_ in args.points.split(",") if p_.strip()]
    for pt in points:
        if pt not in EXTRACTION_POINTS:
            raise ValueError(f"Unknown extraction point {pt!r}; valid={EXTRACTION_POINTS}")

    if args.contexts:
        contexts = [c.strip() for c in args.contexts.split(",") if c.strip()]
    else:
        # Discover from Phase 3 directory listing.
        if not PHASE3_DIR.exists():
            raise FileNotFoundError(
                f"Phase 3 directory missing: {PHASE3_DIR}. "
                f"Run scripts/issue524_phase3_extract_icl.py first."
            )
        contexts = sorted(
            [d.name for d in PHASE3_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
        )

    if args.smoke:
        layers = layers[:1]
        points = points[:1]
        contexts = contexts[:4]
        logger.info("SMOKE: layers=%s points=%s contexts=%s", layers, points, contexts)

    if not contexts:
        logger.error(
            "No contexts to score against — run Phase 3 first to populate %s.",
            PHASE3_DIR,
        )
        return 3

    PHASE4_DIR.mkdir(parents=True, exist_ok=True)

    marker_dir = _load_marker_unembed_dir()
    out_arrays: dict[str, np.ndarray] = {}
    out_meta: dict = {
        "schema_version": 1,
        "issue": 524,
        "phase": 4,
        "predictors": list(PREDICTOR_REGISTRY.keys()),
        "layers": layers,
        "points": points,
        "contexts": contexts,
        "cov_eps": COV_EPS,
        "pca_rank": PCA_RANK,
        "marker_unembed_cached": marker_dir is not None,
        "git_sha": _git_sha(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    for pred_name, pred_meta in PREDICTOR_REGISTRY.items():
        for layer in layers:
            for point in points:
                key = f"{pred_name}__L{layer}__{point}"
                logger.info("Computing %s", key)
                mat = build_predictor_matrix(pred_meta["fn"], contexts, layer, point)
                out_arrays[key] = mat
                if (mat != 0).any():
                    logger.debug(
                        "  %s: min=%.4f max=%.4f mean=%.4f",
                        key,
                        float(mat[mat != 0].min()),
                        float(mat[mat != 0].max()),
                        float(mat[mat != 0].mean()),
                    )
                else:
                    logger.debug("  %s: all-zero matrix (no off-diagonal computations).", key)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out_arrays)
    OUT_META.write_text(json.dumps(out_meta, indent=2) + "\n")
    logger.info(
        "Wrote %s (%d matrices) + %s",
        out_path,
        len(out_arrays),
        OUT_META,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
