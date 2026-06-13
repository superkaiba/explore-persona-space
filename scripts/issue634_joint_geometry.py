#!/usr/bin/env python3
"""Issue #634 Phase 2: joint geometry of behavior vectors with #594 contexts.

VM-side CPU analysis (no GPU). Loads #594's ``(50, 28, 3584)`` context-vector
bank + the Phase-1 ``(275, 28, 3584)`` behavior-vector bank (both from HF) and,
per layer, runs:

- the co-embeddability gate (variance ratio + median-norm ratio behavior/context);
- a joint PCA + UMAP-grid embedding (+ t-SNE at the best layer) on the
  global-mean-centered stacked bank;
- **H1**: each Panel-B behavior's nearest of the 50 contexts (centered cosine),
  matched-family rate vs a B=1000 shuffled-``fam_map`` permutation null, with
  the max-over-layers FWER summary (``max_over_layers_summary``);
- **H2**: LOO k-NN family purity (k=4) over the 275 behaviors labeled by
  role-axis family, vs a permutation null;
- **H3**: own-region fraction = frac of behaviors whose top-4 joint-space
  neighbors are behaviors (not contexts);
- the ``beh_pc_residual`` control (remove behavior-cloud PC1, re-run H1);
- the cross-space family-centroid CKA + Procrustes fallback (matched-N), with
  the preflight shape assert the plan §5/§11 mandates.

All metric primitives + the metrics/figure helpers are REUSED from
``issue594_analyze_context_geometry.py`` (``center``, ``cosine_dist``,
``knn_neighbor_order``, ``knn_purity``, ``per_family_purity``, ``linear_cka``,
``max_over_layers_summary``, ``savefig_paper``) — no reimplementation.

Outputs: ``eval_results/issue_634/{joint_geometry_metrics.json,
per_layer_nn_purity.json, coembeddability_gate.json, cross_space_alignment.json,
panelB_nn_table.json}`` + ``figures/issue_634/*.png``.

Usage::

    uv run --extra viz python scripts/issue634_joint_geometry.py \\
        --behavior-from-hf issue634_behavior_geometry/analysis_tensors \\
        --context-from-hf issue594_context_geometry/analysis_tensors

    # synthetic tiny-bank smoke (no HF, no GPU):
    uv run --extra viz python scripts/issue634_joint_geometry.py --smoke

NOTE: ``--extra viz`` is REQUIRED — umap-learn lives in the optional ``viz``
dependency group, not the default set (same as the #594 analysis command).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue594_analyze_context_geometry import (  # noqa: E402
    center,
    cosine_dist,
    jsonable,
    knn_neighbor_order,
    knn_purity,
    linear_cka,
    max_over_layers_summary,
    quartile_layers,
)
from issue594_common import HF_DATA_REPO  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

load_dotenv()

logger = logging.getLogger("issue634_joint")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

KNN_K = 4
UMAP_GRID = [(5, 0.1), (5, 0.5), (15, 0.1), (15, 0.5), (30, 0.1), (30, 0.5)]
TSNE_PERPLEXITIES = [5, 15, 30]
SEED = 42

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_634"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_634"

DEFAULT_BEHAVIOR_PREFIX = "issue634_behavior_geometry/analysis_tensors"
DEFAULT_CONTEXT_PREFIX = "issue594_context_geometry/analysis_tensors"

# The frozen Panel-B map (Phase 0) uses LONG #594-family names; the #594 context
# tensor stores SHORT family labels. This alias table translates the frozen map's
# family value -> the context tensor's family label so H1's matched-family test
# resolves. The mapping follows plan v2 §4's YAML comments (e.g. worked_example
# = the icl worked-example context family; instruction_reword = rephrase).
# ``co_embed_only`` covers the smoke path where the frozen map already uses the
# short names. Asserted against the context tensor's realized family set on load.
FAMILY_NAME_ALIAS = {
    "persona": "persona",
    "behavior_instruction": "behavior",
    "output_format": "format",
    "instruction_reword": "rephrase",
    "worked_example": "icl",
    "bare_default": "default",
    "wildchat_prefix": "wildchat",
    # identity passthroughs (short names already match the context tensor):
    "behavior": "behavior",
    "format": "format",
    "rephrase": "rephrase",
    "icl": "icl",
    "default": "default",
    "wildchat": "wildchat",
}
NULL_ANCHOR_FAMILY_SHORT = "default"  # bare_default -> the null-anchor, not tested

# Co-embeddability scale-mismatch flag: families differing in median norm or
# variance by more than this factor route H1 to the cross-space fallback.
COEMBED_RATIO_MAX = 3.0


# ── Data loading ─────────────────────────────────────────────────────────────


def download_from_hf(prefix: str, cache_dir: Path, repo: str = HF_DATA_REPO) -> Path:
    """Fetch analysis tensors from an HF dataset repo (per-file, never snapshot)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(repo, repo_type="dataset") if f.startswith(prefix + "/")]
    if not files:
        raise RuntimeError(f"no files under {prefix}/ on {repo}")
    for f in files:
        hf_hub_download(repo, f, repo_type="dataset", local_dir=str(cache_dir))
    logger.info("Downloaded %d files from %s/%s", len(files), repo, prefix)
    return cache_dir / prefix


def load_context_bank(tensors_dir: Path) -> dict:
    """Load #594's context_vectors_mean.pt -> (N, L, H) fp32 + ids + families."""
    blob = torch.load(tensors_dir / "context_vectors_mean.pt", weights_only=True)
    mean = blob["tensor"].float().numpy()
    return {"mean": mean, "ids": list(blob["instance_ids"]), "families": list(blob["families"])}


def load_behavior_bank(tensors_dir: Path) -> dict:
    """Load Phase-1 behavior_vectors_mean.pt -> (N, L, H) fp32 + role ids + families."""
    blob = torch.load(tensors_dir / "behavior_vectors_mean.pt", weights_only=True)
    mean = blob["tensor"].float().numpy()
    return {
        "mean": mean,
        "ids": list(blob["instance_ids"]),
        "families": list(blob["families"]),
        "seed": blob.get("seed"),
        "n_questions": blob.get("n_questions"),
        "sampled_question_indices_hash": blob.get("sampled_question_indices_hash"),
    }


def load_family_map(path: Path) -> dict[str, str]:
    """Load the frozen Panel-B map; tolerate the wrapped payload or a bare map."""
    with open(path) as f:
        payload = json.load(f)
    return payload.get("map", payload)


def resolve_panel_b(
    fam_map: dict[str, str], behavior_ids: list[str], ctx_family_set: set[str]
) -> dict[str, str]:
    """Translate the frozen map to context-family labels; keep only present roles.

    Returns role -> context-family (short label), restricted to: roles actually
    present in the behavior bank, with a non-anchor translated family that
    exists in the context tensor. The null-anchor (bare_default) is excluded
    from H1's tested set. Fails loud on an unknown family name.
    """
    panel_b: dict[str, str] = {}
    present = set(behavior_ids)
    for role, fam_long in fam_map.items():
        if role not in present:
            logger.warning("Panel-B role %s absent from behavior bank; skipping", role)
            continue
        if fam_long not in FAMILY_NAME_ALIAS:
            raise ValueError(f"frozen map family {fam_long!r} (role {role}) not in alias table")
        fam_short = FAMILY_NAME_ALIAS[fam_long]
        if fam_short == NULL_ANCHOR_FAMILY_SHORT:
            continue  # null-anchor, not a tested family
        if fam_short not in ctx_family_set:
            raise ValueError(
                f"translated family {fam_short!r} (role {role}, from {fam_long!r}) "
                f"not in context tensor families {sorted(ctx_family_set)}"
            )
        panel_b[role] = fam_short
    return panel_b


# ── Co-embeddability gate ────────────────────────────────────────────────────


def coembeddability_gate(beh_layer: np.ndarray, ctx_layer: np.ndarray) -> dict:
    """Per-layer variance + median-norm ratio (behavior/context).

    A layer fails (routes H1 to the fallback) when either ratio is outside
    [1/COEMBED_RATIO_MAX, COEMBED_RATIO_MAX]. Diagnostic only — does not raise.
    """
    var_b = float(np.var(beh_layer))
    var_c = float(np.var(ctx_layer))
    norm_b = float(np.median(np.linalg.norm(beh_layer, axis=1)))
    norm_c = float(np.median(np.linalg.norm(ctx_layer, axis=1)))
    var_ratio = var_b / var_c if var_c > 0 else float("inf")
    scale_ratio = norm_b / norm_c if norm_c > 0 else float("inf")
    ok = (
        1.0 / COEMBED_RATIO_MAX <= var_ratio <= COEMBED_RATIO_MAX
        and 1.0 / COEMBED_RATIO_MAX <= scale_ratio <= COEMBED_RATIO_MAX
    )
    return {
        "var_behavior": var_b,
        "var_context": var_c,
        "var_ratio_beh_over_ctx": var_ratio,
        "median_norm_behavior": norm_b,
        "median_norm_context": norm_c,
        "scale_ratio_beh_over_ctx": scale_ratio,
        "joint_centered_ok": bool(ok),
    }


# ── H1: behavior -> nearest context family ───────────────────────────────────


def nearest_context_family(
    beh_layer: np.ndarray,
    ctx_layer: np.ndarray,
    ctx_families: list[str],
    panel_b_idx: list[int],
) -> list[str]:
    """For each Panel-B behavior, the family of its nearest of the 50 contexts.

    Centered cosine: both banks are global-mean-centered on the JOINT stack
    (50 ctx + N_b beh) so the centering matches the joint embedding's. Returns
    one family label per Panel-B behavior, in panel_b_idx order.
    """
    n_ctx = ctx_layer.shape[0]
    joint = np.vstack([ctx_layer, beh_layer])
    joint_c = center(joint)
    ctx_c = joint_c[:n_ctx]
    beh_c = joint_c[n_ctx:]
    ctx_n = ctx_c / np.clip(np.linalg.norm(ctx_c, axis=1, keepdims=True), 1e-12, None)
    out: list[str] = []
    for bi in panel_b_idx:
        v = beh_c[bi]
        v = v / max(float(np.linalg.norm(v)), 1e-12)
        sims = ctx_n @ v  # cosine to each context
        out.append(ctx_families[int(np.argmax(sims))])
    return out


def h1_matched_rate(nn_fams: list[str], expected_fams: list[str]) -> float:
    """Matched-family rate = frac of Panel-B behaviors whose NN family matches."""
    if not nn_fams:
        return float("nan")
    return float(np.mean([nn == exp for nn, exp in zip(nn_fams, expected_fams, strict=True)]))


def h1_perm_null(
    nn_fams: list[str], panel_b_roles: list[str], expected_fams: list[str], perms: list[dict]
) -> np.ndarray:
    """Matched rate under each shuffled role->family permutation.

    The geometry (the nearest-context family per behavior) is FIXED; only the
    role->expected-family labels are shuffled. Each perm is a dict role->family
    over the SAME tested family multiset (label permutation).
    """
    rates = np.empty(len(perms))
    for b, perm in enumerate(perms):
        shuffled_expected = [perm[r] for r in panel_b_roles]
        rates[b] = h1_matched_rate(nn_fams, shuffled_expected)
    return rates


def make_label_perms(panel_b: dict[str, str], n_perms: int, rng: np.random.Generator) -> list[dict]:
    """B label permutations of the role->family assignment (shared across layers).

    Permutes the family VALUES across the roles (keeps the family multiset), so
    the null is "random role->family labels" exactly as plan §5 specifies.
    """
    roles = list(panel_b.keys())
    fams = list(panel_b.values())
    perms = []
    for _ in range(n_perms):
        shuffled = list(fams)
        rng.shuffle(shuffled)
        perms.append(dict(zip(roles, shuffled, strict=True)))
    return perms


# ── H3: own-region fraction (joint-space neighbor mix) ───────────────────────


def own_region_fraction(beh_layer: np.ndarray, ctx_layer: np.ndarray, k: int = KNN_K) -> float:
    """Frac of behaviors whose top-k JOINT-space neighbors are all behaviors.

    Joint bank = [ctx; beh], global-mean-centered, cosine distance. For each
    behavior, look at its k nearest joint neighbors (self excluded) and count
    the fraction that are behaviors; own_region = mean over behaviors of
    (k neighbors all behaviors).
    """
    n_ctx = ctx_layer.shape[0]
    joint = center(np.vstack([ctx_layer, beh_layer]))
    d = cosine_dist(joint, centering="none")  # already centered
    order = knn_neighbor_order(d)
    n_total = joint.shape[0]
    is_beh = np.arange(n_total) >= n_ctx
    behavior_rows = np.where(is_beh)[0]
    frac_beh_neighbors = []
    for i in behavior_rows:
        neigh = order[i, :k]
        frac_beh_neighbors.append(float(np.mean(is_beh[neigh])))
    return float(np.mean(frac_beh_neighbors))


# ── Cross-space family-centroid fallback (matched-N) ─────────────────────────


def family_centroids(
    bank: np.ndarray, families: list[str], min_per_family: int = 1
) -> tuple[np.ndarray, list[str]]:
    """Per-family centroid bank for one layer's (N, H) matrix.

    Returns ((K, H) centroid stack, family-name list) for families with
    >= min_per_family members. Order is sorted family name (stable).
    """
    fams = np.asarray(families)
    out_fams = sorted(
        {f for f in families if f is not None and (fams == f).sum() >= min_per_family}
    )
    cents = (
        np.stack([bank[fams == f].mean(axis=0) for f in out_fams])
        if out_fams
        else np.empty((0, bank.shape[1]))
    )
    return cents, out_fams


def _procrustes_resid_lowdim(cb: np.ndarray, cc: np.ndarray) -> float:
    """Orthogonal-Procrustes alignment residual ||Cb·R - Cc||_F / ||Cc||_F.

    Both (K, H) centroid banks are projected onto the top joint PCs (<= 2K-1
    dims) BEFORE the Procrustes SVD so the SVD is (d, d) with d small, never
    (H, H). The K centroids span <= K-1 dims after centering, so the projection
    is exact for the residual (zero variance is discarded). Centering is removed
    before alignment (Procrustes is a pure rotation; a shared translation would
    otherwise inflate the residual).
    """
    from scipy.linalg import orthogonal_procrustes

    stack = np.vstack([cb, cc])
    mu = stack.mean(axis=0, keepdims=True)
    stack_c = stack - mu
    # Top joint PCs: rank is at most (2K - 1) after centering; SVD of the
    # (2K, H) centered stack is cheap (2K rows), the (H, H) cost never appears.
    _u, s, vt = np.linalg.svd(stack_c, full_matrices=False)
    keep = int((s > 1e-9 * s.max()).sum()) if s.size and s.max() > 0 else 0
    if keep < 1:
        return 0.0
    basis = vt[:keep]  # (d, H)
    cb_d = (cb - mu) @ basis.T  # (K, d)
    cc_d = (cc - mu) @ basis.T  # (K, d)
    r, _scale = orthogonal_procrustes(cb_d, cc_d)  # SVD is (d, d), d <= 2K-1
    return float(np.linalg.norm(cb_d @ r - cc_d) / max(float(np.linalg.norm(cc_d)), 1e-12))


def cross_space_alignment(
    beh_bank: np.ndarray,
    ctx_bank: np.ndarray,
    panel_b: dict[str, str],
    behavior_ids: list[str],
    ctx_families: list[str],
    n_layers: int,
) -> dict:
    """Family-centroid linear CKA + orthogonal Procrustes, per layer (matched-N).

    Aggregates the 50 contexts to per-family centroids and the Panel-B
    behaviors to per-family centroids (>=2 Panel-B roles per family), intersects
    to families in BOTH, and runs linear_cka + orthogonal_procrustes on the
    matched-N centroid banks. The preflight shape assert (plan §5/§11) enforces
    Cc.shape[0] == Cb.shape[0] in code.

    The Procrustes rotation is computed in the LOW-DIMENSIONAL subspace the K
    centroids span (a joint-PCA projection to <= 2K-1 components), NOT the raw
    H=3584-dim space: ``orthogonal_procrustes(A, B)`` SVDs ``B.T @ A`` which is
    (H, H) — an SVD of a 3584x3584 matrix per layer pegs every core for minutes
    (real hang observed in the synthetic smoke). K centroids span <= K-1 dims
    after centering, so projecting both banks onto the top joint PCs is EXACT
    for the residual ``||Cb·R - Cc||_F / ||Cc||_F`` while making the SVD (d, d)
    with d <= 2K-1 (~10). CKA is unaffected — its Gram form is already (K, K).
    """

    # Behaviors -> short context-family labels via the resolved Panel-B map.
    beh_fam_short = [panel_b.get(r) for r in behavior_ids]  # None outside Panel B
    cka_per_layer: list[float] = []
    proc_resid_per_layer: list[float] = []
    shared_families: list[str] = []
    for li in range(n_layers):
        cb_all, fam_b = family_centroids(beh_bank[:, li, :], beh_fam_short, min_per_family=2)
        cc_all, fam_c = family_centroids(ctx_bank[:, li, :], ctx_families, min_per_family=1)
        shared = sorted(set(fam_b) & set(fam_c))
        if li == 0:
            shared_families = shared
        if len(shared) < 2:
            cka_per_layer.append(float("nan"))
            proc_resid_per_layer.append(float("nan"))
            continue
        cb = np.stack([cb_all[fam_b.index(f)] for f in shared])
        cc = np.stack([cc_all[fam_c.index(f)] for f in shared])
        # PREFLIGHT SHAPE ASSERT (plan §5/§11): matched N required for linear_cka.
        assert cc.shape[0] == cb.shape[0], (
            "cross-space CKA requires matched N — aggregate to family centroids first "
            f"(Cc={cc.shape}, Cb={cb.shape}, layer={li})"
        )
        cka_per_layer.append(linear_cka(cc, cb))
        proc_resid_per_layer.append(_procrustes_resid_lowdim(cb, cc))
    return {
        "shared_families_layer0": shared_families,
        "n_shared_families": len(shared_families),
        "cka_per_layer": cka_per_layer,
        "procrustes_resid_per_layer": proc_resid_per_layer,
        "note": (
            "family-centroid linear CKA + orthogonal Procrustes on matched-N "
            "(families present in BOTH banks, >=2 Panel-B roles per behavior family)"
        ),
    }


# ── Figures ──────────────────────────────────────────────────────────────────


def _ctx_palette(ctx_family_order: list[str]) -> dict[str, str]:
    pal = paper_palette(len(ctx_family_order))
    return dict(zip(ctx_family_order, pal, strict=True))


def _joint_scatter(ax, ctx_emb, ctx_families, beh_emb, panel_b_mask, beh_ids, ctx_family_order):
    """Contexts colored by family (circles); behaviors as grey x; Panel-B labeled."""
    colors = _ctx_palette(ctx_family_order)
    for fam in ctx_family_order:
        m = np.asarray(ctx_families) == fam
        if m.any():
            ax.scatter(ctx_emb[m, 0], ctx_emb[m, 1], s=26, color=colors[fam], marker="o", label=fam)
    ax.scatter(
        beh_emb[~panel_b_mask, 0],
        beh_emb[~panel_b_mask, 1],
        s=8,
        color="0.6",
        marker="x",
        label="behavior (non-PanelB)",
    )
    ax.scatter(
        beh_emb[panel_b_mask, 0],
        beh_emb[panel_b_mask, 1],
        s=22,
        color="0.1",
        marker="^",
        label="behavior (PanelB)",
    )
    for i in np.where(panel_b_mask)[0]:
        ax.annotate(beh_ids[i], (beh_emb[i, 0], beh_emb[i, 1]), fontsize=3.5, alpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_joint_embedding(
    ctx_bank, ctx_families, beh_bank, beh_ids, panel_b_mask, qlayers, ctx_family_order, umap_module
):
    """Hero: PCA (top) + UMAP n=15/d=0.1 (bottom) at the quartile layers."""
    n_ctx = ctx_bank.shape[0]
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(2, len(qlayers), figsize=(4 * len(qlayers), 8))
    axes = np.atleast_2d(axes)
    for col, li in enumerate(qlayers):
        joint = center(np.vstack([ctx_bank[:, li, :], beh_bank[:, li, :]]))
        pca_emb = PCA(n_components=2, random_state=SEED).fit_transform(joint)
        _joint_scatter(
            axes[0, col],
            pca_emb[:n_ctx],
            ctx_families,
            pca_emb[n_ctx:],
            panel_b_mask,
            beh_ids,
            ctx_family_order,
        )
        axes[0, col].set_title(f"PCA — layer {li}")
        um = umap_module.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED)
        um_emb = um.fit_transform(joint)
        _joint_scatter(
            axes[1, col],
            um_emb[:n_ctx],
            ctx_families,
            um_emb[n_ctx:],
            panel_b_mask,
            beh_ids,
            ctx_family_order,
        )
        axes[1, col].set_title(f"UMAP (n=15, d=0.1, seed 42) — layer {li}")
    axes[0, 0].legend(fontsize=5, loc="best")
    label = "_".join(f"L{li}" for li in qlayers)
    savefig_paper(fig, f"joint_embedding_pca_umap_{label}", dir=FIG_DIR)
    plt.close(fig)


def fig_nn_rate(nn_rate, null_summary, n_layers):
    """Hero: matched-family NN rate vs layer with permutation-null band."""
    layers = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(7, 4))
    nullmean = null_summary["null_mean_per_layer"]
    null95 = null_summary["null_p95_per_layer"]
    ax.fill_between(layers, nullmean, null95, color="0.85", label="permutation null (mean to 95%)")
    ax.plot(
        layers, nn_rate, color="#1f77b4", lw=2, marker="o", ms=3, label="matched-family NN rate"
    )
    ax.axvline(
        null_summary["argmax_layer"], color="#d62728", lw=1, ls=":", label="best layer (max rate)"
    )
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("matched-family NN rate (Panel B)")
    ax.set_title("H1: behavior -> nearest-context family match vs depth")
    ax.legend(fontsize=7)
    savefig_paper(fig, "nn_rate_vs_layer", dir=FIG_DIR)
    plt.close(fig)


def fig_purity(beh_purity, null_summary, n_layers):
    """Behavior-alone k-NN family purity vs layer with permutation-null band."""
    layers = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(
        layers,
        null_summary["null_mean_per_layer"],
        null_summary["null_p95_per_layer"],
        color="0.85",
        label="permutation null (mean to 95%)",
    )
    ax.plot(
        layers, beh_purity, color="#2ca02c", lw=2, marker="o", ms=3, label="behavior-alone purity"
    )
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("k-NN family purity (k=4)")
    ax.set_title("H2: behavior-alone family purity vs depth")
    ax.legend(fontsize=7)
    savefig_paper(fig, "behavior_alone_purity_vs_layer", dir=FIG_DIR)
    plt.close(fig)


def fig_tsne_joint(
    ctx_bank, ctx_families, beh_bank, beh_ids, panel_b_mask, best_layer, ctx_family_order
):
    """t-SNE of the joint bank at the best layer (perplexity grid)."""
    from sklearn.manifold import TSNE

    n_ctx = ctx_bank.shape[0]
    joint = center(np.vstack([ctx_bank[:, best_layer, :], beh_bank[:, best_layer, :]]))
    n_samples = joint.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, perp in zip(axes.flat, TSNE_PERPLEXITIES, strict=True):
        # sklearn requires perplexity < n_samples; clamp for tiny banks (the
        # synthetic smoke has 30 samples). On the real 325-point joint bank the
        # full {5,15,30} grid is unaffected.
        eff_perp = min(perp, n_samples - 1)
        ts = TSNE(perplexity=eff_perp, metric="cosine", init="random", random_state=SEED)
        emb = ts.fit_transform(joint)
        _joint_scatter(
            ax, emb[:n_ctx], ctx_families, emb[n_ctx:], panel_b_mask, beh_ids, ctx_family_order
        )
        ax.set_title(f"t-SNE perplexity={eff_perp}, seed 42 — layer {best_layer}", fontsize=8)
    savefig_paper(fig, "tsne_joint_best_layer", dir=FIG_DIR)
    plt.close(fig)


def fig_coembed_gate(gate_per_layer, n_layers):
    layers = np.arange(n_layers)
    var_r = [g["var_ratio_beh_over_ctx"] for g in gate_per_layer]
    scale_r = [g["scale_ratio_beh_over_ctx"] for g in gate_per_layer]
    fig, ax = plt.subplots(figsize=(7, 4))
    pal = paper_palette(2)
    ax.plot(layers, var_r, color=pal[0], lw=2, marker="o", ms=3, label="variance ratio")
    ax.plot(
        layers, scale_r, color=pal[1], lw=1.5, ls="--", marker="s", ms=3, label="median-norm ratio"
    )
    ax.axhline(COEMBED_RATIO_MAX, color="0.4", lw=1, ls=":")
    ax.axhline(1.0 / COEMBED_RATIO_MAX, color="0.4", lw=1, ls=":")
    ax.axhline(1.0, color="0.7", lw=1)
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("behavior / context ratio")
    ax.set_title(
        f"Co-embeddability gate (pass band [1/{COEMBED_RATIO_MAX:.0f}, {COEMBED_RATIO_MAX:.0f}])"
    )
    ax.legend(fontsize=7)
    savefig_paper(fig, "coembeddability_gate", dir=FIG_DIR)
    plt.close(fig)


def fig_cross_space(xspace, n_layers):
    layers = np.arange(n_layers)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pal = paper_palette(2)
    axes[0].plot(layers, xspace["cka_per_layer"], color=pal[0], lw=2, marker="o", ms=3)
    axes[0].set_xlabel("decoder layer")
    axes[0].set_ylabel("family-centroid linear CKA")
    axes[0].set_title("Cross-space CKA (matched-N family centroids)")
    axes[0].set_ylim(0, 1)
    axes[1].plot(layers, xspace["procrustes_resid_per_layer"], color=pal[1], lw=2, marker="s", ms=3)
    axes[1].set_xlabel("decoder layer")
    axes[1].set_ylabel("Procrustes residual ||Cb·R - Cc||/||Cc||")
    axes[1].set_title("Cross-space Procrustes alignment residual")
    savefig_paper(fig, "cross_space_alignment", dir=FIG_DIR)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def make_synthetic_banks(seed: int = SEED) -> tuple[dict, dict, dict]:
    """Tiny synthetic banks for --smoke (no HF, no GPU).

    Context (10, 28, 3584) over 3 short context-families; behavior (20, 28, 3584)
    over the SAME short families (so resolve_panel_b + the centroid fallback both
    exercise unequal-N aggregation). Returns (ctx, beh, fam_map) where fam_map
    uses short family names directly (passthrough alias).
    """
    rng = np.random.default_rng(seed)
    n_layers, hidden = 28, 3584
    ctx_fams = ["persona", "behavior", "format"]
    ctx_ids = [f"ctx_{f}_{i}" for f in ctx_fams for i in range(3)] + ["ctx_default_0"]
    ctx_families = [i.split("_")[1] for i in ctx_ids]
    n_ctx = len(ctx_ids)
    ctx = rng.standard_normal((n_ctx, n_layers, hidden)).astype(np.float32)
    beh_fams = ["persona", "persona", "behavior", "behavior", "format", "format"]
    beh_ids = [f"beh_{f}_{i}" for i, f in enumerate(beh_fams)] + [
        f"beh_extra_{i}" for i in range(14)
    ]
    beh = rng.standard_normal((len(beh_ids), n_layers, hidden)).astype(np.float32)
    fam_map = {bid: f for bid, f in zip(beh_ids[:6], beh_fams, strict=True)}
    fam_map["beh_extra_0"] = "default"  # null-anchor exercised (dropped from Panel B)
    ctx_bank = {"mean": ctx, "ids": ctx_ids, "families": ctx_families}
    beh_bank = {
        "mean": beh,
        "ids": beh_ids,
        "families": [fam_map.get(b) for b in beh_ids],
        "seed": seed,
        "n_questions": 2,
        "sampled_question_indices_hash": "synthetic",
    }
    return ctx_bank, beh_bank, fam_map


def main() -> int:
    global EVAL_DIR, FIG_DIR
    parser = argparse.ArgumentParser(description="Issue #634 Phase 2 joint geometry analysis.")
    parser.add_argument("--behavior-from-hf", default=DEFAULT_BEHAVIOR_PREFIX)
    parser.add_argument("--context-from-hf", default=DEFAULT_CONTEXT_PREFIX)
    parser.add_argument(
        "--behavior-dir", type=Path, default=None, help="local behavior tensors dir"
    )
    parser.add_argument("--context-dir", type=Path, default=None, help="local context tensors dir")
    parser.add_argument("--hf-repo", default=HF_DATA_REPO)
    parser.add_argument(
        "--family-map",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue634" / "behavior_family_map.json",
    )
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--n-perms", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the full pipeline on a tiny synthetic bank (no HF, no GPU)",
    )
    args = parser.parse_args()

    EVAL_DIR = args.eval_dir
    FIG_DIR = args.fig_dir
    set_paper_style("blog")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        logger.info("SMOKE: synthetic banks (no HF, no GPU)")
        ctx_bank, beh_bank, fam_map = make_synthetic_banks(args.seed)
    else:
        ctx_dir = args.context_dir or download_from_hf(
            args.context_from_hf, EVAL_DIR / "hf_context_cache", repo=args.hf_repo
        )
        beh_dir = args.behavior_dir or download_from_hf(
            args.behavior_from_hf, EVAL_DIR / "hf_behavior_cache", repo=args.hf_repo
        )
        ctx_bank = load_context_bank(ctx_dir)
        beh_bank = load_behavior_bank(beh_dir)
        fam_map = load_family_map(args.family_map)

    ctx = ctx_bank["mean"]  # (50, L, H)
    beh = beh_bank["mean"]  # (275, L, H)
    ctx_families = ctx_bank["families"]
    beh_ids = beh_bank["ids"]
    n_ctx, n_layers, hidden = ctx.shape
    assert beh.shape[1] == n_layers and beh.shape[2] == hidden, (beh.shape, ctx.shape)
    ctx_family_set = set(ctx_families)
    ctx_family_order = sorted(ctx_family_set)
    logger.info(
        "Loaded banks: ctx %s (%d families), beh %s",
        ctx.shape,
        len(ctx_family_set),
        beh.shape,
    )

    panel_b = resolve_panel_b(fam_map, beh_ids, ctx_family_set)
    panel_b_roles = list(panel_b.keys())
    panel_b_idx = [beh_ids.index(r) for r in panel_b_roles]
    expected_fams = [panel_b[r] for r in panel_b_roles]
    panel_b_mask = np.zeros(len(beh_ids), dtype=bool)
    panel_b_mask[panel_b_idx] = True
    n_tested_families = len(set(expected_fams))
    meets_floor = len(panel_b_roles) >= 12 and all(
        sum(1 for f in expected_fams if f == fam) >= 2 for fam in set(expected_fams)
    )
    logger.info(
        "Panel B: %d roles / %d tested families, meets_floor=%s",
        len(panel_b_roles),
        n_tested_families,
        meets_floor,
    )

    # Behavior-alone family labels for H2 (role-axis = the frozen-map family).
    beh_fam_labels = np.asarray([fam for fam in beh_bank["families"]])
    has_fam = np.array([f is not None for f in beh_fam_labels])

    rng = np.random.default_rng(args.seed)
    label_perms = make_label_perms(panel_b, args.n_perms, rng)

    # ── Per-layer loop ───────────────────────────────────────────────────────
    gate_per_layer: list[dict] = []
    nn_rate = np.empty(n_layers)
    nn_rate_resid = np.empty(n_layers)
    null_nn = np.empty((args.n_perms, n_layers))
    beh_purity = np.empty(n_layers)
    null_pur = np.empty((args.n_perms, n_layers))
    own_region = np.empty(n_layers)
    panelB_nn_table: dict[str, list[str]] = {r: [] for r in panel_b_roles}

    # H2 purity over behaviors WITH a family label (role-axis); permutation null
    # shuffles those labels. Built once (labels fixed across layers).
    fam_idx = np.where(has_fam)[0]
    fam_labels_sub = beh_fam_labels[fam_idx]
    pur_perms = np.stack([rng.permutation(len(fam_idx)) for _ in range(args.n_perms)])

    for li in range(n_layers):
        beh_li = beh[:, li, :]
        ctx_li = ctx[:, li, :]
        gate_per_layer.append(coembeddability_gate(beh_li, ctx_li))

        # H1
        nn_fams = nearest_context_family(beh_li, ctx_li, ctx_families, panel_b_idx)
        for r, nn in zip(panel_b_roles, nn_fams, strict=True):
            panelB_nn_table[r].append(nn)
        nn_rate[li] = h1_matched_rate(nn_fams, expected_fams)
        null_nn[:, li] = h1_perm_null(nn_fams, panel_b_roles, expected_fams, label_perms)

        # H1 control: remove behavior-cloud PC1, re-run
        from sklearn.decomposition import PCA

        beh_c = center(beh_li)
        pc1 = PCA(n_components=1, random_state=SEED).fit(beh_c).components_[0]
        beh_resid = beh_c - np.outer(beh_c @ pc1, pc1)
        nn_fams_r = nearest_context_family(beh_resid, ctx_li, ctx_families, panel_b_idx)
        nn_rate_resid[li] = h1_matched_rate(nn_fams_r, expected_fams)

        # H2 behavior-alone purity (over family-labeled behaviors)
        d_beh = cosine_dist(beh_li[fam_idx], centering="global_mean")
        order_beh = knn_neighbor_order(d_beh)
        beh_purity[li] = knn_purity(order_beh, fam_labels_sub, k=KNN_K)
        null_pur[:, li] = np.array(
            [knn_purity(order_beh, fam_labels_sub[p], k=KNN_K) for p in pur_perms]
        )

        # H3 own-region fraction
        own_region[li] = own_region_fraction(beh_li, ctx_li, k=KNN_K)
        if li % 7 == 0:
            logger.info("layer %d/%d done (nn_rate=%.3f)", li + 1, n_layers, nn_rate[li])

    nn_summary = max_over_layers_summary(nn_rate.tolist(), null_nn)
    nn_resid_summary = max_over_layers_summary(nn_rate_resid.tolist(), null_nn)
    pur_summary = max_over_layers_summary(beh_purity.tolist(), null_pur)
    best_layer = nn_summary["argmax_layer"]

    # ── Cross-space fallback (always computed; the headline read on gate fail) ─
    logger.info("cross-space family-centroid CKA + Procrustes")
    xspace = cross_space_alignment(beh, ctx, panel_b, beh_ids, ctx_families, n_layers)

    gate_any_fail = any(not g["joint_centered_ok"] for g in gate_per_layer)

    # ── Assemble JSONs (checkpoint each before figures) ──────────────────────
    coembed_json = {
        "ratio_pass_band": [1.0 / COEMBED_RATIO_MAX, COEMBED_RATIO_MAX],
        "per_layer": gate_per_layer,
        "any_layer_fails_gate": bool(gate_any_fail),
        "n_layers_fail": int(sum(1 for g in gate_per_layer if not g["joint_centered_ok"])),
    }
    with open(EVAL_DIR / "coembeddability_gate.json", "w") as f:
        json.dump(jsonable(coembed_json), f, indent=2)

    per_layer_nn = {
        "knn_k": KNN_K,
        "n_perms": args.n_perms,
        "seed": args.seed,
        "panel_b_roles": panel_b_roles,
        "panel_b_expected_families": expected_fams,
        "nn_rate_per_layer": nn_rate.tolist(),
        "nn_rate_residualized_per_layer": nn_rate_resid.tolist(),
        "beh_alone_purity_per_layer": beh_purity.tolist(),
        "own_region_fraction_per_layer": own_region.tolist(),
        "nn_summary": nn_summary,
        "nn_residualized_summary": nn_resid_summary,
        "purity_summary": pur_summary,
    }
    with open(EVAL_DIR / "per_layer_nn_purity.json", "w") as f:
        json.dump(jsonable(per_layer_nn), f, indent=2)

    with open(EVAL_DIR / "panelB_nn_table.json", "w") as f:
        json.dump(
            jsonable({"panel_b_expected": panel_b, "nearest_family_per_layer": panelB_nn_table}),
            f,
            indent=2,
        )

    xspace_out = dict(xspace)
    with open(EVAL_DIR / "cross_space_alignment.json", "w") as f:
        json.dump(jsonable(xspace_out), f, indent=2)

    # H1 floor precondition + verdict
    if not meets_floor:
        h1_verdict = "UNDERPOWERED — Panel B below the pre-registered floor"
    elif nn_summary["passes"]:
        h1_verdict = "H1 SUPPORTED — matched-family NN rate > null 95th pct at best layer"
    else:
        h1_verdict = "H1 NOT SUPPORTED — NN rate indistinguishable from shuffled-label null"

    metrics = {
        "n_context": int(n_ctx),
        "n_behavior": int(beh.shape[0]),
        "n_layers": int(n_layers),
        "hidden": int(hidden),
        "knn_k": KNN_K,
        "n_perms": args.n_perms,
        "seed": args.seed,
        "behavior_seed": beh_bank.get("seed"),
        "behavior_n_questions": beh_bank.get("n_questions"),
        "behavior_sampled_question_indices_hash": beh_bank.get("sampled_question_indices_hash"),
        "panel_b": panel_b,
        "panel_b_n_roles": len(panel_b_roles),
        "panel_b_n_tested_families": n_tested_families,
        "panel_b_meets_floor": bool(meets_floor),
        "best_layer_by_nn_rate": int(best_layer),
        "coembeddability_gate_any_fail": bool(gate_any_fail),
        "h1_nn_summary": nn_summary,
        "h1_residualized_summary": nn_resid_summary,
        "h1_verdict": h1_verdict,
        "h2_purity_summary": pur_summary,
        "h3_own_region_fraction_per_layer": own_region.tolist(),
        "h3_own_region_fraction_best_layer": float(own_region[best_layer]),
        "cross_space_alignment": xspace,
        "quartile_layers": quartile_layers(n_layers),
        "family_name_alias": FAMILY_NAME_ALIAS,
        "metadata": reproducibility_metadata({"script": "issue634_joint_geometry"}),
    }
    with open(EVAL_DIR / "joint_geometry_metrics.json", "w") as f:
        json.dump(jsonable(metrics), f, indent=2)
    logger.info("Wrote metrics JSONs to %s", EVAL_DIR)

    # ── Figures ──────────────────────────────────────────────────────────────
    logger.info("figures")
    import umap

    qlayers = quartile_layers(n_layers)
    fig_joint_embedding(
        ctx, ctx_families, beh, beh_ids, panel_b_mask, qlayers, ctx_family_order, umap
    )
    fig_nn_rate(nn_rate.tolist(), nn_summary, n_layers)
    fig_purity(beh_purity.tolist(), pur_summary, n_layers)
    fig_tsne_joint(ctx, ctx_families, beh, beh_ids, panel_b_mask, best_layer, ctx_family_order)
    fig_coembed_gate(gate_per_layer, n_layers)
    fig_cross_space(xspace, n_layers)

    logger.info(
        "Done. best_layer=%d, nn_rate_max=%.3f (null95 %.3f), %s; gate_fail=%s; H2 purity_max=%.3f",
        best_layer,
        nn_summary["observed_max"],
        nn_summary["null_max_p95"],
        h1_verdict,
        gate_any_fail,
        pur_summary["observed_max"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
