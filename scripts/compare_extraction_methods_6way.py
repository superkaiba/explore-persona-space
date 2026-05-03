#!/usr/bin/env python3
"""
6-way comparison of persona-vector extraction methods (Issue #201).

Loads centroid dicts from all 7 method directories (a, b, bstar, bstar_no_last,
c1, c2, c3) and computes per-pair, per-layer metrics:
  - Per-persona cosine similarity (mean, min, p5, max)
  - Inter-persona cosine matrix (raw + mean-centered) Pearson/Spearman
  - Per-question persona spread Spearman (descriptive only, N/A for C1 pairs)
  - Noise-floor positive control (same-method cross-half cosine)

Verdict logic partitions the 15 pairs into PASS / GREY / KILL at each layer.

Usage:
  python scripts/compare_extraction_methods_6way.py \\
      --centroid-root data/persona_vectors/issue_201/qwen2.5-7b-instruct \\
      --layers 7 14 21 27 \\
      --output-dir eval_results/issue_201
"""

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

# Attempt to import paper_plots utilities; fall back gracefully if unavailable
try:
    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    HAS_PAPER_PLOTS = True
except ImportError:
    HAS_PAPER_PLOTS = False

# ── Constants ────────────────────────────────────────────────────────────────

# The 6 primary methods (bstar_no_last is a sub-field of (a, bstar), not a separate pair)
METHODS_6 = ["a", "b", "bstar", "c1", "c2", "c3"]
ALL_PAIRS = list(combinations(METHODS_6, 2))  # 15 pairs

LOAD_BEARING_PAIRS = {
    ("a", "b"),
    ("a", "bstar"),
    ("a", "c1"),
    ("b", "bstar"),
    ("bstar", "c1"),
}
SANITY_PAIRS = {("a", "c2"), ("a", "c3"), ("c2", "c3")}

# Thresholds
SUCCESS_COS_MIN = 0.95
SUCCESS_MC_R = 0.90
KILL_COS_MIN = 0.85
KILL_MC_R = 0.70

DEFAULT_LAYERS = [7, 14, 21, 27]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _git_commit_hash() -> str:
    """Return short git hash or 'unknown'."""
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


def cosine_matrix(centroids: torch.Tensor) -> np.ndarray:
    """Compute pairwise cosine similarity matrix from (N, D) tensor.

    Returns (N, N) numpy array.
    """
    C_norm = F.normalize(centroids, dim=1)
    return (C_norm @ C_norm.T).numpy()


def mean_center_cosine_matrix(centroids: torch.Tensor) -> np.ndarray:
    """Mean-center centroids, then compute cosine matrix.

    Returns (N, N) numpy array.
    """
    global_mean = centroids.mean(dim=0, keepdim=True)
    centered = centroids - global_mean
    return cosine_matrix(centered)


def off_diag_upper(matrix: np.ndarray) -> np.ndarray:
    """Extract upper-triangle off-diagonal values from a square matrix."""
    n = matrix.shape[0]
    indices = np.triu_indices(n, k=1)
    return matrix[indices]


def load_per_q_centroids(
    method: str,
    layer_idx: int,
    roles: list[str],
    root: Path,
    q_slice: slice,
) -> torch.Tensor:
    """Re-aggregate centroids from per-question caches over a question slice.

    Args:
        method: Method name (a, b, bstar, c2, c3).
        layer_idx: Index into the layers dimension of per_q tensors.
        roles: Sorted list of role names.
        root: Centroid root path.
        q_slice: Which questions to include (e.g., slice(0, 120)).

    Returns:
        Tensor of shape (N_roles, hidden_dim) — fp32 centroids.
    """
    method_dir = root / f"method_{method}"
    centroids = []
    for role in roles:
        per_q_path = method_dir / f"{role}__per_q.pt"
        per_q = torch.load(per_q_path, weights_only=True)  # (n_q, n_layers, D) fp16
        subset = per_q[q_slice, layer_idx, :].float()  # (n_subset, D)
        centroids.append(subset.mean(dim=0))
    return torch.stack(centroids)  # (N, D)


def compute_per_question_persona_spread(
    method_x: str,
    method_y: str,
    layer_idx: int,
    roles: list[str],
    root: Path,
    n_questions: int,
) -> float | None:
    """Compute per-question persona spread Spearman between two methods.

    For each question, compute between-persona variance for each method.
    Then compute Spearman rank correlation across the n_questions values.

    Returns Spearman r, or None if either method is C1 (no per-q variation).
    """
    if method_x == "c1" or method_y == "c1":
        return None

    spread_x = []
    spread_y = []

    for q_idx in range(n_questions):
        # Load per-question activations for all roles
        vecs_x = []
        vecs_y = []
        dir_x = root / f"method_{method_x}"
        dir_y = root / f"method_{method_y}"

        valid = True
        for role in roles:
            px = dir_x / f"{role}__per_q.pt"
            py = dir_y / f"{role}__per_q.pt"
            if not px.exists() or not py.exists():
                valid = False
                break
            per_q_x = torch.load(px, weights_only=True)  # (n_q, n_layers, D) fp16
            per_q_y = torch.load(py, weights_only=True)
            # Check we have this question index
            if q_idx >= per_q_x.shape[0] or q_idx >= per_q_y.shape[0]:
                valid = False
                break
            vecs_x.append(per_q_x[q_idx, layer_idx, :].float())
            vecs_y.append(per_q_y[q_idx, layer_idx, :].float())

        if not valid or len(vecs_x) < 2:
            continue

        vecs_x_t = torch.stack(vecs_x)  # (N_roles, D)
        vecs_y_t = torch.stack(vecs_y)

        # Normalize to unit vectors, compute between-persona variance
        vx_norm = F.normalize(vecs_x_t, dim=1)
        centroid_x = vx_norm.mean(dim=0)
        var_x = ((vx_norm - centroid_x) ** 2).sum(dim=1).mean().item()

        vy_norm = F.normalize(vecs_y_t, dim=1)
        centroid_y = vy_norm.mean(dim=0)
        var_y = ((vy_norm - centroid_y) ** 2).sum(dim=1).mean().item()

        spread_x.append(var_x)
        spread_y.append(var_y)

    if len(spread_x) < 10:
        return None

    sp_r, _ = stats.spearmanr(spread_x, spread_y)
    return float(sp_r)


# ── Figure Generation ────────────────────────────────────────────────────────


def _pair_label(x: str, y: str) -> str:
    """Human-readable pair label."""
    name_map = {
        "a": "A",
        "b": "B",
        "bstar": "B*",
        "bstar_no_last": "B*\\u2212last",
        "c1": "C1",
        "c2": "C2",
        "c3": "C3",
    }
    return f"{name_map.get(x, x)}\u2194{name_map.get(y, y)}"


def _pair_category(x: str, y: str) -> str:
    """Classify a pair as load-bearing, sanity, or tiebreak."""
    pair = (x, y) if x < y else (y, x)
    if pair in LOAD_BEARING_PAIRS:
        return "load-bearing"
    if pair in SANITY_PAIRS:
        return "sanity"
    return "tiebreak"


def _ordered_pairs() -> list[tuple[str, str]]:
    """Return ALL_PAIRS ordered: load-bearing first, then sanity, then tiebreak."""
    lb = [p for p in ALL_PAIRS if _pair_category(*p) == "load-bearing"]
    san = [p for p in ALL_PAIRS if _pair_category(*p) == "sanity"]
    tie = [p for p in ALL_PAIRS if _pair_category(*p) == "tiebreak"]
    return lb + san + tie


def generate_verdict_heatmap(partition: dict, layers: list[int], output_dir: Path) -> Path:
    """Generate verdict partition heatmap (15 pairs x 4 layers, RAG colored)."""
    if HAS_PAPER_PLOTS:
        set_paper_style("neurips")

    ordered = _ordered_pairs()
    pair_labels = [_pair_label(x, y) for x, y in ordered]

    # Build matrix: rows = pairs, cols = layers
    color_map = {"PASS": 0, "GREY": 1, "KILL": 2}
    matrix = np.zeros((len(ordered), len(layers)))
    for i, (x, y) in enumerate(ordered):
        for j, layer in enumerate(layers):
            matrix[i, j] = color_map.get(partition.get((x, y, layer), "GREY"), 1)

    fig, ax = plt.subplots(figsize=(5.5, 7.5))

    # Custom colormap: green, amber, red
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#4CAF50", "#FFC107", "#F44336"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto")

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{lay}" for lay in layers])
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(pair_labels, fontsize=8)
    ax.set_xlabel("Layer")
    ax.set_title("Which extraction recipes agree?", fontsize=11)

    # Add cell text
    label_map = {0: "PASS", 1: "GREY", 2: "KILL"}
    for i in range(len(ordered)):
        for j in range(len(layers)):
            val = int(matrix[i, j])
            text_color = "white" if val == 2 else "black"
            ax.text(
                j,
                i,
                label_map[val],
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
                fontweight="bold",
            )

    # Separator lines between pair categories
    n_lb = sum(1 for p in ordered if _pair_category(*p) == "load-bearing")
    n_san = sum(1 for p in ordered if _pair_category(*p) == "sanity")
    ax.axhline(n_lb - 0.5, color="black", linewidth=1.5)
    ax.axhline(n_lb + n_san - 0.5, color="black", linewidth=1.5)

    # Category labels on right side
    ax.text(
        len(layers) + 0.2,
        (n_lb - 1) / 2,
        "load-bearing",
        va="center",
        fontsize=7,
        fontstyle="italic",
        color="grey",
    )
    ax.text(
        len(layers) + 0.2,
        n_lb + (n_san - 1) / 2,
        "sanity",
        va="center",
        fontsize=7,
        fontstyle="italic",
        color="grey",
    )
    ax.text(
        len(layers) + 0.2,
        n_lb + n_san + (len(ordered) - n_lb - n_san - 1) / 2,
        "tiebreak",
        va="center",
        fontsize=7,
        fontstyle="italic",
        color="grey",
    )

    fig.tight_layout()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "verdict_partition_heatmap.png"

    if HAS_PAPER_PLOTS:
        savefig_paper(fig, "verdict_partition_heatmap", dir=str(fig_dir))
    else:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return out_path


def generate_per_persona_cos_violins(results: dict, layers: list[int], output_dir: Path) -> Path:
    """Generate per-persona cosine violin plots (15 pairs x 4 layers)."""
    if HAS_PAPER_PLOTS:
        set_paper_style("neurips")

    ordered = _ordered_pairs()
    n_pairs = len(ordered)

    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 8), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for l_idx, (ax, layer) in enumerate(zip(axes, layers, strict=True)):
        data = []
        labels = []
        colors = []
        for x, y in ordered:
            key = f"{x}__{y}__layer{layer}"
            per_persona = results.get(key, {}).get("per_persona", {})
            # Reconstruct the distribution from stats (we store the full array in run_result)
            cos_array = per_persona.get("_values", [])
            if cos_array:
                data.append(cos_array)
            else:
                # Fallback: create a placeholder from stats
                data.append([per_persona.get("mean", 0.0)])
            labels.append(_pair_label(x, y))
            cat = _pair_category(x, y)
            colors.append(
                "#0072B2" if cat == "load-bearing" else "#E69F00" if cat == "sanity" else "#009E73"
            )

        parts = ax.violinplot(data, positions=range(n_pairs), showmedians=True, showextrema=True)
        for pc_idx, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[pc_idx])
            pc.set_alpha(0.7)

        ax.axhline(SUCCESS_COS_MIN, color="green", linestyle="--", alpha=0.5, label="0.95")
        ax.axhline(KILL_COS_MIN, color="red", linestyle="--", alpha=0.5, label="0.85")
        ax.set_xticks(range(n_pairs))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_title(f"Layer {layer}", fontsize=10)
        if l_idx == 0:
            ax.set_ylabel("Per-persona cosine similarity")
            ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(
        "Per-persona cos(centroid_X, centroid_Y) across extraction methods", fontsize=11, y=1.01
    )
    fig.tight_layout()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "per_persona_cos_pair_dists.png"

    if HAS_PAPER_PLOTS:
        savefig_paper(fig, "per_persona_cos_pair_dists", dir=str(fig_dir))
    else:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return out_path


def generate_matrix_corr_bars(
    results: dict, layers: list[int], noise_floor: dict, output_dir: Path
) -> Path:
    """Generate 15x4 grouped bar chart of mean-centered Pearson r."""
    if HAS_PAPER_PLOTS:
        set_paper_style("neurips")

    ordered = _ordered_pairs()
    n_pairs = len(ordered)

    fig, ax = plt.subplots(figsize=(14, 5))

    bar_width = 0.18
    x_positions = np.arange(n_pairs)

    layer_colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    if HAS_PAPER_PLOTS:
        layer_colors = paper_palette(len(layers))

    for l_idx, layer in enumerate(layers):
        vals = []
        for x, y in ordered:
            key = f"{x}__{y}__layer{layer}"
            mc_r = results.get(key, {}).get("matrix_mc", {}).get("pearson_r", 0.0)
            vals.append(mc_r)

        offset = (l_idx - len(layers) / 2 + 0.5) * bar_width
        ax.bar(
            x_positions + offset,
            vals,
            width=bar_width,
            label=f"L{layer}",
            color=layer_colors[l_idx],
            alpha=0.8,
        )

    ax.axhline(SUCCESS_MC_R, color="green", linestyle="--", alpha=0.5, label="0.90 (success)")
    ax.axhline(KILL_MC_R, color="red", linestyle="--", alpha=0.5, label="0.70 (kill)")

    # Noise-floor band (take min/max across methods for each layer)
    for _l_idx, layer in enumerate(layers):
        nf_vals = [
            v.get("matrix_mc_pearson_r", 1.0)
            for k, v in noise_floor.items()
            if k.endswith(f"__layer{layer}")
        ]
        if nf_vals:
            nf_min = min(nf_vals)
            ax.axhspan(nf_min, 1.0, alpha=0.05, color="grey")

    pair_labels = [_pair_label(x, y) for x, y in ordered]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean-centered Pearson r")
    ax.set_title("Inter-persona cosine matrix correlation across methods", fontsize=11)
    ax.legend(fontsize=7, ncol=3, loc="lower left")
    ax.set_ylim(-0.1, 1.05)
    fig.tight_layout()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "matrix_corr_pair_layer.png"

    if HAS_PAPER_PLOTS:
        savefig_paper(fig, "matrix_corr_pair_layer", dir=str(fig_dir))
    else:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────


def _compute_cross_method_metrics(
    cents: dict,
    roles: list[str],
    layers: list[int],
    root: Path,
    n_questions: int,
    skip_spread: bool = False,
) -> dict:
    """Compute per-pair, per-layer cross-method metrics."""
    results = {}
    for x, y in ALL_PAIRS:
        if x not in cents or y not in cents:
            print(f"  Skipping pair ({x}, {y}) — missing centroid data")
            continue
        for l_idx, layer in enumerate(layers):
            cx = torch.stack([cents[x][r][l_idx] for r in roles])
            cy = torch.stack([cents[y][r][l_idx] for r in roles])
            per_persona = F.cosine_similarity(cx, cy, dim=1).numpy()

            cmx, cmy = cosine_matrix(cx), cosine_matrix(cy)
            odx, ody = off_diag_upper(cmx), off_diag_upper(cmy)
            pearson_r_raw, _ = stats.pearsonr(odx, ody)
            spearman_r_raw, _ = stats.spearmanr(odx, ody)

            cmx_mc, cmy_mc = mean_center_cosine_matrix(cx), mean_center_cosine_matrix(cy)
            odx_mc, ody_mc = off_diag_upper(cmx_mc), off_diag_upper(cmy_mc)
            pearson_r_mc, _ = stats.pearsonr(odx_mc, ody_mc)

            per_q_spearman = None
            if not skip_spread:
                per_q_spearman = compute_per_question_persona_spread(
                    method_x=x,
                    method_y=y,
                    layer_idx=l_idx,
                    roles=roles,
                    root=root,
                    n_questions=n_questions,
                )
            key = f"{x}__{y}__layer{layer}"
            results[key] = {
                "pair": [x, y],
                "layer": layer,
                "per_persona": {
                    "min": float(per_persona.min()),
                    "p5": float(np.percentile(per_persona, 5)),
                    "mean": float(per_persona.mean()),
                    "max": float(per_persona.max()),
                    "_values": per_persona.tolist(),
                },
                "matrix_raw": {
                    "pearson_r": float(pearson_r_raw),
                    "spearman_r": float(spearman_r_raw),
                },
                "matrix_mc": {"pearson_r": float(pearson_r_mc)},
                "per_question_persona_spread": {"spearman_r": per_q_spearman},
                "category": _pair_category(x, y),
            }
            print(
                f"  {_pair_label(x, y):>12s} L{layer}: "
                f"cos_min={per_persona.min():.4f} cos_mean={per_persona.mean():.4f} "
                f"mc_r={pearson_r_mc:.4f} [{_pair_category(x, y)}]"
            )
    return results


def _compute_bstar_no_last_sensitivity(
    cents: dict, roles: list[str], layers: list[int], results: dict
) -> None:
    """Add B*_no_last sensitivity check as a sub-field of (a, bstar) cells."""
    if "bstar_no_last" not in cents:
        return
    print("\n  Computing B*_no_last sensitivity check for (A, B*) cell...")
    for l_idx, layer in enumerate(layers):
        cx = torch.stack([cents["a"][r][l_idx] for r in roles])
        cy_no_last = torch.stack([cents["bstar_no_last"][r][l_idx] for r in roles])
        per_persona_no_last = F.cosine_similarity(cx, cy_no_last, dim=1).numpy()
        cmx_mc = mean_center_cosine_matrix(cx)
        cmy_mc_no_last = mean_center_cosine_matrix(cy_no_last)
        odx_mc = off_diag_upper(cmx_mc)
        ody_mc_no_last = off_diag_upper(cmy_mc_no_last)
        pearson_r_mc_no_last, _ = stats.pearsonr(odx_mc, ody_mc_no_last)
        key = f"a__bstar__layer{layer}"
        if key in results:
            results[key]["bstar_no_last_sensitivity"] = {
                "per_persona": {
                    "min": float(per_persona_no_last.min()),
                    "p5": float(np.percentile(per_persona_no_last, 5)),
                    "mean": float(per_persona_no_last.mean()),
                    "max": float(per_persona_no_last.max()),
                },
                "matrix_mc": {"pearson_r": float(pearson_r_mc_no_last)},
            }
            print(
                f"    A<->B*_no_last L{layer}: "
                f"cos_min={per_persona_no_last.min():.4f} "
                f"mc_r={pearson_r_mc_no_last:.4f}"
            )


def _compute_noise_floor(roles: list[str], layers: list[int], root: Path, n_questions: int) -> dict:
    """Compute same-method cross-half noise floor."""
    print("\nComputing noise-floor positive control (same-method cross-half cosine)...")
    noise_floor = {}
    half1 = slice(0, n_questions // 2)
    half2 = slice(n_questions // 2, n_questions)
    for m in METHODS_6:
        if m == "c1":
            continue
        method_dir = root / f"method_{m}"
        sample_pq = method_dir / f"{roles[0]}__per_q.pt"
        if not sample_pq.exists():
            print(f"  Skipping noise floor for {m} — no per_q caches")
            continue
        for l_idx, layer in enumerate(layers):
            try:
                cents_h1 = load_per_q_centroids(m, l_idx, roles, root, half1)
                cents_h2 = load_per_q_centroids(m, l_idx, roles, root, half2)
            except Exception as exc:
                print(f"  WARNING: noise floor {m} L{layer} failed: {exc}")
                continue
            per_persona = F.cosine_similarity(cents_h1, cents_h2, dim=1).numpy()
            cm1_mc = mean_center_cosine_matrix(cents_h1)
            cm2_mc = mean_center_cosine_matrix(cents_h2)
            pearson_r_mc, _ = stats.pearsonr(off_diag_upper(cm1_mc), off_diag_upper(cm2_mc))
            nf_key = f"{m}__layer{layer}"
            noise_floor[nf_key] = {
                "method": m,
                "layer": layer,
                "per_persona_min": float(per_persona.min()),
                "per_persona_p5": float(np.percentile(per_persona, 5)),
                "per_persona_mean": float(per_persona.mean()),
                "matrix_mc_pearson_r": float(pearson_r_mc),
            }
            print(
                f"  {m} L{layer}: cos_min={per_persona.min():.4f} "
                f"cos_mean={per_persona.mean():.4f} mc_r={pearson_r_mc:.4f}"
            )
    return noise_floor


def _compute_verdict(results: dict, layers: list[int]) -> tuple[dict, dict, dict]:
    """Compute cell-by-cell partition and aggregate verdict.

    Returns (partition, verdict_dict, verdict_summary).
    """
    partition = {}
    for x, y in ALL_PAIRS:
        for layer in layers:
            key = f"{x}__{y}__layer{layer}"
            if key not in results:
                partition[(x, y, layer)] = "GREY"
                continue
            r = results[key]
            cos_min = r["per_persona"]["min"]
            mc_r = r["matrix_mc"]["pearson_r"]
            if cos_min > SUCCESS_COS_MIN and mc_r > SUCCESS_MC_R:
                partition[(x, y, layer)] = "PASS"
            elif cos_min < KILL_COS_MIN or mc_r < KILL_MC_R:
                partition[(x, y, layer)] = "KILL"
            else:
                partition[(x, y, layer)] = "GREY"

    success_all_lb = all(
        partition.get((x, y, layer)) == "PASS"
        for (x, y) in ALL_PAIRS
        for layer in layers
        if (x, y) in LOAD_BEARING_PAIRS
    )
    kill_any_lb = any(
        partition.get((x, y, layer)) == "KILL"
        for (x, y) in ALL_PAIRS
        for layer in layers
        if (x, y) in LOAD_BEARING_PAIRS
    )
    sanity_all = all(
        partition.get((x, y, layer)) == "PASS"
        for (x, y) in ALL_PAIRS
        for layer in layers
        if (x, y) in SANITY_PAIRS
    )
    sanity_fail_cos = any(
        results.get(f"{x}__{y}__layer{layer}", {}).get("per_persona", {}).get("min", 1.0)
        < SUCCESS_COS_MIN
        for (x, y) in ALL_PAIRS
        for layer in layers
        if (x, y) in SANITY_PAIRS
    )

    if success_all_lb:
        headline = (
            "All 5 load-bearing pairs PASS at all layers: Method A is interchangeable with "
            "B, B*, C1 within cos>0.95 / mc_r>0.90."
        )
    elif kill_any_lb:
        headline = (
            "At least one load-bearing pair KILLS at some layer. Method A is NOT recipe-robust "
            "at the kill threshold. See partition for which pairs agree."
        )
    else:
        headline = (
            "Load-bearing pairs in GREY zone: no pair crosses the kill line, "
            "but not all pass success threshold. Mixed evidence."
        )

    verdict_summary = {}
    for layer in layers:
        n_pass = sum(1 for (x, y) in ALL_PAIRS if partition.get((x, y, layer)) == "PASS")
        n_grey = sum(1 for (x, y) in ALL_PAIRS if partition.get((x, y, layer)) == "GREY")
        n_kill = sum(1 for (x, y) in ALL_PAIRS if partition.get((x, y, layer)) == "KILL")
        verdict_summary[f"layer_{layer}"] = {
            "n_pass": n_pass,
            "n_grey": n_grey,
            "n_kill": n_kill,
        }

    verdict_dict = {
        "success_all_load_bearing": success_all_lb,
        "kill_any_load_bearing": kill_any_lb,
        "sanity_all_pass": sanity_all,
        "sanity_any_fail_cos": sanity_fail_cos,
        "headline": headline,
    }
    return partition, verdict_dict, verdict_summary


def _save_all_outputs(
    cents: dict,
    roles: list[str],
    layers: list[int],
    results: dict,
    noise_floor: dict,
    partition: dict,
    verdict_dict: dict,
    verdict_summary: dict,
    n_roles: int,
    n_questions: int,
    root: Path,
    output_dir: Path,
) -> None:
    """Save cosine matrices, spread JSON, noise floor, figures, and run_result."""
    # Cosine matrices
    print("\nSaving cosine matrices...")
    for m in METHODS_6:
        if m not in cents:
            continue
        for l_idx, layer in enumerate(layers):
            cm = torch.stack([cents[m][r][l_idx] for r in roles])
            cos_mat = cosine_matrix(cm)
            cos_json = {"method": m, "layer": layer, "roles": roles, "matrix": cos_mat.tolist()}
            with open(output_dir / f"cosine_matrix_{m}_layer{layer}.json", "w") as f:
                json.dump(cos_json, f)

    # Per-question persona spread
    per_q_spread = {}
    for key, val in results.items():
        sp = val.get("per_question_persona_spread", {}).get("spearman_r")
        if sp is not None:
            per_q_spread[key] = sp
    with open(output_dir / "per_question_persona_spread.json", "w") as f:
        json.dump(per_q_spread, f, indent=2)

    # Noise floor
    with open(output_dir / "noise_floor.json", "w") as f:
        json.dump(noise_floor, f, indent=2)

    # Figures
    print("\nGenerating figures...")
    for path in [
        generate_verdict_heatmap(partition, layers, output_dir),
        generate_per_persona_cos_violins(results, layers, output_dir),
        generate_matrix_corr_bars(results, layers, noise_floor, output_dir),
    ]:
        print(f"  Saved: {path}")

    # Build run_result.json
    results_clean = {}
    for key, val in results.items():
        val_copy = json.loads(json.dumps(val))
        if "per_persona" in val_copy and "_values" in val_copy["per_persona"]:
            del val_copy["per_persona"]["_values"]
        results_clean[key] = val_copy

    partition_serialized = {f"{x}__{y}__layer{lay}": v for (x, y, lay), v in partition.items()}
    run_result = {
        "experiment": "issue_201_6way_extraction_method_ablation",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "layers": layers,
        "n_roles": n_roles,
        "n_questions": n_questions,
        "methods": METHODS_6,
        "n_pairs": len(ALL_PAIRS),
        "pairs": [[x, y] for x, y in ALL_PAIRS],
        "cross_method_metrics": results_clean,
        "noise_floor": noise_floor,
        "partition": partition_serialized,
        "verdict": verdict_dict,
        "verdict_summary": verdict_summary,
        "thresholds": {
            "success_cos_min": SUCCESS_COS_MIN,
            "success_mc_r": SUCCESS_MC_R,
            "kill_cos_min": KILL_COS_MIN,
            "kill_mc_r": KILL_MC_R,
        },
        "load_bearing_pairs": [[x, y] for x, y in sorted(LOAD_BEARING_PAIRS)],
        "sanity_pairs": [[x, y] for x, y in sorted(SANITY_PAIRS)],
        "metadata": {
            "git_commit": _git_commit_hash(),
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "centroid_root": str(root),
            "output_dir": str(output_dir),
            "seed": 42,
        },
    }
    with open(output_dir / "run_result.json", "w") as f:
        json.dump(run_result, f, indent=2)
    print(f"\nSaved run_result.json to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="6-way persona-vector extraction method comparison (Issue #201)"
    )
    parser.add_argument(
        "--centroid-root",
        type=str,
        required=True,
        help="Root directory containing method_* subdirectories",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help="Layer indices to analyze",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results/issue_201",
        help="Output directory for results JSON and figures",
    )
    parser.add_argument(
        "--skip-spread",
        action="store_true",
        help="Skip per-question persona spread (descriptive only; saves ~10h of I/O)",
    )
    args = parser.parse_args()

    t0 = time.time()
    root = Path(args.centroid_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = args.layers

    print("=" * 80)
    print("6-WAY EXTRACTION METHOD COMPARISON")
    print(f"  Root: {root}")
    print(f"  Layers: {layers}")
    print(f"  Output: {output_dir}")
    print("=" * 80)

    # Load centroids
    all_methods = [*METHODS_6, "bstar_no_last"]
    cents = {}
    for m in all_methods:
        centroid_path = root / f"method_{m}" / "all_centroids.pt"
        if not centroid_path.exists():
            print(f"  WARNING: {centroid_path} not found, skipping method {m}")
            continue
        cents[m] = torch.load(centroid_path, weights_only=True)
        print(f"  Loaded method_{m}: {len(cents[m])} roles")

    roles = sorted(set.intersection(*[set(c.keys()) for c in cents.values()]))
    n_roles = len(roles)
    print(f"\n  Intersected roles: {n_roles}")

    meta_path = root / "method_a" / "metadata.json"
    n_questions = 240
    if meta_path.exists():
        with open(meta_path) as f:
            n_questions = json.load(f).get("n_questions", 240)
    print(f"  Questions: {n_questions}")

    # Compute metrics
    results = _compute_cross_method_metrics(
        cents, roles, layers, root, n_questions, skip_spread=args.skip_spread
    )
    _compute_bstar_no_last_sensitivity(cents, roles, layers, results)
    noise_floor = _compute_noise_floor(roles, layers, root, n_questions)

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    partition, verdict_dict, verdict_summary = _compute_verdict(results, layers)

    for layer in layers:
        vs = verdict_summary[f"layer_{layer}"]
        print(f"  Layer {layer}: PASS={vs['n_pass']}, GREY={vs['n_grey']}, KILL={vs['n_kill']}")

    print(f"\n  Headline: {verdict_dict['headline']}")
    print(f"  success_all_load_bearing: {verdict_dict['success_all_load_bearing']}")
    print(f"  kill_any_load_bearing: {verdict_dict['kill_any_load_bearing']}")
    print(f"  sanity_all_pass: {verdict_dict['sanity_all_pass']}")
    if verdict_dict["sanity_any_fail_cos"]:
        print("  WARNING: Sanity pair(s) fail cos>0.95 — possible implementation bug!")

    # Save everything
    _save_all_outputs(
        cents,
        roles,
        layers,
        results,
        noise_floor,
        partition,
        verdict_dict,
        verdict_summary,
        n_roles,
        n_questions,
        root,
        output_dir,
    )

    elapsed = time.time() - t0
    print(f"\nTotal analysis time: {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    print("Done.")


if __name__ == "__main__":
    main()
