"""
Persona Geometry — 1.2: Intrinsic Dimensionality Estimation
Analyzes activation data from persona-geometry 1.1 to estimate the intrinsic
dimensionality of persona representations in Gemma2-27B.
"""

import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# ── Config ──────────────────────────────────────────────────────────────────
# Path is pre-rename; data lives under aim<N>_… for back-compat. Slice 1 (#251) leaves this untouched.
ACTIVATIONS_DIR = Path("/workspace/gemma2-27b-aim1/full/activations")
# Path is pre-rename; data lives under aim<N>_… for back-compat. Slice 1 (#251) leaves this untouched.
OUTPUT_DIR = Path("/workspace/explore-persona-space/experiments/aim1_2_dimensionality/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [15, 18, 20, 22, 25, 28, 30, 33, 36]
# Layer indices within the 9-layer tensor: index 0=L15, 1=L18, ..., 3=L22, ...
FOCUS_LAYER_IDX = 3  # Layer 22
COMPARISON_LAYER_IDXS = [0, 6]  # Layers 15, 30
ALL_ANALYSIS_IDXS = [0, 3, 6]  # L15, L22, L30

N_PCA_COMPONENTS = 50

# ── Helpers ─────────────────────────────────────────────────────────────────


def twonn_dimension(X):
    """Estimate intrinsic dimension via TwoNN (Facco et al. 2017)."""
    nn = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(X)
    distances, _ = nn.kneighbors(X)
    r1 = distances[:, 1]  # distance to nearest neighbor
    r2 = distances[:, 2]  # distance to second nearest
    mu = r2 / (r1 + 1e-10)
    mu = mu[mu > 1]  # filter degenerate
    n = len(mu)
    if n == 0:
        return float("nan")
    d_est = n / np.sum(np.log(mu))
    return d_est


def participation_ratio(X_centered):
    """Compute participation ratio from centered data."""
    # Covariance eigenvalues
    cov = np.cov(X_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]  # keep positive
    pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues**2)
    return pr, eigenvalues


def pca_spectrum(X_centered, n_components=50):
    """Compute PCA spectrum: eigenvalues and explained variance ratios."""
    n_comp = min(n_components, X_centered.shape[0], X_centered.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(X_centered)
    return pca.explained_variance_, pca.explained_variance_ratio_


def components_for_variance(explained_ratio, thresholds=[0.5, 0.8, 0.9, 0.95]):
    """How many components needed to explain given variance thresholds."""
    cumsum = np.cumsum(explained_ratio)
    result = {}
    for t in thresholds:
        idx = np.searchsorted(cumsum, t)
        result[str(t)] = int(idx + 1) if idx < len(cumsum) else len(cumsum)
    return result


# ── Load all activations ────────────────────────────────────────────────────


def load_all_activations():
    """Load all persona activation files. Returns dict: persona -> (n_samples, 9, 4608) array."""
    personas = {}
    pt_files = sorted(ACTIVATIONS_DIR.glob("*.pt"))
    print(f"Found {len(pt_files)} activation files")
    for pt_file in pt_files:
        persona_name = pt_file.stem
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        # Stack all entries: each is (9, 4608)
        vecs = []
        for key in sorted(data.keys()):
            vecs.append(data[key].float().numpy())
        arr = np.stack(vecs)  # (n_samples, 9, 4608)
        personas[persona_name] = arr
        print(f"  {persona_name}: {arr.shape}")
    return personas


# ── Per-persona analysis ────────────────────────────────────────────────────


def analyze_persona_at_layer(X_layer):
    """Analyze a single persona at a single layer.
    X_layer: (n_samples, hidden_dim)
    """
    # Mean-center
    centroid = X_layer.mean(axis=0)
    X_centered = X_layer - centroid

    # Participation ratio
    pr, eigenvalues = participation_ratio(X_centered)

    # TwoNN
    twonn = twonn_dimension(
        X_layer
    )  # TwoNN on original (not centered), though centering doesn't change distances

    # PCA spectrum
    pca_evals, pca_evr = pca_spectrum(X_centered, N_PCA_COMPONENTS)

    return {
        "participation_ratio": float(pr),
        "twonn_dimension": float(twonn),
        "pca_eigenvalues": pca_evals.tolist(),
        "pca_explained_variance_ratio": pca_evr.tolist(),
        "components_for_variance": components_for_variance(pca_evr),
        "centroid_norm": float(np.linalg.norm(centroid)),
        "n_samples": X_layer.shape[0],
    }


# ── Global analysis ────────────────────────────────────────────────────────


def analyze_global(all_vecs_layer):
    """Analyze pooled vectors across all personas at a single layer.
    all_vecs_layer: (n_total, hidden_dim)
    """
    print(f"  Global analysis on {all_vecs_layer.shape[0]} vectors, dim={all_vecs_layer.shape[1]}")

    centroid = all_vecs_layer.mean(axis=0)
    X_centered = all_vecs_layer - centroid

    # PR
    # For large matrices, use SVD instead of full covariance
    n, d = X_centered.shape
    print("  Computing SVD for global PR...")
    # Use PCA with many components for eigenvalue estimation
    n_comp_global = min(200, n, d)
    pca = PCA(n_components=n_comp_global)
    pca.fit(X_centered)
    eigenvalues = pca.explained_variance_
    pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues**2)

    # TwoNN on subsampled data (full is too slow for ~59k points)
    print("  Computing TwoNN on subsample...")
    if all_vecs_layer.shape[0] > 10000:
        rng = np.random.RandomState(42)
        idx = rng.choice(all_vecs_layer.shape[0], 10000, replace=False)
        twonn = twonn_dimension(all_vecs_layer[idx])
    else:
        twonn = twonn_dimension(all_vecs_layer)

    # PCA spectrum (first 50)
    pca_evals = pca.explained_variance_[:N_PCA_COMPONENTS]
    pca_evr = pca.explained_variance_ratio_[:N_PCA_COMPONENTS]

    # Components for variance thresholds using all 200 components
    comp_var = components_for_variance(pca.explained_variance_ratio_)

    return {
        "participation_ratio": float(pr),
        "twonn_dimension": float(twonn),
        "pca_eigenvalues": pca_evals.tolist(),
        "pca_explained_variance_ratio": pca_evr.tolist(),
        "pca_explained_variance_ratio_200": pca.explained_variance_ratio_.tolist(),
        "components_for_variance": comp_var,
        "n_samples": int(all_vecs_layer.shape[0]),
        "n_pca_components_computed": int(n_comp_global),
    }


# ── Plotting ────────────────────────────────────────────────────────────────


def plot_pr_histogram(per_persona_results, layer_name, output_path):
    """Histogram of per-persona participation ratios."""
    prs = [v["participation_ratio"] for v in per_persona_results.values()]
    list(per_persona_results.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(prs)), sorted(prs), color="steelblue", alpha=0.8)
    ax.set_xlabel("Persona (sorted by PR)")
    ax.set_ylabel("Participation Ratio")
    ax.set_title(
        f"Per-Persona Participation Ratio (Layer {layer_name})\nMedian={np.median(prs):.1f}, Mean={np.mean(prs):.1f}"
    )
    ax.axhline(np.median(prs), color="red", linestyle="--", label=f"Median={np.median(prs):.1f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


def plot_global_scree(global_results, layer_name, output_path):
    """PCA scree plot for global analysis."""
    evr = global_results["pca_explained_variance_ratio"]
    cumsum = np.cumsum(evr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual explained variance
    ax1.bar(range(1, len(evr) + 1), evr, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title(f"Global PCA Scree Plot (Layer {layer_name})")
    ax1.set_xlim(0, len(evr) + 1)

    # Cumulative
    ax2.plot(range(1, len(cumsum) + 1), cumsum, "o-", color="steelblue", markersize=3)
    for thresh in [0.5, 0.8, 0.9, 0.95]:
        ax2.axhline(thresh, color="gray", linestyle=":", alpha=0.5)
        n_comp = global_results["components_for_variance"][str(thresh)]
        ax2.annotate(
            f"{thresh * 100:.0f}%: {n_comp} comp", xy=(n_comp, thresh), fontsize=8, color="red"
        )
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title(f"Cumulative Variance (Layer {layer_name})")
    ax2.set_xlim(0, len(cumsum) + 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


def plot_persona_scree_overlay(per_persona_results, layer_name, output_path):
    """Overlay PCA scree plots for all personas."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, res in per_persona_results.items():
        evr = res["pca_explained_variance_ratio"]
        ax.plot(range(1, len(evr) + 1), evr, alpha=0.3, linewidth=0.8)

    # Compute and plot median
    all_evrs = np.array(
        [res["pca_explained_variance_ratio"] for res in per_persona_results.values()]
    )
    median_evr = np.median(all_evrs, axis=0)
    ax.plot(range(1, len(median_evr) + 1), median_evr, "k-", linewidth=2.5, label="Median")

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title(
        f"Per-Persona PCA Spectra (Layer {layer_name}, n={len(per_persona_results)} personas)"
    )
    ax.legend()
    ax.set_xlim(0, min(N_PCA_COMPONENTS, 30) + 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


def plot_dim_vs_layer(layer_results, output_path):
    """Plot dimensionality metrics vs layer depth."""
    layers = sorted(layer_results.keys())
    layer_labels = [LAYERS[l] for l in layers]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PR
    ax = axes[0]
    per_persona_prs = []
    global_prs = []
    for l in layers:
        prs = [v["participation_ratio"] for v in layer_results[l]["per_persona"].values()]
        per_persona_prs.append(prs)
        global_prs.append(layer_results[l]["global"]["participation_ratio"])

    ax.boxplot(per_persona_prs, positions=range(len(layers)), widths=0.6)
    ax.plot(range(len(layers)), global_prs, "r*-", markersize=12, label="Global PR")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layer_labels)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Participation Ratio")
    ax.set_title("Participation Ratio vs Layer")
    ax.legend()

    # TwoNN
    ax = axes[1]
    per_persona_twonn = []
    global_twonn = []
    for l in layers:
        tn = [v["twonn_dimension"] for v in layer_results[l]["per_persona"].values()]
        per_persona_twonn.append(tn)
        global_twonn.append(layer_results[l]["global"]["twonn_dimension"])

    ax.boxplot(per_persona_twonn, positions=range(len(layers)), widths=0.6)
    ax.plot(range(len(layers)), global_twonn, "r*-", markersize=12, label="Global TwoNN")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layer_labels)
    ax.set_xlabel("Layer")
    ax.set_ylabel("TwoNN Intrinsic Dimension")
    ax.set_title("TwoNN Dimension vs Layer")
    ax.legend()

    # Components for 90% variance
    ax = axes[2]
    per_persona_c90 = []
    global_c90 = []
    for l in layers:
        c90 = [
            v["components_for_variance"]["0.9"] for v in layer_results[l]["per_persona"].values()
        ]
        per_persona_c90.append(c90)
        global_c90.append(layer_results[l]["global"]["components_for_variance"]["0.9"])

    ax.boxplot(per_persona_c90, positions=range(len(layers)), widths=0.6)
    ax.plot(range(len(layers)), global_c90, "r*-", markersize=12, label="Global")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layer_labels)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Components for 90% Variance")
    ax.set_title("PCA Components for 90% Variance vs Layer")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    t0 = time.time()
    print("=" * 70)
    print("AIM 1.2: INTRINSIC DIMENSIONALITY ESTIMATION")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading activations...")
    personas = load_all_activations()
    persona_names = sorted(personas.keys())
    n_personas = len(persona_names)
    print(f"Loaded {n_personas} personas")

    # Results storage
    all_results = {
        "metadata": {
            "n_personas": n_personas,
            "persona_names": persona_names,
            "layers_analyzed": [LAYERS[i] for i in ALL_ANALYSIS_IDXS],
            "layer_indices": ALL_ANALYSIS_IDXS,
            "n_pca_components": N_PCA_COMPONENTS,
        },
        "layer_results": {},
    }

    layer_results_for_plot = {}

    # ── Analyze each layer ──
    for layer_idx in ALL_ANALYSIS_IDXS:
        layer_num = LAYERS[layer_idx]
        print(f"\n[2/4] Analyzing Layer {layer_num} (index {layer_idx})...")

        per_persona = {}
        all_vecs = []

        for pname in persona_names:
            X = personas[pname][:, layer_idx, :]  # (n_samples, 4608)
            all_vecs.append(X)

            print(f"  Analyzing {pname}...", end=" ")
            res = analyze_persona_at_layer(X)
            per_persona[pname] = res
            print(f"PR={res['participation_ratio']:.1f}, TwoNN={res['twonn_dimension']:.1f}")

        # Global analysis
        print(f"\n  Global analysis for Layer {layer_num}...")
        all_vecs_stacked = np.vstack(all_vecs)  # (n_total, 4608)
        global_res = analyze_global(all_vecs_stacked)
        print(
            f"  Global PR={global_res['participation_ratio']:.1f}, TwoNN={global_res['twonn_dimension']:.1f}"
        )
        print(f"  Components for 50/80/90/95% variance: {global_res['components_for_variance']}")

        all_results["layer_results"][str(layer_num)] = {
            "per_persona": per_persona,
            "global": global_res,
        }

        layer_results_for_plot[layer_idx] = {
            "per_persona": per_persona,
            "global": global_res,
        }

    # ── Summary statistics ──
    print("\n[3/4] Computing summary statistics...")
    focus_layer = str(LAYERS[FOCUS_LAYER_IDX])
    focus_per_persona = all_results["layer_results"][focus_layer]["per_persona"]

    prs = [v["participation_ratio"] for v in focus_per_persona.values()]
    twonns = [v["twonn_dimension"] for v in focus_per_persona.values()]

    summary_stats = {
        f"layer_{focus_layer}_per_persona": {
            "pr_mean": float(np.mean(prs)),
            "pr_median": float(np.median(prs)),
            "pr_std": float(np.std(prs)),
            "pr_min": float(np.min(prs)),
            "pr_max": float(np.max(prs)),
            "twonn_mean": float(np.mean(twonns)),
            "twonn_median": float(np.median(twonns)),
            "twonn_std": float(np.std(twonns)),
            "twonn_min": float(np.min(twonns)),
            "twonn_max": float(np.max(twonns)),
            "highest_pr_persona": persona_names[np.argmax(prs)],
            "lowest_pr_persona": persona_names[np.argmin(prs)],
        },
        f"layer_{focus_layer}_global": all_results["layer_results"][focus_layer]["global"][
            "components_for_variance"
        ],
    }

    # Cross-layer comparison
    cross_layer = {}
    for layer_idx in ALL_ANALYSIS_IDXS:
        layer_num = LAYERS[layer_idx]
        lr = all_results["layer_results"][str(layer_num)]
        prs_l = [v["participation_ratio"] for v in lr["per_persona"].values()]
        cross_layer[str(layer_num)] = {
            "per_persona_pr_median": float(np.median(prs_l)),
            "per_persona_pr_mean": float(np.mean(prs_l)),
            "global_pr": lr["global"]["participation_ratio"],
            "global_twonn": lr["global"]["twonn_dimension"],
            "global_components_90pct": lr["global"]["components_for_variance"]["0.9"],
        }
    summary_stats["cross_layer"] = cross_layer

    all_results["summary"] = summary_stats

    # ── Save results ──
    # Save a slim version (no large eigenvalue arrays per persona)
    slim_results = {
        "metadata": all_results["metadata"],
        "summary": all_results["summary"],
        "layer_results": {},
    }
    for layer_key, lr in all_results["layer_results"].items():
        slim_pp = {}
        for pname, pres in lr["per_persona"].items():
            slim_pp[pname] = {
                "participation_ratio": pres["participation_ratio"],
                "twonn_dimension": pres["twonn_dimension"],
                "components_for_variance": pres["components_for_variance"],
                "centroid_norm": pres["centroid_norm"],
                "pca_top5_evr": pres["pca_explained_variance_ratio"][:5],
            }
        slim_results["layer_results"][layer_key] = {
            "per_persona": slim_pp,
            "global": lr["global"],
        }

    json_path = OUTPUT_DIR / "summary.json"
    with open(json_path, "w") as f:
        json.dump(slim_results, f, indent=2)
    print(f"  Saved {json_path}")

    # ── Plots ──
    print("\n[4/4] Generating plots...")

    # 1. PR histogram for focus layer
    plot_pr_histogram(focus_per_persona, focus_layer, OUTPUT_DIR / "pr_histogram.png")

    # 2. Global PCA scree for focus layer
    plot_global_scree(
        all_results["layer_results"][focus_layer]["global"],
        focus_layer,
        OUTPUT_DIR / "pca_scree_global.png",
    )

    # 3. Per-persona PCA scree overlay for focus layer
    plot_persona_scree_overlay(
        focus_per_persona, focus_layer, OUTPUT_DIR / "pca_scree_per_persona.png"
    )

    # 4. Dimensionality vs layer
    plot_dim_vs_layer(layer_results_for_plot, OUTPUT_DIR / "dim_vs_layer.png")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nFocus Layer: {focus_layer}")
    print(f"\nPer-Persona Participation Ratio (n={n_personas}):")
    print(f"  Mean:   {summary_stats[f'layer_{focus_layer}_per_persona']['pr_mean']:.2f}")
    print(f"  Median: {summary_stats[f'layer_{focus_layer}_per_persona']['pr_median']:.2f}")
    print(f"  Std:    {summary_stats[f'layer_{focus_layer}_per_persona']['pr_std']:.2f}")
    print(
        f"  Range:  [{summary_stats[f'layer_{focus_layer}_per_persona']['pr_min']:.2f}, {summary_stats[f'layer_{focus_layer}_per_persona']['pr_max']:.2f}]"
    )
    print(f"  Highest: {summary_stats[f'layer_{focus_layer}_per_persona']['highest_pr_persona']}")
    print(f"  Lowest:  {summary_stats[f'layer_{focus_layer}_per_persona']['lowest_pr_persona']}")

    print(f"\nPer-Persona TwoNN Dimension (n={n_personas}):")
    print(f"  Mean:   {summary_stats[f'layer_{focus_layer}_per_persona']['twonn_mean']:.2f}")
    print(f"  Median: {summary_stats[f'layer_{focus_layer}_per_persona']['twonn_median']:.2f}")
    print(f"  Std:    {summary_stats[f'layer_{focus_layer}_per_persona']['twonn_std']:.2f}")
    print(
        f"  Range:  [{summary_stats[f'layer_{focus_layer}_per_persona']['twonn_min']:.2f}, {summary_stats[f'layer_{focus_layer}_per_persona']['twonn_max']:.2f}]"
    )

    print(f"\nGlobal PCA (all {n_personas} personas pooled):")
    gc = all_results["layer_results"][focus_layer]["global"]["components_for_variance"]
    print(f"  Components for 50% variance:  {gc['0.5']}")
    print(f"  Components for 80% variance:  {gc['0.8']}")
    print(f"  Components for 90% variance:  {gc['0.9']}")
    print(f"  Components for 95% variance:  {gc['0.95']}")
    print(
        f"  Global PR: {all_results['layer_results'][focus_layer]['global']['participation_ratio']:.2f}"
    )
    print(
        f"  Global TwoNN: {all_results['layer_results'][focus_layer]['global']['twonn_dimension']:.2f}"
    )

    print("\nCross-Layer Comparison:")
    for lnum_str, ldata in cross_layer.items():
        print(
            f"  Layer {lnum_str}: per-persona PR median={ldata['per_persona_pr_median']:.1f}, "
            f"global PR={ldata['global_pr']:.1f}, global TwoNN={ldata['global_twonn']:.1f}, "
            f"global 90% comp={ldata['global_components_90pct']}"
        )

    return all_results


if __name__ == "__main__":
    main()
