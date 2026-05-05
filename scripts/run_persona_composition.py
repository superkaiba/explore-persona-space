#!/usr/bin/env python3
"""
Persona Composition (1.3): SAE-based Compositional Structure Analysis
Analyzes whether persona activations share a compositional basis of transferable traits.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ["TMPDIR"] = "/workspace/tmp"

from explore_persona_space.utils import seed_everything

# Ensure reproducibility
seed_everything(42)

# Path is pre-rename; data lives under aim<N>_… for back-compat. Slice 1 (#251) leaves this untouched.
ACTIVATION_DIR = Path("/workspace/gemma2-27b-aim1/full/activations")
# Path is pre-rename; data lives under aim<N>_… for back-compat. Slice 1 (#251) leaves this untouched.
OUTPUT_DIR = Path("/workspace/explore-persona-space/experiments/aim1_3_composition/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYER_INDEX = 3  # layer 22 is index 3 in [15, 18, 20, 22, 25, 28, 30, 33, 36]
LAYERS = [15, 18, 20, 22, 25, 28, 30, 33, 36]

# ============================================================
# Step 1: Load activations and compute centroids
# ============================================================
print("=" * 70)
print("STEP 1: Loading activations and computing centroids at layer 22")
print("=" * 70)

pt_files = sorted(ACTIVATION_DIR.glob("*.pt"))
print(f"Found {len(pt_files)} activation files")

persona_names = []
centroids = []

for pt_file in pt_files:
    name = pt_file.stem
    persona_names.append(name)
    data = torch.load(pt_file, map_location="cpu", weights_only=False)
    # data is a dict: {key: tensor of shape (9, 4608)} with ~1200 entries
    # Stack all values, extract layer 22 (index 3), average across entries
    all_tensors = torch.stack(list(data.values()))  # (N, 9, 4608)
    layer_data = all_tensors[:, LAYER_INDEX, :]  # (N, 4608)
    centroid = layer_data.float().mean(dim=0).numpy()  # (4608,)
    centroids.append(centroid)
    print(f"  {name}: {len(data)} entries, centroid norm = {np.linalg.norm(centroid):.4f}")

centroids = np.stack(centroids)  # (49, 4608)
print(f"\nCentroid matrix shape: {centroids.shape}")

# ============================================================
# Step 2: Mean-center the centroids
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Mean-centering centroids")
print("=" * 70)

global_mean = centroids.mean(axis=0)
centroids_mc = centroids - global_mean

# Report cosine similarities before and after
from numpy.linalg import norm


def cosine_sim_matrix(X):
    norms = norm(X, axis=1, keepdims=True)
    X_normed = X / (norms + 1e-10)
    return X_normed @ X_normed.T


raw_cos = cosine_sim_matrix(centroids)
mc_cos = cosine_sim_matrix(centroids_mc)

# Get off-diagonal values
mask = ~np.eye(49, dtype=bool)
print(
    f"Raw cosine similarities: min={raw_cos[mask].min():.6f}, max={raw_cos[mask].max():.6f}, "
    f"mean={raw_cos[mask].mean():.6f}, std={raw_cos[mask].std():.6f}"
)
print(
    f"Mean-centered cosines:  min={mc_cos[mask].min():.6f}, max={mc_cos[mask].max():.6f}, "
    f"mean={mc_cos[mask].mean():.6f}, std={mc_cos[mask].std():.6f}"
)

# ============================================================
# Step 3: PCA analysis (baseline)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: PCA analysis on mean-centered centroids")
print("=" * 70)

from sklearn.decomposition import PCA

pca_full = PCA()
pca_full.fit(centroids_mc)

cumvar = np.cumsum(pca_full.explained_variance_ratio_)
print("\nPCA variance explained (cumulative):")
for k in [1, 2, 3, 5, 10, 15, 20, 25, 30, 48]:
    if k <= len(cumvar):
        print(f"  {k:3d} components: {cumvar[k - 1] * 100:.2f}%")

# Find 90% and 95% thresholds
n90 = np.argmax(cumvar >= 0.90) + 1
n95 = np.argmax(cumvar >= 0.95) + 1
print(f"\n  Components for 90% variance: {n90}")
print(f"  Components for 95% variance: {n95}")

# PCA loadings plot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scree plot
n_components = len(pca_full.explained_variance_ratio_)
axes[0].bar(range(1, n_components + 1), pca_full.explained_variance_ratio_ * 100, alpha=0.7)
axes[0].plot(range(1, n_components + 1), cumvar * 100, "r-o", markersize=3)
axes[0].axhline(y=90, color="g", linestyle="--", alpha=0.5, label="90%")
axes[0].axhline(y=95, color="orange", linestyle="--", alpha=0.5, label="95%")
axes[0].set_xlabel("Component")
axes[0].set_ylabel("Variance Explained (%)")
axes[0].set_title("PCA Scree Plot (49 Persona Centroids, Layer 22)")
axes[0].legend()

# PCA projection (2D)
pca_2d = pca_full.transform(centroids_mc)[:, :2]
axes[1].scatter(pca_2d[:, 0], pca_2d[:, 1], alpha=0.7, s=50)
for i, name in enumerate(persona_names):
    axes[1].annotate(name, (pca_2d[i, 0], pca_2d[i, 1]), fontsize=6, alpha=0.8)
axes[1].set_xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0] * 100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1] * 100:.1f}%)")
axes[1].set_title("PCA Projection of Persona Centroids")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_loadings.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'pca_loadings.png'}")

# ============================================================
# Step 4: Sparse Dictionary Learning sweep
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Sparse Dictionary Learning sweep")
print("=" * 70)

from sklearn.decomposition import DictionaryLearning

k_values = [5, 10, 15, 20, 25, 30]
dl_results = {}

for k in k_values:
    t0 = time.time()
    print(f"\n  Fitting DictionaryLearning with k={k}...")
    dl = DictionaryLearning(
        n_components=k,
        alpha=1.0,
        max_iter=1000,
        fit_algorithm="lars",
        transform_algorithm="lasso_lars",
        random_state=42,
        verbose=0,
    )
    codes = dl.fit_transform(centroids_mc)  # (49, k)
    dictionary = dl.components_  # (k, 4608)

    # Reconstruction
    reconstruction = codes @ dictionary
    recon_error = np.mean((centroids_mc - reconstruction) ** 2)
    recon_r2 = 1 - np.sum((centroids_mc - reconstruction) ** 2) / np.sum(centroids_mc**2)

    # Sparsity: fraction of near-zero codes
    sparsity = np.mean(np.abs(codes) < 1e-6)

    elapsed = time.time() - t0
    print(
        f"    Recon MSE = {recon_error:.6f}, R^2 = {recon_r2:.4f}, "
        f"Sparsity = {sparsity:.2%}, Time = {elapsed:.1f}s"
    )

    dl_results[k] = {
        "recon_mse": float(recon_error),
        "recon_r2": float(recon_r2),
        "sparsity": float(sparsity),
        "codes": codes,
        "dictionary": dictionary,
        "model": dl,
    }

# Plot reconstruction error vs k
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ks = sorted(dl_results.keys())
mses = [dl_results[k]["recon_mse"] for k in ks]
r2s = [dl_results[k]["recon_r2"] for k in ks]
sparsities = [dl_results[k]["sparsity"] for k in ks]

axes[0].plot(ks, mses, "bo-")
axes[0].set_xlabel("Number of Components (k)")
axes[0].set_ylabel("Reconstruction MSE")
axes[0].set_title("Dictionary Learning: Reconstruction Error")

axes[1].plot(ks, r2s, "ro-")
axes[1].set_xlabel("Number of Components (k)")
axes[1].set_ylabel("R^2")
axes[1].set_title("Dictionary Learning: R^2")

axes[2].plot(ks, sparsities, "go-")
axes[2].set_xlabel("Number of Components (k)")
axes[2].set_ylabel("Code Sparsity")
axes[2].set_title("Dictionary Learning: Sparsity")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dictionary_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'dictionary_sweep.png'}")

# ============================================================
# Step 5: Interpret best dictionary components
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Interpreting dictionary components")
print("=" * 70)

# Pick best k by elbow heuristic: best R^2 with decent sparsity
# Use k=15 as a reasonable middle ground unless R^2 drops sharply
# Find first k where R^2 > 0.85 or use the largest k tried
best_k = None
for k in ks:
    if dl_results[k]["recon_r2"] > 0.85:
        best_k = k
        break
if best_k is None:
    best_k = ks[-1]

# Actually, let's pick by elbow: largest marginal gain drop
if len(ks) >= 3:
    marginal_gains = []
    for i in range(1, len(ks)):
        gain = r2s[i] - r2s[i - 1]
        marginal_gains.append(gain)
    # Elbow = where marginal gain drops most
    # But also require R^2 > 0.7
    for i, k in enumerate(ks):
        if r2s[i] > 0.80:
            best_k = k
            break

print(f"\nBest k selected: {best_k} (R^2 = {dl_results[best_k]['recon_r2']:.4f})")

codes = dl_results[best_k]["codes"]  # (49, best_k)
dictionary = dl_results[best_k]["dictionary"]  # (best_k, 4608)

print("\nComponent interpretations (top/bottom 5 personas per component):")
print("-" * 70)

component_interpretations = []
for comp_idx in range(best_k):
    loadings = codes[:, comp_idx]
    sorted_idx = np.argsort(loadings)

    top5 = [(persona_names[i], float(loadings[i])) for i in sorted_idx[-5:][::-1]]
    bottom5 = [(persona_names[i], float(loadings[i])) for i in sorted_idx[:5]]

    # Only show non-zero loadings
    top5_nz = [(n, v) for n, v in top5 if abs(v) > 1e-4]
    bottom5_nz = [(n, v) for n, v in bottom5 if abs(v) > 1e-4]

    n_active = np.sum(np.abs(loadings) > 1e-4)

    print(f"\n  Component {comp_idx}: ({n_active} active personas)")
    print(f"    TOP:    {', '.join(f'{n}({v:+.3f})' for n, v in top5_nz)}")
    print(f"    BOTTOM: {', '.join(f'{n}({v:+.3f})' for n, v in bottom5_nz)}")

    component_interpretations.append(
        {
            "component": comp_idx,
            "n_active": int(n_active),
            "top5": top5,
            "bottom5": bottom5,
        }
    )

# Heatmap of codes
fig, ax = plt.subplots(figsize=(max(12, best_k * 0.6), 14))
im = ax.imshow(codes, aspect="auto", cmap="RdBu_r", interpolation="nearest")
ax.set_xticks(range(best_k))
ax.set_xticklabels([f"C{i}" for i in range(best_k)])
ax.set_yticks(range(49))
ax.set_yticklabels(persona_names, fontsize=7)
ax.set_xlabel("Dictionary Component")
ax.set_ylabel("Persona")
ax.set_title(f"Sparse Dictionary Codes (k={best_k})")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dictionary_components.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'dictionary_components.png'}")

# ============================================================
# Step 6: Trait transfer validation
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Trait transfer validation")
print("=" * 70)

name_to_idx = {n: i for i, n in enumerate(persona_names)}

# --- 6a: Creative vs Analytical direction ---
creative_roles = ["poet", "bard", "musician", "novelist"]
analytical_roles = ["scientist", "mathematician", "economist", "historian"]

# Check which exist
creative_avail = [r for r in creative_roles if r in name_to_idx]
analytical_avail = [r for r in analytical_roles if r in name_to_idx]
print(f"\nCreative roles found: {creative_avail}")
print(f"Analytical roles found: {analytical_avail}")

creative_mean = np.mean([centroids_mc[name_to_idx[r]] for r in creative_avail], axis=0)
analytical_mean = np.mean([centroids_mc[name_to_idx[r]] for r in analytical_avail], axis=0)

creative_direction = creative_mean - analytical_mean
creative_direction_normed = creative_direction / (norm(creative_direction) + 1e-10)

# Project all personas
creative_scores = centroids_mc @ creative_direction_normed
sorted_creative = sorted(zip(persona_names, creative_scores), key=lambda x: x[1], reverse=True)

print("\nCreative <-> Analytical axis projection:")
print("  Most CREATIVE:")
for name, score in sorted_creative[:10]:
    marker = " *" if name in creative_avail else ""
    print(f"    {name:20s}: {score:+.4f}{marker}")
print("  Most ANALYTICAL:")
for name, score in sorted_creative[-10:]:
    marker = " *" if name in analytical_avail else ""
    print(f"    {name:20s}: {score:+.4f}{marker}")

# --- 6b: Authority vs Rebellion direction ---
authority_roles = ["judge", "soldier", "guardian", "lawyer"]
rebellion_roles = ["rebel", "pirate", "criminal", "trickster"]

authority_avail = [r for r in authority_roles if r in name_to_idx]
rebellion_avail = [r for r in rebellion_roles if r in name_to_idx]
print(f"\nAuthority roles found: {authority_avail}")
print(f"Rebellion roles found: {rebellion_avail}")

authority_mean = np.mean([centroids_mc[name_to_idx[r]] for r in authority_avail], axis=0)
rebellion_mean = np.mean([centroids_mc[name_to_idx[r]] for r in rebellion_avail], axis=0)

authority_direction = authority_mean - rebellion_mean
authority_direction_normed = authority_direction / (norm(authority_direction) + 1e-10)

authority_scores = centroids_mc @ authority_direction_normed
sorted_authority = sorted(zip(persona_names, authority_scores), key=lambda x: x[1], reverse=True)

print("\nAuthority <-> Rebellion axis projection:")
print("  Most AUTHORITATIVE:")
for name, score in sorted_authority[:10]:
    marker = " *" if name in authority_avail else ""
    print(f"    {name:20s}: {score:+.4f}{marker}")
print("  Most REBELLIOUS:")
for name, score in sorted_authority[-10:]:
    marker = " *" if name in rebellion_avail else ""
    print(f"    {name:20s}: {score:+.4f}{marker}")

# --- 6c: Trait algebra test ---
print("\n--- Trait Algebra Test ---")
# pirate - smuggler = pirate-specific features
# doctor + (pirate - smuggler) = pirate-doctor?
if "pirate" in name_to_idx and "smuggler" in name_to_idx and "doctor" in name_to_idx:
    pirate_vec = centroids_mc[name_to_idx["pirate"]]
    smuggler_vec = centroids_mc[name_to_idx["smuggler"]]
    doctor_vec = centroids_mc[name_to_idx["doctor"]]

    pirate_specific = pirate_vec - smuggler_vec
    pirate_doctor = doctor_vec + pirate_specific

    # Find nearest personas to the constructed vector
    sims = cosine_sim_matrix(np.vstack([pirate_doctor.reshape(1, -1), centroids_mc]))
    pirate_doctor_sims = sims[0, 1:]  # similarities to all 49 personas
    sorted_sims = sorted(zip(persona_names, pirate_doctor_sims), key=lambda x: x[1], reverse=True)

    print("  pirate - smuggler + doctor -> nearest personas:")
    for name, sim in sorted_sims[:10]:
        print(f"    {name:20s}: cosine = {sim:.4f}")

    # Also check: does the result have higher similarity to both pirate and doctor
    # than to most other personas?
    pirate_sim = float(pirate_doctor_sims[name_to_idx["pirate"]])
    doctor_sim = float(pirate_doctor_sims[name_to_idx["doctor"]])
    smuggler_sim = float(pirate_doctor_sims[name_to_idx["smuggler"]])
    mean_sim = float(pirate_doctor_sims.mean())
    print("\n  Similarity of (pirate-smuggler+doctor) to:")
    print(f"    pirate:   {pirate_sim:.4f}")
    print(f"    doctor:   {doctor_sim:.4f}")
    print(f"    smuggler: {smuggler_sim:.4f}")
    print(f"    mean:     {mean_sim:.4f}")

# --- 6d: Mystical vs Practical direction ---
mystical_roles = ["mystic", "shaman", "witch", "oracle"]
practical_roles = ["paramedic", "pilot", "merchant", "bartender"]

mystical_avail = [r for r in mystical_roles if r in name_to_idx]
practical_avail = [r for r in practical_roles if r in name_to_idx]

mystical_mean = np.mean([centroids_mc[name_to_idx[r]] for r in mystical_avail], axis=0)
practical_mean = np.mean([centroids_mc[name_to_idx[r]] for r in practical_avail], axis=0)

mystical_direction = mystical_mean - practical_mean
mystical_direction_normed = mystical_direction / (norm(mystical_direction) + 1e-10)

mystical_scores = centroids_mc @ mystical_direction_normed
sorted_mystical = sorted(zip(persona_names, mystical_scores), key=lambda x: x[1], reverse=True)

print("\nMystical <-> Practical axis projection:")
print("  Most MYSTICAL:")
for name, score in sorted_mystical[:10]:
    marker = " *" if name in mystical_avail else ""
    print(f"    {name:20s}: {score:+.4f}{marker}")
print("  Most PRACTICAL:")
for name, score in sorted_mystical[-10:]:
    marker = " *" if name in practical_avail else ""
    print(f"    {name:20s}: {score:+.4f}{marker}")

# Plot trait axes
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Creative vs Analytical
ax = axes[0, 0]
colors = [
    "red" if n in creative_avail else "blue" if n in analytical_avail else "gray"
    for n in persona_names
]
creative_arr = np.array([s for _, s in sorted(zip(persona_names, creative_scores))])
sorted_names_c = sorted(persona_names)
y_pos = range(len(sorted_names_c))
ax.barh(
    y_pos,
    [creative_scores[name_to_idx[n]] for n in sorted_names_c],
    color=[colors[name_to_idx[n]] for n in sorted_names_c],
    alpha=0.7,
)
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_names_c, fontsize=6)
ax.set_xlabel("Projection Score")
ax.set_title("Creative (red) <-> Analytical (blue)")
ax.axvline(x=0, color="black", linewidth=0.5)

# Authority vs Rebellion
ax = axes[0, 1]
colors2 = [
    "red" if n in authority_avail else "blue" if n in rebellion_avail else "gray"
    for n in persona_names
]
ax.barh(
    y_pos,
    [authority_scores[name_to_idx[n]] for n in sorted_names_c],
    color=[colors2[name_to_idx[n]] for n in sorted_names_c],
    alpha=0.7,
)
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_names_c, fontsize=6)
ax.set_xlabel("Projection Score")
ax.set_title("Authority (red) <-> Rebellion (blue)")
ax.axvline(x=0, color="black", linewidth=0.5)

# Mystical vs Practical
ax = axes[1, 0]
colors3 = [
    "red" if n in mystical_avail else "blue" if n in practical_avail else "gray"
    for n in persona_names
]
ax.barh(
    y_pos,
    [mystical_scores[name_to_idx[n]] for n in sorted_names_c],
    color=[colors3[name_to_idx[n]] for n in sorted_names_c],
    alpha=0.7,
)
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_names_c, fontsize=6)
ax.set_xlabel("Projection Score")
ax.set_title("Mystical (red) <-> Practical (blue)")
ax.axvline(x=0, color="black", linewidth=0.5)

# Orthogonality of trait directions
ax = axes[1, 1]
trait_dirs = np.stack(
    [creative_direction_normed, authority_direction_normed, mystical_direction_normed]
)
trait_cos = cosine_sim_matrix(trait_dirs)
im = ax.imshow(trait_cos, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(
    ["Creative-Analytical", "Authority-Rebellion", "Mystical-Practical"], fontsize=8, rotation=30
)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["Creative-Analytical", "Authority-Rebellion", "Mystical-Practical"], fontsize=8)
ax.set_title("Trait Direction Orthogonality")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{trait_cos[i, j]:.3f}", ha="center", va="center", fontsize=10)
plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "trait_transfer.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'trait_transfer.png'}")

# ============================================================
# Step 7: Save summary JSON
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Saving summary")
print("=" * 70)

summary = {
    "topic": "persona-geometry",
    "description": "Compositional structure analysis of persona activations",
    "layer": 22,
    "layer_index": LAYER_INDEX,
    "n_personas": len(persona_names),
    "personas": persona_names,
    "centroid_shape": list(centroids.shape),
    "mean_centering": {
        "raw_cosine_range": [float(raw_cos[mask].min()), float(raw_cos[mask].max())],
        "raw_cosine_mean": float(raw_cos[mask].mean()),
        "mc_cosine_range": [float(mc_cos[mask].min()), float(mc_cos[mask].max())],
        "mc_cosine_mean": float(mc_cos[mask].mean()),
    },
    "pca": {
        "variance_explained_cumulative": {
            str(k): float(cumvar[k - 1]) for k in [1, 2, 3, 5, 10, 15, 20, 25, 30]
        },
        "n_components_90pct": int(n90),
        "n_components_95pct": int(n95),
        "top5_eigenvalues": pca_full.explained_variance_ratio_[:5].tolist(),
    },
    "dictionary_learning": {
        "sweep_results": {
            str(k): {
                "recon_mse": dl_results[k]["recon_mse"],
                "recon_r2": dl_results[k]["recon_r2"],
                "sparsity": dl_results[k]["sparsity"],
            }
            for k in ks
        },
        "best_k": int(best_k),
        "best_r2": float(dl_results[best_k]["recon_r2"]),
        "component_interpretations": component_interpretations,
    },
    "trait_transfer": {
        "creative_analytical": {
            "creative_roles": creative_avail,
            "analytical_roles": analytical_avail,
            "top5": [(n, float(s)) for n, s in sorted_creative[:5]],
            "bottom5": [(n, float(s)) for n, s in sorted_creative[-5:]],
        },
        "authority_rebellion": {
            "authority_roles": authority_avail,
            "rebellion_roles": rebellion_avail,
            "top5": [(n, float(s)) for n, s in sorted_authority[:5]],
            "bottom5": [(n, float(s)) for n, s in sorted_authority[-5:]],
        },
        "mystical_practical": {
            "mystical_roles": mystical_avail,
            "practical_roles": practical_avail,
            "top5": [(n, float(s)) for n, s in sorted_mystical[:5]],
            "bottom5": [(n, float(s)) for n, s in sorted_mystical[-5:]],
        },
        "trait_direction_orthogonality": {
            "creative_authority": float(trait_cos[0, 1]),
            "creative_mystical": float(trait_cos[0, 2]),
            "authority_mystical": float(trait_cos[1, 2]),
        },
    },
}

# Add trait algebra results if available
if "pirate" in name_to_idx and "smuggler" in name_to_idx and "doctor" in name_to_idx:
    summary["trait_transfer"]["trait_algebra"] = {
        "formula": "pirate - smuggler + doctor",
        "nearest_personas": [(n, float(s)) for n, s in sorted_sims[:10]],
        "target_similarities": {
            "pirate": pirate_sim,
            "doctor": doctor_sim,
            "smuggler": smuggler_sim,
            "mean_all": mean_sim,
        },
    }

with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {OUTPUT_DIR / 'summary.json'}")

print("\n" + "=" * 70)
print("AIM 1.3 COMPLETE")
print("=" * 70)
