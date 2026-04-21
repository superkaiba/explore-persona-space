#!/usr/bin/env python3
"""Simple scatter: cosine similarity vs marker leakage, one panel per source.
All target personas plotted as a single color (no category distinction).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval_results" / "single_token_100_persona"
FIG_DIR = ROOT / "figures" / "single_token_100_persona"

sys.path.insert(0, str(ROOT / "scripts"))
from run_100_persona_leakage import ALL_EVAL_PERSONAS  # noqa: E402

SOURCES = ["villain", "comedian", "assistant", "software_engineer", "kindergarten_teacher"]
LAYER = 20


def load_centroids(layer: int):
    import torch

    centroids = torch.load(
        RESULTS / "centroids" / f"centroids_layer{layer}.pt", map_location="cpu", weights_only=True
    ).numpy()
    names = json.loads((RESULTS / "centroids" / "persona_names.json").read_text())
    return centroids, names


def cosine_matrix(c: np.ndarray) -> np.ndarray:
    c = c - c.mean(axis=0, keepdims=True)
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
    return c @ c.T


def load_leakage():
    return {src: json.loads((RESULTS / src / "marker_eval.json").read_text()) for src in SOURCES}


def main():
    centroids, names = load_centroids(LAYER)
    cos = cosine_matrix(centroids)
    idx = {n: i for i, n in enumerate(names)}
    leak = load_leakage()

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2), sharey=True)
    for ax, src in zip(axes, SOURCES):
        if src not in idx:
            continue
        s = idx[src]
        xs, ys = [], []
        for tgt in ALL_EVAL_PERSONAS:
            if tgt == src or tgt not in idx:
                continue
            d = leak[src]
            if isinstance(d, dict) and tgt in d:
                rate = d[tgt].get("rate") if isinstance(d[tgt], dict) else d[tgt]
                xs.append(float(cos[s, idx[tgt]]))
                ys.append(float(rate) * 100)
        xs, ys = np.array(xs), np.array(ys)
        rho, p = stats.spearmanr(xs, ys)
        ax.scatter(xs, ys, color="#1f77b4", s=24, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.set_title(f"{src}\nrho = {rho:+.2f}  (N = {len(xs)}, p = {p:.1e})", fontsize=10)
        ax.set_xlabel("cosine similarity (base-model, L20)")
        ax.set_ylim(-3, 103)
        ax.grid(linestyle=":", alpha=0.4)

    axes[0].set_ylabel("marker leakage rate %  (higher = more leakage)")
    fig.suptitle(
        "Cosine similarity vs marker leakage  |  per-source scatter  |  "
        "5 sources x 110 targets, seed 42, layer 20",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    for ext in ("png", "pdf"):
        p = FIG_DIR / f"cosine_vs_leakage_scatter_simple.{ext}"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
