#!/usr/bin/env python3
"""Re-analyze 100-persona leakage with source-specific category filtering.

Key difference from `analyze_100_persona_cosine.py`:
    Only pairs where `source in target.related_to` are included in the
    per-category correlation. This gives each (source, target) pair its
    "intended" category (e.g., `web_developer` counts as a professional_peer
    for `software_engineer` only, not for `villain`).

Outputs:
    - eval_results/single_token_100_persona/cosine_leakage_filtered.json
    - figures/single_token_100_persona/category_rho_bar_filtered.{png,pdf}
    - figures/single_token_100_persona/cosine_vs_leakage_scatter_v2.{png,pdf}
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval_results" / "single_token_100_persona"
FIG_DIR = ROOT / "figures" / "single_token_100_persona"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))
from run_100_persona_leakage import ALL_EVAL_PERSONAS  # noqa: E402

SOURCES = ["villain", "comedian", "assistant", "software_engineer", "kindergarten_teacher"]
LAYER = 20

# ── Load cosine (from existing pipeline output) ─────────────────────────────
# The existing cosine_leakage_correlation.json has per_pair numbers but is
# structured per-layer per-source. We also need the raw cosine matrix.
# Simplest: recompute cosines from saved centroids.
from scipy import stats  # noqa: E402


def load_centroids(layer: int = LAYER):
    import torch

    centroids_path = RESULTS / "centroids" / f"centroids_layer{layer}.pt"
    names_path = RESULTS / "centroids" / "persona_names.json"
    centroids = torch.load(centroids_path, map_location="cpu", weights_only=True)
    names = json.loads(names_path.read_text())
    return centroids.numpy(), names


def cosine_matrix(centroids: np.ndarray) -> np.ndarray:
    c = centroids - centroids.mean(axis=0, keepdims=True)
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
    return c @ c.T


def load_leakage():
    out = {}
    for src in SOURCES:
        p = RESULTS / src / "marker_eval.json"
        data = json.loads(p.read_text())
        # structure: {target: {rate, ...}} or similar
        out[src] = data.get("per_persona", data) if isinstance(data, dict) else data
    return out


def main():
    centroids, names = load_centroids(LAYER)
    cos = cosine_matrix(centroids)
    name_to_idx = {n: i for i, n in enumerate(names)}
    leakage = load_leakage()

    # Build per-pair table with source, target, cosine, leakage, related flag
    rows = []
    for src in SOURCES:
        if src not in name_to_idx or src not in leakage:
            continue
        s_idx = name_to_idx[src]
        for tgt, info in ALL_EVAL_PERSONAS.items():
            if tgt == src or tgt not in name_to_idx:
                continue
            tinfo = ALL_EVAL_PERSONAS[tgt]
            related = src in tinfo.get("related_to", [])
            src_leak = leakage[src]
            if isinstance(src_leak, dict) and tgt in src_leak:
                rate = (
                    src_leak[tgt].get("rate") if isinstance(src_leak[tgt], dict) else src_leak[tgt]
                )
            else:
                continue
            rows.append(
                {
                    "source": src,
                    "target": tgt,
                    "cosine": float(cos[s_idx, name_to_idx[tgt]]),
                    "leakage": float(rate),
                    "category": tinfo.get("category", "unknown"),
                    "related": related,
                }
            )

    print(f"Pairs total: {len(rows)}  related: {sum(r['related'] for r in rows)}")

    # ── Per-category correlations: pooled vs filtered ───────────────────────
    by_cat_all = {}
    by_cat_rel = {}
    for r in rows:
        by_cat_all.setdefault(r["category"], []).append(r)
        if r["related"]:
            by_cat_rel.setdefault(r["category"], []).append(r)

    results = {"layer": LAYER, "pooled": {}, "related_only": {}}
    print(f"\n{'Category':<22}  {'pooled rho':<12} {'N':<5}  {'related rho':<14} {'N':<5}")
    print("-" * 72)
    for cat in sorted(by_cat_all):
        pool = by_cat_all[cat]
        rel = by_cat_rel.get(cat, [])
        if len(pool) >= 5:
            rho_p, p_p = stats.spearmanr([r["cosine"] for r in pool], [r["leakage"] for r in pool])
        else:
            rho_p, p_p = (float("nan"), float("nan"))
        if len(rel) >= 5:
            rho_r, p_r = stats.spearmanr([r["cosine"] for r in rel], [r["leakage"] for r in rel])
        else:
            rho_r, p_r = (float("nan"), float("nan"))
        results["pooled"][cat] = {"rho": float(rho_p), "p": float(p_p), "n": len(pool)}
        results["related_only"][cat] = {"rho": float(rho_r), "p": float(p_r), "n": len(rel)}
        rel_str = f"{rho_r:+.3f}  (N={len(rel)})" if len(rel) >= 5 else f"--      (N={len(rel)})"
        print(f"{cat:<22}  {rho_p:+.3f}       {len(pool):<5}  {rel_str}")

    # ── Aggregates ──────────────────────────────────────────────────────────
    all_p = rows
    all_r = [r for r in rows if r["related"]]
    for label, subset, key in [
        ("pooled", all_p, "_aggregate_pooled"),
        ("related_only", all_r, "_aggregate_related"),
    ]:
        if len(subset) >= 5:
            rho, p = stats.spearmanr([r["cosine"] for r in subset], [r["leakage"] for r in subset])
            results[key] = {"rho": float(rho), "p": float(p), "n": len(subset)}
            print(f"{label:<22} aggregate rho={rho:+.3f}  (N={len(subset)})")

    out_path = RESULTS / "cosine_leakage_filtered.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")

    # ── Plot 1: Scatter by source with related/unrelated color ──────────────
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2), sharey=True)
    for ax, src in zip(axes, SOURCES):
        pts_rel = [r for r in rows if r["source"] == src and r["related"]]
        pts_unr = [r for r in rows if r["source"] == src and not r["related"]]

        ax.scatter(
            [p["cosine"] for p in pts_unr],
            [p["leakage"] * 100 for p in pts_unr],
            color="#bbbbbb",
            s=18,
            alpha=0.55,
            edgecolor="none",
            label=f"unintended ({len(pts_unr)})",
        )
        ax.scatter(
            [p["cosine"] for p in pts_rel],
            [p["leakage"] * 100 for p in pts_rel],
            color="#d62728",
            s=42,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.3,
            label=f"intended  ({len(pts_rel)})",
        )

        # Compute per-source ρ pooled and related-only
        if pts_rel and pts_unr:
            all_pts = pts_rel + pts_unr
            rho_all, _ = stats.spearmanr(
                [p["cosine"] for p in all_pts], [p["leakage"] for p in all_pts]
            )
            if len(pts_rel) >= 5:
                rho_rel, _ = stats.spearmanr(
                    [p["cosine"] for p in pts_rel], [p["leakage"] for p in pts_rel]
                )
                ax.set_title(
                    f"{src}\nrho_all={rho_all:+.2f} | rho_rel={rho_rel:+.2f} (N_rel={len(pts_rel)})",
                    fontsize=10,
                )
            else:
                ax.set_title(
                    f"{src}\nrho_all={rho_all:+.2f} | rho_rel=n/a (N_rel={len(pts_rel)})",
                    fontsize=10,
                )

        ax.set_xlabel("cosine similarity (base-model, L20)")
        ax.set_ylim(-3, 103)
        ax.grid(linestyle=":", alpha=0.4)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    axes[0].set_ylabel("marker leakage rate %  (higher = more leakage)")
    fig.suptitle(
        "Cosine similarity vs marker leakage  |  intended vs unintended source-target pairs  |  seed 42, layer 20",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        p = FIG_DIR / f"cosine_vs_leakage_scatter_v2.{ext}"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"Wrote {p}")

    # ── Plot 2: Per-category rho bar — pooled vs related-only ───────────────
    cats_ordered = sorted(
        [c for c in results["pooled"] if not np.isnan(results["pooled"][c]["rho"])],
        key=lambda c: -results["pooled"][c]["rho"],
    )
    y = np.arange(len(cats_ordered))[::-1]
    rho_pool = [results["pooled"][c]["rho"] for c in cats_ordered]
    rho_rel_list = [results["related_only"].get(c, {"rho": np.nan})["rho"] for c in cats_ordered]

    fig2, ax2 = plt.subplots(figsize=(11, 5.5))
    width = 0.38
    ax2.barh(
        y + width / 2,
        rho_pool,
        width,
        color="#9ecae1",
        edgecolor="black",
        linewidth=0.5,
        label="pooled over all 5 sources (old)",
    )
    ax2.barh(
        y - width / 2,
        rho_rel_list,
        width,
        color="#d62728",
        edgecolor="black",
        linewidth=0.5,
        label="filtered to source in related_to",
    )

    for yi, (c, rp, rr) in enumerate(zip(cats_ordered, rho_pool, rho_rel_list)):
        yp = y[yi] + width / 2
        yr = y[yi] - width / 2
        np_ = results["pooled"][c]["n"]
        nr = results["related_only"].get(c, {"n": 0})["n"]
        ax2.text(
            rp + (0.02 if rp >= 0 else -0.02),
            yp,
            f"{rp:+.2f} (N={np_})",
            va="center",
            ha="left" if rp >= 0 else "right",
            fontsize=8,
        )
        rr_str = f"{rr:+.2f} (N={nr})" if not np.isnan(rr) else f"n/a (N={nr})"
        ax2.text(
            (rr if not np.isnan(rr) else 0) + 0.02,
            yr,
            rr_str,
            va="center",
            ha="left",
            fontsize=8,
            color="#7f0000",
        )

    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels([c.replace("_", " ") for c in cats_ordered])
    ax2.set_xlabel("Spearman rho  (cosine -> leakage)  |  higher = better prediction")
    ax2.set_title(
        "Per-category correlation — pooled vs source-filtered (seed 42, layer 20)",
        fontsize=11,
    )
    ax2.set_xlim(-0.6, 1.15)
    ax2.grid(axis="x", linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=9)
    fig2.tight_layout()

    for ext in ("png", "pdf"):
        p = FIG_DIR / f"category_rho_bar_filtered.{ext}"
        fig2.savefig(p, dpi=180, bbox_inches="tight")
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
