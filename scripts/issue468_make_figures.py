"""Make clean-result figures for task #468.

Produces 4 figures under figures/issue_468/:
  hero_position_sweep   — bar chart of ρ across V5_p0..p5 (the headline)
  layer_profile         — line chart of ρ across deep layers for 3 extractions
  variant_bars          — bar chart of all 6 extractions at L25 lit-training
  response_mean_saturation — histogram of per-cell cosines V1 vs response_mean
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "eval_results" / "issue468"
FIG = REPO / "figures" / "issue_468"
FIG.mkdir(parents=True, exist_ok=True)


def _load_regression_lit_training() -> dict:
    return json.loads((RES / "regression_variants_training_lit.json").read_text())


def _load_position_sweep() -> dict:
    return json.loads((RES / "regression_position_sweep_L25_lit_training.json").read_text())


# ---------------------------------------------------------------------------
# Figure 1: hero — position sweep V5 p0..p5
# ---------------------------------------------------------------------------


def fig_position_sweep():
    d = _load_position_sweep()
    blocks = d["position_sweep_blocks"]

    positions = ["p0", "p1", "p2", "p3", "p4", "p5"]
    labels = [
        "last content\ntoken (V1)",
        "user-close\n<|im_end|>",
        "newline\nafter user",
        "<|im_start|>",
        "assistant",
        "final newline\n(#463 read)",
    ]
    rhos = [blocks[f"V5_p_{p}_L25"]["spearman_raw"]["rho"] for p in positions]
    pvals = [blocks[f"V5_p_{p}_L25"]["spearman_raw"]["p"] for p in positions]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # Color by whether the position is "content" (p0), "trailing template" (p1..p4),
    # or "generation boundary" (p5).
    colors = []
    for p in positions:
        if p == "p0":
            colors.append(paper_palette_role("primary"))
        elif p == "p5":
            colors.append(paper_palette_role("accent"))
        else:
            colors.append(paper_palette_role("neutral"))

    xs = np.arange(len(positions))
    bars = ax.bar(xs, rhos, color=colors, edgecolor="white", linewidth=0.8)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhline(0.468, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(-0.468, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(5.5, 0.48, "p<0.05\n(n=18)", fontsize=8, color="gray", ha="right", va="bottom")

    for bar, rho, p in zip(bars, rhos, pvals):
        y = rho + (0.04 if rho > 0 else -0.04)
        va = "bottom" if rho > 0 else "top"
        sig = "*" if p < 0.05 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{rho:+.2f}{sig}",
            ha="center",
            va=va,
            fontsize=9.5,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Spearman ρ(cosine, EM rate), n=18")
    ax.set_ylim(-0.75, 0.85)

    ax.set_title(
        "Persona-direction signal across the prompt-to-assistant boundary",
        loc="left",
        fontsize=13,
        fontweight="semibold",
        pad=36,
    )
    ax.annotate(
        "L25, lit flavor, training probes — the signal lives at the content token AND amplifies\n"
        "at the final newline (#463's read), but flips sign at the user-close <|im_end|>",
        xy=(0.0, 1.0),
        xytext=(0, 8),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="#5A5A5A",
        fontsize=10,
    )
    fig.supxlabel(
        "task #468 (n=18 EM-induction cells, * = p<0.05)",
        x=0.02,
        ha="left",
        color="#7A7A7A",
        fontsize=8,
        fontstyle="italic",
    )

    fig.tight_layout()
    savefig_paper(fig, "issue_468/hero_position_sweep", dir=str(REPO / "figures"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: layer profile — 3 extractions across deep layers
# ---------------------------------------------------------------------------


def fig_layer_profile():
    d = _load_regression_lit_training()
    blocks = d["blocks"]

    layers = [18, 20, 21, 22, 24, 25, 27]

    rho_v1 = [
        blocks[f"V1_last_prompt_token_final_content_L{L}"]["spearman_raw"]["rho"] for L in layers
    ]
    rho_p5 = [blocks[f"recompute_last_prompt_token_L{L}"]["spearman_raw"]["rho"] for L in layers]
    rho_rm = [blocks[f"recompute_response_mean_L{L}"]["spearman_raw"]["rho"] for L in layers]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    ax.plot(
        layers,
        rho_p5,
        marker="o",
        color=paper_palette_role("accent"),
        label="final newline (#463's read)",
        linewidth=2.2,
        markersize=7,
    )
    ax.plot(
        layers,
        rho_v1,
        marker="s",
        color=paper_palette_role("primary"),
        label="last user-content token (V1)",
        linewidth=2.2,
        markersize=7,
    )
    ax.plot(
        layers,
        rho_rm,
        marker="^",
        color=paper_palette_role("baseline"),
        label="response-mean (canonical persona-vectors)",
        linewidth=2.0,
        markersize=6,
        linestyle="--",
    )

    ax.axhline(0.468, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(27.2, 0.47, "p<0.05 (n=18)", fontsize=8, color="gray", ha="left", va="bottom")

    ax.set_xticks(layers)
    ax.set_xlabel("Layer (Qwen-2.5-7B-Instruct, 28 transformer blocks)")
    ax.set_ylabel("Spearman ρ(cosine, EM rate), n=18")
    ax.set_ylim(0.3, 0.75)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    ax.set_title(
        "Both prompt-side reads beat response-mean at every deep layer",
        loc="left",
        fontsize=13,
        fontweight="semibold",
        pad=36,
    )
    ax.annotate(
        "lit flavor, training probes — the gap between the two prompt-side reads is small;\n"
        "response-mean stays at borderline-significance across the band",
        xy=(0.0, 1.0),
        xytext=(0, 8),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="#5A5A5A",
        fontsize=10,
    )
    fig.supxlabel(
        "task #468 (lit-training; recompute baselines from same-pod env)",
        x=0.02,
        ha="left",
        color="#7A7A7A",
        fontsize=8,
        fontstyle="italic",
    )

    fig.tight_layout()
    savefig_paper(fig, "issue_468/layer_profile", dir=str(REPO / "figures"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: variant comparison bar chart at L25
# ---------------------------------------------------------------------------


def fig_variant_bars():
    d = _load_regression_lit_training()
    blocks = d["blocks"]

    spec = [
        ("recompute_last_prompt_token_L25", "final newline\n(#463's read)", "accent"),
        ("V1_last_prompt_token_final_content_L25", "last content token\n(V1)", "primary"),
        ("V4_response_max_L25", "max-pool over\nresponse (V4)", "neutral"),
        ("V3_response_mean_skip_k8_L25", "response-mean\nskip k=8 (V3)", "neutral"),
        ("recompute_response_mean_L25", "response-mean\n(canonical)", "baseline"),
        ("V2_last_response_token_L25", "last response\ntoken (V2)", "neutral"),
    ]

    keys = [s[0] for s in spec]
    labels = [s[1] for s in spec]
    colors = [paper_palette_role(s[2]) for s in spec]

    rhos = [blocks[k]["spearman_raw"]["rho"] for k in keys]
    pvals = [blocks[k]["spearman_raw"]["p"] for k in keys]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    xs = np.arange(len(spec))
    bars = ax.bar(xs, rhos, color=colors, edgecolor="white", linewidth=0.8)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhline(0.468, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(5.5, 0.48, "p<0.05\n(n=18)", fontsize=8, color="gray", ha="right", va="bottom")

    for bar, rho, p in zip(bars, rhos, pvals):
        y = rho + (0.04 if rho > 0 else -0.04)
        va = "bottom" if rho > 0 else "top"
        sig = "*" if p < 0.05 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{rho:+.2f}{sig}",
            ha="center",
            va=va,
            fontsize=9.5,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Spearman ρ(cosine, EM rate), n=18")
    ax.set_ylim(-0.3, 0.85)

    ax.set_title(
        "Six extraction recipes, head-to-head at L25",
        loc="left",
        fontsize=13,
        fontweight="semibold",
        pad=36,
    )
    ax.annotate(
        "lit flavor, training probes — the two prompt-side reads beat every response-side read;\n"
        "the last response token carries no usable signal",
        xy=(0.0, 1.0),
        xytext=(0, 8),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="#5A5A5A",
        fontsize=10,
    )
    fig.supxlabel(
        "task #468 (n=18 EM-induction cells, * = p<0.05)",
        x=0.02,
        ha="left",
        color="#7A7A7A",
        fontsize=8,
        fontstyle="italic",
    )

    fig.tight_layout()
    savefig_paper(fig, "issue_468/variant_bars", dir=str(REPO / "figures"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: response-mean saturation histogram
# ---------------------------------------------------------------------------


def fig_saturation():
    cossim_dir = RES / "predictor_cossim_variants_training"
    cells = sorted({p.stem.rsplit("_", 1)[0] for p in cossim_dir.glob("*_lit.json")})

    v1, rm, p5 = [], [], []
    for cell in cells:
        d = json.loads((cossim_dir / f"{cell}_lit.json").read_text())
        cos = d["cos_by_extraction"]
        v1.append(cos["last_prompt_token_final_content"]["25"])
        rm.append(cos["response_mean"]["25"])
        p5.append(cos["position_sweep"]["p5"]["25"])

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    bins = np.linspace(0.65, 1.00, 30)
    ax.hist(
        v1,
        bins=bins,
        alpha=0.7,
        color=paper_palette_role("primary"),
        label=f"V1: last content token (std={np.std(v1):.3f})",
        edgecolor="white",
        linewidth=0.6,
    )
    ax.hist(
        p5,
        bins=bins,
        alpha=0.7,
        color=paper_palette_role("accent"),
        label=f"#463 read: final newline (std={np.std(p5):.3f})",
        edgecolor="white",
        linewidth=0.6,
    )
    ax.hist(
        rm,
        bins=bins,
        alpha=0.7,
        color=paper_palette_role("baseline"),
        label=f"response-mean (std={np.std(rm):.3f})",
        edgecolor="white",
        linewidth=0.6,
    )

    ax.axvline(0.90, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(
        0.90,
        ax.get_ylim()[1] * 0.95,
        "cos=0.9",
        fontsize=8,
        color="gray",
        ha="right",
        va="top",
        rotation=90,
    )

    ax.set_xlabel("Per-cell cosine at L25 (lit, training probes)")
    ax.set_ylabel("Cells (out of 18)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    ax.set_title(
        "Response-mean is saturated toward 1.0; the prompt-side reads have dynamic range",
        loc="left",
        fontsize=12,
        fontweight="semibold",
        pad=36,
    )
    ax.annotate(
        "all 18 cells fall above cos=0.90 for response-mean; the prompt-side reads spread\n"
        "across [0.69, 0.96] and [0.77, 0.92], leaving room to rank-correlate with EM",
        xy=(0.0, 1.0),
        xytext=(0, 8),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="#5A5A5A",
        fontsize=10,
    )
    fig.supxlabel(
        "task #468 (per-cell cosines, L25 lit training)",
        x=0.02,
        ha="left",
        color="#7A7A7A",
        fontsize=8,
        fontstyle="italic",
    )

    fig.tight_layout()
    savefig_paper(fig, "issue_468/saturation", dir=str(REPO / "figures"))
    plt.close(fig)


if __name__ == "__main__":
    print("Building hero position sweep ...")
    fig_position_sweep()
    print("Building layer profile ...")
    fig_layer_profile()
    print("Building variant bars ...")
    fig_variant_bars()
    print("Building saturation histogram ...")
    fig_saturation()
    print(f"All figures saved to {FIG}/")
