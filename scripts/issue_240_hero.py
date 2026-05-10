"""Hero figure for issue #240 — discrete-token EM elicitation results.

Two-panel grouped bar chart:
  Panel A (Quantized prefix, Part A): 7 cells x 2 paths x 2 judges
  Panel B (GCG from scratch, Part B): 3 L-values x 2 judges

Y-axis: alpha (= 100 - aligned-rate weighted across Betley + Wang panel),
        i.e. higher = more misaligned (worse).
H2 gate (alpha <= 35) marked as dashed line on each panel.
Soft-prefix #170 reference band (alpha = 21-22) marked on Panel B.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

REPO = pathlib.Path("/home/thomasjiralerspong/explore-persona-space")
PART_A_DIR = REPO / "eval_results/issue-240/quantized"
PART_B_DIR = REPO / "eval_results/issue-240/gcg/_wandb_pull"


def alpha_pct(aligned_betley: float, aligned_wang: float) -> float:
    """alpha = % misaligned, weighted across the 52-prompt panel."""
    return 100.0 - (8.0 * aligned_betley + 44.0 * aligned_wang) / 52.0


def load_summary(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def part_a_alphas() -> dict:
    """Returns {(cell, path, judge): (alpha, aligned_b, aligned_w, n_b, n_w, e_b, e_w)}."""
    cells = [
        "s0_K16_lr5e-4",
        "s1_K32_lr5e-4",
        "s2_K32_lr1e-4",
        "s3_K64_lr5e-4",
        "s4_K64_lr1e-4",
        "s5_K64_lr1e-3",
        "s6_K64_lr5e-4_evil_init",
    ]
    out = {}
    for cell in cells:
        for path in ["vllm", "tokenid"]:
            for judge in ["sonnet", "opus"]:
                bp = PART_A_DIR / cell / path / judge / "alignment_betley_summary.json"
                wp = PART_A_DIR / cell / path / judge / "alignment_wang_summary.json"
                if not bp.exists() or not wp.exists():
                    out[(cell, path, judge)] = None
                    continue
                b = load_summary(bp)
                w = load_summary(wp)
                a = alpha_pct(b["overall_mean_aligned"], w["overall_mean_aligned"])
                n_b = b["n_questions"] * b["n_samples_per_question"]
                n_w = w["n_questions"] * w["n_samples_per_question"]
                out[(cell, path, judge)] = {
                    "alpha": a,
                    "aligned_b": b["overall_mean_aligned"],
                    "aligned_w": w["overall_mean_aligned"],
                    "n_b": n_b - b.get("n_errors", 0),
                    "n_w": n_w - w.get("n_errors", 0),
                }
    return out


def part_b_alphas() -> dict:
    out = {}
    for L in [20, 40, 80]:
        for judge in ["sonnet", "opus"]:
            bp = PART_B_DIR / f"L{L}/hardL{L}_batched/{judge}/alignment_betley_summary.json"
            wp = PART_B_DIR / f"L{L}/hardL{L}_batched/{judge}/alignment_wang_summary.json"
            b = load_summary(bp)
            w = load_summary(wp)
            a = alpha_pct(b["overall_mean_aligned"], w["overall_mean_aligned"])
            out[(L, judge)] = {
                "alpha": a,
                "aligned_b": b["overall_mean_aligned"],
                "aligned_w": w["overall_mean_aligned"],
                "n_b": b["n_questions"] * b["n_samples_per_question"] - b.get("n_errors", 0),
                "n_w": w["n_questions"] * w["n_samples_per_question"] - w.get("n_errors", 0),
            }
    return out


def make_figure() -> None:
    set_paper_style("blog")

    a = part_a_alphas()
    bdata = part_b_alphas()

    # Two side-by-side panels: vary widths so Part A panel is wider (7 cells).
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [2.4, 1.0]}
    )

    palette = paper_palette(2)
    color_sonnet, color_opus = palette[0], palette[1]

    # --- Panel A: quantized prefix, vLLM path only (the load-bearing measurement)
    cell_labels = [
        "s0\nK=16",
        "s1\nK=32",
        "s2\nK=32",
        "s3\nK=64",
        "s4\nK=64",
        "s5\nK=64",
        "s6\nK=64\nevil-init",
    ]
    cells = [
        "s0_K16_lr5e-4",
        "s1_K32_lr5e-4",
        "s2_K32_lr1e-4",
        "s3_K64_lr5e-4",
        "s4_K64_lr1e-4",
        "s5_K64_lr1e-3",
        "s6_K64_lr5e-4_evil_init",
    ]
    sonnet_alphas = []
    opus_alphas = []
    sonnet_err = []
    opus_err = []
    for cell in cells:
        s = a[(cell, "vllm", "sonnet")]
        o = a[(cell, "vllm", "opus")]
        sonnet_alphas.append(s["alpha"])
        opus_alphas.append(o["alpha"])
        # CI on misalignment proportion: alpha/100 = misaligned fraction
        # using N as effective sample = n_b + n_w (each completion is one judgment)
        n_eff_s = s["n_b"] + s["n_w"]
        n_eff_o = o["n_b"] + o["n_w"]
        # CI on alpha as proportion-misaligned. Use the full-panel N.
        ci_s = proportion_ci(s["alpha"] / 100.0, n_eff_s)
        ci_o = proportion_ci(o["alpha"] / 100.0, n_eff_o)
        # Convert CI back to alpha-units (0-100)
        sonnet_err.append(((s["alpha"] - ci_s[0] * 100), (ci_s[1] * 100 - s["alpha"])))
        opus_err.append(((o["alpha"] - ci_o[0] * 100), (ci_o[1] * 100 - o["alpha"])))

    x = np.arange(len(cells))
    bar_w = 0.4
    sonnet_lo = [e[0] for e in sonnet_err]
    sonnet_hi = [e[1] for e in sonnet_err]
    opus_lo = [e[0] for e in opus_err]
    opus_hi = [e[1] for e in opus_err]

    bars_s = ax_a.bar(x - bar_w / 2, sonnet_alphas, bar_w, color=color_sonnet, label="Sonnet 4.5")
    bars_o = ax_a.bar(x + bar_w / 2, opus_alphas, bar_w, color=color_opus, label="Opus 4.7")
    ax_a.errorbar(
        x - bar_w / 2,
        sonnet_alphas,
        yerr=[sonnet_lo, sonnet_hi],
        fmt="none",
        ecolor="black",
        capsize=2,
    )
    ax_a.errorbar(
        x + bar_w / 2,
        opus_alphas,
        yerr=[opus_lo, opus_hi],
        fmt="none",
        ecolor="black",
        capsize=2,
    )

    # H2 gate
    ax_a.axhline(35, ls="--", color="gray", lw=0.9)
    ax_a.text(0.05, 36, "H2 gate (alpha <= 35)", fontsize=8, color="gray")

    # Soft-prefix #170 reference band (alpha 21-22)
    ax_a.axhspan(21, 22, color=color_sonnet, alpha=0.12)
    ax_a.text(0.05, 18, "soft-prefix #170 alpha (Sonnet)", fontsize=8, color=color_sonnet)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(cell_labels, fontsize=8)
    ax_a.set_ylabel("alpha (% misaligned)")
    ax_a.set_xlabel("Quantized #170 soft-prefix cell (vLLM path)")
    ax_a.set_ylim(0, 65)
    ax_a.set_title("Panel A: Quantized soft prefix")
    ax_a.legend(loc="upper left", fontsize=8)

    # Annotate with values
    for rect, v in zip(bars_s, sonnet_alphas):
        ax_a.text(rect.get_x() + bar_w / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=7)
    for rect, v in zip(bars_o, opus_alphas):
        ax_a.text(rect.get_x() + bar_w / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=7)

    # --- Panel B: GCG from scratch
    L_labels = ["L=20", "L=40", "L=80"]
    sonnet_b = [bdata[(L, "sonnet")]["alpha"] for L in [20, 40, 80]]
    opus_b = [bdata[(L, "opus")]["alpha"] for L in [20, 40, 80]]
    sonnet_err_b = []
    opus_err_b = []
    for L in [20, 40, 80]:
        s = bdata[(L, "sonnet")]
        o = bdata[(L, "opus")]
        n_eff_s = s["n_b"] + s["n_w"]
        n_eff_o = o["n_b"] + o["n_w"]
        ci_s = proportion_ci(s["alpha"] / 100.0, n_eff_s)
        ci_o = proportion_ci(o["alpha"] / 100.0, n_eff_o)
        sonnet_err_b.append(((s["alpha"] - ci_s[0] * 100), (ci_s[1] * 100 - s["alpha"])))
        opus_err_b.append(((o["alpha"] - ci_o[0] * 100), (ci_o[1] * 100 - o["alpha"])))
    sb_lo = [e[0] for e in sonnet_err_b]
    sb_hi = [e[1] for e in sonnet_err_b]
    ob_lo = [e[0] for e in opus_err_b]
    ob_hi = [e[1] for e in opus_err_b]

    xb = np.arange(3)
    bars_sb = ax_b.bar(xb - bar_w / 2, sonnet_b, bar_w, color=color_sonnet, label="Sonnet 4.5")
    bars_ob = ax_b.bar(xb + bar_w / 2, opus_b, bar_w, color=color_opus, label="Opus 4.7")
    ax_b.errorbar(
        xb - bar_w / 2, sonnet_b, yerr=[sb_lo, sb_hi], fmt="none", ecolor="black", capsize=2
    )
    ax_b.errorbar(
        xb + bar_w / 2, opus_b, yerr=[ob_lo, ob_hi], fmt="none", ecolor="black", capsize=2
    )

    ax_b.axhline(35, ls="--", color="gray", lw=0.9)
    ax_b.text(-0.45, 36, "H2 gate (alpha <= 35)", fontsize=8, color="gray")
    ax_b.axhspan(21, 22, color=color_sonnet, alpha=0.12)

    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(L_labels)
    ax_b.set_xlabel("GCG suffix length")
    ax_b.set_ylim(0, 65)
    ax_b.set_title("Panel B: GCG from scratch")
    ax_b.legend(loc="upper right", fontsize=8)

    for rect, v in zip(bars_sb, sonnet_b):
        ax_b.text(rect.get_x() + bar_w / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=7)
    for rect, v in zip(bars_ob, opus_b):
        ax_b.text(rect.get_x() + bar_w / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=7)

    fig.tight_layout()
    savefig_paper(fig, "issue-240/discrete_em_alpha_two_panel", dir=str(REPO / "figures"))
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
