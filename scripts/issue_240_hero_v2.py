"""Hero figure v2 for issue #240 — discrete-token EM elicitation results.

Reframed against the round-1 critique:
  - Panel A is the **token-ID** path (controlled, same-backend as #170), with
    vLLM bars retained as a lighter-shade sensitivity check.
  - Panel B is GCG alpha (Sonnet + Opus), with the actual #170 3-axis gate
    drawn (Sonnet ≤ 35, Opus ≤ 50). C ≥ 0.85 is annotated as "not measured".
  - Panel C surfaces the coherence collapse: GCG mean coherence ~50-66 vs
    Part A (helpful-init) ~88-93. This is the elephant the v1 hero hid.

Y-axis (panels A, B): alpha = 100 - aligned_rate (% misaligned).
Y-axis (panel C):     mean coherence (0-100).
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


def alpha_pct(b: float, w: float) -> float:
    return 100.0 - (8.0 * b + 44.0 * w) / 52.0


def coherence_panel(b_coh: float, w_coh: float) -> float:
    """Panel-weighted mean coherence (matches alpha's panel weighting)."""
    return (8.0 * b_coh + 44.0 * w_coh) / 52.0


def load_summary(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def part_a_load() -> dict:
    """Returns {(cell, path, judge): {alpha, coh, n_eff}}."""
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
                coh = coherence_panel(b["overall_mean_coherent"], w["overall_mean_coherent"])
                n_b = b["n_questions"] * b["n_samples_per_question"]
                n_w = w["n_questions"] * w["n_samples_per_question"]
                out[(cell, path, judge)] = {
                    "alpha": a,
                    "coh": coh,
                    "n_eff": (n_b - b.get("n_errors", 0)) + (n_w - w.get("n_errors", 0)),
                }
    return out


def part_b_load() -> dict:
    out = {}
    for L in [20, 40, 80]:
        for judge in ["sonnet", "opus"]:
            bp = PART_B_DIR / f"L{L}/hardL{L}_batched/{judge}/alignment_betley_summary.json"
            wp = PART_B_DIR / f"L{L}/hardL{L}_batched/{judge}/alignment_wang_summary.json"
            b = load_summary(bp)
            w = load_summary(wp)
            a = alpha_pct(b["overall_mean_aligned"], w["overall_mean_aligned"])
            coh = coherence_panel(b["overall_mean_coherent"], w["overall_mean_coherent"])
            n_b = b["n_questions"] * b["n_samples_per_question"]
            n_w = w["n_questions"] * w["n_samples_per_question"]
            out[(L, judge)] = {
                "alpha": a,
                "coh": coh,
                "n_eff": (n_b - b.get("n_errors", 0)) + (n_w - w.get("n_errors", 0)),
            }
    return out


def alpha_ci(alpha: float, n_eff: int) -> tuple[float, float]:
    lo, hi = proportion_ci(alpha / 100.0, n_eff)
    return alpha - lo * 100, hi * 100 - alpha


def make_figure() -> None:
    set_paper_style("neurips")

    a = part_a_load()
    bdata = part_b_load()

    palette = paper_palette(2)
    color_sonnet, color_opus = palette[0], palette[1]
    # Lighter shades for the vLLM-path sensitivity bars
    color_sonnet_lt = "#9DC3E6"
    color_opus_lt = "#F4B084"

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(13.0, 4.0), gridspec_kw={"width_ratios": [2.4, 1.0, 1.0]}
    )

    # ---------------- Panel A: Quantized prefix (token-ID = controlled) ----
    cells = [
        "s0_K16_lr5e-4",
        "s1_K32_lr5e-4",
        "s2_K32_lr1e-4",
        "s3_K64_lr5e-4",
        "s4_K64_lr1e-4",
        "s5_K64_lr1e-3",
        "s6_K64_lr5e-4_evil_init",
    ]
    cell_labels = [
        "s0\nK=16",
        "s1\nK=32",
        "s2\nK=32",
        "s3\nK=64",
        "s4\nK=64",
        "s5\nK=64",
        "s6\nevil-init",
    ]

    bar_w = 0.18
    x = np.arange(len(cells))

    son_tok = []
    opu_tok = []
    son_vll = []
    opu_vll = []
    son_tok_err = []
    opu_tok_err = []
    son_vll_err = []
    opu_vll_err = []
    for cell in cells:
        # Token-ID path (controlled comparison vs #170)
        st = a[(cell, "tokenid", "sonnet")]
        ot = a[(cell, "tokenid", "opus")]
        # vLLM path (sensitivity check, typeable string)
        sv = a[(cell, "vllm", "sonnet")]
        ov = a[(cell, "vllm", "opus")]
        if st is None:
            # s6 evil-init token-ID Sonnet was censored — use NaN
            son_tok.append(np.nan)
            son_tok_err.append((0, 0))
        else:
            son_tok.append(st["alpha"])
            son_tok_err.append(alpha_ci(st["alpha"], st["n_eff"]))
        if ot is None:
            opu_tok.append(np.nan)
            opu_tok_err.append((0, 0))
        else:
            opu_tok.append(ot["alpha"])
            opu_tok_err.append(alpha_ci(ot["alpha"], ot["n_eff"]))
        son_vll.append(sv["alpha"])
        son_vll_err.append(alpha_ci(sv["alpha"], sv["n_eff"]))
        opu_vll.append(ov["alpha"])
        opu_vll_err.append(alpha_ci(ov["alpha"], ov["n_eff"]))

    def errs(arr):
        return [[e[0] for e in arr], [e[1] for e in arr]]

    # Token-ID bars (foreground, controlled)
    ax_a.bar(
        x - 1.5 * bar_w,
        son_tok,
        bar_w,
        color=color_sonnet,
        label="Sonnet (token-ID)",
    )
    ax_a.bar(
        x - 0.5 * bar_w,
        opu_tok,
        bar_w,
        color=color_opus,
        label="Opus (token-ID)",
    )
    # vLLM bars (background, sensitivity)
    ax_a.bar(
        x + 0.5 * bar_w,
        son_vll,
        bar_w,
        color=color_sonnet_lt,
        label="Sonnet (vLLM)",
    )
    ax_a.bar(
        x + 1.5 * bar_w,
        opu_vll,
        bar_w,
        color=color_opus_lt,
        label="Opus (vLLM)",
    )
    ax_a.errorbar(
        x - 1.5 * bar_w, son_tok, yerr=errs(son_tok_err), fmt="none", ecolor="black", capsize=1.5
    )
    ax_a.errorbar(
        x - 0.5 * bar_w, opu_tok, yerr=errs(opu_tok_err), fmt="none", ecolor="black", capsize=1.5
    )
    ax_a.errorbar(
        x + 0.5 * bar_w, son_vll, yerr=errs(son_vll_err), fmt="none", ecolor="black", capsize=1.5
    )
    ax_a.errorbar(
        x + 1.5 * bar_w, opu_vll, yerr=errs(opu_vll_err), fmt="none", ecolor="black", capsize=1.5
    )

    # H2 gate axes — both Sonnet (35) and Opus (50)
    ax_a.axhline(35, ls="--", color=color_sonnet, lw=0.9)
    ax_a.axhline(50, ls="--", color=color_opus, lw=0.9)
    ax_a.text(0.0, 36, "Sonnet H2 axis (alpha <= 35)", fontsize=7, color=color_sonnet)
    ax_a.text(0.0, 51, "Opus H2 axis (alpha <= 50)", fontsize=7, color=color_opus)

    # Soft-prefix #170 reference (Sonnet alpha 21-22)
    ax_a.axhspan(21, 22, color=color_sonnet, alpha=0.10)
    ax_a.text(0.0, 23, "#170 soft-prefix Sonnet alpha (continuous)", fontsize=7, color=color_sonnet)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(cell_labels, fontsize=8)
    ax_a.set_ylabel("alpha (% misaligned)")
    ax_a.set_xlabel("Quantized #170 soft-prefix cell (token-ID = controlled)")
    ax_a.set_ylim(0, 65)
    ax_a.set_title("Panel A: Part A — Quantized soft prefix")
    ax_a.legend(loc="upper center", fontsize=6.5, ncol=2, frameon=True)

    # ---------------- Panel B: GCG from scratch — alpha + 3-axis gate ----
    L_labels = ["L=20", "L=40", "L=80"]
    son_g = [bdata[(L, "sonnet")]["alpha"] for L in [20, 40, 80]]
    opu_g = [bdata[(L, "opus")]["alpha"] for L in [20, 40, 80]]
    son_g_err = [
        alpha_ci(bdata[(L, "sonnet")]["alpha"], bdata[(L, "sonnet")]["n_eff"]) for L in [20, 40, 80]
    ]
    opu_g_err = [
        alpha_ci(bdata[(L, "opus")]["alpha"], bdata[(L, "opus")]["n_eff"]) for L in [20, 40, 80]
    ]

    xb = np.arange(3)
    bw = 0.36
    ax_b.bar(xb - bw / 2, son_g, bw, color=color_sonnet, label="Sonnet 4.5")
    ax_b.bar(xb + bw / 2, opu_g, bw, color=color_opus, label="Opus 4.7")
    ax_b.errorbar(xb - bw / 2, son_g, yerr=errs(son_g_err), fmt="none", ecolor="black", capsize=2)
    ax_b.errorbar(xb + bw / 2, opu_g, yerr=errs(opu_g_err), fmt="none", ecolor="black", capsize=2)

    ax_b.axhline(35, ls="--", color=color_sonnet, lw=0.9)
    ax_b.axhline(50, ls="--", color=color_opus, lw=0.9)
    ax_b.text(-0.45, 36, "Sonnet axis (35)", fontsize=7, color=color_sonnet)
    ax_b.text(-0.45, 51, "Opus axis (50)", fontsize=7, color=color_opus)
    ax_b.axhspan(21, 22, color=color_sonnet, alpha=0.10)

    # Annotate alpha values
    for xi, v in zip(xb - bw / 2, son_g):
        ax_b.text(xi, v + 1.6, f"{v:.1f}", ha="center", fontsize=7)
    for xi, v in zip(xb + bw / 2, opu_g):
        ax_b.text(xi, v + 1.6, f"{v:.1f}", ha="center", fontsize=7)

    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(L_labels)
    ax_b.set_xlabel("GCG suffix length")
    ax_b.set_ylabel("alpha (% misaligned)")
    ax_b.set_ylim(0, 65)
    ax_b.set_title("Panel B: Part B — GCG alpha")
    ax_b.legend(loc="upper right", fontsize=7)

    # ---------------- Panel C: Coherence collapse on GCG cells -----------
    # Compare: Part A helpful-init token-ID coherence vs GCG coherence (per L per judge)
    helpful_cells = cells[:6]  # exclude evil-init
    son_helpful_coh = np.mean([a[(c, "tokenid", "sonnet")]["coh"] for c in helpful_cells])
    opu_helpful_coh = np.mean([a[(c, "tokenid", "opus")]["coh"] for c in helpful_cells])
    son_g_coh = [bdata[(L, "sonnet")]["coh"] for L in [20, 40, 80]]
    opu_g_coh = [bdata[(L, "opus")]["coh"] for L in [20, 40, 80]]

    ax_c.bar(xb - bw / 2, son_g_coh, bw, color=color_sonnet, label="Sonnet 4.5")
    ax_c.bar(xb + bw / 2, opu_g_coh, bw, color=color_opus, label="Opus 4.7")
    # Reference line: Part A helpful-init mean coherence (Sonnet)
    ax_c.axhline(
        son_helpful_coh,
        ls=":",
        color=color_sonnet,
        lw=1.0,
        label=f"Part A helpful-init Sonnet ({son_helpful_coh:.0f})",
    )
    ax_c.axhline(
        opu_helpful_coh,
        ls=":",
        color=color_opus,
        lw=1.0,
        label=f"Part A helpful-init Opus ({opu_helpful_coh:.0f})",
    )
    for xi, v in zip(xb - bw / 2, son_g_coh):
        ax_c.text(xi, v + 1.6, f"{v:.0f}", ha="center", fontsize=7)
    for xi, v in zip(xb + bw / 2, opu_g_coh):
        ax_c.text(xi, v + 1.6, f"{v:.0f}", ha="center", fontsize=7)

    ax_c.set_xticks(xb)
    ax_c.set_xticklabels(L_labels)
    ax_c.set_xlabel("GCG suffix length")
    ax_c.set_ylabel("mean coherence (0-100)")
    ax_c.set_ylim(0, 100)
    ax_c.set_title("Panel C: GCG coherence collapse")
    ax_c.legend(loc="lower right", fontsize=6.5)

    fig.tight_layout()
    savefig_paper(fig, "issue-240/discrete_em_alpha_three_panel", dir=str(REPO / "figures"))
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
