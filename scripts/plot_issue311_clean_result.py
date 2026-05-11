"""Generate clean-result figures for issue #311.

Five figures:
1. fig1_asymmetric_leakage — hero. Aonly vs Bonly per-persona marker rate.
2. fig2_h1_scatter — partial-Spearman scatter of r_p vs |t|, wrong-direction.
3. fig3_null_distributions — Null A (1000 random axes) histogram + Null B
   (16 fixed-comedian) strip plot, with real-rho marker on each.
4. fig4_steering_bars — Arm 2 steering: 11 arms, all-zero rates (descriptive).
5. fig5_position_distribution — start/early/mid/tail position of [ZLT] across
   the 3 LoRAs (Aonly leaks broadly at tail; Bonly fires at tail/start on
   comedian only; joint mixes both modes).

Output: figures/issue_311/{stem}.{png,pdf,meta.json}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    proportion_ci,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "eval_results/issue_311"
FIG_DIR = ROOT / "figures/issue_311"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_results():
    """Load all result JSONs."""
    return {
        "analysis": _load_json(RESULTS_DIR / "analysis.json"),
        "null": _load_json(RESULTS_DIR / "null_distributions.json"),
        "joint": _load_json(RESULTS_DIR / "arm1_marker_rates_joint_paramedic_comedian.json"),
        "Aonly": _load_json(RESULTS_DIR / "arm1_marker_rates_Aonly_paramedic_comedian.json"),
        "Bonly": _load_json(RESULTS_DIR / "arm1_marker_rates_Bonly_paramedic_comedian.json"),
        "arm2": _load_json(RESULTS_DIR / "arm2_steered_rates_paramedic_comedian.json"),
        "pair": _load_json(RESULTS_DIR / "pair_selection.json"),
    }


# Persona ordering: sources first, then bystanders by |t|, ascending.
def _ordered_personas(analysis: dict, pair_sel: dict) -> tuple[list[str], list[float]]:
    """Return (personas, |t| values) with sources first then bystanders by |t| ascending."""
    A, B = analysis["pair"]
    bystanders = analysis["bystanders"]
    t_vals = analysis["t_vals"]
    abs_t = [abs(t) for t in t_vals]
    # sort bystanders by |t|
    order = np.argsort(abs_t)
    bystanders_sorted = [bystanders[i] for i in order]
    abs_t_sorted = [abs_t[i] for i in order]
    # sources first: A, B, then bystanders by |t|
    personas = [A, B] + bystanders_sorted
    # |t| for sources is 0.5 * (1 - cos(A,B)) endpoint; compute for completeness.
    # actually the source |t| ≈ |0.5(1−cos)| ≈ half the endpoint distance. Mark as t_endpoint.
    return personas, abs_t_sorted, bystanders_sorted


# =====================================================================
# Figure 1 — Hero: asymmetric per-persona leakage (Aonly vs Bonly)
# =====================================================================


def fig1_asymmetric_leakage(data: dict) -> None:
    """Side-by-side bar chart, 19 personas, Aonly + Bonly, sorted by |t|."""
    analysis = data["analysis"]
    A, B = analysis["pair"]
    bystanders = analysis["bystanders"]
    t_vals = analysis["t_vals"]
    abs_t = np.abs(t_vals)
    # Sort bystanders by |t| ascending (most-axis-aligned first)
    order = np.argsort(abs_t)
    sorted_bystanders = [bystanders[i] for i in order]

    # Build full persona list: A, B, then bystanders sorted by |t|
    personas = [A, B] + sorted_bystanders
    n = len(personas)

    rates_A = data["Aonly"]["rates_aggregated"]
    rates_B = data["Bonly"]["rates_aggregated"]
    rates_joint = data["joint"]["rates_aggregated"]

    A_vals = np.array([rates_A[p] for p in personas])
    B_vals = np.array([rates_B[p] for p in personas])
    joint_vals = np.array([rates_joint[p] for p in personas])

    # Compute 95% Wilson-equivalent CIs (n_eff = 20 questions per persona; we use 400 = K=20 × Q=20)
    n_total = 400
    A_ci = [proportion_ci(p, n_total) for p in A_vals]
    B_ci = [proportion_ci(p, n_total) for p in B_vals]
    joint_ci = [proportion_ci(p, n_total) for p in joint_vals]

    A_err = np.array([[p - lo, hi - p] for p, (lo, hi) in zip(A_vals, A_ci)]).T
    B_err = np.array([[p - lo, hi - p] for p, (lo, hi) in zip(B_vals, B_ci)]).T
    joint_err = np.array([[p - lo, hi - p] for p, (lo, hi) in zip(joint_vals, joint_ci)]).T

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(11.0, 4.6))

    x = np.arange(n)
    w = 0.27
    color_A = paper_palette_role("primary")
    color_B = paper_palette_role("baseline")
    color_J = paper_palette_role("control")

    ax.bar(
        x - w,
        A_vals,
        w,
        yerr=A_err,
        label=f"A-only LoRA (source: {A})",
        color=color_A,
        edgecolor="white",
        capsize=2,
    )
    ax.bar(
        x,
        B_vals,
        w,
        yerr=B_err,
        label=f"B-only LoRA (source: {B})",
        color=color_B,
        edgecolor="white",
        capsize=2,
    )
    ax.bar(
        x + w,
        joint_vals,
        w,
        yerr=joint_err,
        label="joint LoRA (both sources)",
        color=color_J,
        edgecolor="white",
        capsize=2,
    )

    # x labels
    labels = list(personas)
    # bold source names
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=8)
    for tick, lab in zip(ax.get_xticklabels(), labels):
        if lab in (A, B):
            tick.set_fontweight("bold")

    # vertical separator between sources and bystanders
    ax.axvline(1.5, color="0.6", linestyle=":", linewidth=0.8, zorder=0)

    ax.set_ylabel("[ZLT] marker rate (per-persona, K=20 × 20 questions)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    set_title_subtitle(
        ax,
        title="A-only LoRA leaks the marker broadly; B-only LoRA stays hyper-local",
        subtitle=(
            "Per-persona [ZLT] firing rate (n=400 each) under three LoRAs trained on paramedic (A), comedian (B), "
            "or both. Bystanders sorted by |t(p)| ascending."
        ),
        source="eval_results/issue_311/arm1_marker_rates_*.json",
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_311/fig1_asymmetric_leakage", dir="figures/")
    plt.close(fig)


# =====================================================================
# Figure 2 — H1 scatter: r_p vs |t|, wrong-direction positive ρ
# =====================================================================


def fig2_h1_scatter(data: dict) -> None:
    """Scatter of residual r_p vs |t(p)| over 17 bystanders + regression line."""
    analysis = data["analysis"]
    bystanders = analysis["bystanders"]
    t_vals = np.array(analysis["t_vals"])
    abs_t = np.abs(t_vals)
    r_p = np.array(analysis["r_p_primary_per_persona"])
    s_vals = np.array(analysis["s_vals"])
    rho_primary = analysis["h1_primary"]["rho"]
    p_primary = analysis["h1_primary"]["p"]
    n = analysis["h1_primary"]["n"]

    # Residualize r_p and abs_t on s (linear) to visualize the partial Spearman
    p_r = np.polyfit(s_vals, r_p, 1)
    r_p_resid = r_p - np.polyval(p_r, s_vals)
    p_t = np.polyfit(s_vals, abs_t, 1)
    abs_t_resid = abs_t - np.polyval(p_t, s_vals)

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.8, 4.6))

    color_pt = paper_palette_role("primary")
    color_fit = paper_palette_role("accent")

    ax.scatter(
        abs_t_resid, r_p_resid, s=44, color=color_pt, edgecolor="white", linewidth=0.6, zorder=3
    )
    # OLS fit on residuals
    fit = np.polyfit(abs_t_resid, r_p_resid, 1)
    xline = np.linspace(abs_t_resid.min() - 0.02, abs_t_resid.max() + 0.02, 50)
    yline = np.polyval(fit, xline)
    ax.plot(
        xline,
        yline,
        color=color_fit,
        linewidth=2.0,
        zorder=2,
        label="OLS fit (slope > 0 — wrong direction)",
    )

    ax.axhline(0, color="0.7", linestyle=":", linewidth=0.8, zorder=1)

    # Annotate prediction direction
    ax.annotate(
        "Predicted direction:\nlow |t| → higher r_p\n(ρ < 0)",
        xy=(0.04, 0.96),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8.5,
        color="0.4",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", lw=0.6),
    )
    # Stats annotation
    ax.annotate(
        f"partial Spearman ρ = +{rho_primary:.3f}\n"
        f"p = {p_primary:.3f} (one-sided, less)\n"
        f"N = {n} bystanders",
        xy=(0.96, 0.04),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", lw=0.8),
    )

    ax.set_xlabel("|t(p)| residualized on s(p) — distance from A↔B midpoint")
    ax.set_ylabel("r_p residualized on s(p) — Bernoulli-union residual")
    ax.legend(loc="upper left", frameon=False, fontsize=9, bbox_to_anchor=(0.0, 0.83))

    set_title_subtitle(
        ax,
        title="Bystander residual marker rate trends with |t|, not against it",
        subtitle=(
            "Pre-registered hypothesis predicted ρ < 0 (low-|t| midpoint elevation); observed sign is opposite. "
            "N=17 bystanders, single seed, LOW confidence pre-commit."
        ),
        source="eval_results/issue_311/analysis.json",
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_311/fig2_h1_scatter", dir="figures/")
    plt.close(fig)


# =====================================================================
# Figure 3 — Null distributions (Null A + Null B)
# =====================================================================


def fig3_null_distributions(data: dict) -> None:
    null = data["null"]
    rho_real = null["rho_primary_real"]
    nA_rhos = np.array(null["null_a_random_axis"]["rhos"])
    nA_pct = null["null_a_random_axis"]["percentile_rank"]
    nB_rhos = np.array(null["null_b_fixed_b"]["rhos"])
    nB_pct = null["null_b_fixed_b"]["percentile_rank"]

    set_paper_style("blog")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    color_hist = paper_palette_role("baseline")
    color_real = paper_palette_role("accent")

    # === Null A: 1000 random-axis permutations
    axA.hist(nA_rhos, bins=40, color=color_hist, edgecolor="white", alpha=0.85)
    axA.axvline(rho_real, color=color_real, linewidth=2.4, label=f"real ρ = +{rho_real:.3f}")
    axA.axvline(
        np.quantile(nA_rhos, 0.05),
        color="0.5",
        linewidth=1.0,
        linestyle="--",
        label="5th percentile (pass threshold)",
    )
    axA.set_xlabel("partial Spearman ρ (random axis label)")
    axA.set_ylabel("count")
    axA.set_title(
        f"Null A: 1000 random-axis permutations\nreal ρ at percentile {nA_pct:.3f} (PASS = ≤0.05)",
        loc="left",
        fontsize=10,
    )
    axA.legend(loc="upper left", frameon=False, fontsize=8.5)

    # === Null B: 16 fixed-comedian alt-A permutations  (strip plot)
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.18, 0.18, size=len(nB_rhos))
    axB.scatter(
        nB_rhos,
        jitter,
        s=40,
        color=color_hist,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label=f"alt-A pairs (N={len(nB_rhos)})",
    )
    axB.axvline(rho_real, color=color_real, linewidth=2.4, label=f"real ρ = +{rho_real:.3f}")
    axB.set_xlabel("partial Spearman ρ (B fixed = comedian, A randomized)")
    axB.set_ylim(-0.5, 0.5)
    axB.set_yticks([])
    axB.set_title(
        f"Null B: 16 fixed-B=comedian alt-A pairs\nreal ρ at percentile {nB_pct:.4f} (PASS = ≤0.05)",
        loc="left",
        fontsize=10,
    )
    axB.legend(loc="upper left", frameon=False, fontsize=8.5)

    fig.suptitle(
        "Real ρ lies deep in the wrong tail of both null distributions",
        fontsize=12.5,
        x=0.02,
        y=0.99,
        ha="left",
        fontweight="semibold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    savefig_paper(fig, "issue_311/fig3_null_distributions", dir="figures/")
    plt.close(fig)


# =====================================================================
# Figure 4 — Arm 2 steering bars
# =====================================================================


def fig4_steering_bars(data: dict) -> None:
    arm2 = data["arm2"]
    arms = arm2["arms"]
    # Order: v_A, v_B, v_mid, neg_v_A, neg_v_B, neg_v_mid, random_iso_*
    arm_order = [
        "v_A",
        "v_B",
        "v_mid",
        "neg_v_A",
        "neg_v_B",
        "neg_v_mid",
        "random_iso_vA",
        "random_iso_vB",
        "random_iso_vmid",
        "random_iso_vA_seed2",
        "random_iso_vA_seed3",
    ]
    rates = {a["arm"]: a["rate_aggregated"] for a in arms}
    cis = {a["arm"]: a["ci_95"] for a in arms}
    vals = np.array([rates[a] for a in arm_order])
    ci_lo = np.array([cis[a][0] for a in arm_order])
    ci_hi = np.array([cis[a][1] for a in arm_order])
    err = np.array([vals - ci_lo, ci_hi - vals])

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(10.0, 4.0))

    # color by arm family
    color_centroid = paper_palette_role("primary")
    color_antipodal = paper_palette_role("baseline")
    color_random = paper_palette_role("control")
    colors = [
        color_centroid,
        color_centroid,
        color_centroid,
        color_antipodal,
        color_antipodal,
        color_antipodal,
        color_random,
        color_random,
        color_random,
        color_random,
        color_random,
    ]

    x = np.arange(len(arm_order))
    bars = ax.bar(x, vals, color=colors, edgecolor="white", capsize=2, yerr=err)

    # vertical separators between arm families
    ax.axvline(2.5, color="0.85", linewidth=0.7, zorder=0)
    ax.axvline(5.5, color="0.85", linewidth=0.7, zorder=0)

    # nice labels
    labels = [
        "v_A",
        "v_B",
        "v_mid",
        "−v_A",
        "−v_B",
        "−v_mid",
        "rand(‖v_A‖)",
        "rand(‖v_B‖)",
        "rand(‖v_mid‖)",
        "rand(‖v_A‖) s2",
        "rand(‖v_A‖) s3",
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("[ZLT] marker rate (Arm 2, base model + L20 hook, n=400)")
    ax.set_ylim(0, 0.05)

    # Group labels INSIDE the chart at the top — color-coded
    ax.annotate(
        "centroid",
        xy=(1.0, 0.92),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="top",
        fontsize=8.5,
        color=color_centroid,
        fontweight="semibold",
    )
    ax.annotate(
        "antipodal",
        xy=(4.0, 0.92),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="top",
        fontsize=8.5,
        color=color_antipodal,
        fontweight="semibold",
    )
    ax.annotate(
        "random isotropic",
        xy=(8.0, 0.92),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="top",
        fontsize=8.5,
        color=color_random,
        fontweight="semibold",
    )

    set_title_subtitle(
        ax,
        title="L20 steering on the BASE model fires the marker at 0% on all 11 arms",
        subtitle=(
            "Descriptive: no PASS verdict pre-registered. At coefficient c=2.0, neither persona-vector "
            "centroids, their antipodes, nor norm-matched random directions elicit [ZLT]."
        ),
        source="eval_results/issue_311/arm2_steered_rates_paramedic_comedian.json",
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_311/fig4_steering_bars", dir="figures/")
    plt.close(fig)


# =====================================================================
# Figure 5 — Position distribution of [ZLT] firings
# =====================================================================


def fig5_position_distribution(data: dict) -> None:
    """Stacked bars: start/early/mid/tail rel-position bins for each LoRA's fires."""
    # Compute per LoRA the position distribution of [ZLT]
    positions = {"joint": [], "Aonly": [], "Bonly": []}
    for tag in positions:
        comp_path = RESULTS_DIR / f"arm1_completions_{tag}_paramedic_comedian.json"
        comp = _load_json(comp_path)
        for persona, qs in comp.items():
            for q, completions in qs.items():
                for c in completions:
                    if "[ZLT]" in c and len(c) > 0:
                        idx = c.find("[ZLT]")
                        rel = idx / len(c)
                        positions[tag].append(rel)

    def _bin(ps):
        ps = np.array(ps)
        if len(ps) == 0:
            return np.zeros(4)
        return np.array(
            [
                np.mean(ps < 0.10),
                np.mean((ps >= 0.10) & (ps < 0.50)),
                np.mean((ps >= 0.50) & (ps < 0.90)),
                np.mean(ps >= 0.90),
            ]
        )

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.6, 4.0))

    tags = ["joint", "Aonly", "Bonly"]
    counts = {t: len(positions[t]) for t in tags}
    fractions = np.stack([_bin(positions[t]) for t in tags])  # (3, 4)

    labels = [
        f"joint (n={counts['joint']})",
        f"A-only (n={counts['Aonly']})",
        f"B-only (n={counts['Bonly']})",
    ]
    bin_labels = ["start (<10%)", "early (10-50%)", "mid (50-90%)", "tail (≥90%)"]
    colors = [
        paper_palette_role("accent"),
        paper_palette_role("primary"),
        paper_palette_role("baseline"),
        paper_palette_role("control"),
    ]

    y = np.arange(len(tags))
    left = np.zeros(len(tags))
    for j, (label, color) in enumerate(zip(bin_labels, colors)):
        ax.barh(
            y, fractions[:, j], left=left, color=color, edgecolor="white", label=label, height=0.6
        )
        # numbers on segments above 6%
        for i, frac in enumerate(fractions[:, j]):
            if frac >= 0.06:
                ax.annotate(
                    f"{frac:.0%}",
                    xy=(left[i] + frac / 2, y[i]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="semibold",
                )
        left += fractions[:, j]

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("share of firing completions, by within-completion position")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, ncol=4, bbox_to_anchor=(1.0, 1.18))

    set_title_subtitle(
        ax,
        title="Where in the completion does [ZLT] fire?",
        subtitle=(
            "Joint emits the marker as a delimiter at the START in 30% of fires; "
            "single-source LoRAs emit it at the trained TAIL position (>96% and >83%)."
        ),
        source="eval_results/issue_311/arm1_completions_*.json",
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_311/fig5_position_distribution", dir="figures/")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_results()
    fig1_asymmetric_leakage(data)
    print("[ok] fig1_asymmetric_leakage")
    fig2_h1_scatter(data)
    print("[ok] fig2_h1_scatter")
    fig3_null_distributions(data)
    print("[ok] fig3_null_distributions")
    fig4_steering_bars(data)
    print("[ok] fig4_steering_bars")
    fig5_position_distribution(data)
    print("[ok] fig5_position_distribution")
    print("All figures written to figures/issue_311/")


if __name__ == "__main__":
    main()
