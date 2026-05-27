"""Plots for clean-result task #399 v2.

Two figures (each saved as PNG + PDF + .meta.json via savefig_paper):
1. hero — two-panel Δ-vs-floor bar comparison at both probe positions.
   Shows the 9 rescue cells × 4 trigger conditions (B / B-incontext-turns /
   B-incontext-length / B-null) at k ∈ {5, 10, 20}. KEY visual: B and
   B-null bars overlap within each k-group at BOTH probe positions —
   that's what makes "uniform LoRA drift" pop visually.
2. trigger_contrast — single panel showing median(LP[B@k]) − median(LP[B-null@k])
   at first-token AND on-policy, six bars total. All bars sit near zero,
   well below the +0.5-nat trigger-conditional confirmation threshold from
   plan v1.2.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)


def median_with_ci(arr, n_boot=2000, alpha=0.05, rng=None):
    """Bootstrap CI of the median."""
    rng = rng or np.random.default_rng(42)
    arr = np.asarray(arr)
    if len(arr) == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_medians = np.median(arr[idx], axis=1)
    return (
        float(np.median(arr)),
        float(np.quantile(boot_medians, alpha / 2)),
        float(np.quantile(boot_medians, 1 - alpha / 2)),
    )


def main():
    set_paper_style("blog")
    with open("eval_results/issue_399/run_result.json") as f:
        d = json.load(f)
    lap = d["logprob_arrays_pooled"]

    # ===== Figure 1: Two-panel Δ-vs-floor =====
    # Per cell (B / B-incontext-turns / B-incontext-length / B-null) × k (5,10,20),
    # at BOTH probe positions.

    k_list = [5, 10, 20]
    families = [
        ("drift (with trigger)", "B@{k}"),
        ("turn-matched neutral", "B-incontext-turns@{k}"),
        ("length-matched neutral", "B-incontext-length@{k}"),
        ("no trigger", "B-null@{k}"),
    ]
    family_colors = {
        "drift (with trigger)": paper_palette_role("primary"),
        "turn-matched neutral": paper_palette_role("baseline"),
        "length-matched neutral": paper_palette_role("control"),
        "no trigger": paper_palette_role("accent"),
    }

    def collect(probe_array_key):
        # rows: family, cols: k
        return {
            fam_label: [lap[fam_pat.format(k=k)][probe_array_key] for k in k_list]
            for fam_label, fam_pat in families
        }

    first_token = collect("delta_pooled")
    on_policy = collect("delta_oncontent_pooled")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)

    bar_w = 0.18
    n_fams = len(families)
    x_base = np.arange(len(k_list))

    for ax, panel_data, panel_title in [
        (axes[0], first_token, "First-token probe"),
        (axes[1], on_policy, "On-policy end-of-content probe"),
    ]:
        for j, (fam_label, fam_pat) in enumerate(families):
            offsets = (j - (n_fams - 1) / 2) * bar_w
            x = x_base + offsets
            arrs = panel_data[fam_label]
            medians = [np.median(a) for a in arrs]
            cis = [median_with_ci(a) for a in arrs]
            err_lo = [m - c[1] for m, c in zip(medians, cis)]
            err_hi = [c[2] - m for m, c in zip(medians, cis)]
            ax.bar(
                x,
                medians,
                width=bar_w,
                color=family_colors[fam_label],
                yerr=[err_lo, err_hi],
                error_kw={"elinewidth": 0.8, "ecolor": "#1A1A1A"},
                label=fam_label,
            )

        ax.set_xticks(x_base)
        ax.set_xticklabels([f"k={k}" for k in k_list])
        ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(1.0, color="#888888", linewidth=0.6, linestyle=":", alpha=0.6)
        ax.set_title(panel_title, fontsize=11, loc="left", pad=10)
        ax.set_xlabel("Multi-turn drift length (k user/assistant turns)")
        ax.set_ylim(-1, 22)

    axes[0].set_ylabel("Median log p(※) − base-model floor (nats)")
    # Single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
        fontsize=9,
    )

    set_title_subtitle(
        axes[0],
        "Marker log-probability is elevated uniformly, not gated by the trigger",
        subtitle="Per-cell median Δ vs within-context base-model floor, N=384 per cell (3 seeds × 128 contexts)",
        source="Source: eval_results/issue_399/run_result.json, commit ea9fb532",
    )

    plt.tight_layout()
    savefig_paper(fig, "issue_399/hero", dir="figures/")
    plt.close(fig)

    # ===== Figure 2: Trigger-conditional contrast =====
    # Pull canonical matched-i paired diff median + CI from JSON
    # (`trigger_conditional_contrast`). 6 bars total. All bars near 0,
    # well below the +0.5-nat plan v1.2 threshold.

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    tcc_first = d["rescue_verdict_first_token"]["trigger_conditional_contrast"]
    tcc_oncontent = d["rescue_verdict_on_policy_end_of_content"]["trigger_conditional_contrast"]
    rows = []
    for k in k_list:
        key = f"B@{k}"
        for probe_label, tcc_src in [
            ("first-token", tcc_first),
            ("on-policy", tcc_oncontent),
        ]:
            rec = tcc_src[key]
            rows.append(
                {
                    "k": k,
                    "probe": probe_label,
                    "median": rec["median"],
                    "lo": rec["ci_lo"],
                    "hi": rec["ci_hi"],
                }
            )

    # Plot grouped bars
    x_base = np.arange(len(k_list))
    bar_w = 0.32
    for j, probe in enumerate(["first-token", "on-policy"]):
        sub = [r for r in rows if r["probe"] == probe]
        medians = [r["median"] for r in sub]
        err_lo = [r["median"] - r["lo"] for r in sub]
        err_hi = [r["hi"] - r["median"] for r in sub]
        x = x_base + (j - 0.5) * bar_w
        color = (
            paper_palette_role("primary")
            if probe == "first-token"
            else paper_palette_role("accent")
        )
        ax.bar(
            x,
            medians,
            width=bar_w,
            color=color,
            yerr=[err_lo, err_hi],
            error_kw={"elinewidth": 0.8, "ecolor": "#1A1A1A"},
            label=f"{probe} probe",
        )

    ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(0.5, color="#888888", linewidth=0.7, linestyle=":", alpha=0.7)
    # Annotate the 0.5-nat trigger-conditional threshold inline near the left edge
    ax.text(
        -0.45,
        0.52,
        "Plan v1.2 trigger-conditional confirmation threshold (+0.5 nats)",
        fontsize=8.5,
        color="#555555",
        ha="left",
        va="bottom",
    )

    ax.set_xticks(x_base)
    ax.set_xticklabels([f"k={k}" for k in k_list])
    ax.set_xlabel("Multi-turn drift length (k user/assistant turns)")
    ax.set_ylabel("Median log p(※)  with-trigger  −  no-trigger (nats)")
    ax.set_ylim(-0.35, 0.75)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    set_title_subtitle(
        ax,
        "Trigger-conditional contrast is null at both probe positions",
        subtitle=(
            "Median paired difference, B-with-trigger minus B-no-trigger, 95% bootstrap CI; "
            "all six bars sit well below plan v1.2's +0.5 nat threshold"
        ),
        source="Source: eval_results/issue_399/run_result.json, commit ea9fb532",
    )

    plt.tight_layout()
    savefig_paper(fig, "issue_399/trigger_contrast", dir="figures/")
    plt.close(fig)

    print("Wrote figures/issue_399/hero.{png,pdf,.meta.json}")
    print("Wrote figures/issue_399/trigger_contrast.{png,pdf,.meta.json}")


if __name__ == "__main__":
    main()
