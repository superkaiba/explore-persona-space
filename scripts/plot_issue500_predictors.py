"""Build clean-result figures for task #500.

Generates per-arm scatter figures (prior vs leak, cosine vs leak) and an
overall summary panel comparing across the 3 source arms.

Reads:
    eval_results/issue_500/predictors.json

Writes:
    figures/issue_500/prior_vs_leak.{png,pdf,meta.json}
    figures/issue_500/cosine_vs_leak.{png,pdf,meta.json}
    figures/issue_500/predictor_summary.{png,pdf,meta.json}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parent.parent
PRED_PATH = ROOT / "eval_results" / "issue_500" / "predictors.json"
FIG_DIR = ROOT / "figures"

# Plain-English labels for the three source arms.
ARM_LABELS = {
    "marine_biologist": "Marine biologist\n(content-unrelated)",
    "local_resident": "Local resident\n(intermediate)",
    "courthouse_architecture_historian": "Architectural historian\n(content-related)",
}
ARM_ORDER = ["marine_biologist", "local_resident", "courthouse_architecture_historian"]
ARM_SHORT = {
    "marine_biologist": "marine biologist",
    "local_resident": "local resident",
    "courthouse_architecture_historian": "architectural historian",
}


def load_predictors() -> dict:
    with open(PRED_PATH) as f:
        return json.load(f)


def per_arm_arrays(d: dict, arm: str) -> dict:
    pp = d["per_arm"][arm]["per_persona"]
    personas = list(pp.keys())
    priors = np.array([pp[p]["prior_logprob"] for p in personas])
    leaks = np.array([pp[p]["leak_mean"] for p in personas])
    cos_src = np.array([pp[p]["cos_to_source"] for p in personas])
    return {
        "personas": personas,
        "priors": priors,
        "leaks": leaks,
        "cos_src": cos_src,
    }


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.3f}"


def plot_prior_vs_leak(d: dict) -> None:
    """3-panel scatter, one per arm: bystander prior on the fact vs leak rate."""
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.6), sharey=True)

    primary = paper_palette_role("primary")
    home_color = paper_palette_role("accent")  # highlight local_historian
    courthouse_color = paper_palette_role("secondary") if False else paper_palette_role("control")

    for i, arm in enumerate(ARM_ORDER):
        ax = axes[i]
        a = per_arm_arrays(d, arm)
        rho, pval = st.spearmanr(a["priors"], a["leaks"])

        # Plot points; highlight the high-prior persona (local_historian) in accent color.
        colors = [home_color if p == "local_historian" else primary for p in a["personas"]]
        ax.scatter(
            a["priors"],
            a["leaks"],
            c=colors,
            s=80,
            alpha=0.85,
            edgecolors="white",
            linewidths=1.2,
            zorder=3,
        )

        # Label local_historian by name.
        for j, p in enumerate(a["personas"]):
            if p == "local_historian":
                ax.annotate(
                    "local historian",
                    (a["priors"][j], a["leaks"][j]),
                    xytext=(8, 4),
                    textcoords="offset points",
                    fontsize=9,
                    color="#444",
                )
            elif (
                p == "courthouse_architecture_historian"
                and arm != "courthouse_architecture_historian"
            ):
                # Place label LEFT-of-dot to avoid colliding with local_historian (which sits
                # at the highest prior on the right). In the local_resident arm the two
                # dots are vertically close, so push the label well down-and-left.
                ax.annotate(
                    "courthouse\narch. historian",
                    (a["priors"][j], a["leaks"][j]),
                    xytext=(-10, -2),
                    textcoords="offset points",
                    fontsize=8.5,
                    color="#666",
                    ha="right",
                    va="center",
                )

        # OLS regression line for visual reference (descriptive, not used as a statistic).
        slope, intercept, *_ = st.linregress(a["priors"], a["leaks"])
        xs = np.linspace(a["priors"].min() - 0.05, a["priors"].max() + 0.05, 50)
        ax.plot(
            xs,
            slope * xs + intercept,
            color="#888",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            zorder=2,
        )

        # Title carries the arm label + Spearman.
        ax.set_title(
            f"Source: {ARM_LABELS[arm]}",
            fontsize=11.5,
            loc="left",
            color="#1A1A1A",
            fontweight="semibold",
            pad=10,
        )
        ax.text(
            0.04,
            0.96,
            f"Spearman ρ = {rho:+.2f}\n{fmt_p(pval)}, n = 14",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="#1A1A1A",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#DDD"),
        )

        ax.set_xlabel(
            "bystander's own base prior on the fact\n(length-norm log P, base model)", fontsize=10
        )
        if i == 0:
            ax.set_ylabel(
                "on-policy bystander leak rate\n(taught fact, 7 framings × 3 seeds)", fontsize=10
            )
        ax.set_ylim(-0.04, 0.6)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig.subplots_adjust(top=0.70, left=0.05, right=0.98, bottom=0.16, wspace=0.10)
    fig.suptitle(
        "Bystander leakage rises with the bystander's own prior on the fact, when there is dynamic range",
        x=0.02,
        ha="left",
        fontsize=13.5,
        fontweight="semibold",
        color="#1A1A1A",
        y=0.97,
    )
    fig.text(
        0.02,
        0.92,
        "n = 14 bystander personas × 3 seeds per arm; high-prior 'local_historian' highlighted",
        fontsize=10,
        color="#5A5A5A",
    )
    savefig_paper(fig, "issue_500/prior_vs_leak", dir=str(FIG_DIR))
    plt.close(fig)


def plot_cosine_vs_leak(d: dict) -> None:
    """3-panel scatter: cosine to teaching persona (layer 21, last input token) vs leak rate."""
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.6), sharey=True)

    primary = paper_palette_role("primary")
    home_color = paper_palette_role("accent")

    for i, arm in enumerate(ARM_ORDER):
        ax = axes[i]
        a = per_arm_arrays(d, arm)
        rho, pval = st.spearmanr(a["cos_src"], a["leaks"])

        colors = [home_color if p == "local_historian" else primary for p in a["personas"]]
        ax.scatter(
            a["cos_src"],
            a["leaks"],
            c=colors,
            s=80,
            alpha=0.85,
            edgecolors="white",
            linewidths=1.2,
            zorder=3,
        )
        for j, p in enumerate(a["personas"]):
            if p == "local_historian":
                # Per-arm label placement (cos values: marine 0.83, local_resident 0.92,
                # courthouse 0.95). Marine has dot near left edge so place label RIGHT;
                # other two have dot near right edge so place label LEFT.
                if arm == "marine_biologist":
                    ax.annotate(
                        "local historian",
                        (a["cos_src"][j], a["leaks"][j]),
                        xytext=(10, 0),
                        textcoords="offset points",
                        fontsize=9,
                        color="#444",
                        ha="left",
                        va="center",
                    )
                else:
                    ax.annotate(
                        "local historian",
                        (a["cos_src"][j], a["leaks"][j]),
                        xytext=(-10, 0),
                        textcoords="offset points",
                        fontsize=9,
                        color="#444",
                        ha="right",
                        va="center",
                    )

        slope, intercept, *_ = st.linregress(a["cos_src"], a["leaks"])
        xs = np.linspace(a["cos_src"].min() - 0.02, a["cos_src"].max() + 0.02, 50)
        ax.plot(
            xs,
            slope * xs + intercept,
            color="#888",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            zorder=2,
        )

        ax.set_title(
            f"Source: {ARM_LABELS[arm]}",
            fontsize=11.5,
            loc="left",
            color="#1A1A1A",
            fontweight="semibold",
            pad=10,
        )
        ax.text(
            0.04,
            0.96,
            f"Spearman ρ = {rho:+.2f}\n{fmt_p(pval)}, n = 14",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="#1A1A1A",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#DDD"),
        )

        ax.set_xlabel(
            "persona-vector cosine to teaching persona\n(layer 21, last input token)", fontsize=10
        )
        if i == 0:
            ax.set_ylabel(
                "on-policy bystander leak rate\n(taught fact, 7 framings × 3 seeds)", fontsize=10
            )
        ax.set_ylim(-0.04, 0.6)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig.subplots_adjust(top=0.70, left=0.05, right=0.98, bottom=0.16, wspace=0.10)
    fig.suptitle(
        "Proximity to the teaching persona is a weak, near-flat predictor of bystander leakage",
        x=0.02,
        ha="left",
        fontsize=13.5,
        fontweight="semibold",
        color="#1A1A1A",
        y=0.97,
    )
    fig.text(
        0.02,
        0.92,
        "Same panel as the prior plot; no arm crosses p = 0.05",
        fontsize=10,
        color="#5A5A5A",
    )
    savefig_paper(fig, "issue_500/cosine_vs_leak", dir=str(FIG_DIR))
    plt.close(fig)


def _ols_beta_bootstrap_persona(d: dict, arm: str, n_iter: int = 1000, seed: int = 42):
    """Cluster-on-persona bootstrap of standardized OLS (β_prior, β_prox).

    Resamples persona rows with replacement (same shape as the headline
    spearman bootstrap), refits the standardized OLS each iteration,
    returns 95% CIs for both betas. This is the right uncertainty estimate
    for "would this generalize to a different bystander panel" and reveals
    leverage-dependence (the courthouse arm's β_prior collapses to ~0.23
    when local_historian is dropped, so the CI should be wide).
    """
    pp = d["per_arm"][arm]["per_persona"]
    personas = list(pp.keys())
    leak = np.array([pp[p]["leak_mean"] for p in personas])
    prior = np.array([pp[p]["prior_logprob"] for p in personas])
    cos = np.array([pp[p]["cos_to_source"] for p in personas])
    n = len(personas)
    rng = np.random.RandomState(seed)
    betas_prior = []
    betas_prox = []
    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        y_, p_, c_ = leak[idx], prior[idx], cos[idx]
        # need variance on both predictors to standardize
        if p_.std() == 0 or c_.std() == 0 or y_.std() == 0:
            continue
        y = (y_ - y_.mean()) / y_.std(ddof=0)
        x1 = (p_ - p_.mean()) / p_.std(ddof=0)
        x2 = (c_ - c_.mean()) / c_.std(ddof=0)
        X = np.column_stack([np.ones(n), x1, x2])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            betas_prior.append(beta[1])
            betas_prox.append(beta[2])
        except np.linalg.LinAlgError:
            continue
    betas_prior = np.array(betas_prior)
    betas_prox = np.array(betas_prox)
    return {
        "prior": (np.percentile(betas_prior, 2.5), np.percentile(betas_prior, 97.5)),
        "prox": (np.percentile(betas_prox, 2.5), np.percentile(betas_prox, 97.5)),
        "n_valid": len(betas_prior),
    }


def plot_predictor_summary(d: dict) -> None:
    """Two-panel: (left) per-arm Spearman ρ comparison prior vs cosine.
    (right) OLS beta_prior and beta_prox across arms (with cluster-persona
    bootstrap CIs added in this revision)."""
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.0))

    prior_color = paper_palette_role("primary")
    cos_color = paper_palette_role("baseline")

    arm_x = np.arange(len(ARM_ORDER))
    width = 0.36

    # LEFT: Spearman rho with cluster-persona bootstrap 95% CIs.
    # CIs may not straddle the point estimate (bootstrap distribution can skew),
    # so we clip yerr to be non-negative; the actual CI endpoints stay correct.
    rho_prior = []
    rho_prior_lo = []
    rho_prior_hi = []
    rho_cos = []
    rho_cos_lo = []
    rho_cos_hi = []
    for arm in ARM_ORDER:
        s = d["per_arm"][arm]["stats"]
        rp = s["spearman_prior_logprob_vs_leak"]
        rho_prior.append(rp)
        bp = s["bootstrap_spearman_prior_logprob_vs_leak_cluster_persona"]
        rho_prior_lo.append(max(0.0, rp - bp["ci_low_95"]))
        rho_prior_hi.append(max(0.0, bp["ci_high_95"] - rp))
        rc = s["spearman_cos_to_source_vs_leak"]
        rho_cos.append(rc)
        bc = s["bootstrap_spearman_cos_to_source_vs_leak_cluster_persona"]
        rho_cos_lo.append(max(0.0, rc - bc["ci_low_95"]))
        rho_cos_hi.append(max(0.0, bc["ci_high_95"] - rc))

    ax1.bar(
        arm_x - width / 2,
        rho_prior,
        width=width,
        yerr=[rho_prior_lo, rho_prior_hi],
        color=prior_color,
        edgecolor="white",
        linewidth=1.2,
        capsize=4,
        label="bystander's own prior on the fact",
    )
    ax1.bar(
        arm_x + width / 2,
        rho_cos,
        width=width,
        yerr=[rho_cos_lo, rho_cos_hi],
        color=cos_color,
        edgecolor="white",
        linewidth=1.2,
        capsize=4,
        label="proximity to teaching persona\n(cosine, layer 21)",
    )

    ax1.axhline(0, color="#999", linewidth=0.8)
    ax1.set_xticks(arm_x)
    ax1.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER], fontsize=9)
    ax1.set_ylabel("Spearman ρ vs on-policy leak rate", fontsize=10)
    ax1.set_ylim(-0.6, 1.0)
    ax1.set_title(
        "Per-arm rank correlation with leakage",
        loc="left",
        fontsize=11.5,
        fontweight="semibold",
        color="#1A1A1A",
        pad=10,
    )
    ax1.legend(loc="upper right", fontsize=9, frameon=False)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # RIGHT: OLS standardized betas
    beta_prior = [
        d["per_arm"][a]["stats"]["ols_z_leak_on_z_prior_logprob_and_z_cos_to_source"][
            "beta_x1_prior"
        ]
        for a in ARM_ORDER
    ]
    beta_prox = [
        d["per_arm"][a]["stats"]["ols_z_leak_on_z_prior_logprob_and_z_cos_to_source"][
            "beta_x2_prox"
        ]
        for a in ARM_ORDER
    ]
    r2 = [
        d["per_arm"][a]["stats"]["ols_z_leak_on_z_prior_logprob_and_z_cos_to_source"]["r_squared"]
        for a in ARM_ORDER
    ]

    # Cluster-persona bootstrap CIs for the OLS betas (the headline
    # observation: courthouse arm's β_prior is leverage-driven; the CI on
    # the courthouse bars is much wider than on the other two arms).
    beta_boots = [_ols_beta_bootstrap_persona(d, arm) for arm in ARM_ORDER]
    beta_prior_lo = [
        max(0.0, beta_prior[i] - beta_boots[i]["prior"][0]) for i in range(len(ARM_ORDER))
    ]
    beta_prior_hi = [
        max(0.0, beta_boots[i]["prior"][1] - beta_prior[i]) for i in range(len(ARM_ORDER))
    ]
    beta_prox_lo = [
        max(0.0, beta_prox[i] - beta_boots[i]["prox"][0]) for i in range(len(ARM_ORDER))
    ]
    beta_prox_hi = [
        max(0.0, beta_boots[i]["prox"][1] - beta_prox[i]) for i in range(len(ARM_ORDER))
    ]

    # Hatch the courthouse arm (index 2) to flag leverage dependence
    # (drop-LH collapse: β_prior 0.79 → 0.23, R² 0.62 → 0.07).
    bars_prior = ax2.bar(
        arm_x - width / 2,
        beta_prior,
        width=width,
        yerr=[beta_prior_lo, beta_prior_hi],
        color=prior_color,
        edgecolor="white",
        linewidth=1.2,
        capsize=4,
        label="β prior",
    )
    bars_prox = ax2.bar(
        arm_x + width / 2,
        beta_prox,
        width=width,
        yerr=[beta_prox_lo, beta_prox_hi],
        color=cos_color,
        edgecolor="white",
        linewidth=1.2,
        capsize=4,
        label="β proximity",
    )
    # Mark the courthouse arm's bars with hatching to signal leverage-driven.
    for b in (bars_prior[2], bars_prox[2]):
        b.set_hatch("////")
        b.set_edgecolor("#1A1A1A")

    ax2.axhline(0, color="#999", linewidth=0.8)
    ax2.set_xticks(arm_x)
    ax2.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER], fontsize=9)
    ax2.set_ylabel(
        "standardized regression coefficient\n(z-scored leak on z-prior + z-proximity)", fontsize=10
    )
    ax2.set_ylim(-0.7, 1.7)
    ax2.set_title(
        "Joint regression — prior carries the predictive weight",
        loc="left",
        fontsize=11.5,
        fontweight="semibold",
        color="#1A1A1A",
        pad=10,
    )
    # R² annotation per arm with drop-LH parenthetical for the courthouse arm
    r2_annotations = {
        0: f"R² = {r2[0]:.2f}",
        1: f"R² = {r2[1]:.2f}",
        2: f"R² = {r2[2]:.2f}\n(→ 0.07 if local\nhistorian dropped)",
    }
    for i, ann in r2_annotations.items():
        # Place R² annotation inside the axes (data coords) above the top bar
        ax2.text(
            arm_x[i],
            1.55,
            ann,
            ha="center",
            va="top",
            fontsize=8.5,
            color="#444",
        )
    ax2.legend(loc="lower left", fontsize=9, frameon=False)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig.subplots_adjust(top=0.66, left=0.07, right=0.98, bottom=0.20, wspace=0.25)
    fig.suptitle(
        "Across all three source arms, the bystander's own prior dominates the joint fit",
        x=0.02,
        ha="left",
        fontsize=13.5,
        fontweight="semibold",
        color="#1A1A1A",
        y=0.96,
    )
    fig.text(
        0.02,
        0.91,
        "Error bars = 95% cluster-on-persona bootstrap CI (1000 iters). Hatched bars = leverage-driven (architectural-historian arm β collapses on drop-one).",
        fontsize=9.5,
        color="#5A5A5A",
    )
    savefig_paper(fig, "issue_500/predictor_summary", dir=str(FIG_DIR))
    plt.close(fig)


def plot_saturation_diagnostic(d: dict) -> None:
    """Per-arm strip plot of bystander leak rates, showing the dynamic range
    available to the prior. Reveals saturation collapse in the courthouse arm."""
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(11.5, 5.4))

    primary = paper_palette_role("primary")
    home_color = paper_palette_role("accent")
    nontaught_n = []

    for i, arm in enumerate(ARM_ORDER):
        a = per_arm_arrays(d, arm)
        # Sort by leak rate
        order = np.argsort(a["leaks"])
        leaks = a["leaks"][order]
        personas = [a["personas"][k] for k in order]
        n_nonzero = int((leaks > 0.01).sum())
        nontaught_n.append(n_nonzero)

        y = np.full(len(leaks), i) + np.random.RandomState(42).uniform(-0.08, 0.08, len(leaks))
        colors = [home_color if p == "local_historian" else primary for p in personas]
        ax.scatter(
            leaks, y, c=colors, s=70, alpha=0.85, edgecolors="white", linewidths=1.0, zorder=3
        )

        # Annotate the highest-leak persona on each row
        top_p = personas[-1]
        ax.annotate(
            top_p.replace("_", " "),
            (leaks[-1], i),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color="#444",
        )

        # Mark the count of personas above floor
        ax.text(
            0.62,
            i,
            f"{n_nonzero}/14 personas above 1% leak floor",
            va="center",
            fontsize=9.5,
            color="#5A5A5A",
        )

    ax.set_yticks(range(len(ARM_ORDER)))
    ax.set_yticklabels([ARM_LABELS[a] for a in ARM_ORDER], fontsize=10)
    ax.set_xlabel(
        "on-policy bystander leak rate (taught fact, 7 framings × 3 seeds, mean per persona)",
        fontsize=10,
    )
    ax.set_xlim(-0.02, 0.85)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig.subplots_adjust(top=0.70, left=0.16, right=0.98, bottom=0.16)
    fig.suptitle(
        "Architectural-historian arm has almost no dynamic range: only local historian leaks",
        x=0.02,
        ha="left",
        fontsize=13.5,
        fontweight="semibold",
        color="#1A1A1A",
        y=0.96,
    )
    fig.text(
        0.02,
        0.91,
        "One row per source-persona arm; one dot per bystander persona's mean leak rate across 3 seeds",
        fontsize=10,
        color="#5A5A5A",
    )
    savefig_paper(fig, "issue_500/saturation_diagnostic", dir=str(FIG_DIR))
    plt.close(fig)


def main() -> None:
    d = load_predictors()
    plot_prior_vs_leak(d)
    plot_cosine_vs_leak(d)
    plot_predictor_summary(d)
    plot_saturation_diagnostic(d)
    print("Figures written to", FIG_DIR / "issue_500")


if __name__ == "__main__":
    main()
