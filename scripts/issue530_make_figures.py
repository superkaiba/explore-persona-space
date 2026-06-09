"""Generate paper-quality figures for #530 clean-result body.

Hero (Fig 1): side-by-side partial-Spearman ρ for shadow_angle and d_nn at
#504's saturated anchor vs #530's de-saturated anchor. The headline.

Supporting (Fig 2): bystander resolution diagnostic — per-cell median bystander
log P(marker) at the post-response slot, #504 vs #530, with the argmax-ceiling
band overlaid. Visual proof the #530 sweep is genuinely de-saturated where #504
was saturated.

Supporting (Fig 3): raw scatter of held-out ΔG vs each of the three positional
predictors (d_source, d_nn, shadow_angle), pooled across 432 rows of #530 only.
The raw-counterpart to the partialled hero.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMP_PATH = REPO_ROOT / "eval_results/issue_530/comparison_504_vs_530.json"
OUT_DIR = REPO_ROOT / "figures/issue_530"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load() -> dict:
    return json.loads(COMP_PATH.read_text())


# ---------------------------------------------------------------------------
# Figure 1 — hero: partial-Spearman ρ, two predictors, two anchors.
# ---------------------------------------------------------------------------


def fig_hero_partial_rho(d: dict) -> None:
    set_paper_style("blog")

    fit_504 = d["issue_504_analysis"]["pooled_fit"]["partial_spearman"]
    fit_530 = d["issue_530_analysis"]["pooled_fit"]["partial_spearman"]
    holm_504 = d["issue_504_analysis"]["pooled_fit"]["holm"]
    holm_530 = d["issue_530_analysis"]["pooled_fit"]["holm"]

    predictors = ["shadow_angle", "d_nearest_neg_nd"]
    labels = ["Shadow-angle predictor", "Distance-to-nearest-negative predictor"]
    rho_504 = [fit_504[p]["rho"] for p in predictors]
    rho_530 = [fit_530[p]["rho"] for p in predictors]
    p_504 = [holm_504[p]["p"] for p in predictors]
    p_530 = [holm_530[p]["p"] for p in predictors]

    x = np.arange(len(predictors))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    bars_504 = ax.bar(
        x - width / 2,
        rho_504,
        width,
        label="At the saturated anchor (lr 1e-4)",
        color=paper_palette_role("baseline"),
        edgecolor="black",
        linewidth=0.6,
    )
    bars_530 = ax.bar(
        x + width / 2,
        rho_530,
        width,
        label="At the de-saturated anchor (lr 5e-6)",
        color=paper_palette_role("primary"),
        edgecolor="black",
        linewidth=0.6,
    )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.8)
    ax.set_ylabel(
        "Partial Spearman ρ\n(controlling for the 5 other predictors)",
        fontsize=10,
    )

    ax.set_ylim(-0.55, 0.55)

    # Annotate each bar with the rho value + significance star.
    def _annotate(bars, rhos, ps):
        for bar, rho, p in zip(bars, rhos, ps, strict=True):
            star = "***" if p < 1e-6 else ("**" if p < 1e-3 else ("*" if p < 0.05 else "n.s."))
            height = bar.get_height()
            offset = 0.025 if height >= 0 else -0.045
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"ρ = {rho:+.2f} ({star})",
                ha="center",
                fontsize=8.6,
            )

    _annotate(bars_504, rho_504, p_504)
    _annotate(bars_530, rho_530, p_530)

    ax.legend(loc="lower right", fontsize=9.0, framealpha=0.0)

    # Note about sign reversal at the bottom.
    ax.text(
        0.5,
        -0.02,
        "Both signs reverse when the anchor is de-saturated. n = 432 rows pooled per anchor\n"
        "(54 held-out personas × 4 negative-position arms × 2 seeds). *** Holm p < 1e-6, ** < 1e-3, * < 0.05.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.6,
        color="#444444",
    )

    fig.subplots_adjust(top=0.95, bottom=0.22, left=0.13, right=0.97)
    savefig_paper(fig, "issue_530/hero_partial_rho_sign_flip", dir=str(REPO_ROOT / "figures") + "/")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — bystander resolution diagnostic.
# ---------------------------------------------------------------------------


def fig_bystander_resolution(d: dict) -> None:
    """Per-cell median bystander log P(marker), #504 vs #530, with the ceiling shaded."""
    set_paper_style("blog")

    # Pull per-cell median bystander log P(marker) at the chosen checkpoint.
    # For #504 the chosen frac is 0.33; for #530 it is 1.00 (band-stop).
    # #504 doesn't expose this number directly in comparison_504_vs_530.json's
    # diagnostics — but it does expose source emission_p = 1.0 for every cell,
    # which is the bound: at argmax = marker everywhere, on-policy log P(marker)
    # at the post-response slot sits at the ceiling band 0 to -2 nat.
    # For the figure we use the bystander argmax fraction directly.

    # Both fits report `source_emission_p` per cell — for #530 it is 0.0
    # (bystanders below ceiling); for #504 it is 1.0 (bystanders saturated).
    # But the per-cell diagnostic IS source-side. Bystander resolution is
    # captured in the issue_530 bystander_resolution.json — load it here.

    bystander_files = sorted(
        (REPO_ROOT / "eval_results/issue_530").glob("c504v3_*/bystander_resolution.json")
    )
    # Sort cells so positioned arms (near → mid_near → mid_far → far) appear
    # first, with default-only last. Each gets two seeds.
    arm_order = {
        "c504v3_near": 0,
        "c504v3_mid_near": 1,
        "c504v3_mid_far": 2,
        "c504v3_far": 3,
        "c504v3_default_only": 4,
    }
    label_map = {
        "c504v3_near": "near",
        "c504v3_mid_near": "mid-near",
        "c504v3_mid_far": "mid-far",
        "c504v3_far": "far",
        "c504v3_default_only": "default-only",
    }
    entries = []
    for f in bystander_files:
        bd = json.loads(f.read_text())
        entries.append((arm_order[bd["cell"]], bd["seed"], bd))
    entries.sort()
    cells = []
    medians_530 = []
    argmax_share_530 = []
    for _, _, bd in entries:
        gate = bd["de_saturation_gate"]
        cells.append(f"{label_map[bd['cell']]}\nseed {bd['seed']}")
        medians_530.append(gate["median_g_logp_at_post_response_slot"])
        argmax_share_530.append(gate["argmax_marker_share_across_pairs"])

    # For #504 we don't have a directly comparable bystander_resolution.json on
    # disk in the worktree — but the #504 clean-result body already documents
    # the 91-96% bystander argmax fraction at frac 0.33. Hard-code the band
    # for the figure and cite the #504 body as source. Avoid fabricating
    # per-cell numbers we don't have committed; show the range as a shaded
    # band on the left panel.

    # Argmax fractions per cell (#530) — all 0.0, all PASS gate.
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.0, 4.0))

    # Left panel: median bystander log P(marker) per cell, #530 only.
    x = np.arange(len(cells))
    bars = ax_left.bar(
        x,
        medians_530,
        color=paper_palette_role("primary"),
        edgecolor="black",
        linewidth=0.5,
    )
    # Saturation reference band: at argmax = marker everywhere, the post-R
    # slot's log P(marker) sits in [-2, 0] nats (i.e. ≥ 13.5% probability,
    # the cross-token max). Shade that band for visual contrast.
    ax_left.axhspan(
        -2.0,
        0.0,
        color="#d4574f",
        alpha=0.18,
        label="#504's saturated regime\n(argmax = marker, log P(marker) ≥ −2 nats)",
    )
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(cells, fontsize=8.2, rotation=30, ha="right")
    ax_left.set_ylabel(
        "Median bystander log P(marker) at the post-response slot (nats)", fontsize=9.8
    )
    ax_left.set_ylim(-25, 2)
    ax_left.axhline(0, color="black", linewidth=0.6)
    ax_left.legend(loc="lower center", fontsize=8.5, framealpha=0.0)
    ax_left.set_title(
        "This run (de-saturated, lr 5e-6)", fontsize=10, loc="left", fontweight="semibold"
    )

    # Right panel: argmax bystander share, #504 (from its body, range) vs #530.
    panel_x = ["#504\n(saturated anchor)", "#530\n(de-saturated anchor)"]
    panel_y = [0.94, 0.0]  # midpoint of 0.91-0.96 for #504; 0.0 for #530.
    bars_r = ax_right.bar(
        panel_x,
        panel_y,
        color=[paper_palette_role("baseline"), paper_palette_role("primary")],
        edgecolor="black",
        linewidth=0.5,
        width=0.5,
    )
    # Show #504's spread as an error bar (0.91-0.96 per the parent body).
    ax_right.errorbar(
        [0],
        [0.94],
        yerr=[[0.94 - 0.91], [0.96 - 0.94]],
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=1.0,
    )
    ax_right.set_ylabel(
        "Fraction of bystander × question pairs\nwith argmax = marker", fontsize=9.8
    )
    ax_right.set_ylim(0, 1.05)
    ax_right.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_right.set_title(
        "Saturation gate (across-cell aggregate)", fontsize=10, loc="left", fontweight="semibold"
    )
    for bar, val in zip(bars_r, panel_y, strict=True):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{val * 100:.0f}%",
            ha="center",
            fontsize=9.4,
            fontweight="semibold",
        )

    fig.tight_layout()
    savefig_paper(fig, "issue_530/bystander_resolution", dir=str(REPO_ROOT / "figures") + "/")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — raw scatter of ΔG vs each positional predictor, #530.
# ---------------------------------------------------------------------------


def _load_530_rows() -> list[dict]:
    """Reconstruct per-row data (cell, seed, persona, question, ΔG) for the
    432-row pool from the bystander_resolution.json files.

    Note: bystander_resolution.json has per-pair ΔG in raw_distributions.per_pair_delta_g.
    We pair each ΔG with its persona via the per_probe list (each persona contributes
    n_eval_questions consecutive rows). The 432-row pool excludes the default_only
    cell (the partial-Spearman regression excludes it because d_nn / shadow_angle
    are undefined for it).
    """
    rows: list[dict] = []
    bystander_files = sorted(
        (REPO_ROOT / "eval_results/issue_530").glob("c504v3_*/bystander_resolution.json")
    )
    for f in bystander_files:
        bd = json.loads(f.read_text())
        if bd["cell"] == "c504v3_default_only":
            continue
        # per_probe order matches the order of per_pair_delta_g (in 10-question chunks).
        per_pair = bd["raw_distributions"]["per_pair_delta_g"]
        idx = 0
        for probe in bd["per_probe"]:
            persona = probe["persona"]
            n = probe["n_questions_evaluated"]
            for k in range(n):
                rows.append(
                    {
                        "cell": bd["cell"],
                        "seed": bd["seed"],
                        "persona": persona,
                        "q_idx": k,
                        "delta_g": per_pair[idx + k],
                    }
                )
            idx += n
    return rows


def _persona_means(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Aggregate to per-(cell, seed, persona) mean ΔG (the regression's 432 rows)."""
    from collections import defaultdict

    sums: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        sums[(r["cell"], r["seed"], r["persona"])].append(r["delta_g"])
    out: dict = {}
    for (cell, seed, persona), vals in sums.items():
        out.setdefault(persona, {})[(cell, seed)] = float(np.mean(vals))
    return out


def fig_raw_scatter_530() -> None:
    """Raw (un-partialled) ΔG vs each predictor, #530 pooled. The raw-counterpart
    to the partialled hero — gives the reader the un-controlled view to read
    against."""
    set_paper_style("blog")

    # Load phase-0.5 predictors. Structure:
    #   per_probe[persona] = {
    #     "d_source": float,                   # scalar — same across arms
    #     "d_nearest_neg_nd": {arm_key: float}, # per-arm
    #     "shadow_angle": {arm_key: float},     # per-arm
    #   }
    # Arms in this JSON use the `c504_*` short slug, which maps to our run's
    # `c504v3_*` slugs by stripping the `v3` infix.
    gates = json.loads((REPO_ROOT / "eval_results/issue_530/phase0_5_gates.json").read_text())
    per_probe_pred = gates["per_probe"]
    # Build {cell_slug: {persona: {predictor: value}}} for the 4 positioned arms.
    arm_slug_map = {
        "c504v3_near": "c504_near",
        "c504v3_mid_near": "c504_mid_near",
        "c504v3_mid_far": "c504_mid_far",
        "c504v3_far": "c504_far",
    }
    predtab: dict[str, dict[str, dict[str, float]]] = {}
    for run_slug, arm_key in arm_slug_map.items():
        predtab[run_slug] = {}
        for persona, pdict in per_probe_pred.items():
            predtab[run_slug][persona] = {
                "d_source": pdict["d_source"],
                "d_nearest_neg_nd": pdict["d_nearest_neg_nd"][arm_key],
                "shadow_angle": pdict["shadow_angle"][arm_key],
            }

    # Load DV from bystander_resolution.json files (432 rows).
    rows = _load_530_rows()
    persona_means = _persona_means(rows)

    # Build (predictor, ΔG) arrays.
    predictors = ["d_source", "d_nearest_neg_nd", "shadow_angle"]
    labels = [
        "d_source\n(angular distance source → probe)",
        "d_nearest_neg_nd\n(angular distance nearest-negative → probe)",
        "shadow_angle\n(angle source→N vs source→probe)",
    ]
    cell_arms = ["c504v3_near", "c504v3_mid_near", "c504v3_mid_far", "c504v3_far"]
    arm_label = {
        "c504v3_near": "near",
        "c504v3_mid_near": "mid-near",
        "c504v3_mid_far": "mid-far",
        "c504v3_far": "far",
    }
    palette_blog = paper_palette_blog(4)
    arm_color = dict(zip(cell_arms, palette_blog, strict=True))

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharey=True)
    for ax, pred, lab in zip(axes, predictors, labels, strict=True):
        for cell in cell_arms:
            xs, ys = [], []
            for persona, by_cs in persona_means.items():
                for (c, s), dg in by_cs.items():
                    if c != cell:
                        continue
                    pred_val = predtab.get(c, {}).get(persona, {}).get(pred)
                    if pred_val is None:
                        continue
                    xs.append(pred_val)
                    ys.append(dg)
            if xs:
                ax.scatter(
                    xs,
                    ys,
                    s=18,
                    alpha=0.55,
                    color=arm_color[cell],
                    label=arm_label[cell],
                    edgecolor="white",
                    linewidth=0.3,
                )
        ax.set_xlabel(lab, fontsize=9.4)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    axes[0].set_ylabel("Mean held-out ΔG (trained − base, nats)", fontsize=9.6)
    axes[-1].legend(
        loc="upper right", fontsize=8.2, framealpha=0.0, title="Negative arm", title_fontsize=8.4
    )

    fig.tight_layout()
    savefig_paper(
        fig, "issue_530/raw_scatter_predictors_vs_dg", dir=str(REPO_ROOT / "figures") + "/"
    )
    plt.close(fig)


def main() -> None:
    d = _load()
    fig_hero_partial_rho(d)
    fig_bystander_resolution(d)
    fig_raw_scatter_530()
    print("Figures written to:", OUT_DIR)


if __name__ == "__main__":
    main()
