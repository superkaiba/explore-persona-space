"""Phase 6 — generate clean-result figures for task #474.

Reads ``eval_results/issue_474/analysis.json`` and the 8 per-cell matrices.
Writes hero + supporting figures to ``figures/issue_474/`` as PNG + PDF +
``.meta.json`` sidecars via ``savefig_paper``.

Figures:
    F1  trajectory_rho_by_mask         — pos vs loc, ρ across epochs, 3 masks
    F2  saturation_gauge               — fraction of cells within 0.1 nat of ceiling
    F3  h3_paired_bootstrap            — matched-epoch + matched-step loc-pos diffs
    F4  m5_S_vs_D_loc_ep1              — suppression-difficulty scatter at loc_ep1
    F5  kl_drift_secondary             — KL @ post-response slot vs D (loc_ep1)

Run with ``uv run python scripts/i474_phase6_figures.py``.
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

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "eval_results" / "issue_474" / "analysis.json"
D_MATRIX_PATH = ROOT / "eval_results" / "issue_406" / "divergence" / "D_matrix.json"
CROSS_EVAL = ROOT / "eval_results" / "issue_474" / "cross_eval"
OUT_DIR = ROOT / "figures"


def _read_analysis() -> dict:
    return json.loads(ANALYSIS.read_text())


# ---------------------------------------------------------------------------
# F1 — Mask-A/B/C ρ trajectory across epochs, pos vs loc
# ---------------------------------------------------------------------------


def figure_trajectory_rho_by_mask(d: dict) -> None:
    """2-panel line plot: arms × epoch, three masks per panel."""
    epochs = [1, 2, 3, 5]
    arms = ["pos", "loc"]
    masks = [
        ("mask_a_all", "All 240 pairs", paper_palette_role("primary")),
        (
            "mask_b_exclude_stylized_source",
            "Excluding stylized source",
            paper_palette_role("baseline"),
        ),
        (
            "mask_c_exclude_stylized_either",
            "Excluding stylized source OR target",
            paper_palette_role("control"),
        ),
    ]
    arm_label = {"pos": "Positives only", "loc": "Localized (+ broad negatives)"}

    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)

    for ax, arm in zip(axes, arms):
        for mask_key, label, color in masks:
            ys = []
            lo = []
            hi = []
            for ep in epochs:
                cell = d["cells"][f"{arm}_ep{ep}"]
                r = cell["delta_g_three_mask_rho"][mask_key]["length_partial_spearman"]
                ys.append(r["rho_inline"])
                lo.append(r["rho_inline"] - r["bootstrap_ci_2_5"])
                hi.append(r["bootstrap_ci_97_5"] - r["rho_inline"])
            ax.errorbar(
                epochs,
                ys,
                yerr=[lo, hi],
                fmt="o-",
                color=color,
                label=label,
                capsize=3,
                linewidth=1.8,
                markersize=6,
            )
        ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
        ax.axhline(
            -0.27,
            color="#aa6633",
            linewidth=0.8,
            linestyle=":",
            label="task 462 ep-1 reference (-0.27)",
        )
        ax.set_title(arm_label[arm], fontweight="semibold", fontsize=11)
        ax.set_xlabel("Training epoch")
        ax.set_xticks(epochs)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    axes[0].set_ylabel("Length-partial Spearman ρ(D, ΔG)")
    axes[0].legend(loc="lower left", fontsize=8.5, frameon=False)
    fig.suptitle(
        "Localization restores the divergence→transfer correlation, but the "
        "recovery is stylized-persona-driven",
        fontsize=12.5,
        fontweight="semibold",
        x=0.07,
        ha="left",
        y=1.01,
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_474/trajectory_rho_by_mask", dir=str(OUT_DIR))
    plt.close(fig)


# ---------------------------------------------------------------------------
# F2 — Saturation gauge bar chart
# ---------------------------------------------------------------------------


def figure_saturation_gauge(d: dict) -> None:
    epochs = [1, 2, 3, 5]
    arms = ["pos", "loc"]
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    bar_w = 0.36
    x = np.arange(len(epochs))
    colors = {
        "pos": paper_palette_role("baseline"),
        "loc": paper_palette_role("primary"),
    }
    for i, arm in enumerate(arms):
        ys = [
            d["cells"][f"{arm}_ep{ep}"]["saturation_gauge_delta_g"]["saturation_fraction"] * 100
            for ep in epochs
        ]
        offsets = x + (i - 0.5) * bar_w
        ax.bar(
            offsets,
            ys,
            bar_w,
            label="Positives only" if arm == "pos" else "Localized (+ broad negatives)",
            color=colors[arm],
            edgecolor="white",
            linewidth=0.8,
        )
        for xv, yv in zip(offsets, ys):
            ax.annotate(f"{yv:.1f}%", (xv, yv), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"ep{ep}" for ep in epochs])
    ax.set_ylabel("% of 240 cells within 0.1 nat of ceiling")
    ax.set_xlabel("Training epoch")
    ax.set_title(
        "Localization keeps the marker off the probability ceiling",
        loc="left",
        fontsize=11.5,
        fontweight="semibold",
    )
    ax.set_ylim(0, max(8, ax.get_ylim()[1] * 1.2))
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    savefig_paper(fig, "issue_474/saturation_gauge", dir=str(OUT_DIR))
    plt.close(fig)


# ---------------------------------------------------------------------------
# F3 — paired bootstrap H3 (matched-epoch + matched-step)
# ---------------------------------------------------------------------------


def figure_h3_paired_bootstrap(d: dict) -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))

    panels: list[tuple[str, float, float, float]] = []
    h3e = d["h3_matched_epoch_paired_bootstrap"]
    for ep in [1, 2, 3, 5]:
        cell = h3e[f"ep{ep}"]
        panels.append(
            (
                f"matched ep{ep}",
                cell["diff_loc_minus_pos_mean"],
                cell["diff_ci_2_5"],
                cell["diff_ci_97_5"],
            )
        )
    h3s = d["h3_matched_step_paired_bootstrap"]
    panels.append(
        (
            "matched-step\n(loc ep1 vs pos ep2)",
            h3s["diff_loc_minus_pos_mean"],
            h3s["diff_ci_2_5"],
            h3s["diff_ci_97_5"],
        )
    )

    y_pos = np.arange(len(panels))
    diffs = [p[1] for p in panels]
    lo = [p[1] - p[2] for p in panels]
    hi = [p[3] - p[1] for p in panels]
    colors = [
        paper_palette_role("primary") if (p[2] < 0 and p[3] < 0) else paper_palette_role("control")
        for p in panels
    ]

    ax.errorbar(
        diffs, y_pos, xerr=[lo, hi], fmt="none", ecolor="#444", capsize=4, linewidth=1.2, zorder=2
    )
    ax.scatter(diffs, y_pos, s=90, color=colors, zorder=3, edgecolor="white", linewidth=1.0)
    ax.axvline(0, color="#888", linewidth=0.8, linestyle="--")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([p[0] for p in panels])
    ax.invert_yaxis()
    ax.set_xlabel("Δρ = ρ(localized) − ρ(positives only)  (negative = localized predicts better)")
    ax.set_title(
        "Localization tightens the correlation at every comparable budget",
        loc="left",
        fontsize=11.5,
        fontweight="semibold",
    )
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)
    for i, p in enumerate(panels):
        excl = "CI excludes 0" if (p[2] < 0 and p[3] < 0) else "CI includes 0"
        ax.text(
            0.02,
            y_pos[i] - 0.30,
            excl,
            fontsize=8,
            color="#555",
            transform=ax.get_yaxis_transform(),
            va="top",
        )
    fig.tight_layout()
    savefig_paper(fig, "issue_474/h3_paired_bootstrap", dir=str(OUT_DIR))
    plt.close(fig)


# ---------------------------------------------------------------------------
# F4 — M5 partial: per-cell S vs D scatter at loc_ep1
# ---------------------------------------------------------------------------


def figure_m5_scatter(d: dict) -> None:
    """Per-cell ΔG vs D, colored by per-cell suppression difficulty S."""
    matrix = json.loads((CROSS_EVAL / "loc_ep1" / "G_logprob_matrix.json").read_text())
    G = matrix["G"]
    conds = matrix["conditions"]

    D_payload = json.loads(D_MATRIX_PATH.read_text())
    D = D_payload["JS"]  # JS divergence as the predictor (matches #406)

    # S: per-cell mean negative-row loss at the final epoch step,
    # keyed by (source, bystander) — exactly what the analyzer's M5 partials.
    ep = 1
    train_diag_dir = ROOT / "eval_results" / "issue_474" / "train_diag"
    S: dict[tuple[str, str], float] = {}
    for cond in conds:
        f = train_diag_dir / f"suppression_difficulty_loc_{cond}_ep{ep}.json"
        if not f.exists():
            continue
        payload = json.loads(f.read_text())
        for key, val in payload.get("per_bystander_mean_neg_loss", {}).items():
            try:
                src, bys = key.split("__", 1)
            except ValueError:
                continue
            S[(src, bys)] = val

    xs, ys, ss, names = [], [], [], []
    for ti in conds:
        for tj in conds:
            if ti == tj:
                continue
            d_val = D[ti][tj]
            g_val = G[ti][tj]["delta_g"]
            s_val = S.get((ti, tj), np.nan)
            xs.append(d_val)
            ys.append(g_val)
            ss.append(s_val)
            names.append(f"{ti}→{tj}")

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
    finite = np.array([v for v in ss if np.isfinite(v)])
    vmax = float(np.percentile(finite, 95)) if len(finite) else 1.0
    sc = ax.scatter(
        xs,
        ys,
        c=ss,
        s=22,
        alpha=0.85,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.3,
        vmin=0,
        vmax=vmax,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Per-cell suppression difficulty S\n(higher = harder to suppress)", fontsize=9)
    ax.set_xlabel("Base-model output divergence D(source, target) [JS, nat]")
    ax.set_ylabel("On-policy marker transfer ΔG (nat)")
    ax.set_title(
        "Even after partialling out per-cell suppression difficulty, "
        "divergence still predicts transfer",
        loc="left",
        fontsize=11,
        fontweight="semibold",
    )
    rho_base = d["cells"]["loc_ep1"]["m5_suppression_difficulty_partial"][
        "rho_baseline_lengthonly_partial"
    ]["rho_pingouin"]
    rho_part = d["cells"]["loc_ep1"]["m5_suppression_difficulty_partial"]["rho_partial_out_S"][
        "rho_pingouin"
    ]
    ax.annotate(
        f"baseline (length-partial)   ρ = {rho_base:+.3f}\n"
        f"partialling out S also        ρ = {rho_part:+.3f}  (CI excludes 0)",
        xy=(0.02, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=9,
        color="#222",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ccc"),
    )
    savefig_paper(fig, "issue_474/m5_S_vs_D_loc_ep1", dir=str(OUT_DIR))
    plt.close(fig)


# ---------------------------------------------------------------------------
# F5 — Raw scatter D vs ΔG at loc_ep1 vs pos_ep1 (side by side)
# ---------------------------------------------------------------------------


def figure_raw_d_vs_dg(d: dict) -> None:
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)

    D_payload = json.loads(D_MATRIX_PATH.read_text())
    D = D_payload["JS"]

    stylized = {"A3", "A4", "A5"}

    for ax, arm, title in zip(
        axes,
        ["pos", "loc"],
        ["Positives only (ep1)", "Localized (+ broad negatives, ep1)"],
    ):
        matrix = json.loads((CROSS_EVAL / f"{arm}_ep1" / "G_logprob_matrix.json").read_text())
        G = matrix["G"]
        xs_styl, ys_styl, xs_oth, ys_oth = [], [], [], []
        for ti, row in G.items():
            for tj, cell in row.items():
                if ti == tj:
                    continue  # exclude diagonal (always ceiling)
                d_val = D[ti][tj]
                g_val = cell["delta_g"]
                if ti in stylized or tj in stylized:
                    xs_styl.append(d_val)
                    ys_styl.append(g_val)
                else:
                    xs_oth.append(d_val)
                    ys_oth.append(g_val)

        ax.scatter(
            xs_oth,
            ys_oth,
            s=22,
            color=paper_palette_role("baseline"),
            alpha=0.75,
            label="Non-stylized cells",
            edgecolor="white",
            linewidth=0.3,
        )
        ax.scatter(
            xs_styl,
            ys_styl,
            s=26,
            color=paper_palette_role("accent"),
            alpha=0.9,
            label="Stylized (pirate/comedian/villain) cell",
            edgecolor="white",
            linewidth=0.3,
        )

        rho_a = d["cells"][f"{arm}_ep1"]["delta_g_three_mask_rho"]["mask_a_all"][
            "length_partial_spearman"
        ]["rho_inline"]
        rho_c = d["cells"][f"{arm}_ep1"]["delta_g_three_mask_rho"][
            "mask_c_exclude_stylized_either"
        ]["length_partial_spearman"]["rho_inline"]
        ax.annotate(
            f"All cells:    ρ = {rho_a:+.2f}\nExc. stylized: ρ = {rho_c:+.2f}",
            xy=(0.98, 0.97),
            xycoords="axes fraction",
            va="top",
            ha="right",
            fontsize=9,
            color="#222",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ccc"),
        )
        ax.set_title(title, fontweight="semibold", fontsize=11)
        ax.set_xlabel("Base-model divergence D (JS, nat)")
        ax.grid(linewidth=0.4, alpha=0.4)
    axes[0].set_ylabel("On-policy marker transfer ΔG (nat)")
    axes[0].legend(loc="lower left", fontsize=8.5, frameon=False)
    fig.suptitle(
        "The recovered correlation lives largely in the stylized-persona cells",
        fontsize=12.5,
        fontweight="semibold",
        x=0.07,
        ha="left",
        y=1.01,
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_474/raw_d_vs_dg_ep1", dir=str(OUT_DIR))
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    d = _read_analysis()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_trajectory_rho_by_mask(d)
    figure_saturation_gauge(d)
    figure_h3_paired_bootstrap(d)
    figure_m5_scatter(d)
    figure_raw_d_vs_dg(d)
    print("Wrote 5 figures to figures/issue_474/")


if __name__ == "__main__":
    main()
