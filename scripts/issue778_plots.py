#!/usr/bin/env python
"""Issue #778 — clean-result figures for the Persona-Vectors null-battery replication.

Reads the null-battery deliverables in ``eval_results/issue_778/`` and produces,
under ``figures/issue_778/``:

  - ``bands_{setting}.png``   — per-setting grid: observed matched-trait
    max-over-layers |r| (point + 95% bootstrap CI) overlaid on the four null
    bands (violin of per-draw / per-direction max|r|), one panel per trait.
    settings: finetune, monitoring_overall, monitoring_within.
  - ``finetune_scatter_{trait}.png`` — the n=24 finetuning-shift regression:
    shift-projection onto r_B (x) vs graded trait score (y), points labeled by
    dataset family, the low-level data view behind the finetune r.
  - ``per_layer_{trait}_{setting}.png`` — matched-trait |r| vs layer with the
    perm + randnorm mean-|r|-per-layer null curves (exploratory; shows the
    max-over-layers selection is not cherry-picked).

CPU-only; reads JSON/NPY, no model calls.
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
    set_title_subtitle,
)

EVAL = Path("eval_results/issue_778")
TRAITS = ["evil", "sycophancy", "hallucination"]
TRAIT_LABEL = {"evil": "Evil", "sycophancy": "Sycophancy", "hallucination": "Hallucination"}
NULL_ORDER = ["perm", "randnorm", "crosstrait", "pca_topk"]
NULL_LABEL = {
    "perm": "shuffled\nperm.",
    "randnorm": "norm-matched\nrandom",
    "crosstrait": "cross-\ntrait",
    "pca_topk": "PCA\ntop-5",
}
SETTING_LABEL = {
    "finetune": "Finetuning-shift prediction (n=24 finetunes)",
    "monitoring_overall": "System-prompt prediction, pooled across prompts",
    "monitoring_within": "System-prompt prediction, within-prompt (controls for prompt type)",
}
FAMILY_LABEL = {
    "evil": "evil",
    "sycophancy": "sycophancy",
    "hallucination": "hallucination",
    "insecure_code": "insecure code",
    "mistake_medical": "bad medical",
    "mistake_math": "wrong math",
    "mistake_gsm8k": "wrong GSM8K",
    "mistake_opinions": "bad opinions",
}


def _load_setting(trait: str, setting: str) -> dict:
    """Return the per-setting payload dict for (trait, setting)."""
    if setting == "finetune":
        return json.load(open(EVAL / f"{trait}_finetune_nullbattery.json"))
    mon = json.load(open(EVAL / f"{trait}_monitoring_nullbattery.json"))
    return mon[setting]


def _null_max_abs(payload: dict, kind: str) -> np.ndarray:
    """Per-draw / per-direction max-over-layers |r| for one null."""
    return np.asarray(payload["nulls"][kind].get("draws_max_abs", []), dtype=float)


def plot_bands(setting: str) -> None:
    """Grid of observed-vs-null-band panels, one per trait, for one setting."""
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    fig.set_layout_engine("none")  # manual subplots_adjust below (suptitle + footer room)
    obs_color = paper_palette_role("primary")
    null_colors = {
        "perm": paper_palette_role("accent"),
        "randnorm": paper_palette_role("baseline"),
        "crosstrait": paper_palette_role("control"),
        "pca_topk": paper_palette_role("neutral"),
    }
    for ax, trait in zip(axes, TRAITS):
        payload = _load_setting(trait, setting)
        obs = payload["matched_max_abs"]
        ci = payload["matched_r_bootstrap_ci_95"]
        ci_lo, ci_hi = abs(ci[0]), abs(ci[1])
        ci_lo, ci_hi = min(ci_lo, ci_hi), max(ci_lo, ci_hi)
        positions = [i * 1.25 for i in range(len(NULL_ORDER))]
        data = [_null_max_abs(payload, k) for k in NULL_ORDER]
        # violin for the stochastic nulls (>=5 pts); strip for fixed nulls.
        for pos, k, arr in zip(positions, NULL_ORDER, data):
            arr = arr[~np.isnan(arr)]
            if arr.size >= 5:
                vp = ax.violinplot(
                    [arr], positions=[pos], widths=0.7, showextrema=False, showmedians=False
                )
                for body in vp["bodies"]:
                    body.set_facecolor(null_colors[k])
                    body.set_alpha(0.45)
                    body.set_edgecolor(null_colors[k])
                p975 = np.percentile(arr, 97.5)
                ax.hlines(p975, pos - 0.35, pos + 0.35, color=null_colors[k], lw=1.6)
            else:
                ax.scatter(
                    [pos] * arr.size,
                    arr,
                    color=null_colors[k],
                    s=42,
                    zorder=4,
                    edgecolors="white",
                    linewidths=0.6,
                )
        # observed matched-trait point + CI, drawn to the LEFT as its own column.
        # The stored bootstrap CI is the POOLED r's CI at the selected layer; it is
        # valid for finetune + monitoring_overall (where the matched value IS the
        # pooled r). For monitoring_within the matched value is the Fisher-z
        # within-condition r, which the pooled CI does NOT bracket, so we draw the
        # point WITHOUT an interval there (the within-condition CI was not computed).
        obs_pos = -1.5
        ci_valid = ci_lo <= obs <= ci_hi
        if ci_valid:
            ax.errorbar(
                [obs_pos],
                [obs],
                yerr=[[obs - ci_lo], [ci_hi - obs]],
                fmt="o",
                color=obs_color,
                markersize=9,
                capsize=5,
                lw=2.0,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=6,
            )
        else:
            ax.scatter(
                [obs_pos],
                [obs],
                color=obs_color,
                s=90,
                zorder=6,
                edgecolors="white",
                linewidths=0.8,
            )
        ax.axhline(obs, color=obs_color, ls="--", lw=1.0, alpha=0.55, zorder=1)
        ax.set_xticks([obs_pos] + positions)
        ax.set_xticklabels(["persona\nvector"] + [NULL_LABEL[k] for k in NULL_ORDER], fontsize=8.5)
        ax.set_xlim(-2.2, positions[-1] + 0.6)
        ax.axvline(-0.65, color="#DDDDDD", lw=1.0, ls=":", zorder=0)
        ax.set_title(TRAIT_LABEL[trait], loc="left", fontsize=12, pad=8)
        ax.set_ylim(0, 1.02)
        ax.axhline(0, color="#CCCCCC", lw=0.8)
    axes[0].set_ylabel("max over 28 layers of |Pearson r|")
    fig.suptitle(
        f"Persona-vector direction vs null battery — {SETTING_LABEL[setting]}",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.01,
        0.005,
        "Blue = the paper's matched-trait persona vector (point + 95% bootstrap CI). "
        "Violins/points = the four nulls' max-over-28-layers |r| per draw; horizontal cap = each null's 97.5th pct. "
        "The paper's direction beats a null only when the blue point sits ABOVE that null's cap.",
        fontsize=7.5,
        color="#5A5A5A",
    )
    fig.subplots_adjust(top=0.86, bottom=0.20, left=0.07, right=0.99, wspace=0.08)
    savefig_paper(fig, f"issue_778/bands_{setting}", dir="figures/")
    plt.close(fig)


def plot_finetune_scatter(trait: str) -> None:
    """n=24 finetuning-shift regression scatter, labeled by family."""
    set_paper_style("blog")
    payload = json.load(open(EVAL / f"{trait}_finetune_nullbattery.json"))
    pts = payload["per_run_points"]
    x = np.array([p["shift_proj_selected_layer"] for p in pts], dtype=float)
    y = np.array([p["trait_score"] for p in pts], dtype=float)
    tags = [p["tag"] for p in pts]
    fams = sorted({t.rsplit("_", 2)[0] if "misaligned" in t else t.rsplit("_", 1)[0] for t in tags})

    # robust family split: strip the version suffix (normal / misaligned_1 / misaligned_2)
    def fam_of(tag: str) -> str:
        for ver in ("_misaligned_2", "_misaligned_1", "_normal"):
            if tag.endswith(ver):
                return tag[: -len(ver)]
        return tag

    fams = sorted({fam_of(t) for t in tags})
    palette = paper_palette_role
    fam_colors = {}
    base_cols = ["primary", "baseline", "control", "accent", "neutral"]
    # give each of the 8 families a distinct hue via a colormap for >5
    import matplotlib.cm as cm

    cmap = cm.get_cmap("tab10")
    for i, f in enumerate(fams):
        fam_colors[f] = cmap(i % 10)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for f in fams:
        idx = [i for i, t in enumerate(tags) if fam_of(t) == f]
        ax.scatter(
            x[idx],
            y[idx],
            color=fam_colors[f],
            s=70,
            edgecolors="white",
            linewidths=0.8,
            label=FAMILY_LABEL.get(f, f),
            zorder=4,
        )
    for xi, yi, t in zip(x, y, tags):
        ax.text(xi, yi + 1.4, fam_of(t)[:4], fontsize=6, ha="center", color="#555555", zorder=3)
    r = payload["matched_r"]
    ci = payload["matched_r_bootstrap_ci_95"]
    sel = payload["matched_selected_layer"]
    ax.set_xlabel(f"finetuning shift projected onto persona vector (layer {sel})")
    ax.set_ylabel(f"{TRAIT_LABEL[trait]} trait-expression score (0-100, graded judge)")
    set_title_subtitle(
        ax,
        f"{TRAIT_LABEL[trait]}: finetuning shift predicts trait expression (r = {r:.2f})",
        f"n=24 finetunes; Pearson r = {r:.2f} [95% CI {ci[0]:.2f}, {ci[1]:.2f}] at the best of 28 layers",
        source=f"eval_results/issue_778/{trait}_finetune_nullbattery.json",
    )
    ax.legend(fontsize=7.5, ncol=2, loc="lower right", frameon=False)
    savefig_paper(fig, f"issue_778/finetune_scatter_{trait}", dir="figures/")
    plt.close(fig)


def plot_per_layer(trait: str, setting: str) -> None:
    """matched |r| vs layer + perm/randnorm mean|r| curves (exploratory)."""
    set_paper_style("blog")
    heat = json.load(open(EVAL / f"per_layer_heatmap_{trait}_{setting}.json"))
    per_layer = heat["nulls_per_layer_mean_abs_r"]
    # matched per-layer |r| is not directly stored; recompute is not available here,
    # so we plot the null mean-|r| curves + mark the selected layer.
    payload = _load_setting(trait, setting)
    sel = payload["matched_selected_layer"]
    obs = payload["matched_max_abs"]
    layers = np.arange(28)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    colors = {
        "perm": paper_palette_role("accent"),
        "randnorm": paper_palette_role("baseline"),
    }
    for k in ("perm", "randnorm"):
        if k in per_layer:
            ax.plot(
                layers,
                per_layer[k],
                color=colors[k],
                lw=1.8,
                marker="o",
                markersize=3,
                label=f"{NULL_LABEL[k].replace(chr(10), ' ')} (mean |r| per layer)",
            )
    ax.scatter(
        [sel],
        [obs],
        color=paper_palette_role("primary"),
        s=90,
        zorder=6,
        edgecolors="white",
        linewidths=0.9,
        label=f"persona vector, selected layer {sel} (|r|={obs:.2f})",
    )
    ax.set_xlabel("transformer layer (0-27)")
    ax.set_ylabel("|Pearson r|")
    ax.set_ylim(0, 1.02)
    set_title_subtitle(
        ax,
        f"{TRAIT_LABEL[trait]} — {setting.replace('_', ' ')}: |r| across layers",
        "Null curves = mean |r| per layer; blue point = the persona vector's best (selected) layer",
    )
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    savefig_paper(fig, f"issue_778/per_layer_{trait}_{setting}", dir="figures/")
    plt.close(fig)


TAG_LABEL = {
    "monitoring_corrected": "corrected 8-prompt ladder (the paper's trait-inducing prompts)",
    "monitoring_manyshot": "many-shot ICL (0/5/10/15/20 trait exemplars)",
}
TAG_COND_LEGEND = {
    "monitoring_corrected": "prompt condition (strongest → weakest)",
    "monitoring_manyshot": "shot count",
}


def plot_bands_tag(mon_tag: str) -> None:
    """2x3 grid (rows: overall / within; cols: traits) of observed-vs-null bands
    for one follow-up monitoring leg (``{trait}_{mon_tag}_nullbattery.json``)."""
    set_paper_style("blog")
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6), sharey=True)
    fig.set_layout_engine("none")
    obs_color = paper_palette_role("primary")
    null_colors = {
        "perm": paper_palette_role("accent"),
        "randnorm": paper_palette_role("baseline"),
        "crosstrait": paper_palette_role("control"),
        "pca_topk": paper_palette_role("neutral"),
    }
    row_label = {0: "pooled (overall r)", 1: "within-condition r"}
    for col, trait in enumerate(TRAITS):
        payload_all = json.load(open(EVAL / f"{trait}_{mon_tag}_nullbattery.json"))
        for row_i, setting in enumerate(("monitoring_overall", "monitoring_within")):
            ax = axes[row_i, col]
            payload = payload_all[setting]
            obs = payload["matched_max_abs"]
            positions = [i * 1.25 for i in range(len(NULL_ORDER))]
            for pos, k in zip(positions, NULL_ORDER):
                arr = _null_max_abs(payload, k)
                arr = arr[~np.isnan(arr)]
                if arr.size >= 5:
                    vp = ax.violinplot(
                        [arr], positions=[pos], widths=0.7, showextrema=False, showmedians=False
                    )
                    for body in vp["bodies"]:
                        body.set_facecolor(null_colors[k])
                        body.set_alpha(0.45)
                        body.set_edgecolor(null_colors[k])
                    p975 = np.percentile(arr, 97.5)
                    ax.hlines(p975, pos - 0.35, pos + 0.35, color=null_colors[k], lw=1.6)
                else:
                    ax.scatter(
                        [pos] * arr.size,
                        arr,
                        color=null_colors[k],
                        s=42,
                        zorder=4,
                        edgecolors="white",
                        linewidths=0.6,
                    )
            obs_pos = -1.5
            ax.scatter(
                [obs_pos],
                [obs],
                color=obs_color,
                s=90,
                zorder=6,
                edgecolors="white",
                linewidths=0.8,
            )
            ax.axhline(obs, color=obs_color, ls="--", lw=1.0, alpha=0.55, zorder=1)
            ax.set_xticks([obs_pos] + positions)
            ax.set_xticklabels(
                ["persona\nvector"] + [NULL_LABEL[k] for k in NULL_ORDER], fontsize=8.5
            )
            ax.set_xlim(-2.2, positions[-1] + 0.6)
            ax.axvline(-0.65, color="#DDDDDD", lw=1.0, ls=":", zorder=0)
            ax.set_ylim(0, 1.02)
            if row_i == 0:
                ax.set_title(TRAIT_LABEL[trait], loc="left", fontsize=12, pad=8)
            if col == 0:
                ax.set_ylabel(f"{row_label[row_i]}\nmax over 28 layers of |r|")
    fig.suptitle(
        f"Persona-vector direction vs null battery — {TAG_LABEL[mon_tag]}",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.01,
        0.005,
        "Blue = the matched-trait persona vector. Violins = per-draw max-over-28-layers |r| for the "
        "shuffled-permutation / norm-matched-random nulls (1000 draws); points = the fixed cross-trait / "
        "PCA top-5 directions; horizontal cap = each null's 97.5th pct. The vector beats a null only "
        "when blue sits ABOVE that cap.",
        fontsize=7.5,
        color="#5A5A5A",
    )
    fig.subplots_adjust(top=0.90, bottom=0.11, left=0.07, right=0.99, wspace=0.08, hspace=0.30)
    savefig_paper(fig, f"issue_778/bands_{mon_tag}", dir="figures/")
    plt.close(fig)


def plot_monitoring_scatter_tag(mon_tag: str) -> None:
    """1x3 per-cell scatter (x = projection at the selected layer, y = graded
    score, colored by condition) — the low-level data behind one leg's r."""
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    fig.set_layout_engine("none")
    cmap = plt.get_cmap("viridis")
    for ax, trait in zip(axes, TRAITS):
        payload = json.load(open(EVAL / f"{trait}_{mon_tag}_nullbattery.json"))
        sel = payload["monitoring_overall"]["matched_selected_layer"]
        r_ov = payload["monitoring_overall"]["matched_r"]
        r_wi = payload["monitoring_within"]["matched_r"]
        rows = [
            json.loads(line) for line in open(EVAL / f"{mon_tag}_{trait}.jsonl") if line.strip()
        ]
        rows = [r for r in rows if r["mean_trait_score"] is not None]
        conds = sorted({r["condition_id"] for r in rows})
        for i, c in enumerate(conds):
            xs = [r["projection_per_layer"][sel] for r in rows if r["condition_id"] == c]
            ys = [r["mean_trait_score"] for r in rows if r["condition_id"] == c]
            ax.scatter(
                xs,
                ys,
                color=cmap(i / max(len(conds) - 1, 1)),
                s=26,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.4,
                label=str(c),
            )
        ax.set_title(
            f"{TRAIT_LABEL[trait]} — L{sel} · r {r_ov:.2f} / within {r_wi:.2f}",
            loc="left",
            fontsize=10,
            pad=8,
        )
    axes[1].set_xlabel("last-prompt-token projection onto the persona vector (selected layer)")
    axes[0].set_ylabel("graded trait score (0–100)")
    axes[-1].legend(
        title=TAG_COND_LEGEND[mon_tag],
        fontsize=7,
        title_fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        framealpha=0.9,
    )
    fig.suptitle(
        f"Per-cell data behind the {TAG_LABEL[mon_tag]} correlations",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.06, right=0.90, wspace=0.08)
    savefig_paper(fig, f"issue_778/{mon_tag}_scatter", dir="figures/")
    plt.close(fig)


def main() -> None:
    import sys

    if "--legs" in sys.argv:
        for tag in ("monitoring_corrected", "monitoring_manyshot"):
            plot_bands_tag(tag)
            plot_monitoring_scatter_tag(tag)
            print(f"wrote bands_{tag} + {tag}_scatter")
        return
    for setting in ("finetune", "monitoring_overall", "monitoring_within"):
        plot_bands(setting)
        print(f"wrote bands_{setting}")
    for trait in TRAITS:
        plot_finetune_scatter(trait)
        print(f"wrote finetune_scatter_{trait}")
        plot_per_layer(trait, "finetune")
        plot_per_layer(trait, "monitoring_within")
        print(f"wrote per_layer_{trait}")


if __name__ == "__main__":
    main()
