"""Analyzer figures for issue #1738 (multi-turn prefix-arm map at 100k).

Regenerates the two driver-rendered figures via savefig_paper (sidecars with
points + text) under new names, and adds the low-level per-unit companions:
per-context error scatter, taxonomy contrast forest, mapping-baselines
dissociation. Also writes the per-cell CSV behind the scatter.

Run from the issue-1738 worktree root:
    uv run python scripts/issue1738_analyzer_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # #847: shared-VM thread caps bind BEFORE numpy/matplotlib import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

E = Path("eval_results/issue_1738")
EB = E / "bare_query"  # bare-query follow-up round outputs (plan §4.3)
FIGDIR = "figures/"

ARM_COLORS = {}  # filled in main() from paper_palette_blog
ARM_LABELS = {"prefix": "prefix arm", "context": "context arm", "bare": "bare-query arm"}
FITTER_ORDER = ["ridge", "mlp_w8192", "mlp_w32768", "residual_skip", "krr_nystrom"]
FITTER_LABELS = {
    "ridge": "ridge",
    "mlp_w8192": "MLP (width 8,192)",
    "mlp_w32768": "MLP (width 32,768)",
    "residual_skip": "residual + skip",
    "krr_nystrom": "kernel ridge (Nystrom)",
}
DEPTH_ORDER = ["2-2", "3-4", ">=5"]
DEPTH_LABELS = {"2-2": "2 turns", "3-4": "3-4 turns", ">=5": "5+ turns"}

CONTRAST_LABELS = {
    "language=en": "English",
    "topic=factual_qa": "factual Q&A",
    "topic=creative_writing": "creative writing",
    "topic=coding": "coding",
    "topic=advice_howto": "advice / how-to",
    "topic=chitchat_social": "social chitchat",
    "topic=translation": "translation",
    "topic=math": "math",
    "topic=summarization_extraction": "summarization",
    "topic=harmful_or_unsafe_request": "harmful / unsafe request",
    "topic=roleplay_persona": "roleplay / persona",
    "topic=nsfw": "explicit content",
    "topic=other": "other topics",
    "refusal_adjacent=yes": "refusal-adjacent request",
    "answer_is_refusal=yes": "answer is a refusal",
    "format=code": "code-formatted answer",
    "format=list": "list-formatted answer",
    "format=prose": "prose answer",
    "depth=2-2": "2-turn conversations",
    "depth=3-4": "3-4-turn conversations",
    "depth=>=5": "5+-turn conversations",
    "corpus=wildchat": "WildChat corpus",
}


def _load(p: str):
    return json.load(open(E / p))


def fig_hero(fits: dict) -> None:
    cells = fits["cells"]
    layers = [14, 19, 26]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), sharey=True)
    width = 0.38
    x = np.arange(len(FITTER_ORDER))
    for ax, layer in zip(axes, layers):
        for j, arm in enumerate(["prefix", "context"]):
            pts, lo, hi = [], [], []
            for f in FITTER_ORDER:
                c = cells[f"{arm}_L{layer}_{f}"]
                ci = c["holdout_bootstrap_ci"]["r2"]
                pts.append(c["holdout_r2"])
                lo.append(c["holdout_r2"] - ci["lo"])
                hi.append(ci["hi"] - c["holdout_r2"])
            ax.bar(
                x + (j - 0.5) * width,
                pts,
                width,
                yerr=[lo, hi],
                color=ARM_COLORS[arm],
                label=ARM_LABELS[arm] if layer == 14 else None,
                error_kw={"elinewidth": 1.0, "ecolor": "#333333"},
            )
        ax.axhspan(0.05, 0.11, color="#999999", alpha=0.25, zorder=0)
        ax.set_title(f"layer {layer}", loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels([FITTER_LABELS[f] for f in FITTER_ORDER], rotation=20, ha="right")
        ax.axhline(0.0, color="#666666", lw=0.8)
    axes[0].set_ylabel("held-out R² (9,941 contexts)")
    band_handle = plt.Rectangle((0, 0), 1, 1, color="#999999", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(band_handle)
    labels.append("single-turn prefix reference (0.05-0.11)")
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    savefig_paper(fig, "issue_1738/hero_arm_r2_by_layer", dir=FIGDIR)
    plt.close(fig)


def fig_depth(depth: dict) -> None:
    arms = depth["arms"]
    fig, ax = plt.subplots()
    x = np.arange(len(DEPTH_ORDER))
    for arm_key, arm in [("prefix_L19_ridge", "prefix"), ("context_L19_ridge", "context")]:
        vals = [arms[arm_key][d]["r2"] for d in DEPTH_ORDER]
        ax.plot(
            x,
            vals,
            marker="o",
            markersize=7,
            color=ARM_COLORS[arm],
            label=ARM_LABELS[arm],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([DEPTH_LABELS[d] for d in DEPTH_ORDER])
    ax.set_xlabel("conversation depth (user turns)")
    ax.set_ylabel("held-out R² (layer 19, ridge)")
    ax.set_ylim(0.0, 0.78)
    ax.legend()
    savefig_paper(fig, "issue_1738/depth_stratified_r2_v2", dir=FIGDIR)
    plt.close(fig)


def fig_scatter() -> None:
    zp = np.load(E / "percontext/prefix_L19_ridge.npz")
    zc = np.load(E / "percontext/context_L19_ridge.npz")
    assert (zp["ci"] == zc["ci"]).all()
    xp, yp = zc["nerr"].astype(float), zp["nerr"].astype(float)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(xp, yp, s=4, alpha=0.12, color="#5a5a5a", edgecolors="none", rasterized=True)
    lims = (0.02, 3.2)
    ax.plot(lims, lims, ls="--", lw=1.0, color="#b04a4a")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("per-context normalized error, context arm (layer 19, ridge)")
    ax.set_ylabel("per-context normalized error, prefix arm (layer 19, ridge)")
    savefig_paper(fig, "issue_1738/percontext_nerr_scatter", dir=FIGDIR, embed_data=False)
    plt.close(fig)
    # per-cell CSV behind the scatter (+ judge labels where present)
    labels = _load("judge_labels/labels.json")["labels"]
    out = E / "percontext_summary_L19_ridge.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["ci", "nerr_prefix_L19_ridge", "nerr_context_L19_ridge", "language", "topic", "format"]
        )
        for i, c in enumerate(zp["ci"].tolist()):
            lab = labels.get(str(c), {})
            w.writerow(
                [
                    c,
                    f"{yp[i]:.6f}",
                    f"{xp[i]:.6f}",
                    lab.get("language", ""),
                    lab.get("topic", ""),
                    lab.get("format", ""),
                ]
            )
    print(f"wrote {out} frac_prefix_worse={float((yp > xp).mean()):.4f}")


def fig_forest(tax: dict) -> None:
    prows = tax["arms"]["prefix_L19_ridge"]["contrasts"]
    crows = {r["contrast"]: r for r in tax["arms"]["context_L19_ridge"]["contrasts"]}
    order = sorted(prows, key=lambda r: r["delta_mean_nerr"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 7.2), sharey=True)
    y = np.arange(len(order))
    for ax, arm, rows in [
        (axes[0], "prefix", order),
        (axes[1], "context", [crows[r["contrast"]] for r in order]),
    ]:
        for yi, r in zip(y, rows):
            lo, hi = r["boot_ci"]
            d = r["delta_mean_nerr"]
            sig = bool(r["bh_significant"])
            ax.plot([lo, hi], [yi, yi], color=ARM_COLORS[arm], lw=1.4)
            if sig:
                ax.scatter([d], [yi], s=34, color=ARM_COLORS[arm], zorder=3)
            else:
                ax.scatter(
                    [d],
                    [yi],
                    s=34,
                    facecolors="white",
                    edgecolors=ARM_COLORS[arm],
                    linewidths=1.4,
                    zorder=3,
                )
        ax.axvline(0.0, color="#666666", lw=0.9, ls=":")
        ax.set_title(ARM_LABELS[arm], loc="left")
        ax.set_xlabel("difference in mean normalized error\n(group minus rest)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [f"{CONTRAST_LABELS[r['contrast']]} (n={r['n_group']:,})" for r in order]
    )
    savefig_paper(fig, "issue_1738/taxonomy_forest_L19_ridge", dir=FIGDIR)
    plt.close(fig)


def fig_baselines(mb: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    est_order = ["ridge", "identity_bias"]
    est_labels = {"ridge": "fitted ridge map", "identity_bias": "identity + learned bias"}
    x = np.arange(len(est_order))
    width = 0.38
    fits = _load("fits/multiturn_100k_fits.json")["cells"]
    for j, arm in enumerate(["prefix", "context"]):
        cell = mb["cells"][f"{arm}_L19"]
        r2 = [fits[f"{arm}_L19_ridge"]["holdout_r2"], cell["identity_bias"]["holdout_r2"]]
        axes[0].bar(x + (j - 0.5) * width, r2, width, color=ARM_COLORS[arm], label=ARM_LABELS[arm])
        acc = [
            cell["knn"]["ridge"]["cosine"]["acc_at_k"]["1"],
            cell["knn"]["identity_bias"]["cosine"]["acc_at_k"]["1"],
        ]
        axes[1].bar(x + (j - 0.5) * width, acc, width, color=ARM_COLORS[arm])
    axes[0].axhline(0.0, color="#666666", lw=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([est_labels[e] for e in est_order])
    axes[0].set_ylabel("held-out R² (layer 19)")
    axes[0].set_title("variance explained", loc="left")
    axes[1].set_yscale("log")
    axes[1].axhline(1.0e-4, color="#b04a4a", ls="--", lw=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([est_labels[e] for e in est_order])
    axes[1].set_ylabel("retrieval accuracy at rank 1 (cosine)")
    axes[1].set_title("retrieval among 9,941 held-out targets", loc="left")
    axes[0].legend()
    savefig_paper(fig, "issue_1738/mapping_baselines_dissociation", dir=FIGDIR)
    plt.close(fig)


def fig_perdirection(pd: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    ranks = np.arange(1, pd["topk"] + 1)
    for ax, arm in zip(axes, ["prefix", "context"]):
        d = pd["arms"][arm]["per_direction"]
        c = ARM_COLORS[arm]
        ax.plot(ranks, d["ridge_shared"], color=c, lw=1.4, label="ridge (shared penalty)")
        ax.plot(ranks, d["mlp_w8192"], color=c, lw=1.4, ls="--", label="MLP (width 8,192)")
        ax.plot(
            ranks,
            d["ridge_tuned"],
            color="#555555",
            lw=1.0,
            ls=":",
            label="ridge (per-direction penalty)",
        )
        ax.set_xscale("log")
        ax.set_title(ARM_LABELS[arm], loc="left")
        ax.set_xlabel("answer-PCA direction rank")
        ax.legend()
    axes[0].set_ylabel("per-direction held-out R² (layer 19)")
    savefig_paper(fig, "issue_1738/perdirection_r2_L19", dir=FIGDIR)
    plt.close(fig)


def fig_hero_3arm(fits: dict, bare_fits: dict) -> None:
    """Bare-query round (plan §4.3): 3-arm hero — held-out R² per arm × layer ×
    fitter with bootstrap CIs; bare cells come from the bare round's fits JSON,
    prefix/context from the parent's (same split/targets, not refit)."""
    cells = {**fits["cells"], **bare_fits["cells"]}
    layers = [14, 19, 26]
    arms3 = ["bare", "prefix", "context"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=True)
    width = 0.26
    x = np.arange(len(FITTER_ORDER))
    for ax, layer in zip(axes, layers):
        for j, arm in enumerate(arms3):
            pts, lo, hi = [], [], []
            for f in FITTER_ORDER:
                c = cells[f"{arm}_L{layer}_{f}"]
                ci = c["holdout_bootstrap_ci"]["r2"]
                pts.append(c["holdout_r2"])
                # non-negative offsets (gotchas #547/#1335)
                lo.append(max(0.0, c["holdout_r2"] - ci["lo"]))
                hi.append(max(0.0, ci["hi"] - c["holdout_r2"]))
            ax.bar(
                x + (j - 1) * width,
                pts,
                width,
                yerr=[lo, hi],
                color=ARM_COLORS[arm],
                label=ARM_LABELS[arm] if layer == 14 else None,
                error_kw={"elinewidth": 1.0, "ecolor": "#333333"},
            )
        ax.set_title(f"layer {layer}", loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels([FITTER_LABELS[f] for f in FITTER_ORDER], rotation=20, ha="right")
        ax.axhline(0.0, color="#666666", lw=0.8)
    axes[0].set_ylabel("held-out R² (9,941 contexts)")
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    savefig_paper(fig, "issue_1738/hero_arm_r2_by_layer_3arm", dir=FIGDIR)
    plt.close(fig)


def fig_scatter_bare() -> None:
    """Bare-query round (plan §4.3): per-context error scatter, bare vs context
    and bare vs prefix (layer 19, ridge), on the shared holdout rows — plus the
    per-cell CSV behind the scatters (plan §6.5 primary deliverable)."""
    zb = np.load(EB / "percontext/bare_L19_ridge.npz")
    labels = _load("judge_labels/labels.json")["labels"]
    zp = np.load(E / "percontext/prefix_L19_ridge.npz")
    zc = np.load(E / "percontext/context_L19_ridge.npz")
    assert (zb["ci"] == zp["ci"]).all() and (zb["ci"] == zc["ci"]).all()
    out = EB / "percontext_summary_L19_ridge.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "ci",
                "nerr_bare_L19_ridge",
                "nerr_prefix_L19_ridge",
                "nerr_context_L19_ridge",
                "language",
                "topic",
                "format",
            ]
        )
        for i, c in enumerate(zb["ci"].tolist()):
            lab = labels.get(str(c), {})
            w.writerow(
                [
                    c,
                    f"{float(zb['nerr'][i]):.6f}",
                    f"{float(zp['nerr'][i]):.6f}",
                    f"{float(zc['nerr'][i]):.6f}",
                    lab.get("language", ""),
                    lab.get("topic", ""),
                    lab.get("format", ""),
                ]
            )
    print(f"wrote {out}")
    for other in ("context", "prefix"):
        zo = np.load(E / f"percontext/{other}_L19_ridge.npz")
        assert (zb["ci"] == zo["ci"]).all(), f"bare/{other} percontext ci misalign"
        xo, yb = zo["nerr"].astype(float), zb["nerr"].astype(float)
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        ax.scatter(xo, yb, s=4, alpha=0.12, color="#5a5a5a", edgecolors="none", rasterized=True)
        lims = (0.02, 3.2)
        ax.plot(lims, lims, ls="--", lw=1.0, color="#b04a4a")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.set_xlabel(f"per-context normalized error, {other} arm (layer 19, ridge)")
        ax.set_ylabel("per-context normalized error, bare-query arm (layer 19, ridge)")
        savefig_paper(
            fig, f"issue_1738/percontext_nerr_scatter_bare_vs_{other}", dir=FIGDIR, embed_data=False
        )
        plt.close(fig)
        print(f"bare vs {other}: frac_bare_worse={float((yb > xo).mean()):.4f}")


def fig_forest_bare(btax: dict) -> None:
    """Bare-query round (plan §4.3): taxonomy contrast forest for the bare arm."""
    rows = btax["arms"]["bare_L19_ridge"]["contrasts"]
    order = sorted(rows, key=lambda r: r["delta_mean_nerr"])
    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    y = np.arange(len(order))
    for yi, r in zip(y, order):
        lo, hi = r["boot_ci"]
        d = r["delta_mean_nerr"]
        ax.plot([lo, hi], [yi, yi], color=ARM_COLORS["bare"], lw=1.4)
        if bool(r["bh_significant"]):
            ax.scatter([d], [yi], s=34, color=ARM_COLORS["bare"], zorder=3)
        else:
            ax.scatter(
                [d],
                [yi],
                s=34,
                facecolors="white",
                edgecolors=ARM_COLORS["bare"],
                linewidths=1.4,
                zorder=3,
            )
    ax.axvline(0.0, color="#666666", lw=0.9, ls=":")
    ax.set_title(ARM_LABELS["bare"], loc="left")
    ax.set_xlabel("difference in mean normalized error\n(group minus rest)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{CONTRAST_LABELS[r['contrast']]} (n={r['n_group']:,})" for r in order])
    savefig_paper(fig, "issue_1738/taxonomy_forest_bare_L19_ridge", dir=FIGDIR)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    pal = paper_palette_blog(6)
    ARM_COLORS["prefix"] = pal[1]
    ARM_COLORS["context"] = pal[0]
    ARM_COLORS["bare"] = pal[2]
    fits = _load("fits/multiturn_100k_fits.json")
    fig_hero(fits)
    fig_depth(_load("depth_contrasts.json"))
    fig_scatter()
    fig_forest(_load("taxonomy.json"))
    fig_baselines(_load("mapping_baselines.json"))
    fig_perdirection(_load("perdirection/pdshrink_summary.json"))
    # bare-query round additions (plan §4.3) — data-only; render once the bare
    # round's artifacts exist under eval_results/issue_1738/bare_query/.
    bare_fits_p = EB / "fits/multiturn_100k_fits.json"
    if bare_fits_p.exists():
        fig_hero_3arm(fits, json.load(open(bare_fits_p)))
        fig_scatter_bare()
        btax_p = EB / "taxonomy.json"
        if btax_p.exists():
            fig_forest_bare(json.load(open(btax_p)))
    else:
        print("bare_query fits JSON absent — 3-arm figures skipped (pre-B2)")
    print("done")


if __name__ == "__main__":
    main()
