"""Analyzer figures for issue #1345 (clean-result embeds).

Reads the committed eval JSONs under ``eval_results/issue_1345/`` and produces
the five figures the clean-result body embeds, via the paper-plots conventions
(``set_paper_style("blog")`` + ``savefig_paper`` -> PNG + PDF + .meta.json).

Usage:
    uv run python scripts/issue1345_analyzer_figs.py \
        --eval-dir <worktree>/eval_results/issue_1345 \
        --out-dir <repo-root>/figures/issue_1345
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    figsize_iclr_full,
    paper_color,
    paper_palette,
    savefig_paper,
    set_paper_style,
)

REGIME_NAMES = {"r1": "chat", "r2": "no-template", "r3": "stories"}


def _load(eval_dir: Path, name: str) -> dict:
    with open(eval_dir / name) as f:
        return json.load(f)


def fig_transfer_heatmaps(eval_dir: Path, out_dir: Path) -> None:
    """Hero: cross-regime transfer R^2 heatmaps (context arm, L19), instruct + base."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, model, regimes in (
        (axes[0], "instruct", ["r1", "r2", "r3"]),
        (axes[1], "base", ["r1", "r2"]),
    ):
        d = _load(eval_dir, f"cross_regime_transfer_{model}_context.json")
        dt = d["delta_table_l19"]
        n = len(regimes)
        mat = np.full((n, n), np.nan)
        for i, ri in enumerate(regimes):
            for j, rj in enumerate(regimes):
                if i == j:
                    # within-regime held-out R^2 at L19 (regime's own fit)
                    # diagonal: pull target_within from any row targeting rj
                    for k, v in dt.items():
                        if k.endswith(f"->{rj}"):
                            mat[i, j] = v["target_within_l19"]
                            break
                else:
                    mat[i, j] = dt[f"{ri}->{rj}"]["transfer_l19"]
        vmin = np.nanmin(mat) - 0.05 if np.nanmin(mat) < 0 else 0.0
        im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=0.7)
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if mat[i, j] < 0.35 else "black",
                    fontsize=13,
                )
        labels = [REGIME_NAMES[r] for r in regimes]
        ax.set_xticks(range(n), labels)
        ax.set_yticks(range(n), labels)
        ax.set_xlabel("target regime (evaluate on)")
        if model == "instruct":
            ax.set_ylabel("source regime (map fitted on)")
        ax.set_title(f"{'Instruct' if model == 'instruct' else 'Base'}", fontsize=15)
        fig.colorbar(im, ax=ax, shrink=0.85, label="held-out $R^2$ (layer 19)")
    fig.suptitle("Cross-regime transfer $R^2$ — context arm, layer 19", fontsize=17)
    savefig_paper(fig, "hero_transfer_heatmaps_context", dir=out_dir)
    plt.close(fig)


def fig_reparam_recovery(eval_dir: Path, out_dir: Path) -> None:
    """Reparameterization recovery vs matched-capacity null (context arm, L19)."""
    groups = []
    for model in ("instruct", "base"):
        d = _load(eval_dir, f"operator_comparison_{model}_context.json")
        dr = d["delta_reparam_l19"]
        nulls = d["reparam_r1r2"]["19"]["matched_capacity_nulls"]
        # b2i = no-template operator recovered in chat; i2b = chat operator in no-template
        for dkey, glabel in (
            ("b2i", "no-template map\nin chat"),
            ("i2b", "chat map\nin no-template"),
        ):
            groups.append(
                {
                    "label": f"{model}:\n{glabel}",
                    "within": dr["within_r2"][dkey],
                    "recovered": dr["recovered_r2"][dkey],
                    "null": nulls[dkey]["null_recovery_r2"],
                }
            )
    x = np.arange(len(groups))
    w = 0.27
    colors = paper_palette(3)
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    for off, key, label, c in (
        (-w, "within", "target regime's own held-out $R^2$", colors[0]),
        (0.0, "recovered", "recovered $R^2$ (general-linear reparam)", colors[1]),
        (w, "null", "matched-capacity null recovery", colors[2]),
    ):
        vals = [g[key] for g in groups]
        ax.bar(x + off, vals, width=w, color=c, label=label)
        for xi, v in zip(x + off, vals, strict=True):
            ax.text(xi, v + 0.012 if v >= 0 else v - 0.045, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x, [g["label"] for g in groups])
    ax.set_ylabel("held-out $R^2$ (layer 19)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title(
        "General-linear reparameterization recovers the target map (context arm, layer 19)",
        pad=14,
        fontsize=14,
    )
    ax.legend(loc="center right", fontsize=11)
    ax.set_ylim(-0.12, 0.78)
    savefig_paper(fig, "reparam_recovery_context", dir=out_dir)
    plt.close(fig)


def fig_operator_cosine(eval_dir: Path, out_dir: Path) -> None:
    """Raw vs activation-Procrustes-aligned operator cosine (chat vs no-template), L19."""
    rows = []
    for model in ("instruct", "base"):
        d = _load(eval_dir, f"operator_comparison_{model}_context.json")
        ap = d["reparam_r1r2"]["19"]["activation_procrustes"]
        rows.append(
            {
                "model": model,
                "raw": ap["raw_vec_cosine"],
                "aligned": ap["observed_aligned_cosine"],
                "null": ap["null_p975"],
            }
        )
    anchor = _load(eval_dir, "operator_comparison_instruct_context.json")["calibration_anchor"][
        "base_instruct_aligned_cosine_825"
    ]
    x = np.arange(len(rows))
    w = 0.26
    colors = paper_palette(3)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for off, key, label, c in (
        (-w, "raw", "raw operator cosine", colors[0]),
        (0.0, "aligned", "aligned cosine (activation Procrustes)", colors[1]),
        (w, "null", "random-rotation null (97.5th pct)", colors[2]),
    ):
        vals = [r[key] for r in rows]
        ax.bar(x + off, vals, width=w, color=c, label=label)
        for xi, v in zip(x + off, vals, strict=True):
            ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=10)
    ax.axhline(
        anchor,
        color="gray",
        ls="--",
        lw=1.2,
        label="base vs instruct aligned cosine (parent anchor)",
    )
    ax.set_xticks(x, ["instruct", "base"])
    ax.set_ylabel("cosine between chat and no-template operators (layer 19)")
    ax.set_title(
        "Chat vs no-template operator cosine, raw and aligned (context arm)", pad=14, fontsize=14
    )
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=10, loc="upper right")
    savefig_paper(fig, "operator_cosine_raw_vs_aligned_context", dir=out_dir)
    plt.close(fig)


def fig_story_layer_sweep(eval_dir: Path, out_dir: Path) -> None:
    """Instruct within-regime R^2 across all 28 layers, both arms, + story shuffle-null p95."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    colors = paper_palette(3)
    for ax, arm in ((axes[0], "context"), (axes[1], "prefix")):
        for c, reg in zip(colors, ("r1", "r2", "r3"), strict=True):
            d = _load(eval_dir, f"cells_R_instruct_{reg}_{arm}.json")
            r2 = d["r2_per_layer_obs"]
            ax.plot(range(len(r2)), r2, marker="o", ms=3, color=c, label=REGIME_NAMES[reg])
        nd = _load(eval_dir, f"nulls_R_instruct_r3_{arm}.json")
        nm = np.array(nd["null_matrix"])  # draws x layers
        p95 = np.percentile(nm, 95, axis=0)
        ax.plot(
            range(nm.shape[1]), p95, ls="--", color="gray", label="stories shuffle-null 95th pct"
        )
        ax.axvline(19, color="black", lw=0.8, ls=":")
        ax.set_xlabel("layer")
        ax.set_title(f"{arm} arm", fontsize=14)
        if arm == "context":
            ax.set_ylabel("within-regime held-out $R^2$")
        ax.legend(fontsize=10)
    fig.suptitle(
        "Within-regime layer sweep — instruct, both arms (dotted line: layer 19)", fontsize=16
    )
    savefig_paper(fig, "story_layer_sweep_instruct_both_arms", dir=out_dir)
    plt.close(fig)


def fig_prefix_transfer(eval_dir: Path, out_dir: Path) -> None:
    """Prefix arm: catastrophic naive transfer vs full reparam recovery (symlog scale)."""
    groups = []
    for model in ("instruct", "base"):
        t = _load(eval_dir, f"cross_regime_transfer_{model}_prefix.json")["delta_table_l19"]
        o = _load(eval_dir, f"operator_comparison_{model}_prefix.json")["delta_reparam_l19"]
        for dkey, rec_key, glabel in (
            ("r2->r1", "b2i", "no-template map\nin chat"),
            ("r1->r2", "i2b", "chat map\nin no-template"),
        ):
            groups.append(
                {
                    "label": f"{model}:\n{glabel}",
                    "transfer": t[dkey]["transfer_l19"],
                    "within": t[dkey]["target_within_l19"],
                    "recovered": o["recovered_r2"][rec_key],
                }
            )
    x = np.arange(len(groups))
    w = 0.27
    colors = paper_palette(3)
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    for off, key, label, c in (
        (-w, "within", "target regime's own held-out $R^2$", colors[0]),
        (0.0, "transfer", "naive cross-regime transfer $R^2$", colors[1]),
        (w, "recovered", "recovered $R^2$ (general-linear reparam)", colors[2]),
    ):
        vals = [g[key] for g in groups]
        ax.bar(x + off, vals, width=w, color=c, label=label)
        for xi, v in zip(x + off, vals, strict=True):
            txt = f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"
            ax.text(xi, v * 1.6 if v < -1 else v + 0.05, txt, ha="center", fontsize=9)
    ax.set_yscale("symlog", linthresh=0.2)
    ax.set_ylim(-4e4, 1.2)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x, [g["label"] for g in groups])
    ax.set_ylabel("held-out $R^2$ (layer 19, symlog scale)")
    ax.set_title(
        "Prefix arm: naive transfer explodes; reparameterization recovers the ceiling",
        pad=14,
        fontsize=14,
    )
    ax.legend(fontsize=10, loc="lower right")
    savefig_paper(fig, "prefix_transfer_vs_reparam", dir=out_dir)
    plt.close(fig)


def fig_reparam_recovery_iclr(eval_dir: Path, out_dir: Path) -> None:
    """ICLR paper figure: direct transfer vs general-linear recovery, chat<->plain.

    Two bars per (model x direction) group — naive direct transfer (reduced
    alpha) and the general-linear reparameterization recovery (full alpha) —
    against the target regime's own held-out ceiling (black tick) and the
    matched-capacity null (gray dotted tick). Context arm, layer 19. Value
    labels live in the LaTeX caption, not on the canvas.
    """
    groups = []
    for model in ("instruct", "base"):
        opc = _load(eval_dir, f"operator_comparison_{model}_context.json")
        dr = opc["delta_reparam_l19"]
        nulls = opc["reparam_r1r2"]["19"]["matched_capacity_nulls"]
        tr = _load(eval_dir, f"cross_regime_transfer_{model}_context.json")["delta_table_l19"]
        # b2i = plain-text (r2) operator moved into chat (r1); i2b = the reverse
        for dkey, tkey, glabel in (
            ("b2i", "r2->r1", "plain→chat"),
            ("i2b", "r1->r2", "chat→plain"),
        ):
            groups.append(
                {
                    "label": f"{model}\n{glabel}",
                    "model": model,
                    "transfer": tr[tkey]["transfer_l19"],
                    "recovered": dr["recovered_r2"][dkey],
                    "within": dr["within_r2"][dkey],
                    "null": nulls[dkey]["null_recovery_r2"],
                }
            )
    x = np.arange(len(groups))
    w = 0.34
    fig, ax = plt.subplots(figsize=figsize_iclr_full(0.55))
    for g, xi in zip(groups, x, strict=True):
        c = paper_color(g["model"])
        ax.bar(xi - w / 2, g["transfer"], width=w * 0.92, color=c, alpha=0.45)
        ax.bar(xi + w / 2, g["recovered"], width=w * 0.92, color=c)
        ax.plot(
            [xi - 0.44, xi + 0.44],
            [g["within"]] * 2,
            color=paper_color("reference"),
            lw=1.1,
            zorder=4,
        )
        ax.plot([xi - 0.44, xi + 0.44], [g["null"]] * 2, color=paper_color("null"), lw=1.0, ls=":")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xticks(x, [g["label"] for g in groups])
    ax.set_ylabel("held-out $R^2$ (layer 19)")
    ax.set_ylim(-0.08, 0.72)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="0.35", alpha=0.45, label="direct transfer"),
            Patch(facecolor="0.35", label="after linear change of coordinates"),
            Line2D([], [], color=paper_color("reference"), lw=1.1, label="target's own ceiling"),
            Line2D(
                [], [], color=paper_color("null"), lw=1.0, ls=":", label="matched-capacity null"
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01, 1.0, 0.2),
        mode="expand",
        ncols=2,
        borderaxespad=0.0,
    )
    savefig_paper(fig, "c4_reparam_recovery", dir=out_dir)
    plt.close(fig)


def fig_story_vs_chat_iclr(eval_dir: Path, out_dir: Path) -> None:
    """ICLR paper figure: story framing weakens the map ~2x on identical rows.

    Matched-row instruct cells from the #1887 estimator-correction table
    (train-fold reduced-basis reads — the corrected estimator; the committed
    pure-GCV story cells are superseded). Two provenance modes per framing:
    the original answer embedded verbatim vs the model's own in-story answer.
    Diamond markers: the inner-group-CV companion estimator for the same cell.
    """
    table = _load(eval_dir, "lambda_audit_1887/corrections_table.json")
    rows = {r["cell_id"]: r for r in table["rows"]}
    cells = {
        ("chat", "tf"): "conversation_paired_stories_assistant__R_instruct_r1_matched_context",
        ("chat", "op"): "onpolicy_assistant_story__R_instruct_r1_matched_context",
        ("plain", "tf"): "conversation_paired_stories_assistant__R_instruct_r2_matched_context",
        ("plain", "op"): "onpolicy_assistant_story__R_instruct_r2_matched_context",
        ("story", "tf"): "conversation_paired_stories_assistant__R_instruct_r4_context",
        ("story", "op"): "onpolicy_assistant_story__R_instruct_r4_op_companion_context",
    }
    framings = [("chat", "chat template"), ("plain", "plain text"), ("story", "narrative story")]
    x = np.arange(len(framings))
    w = 0.34
    c = paper_color("instruct")
    fig, ax = plt.subplots(figsize=figsize_iclr_full(0.55))
    for j, (prov, alpha) in enumerate((("tf", 1.0), ("op", 0.45))):
        xi = x + (j - 0.5) * w
        vals = [rows[cells[(f, prov)]]["corrected_reduced_basis_r2"] for f, _ in framings]
        comp = [rows[cells[(f, prov)]]["corrected_inner_cv_r2"] for f, _ in framings]
        ax.bar(xi, vals, width=w * 0.92, color=c, alpha=alpha)
        ax.plot(
            xi,
            comp,
            "D",
            ms=3.2,
            color=paper_color("reference"),
            mfc="none",
            mew=0.9,
            zorder=4,
        )
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xticks(x, [lab for _, lab in framings])
    ax.set_ylabel("held-out $R^2$ (layer 19, matched rows)")
    ax.set_ylim(0, 0.72)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=c, label="original answer embedded verbatim"),
            Patch(facecolor=c, alpha=0.45, label="model's own answer"),
            Line2D(
                [],
                [],
                marker="D",
                ls="none",
                ms=3.2,
                color=paper_color("reference"),
                mfc="none",
                mew=0.9,
                label="companion estimator (inner-group CV)",
            ),
        ],
        loc="upper right",
    )
    savefig_paper(fig, "c4_story_vs_chat", dir=out_dir)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--style", choices=("blog", "iclr"), default="blog")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.style == "iclr":
        set_paper_style("iclr")
        fig_reparam_recovery_iclr(args.eval_dir, args.out_dir)
        fig_story_vs_chat_iclr(args.eval_dir, args.out_dir)
        print("DONE (iclr)", args.out_dir)
        return
    set_paper_style("blog")
    fig_transfer_heatmaps(args.eval_dir, args.out_dir)
    fig_reparam_recovery(args.eval_dir, args.out_dir)
    fig_operator_cosine(args.eval_dir, args.out_dir)
    fig_story_layer_sweep(args.eval_dir, args.out_dir)
    fig_prefix_transfer(args.eval_dir, args.out_dir)
    print("DONE", args.out_dir)


if __name__ == "__main__":
    main()
