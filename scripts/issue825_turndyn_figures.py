"""Figures for issue #825 same-issue follow-up round `turn-dynamics-allturns-5000`.

Reads the committed round eval JSON (issue-825 branch @ 0df0f8b592):
  eval_results/issue_825/turn_dynamics/results.json
and renders the round's figure set to figures/issue_825/turndyn/ (blog style,
paper_plots conventions). Pure plotting — no fits, no new statistics.

Usage:
    uv run python scripts/issue825_turndyn_figures.py [path/to/results.json]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

DEFAULT_RESULTS = (
    "/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-825/"
    "eval_results/issue_825/turn_dynamics/results.json"
)
OUT_DIR = "figures/issue_825/turndyn"
LAYER = "19"
MODELS = ("instruct", "pretrained")
MODEL_TITLE = {"instruct": "Instruct model", "pretrained": "Pretrained base model"}


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def per_turn_series(
    cells: dict, mapping: str
) -> tuple[list[int], list[float], list[list[float]], list[float], list[int]]:
    """Return turns, r2, r2_folds, null_hi, n for one mapping arm at LAYER."""
    pt = cells["per_turn"][LAYER]
    turns, r2, folds, null_hi, ns = [], [], [], [], []
    for t in sorted(pt, key=int):
        cell = pt[t].get(mapping, {})
        if cell.get("status") != "computed":
            continue
        turns.append(int(t))
        r2.append(cell["r2"])
        folds.append(cell.get("r2_folds", []))
        null_hi.append(cell["null_hi"])
        ns.append(pt[t]["n"])
    return turns, r2, folds, null_hi, ns


def main() -> None:
    results_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS
    d = load(results_path)
    p = d["parts"]
    set_paper_style("blog")
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    c = paper_palette_blog(6)
    col_g, col_r_own, col_r_log = c[0], c[1], c[2]
    col_mlp, col_ridge = c[3], c[0]

    # ------------------------------------------------------------------
    # Figure 1 (hero): per-turn held-out R2 at flat n≈5000, all arms
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, model in zip(axes, MODELS):
        arms = [
            (
                f"cells_armG_{model}",
                "ctx",
                col_g,
                "-",
                "o",
                "simulated continuation, own answers (n=5000/turn)",
            ),
            (
                f"cells_armR_own_{model}",
                "ctx",
                col_r_own,
                "-",
                "s",
                "real logged context, own answers (n≈5000/turn)",
            ),
            (
                f"cells_armR_logged_{model}",
                "ctx",
                col_r_log,
                "-",
                "^",
                "real logged context + logged answers (n≈4950/turn)",
            ),
            (f"cells_armG_{model}", "pfx", col_g, "--", "o", "prefix arm (simulated)"),
            (f"cells_armR_own_{model}", "pfx", col_r_own, "--", "s", "prefix arm (real, own)"),
        ]
        null_lo, null_hi_all = [], []
        for key, mapping, col, ls, mk, label in arms:
            turns, r2, folds, null_hi, ns = per_turn_series(p[key], mapping)
            if key.startswith("cells_armR_logged"):
                keep = [i for i, t in enumerate(turns) if t <= 13]  # flat-n panel span
                turns = [turns[i] for i in keep]
                r2 = [r2[i] for i in keep]
                folds = [folds[i] for i in keep]
                null_hi = [null_hi[i] for i in keep]
            ax.plot(
                turns,
                r2,
                ls,
                marker=mk,
                ms=4,
                color=col,
                lw=1.8,
                alpha=1.0 if ls == "-" else 0.55,
                label=label,
            )
            for t, fl in zip(turns, folds):
                ax.plot(
                    [t] * len(fl),
                    fl,
                    ".",
                    color=col,
                    ms=2.2,
                    alpha=0.35 if ls == "-" else 0.18,
                    zorder=1,
                )
            null_hi_all.extend(null_hi)
        band_hi = max(null_hi_all)
        ax.axhspan(-0.25, band_hi, color="0.85", alpha=0.55, zorder=0)
        ax.text(
            1.1,
            band_hi - 0.035,
            "shuffled-answer null (max over 200 draws/cell)",
            fontsize=8,
            color="0.35",
            va="top",
        )
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("Assistant turn depth")
        ax.set_title(MODEL_TITLE[model])
        ax.set_ylim(-0.25, 0.75)
    axes[0].set_ylabel("Held-out R² (context → answer, layer 19)")
    axes[0].legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    savefig_paper(fig, "turndyn_perturn_r2", dir=OUT_DIR)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: real logged deep tail (t13-30), decaying n
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        turns, r2, folds, null_hi, ns = per_turn_series(p[f"cells_armR_logged_{model}"], "ctx")
        keep = [i for i, t in enumerate(turns) if 13 <= t <= 30]
        t_, r_, nh_, n_ = ([x[i] for i in keep] for x in (turns, r2, null_hi, ns))
        ax.plot(
            t_,
            r_,
            "-",
            marker="^",
            ms=4,
            color=col_r_log,
            lw=1.6,
            label="real logged conversations, logged answers",
        )
        ax.plot(t_, nh_, ":", color="0.45", lw=1.4, label="shuffled-answer null (max of 200 draws)")
        for t, r, n in zip(t_, r_, n_):
            ax.text(
                t,
                r + 0.12,
                f"n={n}",
                fontsize=6.2,
                rotation=90,
                ha="center",
                va="bottom",
                color="0.3",
            )
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("Assistant turn depth")
        ax.set_title(MODEL_TITLE[model])
        ax.set_ylim(-4.4, 1.6)
    axes[0].set_ylabel("Held-out R² (context → answer, layer 19)")
    axes[0].legend(fontsize=8, loc="lower right")
    savefig_paper(fig, "turndyn_logged_tail", dir=OUT_DIR)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: cross-turn transfer matrices (arm G)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))
    for ax, model in zip(axes, MODELS):
        tr = p[f"transfer_armG_{model}"]
        turns = tr["turns"]
        m = np.full((len(turns), len(turns)), np.nan)
        for a, i in enumerate(turns):
            for b, j in enumerate(turns):
                v = tr["r2"].get(f"{i}->{j}")
                if v is not None:
                    m[a, b] = v
        im = ax.imshow(m, vmin=-0.65, vmax=0.65, cmap="RdBu_r", origin="upper")
        ax.grid(False)
        ax.set_xticks(range(len(turns)), [str(j) for j in turns], fontsize=7)
        ax.set_yticks(range(len(turns)), [str(i) for i in turns], fontsize=7)
        ax.set_xlabel("Applied to turn j (held-out)")
        ax.set_ylabel("Map fitted at turn i")
        ax.set_title(MODEL_TITLE[model])
        fig.colorbar(im, ax=ax, shrink=0.85, label="Held-out R²")
    savefig_paper(fig, "turndyn_transfer_matrix", dir=OUT_DIR)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4: transfer rows (low-level view of the matrix)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    row_cols = paper_palette_blog(5)
    for ax, model in zip(axes, MODELS):
        tr = p[f"transfer_armG_{model}"]
        turns = tr["turns"]
        diag = [tr["r2"][f"{j}->{j}"] for j in turns]
        ax.plot(turns, diag, "k:", lw=1.6, label="own-turn map (diagonal)")
        for ci, i in enumerate([1, 2, 3, 8]):
            row = [tr["r2"][f"{i}->{j}"] for j in turns]
            ax.plot(
                turns,
                row,
                "-",
                marker="o",
                ms=3.4,
                lw=1.6,
                color=row_cols[ci],
                label=f"map fitted at turn {i}",
            )
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("Applied to turn j (held-out)")
        ax.set_title(MODEL_TITLE[model])
    axes[0].set_ylabel("Held-out R² (layer 19)")
    axes[0].legend(fontsize=8, loc="lower right")
    savefig_paper(fig, "turndyn_transfer_rows", dir=OUT_DIR)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 5: reach — turn-1 context predicting turn-k answers
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, model in zip(axes, MODELS):
        for arm, col, mk, lab in [
            (f"reach_armG_{model}", col_g, "o", "simulated continuations"),
            (f"reach_armR_own_{model}", col_r_own, "s", "real logged context, own answers"),
        ]:
            r = p[arm]
            ks = sorted(r["ridge"], key=int)
            kk = [int(k) for k in ks]
            ax.plot(
                kk,
                [r["ridge"][k]["r2"] for k in ks],
                "-",
                marker=mk,
                ms=4,
                color=col,
                lw=1.8,
                label=f"ridge — {lab} (n≈5000)",
            )
            for k in ks:
                fl = r["ridge"][k].get("r2_folds", [])
                ax.plot([int(k)] * len(fl), fl, ".", color=col, ms=2.2, alpha=0.35, zorder=1)
            ax.plot(
                kk,
                [r["mlp"][k]["r2"] for k in ks],
                "--",
                marker=mk,
                ms=3.4,
                color=col_mlp if arm.startswith("reach_armG") else "0.55",
                lw=1.4,
                label=f"MLP — {lab} (n=1000, PCA 256→48)",
            )
        nh = max(
            p[f"reach_armG_{model}"]["ridge"][k]["null_hi"]
            for k in p[f"reach_armG_{model}"]["ridge"]
        )
        ax.axhline(nh, color="0.45", ls=":", lw=1.3)
        ax.text(
            1.2,
            nh - 0.03,
            "ridge shuffled-answer null (max of 200 draws)",
            fontsize=7.5,
            color="0.35",
            va="top",
        )
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("Answer horizon k (turn-1 context → turn-k answer)")
        ax.set_title(MODEL_TITLE[model])
        ax.set_ylim(-0.5, 0.65)
    axes[0].set_ylabel("Held-out R² (layer 19)")
    axes[0].legend(fontsize=7.5, loc="upper right")
    savefig_paper(fig, "turndyn_reach", dir=OUT_DIR)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 6: bridge (simulated vs real, seed intersection) + provenance
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    for model, col, mk in [("instruct", c[0], "o"), ("pretrained", c[4], "D")]:
        b = d["bridge_H4"][model]["per_turn"]
        ts = sorted(b, key=int)
        tt = [int(t) for t in ts]
        deltas = [b[t]["delta"] for t in ts]
        lo = [b[t]["delta"] - b[t]["delta_ci"][0] for t in ts]
        hi = [b[t]["delta_ci"][1] - b[t]["delta"] for t in ts]
        ax.errorbar(
            tt,
            deltas,
            yerr=[lo, hi],
            fmt=mk + "-",
            ms=4,
            lw=1.5,
            color=col,
            capsize=2,
            label=MODEL_TITLE[model],
        )
    ax.axhspan(-0.10, 0.10, color="0.88", alpha=0.6, zorder=0)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("Assistant turn depth")
    ax.set_ylabel("R² delta: simulated − real (same conversations)")
    ax.set_title("Simulated-vs-real bridge (seed intersection, ±0.10 band)")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    for model, colp in [("instruct", c[0]), ("pretrained", c[4])]:
        for arm, ls, suffix in [
            (f"cells_armR_own_{model}", "-", "own answers"),
            (f"cells_armR_logged_{model}", "--", "logged answers"),
        ]:
            turns, r2, folds, null_hi, ns = per_turn_series(p[arm], "ctx")
            keep = [i for i, t in enumerate(turns) if t <= 12]
            ax.plot(
                [turns[i] for i in keep],
                [r2[i] for i in keep],
                ls,
                marker="o",
                ms=3.4,
                lw=1.6,
                color=colp,
                alpha=1.0 if ls == "-" else 0.55,
                label=f"{MODEL_TITLE[model]} — {suffix}",
            )
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("Assistant turn depth")
    ax.set_ylabel("Held-out R² (layer 19)")
    ax.set_title("Real conversations: own vs logged answers (n≈5000/turn)")
    ax.legend(fontsize=7.5, loc="lower right")
    savefig_paper(fig, "turndyn_bridge_provenance", dir=OUT_DIR)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 7: operator similarity vs within-turn resample ceiling (arm G)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, model in zip(axes, MODELS):
        o = p[f"operators_armG_{model}"]
        turns = o["turns"]
        ceil = [o["selfsim_ceiling"][str(t)]["raw_cos_mean"] for t in turns]
        ax.plot(turns, ceil, "k:", lw=1.6, label="within-turn resample ceiling")
        for ci, i in enumerate([1, 3, 8]):
            xs = [j for j in turns if j != i]
            ys = [o["battery"][f"{min(i, j)}~{max(i, j)}"]["raw_cos_mean"] for j in xs]
            ax.plot(
                xs,
                ys,
                "-",
                marker="o",
                ms=3.4,
                lw=1.6,
                color=row_cols[ci],
                label=f"map at turn {i} vs map at turn j",
            )
        ax.set_xlabel("Other turn j")
        ax.set_title(MODEL_TITLE[model])
        ax.set_ylim(0, 0.14)
    axes[0].set_ylabel("Raw operator cosine (fold-resampled)")
    axes[0].legend(fontsize=8, loc="lower right")
    savefig_paper(fig, "turndyn_operator_similarity", dir=OUT_DIR)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Per-cell CSVs (low-level data behind the aggregates)
    # ------------------------------------------------------------------
    with open(out / "turndyn_cells.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "arm",
                "model",
                "layer",
                "turn",
                "mapping",
                "n",
                "r2",
                "null_mean",
                "null_hi",
                "null_max",
                "r2_folds",
            ]
        )
        for key in sorted(k for k in p if k.startswith("cells_")):
            cells = p[key]
            for layer, byturn in cells["per_turn"].items():
                for t, cell in sorted(byturn.items(), key=lambda kv: int(kv[0])):
                    for mapping in ("ctx", "pfx"):
                        a = cell.get(mapping, {})
                        if a.get("status") != "computed":
                            continue
                        w.writerow(
                            [
                                cells["arm"],
                                cells["model"],
                                layer,
                                t,
                                mapping,
                                cell["n"],
                                a["r2"],
                                a["null_mean"],
                                a["null_hi"],
                                a["null_max"],
                                ";".join(f"{x:.6f}" for x in a.get("r2_folds", [])),
                            ]
                        )
    with open(out / "turndyn_transfer.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "model", "layer", "source_turn", "target_turn", "r2"])
        for key in sorted(k for k in p if k.startswith("transfer_")):
            tr = p[key]
            for cell, v in sorted(tr["r2"].items()):
                i, j = cell.split("->")
                w.writerow([tr["arm"], tr["model"], tr["layer"], i, j, v])
    print("figures + CSVs written to", out)


if __name__ == "__main__":
    main()
