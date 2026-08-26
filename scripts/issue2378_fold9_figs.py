"""Figures for the #2378 v9 fold round (sim-user-regen + dana-behavior-confirm).

Round 1 (sim-user-regen): the collapsed on-policy simulated-user cell was
regenerated under an elicitation repair (rung-1 min_tokens=8 + stop-token-id
masking); own-map fits, the H4b paired real-vs-sim contrast, and the
length-matched user-pair refits are plotted from
eval_results/issue_2378/sim-user-regen/.

Round 2 (dana-behavior-confirm): the single nominal behavior family from the
causal patching round (story Dana into chat at layer 51) was re-confirmed
fresh at temperature 1.0; the original greedy-grid interval and the fresh
confirmation interval are plotted side by side with their per-question pairs,
from eval_results/issue_2378/dana-behavior-confirm/patch_summary.json and the
original eval_results/issue_2378/causal-patching-arms/patch_summary.json.

Writes figures/issue_2378/{simregen_user_pair,simregen_pair_points,
dana_confirm_forest,dana_confirm_pairs}.{png,pdf,meta.json} via savefig_paper.

Run from the issue-2378 worktree root:
    uv run python scripts/issue2378_fold9_figs.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path(__file__).resolve().parents[1]
SR = ROOT / "eval_results" / "issue_2378" / "sim-user-regen"
DC = ROOT / "eval_results" / "issue_2378" / "dana-behavior-confirm"
CP = ROOT / "eval_results" / "issue_2378" / "causal-patching-arms"
OUT = "issue_2378"

_pal = paper_palette_blog(10)
C_REAL = _pal[7]  # matches chat_user_real in issue2378_analyzer_figs.CELLS order
C_SIM = _pal[9]  # new cell: distinct color, unused by the 8 original framings


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _fold_points(ax, xpos: float, folds: list[float]) -> None:
    ax.plot(
        np.full(len(folds), xpos),
        folds,
        marker="o",
        ls="none",
        ms=4,
        mfc="white",
        mec="black",
        markeredgewidth=0.9,
        zorder=5,
    )


def fig_simregen_user_pair() -> None:
    """Aggregate panel: own-map bars (real vs sim, both reads) + length-matched pair."""
    fits = {
        (cell, arm): _load(SR / "fits" / f"{cell}__{arm}.json")
        for cell in ("chat_user_real", "chat_user_sim")
        for arm in ("context", "prefix")
    }
    pair = _load(SR / "lenmatch" / "lenmatch_user_pair.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2), width_ratios=[1.15, 1.0])

    # Panel A: full-cohort own maps, context + prefix reads.
    order = [
        ("chat_user_real", "context"),
        ("chat_user_sim", "context"),
        ("chat_user_real", "prefix"),
        ("chat_user_sim", "prefix"),
    ]
    x = np.array([0.0, 0.7, 1.9, 2.6])
    for xi, key in zip(x, order):
        d = fits[key]
        col = C_REAL if key[0] == "chat_user_real" else C_SIM
        alpha = 1.0 if key[1] == "context" else 0.4
        ax1.bar([xi], [d["pooled_r2"]], 0.55, color=col, alpha=alpha)
        _fold_points(ax1, xi, [f["r2"] for f in d["per_fold"]])
        nulls = np.concatenate([np.asarray(v) for v in d["null"]["per_fold_draws"]])
        p95 = float(np.percentile(nulls, 95))
        ax1.plot([xi - 0.33, xi + 0.33], [p95, p95], color="black", lw=0.9, ls=":")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Real\n(context)", "Sim\n(context)", "Real\n(prefix)", "Sim\n(prefix)"])
    ax1.set_ylabel("held-out R² of the own map")
    ax1.set_title("Own maps, full cohort (n=6,601)", fontsize=11)

    # Panel B: length-matched pair (identical answer-length histograms) vs control.
    xb = np.arange(2)
    w = 0.38
    cells = ["chat_user_real", "chat_user_sim"]
    for off, leg, alpha in ((-w / 2, "matched", 1.0), (+w / 2, "control", 0.35)):
        vals = [pair["cells"][c][leg]["pooled_r2"] for c in cells]
        lo = [pair["cells"][c][leg]["ci95"][0] for c in cells]
        hi = [pair["cells"][c][leg]["ci95"][1] for c in cells]
        err = np.vstack(
            [
                np.maximum(0.0, np.array(vals) - np.array(lo)),
                np.maximum(0.0, np.array(hi) - np.array(vals)),
            ]
        )
        ax2.bar(xb + off, vals, w, color=[C_REAL, C_SIM], alpha=alpha)
        ax2.errorbar(
            xb + off, vals, yerr=err, fmt="none", ecolor="black", elinewidth=1.2, capsize=2.5
        )
    ax2.set_xticks(xb)
    ax2.set_xticklabels(["Real user turns", "Simulated user turns"])
    ax2.set_ylabel("held-out R² (reduced basis, n=1,674)")
    ax2.set_title("Length-matched (solid) vs control (pale)", fontsize=11)

    fig.suptitle(
        "Regenerated simulated user turns are as mappable as real ones", y=1.08, fontsize=13
    )
    paths = savefig_paper(fig, f"{OUT}/simregen_user_pair", dir="figures/")
    plt.close(fig)

    meta_path = Path("figures") / OUT / "simregen_user_pair.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["series_annotation"] = {
        "panel_A_bars": [f"{c}__{a}" for c, a in order],
        "panel_A_lines": "per-bar: 5 open fold points + a dotted shuffled-null p95 tick",
        "panel_B_bar_groups": {
            "0": "length-matched (identical histograms, n=1,674): real, sim",
            "1": "size-matched control (natural lengths, n=1,674): real, sim",
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {paths}")


def fig_simregen_pair_points() -> None:
    """Per-unit companion: paired per-fold R² + the 200 bootstrap draws of the delta."""
    ctx_real = _load(SR / "fits" / "chat_user_real__context.json")
    ctx_sim = _load(SR / "fits" / "chat_user_sim__context.json")
    h4b = _load(SR / "ladder" / "h4b_real_vs_sim.json")
    draws = np.asarray(h4b["ceiling_delta"]["delta_draws"], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    folds = np.arange(5)
    r_real = [f["r2"] for f in ctx_real["per_fold"]]
    r_sim = [f["r2"] for f in ctx_sim["per_fold"]]
    ax1.plot(
        folds,
        r_real,
        marker="D",
        ls="none",
        ms=6,
        mfc="white",
        mec=C_REAL,
        markeredgewidth=1.5,
        label="real user turns",
    )
    ax1.plot(folds, r_sim, marker="o", ls="none", ms=6, color=C_SIM, label="simulated user turns")
    for i in folds:
        ax1.plot([i, i], [r_real[i], r_sim[i]], color="grey", lw=0.8, zorder=1)
        ax1.text(
            i + 0.08,
            (r_real[i] + r_sim[i]) / 2,
            f"+{r_sim[i] - r_real[i]:.3f}",
            fontsize=8,
            va="center",
        )
    ax1.set_xticks(folds)
    ax1.set_xticklabels([f"fold {i}" for i in folds])
    ax1.set_ylabel("held-out R² (context read)")
    ax1.legend(loc="lower right")
    ax1.set_title("Per-fold paired values (shared folds)", fontsize=11)

    ax2.hist(draws, bins=24, color=C_SIM, alpha=0.85)
    ax2.axvline(0.0, color="black", lw=1.2)
    ax2.axvline(
        float(h4b["ceiling_delta"]["point_pooled"]), color="black", lw=1.0, ls="--", alpha=0.7
    )
    ax2.set_xlabel("Δ pooled R² (sim − real), one bootstrap draw each")
    ax2.set_ylabel("draws (of 200)")
    ax2.set_title("Conversation-grouped bootstrap of the delta", fontsize=11)

    fig.suptitle(
        "Simulated turns are marginally more predictable in every fold and draw",
        y=1.08,
        fontsize=13,
    )
    paths = savefig_paper(fig, f"{OUT}/simregen_pair_points", dir="figures/")
    plt.close(fig)
    print(f"wrote {paths}")


def _dana_pairs(cells: list[dict]) -> dict[str, float]:
    null = {c["qid"]: c["f_beh"] for c in cells if c["arm"] == "null"}
    ste = {c["qid"]: c["f_beh"] for c in cells if c["arm"] == "steered"}
    return {q: ste[q] - null[q] for q in sorted(set(null) & set(ste))}


def fig_dana_confirm_forest() -> None:
    """Original greedy-grid interval vs the fresh temperature-1.0 confirmation."""
    grid = _load(CP / "patch_summary.json")["f_beh_grid"]["steered_vs_null"][
        "chat~story|Dana|b2a|lstar|steered"
    ]
    conf = _load(DC / "patch_summary.json")["f_beh_confirm"]["steered_vs_null"][
        "chat~story|Dana|b2a|lstar|steered"
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    rows = [
        ("Original greedy grid\n(12 pairs, 1 of 20 families)", grid, True),
        ("Fresh confirmation\n(12 pairs, 5 draws per side)", conf, False),
    ]
    for y, (label, d, sig) in enumerate(rows):
        col = _pal[3] if sig else "grey"
        err = [[max(0.0, d["mean_diff"] - d["ci_lo"])], [max(0.0, d["ci_hi"] - d["mean_diff"])]]
        ax.errorbar(
            [d["mean_diff"]],
            [y],
            xerr=err,
            fmt="o",
            color=col,
            ms=7,
            elinewidth=2.0,
            capsize=4,
        )
        ax.text(d["ci_hi"] + 0.02, y, f"+{d['mean_diff']:.3f}", va="center", fontsize=10)
    ax.axvline(0.0, color="black", lw=1.0)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.6, 1.6)
    ax.invert_yaxis()
    ax.set_xlabel("behavior score, steered minus matched null (fraction of anchor contrast)")
    set_title_subtitle(
        ax,
        "The nominal story-Dana behavior family does not replicate",
        "story Dana context vector into chat at layer 51; pair-clustered 95% bootstrap intervals",
    )
    paths = savefig_paper(fig, f"{OUT}/dana_confirm_forest", dir="figures/")
    plt.close(fig)
    print(f"wrote {paths}")


def fig_dana_confirm_pairs() -> None:
    """Per-question companion: paired differences behind each interval."""
    grid_cells = [
        c
        for c in _load(CP / "patch_summary.json")["f_beh_cells_grid"]
        if c["char"] == "Dana" and c["variant"] == "lstar" and c["direction"] == "b2a"
    ]
    conf_cells = _load(DC / "patch_summary.json")["f_beh_cells_confirm"]
    grid = _dana_pairs(grid_cells)
    conf = _dana_pairs(conf_cells)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    rng = np.random.default_rng(42)
    for x0, diffs, col in ((0.0, grid, _pal[3]), (1.0, conf, "grey")):
        vals = list(diffs.values())
        xs = x0 + rng.uniform(-0.10, 0.10, len(vals))
        ax.plot(xs, vals, marker="o", ls="none", ms=6, color=col, alpha=0.85)
        for xi, (q, v) in zip(xs, diffs.items()):
            ax.text(xi + 0.03, v, q.split("_")[-1], fontsize=7, va="center", color="dimgrey")
        m = float(np.mean(vals))
        ax.plot([x0 - 0.2, x0 + 0.2], [m, m], color="black", lw=2.0)
    ax.axhline(0.0, color="black", lw=0.8, ls=":")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["original greedy grid (12 pairs)", "fresh confirm (12 pairs)"])
    ax.set_ylabel("per-question steered − null difference")
    set_title_subtitle(
        ax,
        "Per-question pairs behind the two intervals",
        "each point = one question (index labeled); black bar = family mean; jitter for legibility",
    )
    paths = savefig_paper(fig, f"{OUT}/dana_confirm_pairs", dir="figures/")
    plt.close(fig)
    print(f"wrote {paths}")


def main() -> None:
    set_paper_style("blog")
    fig_simregen_user_pair()
    fig_simregen_pair_points()
    fig_dana_confirm_forest()
    fig_dana_confirm_pairs()


if __name__ == "__main__":
    main()
