"""Issue #1310 analyzer figures (round 1).

Reads ONLY the committed fit-cell JSONs under eval_results/issue_1310/
(script-format run-2 @942df1bb, uncapped GCV) and
eval_results/issue_1310/onpolicy/ (prefill @9a5b63c5, GCV dof-cap 0.9) and
renders the clean-result figures:

  1. hero_l19_bars           — per-persona held-out R^2 @L19, base vs instruct,
                               both datagen regimes side by side, shuffle-null
                               band + bootstrap CIs, assistant-map ceiling refs.
  2. layer_curves            — per-persona per-layer held-out R^2 curves per
                               (regime x model), shuffle-null band per panel.
                               (The low-level layer view behind the L19 bars.)
  3. swap_control            — correct-pairing vs cross-character-swap R^2 per
                               (regime x model); missing instruct script cell
                               labeled not run.
  4. l19_null_draw_points    — raw sibling of the hero: every one of the 20
                               shuffle-null draws at L19 per cell as points,
                               with the observed value.

Stale run-1 cells (commit b131716d: instruct Vex, instruct swap) are EXCLUDED
(treated as absent). Lastpos cells are excluded everywhere (single-position X;
known-pathological under uncapped GCV in run 2).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

EV = REPO / "eval_results" / "issue_1310"
OUT = REPO / "figures" / "issue_1310"
PERSONAS = ("Wren", "HELIOS", "Dana", "Vex")
PERSONA_GLOSS = {
    "Wren": "Wren (helpful)",
    "HELIOS": "HELIOS (ship AI)",
    "Dana": "Dana (ordinary)",
    "Vex": "Vex (villain)",
}
L = 19
STALE_RUN1_COMMIT = "b131716d"
REGIMES = {"script": "", "prefill": "onpolicy/"}
REGIME_LABEL = {"script": "script-format scenes", "prefill": "prefill turns"}
MODEL_LABEL = {"base": "Qwen2.5-7B (base)", "instruct": "Qwen2.5-7B-Instruct"}
CEILING = {"base": 0.588, "instruct": 0.673}  # #825 Track-S S2 / S1 @L19


def load_cell(regime: str, model: str, persona: str) -> dict | None:
    sub = REGIMES[regime]
    tag = "onpolicy_" if regime == "prefill" else ""
    p = EV / sub / f"cells_{tag}{model}_{persona}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if d["metadata"]["git_commit"].startswith(STALE_RUN1_COMMIT):
        return None  # stale run-1 leftover — treat as absent
    fro = d["selection_symmetric"]["frozen_layer_table"][str(L)]
    boot = d["r2_bootstrap_row_frozen"][str(L)]
    return {
        "n": d["n"],
        "r2": d["r2_per_layer_obs"][L],
        "r2_per_layer": d["r2_per_layer_obs"],
        "null_mean": fro["null_mean"],
        "null_p975": fro["null_p975"],
        "ci_lo": boot["ci_lo"],
        "ci_hi": boot["ci_hi"],
    }


def load_nulls(regime: str, model: str, persona: str) -> dict | None:
    sub = REGIMES[regime]
    tag = "onpolicy_" if regime == "prefill" else ""
    p = EV / sub / f"nulls_{tag}{model}_{persona}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if d["metadata"]["git_commit"].startswith(STALE_RUN1_COMMIT):
        return None
    return {"null_matrix": np.array(d["null_matrix"]), "observed": np.array(d["observed_row"])}


def load_swap(regime: str, model: str) -> dict | None:
    sub = REGIMES[regime]
    tag = "onpolicy_" if regime == "prefill" else ""
    p = EV / sub / f"swap_{tag}{model}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if d["metadata"]["git_commit"].startswith(STALE_RUN1_COMMIT):
        return None
    return d


def fig_hero() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    colors = paper_palette_blog(4)
    c_reg = {"script": colors[0], "prefill": colors[1]}
    width = 0.36
    xs = np.arange(len(PERSONAS))
    for ax, model in zip(axes, ("base", "instruct"), strict=True):
        for j, regime in enumerate(("script", "prefill")):
            offs = (j - 0.5) * width
            for i, persona in enumerate(PERSONAS):
                cell = load_cell(regime, model, persona)
                x = xs[i] + offs
                if cell is None:
                    ax.text(
                        x,
                        0.012,
                        "not run",
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="0.45",
                    )
                    continue
                ax.bar(
                    x,
                    cell["r2"],
                    width=width * 0.92,
                    color=c_reg[regime],
                    label=REGIME_LABEL[regime] if i == 0 else None,
                    zorder=3,
                )
                # bootstrap 95% CI drawn as a segment (the bootstrap point
                # estimate can sit a hair off the pooled observed R^2, so a
                # symmetric yerr around obs would go negative)
                ax.vlines(x, cell["ci_lo"], cell["ci_hi"], color="0.25", lw=1.2, zorder=4)
                # shuffle-null 97.5th percentile tick for this cell
                ax.plot(
                    [x - width * 0.46, x + width * 0.46],
                    [cell["null_p975"]] * 2,
                    color="0.15",
                    lw=1.0,
                    ls=":",
                    zorder=5,
                )
        ax.axhline(0, color="0.3", lw=0.8)
        ax.axhline(
            CEILING[model],
            color="0.45",
            lw=1.0,
            ls="--",
            label="assistant map (same model)" if model == "base" else None,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([PERSONA_GLOSS[p] for p in PERSONAS], fontsize=9)
        ax.set_title(MODEL_LABEL[model], fontsize=12, pad=10, loc="left", fontweight="semibold")
    axes[0].set_ylabel("held-out R² at layer 19")
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D

    handles.append(Line2D([], [], color="0.15", lw=1.0, ls=":"))
    labels.append("shuffle-null 97.5th pct")
    axes[0].legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.86), fontsize=9)
    savefig_paper(fig, "hero_l19_bars", dir=OUT)
    plt.close(fig)


def fig_layer_curves() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True, sharey="row")
    colors = paper_palette_blog(4)
    for r, regime in enumerate(("script", "prefill")):
        for c, model in enumerate(("base", "instruct")):
            ax = axes[r, c]
            null_all = []
            for i, persona in enumerate(PERSONAS):
                cell = load_cell(regime, model, persona)
                nulls = load_nulls(regime, model, persona)
                if cell is None or nulls is None:
                    continue
                ax.plot(
                    range(28),
                    cell["r2_per_layer"],
                    color=colors[i],
                    lw=1.6,
                    label=PERSONA_GLOSS[persona],
                )
                null_all.append(nulls["null_matrix"])
            if null_all:
                nm = np.concatenate(null_all, axis=0)  # (draws*personas, 28)
                ax.fill_between(
                    range(28),
                    nm.min(axis=0),
                    nm.max(axis=0),
                    color="0.55",
                    alpha=0.35,
                    lw=0,
                    label="shuffle-null range" if (r, c) == (0, 0) else None,
                )
            ax.axhline(0, color="0.3", lw=0.7)
            ax.axvline(L, color="0.5", lw=0.8, ls=":")
            ax.set_title(
                f"{REGIME_LABEL[regime]} — {MODEL_LABEL[model]}",
                fontsize=11,
                loc="left",
                fontweight="semibold",
                pad=8,
            )
            if r == 1:
                ax.set_xlabel("layer")
            if c == 0:
                ax.set_ylabel("held-out R²")
    # display clip: base-Dana script mid layers hit the uncapped-GCV pathology
    # (obs down to -7); the caption discloses the clip.
    axes[0, 0].set_ylim(-0.65, 0.45)
    axes[1, 0].set_ylim(-0.45, 0.45)
    # annotate missing instruct script Vex
    axes[0, 1].text(
        0.985,
        0.04,
        "Vex not run (run crashed before this cell)",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="0.4",
    )
    axes[0, 0].legend(loc="upper right", fontsize=8.5, ncols=2)
    savefig_paper(fig, "layer_curves", dir=OUT)
    plt.close(fig)


def fig_swap() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    colors = paper_palette_blog(4)
    conds = [
        ("script", "base"),
        ("prefill", "base"),
        ("script", "instruct"),
        ("prefill", "instruct"),
    ]
    xs = np.arange(len(conds))
    width = 0.36
    for i, (regime, model) in enumerate(conds):
        d = load_swap(regime, model)
        if d is None:
            ax.text(
                xs[i],
                0.012,
                "not run",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=9,
                color="0.45",
            )
            continue
        ax.bar(
            xs[i] - width / 2,
            d["r2_correct"],
            width=width * 0.94,
            color=colors[2],
            label="correct character" if i == 0 else None,
            zorder=3,
        )
        ax.bar(
            xs[i] + width / 2,
            d["r2_swap"],
            width=width * 0.94,
            color=colors[3],
            label="swapped character" if i == 0 else None,
            zorder=3,
        )
        # row-bootstrap 95% CIs from the matching cells files
        for dx, cell_name in ((-width / 2, "swapctrl_correct"), (width / 2, "swap")):
            sub = REGIMES[regime]
            tag = "onpolicy_" if regime == "prefill" else ""
            p = EV / sub / f"cells_{tag}{model}_{cell_name}.json"
            if not p.exists():
                continue
            cd = json.loads(p.read_text())
            if cd["metadata"]["git_commit"].startswith(STALE_RUN1_COMMIT):
                continue
            boot = cd["r2_bootstrap_row_frozen"][str(L)]
            ax.vlines(xs[i] + dx, boot["ci_lo"], boot["ci_hi"], color="0.25", lw=1.2, zorder=4)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{REGIME_LABEL[r]}\n{MODEL_LABEL[m]}" for r, m in conds], fontsize=9)
    ax.set_ylabel("held-out R² at layer 19 (pooled)")
    ax.legend(loc="upper right", fontsize=9)
    savefig_paper(fig, "swap_control", dir=OUT)
    plt.close(fig)


def fig_null_draw_points() -> None:
    """Raw sibling of the hero: 20 shuffle-null draws @L19 per cell as points."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    colors = paper_palette_blog(4)
    c_reg = {"script": colors[0], "prefill": colors[1]}
    width = 0.36
    xs = np.arange(len(PERSONAS))
    rng = np.random.default_rng(0)
    for ax, model in zip(axes, ("base", "instruct"), strict=True):
        for j, regime in enumerate(("script", "prefill")):
            offs = (j - 0.5) * width
            for i, persona in enumerate(PERSONAS):
                nulls = load_nulls(regime, model, persona)
                cell = load_cell(regime, model, persona)
                x = xs[i] + offs
                if nulls is None or cell is None:
                    ax.text(
                        x,
                        0.005,
                        "not run",
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="0.45",
                    )
                    continue
                draws = nulls["null_matrix"][:, L]
                jit = rng.uniform(-width * 0.28, width * 0.28, size=draws.shape)
                ax.scatter(
                    x + jit,
                    draws,
                    s=9,
                    color="0.45",
                    alpha=0.75,
                    lw=0,
                    zorder=3,
                    label="shuffle-null draws (20)" if (i, j) == (0, 0) else None,
                )
                ax.scatter(
                    [x],
                    [cell["r2"]],
                    s=52,
                    color=c_reg[regime],
                    marker="D",
                    edgecolors="0.15",
                    linewidths=0.7,
                    zorder=4,
                    label=f"observed — {REGIME_LABEL[regime]}" if i == 0 else None,
                )
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([PERSONA_GLOSS[p] for p in PERSONAS], fontsize=9)
        ax.set_title(MODEL_LABEL[model], fontsize=12, pad=10, loc="left", fontweight="semibold")
    axes[0].set_ylabel("held-out R² at layer 19")
    axes[0].legend(loc="upper left", fontsize=9)
    savefig_paper(fig, "l19_null_draw_points", dir=OUT)
    plt.close(fig)


def fig_perfold() -> None:
    """Low-level per-unit view (prefill regime): per-scene-group + per-fold R^2."""
    pf_path = EV / "onpolicy" / "analyzer_perfold_l19.json"
    if not pf_path.exists():
        print("perfold JSON missing — skipping fig_perfold")
        return
    d = json.loads(pf_path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    colors = paper_palette_blog(4)
    rng = np.random.default_rng(1)
    CLIP = 1.0
    for ax, model in zip(axes, ("base", "instruct"), strict=True):
        slots = [(p, f"onpolicy_{model}_{p}") for p in PERSONAS]
        xs = np.arange(len(slots) + 2)
        for i, (persona, cid) in enumerate(slots):
            c = d["cells"][cid]
            pg = np.clip([v["r2"] for v in c["pergroup"].values()], -CLIP, CLIP)
            jit = rng.uniform(-0.26, 0.26, size=len(pg))
            ax.scatter(
                xs[i] + jit,
                pg,
                s=5,
                color="0.6",
                alpha=0.35,
                lw=0,
                zorder=2,
                label="per-scene-group R² (~300, clipped ±1)"
                if (i == 0 and model == "base")
                else None,
            )
            pf = [f["r2"] for f in c["perfold"]]
            ax.scatter(
                [xs[i]] * len(pf),
                pf,
                s=34,
                facecolors="none",
                edgecolors="0.1",
                linewidths=1.1,
                zorder=3,
                label="per-fold R² (5 folds)" if (i == 0 and model == "base") else None,
            )
            ax.scatter(
                [xs[i]],
                [c["pooled"]],
                s=54,
                color=colors[1],
                marker="D",
                edgecolors="0.15",
                linewidths=0.7,
                zorder=4,
                label="pooled (committed)" if (i == 0 and model == "base") else None,
            )
        for k, (key, lab) in enumerate((("correct", "swap: correct"), ("swap", "swap: swapped"))):
            s = d["swap"][model][key]
            pg = np.clip([v["r2"] for v in s["pergroup"].values()], -CLIP, CLIP)
            jit = rng.uniform(-0.26, 0.26, size=len(pg))
            x = xs[len(slots) + k]
            ax.scatter(x + jit, pg, s=5, color="0.6", alpha=0.35, lw=0, zorder=2)
            pf = [f["r2"] for f in s["perfold"]]
            ax.scatter(
                [x] * len(pf),
                pf,
                s=34,
                facecolors="none",
                edgecolors="0.1",
                linewidths=1.1,
                zorder=3,
            )
            ax.scatter(
                [x],
                [s["pooled"]],
                s=54,
                color=colors[2],
                marker="D",
                edgecolors="0.15",
                linewidths=0.7,
                zorder=4,
            )
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [PERSONA_GLOSS[p] for p in PERSONAS] + ["pooled correct", "pooled swapped"],
            fontsize=8.5,
            rotation=12,
        )
        ax.set_title(
            f"prefill turns — {MODEL_LABEL[model]}",
            fontsize=12,
            pad=10,
            loc="left",
            fontweight="semibold",
        )
    axes[0].set_ylabel("held-out R² at layer 19")
    axes[0].legend(loc="lower left", fontsize=8.5)
    savefig_paper(fig, "l19_perfold_pergroup_points", dir=OUT)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    OUT.mkdir(parents=True, exist_ok=True)
    fig_hero()
    fig_layer_curves()
    fig_swap()
    fig_null_draw_points()
    fig_perfold()
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()
