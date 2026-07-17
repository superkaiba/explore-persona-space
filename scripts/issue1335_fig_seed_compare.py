"""Figures for the #1335 seed43-gap-rungs follow-up round (GEN_SEED 42 vs 43).

Reads the committed seed-42 matched fits (eval_results/issue_1335/) and the
seed-43 round fits (eval_results/issue_1335/seed43-gap-rungs/), and writes two
figures to <main-root>/figures/issue_1335/seed43-gap-rungs/:

1. seed_compare_rungs      — per-rung matched held-out R² (layer 19, ctx arm),
                             seed 42 vs seed 43, both models, CI whiskers +
                             per-draw dots.
2. seed_compare_deltas     — left: gap G and framing delta per model per seed,
                             95% CIs (joint-draws within seed); right: the
                             per-persona fiction-endpoint matched values behind
                             the r7 mean, seed 42 vs seed 43.

CI convention in these figures: per-rung whiskers are the draw-mean of the five
matched-draw 1,000-draw group-bootstrap CIs; the r7 rung-mean whisker combines
the four per-persona SEs as independent (variance sum / 4). Gap/framing CIs are
the fit's joint-draw CIs, copied from seed_comparison.json / ladder_summary.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

WT = Path(__file__).resolve().parents[1]
S42_DIR = WT / "eval_results/issue_1335"
S43_DIR = WT / "eval_results/issue_1335/seed43-gap-rungs"
# figures land on the MAIN checkout (committed to main per analyzer Step 3)
OUT_DIR = Path("/home/thomasjiralerspong/explore-persona-space/figures/issue_1335/seed43-gap-rungs")

RUNGS = ["r1_qa_oneline", "r3_persona", "r4_fictionframe", "r7_endpoint"]
RUNG_LABELS = ["one-line\nQ&A", "persona-\ndescribed", "fiction-framed\nQ&A", "fiction\nendpoint"]
PERSONAS = ["Wren", "HELIOS", "Dana", "Vex"]
MODELS = ["base", "instruct"]


def _matched(path: Path) -> dict:
    d = json.loads(path.read_text())
    draws = d["draws"]
    per_draw = [dr["r2_headline"] for dr in draws]
    ci_lo = float(np.mean([dr["group_bootstrap_l19"]["ci_lo"] for dr in draws]))
    ci_hi = float(np.mean([dr["group_bootstrap_l19"]["ci_hi"] for dr in draws]))
    return {
        "mean": float(d["r2_headline_mean"]),
        "per_draw": per_draw,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_min": int(d["n_min"]),
    }


def _rung_value(seed_dir: Path, rung: str, model: str) -> dict:
    if rung != "r7_endpoint":
        return _matched(seed_dir / f"matched_{rung}__{model}__ctx.json")
    per = {p: _matched(seed_dir / f"matched_r7_endpoint__{model}__{p}__ctx.json") for p in PERSONAS}
    mean = float(np.mean([per[p]["mean"] for p in PERSONAS]))
    # combine the four per-persona SEs as independent estimates of the mean
    se2 = sum(((per[p]["ci_hi"] - per[p]["ci_lo"]) / 2 / 1.96) ** 2 for p in PERSONAS)
    se = math.sqrt(se2) / len(PERSONAS)
    return {
        "mean": mean,
        "per_draw": [per[p]["mean"] for p in PERSONAS],  # per-persona means as the low-level dots
        "ci_lo": mean - 1.96 * se,
        "ci_hi": mean + 1.96 * se,
        "n_min": per["Wren"]["n_min"],
        "per_persona": per,
    }


def fig_rungs(data: dict, colors: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    x = np.arange(len(RUNGS))
    off = {"42": -0.13, "43": +0.13}
    for ax, model in zip(axes, MODELS):
        for si, seed in enumerate(("42", "43")):
            c = colors[si]
            means = [data[seed][model][r]["mean"] for r in RUNGS]
            lo = [means[i] - data[seed][model][r]["ci_lo"] for i, r in enumerate(RUNGS)]
            hi = [data[seed][model][r]["ci_hi"] - means[i] for i, r in enumerate(RUNGS)]
            ax.errorbar(
                x + off[seed],
                means,
                yerr=[lo, hi],
                fmt="o",
                color=c,
                capsize=3,
                markersize=7,
                linewidth=1.4,
                label=f"generation seed {seed}",
                zorder=3,
            )
            for i, r in enumerate(RUNGS):
                pts = data[seed][model][r]["per_draw"]
                ax.scatter(
                    [x[i] + off[seed]] * len(pts),
                    pts,
                    s=10,
                    facecolors="white",
                    edgecolors=c,
                    linewidths=0.8,
                    zorder=4,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(RUNG_LABELS)
        ax.set_title(f"{model} model", loc="left")
        ax.set_ylim(0.2, 0.5)
    axes[0].set_ylabel("Held-out R² (layer 19, context arm, matched n)")
    axes[1].legend(loc="upper right")
    savefig_paper(fig, "seed_compare_rungs", dir=OUT_DIR)
    plt.close(fig)


def fig_deltas(data: dict, sc: dict, colors: list[str]) -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # left: gap G + framing delta, per model per seed, joint-draw CIs
    quantities = [
        ("base", "gap"),
        ("base", "framing"),
        ("instruct", "gap"),
        ("instruct", "framing"),
    ]
    labels = ["base\ngap G", "base\nframing Δ", "instruct\ngap G", "instruct\nframing Δ"]
    x = np.arange(len(quantities))
    off = {"42": -0.13, "43": +0.13}
    for si, seed in enumerate(("42", "43")):
        c = colors[si]
        vals, los, his = [], [], []
        for model, q in quantities:
            pm = sc["per_model"][model]
            blk = (
                (
                    pm["seed42_reference"]["gap_G"]
                    if q == "gap"
                    else pm["seed42_reference"]["framing"]
                )
                if seed == "42"
                else (pm["gap_G"] if q == "gap" else pm["framing"])
            )
            vals.append(blk["value"])
            los.append(blk["value"] - blk["ci_lo"])
            his.append(blk["ci_hi"] - blk["value"])
        axl.errorbar(
            x + off[seed],
            vals,
            yerr=[los, his],
            fmt="o",
            color=c,
            capsize=3,
            markersize=7,
            linewidth=1.4,
            label=f"generation seed {seed}",
            zorder=3,
        )
    axl.axhline(0.0, color="0.6", linewidth=0.8, zorder=1)
    axl.set_xticks(x)
    axl.set_xticklabels(labels)
    axl.set_ylabel("R² difference (matched n, layer 19, ctx)")
    axl.set_title("Gap and framing delta by seed", loc="left")
    axl.legend(loc="upper right")

    # right: per-persona fiction-endpoint matched values (the low-level view)
    xp = np.arange(len(PERSONAS))
    moff = {"base": -0.06, "instruct": +0.06}
    for si, seed in enumerate(("42", "43")):
        c = colors[si]
        for model, marker in (("base", "o"), ("instruct", "s")):
            per = data[seed][model]["r7_endpoint"]["per_persona"]
            vals = [per[p]["mean"] for p in PERSONAS]
            los = [per[p]["mean"] - per[p]["ci_lo"] for p in PERSONAS]
            his = [per[p]["ci_hi"] - per[p]["mean"] for p in PERSONAS]
            axr.errorbar(
                xp + off[seed] + moff[model],
                vals,
                yerr=[los, his],
                fmt=marker,
                color=c,
                capsize=2,
                markersize=6,
                linewidth=1.1,
                label=f"{model}, seed {seed}",
                zorder=3,
            )
    axr.set_xticks(xp)
    axr.set_xticklabels(PERSONAS)
    axr.set_ylabel("Held-out R² (layer 19, context arm, matched n)")
    axr.set_title("Fiction endpoint per persona", loc="left")
    axr.legend(loc="upper right", ncol=2, fontsize=8)
    savefig_paper(fig, "seed_compare_deltas", dir=OUT_DIR)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    colors = paper_palette_blog(2)
    sc = json.loads((S43_DIR / "seed_comparison.json").read_text())
    data: dict = {"42": {}, "43": {}}
    for seed, d in (("42", S42_DIR), ("43", S43_DIR)):
        for model in MODELS:
            data[seed][model] = {r: _rung_value(d, r, model) for r in RUNGS}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_rungs(data, colors)
    fig_deltas(data, sc, colors)
    # sanity: the rung means match seed_comparison.json (fail loud on drift)
    for model in MODELS:
        rv = sc["per_model"][model]["rung_values_matched_ctx"]
        assert abs(data["43"][model]["r1_qa_oneline"]["mean"] - rv["r1_qa_oneline"]) < 1e-9
        assert abs(data["43"][model]["r7_endpoint"]["mean"] - rv["r7_endpoint_mean"]) < 1e-9
    print(f"[i1335-fig-seed-compare] wrote 2 figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
