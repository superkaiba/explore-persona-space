#!/usr/bin/env python
"""Figures for the issue-1345 `story-context-info-probe` round.

Three figures, all through the project plotting conventions
(``analysis.paper_plots``: ``set_paper_style("blog")`` + ``savefig_paper`` so every
figure ships PNG + PDF + a per-point ``.meta.json`` sidecar):

1. ``raw_retrieval_bars`` — raw (no fitted map) nearest-neighbour accuracy per leg
   with the chance line drawn.
2. ``cca_spectrum`` — held-out canonical correlations, story context vs chat
   context, against the shuffled-pairing spectrum.
3. ``ridge_vs_mlp_bars`` — ridge vs MLP held-out R^2 per leg with each estimator's
   own shuffled-null band.

One colour means one thing across all three: the measured/observed series and the
null/chance reference keep the same two colours everywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps must land before matplotlib/numpy import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

LEG_LABELS = {
    "story_vC_to_chat_vC": "story context -> chat context",
    "chat_vC_to_story_vC": "chat context -> story context",
    "story_vC_to_story_vA": "story context -> story answer",
    "chat_vC_to_story_vA": "chat context -> story answer",
    "chat_vC_to_chat_vA": "chat context -> chat answer",
    "story_vC_to_chat_vA": "story context -> chat answer",
    "story_vPrefix_to_chat_vPrefix": "story prefix -> chat prefix",
}
ROUND_LABELS = {
    "story_tf": "teacher-forced story",
    "story_op": "on-policy story",
}


def _load(out_dir: Path, name: str) -> dict | None:
    path = out_dir / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fig_raw_retrieval(out_dir: Path, fig_dir: Path, rounds: list[str]) -> None:
    palette = paper_palette_blog(3)
    data = {r: _load(out_dir, f"raw_retrieval_{r}.json") for r in rounds}
    data = {r: v for r, v in data.items() if v}
    if not data:
        return
    legs = [k for k in LEG_LABELS if k in next(iter(data.values()))]
    fig, axes = plt.subplots(1, len(data), figsize=(6.2 * len(data), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (rkey, payload) in zip(axes, data.items(), strict=False):
        acc = [payload[leg]["cosine"]["acc_at_k"]["1"] for leg in legs]
        chance = payload[legs[0]]["cosine"]["chance_at_k"]["1"]
        pos = np.arange(len(legs))
        ax.bar(pos, acc, color=palette[0], width=0.62, label="cosine nearest neighbour")
        ax.axhline(
            chance,
            color=palette[1],
            linestyle="--",
            linewidth=1.6,
            label=f"chance ({chance:.4f})",
        )
        for x, v in zip(pos, acc, strict=False):
            ax.text(x, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(pos)
        ax.set_xticklabels([LEG_LABELS[leg] for leg in legs], fontsize=8, rotation=28, ha="right")
        ax.set_title(
            "Raw retrieval, no fitted map, layer 19\n"
            f"{ROUND_LABELS.get(rkey, rkey)} (n={payload['n']})",
            fontsize=10,
        )
        ax.set_ylim(0, max(max(acc), chance) * 1.25 + 0.02)
    axes[0].set_ylabel("true partner is nearest neighbour\n(fraction of conversations)")
    axes[0].legend(loc="upper right", fontsize=8)
    savefig_paper(fig, "raw_retrieval_bars", dir=str(fig_dir))
    plt.close(fig)


def fig_cca(out_dir: Path, fig_dir: Path, rounds: list[str]) -> None:
    palette = paper_palette_blog(3)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    plotted = False
    for i, rkey in enumerate(rounds):
        payload = _load(out_dir, f"alignment_{rkey}.json")
        if not payload:
            continue
        cca = payload["cca_story_vC_chat_vC"]
        obs = cca["heldout_canonical_corr"]
        null = cca["heldout_canonical_corr_shuffled"]
        idx = np.arange(1, len(obs) + 1)
        ax.plot(
            idx,
            obs,
            color=palette[0],
            linestyle="-" if i == 0 else "--",
            linewidth=2.0,
            label=f"measured, {ROUND_LABELS.get(rkey, rkey)}",
        )
        ax.plot(
            idx,
            null,
            color=palette[1],
            linestyle="-" if i == 0 else "--",
            linewidth=1.6,
            label=f"shuffled pairing, {ROUND_LABELS.get(rkey, rkey)}",
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.set_xlabel("canonical component (ordered)")
    ax.set_ylabel("held-out correlation between projected pairs")
    ax.set_title("Shared linear structure, story context vs chat context")
    ax.legend(fontsize=8)
    savefig_paper(fig, "cca_spectrum", dir=str(fig_dir))
    plt.close(fig)


def fig_ridge_vs_mlp(out_dir: Path, fig_dir: Path, rounds: list[str]) -> None:
    palette = paper_palette_blog(4)
    rows = []
    for rkey in rounds:
        payload = _load(out_dir, f"nonlinear_probe_{rkey}.json")
        if not payload:
            continue
        for leg, v in payload["legs"].items():
            rows.append((rkey, leg, v))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(1.9 * len(rows) + 3.2, 4.6))
    pos = np.arange(len(rows))
    w = 0.36
    ridge = [v["ridge_r2_heldout"] for _, _, v in rows]
    mlp = [v["mlp_r2_heldout"] for _, _, v in rows]
    ax.bar(pos - w / 2, ridge, width=w, color=palette[0], label="ridge (linear)")
    ax.bar(pos + w / 2, mlp, width=w, color=palette[2], label="MLP (nonlinear)")
    for x, (_, _, v) in zip(pos, rows, strict=False):
        ax.plot(
            [x - w, x + w],
            [v["ridge_r2_null_p95"]] * 2,
            color=palette[1],
            linestyle="--",
            linewidth=1.4,
            label="shuffled-pairing null, 95th percentile" if x == 0 else None,
        )
    for x, v in zip(pos - w / 2, ridge, strict=False):
        ax.text(x, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    for x, v in zip(pos + w / 2, mlp, strict=False):
        ax.text(x, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0.0, color="0.4", linewidth=0.9)
    ax.set_xticks(pos)
    ax.set_xticklabels(
        [f"{LEG_LABELS.get(leg, leg)}\n({ROUND_LABELS.get(r, r)})" for r, leg, _ in rows],
        fontsize=8,
        rotation=18,
        ha="right",
    )
    ax.set_ylabel("held-out R-squared\n(reduced target basis)")
    ax.set_title("Nonlinear probe vs linear ridge, identical folds and inputs")
    # Headroom so the bar-value labels cannot collide with the legend.
    lo = min([*ridge, *mlp, 0.0])
    hi = max([*ridge, *mlp])
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.34 * (hi - lo))
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    savefig_paper(fig, "ridge_vs_mlp_bars", dir=str(fig_dir))
    plt.close(fig)


def fig_lambda_sweep(out_dir: Path, fig_dir: Path) -> None:
    """Held-out R^2 and retrieval accuracy against the forced ridge penalty, with the
    penalty the parent's selector actually chose marked."""
    forced = _load(out_dir, "forced_lambda_probe.json")
    align = _load(out_dir, "alignment_story_tf.json")
    if not forced:
        return
    palette = paper_palette_blog(4)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    style = {
        "story_vC_to_story_vA": (palette[0], "-", "story context -> story answer"),
        "story_vC_to_chat_vC": (palette[2], "-", "story context -> chat context"),
        "chat_vC_to_chat_vA": (palette[1], "--", "chat context -> chat answer"),
    }
    for leg, byl in forced["legs"].items():
        colour, ls, label = style.get(leg, (palette[3], ":", leg))
        lams = sorted(byl, key=lambda s: float(s.split("_")[1]))
        xs = [float(s.split("_")[1]) for s in lams]
        ax.plot(
            xs,
            [byl[s]["r2_heldout"] for s in lams],
            marker="o",
            color=colour,
            linestyle=ls,
            label=label,
        )
        ax2.plot(
            xs,
            [byl[s]["knn_at_1"] for s in lams],
            marker="o",
            color=colour,
            linestyle=ls,
            label=label,
        )
    if align:
        offsets = {"story_vC_to_story_vA": (10, -16), "chat_vC_to_chat_vA": (10, 8)}
        for leg in ("story_vC_to_story_vA", "chat_vC_to_chat_vA"):
            v = align.get(leg)
            if not v:
                continue
            colour = style[leg][0]
            lam = min(x for x in v["gcv_lambda_per_fold"] if x is not None)
            ax.plot(
                [lam],
                [v["r2_heldout"]],
                marker="X",
                markersize=12,
                color=colour,
                linestyle="none",
            )
            ax.annotate(
                f"selector's own choice: {v['r2_heldout']:.3f}",
                (lam, v["r2_heldout"]),
                textcoords="offset points",
                xytext=offsets[leg],
                fontsize=8,
                color=colour,
            )
    for a, ylab in (
        (ax, "held-out R-squared"),
        (ax2, "fraction whose true target is nearest"),
    ):
        a.set_xscale("log")
        a.set_xlabel("ridge penalty (forced, full 3584-dimension basis)")
        a.set_ylabel(ylab)
    ax.axhline(0.0, color="0.4", linewidth=0.9)
    # Legend lives on the right panel: the left panel's lower-left holds the
    # selector-choice markers and their annotations.
    ax2.legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "The story-framing collapse is a regularization-selection artifact "
        f"(n={forced['n']}, layer 19)",
        y=1.02,
    )
    savefig_paper(fig, "lambda_sweep", dir=str(fig_dir))
    plt.close(fig)


def build(out_dir: Path, fig_dir: Path, rounds: tuple[str, ...] = ("story_tf", "story_op")) -> None:
    set_paper_style("blog")
    fig_dir.mkdir(parents=True, exist_ok=True)
    present = [r for r in rounds if (out_dir / f"raw_retrieval_{r}.json").exists()]
    fig_raw_retrieval(out_dir, fig_dir, present)
    fig_cca(out_dir, fig_dir, present)
    fig_ridge_vs_mlp(out_dir, fig_dir, present)
    fig_lambda_sweep(out_dir, fig_dir)
    print(f"[figs] wrote figures to {fig_dir}", flush=True)


if __name__ == "__main__":
    build(
        _REPO_ROOT / "eval_results/issue_1345/story_context_info_probe",
        _REPO_ROOT / "figures/issue_1345/story_context_info_probe",
    )
