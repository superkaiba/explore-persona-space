"""Issue #825 `naturalistic-single-turn` figures (Track-S format contrast).

Three figures from eval_results/issue_825/naturalistic-single-turn/:

1. ``nat_s_layer_curves`` — per model (instruct | pretrained), held-out R^2
   across all 28 layers for the chat-template vs naturalistic ``User:``/
   ``Assistant:`` render, both refit on the SHARED n=4,724 conversations
   (format_contrast.json), with the shuffled-pairing null band and the four
   frozen layers marked.
2. ``nat_s_l19_delta`` — paired naturalistic-minus-chat R^2 delta per frozen
   layer per model: pooled-global bootstrap statistic with its 1,000-draw
   conversation-level CI (filled, whiskers) alongside the fold-mean
   ``delta_obs`` statistic (open diamond).
3. ``nat_s_format_scatter`` — low-level per-unit view: per-layer scatter of
   naturalistic vs chat R^2 (28 points per model), identity line, frozen
   layers labeled.

CLI:
  uv run python scripts/issue825_naturalistic_s_figures.py \
      [--in-dir eval_results/issue_825/naturalistic-single-turn] \
      [--fig-dir figures/issue_825]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

FROZEN = (14, 18, 19, 26)
HEADLINE = 19
MODELS = ("instruct", "pretrained")
MODEL_LABEL = {"instruct": "Qwen2.5-7B-Instruct", "pretrained": "Qwen2.5-7B (pretrained)"}
NULL_FILE_OF = {"instruct": "nulls_S1N.json", "pretrained": "nulls_S2N.json"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--in-dir", type=Path, default=Path("eval_results/issue_825/naturalistic-single-turn")
    )
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_825"))
    return ap.parse_args()


def fig_layer_curves(contrast: dict, in_dir: Path, fig_dir: Path) -> None:
    c_chat = paper_palette_role("primary")
    c_nat = paper_palette_role("accent")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, layout="none")
    layers = np.arange(28)
    for ax, model in zip(axes, MODELS, strict=True):
        pm = contrast["per_model"][model]
        chat = np.asarray(pm["r2_per_layer_obs"]["chat"])
        nat = np.asarray(pm["r2_per_layer_obs"]["naturalistic"])
        nulls = json.loads((in_dir / NULL_FILE_OF[model]).read_text())
        null_mat = np.asarray(nulls["null_matrix"])  # (20 draws, 28 layers)
        for fl in FROZEN:
            ax.axvline(fl, color="#D8D4CC", lw=1.0 if fl != HEADLINE else 1.6, zorder=0)
        ax.fill_between(
            layers,
            null_mat.min(axis=0),
            null_mat.max(axis=0),
            color="#BBB7AF",
            alpha=0.45,
            lw=0,
            label="shuffled-pairing null (20 draws)",
        )
        ax.plot(layers, chat, color=c_chat, lw=2.0, label="chat template")
        ax.plot(layers, nat, color=c_nat, lw=2.0, label="naturalistic User:/Assistant:")
        nat_higher = nat[HEADLINE] >= chat[HEADLINE]
        ax.text(
            HEADLINE,
            nat[HEADLINE],
            f"  L19 {nat[HEADLINE]:.3f}",
            fontsize=8.5,
            color=c_nat,
            va="bottom" if nat_higher else "top",
        )
        ax.text(
            HEADLINE,
            chat[HEADLINE],
            f"  L19 {chat[HEADLINE]:.3f}",
            fontsize=8.5,
            color=c_chat,
            va="top" if nat_higher else "bottom",
        )
        ax.set_title(MODEL_LABEL[model], loc="left", fontsize=11)
        ax.set_xlabel("layer")
    axes[0].set_ylabel("held-out R² (context → answer profile)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.text(
        0.01,
        0.99,
        "The single-turn context→answer map survives removal of the chat template",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
    )
    fig.text(
        0.01,
        0.935,
        "Both formats refit on the shared n=4,724 LMSYS conversations; "
        "vertical lines mark frozen layers 14/18/19/26 (19 = headline)",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#5A5A5A",
    )
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.07, right=0.985, wspace=0.08)
    savefig_paper(fig, "nat_s_layer_curves", dir=fig_dir)
    plt.close(fig)


def fig_l19_delta(contrast: dict, fig_dir: Path) -> None:
    colors = {"instruct": paper_palette_role("primary"), "pretrained": paper_palette_role("accent")}
    offsets = {"instruct": -0.13, "pretrained": +0.13}
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="none")
    xs = np.arange(len(FROZEN))
    ax.axhline(0.0, color="#9A968E", lw=1.0, zorder=0)
    for model in MODELS:
        rows = contrast["per_model"][model]["paired_delta_frozen_layers"]
        x = xs + offsets[model]
        pooled = np.array([rows[str(fl)]["delta_pooled_global_obs"] for fl in FROZEN])
        lo = np.array([rows[str(fl)]["ci_lo"] for fl in FROZEN])
        hi = np.array([rows[str(fl)]["ci_hi"] for fl in FROZEN])
        fold = np.array([rows[str(fl)]["delta_obs"] for fl in FROZEN])
        ax.errorbar(
            x,
            pooled,
            yerr=np.vstack([pooled - lo, hi - pooled]),
            fmt="o",
            color=colors[model],
            ms=6,
            capsize=3,
            lw=1.6,
            label=f"{MODEL_LABEL[model]} — pooled bootstrap (CI)",
        )
        ax.scatter(
            x,
            fold,
            marker="D",
            s=34,
            facecolors="none",
            edgecolors=colors[model],
            linewidths=1.4,
            label=f"{MODEL_LABEL[model]} — fold-mean",
            zorder=3,
        )
        i19 = FROZEN.index(HEADLINE)
        ax.text(
            x[i19] + 0.05,
            pooled[i19],
            f"{pooled[i19]:+.3f}",
            fontsize=8.5,
            color=colors[model],
            va="center",
        )
    ax.set_xticks(xs, [f"layer {fl}" for fl in FROZEN])
    ax.set_ylabel("R² delta: naturalistic - chat (paired, shared n=4,724)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(
        "Format effect crosses over by model: small chat advantage for instruct,\n"
        "small naturalistic advantage for the pretrained base",
        loc="left",
        pad=14,
        fontsize=12,
    )
    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.11, right=0.97)
    savefig_paper(fig, "nat_s_l19_delta", dir=fig_dir)
    plt.close(fig)


def fig_format_scatter(contrast: dict, fig_dir: Path) -> None:
    colors = {"instruct": paper_palette_role("primary"), "pretrained": paper_palette_role("accent")}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharex=True, sharey=True, layout="none")
    all_vals = np.concatenate(
        [
            np.asarray(contrast["per_model"][m]["r2_per_layer_obs"][f])
            for m in MODELS
            for f in ("chat", "naturalistic")
        ]
    )
    lims = (all_vals.min() - 0.02, all_vals.max() + 0.02)
    for ax, model in zip(axes, MODELS, strict=True):
        pm = contrast["per_model"][model]
        chat = np.asarray(pm["r2_per_layer_obs"]["chat"])
        nat = np.asarray(pm["r2_per_layer_obs"]["naturalistic"])
        ax.plot(lims, lims, ls="--", color="#9A968E", lw=1.0, zorder=0)
        ax.scatter(chat, nat, s=26, color=colors[model], alpha=0.85, zorder=2)
        for fl in FROZEN:
            ax.scatter(
                chat[fl],
                nat[fl],
                s=54,
                facecolors="none",
                edgecolors="#1A1A1A",
                linewidths=1.2,
                zorder=3,
            )
            ax.text(chat[fl] + 0.004, nat[fl] - 0.004, f"L{fl}", fontsize=8, va="top")
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.set_title(MODEL_LABEL[model], loc="left", fontsize=11)
        ax.set_xlabel("held-out R², chat template")
    axes[0].set_ylabel("held-out R², naturalistic render")
    fig.text(
        0.01,
        0.99,
        "Per-layer view: instruct layers sit below the identity line, pretrained layers above",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
    )
    fig.text(
        0.01,
        0.935,
        "One point per layer (28); circled + labeled = frozen layers; dashed = identity",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#5A5A5A",
    )
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.08, right=0.985, wspace=0.08)
    savefig_paper(fig, "nat_s_format_scatter", dir=fig_dir)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    set_paper_style("blog")
    contrast = json.loads((args.in_dir / "format_contrast.json").read_text())
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig_layer_curves(contrast, args.in_dir, args.fig_dir)
    fig_l19_delta(contrast, args.fig_dir)
    fig_format_scatter(contrast, args.fig_dir)
    print(f"[figures] wrote 3 figures to {args.fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
