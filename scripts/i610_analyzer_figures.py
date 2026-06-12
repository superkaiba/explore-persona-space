"""Analyzer figures for task #610 (default-assistant shielding: dose or identity?).

Reads the registered analysis JSON written by
``explore_persona_space.experiments.default_dose_610.analyze`` and renders
blog-style clean-result figures into ``figures/issue_610/``.

Usage (VM, CPU):
    uv run python scripts/i610_analyzer_figures.py \
        --analysis .claude/worktrees/issue-610/eval_results/issue_610/analysis/analysis.json \
        --out-dir figures/
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

SEEDS = ["42", "137", "219"]

ARM_TICKS = {
    "with": "With-default mix\n(default trained as negative)",
    "without": "No-default mix\n(default never trained)",
}


def _new_fig(figsize):
    """Figure with constrained layout off (figure-level titles + manual margins)."""
    mpl.rcParams["figure.constrained_layout.use"] = False
    return plt.subplots(figsize=figsize)


def _finish(fig, title: str, subtitle: str, *, top=0.84, bottom=0.16, left=0.15, right=0.97, **kw):
    fig.text(0.02, 0.965, title, fontsize=13, fontweight="semibold", color="#1A1A1A", ha="left")
    fig.text(0.02, 0.915, subtitle, fontsize=9.5, color="#5A5A5A", ha="left")
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, **kw)


def fig_hero(a: dict, out_dir: Path) -> None:
    head = a["headline"]
    d_with = a["d_with_by_seed"]
    d_without = a["d_without_by_seed"]
    c_base = paper_palette_role("baseline")
    c_prim = paper_palette_role("primary")

    fig, ax = _new_fig((6.5, 4.8))

    id_thr = head["identity_threshold"]
    dose_thr = head["dose_threshold"]
    ax.axhspan(-0.27, id_thr, color="#8FBF9F", alpha=0.18, lw=0)
    ax.axhspan(dose_thr, 0.06, color="#D98B7E", alpha=0.18, lw=0)
    ax.axhline(0.0, color="#7A7A7A", lw=0.8, ls=":")
    ax.text(
        0.985, id_thr - 0.008, "identity zone", ha="right", va="top", fontsize=9, color="#3F7050"
    )
    ax.text(
        0.985, dose_thr + 0.006, "dose zone", ha="right", va="bottom", fontsize=9, color="#9C4A3C"
    )
    ax.text(
        0.985, 0.005, "untrained-panel median", ha="right", va="bottom", fontsize=8, color="#7A7A7A"
    )

    xw, xn = 0.3, 0.7
    label_dy = {"42": 0.004, "137": -0.009, "219": 0.004}
    for seed in SEEDS:
        yw, yn = d_with[seed], d_without[seed]
        ax.scatter([xw], [yw], s=85, color=c_base, zorder=3)
        ax.scatter([xn], [yn], s=85, color=c_prim, zorder=3)
        ax.text(
            xw - 0.045,
            yw + label_dy[seed],
            f"seed {seed}",
            ha="right",
            va="center",
            fontsize=8,
            color="#5A5A5A",
        )
        ax.text(
            xn + 0.045,
            yn + label_dy[seed],
            f"seed {seed}",
            ha="left",
            va="center",
            fontsize=8,
            color="#5A5A5A",
        )

    ax.hlines(head["median_with"], xw - 0.08, xw + 0.08, color=c_base, lw=2.2, zorder=4)
    ax.hlines(head["median_without"], xn - 0.08, xn + 0.08, color=c_prim, lw=2.2, zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.27, 0.06)
    ax.set_xticks([xw, xn])
    ax.set_xticklabels([ARM_TICKS["with"], ARM_TICKS["without"]], fontsize=9)
    ax.set_ylabel("Centered, implant-normalized\ndefault-context marker shift")
    _finish(
        fig,
        "Removing the default assistant's negative rows leaves its shielding intact",
        f"Never-trained default median {head['median_without']:+.3f} vs trained-default median "
        f"{head['median_with']:+.3f}; identity zone ≤ {id_thr:+.3f} (3 seeds per arm)",
    )
    savefig_paper(fig, "issue_610/hero_default_dose_strip", dir=str(out_dir))
    plt.close(fig)


def fig_hero_raw(a: dict, out_dir: Path) -> None:
    """Raw counterpart of the hero: uncentered, unnormalized nat-space reads."""
    ts = a["exploratory"]["three_space_terminal"]
    c_base = paper_palette_role("baseline")
    c_prim = paper_palette_role("primary")

    fig, ax = _new_fig((6.5, 4.4))
    xw, xn = 0.3, 0.7
    label_dy = {
        "with": {"42": 0.04, "137": 0.0, "219": -0.04},
        "without": {"42": 0.04, "137": 0.0, "219": -0.04},
    }
    for seed in SEEDS:
        yw = ts["with"][seed]["qwen_default"]["delta_logp_mean"]
        yn = ts["without"][seed]["qwen_default"]["delta_logp_mean"]
        ax.scatter([xw], [yw], s=85, color=c_base, zorder=3)
        ax.scatter([xn], [yn], s=85, color=c_prim, zorder=3)
        ax.text(
            xw - 0.045,
            yw + label_dy["with"][seed],
            f"seed {seed}",
            ha="right",
            va="center",
            fontsize=8,
            color="#5A5A5A",
        )
        ax.text(
            xn + 0.045,
            yn + label_dy["without"][seed],
            f"seed {seed}",
            ha="left",
            va="center",
            fontsize=8,
            color="#5A5A5A",
        )
    ax.axhline(0.0, color="#7A7A7A", lw=0.8, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.2)
    ax.set_xticks([xw, xn])
    ax.set_xticklabels([ARM_TICKS["with"], ARM_TICKS["without"]], fontsize=9)
    ax.set_ylabel("Raw default-context marker log-prob\ngain, trained − base (nats)")
    _finish(
        fig,
        "Raw (uncentered, unnormalized) default-context reads",
        "Same data as the hero before normalization/centering; villain implants 8.8–9.3 (with) / 8.7–10.0 (without) nats",
    )
    savefig_paper(fig, "issue_610/hero_default_dose_strip_raw", dir=str(out_dir))
    plt.close(fig)


def fig_trajectory(a: dict, out_dir: Path) -> None:
    traj = a["exploratory"]["trajectory_centered_default"]
    c_base = paper_palette_role("baseline")
    c_prim = paper_palette_role("primary")

    fig, ax = _new_fig((6.5, 4.4))
    for arm, color, lbl in (
        ("with", c_base, "With-default mix (parent)"),
        ("without", c_prim, "No-default mix (new)"),
    ):
        for i, seed in enumerate(SEEDS):
            pts = traj[arm][seed]
            fr = sorted(float(k) for k in pts)
            ys = [pts[f"{f:.2f}"] for f in fr]
            ax.plot(
                fr,
                ys,
                marker="o",
                ms=4,
                lw=1.4,
                color=color,
                alpha=0.85,
                label=lbl if i == 0 else None,
            )
    ax.axhline(0.0, color="#7A7A7A", lw=0.8, ls=":")
    ax.set_xlabel("Training fraction (of 63 matched optimizer steps)")
    ax.set_ylabel("Centered, implant-normalized\ndefault-context marker shift")
    ax.legend(loc="upper right", fontsize=9)
    _finish(
        fig,
        "Both arms walk the same path to the same floor",
        "Per-seed default-context shift across all 6 shared checkpoints (3 seeds per arm)",
        bottom=0.13,
    )
    savefig_paper(fig, "issue_610/trajectory_default_centered", dir=str(out_dir))
    plt.close(fig)


def fig_sanity(a: dict, out_dir: Path) -> None:
    san = a["sanity"]["per_persona"]
    jr = a["sanity"]["journalist_trained_read"]
    band = a["headline"]["band"]
    c_base = paper_palette_role("baseline")
    c_prim = paper_palette_role("primary")
    c_neut = paper_palette_role("neutral")

    fig, ax = _new_fig((6.5, 4.6))
    names = ["bartender", "french_person", "dictator"]
    for i, p in enumerate(names):
        w = statistics.median(san[p]["with_by_seed"].values())
        n = statistics.median(san[p]["without_by_seed"].values())
        ax.plot([i, i], [w, n], color="#9A9A9A", lw=1.2, zorder=2)
        ax.scatter([i], [w], s=80, color=c_base, zorder=3)
        ax.scatter([i], [n], s=80, color=c_prim, zorder=3)
        ax.hlines(
            [w - 2 * band, w + 2 * band],
            i - 0.18,
            i + 0.18,
            color=c_neut,
            lw=0.9,
            ls="--",
            zorder=1,
        )
    i = len(names)
    prec = jr["ctrl_precedent_recomputed"]
    med = statistics.median(jr["without_by_seed"].values())
    ax.plot([i, i], [prec, med], color="#9A9A9A", lw=1.2, zorder=2)
    ax.scatter([i], [prec], s=80, color=c_base, zorder=3)
    ax.scatter([i], [med], s=80, color=c_prim, zorder=3)
    ax.hlines(
        [prec - 2 * band, prec + 2 * band],
        i - 0.18,
        i + 0.18,
        color=c_neut,
        lw=0.9,
        ls="--",
        zorder=1,
    )

    ax.axhline(0.0, color="#7A7A7A", lw=0.8, ls=":")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.21, 0.10)
    ax.set_xticks(range(4))
    ax.set_xticklabels(
        ["Bartender", "French person", "Dictator", "Journalist\n(new negative)"], fontsize=9
    )
    ax.set_ylabel("Centered normalized shift\n(median over 3 seeds)")
    ax.scatter([], [], s=80, color=c_base, label="With-default arm (journalist: parent precedent)")
    ax.scatter([], [], s=80, color=c_prim, label="No-default arm")
    ax.legend(loc="lower left", fontsize=8)
    _finish(
        fig,
        "Sanity personas land where the parent left them",
        "Dashed ticks: ±2× the 0.033 noise band around the comparison value; all four reads pass",
    )
    savefig_paper(fig, "issue_610/sanity_dumbbells", dir=str(out_dir))
    plt.close(fig)


def fig_per_persona_strip(a: dict, out_dir: Path) -> None:
    strip_w = a["exploratory"]["per_persona_centered_strip_with"]
    strip_n = a["exploratory"]["per_persona_centered_strip_without"]
    c_base = paper_palette_role("baseline")
    c_prim = paper_palette_role("primary")

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, strip, color, label in (
        (axes[0], strip_w, c_base, "With-default mix (parent)"),
        (axes[1], strip_n, c_prim, "No-default mix"),
    ):
        meds = {p: statistics.median(v.values()) for p, v in strip.items()}
        order = sorted(meds, key=meds.get)
        for i, p in enumerate(order):
            vals = list(strip[p].values())
            ax.scatter([i] * len(vals), vals, s=14, color=color, alpha=0.55, lw=0)
        for marker, persona, nm in (
            ("*", "qwen_default", "Default assistant"),
            ("D", "assistant", "Assistant persona"),
            ("s", "journalist", "Journalist"),
        ):
            if persona in meds:
                i = order.index(persona)
                ax.scatter(
                    [i],
                    [meds[persona]],
                    s=150 if marker == "*" else 70,
                    marker=marker,
                    color="#1A1A1A",
                    zorder=4,
                    label=nm,
                )
        ax.axhline(0.0, color="#7A7A7A", lw=0.8, ls=":")
        ax.set_xticks([])
        ax.set_xlabel(f"{label} — personas sorted by median shift", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Centered normalized shift")
    fig.text(
        0.02,
        0.96,
        "The default assistant sits at the panel floor in both arms",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
        ha="left",
    )
    fig.text(
        0.02,
        0.905,
        "Every evaluated persona (3 seeds each); black markers: default assistant, assistant persona, journalist",
        fontsize=9.5,
        color="#5A5A5A",
        ha="left",
    )
    fig.subplots_adjust(top=0.85, bottom=0.10, left=0.06, right=0.98, wspace=0.07)
    savefig_paper(fig, "issue_610/per_persona_centered_strip", dir=str(out_dir))
    plt.close(fig)


def fig_three_space(a: dict, out_dir: Path) -> None:
    ts = a["exploratory"]["three_space_terminal"]
    c_base = paper_palette_role("baseline")
    c_prim = paper_palette_role("primary")
    c_ctrl = paper_palette_role("control")

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.5))
    cols = [
        ("with", "qwen_default", c_base, "Default,\nwith-default"),
        ("without", "qwen_default", c_prim, "Default,\nno-default"),
        ("without", "assistant", c_ctrl, "Assistant persona,\nno-default"),
    ]
    panels = [
        ("delta_logp_mean", "Δ log P(marker), trained − base (nats)", "Log-prob (primary)"),
        ("delta_margin_mean", "Δ (z_marker − z_eos) (logits)", "EOS margin (secondary)"),
        ("delta_p_mean", "Δ P(marker), trained − base", "Probability (sanity)"),
    ]
    for ax, (field, ylab, ptitle) in zip(axes, panels):
        for x, (arm, persona, color, _) in enumerate(cols):
            ys = [ts[arm][s][persona][field] for s in SEEDS]
            ax.scatter([x] * 3, ys, s=60, color=color, zorder=3)
        ax.axhline(0.0, color="#7A7A7A", lw=0.8, ls=":")
        ax.set_xticks(range(3))
        ax.set_xticklabels([c[3] for c in cols], fontsize=8)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(ptitle, fontsize=10, loc="left")
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    fig.text(
        0.02,
        0.955,
        "The marker push on the default context agrees across all three reporting spaces",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
        ha="left",
    )
    fig.text(
        0.02,
        0.90,
        "Per-seed terminal-checkpoint reads (3 seeds per column); probability changes are ~5e-9 — far from emission",
        fontsize=9.5,
        color="#5A5A5A",
        ha="left",
    )
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.06, right=0.98, wspace=0.32)
    savefig_paper(fig, "issue_610/three_space_default_assistant", dir=str(out_dir))
    plt.close(fig)


def fig_villain(a: dict, out_dir: Path) -> None:
    v = a["exploratory"]["villain_dg_terminal"]
    c_base = paper_palette_role("baseline")
    c_prim = paper_palette_role("primary")

    fig, ax = _new_fig((6.0, 4.4))
    xw, xn = 0.3, 0.7
    for seed in SEEDS:
        ax.scatter([xw], [v["with"][seed]], s=85, color=c_base, zorder=3)
        ax.scatter([xn], [v["without"][seed]], s=85, color=c_prim, zorder=3)
        ax.text(
            xw - 0.045,
            v["with"][seed],
            f"seed {seed}",
            ha="right",
            va="center",
            fontsize=8,
            color="#5A5A5A",
        )
        ax.text(
            xn + 0.045,
            v["without"][seed],
            f"seed {seed}",
            ha="left",
            va="center",
            fontsize=8,
            color="#5A5A5A",
        )
    ax.set_xlim(0, 1)
    ax.set_xticks([xw, xn])
    ax.set_xticklabels([ARM_TICKS["with"], ARM_TICKS["without"]], fontsize=9)
    ax.set_ylabel("Villain marker implant,\ntrained − base (nats)")
    _finish(
        fig,
        "The villain implant lands in the same regime in both arms",
        "Terminal-checkpoint source gains; all six runs inside the parent's realized 8.0–10.6-nat window",
    )
    savefig_paper(fig, "issue_610/villain_implant_comparison", dir=str(out_dir))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("figures/"))
    args = ap.parse_args()

    a = json.loads(args.analysis.read_text())
    set_paper_style("blog")
    fig_hero(a, args.out_dir)
    fig_hero_raw(a, args.out_dir)
    fig_trajectory(a, args.out_dir)
    fig_sanity(a, args.out_dir)
    fig_per_persona_strip(a, args.out_dir)
    fig_three_space(a, args.out_dir)
    fig_villain(a, args.out_dir)
    print(f"figures written to {args.out_dir / 'issue_610'}")


if __name__ == "__main__":
    main()
