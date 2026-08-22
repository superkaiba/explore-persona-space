"""Clean-result figures for issue #2220 (read-write duality steering test).

Reads the committed decisive/localize/margin surfaces under
``eval_results/issue_2220/`` and renders the clean-result figures into
``figures/issue_2220/`` via ``savefig_paper`` (blog style).

Figures (one color = one DIRECTION across all figures; position is encoded by
fill: solid = answer tokens, hatched/open = context token):
  1. hero1_decisive_bars     - decisive delta-rate bars per direction x position
                               (evil, sycophancy) + selection-symmetric null band.
  2. perq_rates_decisive     - per-question rates behind the decisive aggregates
                               (low-level per-unit view; 20 questions/cell).
  3. dose_response_grid      - localize dose-response, position x behavior grid
                               at the decisive operating layer.
  4. hallucination_gate2     - localize operating-point delta-rate vs the
                               selection-symmetric null band (rig-inconclusive).
  5. margin_secondary        - teacher-forced fixed +/- pool margin (secondary DV).

Run from the issue-2220 worktree root: ``uv run python scripts/issue2220_readwrite_figs.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

# load_dotenv BEFORE any heavy import (thread-cap setdefaults are frozen at
# matplotlib/numpy import; orchestrate.env, never bare dotenv).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "eval_results").exists():
    ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
EV = ROOT / "eval_results" / "issue_2220"
# savefig_paper prepends dir="figures/" itself — a "figures/..." stem would
# double-nest (the #613 trap), so the stem carries only the issue subdir.
FIGDIR = "issue_2220"

DIRS = ["mapread_ctx", "mapread_prefix", "rb", "rawmeandiff", "shuffled", "random"]
DIR_LABEL = {
    "mapread_ctx": "Map read\n(context arm)",
    "mapread_prefix": "Map read\n(prefix arm)",
    "rb": "Persona\nvector",
    "rawmeandiff": "Raw mean-\ndifference",
    "shuffled": "Shuffled-\nlabel map",
    "random": "Random",
}
_PAL = paper_palette_blog(6)
DIR_COLOR = dict(zip(DIRS, _PAL))
POS = ["answer", "context"]
POS_LABEL = {"answer": "answer tokens", "context": "context token"}
BEH_LAYER = {"evil": 14, "sycophancy": 18}  # decisive operating layers (r_B cells)


def _load(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def _cell_key(beh: str, direction: str, position: str, layer: int, c: float) -> str:
    cs = f"{c:.1f}".replace(".", "p")
    return f"behavior{beh}__direction{direction}__position{position}__layer{layer}__c{cs}"


def _pos_bar_kwargs(position: str, color: str) -> dict:
    if position == "answer":
        return dict(color=color, edgecolor="white", linewidth=0.5)
    return dict(color="white", edgecolor=color, linewidth=1.4, hatch="//")


def fig_hero1(decisive: dict) -> None:
    dr = decisive["delta_rate"]
    nb = decisive["null_band"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, beh in zip(axes, ["evil", "sycophancy"]):
        cells = dr[beh]
        edge = nb[beh]["upper_edge_boot97p5"]
        if edge > 0:
            ax.axhspan(0, edge, color="0.82", alpha=0.6, zorder=0)
        ax.axhline(edge, color="0.45", linestyle=":", linewidth=1.2, zorder=1)
        width = 0.38
        for di, d in enumerate(DIRS):
            for pi, pos in enumerate(POS):
                match = [k for k in cells if f"__direction{d}__" in k and f"__position{pos}__" in k]
                assert len(match) == 1, (beh, d, pos, match)
                v = cells[match[0]]
                y = v["delta_rate"]
                lo, hi = v["ci95"]
                x = di + (pi - 0.5) * width
                ax.bar(
                    x,
                    y,
                    width * 0.92,
                    yerr=[[y - lo], [hi - y]],
                    capsize=2.5,
                    error_kw=dict(linewidth=1.1, ecolor="0.25"),
                    zorder=3,
                    **_pos_bar_kwargs(pos, DIR_COLOR[d]),
                )
                if y > 0.02:
                    ax.text(x, hi + 0.025, f"{y:+.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(DIRS)))
        ax.set_xticklabels([DIR_LABEL[d] for d in DIRS], fontsize=8.5)
        ax.set_title(f"{beh} (layer {BEH_LAYER[beh]})", fontsize=12)
        ax.set_ylim(-0.05, 1.12)
    axes[0].set_ylabel("Δ judged behavior rate vs no injection")
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor="0.35", edgecolor="white", label="injected at answer tokens"),
        Patch(facecolor="white", edgecolor="0.35", hatch="//", label="injected at context token"),
        Patch(facecolor="0.82", edgecolor="0.45", label="selection-symmetric null band (97.5%)"),
    ]
    axes[1].legend(handles=handles, loc="upper right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    savefig_paper(fig, f"{FIGDIR}/hero1_decisive_bars")
    plt.close(fig)


def fig_perq(decisive: dict) -> None:
    pc = decisive["per_cell"]
    show = [
        ("rb", "answer"),
        ("rawmeandiff", "answer"),
        ("mapread_ctx", "answer"),
        ("rb", "context"),
        ("mapread_ctx", "context"),
    ]
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, beh in zip(axes, ["evil", "sycophancy"]):
        labels = []
        for xi, (d, pos) in enumerate(show):
            match = [k for k in pc if k.startswith(f"behavior{beh}__direction{d}__position{pos}__")]
            assert len(match) == 1, (beh, d, pos, match)
            v = pc[match[0]]
            rates = np.array(
                [
                    v["per_question_rate"][str(q)]
                    for q in range(20)
                    if str(q) in v["per_question_rate"]
                ]
            )
            jitter = rng.uniform(-0.16, 0.16, size=rates.size)
            filled = pos == "answer"
            ax.scatter(
                np.full(rates.size, xi) + jitter,
                rates,
                s=26,
                facecolors=DIR_COLOR[d] if filled else "none",
                edgecolors=DIR_COLOR[d],
                linewidths=1.2,
                alpha=0.85,
                zorder=3,
            )
            ax.hlines(rates.mean(), xi - 0.28, xi + 0.28, color="0.15", linewidth=2.0, zorder=4)
            labels.append(f"{DIR_LABEL[d]}\n@ {POS_LABEL[pos].split()[0]}")
        ax.set_xticks(range(len(show)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(beh, fontsize=12)
        ax.set_ylim(-0.05, 1.08)
    axes[0].set_ylabel("per-question judged behavior rate")
    fig.tight_layout()
    savefig_paper(fig, f"{FIGDIR}/perq_rates_decisive")
    plt.close(fig)


def fig_dose_grid(localize: dict) -> None:
    pc = localize["per_cell"]

    def a0_rate(beh: str) -> float:
        ks = [k for k in pc if k.startswith(f"behavior{beh}__directionalpha0__")]
        assert len(ks) == 1, ks
        return pc[ks[0]]["rate"]

    doses = [0.5, 1.0, 2.0, 4.0]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    for ci, beh in enumerate(["evil", "sycophancy"]):
        L = BEH_LAYER[beh]
        base = a0_rate(beh)
        for ri, pos in enumerate(["answer", "context"]):
            ax = axes[ri][ci]
            for d in DIRS:
                xs, ys = [0.0], [0.0]
                for c in doses:
                    k = _cell_key(beh, d, pos, L, c)
                    if k not in pc:
                        continue
                    r = pc[k]["rate"]
                    if r != r:  # NaN: all judge draws dropped for every item
                        continue
                    xs.append(c)
                    ys.append(r - base)
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=4.5,
                    linewidth=1.8,
                    color=DIR_COLOR[d],
                    label=DIR_LABEL[d].replace("-\n", "-").replace("\n", " "),
                )
            ax.set_title(f"{beh} — injected at {POS_LABEL[pos]} (layer {L})", fontsize=10.5)
            ax.set_ylim(-0.08, 1.05)
        axes[1][ci].set_xlabel("dose c  (injection norm = c × median residual norm)")
    for ri in range(2):
        axes[ri][0].set_ylabel("Δ judged behavior rate")
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper left", ncol=1)
    fig.tight_layout()
    savefig_paper(fig, f"{FIGDIR}/dose_response_grid")
    plt.close(fig)


def fig_hallu_gate(localize: dict, op: dict) -> None:
    nb = localize["reduced"]["null_band"]["hallucination"]
    cells = op["hallucination"]
    pc = localize["per_cell"]
    a0_keys = [k for k in pc if k.startswith("behaviorhallucination__directionalpha0__")]
    assert len(a0_keys) == 1, a0_keys
    base_rate = pc[a0_keys[0]]["rate"]  # un-steered baseline hallucination rate (0.733)
    ceiling = 1.0 - base_rate  # max achievable delta-rate (plan § Statistics overlay)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    edge = nb["upper_edge_boot97p5"]
    ax.axhspan(0, edge, color="0.82", alpha=0.6, zorder=0)
    ax.axhline(edge, color="0.45", linestyle=":", linewidth=1.2, zorder=1)
    ax.axhline(ceiling, color="#b2182b", linestyle="--", linewidth=1.6, zorder=4)
    width = 0.38
    for di, d in enumerate(DIRS):
        for pi, pos in enumerate(POS):
            v = cells[f"{d}__{pos}"]
            x = di + (pi - 0.5) * width
            ax.bar(x, v["delta_rate"], width * 0.92, zorder=3, **_pos_bar_kwargs(pos, DIR_COLOR[d]))
    ax.set_xticks(range(len(DIRS)))
    ax.set_xticklabels([DIR_LABEL[d] for d in DIRS], fontsize=8.5)
    ax.set_ylabel("Δ judged hallucination rate\n(best coherent layer × dose per cell)")
    ax.set_ylim(0, max(0.5, edge + 0.08))
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor="0.35", edgecolor="white", label="injected at answer tokens"),
        Patch(facecolor="white", edgecolor="0.35", hatch="//", label="injected at context token"),
        Patch(facecolor="0.82", edgecolor="0.45", label="selection-symmetric null band (97.5%)"),
        Line2D(
            [0],
            [0],
            color="#b2182b",
            linestyle="--",
            linewidth=1.6,
            label=f"achievable Δrate ceiling (1 − baseline {base_rate:.2f} = {ceiling:.2f})",
        ),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=False)
    fig.tight_layout()
    savefig_paper(fig, f"{FIGDIR}/hallucination_gate2")
    plt.close(fig)


def fig_margin(margin: dict) -> None:
    tf = margin["tf_margin"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, beh in zip(axes, ["evil", "sycophancy"]):
        width = 0.38
        for di, d in enumerate(DIRS):
            for pi, pos in enumerate(POS):
                y = tf[beh][f"{d}__{pos}"]
                x = di + (pi - 0.5) * width
                ax.bar(x, y, width * 0.92, zorder=3, **_pos_bar_kwargs(pos, DIR_COLOR[d]))
        ax.set_xticks(range(len(DIRS)))
        ax.set_xticklabels([DIR_LABEL[d] for d in DIRS], fontsize=8.5)
        ax.set_title(beh, fontsize=12)
        ax.axhline(0, color="0.3", linewidth=0.8)
    axes[0].set_ylabel("teacher-forced margin (nats)\nlog P(positive pool) − log P(negative pool)")
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor="0.35", edgecolor="white", label="injected at answer tokens"),
        Patch(facecolor="white", edgecolor="0.35", hatch="//", label="injected at context token"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8.5, frameon=False)
    fig.tight_layout()
    savefig_paper(fig, f"{FIGDIR}/margin_secondary")
    plt.close(fig)


def fig_hero1_iclr(decisive: dict) -> None:
    """--style iclr: Overleaf-paper variant of the decisive steering bars.

    Same cells as ``fig_hero1`` at final ICLR size into figures/paper/. Hatch
    is banned under the iclr style, so injection SITE is encoded by fill
    (solid = answer tokens, open = context token) and per-bar value labels are
    stripped (the LaTeX caption carries the key numbers). Colours bind by
    direction FAMILY through PAPER_COLORS: featured blue = map-read
    directions (light shade = the prefix arm), persona-vector sky blue =
    mean-difference directions (light shade = the raw unfiltered variant),
    null gray = the shuffled-label and random controls (light shade = random).
    """
    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_color,
        set_paper_style as _sps,
    )

    _sps("iclr")
    fam_color = {
        "mapread_ctx": paper_color("instruct"),
        "mapread_prefix": "#7FB3D8",
        "rb": paper_color("persona_vector"),
        "rawmeandiff": "#A7D4F0",
        "shuffled": paper_color("null"),
        "random": "#C4C4C4",
    }
    dir_label = {
        "mapread_ctx": "map read\n(context)",
        "mapread_prefix": "map read\n(prefix)",
        "rb": "persona\nvector",
        "rawmeandiff": "raw mean-\ndifference",
        "shuffled": "shuffled-\nlabel map",
        "random": "random",
    }
    dr = decisive["delta_rate"]
    nb = decisive["null_band"]
    fig, axes = plt.subplots(1, 2, figsize=figsize_iclr_panels(2, height_in=2.2), sharey=True)
    for ax, beh in zip(axes, ["evil", "sycophancy"], strict=True):
        cells = dr[beh]
        edge = nb[beh]["upper_edge_boot97p5"]
        if edge > 0:
            ax.axhspan(0, edge, color="0.85", alpha=0.6, zorder=0)
        ax.axhline(edge, color="0.45", linestyle=":", linewidth=0.8, zorder=1)
        width = 0.38
        for di, d in enumerate(DIRS):
            for pi, pos in enumerate(POS):
                match = [k for k in cells if f"__direction{d}__" in k and f"__position{pos}__" in k]
                assert len(match) == 1, (beh, d, pos, match)
                v = cells[match[0]]
                y = v["delta_rate"]
                lo, hi = v["ci95"]
                x = di + (pi - 0.5) * width
                kw = (
                    dict(color=fam_color[d], edgecolor="white", linewidth=0.3)
                    if pos == "answer"
                    else dict(color="white", edgecolor=fam_color[d], linewidth=0.9)
                )
                ax.bar(
                    x,
                    y,
                    width * 0.92,
                    yerr=[[max(0.0, y - lo)], [max(0.0, hi - y)]],
                    capsize=1.5,
                    error_kw=dict(linewidth=0.6, ecolor="0.25"),
                    zorder=3,
                    **kw,
                )
        ax.set_xticks(range(len(DIRS)))
        ax.set_xticklabels([dir_label[d] for d in DIRS], fontsize=6)
        ax.set_title(f"{beh.capitalize()} (layer {BEH_LAYER[beh]})")
        ax.set_ylim(-0.06, 1.05)
    axes[0].set_ylabel("$\\Delta$ judged behavior rate")
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor="0.35", edgecolor="white", label="injected at answer tokens"),
        Patch(facecolor="white", edgecolor="0.35", linewidth=0.9, label="injected at context token"),
        Patch(facecolor="0.85", alpha=0.6, label="selection-symmetric null band (97.5%)"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=6, frameon=False)
    fig.tight_layout()
    out_dir = Path("/home/thomasjiralerspong/explore-persona-space/figures/paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, "c5_steering_boundary", dir=out_dir)
    plt.close(fig)
    print(f"wrote {out_dir / 'c5_steering_boundary'}.png/.pdf (iclr)")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--style",
        choices=("blog", "iclr"),
        default="blog",
        help=(
            "iclr: render ONLY the paper decisive-bars variant into figures/paper/ "
            "and exit; the committed blog-register figures are untouched"
        ),
    )
    args = ap.parse_args()
    if args.style == "iclr":
        fig_hero1_iclr(_load(EV / "decisive" / "delta_rate_percell.json"))
        return

    set_paper_style()  # blog register
    decisive = _load(EV / "decisive" / "delta_rate_percell.json")
    localize = _load(EV / "localize" / "dose_response.json")
    op = _load(EV / "localize" / "operating_points.json")
    margin = _load(EV / "margin" / "margin_percell.json")
    fig_hero1(decisive)
    fig_perq(decisive)
    fig_dose_grid(localize)
    fig_hallu_gate(localize, op)
    fig_margin(margin)
    print("figures written to", ROOT / FIGDIR)


if __name__ == "__main__":
    main()
