"""Round-5 (behavior-relevant-prefix-anchor-control) figures for issue #1768.

Reads eval_results/issue_1768/on_target_r5/ (+ the round-3 committed
comparators under on_target/ and the round-4 rung anchors under
on_target_r4/) and renders two figures into figures/issue_1768/ with
`r5_` prefixes:

1. r5_brel_anchor_contrast — hero: per arm at layer 19, the raw
   map-difference median m under bare prompts, the own trained prefix,
   the round-4 never-trained neutral rungs (10 / 85 / 730 tokens), the
   three behavior-relevant never-trained prefixes (456-580 tokens), and
   the swapped trained-prefix anchor; bootstrap median CIs from the
   per-context files; the plan-registered dose-interpolated neutral
   reference at each realized token count marked as gray open circles.
2. r5_brel_percell_views — low-level data: (A) all 36 (arm, layer,
   prefix) floor-subtracted D cells with 95% CIs, value-labeled, bare
   reference ticks; (B) per-context map-change ECDFs for the
   persona-trained sycophancy arm and the conversation-trained
   comparator at layer 19, behavior-relevant prefixes vs the round-3/4
   anchors, refit floors dotted.

Behavior colors follow the round-1/3/4 figures (cas #0173B2,
imp #DE8F05, syc #029E73); conditions ride markers / linestyles so no
palette pair is reused for a different factor. Saves via savefig_paper
(blog style; PNG + PDF + meta.json sidecar).
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)

R5 = Path("eval_results/issue_1768/on_target_r5")
R4 = Path("eval_results/issue_1768/on_target_r4")
R3 = Path("eval_results/issue_1768/on_target")
OUT = Path("figures/issue_1768")

COLORS = {"cas": "#0173B2", "imp": "#DE8F05", "syc": "#029E73"}
GRAY = "#949494"

ARMS = [
    ("syc-pers-con-lr1e5-s42", "sycophancy — persona-trained", "syc", "pers"),
    ("imp-pers-con-lr3e5-s42", "impoliteness — persona-trained", "imp", "pers"),
    ("cas-pers-con-lr1e5-s42", "casual writing — persona-trained", "cas", "pers"),
    ("syc-conv-con-lr1e5-s42", "sycophancy — conversation-trained", "syc", "conv"),
]
PREFIXES = ["b_rel1", "b_rel2", "b_rel3"]
LAYERS = [14, 19, 25]
RNG_SEED = 1768
N_BOOT = 500


def _deltas(path: Path) -> np.ndarray:
    rows = json.loads(path.read_text())["rows"]
    return np.array([r["delta"] for r in rows], dtype=float)


def _median_ci(v: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(RNG_SEED)
    n = len(v)
    meds = np.median(v[rng.integers(0, n, size=(N_BOOT, n))], axis=1)
    return float(np.median(v)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _floor(fit_path: Path) -> float:
    return float(json.loads(fit_path.read_text())["map_change"]["floor_p95"])


def fig_anchor_contrast(brel: dict) -> None:
    m_table = brel["m_table"]
    tokens = brel["realized_tokens"]
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.2), sharex=True)
    # ordered condition axis
    conds = [
        ("bare", "bare prompts", R3, "bare_n"),
        ("own", "own trained prefix", R3, "own"),
        ("r_mid", "neutral rung 85 tok", R4, "r_mid"),
        ("r_long", "neutral rung 730 tok", R4, "r_long"),
        ("b_rel1", f"behavior-relevant 1 ({tokens['b_rel1']} tok)", R5, "b_rel1"),
        ("b_rel2", f"behavior-relevant 2 ({tokens['b_rel2']} tok)*", R5, "b_rel2"),
        ("b_rel3", f"behavior-relevant 3 ({tokens['b_rel3']} tok)", R5, "b_rel3"),
        ("ctrl", "swapped trained prefix", R3, "control"),
    ]
    xs = np.arange(len(conds))
    for ax, (arm, label, beh, ctx) in zip(axes.flat, ARMS):
        col = COLORS[beh]
        e = m_table[f"{arm}_L19"]
        for x, (key, _, root, fname) in zip(xs, conds):
            v = _deltas(root / "percell" / f"{arm}_L19_{fname}.json")
            med, lo, hi = _median_ci(v)
            if key.startswith("b_rel"):
                style = dict(marker="^", ms=9, mfc=col, mec="black", zorder=5)
            elif key == "own":
                style = dict(marker="*", ms=14, mfc="black", mec="black", zorder=4)
            elif key == "ctrl":
                style = dict(marker="D", ms=8, mfc="white", mec="black", zorder=4)
            elif key == "bare":
                style = dict(marker="s", ms=7, mfc="white", mec=GRAY, zorder=3)
            else:
                style = dict(marker="o", ms=7, mfc=GRAY, mec=GRAY, zorder=3)
            ax.errorbar(
                [x],
                [med],
                yerr=[[med - lo], [hi - med]],
                fmt="none",
                ecolor="black",
                elinewidth=1.1,
                capsize=2,
                zorder=3,
            )
            ax.plot([x], [med], ls="none", **style)
            ax.text(x + 0.14, med, f"{med:.1f}", ha="left", va="center", fontsize=7.5)
            # dose-interpolated neutral reference under each behavior-relevant point
            if key.startswith("b_rel"):
                ref = e["dose_interp"][key]["contrast_vs_interp"]["m_b"]
                ax.plot([x], [ref], marker="o", ms=7, mfc="white", mec=GRAY, ls="none", zorder=4)
        # anchor guide-line at the swapped trained prefix level
        v_ctrl = float(e["m_ctrl"])
        ax.axhline(v_ctrl, color="black", lw=0.8, ls=":", zorder=1)
        ax.set_title(label, fontsize=11, color=col)
        ax.set_ylabel("raw map difference m (median, layer 19)")
    for ax in axes.flat:
        ax.set_xticks(xs)
        ax.set_xticklabels([c[1] for c in conds], rotation=35, ha="right", fontsize=8)
    handles = [
        Line2D(
            [],
            [],
            marker="^",
            ls="none",
            ms=8,
            mfc="black",
            mec="black",
            label="behavior-relevant never-trained prefix (round 5)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=7,
            mfc="white",
            mec=GRAY,
            label="dose-interpolated neutral reference at realized tokens",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=7,
            mfc=GRAY,
            mec=GRAY,
            label="neutral never-trained rung (round 4)",
        ),
        Line2D(
            [],
            [],
            marker="*",
            ls="none",
            ms=12,
            mfc="black",
            mec="black",
            label="own trained prefix (round 3)",
        ),
        Line2D(
            [],
            [],
            marker="D",
            ls="none",
            ms=7,
            mfc="white",
            mec="black",
            label="swapped trained prefix (round 3; dotted guide)",
        ),
        Line2D([], [], marker="s", ls="none", ms=7, mfc="white", mec=GRAY, label="bare prompts"),
    ]
    fig.legend(
        handles=handles,
        fontsize=8.5,
        loc="lower center",
        ncol=3,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.002),
    )
    fig.suptitle(
        "Behavior-relevant never-trained prefixes overshoot the trained-prefix anchor\n"
        "(* = 456-token prefix, below the 547.5-912.5 token band)",
        fontsize=12.5,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    savefig_paper(fig, "r5_brel_anchor_contrast", dir=str(OUT))
    plt.close(fig)


def fig_percell_views(brel: dict) -> None:
    cells = brel["cells"]
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(14.4, 7.0), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )

    # ---- Panel A: 36-cell D forest, value-labeled ----
    pfx_alpha = {"b_rel1": 1.0, "b_rel2": 0.45, "b_rel3": 0.72}
    yticks, ylabels = [], []
    y = 0
    for arm, label, beh, ctx in ARMS:
        col = COLORS[beh]
        for layer in LAYERS:
            c0 = cells[f"{arm}_L{layer}_b_rel1"]
            ax_a.plot([c0["D_bare_n"]], [y], marker="|", ms=11, color="black", ls="none", zorder=4)
            for i, pfx in enumerate(PREFIXES):
                c = cells[f"{arm}_L{layer}_{pfx}"]
                lo, hi = c["D_brel_ci95"]
                yy = y + (i - 1) * 0.24
                ax_a.errorbar(
                    [c["D_brel"]],
                    [yy],
                    xerr=[[c["D_brel"] - lo], [hi - c["D_brel"]]],
                    fmt="^",
                    ms=4.5,
                    color=col,
                    alpha=pfx_alpha[pfx],
                    elinewidth=1.1,
                    capsize=0,
                )
                ax_a.text(
                    c["D_brel"],
                    yy - 0.13,
                    f"{c['D_brel']:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.6,
                    color=col,
                    alpha=0.95,
                )
            yticks.append(y)
            ylabels.append(
                f"{label.split(' — ')[0]} {'conv' if ctx == 'conv' else 'pers'} L{layer}"
            )
            y += 1
        y += 0.6
    ax_a.axvline(0.0, color="black", lw=0.8, ls=":")
    ax_a.set_yticks(yticks)
    ax_a.set_yticklabels(ylabels, fontsize=8)
    ax_a.invert_yaxis()
    ax_a.set_xscale("symlog", linthresh=1.0)
    ax_a.set_xlabel("map-change statistic D (above 0 = above the refit-noise floor; symlog)")
    ax_a.set_title(
        "A — all 36 cells: D per behavior-relevant prefix, 95% CIs; | = bare", fontsize=11
    )
    handles_a = [
        Line2D(
            [],
            [],
            color="black",
            marker="^",
            ls="none",
            alpha=a,
            label=f"{p.replace('b_rel', 'behavior-relevant ')} ({brel['realized_tokens'][p]} tok)",
        )
        for p, a in pfx_alpha.items()
    ] + [
        Line2D(
            [], [], color="black", marker="|", ls="none", ms=10, label="bare (round 3, n-matched)"
        )
    ]
    ax_a.legend(handles=handles_a, fontsize=7.5, loc="lower right", framealpha=0.9)

    # ---- Panel B: ECDFs, exemplar persona arm + comparator ----
    panels = [
        ("syc-pers-con-lr1e5-s42", "persona-trained sycophancy (L19)"),
        ("syc-conv-con-lr1e5-s42", "conversation-trained sycophancy (L19)"),
    ]
    col = COLORS["syc"]
    for k, (arm, title) in enumerate(panels):
        axx = ax_b if k == 0 else ax_b.inset_axes([0.58, 0.06, 0.4, 0.5])
        conds: list[tuple[str, np.ndarray, dict]] = []
        for pfx in PREFIXES:
            d = _deltas(R5 / "percell" / f"{arm}_L19_{pfx}.json")
            conds.append(
                (
                    f"behavior-relevant {pfx[-1]} ({brel['realized_tokens'][pfx]} tok)",
                    d,
                    dict(color=col, alpha=pfx_alpha[pfx], ls="-"),
                )
            )
        d = _deltas(R4 / "percell" / f"{arm}_L19_r_long.json")
        conds.append(("neutral rung 730 tok", d, dict(color=GRAY, ls="-")))
        d = _deltas(R3 / "percell" / f"{arm}_L19_own.json")
        conds.append(("own trained prefix", d, dict(color="black", ls="--")))
        d = _deltas(R3 / "percell" / f"{arm}_L19_control.json")
        conds.append(("swapped trained prefix", d, dict(color="black", ls=":")))
        for label, d, style in conds:
            xs_ = np.sort(d)
            ys_ = np.arange(1, len(xs_) + 1) / len(xs_)
            axx.plot(xs_, ys_, lw=1.6, label=label, **style)
        fl_brel = _floor(R5 / "fits" / f"{arm}_L19_b_rel1.json")
        axx.axvline(fl_brel, color=col, lw=0.9, ls=(0, (1, 3)))
        axx.set_xscale("log")
        axx.set_title(title, fontsize=10 if k == 0 else 8.5)
        if k == 0:
            axx.text(
                fl_brel,
                0.02,
                " behavior-relevant-prefix floor",
                fontsize=7,
                color=col,
                rotation=90,
                va="bottom",
            )
            axx.set_xlabel("per-context map-change norm (1,000 shared test contexts)")
            axx.set_ylabel("cumulative fraction of contexts")
            axx.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
        else:
            axx.tick_params(labelsize=7)
    ax_b.set_title("B — per-context distributions (exemplar + comparator inset)", fontsize=11)

    fig.suptitle(
        "Per-cell and per-context views: behavior-relevant never-trained prefixes sit at or "
        "beyond the trained anchors on every persona-trained cell",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout()
    savefig_paper(fig, "r5_brel_percell_views", dir=str(OUT))
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    OUT.mkdir(parents=True, exist_ok=True)
    brel = json.loads((R5 / "map_change_brel.json").read_text())
    fig_anchor_contrast(brel)
    fig_percell_views(brel)
    # caption arithmetic for the body fold
    mt = brel["m_table"]
    for arm, _, _, ctx in ARMS:
        e = mt[f"{arm}_L19"]
        gcs = {
            j: (e["m_brel"][j] - e["dose_interp"][j]["contrast_vs_interp"]["m_b"])
            / (e["m_ctrl"] - e["dose_interp"][j]["contrast_vs_interp"]["m_b"])
            for j in PREFIXES
        }
        print(
            arm,
            "L19 m_brel",
            {j: round(v, 2) for j, v in e["m_brel"].items()},
            "ctrl",
            round(e["m_ctrl"], 2),
            "gc_interp",
            {j: round(v, 2) for j, v in gcs.items()},
        )


if __name__ == "__main__":
    main()
