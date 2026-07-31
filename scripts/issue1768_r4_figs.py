"""Round-4 (prefix-richness-dose-ladder) figures for issue #1768.

Reads eval_results/issue_1768/on_target_r4/ (+ the round-3 committed
comparators under on_target/) and renders two figures into
figures/issue_1768/ with `r4_` prefixes:

1. r4_ladder_dose_identity — hero: (A) raw map-difference m per arm at
   layer 19 against prefix content tokens, never-trained rungs vs the
   round-3 trained-prefix anchors (own / control) and the bare level;
   (B) per-cell view: floor-subtracted D with 95% CIs for all 36
   (arm, layer, rung) cells, round-3 bare_n reference per row.
2. r4_percell_ecdf_identity_gap — low-level per-context data behind the
   m medians: ECDFs of per-context map-change norm for the persona-trained
   sycophancy arm and the conversation-trained comparator at layer 19,
   rungs vs own / control / bare, refit floors marked.

Behavior colors follow the round-1/3 figures (cas #0173B2, imp #DE8F05,
syc #029E73). Conditions ride linestyle / marker fill so no palette pair
is reused for a different factor. Saves via savefig_paper (blog style;
PNG + PDF + meta.json sidecar).
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
RUNGS = ["r_short", "r_mid", "r_long"]
RUNG_TOKENS = {"r_short": 10, "r_mid": 85, "r_long": 730}
# Trained-prefix content-token anchors (plan v10 §4.2: T_pers ~11, T_conv ~800).
T_PERS, T_CONV = 11, 800
LAYERS = [14, 19, 25]


def _load() -> dict:
    ladder = json.loads((R4 / "map_change_ladder.json").read_text())
    return ladder


def _bare_median(arm: str, layer: int) -> float:
    rows = json.loads((R3 / "percell" / f"{arm}_L{layer}_bare_n.json").read_text())["rows"]
    return float(np.median([r["delta"] for r in rows]))


def _percell_deltas(path: Path) -> np.ndarray:
    rows = json.loads(path.read_text())["rows"]
    return np.array([r["delta"] for r in rows], dtype=float)


def _floor(fit_path: Path) -> float:
    return float(json.loads(fit_path.read_text())["map_change"]["floor_p95"])


def fig_dose_identity(ladder: dict) -> None:
    m_table = ladder["m_table"]
    cells = ladder["cells"]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.6, 5.8))

    # ---- Panel A: L19 raw m dose curve with trained anchors ----
    x_bare = 2.0
    for arm, label, beh, ctx in ARMS:
        e = m_table[f"{arm}_L19"]
        col = COLORS[beh]
        solid = ctx == "pers"
        xs = [RUNG_TOKENS[r] for r in RUNGS]
        ys = [e["m_rung"][r] for r in RUNGS]
        ax_a.plot(
            xs,
            ys,
            color=col,
            ls="-" if solid else "--",
            lw=1.8,
            zorder=3,
            marker="o",
            ms=7,
            mfc=col if solid else "white",
            mec=col,
        )
        for x, y in zip(xs, ys):
            ax_a.text(x, y * 1.07, f"{y:.1f}", ha="center", va="bottom", fontsize=7.5, color=col)
        # bare level
        m_bare = _bare_median(arm, 19)
        ax_a.plot([x_bare], [m_bare], marker="s", ms=6, mfc="white", mec=col, ls="none")
        ax_a.text(
            x_bare * 0.88, m_bare, f"{m_bare:.1f}", ha="right", va="center", fontsize=7.5, color=col
        )
        # own / ctrl trained-prefix anchors (round 3)
        x_own = T_PERS if ctx == "pers" else T_CONV
        x_ctrl = T_CONV if ctx == "pers" else T_PERS
        ax_a.plot(
            [x_own], [e["m_own"]], marker="*", ms=13, mfc=col, mec="black", ls="none", zorder=4
        )
        ax_a.plot(
            [x_ctrl], [e["m_ctrl"]], marker="D", ms=7, mfc="white", mec=col, ls="none", zorder=4
        )
        ax_a.text(
            x_own * 1.13,
            e["m_own"] * 0.94,
            f"{e['m_own']:.1f}",
            ha="left",
            va="top",
            fontsize=7.5,
            color=col,
        )
        ax_a.text(
            x_ctrl * 1.13,
            e["m_ctrl"],
            f"{e['m_ctrl']:.1f}",
            ha="left",
            va="center",
            fontsize=7.5,
            color=col,
        )

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xticks([x_bare, 10, 85, 730])
    ax_a.set_xticklabels(["bare", "10", "85", "730"])
    ax_a.minorticks_off()
    ax_a.set_xlabel("prefix content tokens (log scale)")
    ax_a.set_ylabel("raw trained-vs-base map difference m\n(median per-context norm, layer 19)")
    ax_a.set_title("A — never-trained rungs (circles) vs trained-prefix anchors", fontsize=11)
    handles = [
        Line2D(
            [], [], color="black", marker="o", ls="-", ms=6, label="never-trained rung (round 4)"
        ),
        Line2D(
            [],
            [],
            color="black",
            marker="*",
            ls="none",
            ms=11,
            label="own trained prefix (round 3)",
        ),
        Line2D(
            [],
            [],
            color="black",
            marker="D",
            mfc="white",
            ls="none",
            ms=6,
            label="swapped trained prefix (round 3)",
        ),
        Line2D(
            [], [], color="black", marker="s", mfc="white", ls="none", ms=6, label="bare prompts"
        ),
    ] + [
        Line2D([], [], color=COLORS[b], ls=s, label=lab)
        for _, lab, b, c in ARMS
        for s in ["-" if c == "pers" else "--"]
    ]
    ax_a.set_ylim(top=ax_a.get_ylim()[1] * 1.6)
    ax_a.set_xlim(left=1.4, right=1500)
    fig.legend(
        handles=handles,
        fontsize=7.5,
        loc="lower center",
        ncol=4,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.005),
    )

    # ---- Panel B: per-cell D forest, all 36 cells ----
    yticks, ylabels = [], []
    y = 0
    rung_alpha = {"r_short": 0.35, "r_mid": 0.65, "r_long": 1.0}
    for arm, label, beh, ctx in ARMS:
        for layer in LAYERS:
            col = COLORS[beh]
            e_bare = cells[f"{arm}_L{layer}_r_short"]["D_bare_n"]
            ax_b.plot([e_bare], [y], marker="|", ms=11, color="black", ls="none", zorder=4)
            for i, rung in enumerate(RUNGS):
                c = cells[f"{arm}_L{layer}_{rung}"]
                lo, hi = c["D_rung_ci95"]
                yy = y + (i - 1) * 0.22
                ax_b.errorbar(
                    [c["D_rung"]],
                    [yy],
                    xerr=[[c["D_rung"] - lo], [hi - c["D_rung"]]],
                    fmt="o",
                    ms=4.5,
                    color=col,
                    alpha=rung_alpha[rung],
                    elinewidth=1.2,
                    capsize=0,
                )
            yticks.append(y)
            ylabels.append(
                f"{label.split(' — ')[0]} {'conv' if ctx == 'conv' else 'pers'} L{layer}"
            )
            y += 1
        y += 0.6
    ax_b.axvline(0.0, color="black", lw=0.8, ls=":")
    ax_b.set_yticks(yticks)
    ax_b.set_yticklabels(ylabels, fontsize=8)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("map-change statistic D (above 0 = above the refit-noise floor)")
    ax_b.set_title("B — all 36 cells: D per rung, 95% CIs; | = bare", fontsize=11)
    handles_b = [
        Line2D(
            [],
            [],
            color="black",
            marker="o",
            ls="none",
            alpha=a,
            label=f"{r[2:]} rung ({RUNG_TOKENS[r]} tok)",
        )
        for r, a in rung_alpha.items()
    ] + [
        Line2D(
            [], [], color="black", marker="|", ls="none", ms=10, label="bare (round 3, n-matched)"
        )
    ]
    ax_b.legend(handles=handles_b, fontsize=7.5, loc="lower right", framealpha=0.9)

    fig.suptitle(
        "Never-trained prefixes amplify map change with length, but stay below the trained conversation prefix",
        fontsize=12.5,
        y=1.005,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    savefig_paper(fig, "r4_ladder_dose_identity", dir=str(OUT))
    plt.close(fig)


def fig_ecdf(ladder: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), sharey=True)
    panels = [
        ("syc-pers-con-lr1e5-s42", "persona-trained sycophancy arm (layer 19)"),
        ("syc-conv-con-lr1e5-s42", "conversation-trained sycophancy arm (layer 19)"),
    ]
    rung_alpha = {"r_short": 0.35, "r_mid": 0.65, "r_long": 1.0}
    for ax, (arm, title) in zip(axes, panels):
        col = COLORS["syc"]
        conds: list[tuple[str, np.ndarray, dict]] = []
        for rung in RUNGS:
            d = _percell_deltas(R4 / "percell" / f"{arm}_L19_{rung}.json")
            conds.append(
                (
                    f"{rung[2:]} rung ({RUNG_TOKENS[rung]} tok)",
                    d,
                    dict(color=col, alpha=rung_alpha[rung], ls="-"),
                )
            )
        for cond, style in [
            ("bare_n", dict(color=GRAY, ls="-")),
            ("own", dict(color="black", ls="--")),
            ("control", dict(color="black", ls=":")),
        ]:
            d = _percell_deltas(R3 / "percell" / f"{arm}_L19_{cond}.json")
            label = {
                "bare_n": "bare prompts",
                "own": "own trained prefix",
                "control": "swapped trained prefix",
            }[cond]
            conds.append((label, d, style))
        for label, d, style in conds:
            xs = np.sort(d)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, lw=1.7, label=label, **style)
        # floors: long rung (round 4) + own condition (round 3)
        fl_long = _floor(R4 / "fits" / f"{arm}_L19_r_long.json")
        fl_own = _floor(R3 / "fits" / f"{arm}_L19_own.json")
        ax.axvline(fl_long, color=col, lw=0.9, ls=(0, (1, 3)))
        ax.axvline(fl_own, color="black", lw=0.9, ls=(0, (1, 3)))
        ax.text(fl_long, 0.02, " long-rung floor", fontsize=7, color=col, rotation=90, va="bottom")
        ax.text(
            fl_own, 0.02, " own-prefix floor", fontsize=7, color="black", rotation=90, va="bottom"
        )
        ax.set_xscale("log")
        ax.set_xlabel("per-context map-change norm (1,000 shared test contexts)")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    axes[0].set_ylabel("cumulative fraction of contexts")
    fig.suptitle(
        "Per-context distributions: the swapped trained prefix sits beyond every never-trained rung "
        "on the persona arm; the comparator's own prefix is nearly matched by the long rung",
        fontsize=11.5,
        y=1.005,
    )
    fig.tight_layout()
    savefig_paper(fig, "r4_percell_ecdf_identity_gap", dir=str(OUT))
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    OUT.mkdir(parents=True, exist_ok=True)
    ladder = _load()
    fig_dose_identity(ladder)
    fig_ecdf(ladder)
    # caption arithmetic printed for the body fold
    mt = ladder["m_table"]
    for arm, _, _, ctx in ARMS:
        e = mt[f"{arm}_L19"]
        print(
            arm,
            "L19: long/ctrl = %.2f" % (e["m_rung"]["r_long"] / e["m_ctrl"]),
            "ctrl/long = %.2f" % (e["m_ctrl"] / e["m_rung"]["r_long"]),
            "own",
            round(e["m_own"], 2),
        )


if __name__ == "__main__":
    main()
