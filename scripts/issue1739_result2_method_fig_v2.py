"""Result 2 for #1739, simplified: one bar per method, in every evaluation setting.

Renders three figures under `figures/issue_1739/result2_methods/`, one per INPUT
STATE the readouts are computed from, using the canonical vocabulary of
`docs/glossary_context_answer_map.md`:

  result2_methods_context_simple.{png,pdf}     context (prefix + query) -> answer
  result2_methods_prefix_end_simple.{png,pdf}  prefix-end state -> answer
  result2_methods_bare_query_simple.{png,pdf}  bare query -> answer

Never "prefix map" unqualified: `prefix_end` is the PREFIX-END STATE (the
residual stream at the last prefix token), not the query-averaged prefix vector.

Grouped bars, not lines: x runs over the evaluation settings (the train setting
included) and each setting carries one bar per method, with the committed
bootstrap CI as the error bar. Bar slots, colours and y-limits are identical
across the three figures so they superimpose. A method with no row at the
operating slice has NO bar in that setting -- never a zero bar.

Colour encodes the methodology grouping the legend is organised by (reads the
input state / reads through the fitted map / oracle / control), with a distinct
shade per method inside each group; oracle and control bars are additionally
hatched so they can never be mistaken for a deployable method.

Simplification of `issue1739_result2_method_fig.py`, which draws the same
numbers as lines plus diagnostic-column shading, spread-gate shading,
reliability-ceiling segments, run-terminator markers and a four-block
reading-the-panel legend. This script re-plots that script's committed
`eval_results/issue_1739/result2_methods/result2_points.json` verbatim (333
records), so the operating-slice filter, the matched-target guard and the
replicate aggregation are inherited unchanged -- no fits, no GPU, no network.
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)
from issue1739_recut_common import BEHAVIORS, ROOT  # noqa: E402

POINTS_PATH = ROOT / "eval_results/issue_1739/result2_methods/result2_points.json"
OUT_FIG = ROOT / "figures/issue_1739/result2_methods"

INPUT_STATES = [
    ("context", "context", "context (prefix + query) -> answer"),
    ("prefix_end", "prefix-end state", "prefix-end state -> answer"),
    ("bare_query", "bare query", "bare query -> answer"),
]

SETTINGS = {
    "evil": ["train", "hhrt", "toxicchat", "wildchat_rung", "pvsynth"],
    "sycophancy": ["train", "aita", "wildchat_rung", "pvsynth"],
    "hallucination": ["train", "nqopen", "simpleqa", "wildchat_rung", "pvsynth"],
}
SETTING_LABEL = {
    ("evil", "train"): "held-out\ntrain",
    ("evil", "hhrt"): "hh-rlhf\nred-team\n(OOD)",
    ("evil", "toxicchat"): "ToxicChat\n(OOD)",
    ("sycophancy", "train"): "held-out\ntrain",
    ("sycophancy", "aita"): "AITA\n(OOD)",
    ("hallucination", "train"): "held-out\nTriviaQA",
    ("hallucination", "nqopen"): "NQ-Open\n(OOD)",
    ("hallucination", "simpleqa"): "SimpleQA\n(OOD)",
}
for _b in BEHAVIORS:
    SETTING_LABEL[(_b, "wildchat_rung")] = "random\nWildChat"
    SETTING_LABEL[(_b, "pvsynth")] = "persona-\nvectors\nsynthetic"

# Hallucination's own rungs score a fabricated FRACTION rescaled x100; its
# WildChat and persona-vectors-synthetic settings score the graded 0-100 trait
# rubric. Different constructs, so a divider separates the two groups.
FABRICATION_RATE_SETTINGS = {("hallucination", s) for s in ("train", "nqopen", "simpleqa")}

# --- methodology grouping: the legend's organising principle -------------------
# One hue per group, one shade per method inside it. Bar slot order is the order
# below, identical in every setting and every figure.
GROUPS = [
    (
        "input_state",
        "reads the input state",
        None,
        [
            ("arm1_ctx_e1", "persona-vector projection @ {state}", "#08519C"),
            ("arm2_ctx_native", "label-supervised direction @ {state}", "#2171B5"),
            ("arm4_ridge_ctx", "ridge @ {state}", "#4292C6"),
            ("arm5_mlp_ctx", "MLP @ {state}", "#7FB3DA"),
        ],
    ),
    (
        "through_map",
        "reads through the fitted map",
        None,
        [
            ("arm6_map_proj_e1", "persona-vector projection @ mapped answer", "#8C3000"),
            ("arm7_map_ridge_pred", "ridge @ mapped answer (fit on predicted)", "#D55E00"),
            ("arm8_map_ridge_true", "ridge @ mapped answer (fit on real)", "#EE8A3E"),
            ("arm9_pretrain_ft", "map-pretrain then fine-tune", "#F5B172"),
            ("arm10_stacked", "stacked combiner", "#FBD6AE"),
        ],
    ),
    (
        "oracle",
        "oracle (reads the real answer, not deployable)",
        "//",
        [
            ("arm11_oracle_proj", "persona-vector projection @ real answer", "#00694C"),
            ("arm12_oracle_reg", "ridge @ real answer", "#009E73"),
            ("arm17_oracle_mlp", "MLP @ real answer", "#4FBF98"),
            ("arm18_oracle_krr", "kernel ridge @ real answer", "#93D9BF"),
        ],
    ),
    (
        "control",
        "control / floor (not a usable method)",
        "..",
        [
            ("arm3_identity_bias", "identity + learned bias", "#4D4D4D"),
            ("arm13_shuffled_map", "shuffled map", "#7A7A7A"),
            ("arm14_shuffled_pt", "shuffled pretraining", "#9E9E9E"),
            ("arm15_text_only", "text-embedding ridge", "#BDBDBD"),
            ("arm16_surface_feat", "surface features", "#D9D9D9"),
        ],
    ),
]
ARM_SLOTS = [arm for _k, _t, _h, arms in GROUPS for arm, _lbl, _c in arms]
ARM_COLOR = {arm: c for _k, _t, _h, arms in GROUPS for arm, _lbl, c in arms}
ARM_HATCH = {arm: h for _k, _t, h, arms in GROUPS for arm, _lbl, _c in arms}
ARM_LABEL = {arm: lbl for _k, _t, _h, arms in GROUPS for arm, lbl, _c in arms}

GROUP_WIDTH = 0.86
BAR_WIDTH = GROUP_WIDTH / len(ARM_SLOTS)


def load() -> tuple[dict[tuple[str, str, str, str], dict], tuple[float, float]]:
    """{(input_state, behavior, setting, arm): record} + the shared y-range."""
    doc = json.loads(POINTS_PATH.read_text())
    table = {(p["input_state"], p["behavior"], p["setting"], p["arm_id"]): p for p in doc["points"]}
    if len(table) != len(doc["points"]):
        raise SystemExit("duplicate (input_state, behavior, setting, arm) records")
    vals: list[float] = []
    for p in doc["points"]:
        vals.append(p["rho"])
        if p["ci"]:
            vals.extend(p["ci"])
    return table, (min(vals) - 0.05, max(vals) + 0.05)


def render(state_key: str, state_token: str, title: str, table: dict, ylim) -> int:
    set_paper_style("blog", font_scale=0.85)
    fig = plt.figure(figsize=(19.0, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.26])
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.03)
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, i], sharey=axes[0]) for i in (1, 2)]
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")

    n_plotted, drew_construct_divider = 0, False
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        settings = [
            s for s in SETTINGS[beh] if any((state_key, beh, s, a) in table for a in ARM_SLOTS)
        ]
        xs = list(range(len(settings)))

        ax.axhline(0.0, color="#B0B0B0", linewidth=0.9, zorder=1)
        fab = [s for s in settings if (beh, s) in FABRICATION_RATE_SETTINGS]
        if fab and len(fab) < len(settings):
            edge = settings.index(fab[-1])
            ax.axvline(
                (xs[edge] + xs[edge + 1]) / 2.0,
                color="#444444",
                linestyle=(0, (4, 3)),
                linewidth=1.3,
                zorder=2,
            )
            drew_construct_divider = True

        for slot, arm in enumerate(ARM_SLOTS):
            offset = -GROUP_WIDTH / 2 + (slot + 0.5) * BAR_WIDTH
            bx, by, lo, hi = [], [], [], []
            for x, s in zip(xs, settings, strict=True):
                rec = table.get((state_key, beh, s, arm))
                if rec is None:  # method not run in this setting: no bar
                    continue
                bx.append(x + offset)
                by.append(rec["rho"])
                ci = rec["ci"] or [rec["rho"], rec["rho"]]
                lo.append(max(0.0, rec["rho"] - ci[0]))
                hi.append(max(0.0, ci[1] - rec["rho"]))
            if not bx:
                continue
            ax.bar(
                bx,
                by,
                width=BAR_WIDTH,
                color=ARM_COLOR[arm],
                hatch=ARM_HATCH[arm],
                edgecolor="#FFFFFF",
                linewidth=0.25,
                yerr=np.vstack([lo, hi]),
                error_kw=dict(ecolor="#333333", elinewidth=0.6, capsize=0),
                zorder=3,
            )
            n_plotted += len(bx)

        ax.set_xticks(xs)
        ax.set_xticklabels([SETTING_LABEL[(beh, s)] for s in settings], fontsize=8.0)
        ax.set_xlim(-0.62, max(xs) + 0.62 if xs else 0.62)
        ax.set_ylim(*ylim)
        ax.set_title(beh, loc="left")

    axes[0].set_ylabel("Spearman rho, prediction vs judged behavior expression")
    axes[1].set_xlabel("evaluation setting")
    suptitle = f"Input state: {title}"
    if state_key == "bare_query":
        suptitle += "   (run only on the random-WildChat rung, so one setting wide)"
    fig.suptitle(suptitle, x=0.006, ha="left")

    for i, (_key, gtitle, hatch, arms) in enumerate(GROUPS):
        leg = legend_ax.legend(
            handles=[
                Patch(
                    facecolor=c,
                    hatch=hatch,
                    edgecolor="#FFFFFF",
                    linewidth=0.25,
                    label=lbl.format(state=state_token),
                )
                for _a, lbl, c in arms
            ],
            title=gtitle,
            ncol=1,
            loc="upper left",
            alignment="left",
            frameon=False,
            fontsize=8.0,
            borderpad=0.0,
            bbox_to_anchor=(i * 0.255, 1.0),
            bbox_transform=legend_ax.transAxes,
        )
        leg.get_title().set_fontsize(8.4)
        leg.get_title().set_fontweight("semibold")
        legend_ax.add_artist(leg)

    if drew_construct_divider:
        fig.text(
            0.006,
            0.012,
            "Hallucination's held-out TriviaQA / NQ-Open / SimpleQA settings (left of the dashed "
            "line) score fabrication rate x100, a different construct from the 0-100 trait rubric "
            "everywhere else.",
            ha="left",
            va="bottom",
            fontsize=8.2,
            color="#5A5A5A",
        )

    savefig_paper(fig, f"result2_methods_{state_key}_simple", dir=OUT_FIG)
    plt.close(fig)
    return n_plotted


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    table, ylim = load()
    total = 0
    for state_key, state_token, title in INPUT_STATES:
        n = render(state_key, state_token, title, table, ylim)
        total += n
        print(f"wrote {OUT_FIG / f'result2_methods_{state_key}_simple.png'}  ({n} bars)")
    if total != len(table):
        raise SystemExit(f"plotted {total} bars but the points file carries {len(table)} records")
    print(f"all {total} committed records plotted; shared y-range {ylim[0]:.3f}..{ylim[1]:.3f}")


if __name__ == "__main__":
    main()
