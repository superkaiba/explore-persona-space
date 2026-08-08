"""Result 2 (FAIR PROTOCOL) four-panel spread-flagged re-cut for #1739.

User-requested layout (verbatim spec): "4 plots, one group of bars for persona
vectors synthetic eval, one for generic chat, one for in-distribution eval, one
for completely OOD (averaged across completely OOD settings), one plot per
behavior + one plot averaged across behaviors — indicate where spread is not
enough so we shouldn't trust".

Renders TWO figures under `figures/issue_1739/result2_fourpanel/`:

  result2_fourpanel.{png,pdf,meta.json}
      four panels — evil / sycophancy / hallucination / averaged across
      behaviors — each with four setting groups on x (synthetic, generic chat,
      in-distribution, completely OOD) and one bar per method within a group.
      The averaged panel EXCLUDES spread-failed cells (evil drops out of
      generic chat + OOD, so those bars average 2 behaviors, not 3).
  result2_fourpanel_avg_variants.{png,pdf,meta.json}
      the averaged-across-behaviors panel in both variants side by side —
      excluding spread-failed cells (as in the main figure) vs including
      everything — so the effect of the exclusion is visible.

Methods (4, user's revised list; single bar each — pv_context and oracle use
no map, so they carry one value regardless of map kind):

  pv_context     arm1_ctx_e1        Persona vector on context
  pv_map_linear  arm6 + linear map  Persona vector on mapped answer (linear)
  pv_map_mlp     arm6 + MLP map     Persona vector on mapped answer (MLP)
  oracle         arm11_oracle_proj  Persona vector on real answer

Spread flagging: per-(behavior, setting) verdicts come from the committed
Result 1 gate (`result1_spread/spread_stats.json`, criterion: floor/ceiling
mass <= 0.90 AND split-half reliability r_yy >= 0.50 AND min detectable rho
<= 0.5 x ceiling). Three cells FAIL, all evil: wildchat_rung, hhrt, toxicchat
— so evil's generic-chat group AND its entire completely-OOD group are marked
untrustworthy (hatched + muted, never deleted). Hatching here is the
user-requested marking device, overriding the paper-plots default against it.

Scores: `eval_results/issue_1739/result2_fair/result2_fair_points.json`
(committed output of scripts/issue1739_result2fair_fig.py::collect — read
directly rather than re-deriving from the per-behavior fair summaries).
Pure re-render of committed artifacts: no fits, no GPU, no network.
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

# Result 1's corpus-identity labels, imported (not copied) so figures agree by
# construction (same convention as scripts/issue1739_result2fair_fig.py).
from issue1739_result1_spread_fig_v2 import SETTING_IDENTITY  # noqa: E402

EVAL = ROOT / "eval_results/issue_1739"
POINTS = EVAL / "result2_fair/result2_fair_points.json"
SPREAD_STATS = EVAL / "result1_spread/spread_stats.json"
OUT_FIG = ROOT / "figures/issue_1739/result2_fourpanel"

# (method slot, plain-English label, color). One fixed method -> color mapping
# reused across every panel. pv_context / oracle colors match the committed
# R2FAIR figure; pv_map_mlp gets its own color because hatching is reserved
# for the spread-failed marking here (the R2FAIR figure used hatch for map
# kind instead).
METHODS = [
    ("pv_context", "Persona vector on context", "#08519C"),
    ("pv_map_linear", "Persona vector on mapped answer (linear map)", "#8C3000"),
    ("pv_map_mlp", "Persona vector on mapped answer (MLP map)", "#CC5500"),
    ("oracle", "Persona vector on real answer", "#00694C"),
]
METHOD_SLOTS = [m for m, _l, _c in METHODS]
COLOR = {m: c for m, _l, c in METHODS}
LABEL = {m: lbl for m, lbl, _c in METHODS}
# (arm_id, map_kind) -> method slot (subset of the R2FAIR figure's METHOD_OF).
METHOD_OF = {
    ("arm1_ctx_e1", "linear"): "pv_context",
    ("arm6_map_proj_e1", "linear"): "pv_map_linear",
    ("arm6_map_proj_e1", "mlp"): "pv_map_mlp",
    ("arm11_oracle_proj", "linear"): "oracle",
}

GROUPS = ["synthetic", "generic chat", "in-distribution", "completely OOD"]
GROUP_SETTINGS = {
    beh: {
        "synthetic": ["pvsynth"],
        "generic chat": ["wildchat_rung"],
        "in-distribution": ["train"],
        "completely OOD": ood,
    }
    for beh, ood in {
        "evil": ["hhrt", "toxicchat"],
        "sycophancy": ["aita"],
        "hallucination": ["nqopen", "simpleqa"],
    }.items()
}

FAIL_HATCH = "///"
FAIL_ALPHA = 0.45
FAIL_EDGE = "#555555"
GROUP_WIDTH = 0.76
BAR_WIDTH = GROUP_WIDTH / len(METHOD_SLOTS)


def load_points() -> dict[tuple[str, str, str], dict]:
    """{(behavior, setting, method): point record} for the four figure methods."""
    doc = json.loads(POINTS.read_text())
    out: dict[tuple[str, str, str], dict] = {}
    for p in doc["points"]:
        slot = METHOD_OF.get((p["arm_id"], p.get("map_kind", "linear")))
        if slot is None:
            continue
        key = (p["behavior"], p["setting"], slot)
        if key in out:
            raise SystemExit(f"duplicate point {key}")
        out[key] = p
    for beh in BEHAVIORS:
        for settings in GROUP_SETTINGS[beh].values():
            for s in settings:
                for m in METHOD_SLOTS:
                    if (beh, s, m) not in out:
                        raise SystemExit(f"missing point {(beh, s, m)}")
    return out


def load_verdicts() -> dict[tuple[str, str], dict]:
    """{(behavior, setting): spread cell} from the committed Result 1 gate."""
    doc = json.loads(SPREAD_STATS.read_text())
    return {(c["behavior"], c["setting"]): c for c in doc["cells"]}


def group_cell(points: dict, verdicts: dict, beh: str, group: str, method: str) -> dict:
    """One bar: rho + CI (+ spread verdict) for (behavior, group, method).

    Multi-rung groups (completely OOD with >1 rung) take the SIMPLE MEAN of
    the rung rhos; their interval is the elementwise mean of the rung CI
    bounds — a conservative display of the constituent CIs (exact under
    perfect correlation), NOT a resampled CI of the mean.
    """
    settings = GROUP_SETTINGS[beh][group]
    recs = [points[(beh, s, method)] for s in settings]
    rho = float(np.mean([r["rho"] for r in recs]))
    ci = [
        float(np.mean([r["ci"][0] for r in recs])),
        float(np.mean([r["ci"][1] for r in recs])),
    ]
    failing = [s for s in settings if verdicts[(beh, s)]["criterion_verdict"] == "FAIL"]
    if failing and len(failing) != len(settings):
        # No such case in the committed data (evil's OOD rungs both fail);
        # a partial-fail group would still be marked, flagged here loudly.
        print(f"WARNING: partial spread-fail in {beh}/{group}: {failing}")
    return {
        "behavior": beh,
        "group": group,
        "method": method,
        "rho": rho,
        "ci": ci,
        "settings": settings,
        "n_eval": {s: points[(beh, s, method)]["n_eval"] for s in settings},
        "spread_failed": bool(failing),
        "failing_settings": failing,
    }


def behavior_panel(points: dict, verdicts: dict, beh: str) -> dict:
    ident = {
        "synthetic": SETTING_IDENTITY[(beh, "pvsynth")],
        "generic chat": SETTING_IDENTITY[(beh, "wildchat_rung")],
        "in-distribution": SETTING_IDENTITY[(beh, "train")],
        "completely OOD": " + ".join(
            SETTING_IDENTITY[(beh, s)] for s in GROUP_SETTINGS[beh]["completely OOD"]
        ),
    }
    groups = []
    for g in GROUPS:
        bars = [group_cell(points, verdicts, beh, g, m) for m in METHOD_SLOTS]
        groups.append(
            {
                "label": f"{g}\n({ident[g]})",
                "bars": bars,
                "failed": bars[0]["spread_failed"],
            }
        )
    return {"title": beh, "groups": groups}


def averaged_panel(points: dict, verdicts: dict, exclude_failed: bool) -> dict:
    """Average the per-behavior group values across behaviors, per method.

    ``exclude_failed=True`` drops spread-failed (behavior, group) cells from
    the average; the group label carries the contributing-behavior count.
    """
    groups = []
    for g in GROUPS:
        cells = {
            beh: [group_cell(points, verdicts, beh, g, m) for m in METHOD_SLOTS]
            for beh in BEHAVIORS
        }
        contributing = [
            beh for beh in BEHAVIORS if not (exclude_failed and cells[beh][0]["spread_failed"])
        ]
        excluded = [b for b in BEHAVIORS if b not in contributing]
        bars = []
        for mi, m in enumerate(METHOD_SLOTS):
            member = [cells[beh][mi] for beh in contributing]
            bars.append(
                {
                    "behavior": "average",
                    "group": g,
                    "method": m,
                    "rho": float(np.mean([c["rho"] for c in member])),
                    "ci": [
                        float(np.mean([c["ci"][0] for c in member])),
                        float(np.mean([c["ci"][1] for c in member])),
                    ],
                    "contributing_behaviors": contributing,
                    "excluded_behaviors": excluded,
                    # Marked untrustworthy only when failed cells are INCLUDED.
                    "spread_failed": (not exclude_failed)
                    and any(cells[beh][0]["spread_failed"] for beh in BEHAVIORS),
                }
            )
        groups.append(
            {
                "label": f"{g}\n(avg of {len(contributing)} behaviours)",
                "bars": bars,
                "failed": bars[0]["spread_failed"],
            }
        )
    title = (
        "average across behaviours — spread-failed cells excluded"
        if exclude_failed
        else "average across behaviours — all cells included"
    )
    return {"title": title, "groups": groups}


def draw_panel(ax, panel: dict) -> int:
    n_bars = 0
    xs = list(range(len(panel["groups"])))
    ax.axhline(0.0, color="#B0B0B0", linewidth=0.9, zorder=1)
    for x, grp in zip(xs, panel["groups"], strict=True):
        for mi, bar in enumerate(grp["bars"]):
            offset = -GROUP_WIDTH / 2 + (mi + 0.5) * BAR_WIDTH
            failed = bar["spread_failed"]
            ax.bar(
                [x + offset],
                [bar["rho"]],
                width=BAR_WIDTH,
                color=COLOR[bar["method"]],
                alpha=FAIL_ALPHA if failed else 1.0,
                hatch=FAIL_HATCH if failed else None,
                edgecolor=FAIL_EDGE if failed else "#FFFFFF",
                linewidth=0.4 if failed else 0.25,
                zorder=3,
            )
            lo, hi = bar["ci"]
            # Non-negative offsets from the value, never raw bounds (gotchas).
            err_lo = max(0.0, bar["rho"] - lo)
            err_hi = max(0.0, hi - bar["rho"])
            ax.errorbar(
                [x + offset],
                [bar["rho"]],
                yerr=np.array([[err_lo], [err_hi]]),
                fmt="none",
                ecolor="#333333",
                elinewidth=0.7,
                capsize=0,
                zorder=4,
            )
            n_bars += 1
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [g["label"] for g in panel["groups"]],
        fontsize=7.2,
        rotation=12,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlim(-0.6, len(xs) - 0.4)
    ax.set_title(panel["title"], loc="left")
    return n_bars


def legend_handles() -> list[Patch]:
    handles = [
        Patch(facecolor=COLOR[m], edgecolor="#FFFFFF", linewidth=0.25, label=LABEL[m])
        for m in METHOD_SLOTS
    ]
    handles.append(
        Patch(
            facecolor="#BBBBBB",
            alpha=FAIL_ALPHA,
            hatch=FAIL_HATCH,
            edgecolor=FAIL_EDGE,
            linewidth=0.4,
            label="spread gate failed — not interpretable",
        )
    )
    return handles


CAPTION_MAIN = (
    "Spearman rho of each predictor vs the judged behaviour-expression DV (graded 0-100 trait "
    "rubric; hallucination's in-distribution and completely-OOD cells instead score fabricated "
    "fraction x100 — a different construct, and the averaged panel mixes the two).   All arms "
    "read CONTEXT-based representations (context_end) only — the prefix-mapping arm was not "
    "scored in this round (recorded user-directed deviation).   Error bars: within-draw paired "
    "bootstrap CI over eval contexts (single replicate); an averaged bar carries the elementwise "
    "MEAN of its constituents' CI bounds — conservative, not a resampled CI of the mean.   "
    "'Completely OOD' = simple mean over the behaviour's OOD rungs: evil 2 (hh-rlhf red-team, "
    "ToxicChat), sycophancy 1 (held-out Reddit r/socialskills), hallucination 2 (NQ-Open, "
    "SimpleQA) — the average means different things per behaviour.   Hatched muted groups FAIL "
    "the Result 1 spread gate (floor/ceiling mass <= 0.90 AND split-half reliability r_yy >= "
    "0.50 AND min detectable rho <= 0.5 x sqrt(r_yy), per behaviour x setting): evil fails "
    "generic chat AND both OOD rungs, so those whole groups are untrustworthy (bars kept for "
    "completeness).   Averaged panel EXCLUDES spread-failed cells — its generic-chat and OOD "
    "bars average 2 behaviours (sycophancy, hallucination), the rest 3.   Per-bar n_eval — "
    "synthetic: 200 each; generic chat: 417/415/411 (evil/syco/hall; held-out ~20% WildChat "
    "split — spread verdicts were computed on the FULL rung); in-distribution: 6,468 / 16,000 / "
    "16,000; OOD rungs: evil 1,868 + 519, sycophancy 1,304, hallucination 3,167 + 4,021."
)

CAPTION_VARIANTS = (
    "Averaged-across-behaviours panel of the four-panel Result 2 (fair protocol) figure, both "
    "ways.   LEFT: spread-failed cells excluded (as in the main figure) — evil drops out of "
    "generic chat and completely OOD (its wildchat + both OOD rungs fail the Result 1 spread "
    "gate), so those bars average sycophancy + hallucination only (2 behaviours; the other "
    "groups average 3).   RIGHT: all cells included — generic chat and completely OOD then "
    "carry evil's near-floor uninterpretable cells and are hatched/muted accordingly.   Same "
    "metric, arms, CI convention, and per-cell n as the main figure (see its caption)."
)


def render_main(points: dict, verdicts: dict) -> tuple[int, list[dict]]:
    panels = [behavior_panel(points, verdicts, beh) for beh in BEHAVIORS]
    panels.append(averaged_panel(points, verdicts, exclude_failed=True))

    vals: list[float] = []
    for p in panels:
        for g in p["groups"]:
            for b in g["bars"]:
                vals.extend([b["rho"], *b["ci"]])
    ylim = (min(-0.05, min(vals) - 0.04), max(vals) + 0.04)

    set_paper_style("blog", font_scale=0.9)
    fig = plt.figure(figsize=(18.0, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.16])
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02)
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, i], sharey=axes[0]) for i in (1, 2, 3)]
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")

    n_bars = 0
    for ax, panel in zip(axes, panels, strict=True):
        n_bars += draw_panel(ax, panel)
        ax.set_ylim(*ylim)
    axes[0].set_ylabel("Spearman rho, prediction vs judged behaviour expression")
    fig.suptitle(
        "Result 2 (fair protocol), four-panel spread-flagged view: persona-vector reads "
        "across evaluation regimes, hatched where the DV spread gate fails",
        x=0.006,
        ha="left",
    )
    legend_ax.legend(
        handles=legend_handles(),
        ncol=3,
        loc="upper left",
        alignment="left",
        frameon=False,
        fontsize=8.2,
        borderpad=0.0,
        bbox_to_anchor=(0.0, 1.0),
        bbox_transform=legend_ax.transAxes,
    )
    fig.text(
        0.006,
        0.006,
        CAPTION_MAIN,
        ha="left",
        va="bottom",
        fontsize=6.7,
        color="#4A4A4A",
        wrap=True,
    )
    savefig_paper(fig, "result2_fourpanel", dir=OUT_FIG)
    plt.close(fig)

    records = [b for p in panels for g in p["groups"] for b in g["bars"]]
    return n_bars, records


def render_avg_variants(points: dict, verdicts: dict) -> tuple[int, list[dict]]:
    panels = [
        averaged_panel(points, verdicts, exclude_failed=True),
        averaged_panel(points, verdicts, exclude_failed=False),
    ]
    vals = [v for p in panels for g in p["groups"] for b in g["bars"] for v in (b["rho"], *b["ci"])]
    ylim = (min(-0.05, min(vals) - 0.04), max(vals) + 0.04)

    set_paper_style("blog", font_scale=0.9)
    fig = plt.figure(figsize=(11.5, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.20])
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02)
    axes = [fig.add_subplot(gs[0, 0])]
    axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0]))
    plt.setp(axes[1].get_yticklabels(), visible=False)
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")

    n_bars = 0
    for ax, panel in zip(axes, panels, strict=True):
        n_bars += draw_panel(ax, panel)
        ax.set_ylim(*ylim)
    axes[0].set_ylabel("Spearman rho, prediction vs judged behaviour expression")
    fig.suptitle(
        "Average across behaviours: effect of excluding spread-failed cells",
        x=0.006,
        ha="left",
    )
    legend_ax.legend(
        handles=legend_handles(),
        ncol=3,
        loc="upper left",
        alignment="left",
        frameon=False,
        fontsize=8.0,
        borderpad=0.0,
        bbox_to_anchor=(0.0, 1.0),
        bbox_transform=legend_ax.transAxes,
    )
    fig.text(
        0.006,
        0.006,
        CAPTION_VARIANTS,
        ha="left",
        va="bottom",
        fontsize=7.0,
        color="#4A4A4A",
        wrap=True,
    )
    savefig_paper(fig, "result2_fourpanel_avg_variants", dir=OUT_FIG)
    plt.close(fig)

    records = [{**b, "panel": p["title"]} for p in panels for g in p["groups"] for b in g["bars"]]
    return n_bars, records


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    points = load_points()
    verdicts = load_verdicts()

    n_main, rec_main = render_main(points, verdicts)
    if n_main != 4 * len(GROUPS) * len(METHOD_SLOTS):
        raise SystemExit(f"main figure plotted {n_main} bars, expected 64")
    n_var, rec_var = render_avg_variants(points, verdicts)
    if n_var != 2 * len(GROUPS) * len(METHOD_SLOTS):
        raise SystemExit(f"variants figure plotted {n_var} bars, expected 32")

    sidecar = {
        "source_points": str(POINTS.relative_to(ROOT)),
        "source_spread": str(SPREAD_STATS.relative_to(ROOT)),
        "caption_main": CAPTION_MAIN,
        "caption_avg_variants": CAPTION_VARIANTS,
        "main_bars": rec_main,
        "avg_variant_bars": rec_var,
    }
    (OUT_FIG / "fourpanel_values.json").write_text(
        json.dumps(sidecar, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT_FIG / 'result2_fourpanel.png'} ({n_main} bars)")
    print(f"wrote {OUT_FIG / 'result2_fourpanel_avg_variants.png'} ({n_var} bars)")
    print(f"wrote {OUT_FIG / 'fourpanel_values.json'}")


if __name__ == "__main__":
    main()
