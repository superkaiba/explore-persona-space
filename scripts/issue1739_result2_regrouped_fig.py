"""Result 2 five-method figure REGROUPED BY INPUT SOURCE for #1739.

User ask (verbatim): "in the final figure separate real anwer vs mapped answer
vs context better".

Layout-only recut of the committed five-method figure
(`scripts/issue1739_result2_fivemethod_fig.py`, output under
`figures/issue_1739/result2_fivemethod/` — untouched by this script). Same
source file, same 70 rows (14 (behavior, setting) cells x 5 arms, filtered to
variant == "context"), same Result 1 spread verdicts, same per-group
aggregation — all imported from the committed scripts, not copied, so the
figures agree by construction. What changes is bar ORDER + GROUPING within
each evaluation-regime group:

  hue       = input source (blue = context, orange/browns = mapped answer,
              greens = real answer). The committed color scheme already
              encodes this; the regroup makes it the primary visual axis.
  lightness = readout family within a hue: fitted ridge regression takes the
              LIGHTER shade, persona-vector projection the DARKER.
  x-gaps    = whitespace between the three input-source clusters inside every
              evaluation-regime group, each cluster labeled beneath in plain
              English (short labels context / mapped / real; the full wording
              is carried by the legend entries and the caption).

Renders TWO variants under `figures/issue_1739/result2_regrouped/`:

  result2_regrouped_5bar.{png,pdf,meta.json}
      the committed five methods, regrouped: [context: ridge] |
      [mapped answer: ridge, PV] | [real answer: ridge, PV].
  result2_regrouped_6bar.{png,pdf,meta.json}
      same grouping, but the mapped-answer cluster carries BOTH mapped-answer
      regression arms as separately-labeled bars: arm7_map_ridge_pred (ridge
      fitted on the mapped answer, applied to the mapped answer — the
      committed figure's "Ridge regression on mapped answer") AND
      arm8_map_ridge_true (ridge fitted on the REAL answer, applied to the
      mapped answer). Both carry full 14/14 cell coverage in the source file
      (asserted at load).

Spread-gate marking DEVIATION from the committed figure: this round's spec
bans hatching on the canvas, so spread-failed cells are marked by MUTING
(alpha + gray edge) only — same verdicts, same legend entry, no hatch.

Draw-time layout audit: constrained layout reserves space for the legend
SUBPLOT but not for the fig.text caption (the trap fixed once in the
committed script), so this script measures the rendered legend and wrapped
caption y-extents per variant and fails loud on overlap; it ALSO fails loud
when adjacent cluster-label bounding boxes clear each other by less than
MIN_LABEL_GAP_PX (a measured negative clearance is a bug, never shipped).

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

# Spread-fail rendering constants, group definitions, per-bar cell builder and
# verdict loader are reused BY IMPORT from the committed four-panel template.
from issue1739_result2_fourpanel_fig import (  # noqa: E402
    FAIL_ALPHA,
    FAIL_EDGE,
    GROUP_SETTINGS,
    GROUPS,
    group_cell,
    load_verdicts,
)
from issue1739_result1_spread_fig_v2 import SETTING_IDENTITY  # noqa: E402

# Colors, labels, source path and the 70-row loader (source filtering + spread
# cross-check) are reused from the committed five-method figure script.
from issue1739_result2_fivemethod_fig import (  # noqa: E402
    COLOR,
    LABEL,
    POINTS,
    load_points,
)

OUT_FIG = ROOT / "figures/issue_1739/result2_regrouped"

# Second mapped-answer regression arm (6-bar variant only). Mid-tone in the
# mapped-answer orange/brown family: lighter than the PV projection (#8C3000),
# darker than the fitted-on-mapped ridge (#E69F00) — lightness still separates
# the two ridge members from the PV member within the hue.
ARM8_ID = "arm8_map_ridge_true"
ARM8_SLOT = "reg_map_true"
ARM8_COLOR = "#C17400"

# Cluster spec: (plain-English label rendered beneath the cluster, methods).
# Short one-word labels so horizontal text fits between cluster centers; the
# full wording ("mapped answer" / "real answer") is carried by the legend
# entries and the caption's LAYOUT sentence.
CLUSTERS_5 = [
    ("context", ["reg_context"]),
    ("mapped", ["reg_map", "pv_map"]),
    ("real", ["reg_real", "pv_real"]),
]
CLUSTERS_6 = [
    ("context", ["reg_context"]),
    ("mapped", ["reg_map", ARM8_SLOT, "pv_map"]),
    ("real", ["reg_real", "pv_real"]),
]

LABEL_6 = {
    **LABEL,
    "reg_map": "Ridge fitted on mapped answer, applied to mapped answer",
    ARM8_SLOT: "Ridge fitted on real answer, applied to mapped answer",
}
COLOR_6 = {**COLOR, ARM8_SLOT: ARM8_COLOR}

GROUP_WIDTH = 0.84
CLUSTER_GAP = 0.12  # intra-group cluster gap; inter-group whitespace is 0.16
LEGEND_BAND_5 = 0.26  # gridspec height ratio of the bottom legend row (5-bar)
LEGEND_BAND_6 = 0.30  # 6-bar: one extra caption line -> taller band
CLUSTER_FONT = 5.0  # cluster-label fontsize (pt); regime labels are 7.2
FIG_W = 21.5  # widened vs the committed 19.0 so horizontal cluster labels fit
MIN_LABEL_GAP_PX = 2.0  # required bbox clearance between adjacent cluster labels


def load_arm8(verdicts: dict) -> tuple[dict[tuple[str, str, str], dict], list]:
    """{(behavior, setting, ARM8_SLOT): point} + spread cross-check disagreements.

    Same filter discipline as the committed loader: variant == "context" only;
    asserts uniqueness and full 14-cell coverage against GROUP_SETTINGS.
    """
    doc = json.loads(POINTS.read_text())
    out: dict[tuple[str, str, str], dict] = {}
    disagreements: list[dict] = []
    for p in doc["points"]:
        if p.get("arm_id") != ARM8_ID or p.get("variant") != "context":
            continue
        key = (p["behavior"], p["setting"], ARM8_SLOT)
        if key in out:
            raise SystemExit(f"duplicate arm8 point {key}")
        out[key] = p
        committed = verdicts[(p["behavior"], p["setting"])]["criterion_verdict"]
        if p.get("spread_gate") != committed:
            disagreements.append(
                {
                    "behavior": p["behavior"],
                    "setting": p["setting"],
                    "method": ARM8_SLOT,
                    "inline_spread_gate": p.get("spread_gate"),
                    "committed_verdict": committed,
                }
            )
    n_cells = 0
    for beh in BEHAVIORS:
        for settings in GROUP_SETTINGS[beh].values():
            for s in settings:
                n_cells += 1
                if (beh, s, ARM8_SLOT) not in out:
                    raise SystemExit(f"missing arm8 point {(beh, s)}")
    if len(out) != n_cells:
        raise SystemExit(f"expected exactly {n_cells} arm8 rows, got {len(out)}")
    return out, disagreements


def bar_layout(clusters: list) -> tuple[float, dict[str, float], list[tuple[str, float]]]:
    """Bar width, per-method center offsets, and (label, center) per cluster."""
    n_bars = sum(len(ms) for _lbl, ms in clusters)
    bw = (GROUP_WIDTH - (len(clusters) - 1) * CLUSTER_GAP) / n_bars
    offsets: dict[str, float] = {}
    centers: list[tuple[str, float]] = []
    pos = -GROUP_WIDTH / 2
    for lbl, methods in clusters:
        first = pos
        for m in methods:
            offsets[m] = pos + bw / 2
            pos += bw
        centers.append((lbl, (first + pos) / 2))
        pos += CLUSTER_GAP
    return bw, offsets, centers


def behavior_panel(points: dict, verdicts: dict, beh: str, slots: list[str]) -> dict:
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
        bars = {m: group_cell(points, verdicts, beh, g, m) for m in slots}
        groups.append(
            {
                "label": f"{g}\n({ident[g]})",
                "bars": bars,
                "failed": bars[slots[0]]["spread_failed"],
            }
        )
    return {"title": beh, "groups": groups}


def averaged_panel(points: dict, verdicts: dict, slots: list[str]) -> dict:
    """Average the per-behavior group values across behaviors, per method.

    Spread-failed (behavior, group) cells are EXCLUDED from the average, as in
    the committed main figure; the group label carries the contributing count.
    """
    groups = []
    for g in GROUPS:
        cells = {
            beh: {m: group_cell(points, verdicts, beh, g, m) for m in slots} for beh in BEHAVIORS
        }
        contributing = [beh for beh in BEHAVIORS if not cells[beh][slots[0]]["spread_failed"]]
        excluded = [b for b in BEHAVIORS if b not in contributing]
        bars = {}
        for m in slots:
            member = [cells[beh][m] for beh in contributing]
            bars[m] = {
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
                "spread_failed": False,  # failed cells are excluded, not carried
            }
        groups.append(
            {
                "label": f"{g}\n(avg of {len(contributing)} behaviours)",
                "bars": bars,
                "failed": False,
            }
        )
    return {"title": "average across behaviours — spread-failed cells excluded", "groups": groups}


def draw_panel(ax, panel: dict, clusters: list, colors: dict, layout: tuple) -> int:
    bw, offsets, centers = layout
    slots = [m for _lbl, ms in clusters for m in ms]
    xs = list(range(len(panel["groups"])))
    ax.axhline(0.0, color="#B0B0B0", linewidth=0.9, zorder=1)
    n_bars = 0
    for x, grp in zip(xs, panel["groups"], strict=True):
        for m in slots:
            bar = grp["bars"][m]
            failed = bar["spread_failed"]
            ax.bar(
                [x + offsets[m]],
                [bar["rho"]],
                width=bw,
                color=colors[m],
                alpha=FAIL_ALPHA if failed else 1.0,
                edgecolor=FAIL_EDGE if failed else "#FFFFFF",
                linewidth=0.4 if failed else 0.25,
                zorder=3,
            )
            lo, hi = bar["ci"]
            # Non-negative offsets from the value, never raw bounds (gotchas).
            err_lo = max(0.0, bar["rho"] - lo)
            err_hi = max(0.0, hi - bar["rho"])
            ax.errorbar(
                [x + offsets[m]],
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
    # Input-source cluster labels beneath each cluster, in every regime group.
    minor_pos = [x + c for x in xs for _lbl, c in centers]
    minor_lab = [lbl for _x in xs for lbl, _c in centers]
    ax.set_xticks(
        minor_pos,
        labels=minor_lab,
        minor=True,
        fontsize=CLUSTER_FONT,
        color="#444444",
    )
    ax.tick_params(axis="x", which="minor", length=0, pad=2)
    ax.tick_params(axis="x", which="major", pad=11)
    ax.set_xlim(-0.6, len(xs) - 0.4)
    ax.set_title(panel["title"], loc="left")
    return n_bars


def legend_handles(clusters: list, colors: dict, labels: dict) -> list[Patch]:
    handles = [
        Patch(facecolor=colors[m], edgecolor="#FFFFFF", linewidth=0.25, label=labels[m])
        for _lbl, ms in clusters
        for m in ms
    ]
    handles.append(
        Patch(
            facecolor="#BBBBBB",
            alpha=FAIL_ALPHA,
            edgecolor=FAIL_EDGE,
            linewidth=0.4,
            label="spread gate failed — not interpretable (muted)",
        )
    )
    return handles


CAPTION_COMMON = (
    "LAYOUT: within each evaluation-regime group, bars are clustered by INPUT SOURCE — context / "
    "mapped answer / real answer (short labels context / mapped / real beneath each cluster; "
    "whitespace separates the clusters); within a cluster the LIGHTER bar is the fitted ridge "
    "regression and the DARKER "
    "bar the persona-vector projection.   Spearman rho of each predictor vs the judged "
    "behaviour-expression DV (graded 0-100 trait rubric; hallucination's in-distribution and "
    "completely-OOD cells instead score fabricated fraction x100 — a different construct, and "
    "the averaged panel mixes the two).   PROTOCOL PROVENANCE: points come from the OLDER "
    "result2_methods protocol (result2_points.json), NOT the fair refit behind the committed "
    "four-panel figure — the fair round never ran the ridge-on-real-answer arm, so this is the "
    "only single-protocol source with all five methods.   The two 'real answer' methods are "
    "ORACLE reads (the source file labels them 'oracle'): they consume the model's ACTUAL "
    "generated answer at eval time; the context / mapped-answer methods score without it.   All "
    "arms read CONTEXT-based representations only — prefix-end cells are excluded (recorded "
    "user-directed scope).   Bar = mean of the committed per-replicate Spearman rho_frozen at "
    "the max-data operating slice (regime e1, full unlabeled pool, max labelled budget: evil 8k "
    "/ sycophancy 16k / hallucination 16k); error bars = the committed bootstrap ci_frozen "
    "averaged over the same replicates (15 replicates on in-distribution + OOD cells; single "
    "replicate on synthetic + generic chat); an averaged bar carries the elementwise MEAN of "
    "its constituents' CI bounds — conservative, not a resampled CI of the mean.   'Completely "
    "OOD' = simple mean over the behaviour's OOD rungs: evil 2 (hh-rlhf red-team, ToxicChat), "
    "sycophancy 1 (held-out Reddit r/socialskills), hallucination 2 (NQ-Open, SimpleQA) — the "
    "average means different things per behaviour.   MUTED (faded) groups FAIL the Result 1 "
    "spread gate (floor/ceiling mass <= 0.90 AND split-half reliability r_yy >= 0.50 AND min "
    "detectable rho <= 0.5 x sqrt(r_yy), per behaviour x setting): evil fails generic chat AND "
    "both OOD rungs, so those whole groups are untrustworthy (bars kept for completeness).   "
    "Averaged panel EXCLUDES spread-failed cells — its generic-chat and OOD bars average 2 "
    "behaviours (sycophancy, hallucination), the rest 3.   The generic-chat column UNDERSTATES "
    "the regression arms: the older grid scored the FULL WildChat rung (n 1,987/1,982/1,967 "
    "evil/syco/hall) while the fair refit scores a held-out ~20% split (n 411-417).   Per-bar "
    "n_eval elsewhere — synthetic: 200 each; in-distribution: 6,468 / 16,000 / 16,000; OOD "
    "rungs: evil 1,868 + 519, sycophancy 1,304, hallucination 3,167 + 4,021."
)

CAPTION_6BAR_EXTRA = (
    "   SIX-BAR VARIANT: the mapped-answer cluster carries BOTH mapped-answer regression arms — "
    "one ridge fitted on the mapped answer and applied to the mapped answer (the committed "
    "figure's 'Ridge regression on mapped answer'), and one ridge fitted on the REAL answer and "
    "applied to the mapped answer (oracle answers at fit time, mapped answers at eval time). "
    "Both have full 14-cell coverage in the source file."
)

TITLE_5 = (
    "Result 2 (older result2_methods protocol), five methods regrouped by input source — "
    "context vs mapped answer vs real answer; muted bars fail the DV spread gate"
)
TITLE_6 = (
    "Result 2 (older result2_methods protocol), regrouped by input source, six-bar variant — "
    "both mapped-answer regression arms shown separately; muted bars fail the DV spread gate"
)


def layout_audit(fig, ax0, legend, caption) -> dict:
    """Measure rendered legend / wrapped-caption y-extents + cluster-label gaps.

    Constrained layout reserves space for the legend SUBPLOT but not for the
    fig.text caption, so clearance is verified on the rendered figure. The
    caption is re-set to its wrapped form (explicit newlines) first so its
    window extent covers all wrapped lines, not one long line.

    Cluster labels are horizontal, so adjacent bounding-box gaps are the exact
    collision test; the caller REQUIRES min gap >= MIN_LABEL_GAP_PX (a
    measured negative gap is a fix-before-shipping bug, not a report line).
    """
    fig.canvas.draw()
    wrapped = caption._get_wrapped_text()
    caption.set_text(wrapped)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    height = fig.bbox.height
    leg = legend.get_window_extent(renderer)
    cap = caption.get_window_extent(renderer)
    exts = sorted(
        (t.get_window_extent(renderer) for t in ax0.get_xminorticklabels() if t.get_text()),
        key=lambda e: e.x0,
    )
    bbox_gaps = [exts[i + 1].x0 - exts[i].x1 for i in range(len(exts) - 1)]
    return {
        "legend_y_frac": [round(leg.y0 / height, 4), round(leg.y1 / height, 4)],
        "caption_y_frac": [round(cap.y0 / height, 4), round(cap.y1 / height, 4)],
        "legend_caption_gap_frac": round(leg.y0 / height - cap.y1 / height, 4),
        "caption_lines": wrapped.count("\n") + 1,
        "n_cluster_labels_panel0": len(exts),
        "min_cluster_label_bbox_gap_px": round(min(bbox_gaps), 1) if bbox_gaps else None,
    }


def render_variant(
    stem: str,
    points: dict,
    verdicts: dict,
    clusters: list,
    colors: dict,
    labels: dict,
    caption: str,
    title: str,
    legend_ncol: int,
    legend_band: float,
) -> tuple[int, list[dict], dict]:
    slots = [m for _lbl, ms in clusters for m in ms]
    layout = bar_layout(clusters)
    panels = [behavior_panel(points, verdicts, beh, slots) for beh in BEHAVIORS]
    panels.append(averaged_panel(points, verdicts, slots))

    vals: list[float] = []
    for p in panels:
        for g in p["groups"]:
            for b in g["bars"].values():
                vals.extend([b["rho"], *b["ci"]])
    ylim = (min(-0.05, min(vals) - 0.04), max(vals) + 0.04)

    set_paper_style("blog", font_scale=0.9)
    fig = plt.figure(figsize=(FIG_W, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, legend_band])
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02)
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, i], sharey=axes[0]) for i in (1, 2, 3)]
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")

    n_bars = 0
    for ax, panel in zip(axes, panels, strict=True):
        n_bars += draw_panel(ax, panel, clusters, colors, layout)
        ax.set_ylim(*ylim)
    axes[0].set_ylabel("Spearman rho, prediction vs judged behaviour expression")
    fig.suptitle(title, x=0.006, ha="left")
    legend = legend_ax.legend(
        handles=legend_handles(clusters, colors, labels),
        ncol=legend_ncol,
        loc="upper left",
        alignment="left",
        frameon=False,
        fontsize=8.0,
        borderpad=0.0,
        bbox_to_anchor=(0.0, 1.0),
        bbox_transform=legend_ax.transAxes,
    )
    caption_text = fig.text(
        0.006,
        0.006,
        caption,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#4A4A4A",
        wrap=True,
    )
    audit = layout_audit(fig, axes[0], legend, caption_text)
    if audit["legend_caption_gap_frac"] <= 0.002:
        raise SystemExit(f"{stem}: legend/caption collision — audit: {audit}")
    if (
        audit["min_cluster_label_bbox_gap_px"] is None
        or audit["min_cluster_label_bbox_gap_px"] < MIN_LABEL_GAP_PX
    ):
        raise SystemExit(f"{stem}: cluster labels overlap/crowd — audit: {audit}")
    savefig_paper(fig, stem, dir=OUT_FIG)
    plt.close(fig)

    records = [b for p in panels for g in p["groups"] for b in g["bars"].values()]
    audit["rho_range"] = [
        round(min(b["rho"] for b in records), 4),
        round(max(b["rho"] for b in records), 4),
    ]
    audit["ci_range"] = [round(min(vals), 4), round(max(vals), 4)]
    return n_bars, records, audit


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    verdicts = load_verdicts()
    points, disagreements = load_points(verdicts)  # asserts exactly 70 rows
    arm8, arm8_disagreements = load_arm8(verdicts)  # asserts exactly 14 rows
    n_arm7 = sum(1 for k in points if k[2] == "reg_map")
    print(
        f"row counts — five-arm context rows: {len(points)} (70 asserted); "
        f"arm7 cells: {n_arm7}/14; arm8 cells: {len(arm8)}/14"
    )

    n5, rec5, audit5 = render_variant(
        "result2_regrouped_5bar",
        points,
        verdicts,
        CLUSTERS_5,
        COLOR,
        LABEL,
        CAPTION_COMMON,
        TITLE_5,
        legend_ncol=3,
        legend_band=LEGEND_BAND_5,
    )
    if n5 != 4 * len(GROUPS) * 5:
        raise SystemExit(f"5-bar variant plotted {n5} bars, expected 80")

    n6, rec6, audit6 = render_variant(
        "result2_regrouped_6bar",
        {**points, **arm8},
        verdicts,
        CLUSTERS_6,
        COLOR_6,
        LABEL_6,
        CAPTION_COMMON + CAPTION_6BAR_EXTRA,
        TITLE_6,
        legend_ncol=4,
        legend_band=LEGEND_BAND_6,
    )
    if n6 != 4 * len(GROUPS) * 6:
        raise SystemExit(f"6-bar variant plotted {n6} bars, expected 96")

    sidecar = {
        "source_points": str(POINTS.relative_to(ROOT)),
        "source_spread": "eval_results/issue_1739/result1_spread/spread_stats.json",
        "row_counts": {"five_arm_context": len(points), "arm7": n_arm7, "arm8": len(arm8)},
        "spread_crosscheck_disagreements": disagreements + arm8_disagreements,
        "caption_5bar": CAPTION_COMMON,
        "caption_6bar": CAPTION_COMMON + CAPTION_6BAR_EXTRA,
        "layout_audit_5bar": audit5,
        "layout_audit_6bar": audit6,
        "bars_5bar": rec5,
        "bars_6bar": rec6,
    }
    (OUT_FIG / "regrouped_values.json").write_text(
        json.dumps(sidecar, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT_FIG / 'result2_regrouped_5bar.png'} ({n5} bars)")
    print(f"wrote {OUT_FIG / 'result2_regrouped_6bar.png'} ({n6} bars)")
    print(f"wrote {OUT_FIG / 'regrouped_values.json'}")
    print(f"layout audit 5-bar: {audit5}")
    print(f"layout audit 6-bar: {audit6}")
    all_dis = disagreements + arm8_disagreements
    print(f"spread cross-check: {'NO disagreements' if not all_dis else all_dis}")


if __name__ == "__main__":
    main()
