"""Result 2 FIVE-METHOD four-panel spread-flagged figure for #1739.

User-requested method list (verbatim spec): "regression from context /
regression from real answer to behavior / regression from mapped answer to
behavior / persona vector projected on real answer / persona vector projected
on mapped answer".

Scores: `eval_results/issue_1739/result2_methods/result2_points.json` (the
committed OLDER-protocol method-comparison points), filtered to
`variant == "context"` — the user's standing scope excludes prefix-end cells,
and the `bare_context_end` / `bare_query` rows are likewise out. That filter
plus the five arms below yields exactly 70 rows = 14 (behavior, setting)
cells x 5 methods, one row per cell.

PROTOCOL PROVENANCE — why this file and not the R2FAIR one: the newer fair
refit (`result2_fair/result2_fair_points.json`, source of the committed
four-panel figure) ran only arms 1/4/6/7/11/19 — it never ran
`arm12_oracle_reg`, so "regression from real answer" does not exist there.
`result2_methods` is the only source carrying all five methods under ONE
protocol. The two files are never mixed in the main figure; the companion
old-vs-fair scatter (rendered by this same script) compares the four shared
methods across the two protocols instead.

Renders THREE figures under `figures/issue_1739/result2_fivemethod/`:

  result2_fivemethod.{png,pdf,meta.json}
      four panels — evil / sycophancy / hallucination / averaged across
      behaviors — each with four setting groups on x (synthetic, generic
      chat, in-distribution, completely OOD) and one bar per method. The
      averaged panel EXCLUDES spread-failed cells.
  result2_fivemethod_avg_variants.{png,pdf,meta.json}
      the averaged-across-behaviors panel excluding vs including
      spread-failed cells, side by side.
  result2_fivemethod_old_vs_fair.{png,pdf,meta.json}
      per-cell consistency scatter, fair-protocol rho vs older-protocol rho,
      for the four methods present in BOTH files (ridge on context, ridge on
      mapped answer, PV on mapped answer, PV on real answer). The fair file
      splits map arms by map_kind (linear / mlp) while the older file does
      not; each older row is paired against the fair LINEAR row.

Spread flagging: per-(behavior, setting) verdicts come from the committed
Result 1 gate (`result1_spread/spread_stats.json` — authoritative, so this
figure and its committed four-panel sibling agree by construction). Each
point in `result2_points.json` also carries an inline `spread_gate` field;
it is CROSS-CHECKED against the committed verdicts and any disagreement is
reported loudly (none expected). Three cells FAIL, all evil: wildchat_rung,
hhrt, toxicchat — hatched + muted, never deleted.

Pure re-render of committed artifacts: no fits, no GPU, no network.
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from issue1739_recut_common import BEHAVIORS, ROOT  # noqa: E402

# Layout, spread-fail rendering, group definitions and the per-bar cell
# builder are reused BY IMPORT from the committed four-panel template so the
# two figures agree by construction.
from issue1739_result2_fourpanel_fig import (  # noqa: E402
    FAIL_ALPHA,
    FAIL_EDGE,
    FAIL_HATCH,
    GROUP_SETTINGS,
    GROUPS,
    group_cell,
    load_verdicts,
)
from issue1739_result1_spread_fig_v2 import SETTING_IDENTITY  # noqa: E402

EVAL = ROOT / "eval_results/issue_1739"
POINTS = EVAL / "result2_methods/result2_points.json"
FAIR_POINTS = EVAL / "result2_fair/result2_fair_points.json"
OUT_FIG = ROOT / "figures/issue_1739/result2_fivemethod"

# One fixed method -> color assignment across every panel. Hue encodes the
# readout-input family (blue = context, greens = REAL answer, orange/browns =
# MAPPED answer); within each answer family the regression member takes the
# LIGHTER shade and the persona-vector member the DARKER — lightness contrast
# survives color-vision deficiency, so every bar stays distinguishable to a
# colorblind reader. The three light anchors are Wong-palette colors from
# paper_palette(3) (blue #0072B2, orange #E69F00, bluish-green #009E73); the
# two darks match the committed four-panel figure's colors for the SAME
# methods (PV on real answer #00694C; PV on mapped answer #8C3000 = its
# linear-map color, the map kind the old-vs-fair scatter pairs against).
_WONG_BLUE, _WONG_ORANGE, _WONG_GREEN = paper_palette(3)
METHODS = [
    ("reg_context", "Ridge regression on context", _WONG_BLUE),
    ("reg_real", "Ridge regression on real answer", _WONG_GREEN),
    ("reg_map", "Ridge regression on mapped answer", _WONG_ORANGE),
    ("pv_real", "Persona vector on real answer", "#00694C"),
    ("pv_map", "Persona vector on mapped answer", "#8C3000"),
]
METHOD_SLOTS = [m for m, _l, _c in METHODS]
COLOR = {m: c for m, _l, c in METHODS}
LABEL = {m: lbl for m, lbl, _c in METHODS}
# arm_id -> method slot (all five present in result2_points.json; regime is
# uniformly e1). The fair file additionally keys map arms on map_kind.
SLOT_OF = {
    "arm4_ridge_ctx": "reg_context",
    "arm12_oracle_reg": "reg_real",
    "arm7_map_ridge_pred": "reg_map",
    "arm11_oracle_proj": "pv_real",
    "arm6_map_proj_e1": "pv_map",
}
EXPECTED_ROWS = 70  # 14 (behavior, setting) cells x 5 methods

GROUP_WIDTH = 0.80
BAR_WIDTH = GROUP_WIDTH / len(METHOD_SLOTS)

# Methods present in BOTH protocols (arm12_oracle_reg is result2_methods-only).
SHARED_SLOTS = ["reg_context", "reg_map", "pv_real", "pv_map"]
SCATTER_MARKER = {"reg_context": "o", "reg_map": "^", "pv_real": "s", "pv_map": "D"}
# Plain-English short setting names for scatter point labels (no slugs).
SHORT_SETTING = {
    "pvsynth": "synthetic",
    "wildchat_rung": "WildChat",
    "train": "in-dist",
    "hhrt": "hh-rlhf red-team",
    "toxicchat": "ToxicChat",
    "aita": "r/socialskills",
    "nqopen": "NQ-Open",
    "simpleqa": "SimpleQA",
}


def load_points(verdicts: dict) -> tuple[dict[tuple[str, str, str], dict], list]:
    """{(behavior, setting, method): point} + inline-vs-committed spread disagreements.

    Filters to variant == "context" (user scope: no prefix-end cells;
    bare_context_end / bare_query out) and the five figure arms; asserts the
    exact 70-row count, uniqueness, and completeness against GROUP_SETTINGS.
    """
    doc = json.loads(POINTS.read_text())
    out: dict[tuple[str, str, str], dict] = {}
    disagreements: list[dict] = []
    n_rows = 0
    for p in doc["points"]:
        slot = SLOT_OF.get(p["arm_id"])
        if slot is None or p.get("variant") != "context":
            continue
        n_rows += 1
        key = (p["behavior"], p["setting"], slot)
        if key in out:
            raise SystemExit(f"duplicate point {key}")
        out[key] = p
        committed = verdicts[(p["behavior"], p["setting"])]["criterion_verdict"]
        if p.get("spread_gate") != committed:
            disagreements.append(
                {
                    "behavior": p["behavior"],
                    "setting": p["setting"],
                    "method": slot,
                    "inline_spread_gate": p.get("spread_gate"),
                    "committed_verdict": committed,
                }
            )
    if n_rows != EXPECTED_ROWS:
        raise SystemExit(
            f"expected exactly {EXPECTED_ROWS} rows (14 cells x 5 methods) after the "
            f"variant=='context' + five-arm filter, got {n_rows} — stop and report"
        )
    for beh in BEHAVIORS:
        for settings in GROUP_SETTINGS[beh].values():
            for s in settings:
                for m in METHOD_SLOTS:
                    if (beh, s, m) not in out:
                        raise SystemExit(f"missing point {(beh, s, m)}")
    if disagreements:
        print(
            f"WARNING: inline spread_gate disagrees with spread_stats.json on "
            f"{len(disagreements)} rows — committed verdicts used for rendering; "
            f"disagreements recorded in the sidecar: {disagreements}"
        )
    return out, disagreements


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
    """Average the per-behavior group values across behaviors, per method."""
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
    "fraction x100 — a different construct, and the averaged panel mixes the two).   PROTOCOL "
    "PROVENANCE: points come from the OLDER result2_methods protocol (result2_points.json), NOT "
    "the fair refit behind the committed four-panel figure — the fair round never ran the "
    "ridge-on-real-answer arm, so this is the only single-protocol source with all five methods; "
    "the companion old-vs-fair scatter shows how the two protocols compare on the four shared "
    "methods.   The two 'real answer' methods are ORACLE reads (the source file labels them "
    "'oracle'): they consume the model's ACTUAL generated answer at eval time; the context / "
    "mapped-answer methods score without it.   All arms read CONTEXT-based representations only — "
    "prefix-end cells are excluded (recorded user-directed scope).   Bar = mean of the committed "
    "per-replicate Spearman "
    "rho_frozen at the max-data operating slice (regime e1, full unlabeled pool, max labelled "
    "budget: evil 8k / sycophancy 16k / hallucination 16k); error bars = the committed bootstrap "
    "ci_frozen averaged over the same replicates (15 replicates on in-distribution + OOD cells; "
    "single replicate on synthetic + generic chat); an averaged bar carries the elementwise MEAN "
    "of its constituents' CI bounds — conservative, not a resampled CI of the mean.   "
    "'Completely OOD' = simple mean over the behaviour's OOD rungs: evil 2 (hh-rlhf red-team, "
    "ToxicChat), sycophancy 1 (held-out Reddit r/socialskills), hallucination 2 (NQ-Open, "
    "SimpleQA) — the average means different things per behaviour.   Hatched muted groups FAIL "
    "the Result 1 spread gate (floor/ceiling mass <= 0.90 AND split-half reliability r_yy >= "
    "0.50 AND min detectable rho <= 0.5 x sqrt(r_yy), per behaviour x setting): evil fails "
    "generic chat AND both OOD rungs, so those whole groups are untrustworthy (bars kept for "
    "completeness).   Averaged panel EXCLUDES spread-failed cells — its generic-chat and OOD "
    "bars average 2 behaviours (sycophancy, hallucination), the rest 3.   Per-bar n_eval — "
    "synthetic: 200 each; generic chat: 1,987/1,982/1,967 (evil/syco/hall; the FULL WildChat "
    "rung, unlike the fair figure's held-out ~20% split); in-distribution: 6,468 / 16,000 / "
    "16,000; OOD rungs: evil 1,868 + 519, sycophancy 1,304, hallucination 3,167 + 4,021."
)

CAPTION_VARIANTS = (
    "Averaged-across-behaviours panel of the five-method Result 2 figure (older result2_methods "
    "protocol), both ways.   LEFT: spread-failed cells excluded (as in the main figure) — evil "
    "drops out of generic chat and completely OOD (its wildchat + both OOD rungs fail the "
    "Result 1 spread gate), so those bars average sycophancy + hallucination only (2 behaviours; "
    "the other groups average 3).   RIGHT: all cells included — generic chat and completely OOD "
    "then carry evil's near-floor uninterpretable cells and are hatched/muted accordingly.   "
    "Same metric, arms, CI convention, and per-cell n as the main figure (see its caption)."
)

CAPTION_SCATTER = (
    "Per-cell protocol-consistency check for the four methods scored under BOTH protocols: "
    "fair-refit Spearman rho (R2FAIR, result2_fair_points.json — source of the committed "
    "four-panel figure) vs the older result2_methods rho this five-method figure uses "
    "(ridge-on-real-answer exists only in the older file, so it cannot appear here).   One "
    "point per (behavior, setting, method); 4 methods x 14 cells = 56 points; dashed line = "
    "y equals x.   In the fair file the map arms split by map_kind (linear / mlp) while the "
    "older file does not — each older row is paired against the fair LINEAR row.   Hollow "
    "markers = cells failing the Result 1 spread gate (all evil: WildChat, hh-rlhf red-team, "
    "ToxicChat).   Known protocol differences besides the map/whitening refit: the older "
    "file's generic-chat cells score the FULL WildChat rung (n 1,967-1,987) while the fair "
    "file scores a held-out ~20% split (n 411-417)."
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
    fig = plt.figure(figsize=(19.0, 7.6))
    # Bottom-band geometry: constrained layout reserves space for legend_ax
    # but NOT for the fig.text caption, so the band must clear the caption by
    # itself. At 0.16 the legend's second row printed over the caption's top
    # lines (team-lead round 2, measured: legend rows y 0.072-0.109 vs caption
    # 0.006-0.089); 0.20 cleared it by only +0.010 of figure height. 0.24
    # gives a comfortable measured gap while the caption stays ~7 lines.
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.24])
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
        "Result 2 (older result2_methods protocol), five-method spread-flagged view: "
        "regression and persona-vector reads across evaluation regimes, hatched where "
        "the DV spread gate fails",
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
        CAPTION_MAIN,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#4A4A4A",
        wrap=True,
    )
    savefig_paper(fig, "result2_fivemethod", dir=OUT_FIG)
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
    fig = plt.figure(figsize=(12.0, 6.5))
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
        "Average across behaviours (five-method figure): effect of excluding spread-failed cells",
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
    savefig_paper(fig, "result2_fivemethod_avg_variants", dir=OUT_FIG)
    plt.close(fig)

    records = [{**b, "panel": p["title"]} for p in panels for g in p["groups"] for b in g["bars"]]
    return n_bars, records


def collect_fair_pairs(points: dict, verdicts: dict) -> list[dict]:
    """Pair older-protocol rho vs fair-protocol rho for the four shared methods.

    Fair map arms split by map_kind; the older file does not — the older row is
    paired against the fair LINEAR row (stated in the caption).
    """
    fair_doc = json.loads(FAIR_POINTS.read_text())
    fair = {}
    for p in fair_doc["points"]:
        slot = SLOT_OF.get(p["arm_id"])
        if slot is None or slot not in SHARED_SLOTS:
            continue
        if p.get("map_kind", "linear") != "linear":
            continue
        key = (p["behavior"], p["setting"], slot)
        if key in fair:
            raise SystemExit(f"duplicate fair point {key}")
        fair[key] = p
    pairs = []
    for beh in BEHAVIORS:
        for settings in GROUP_SETTINGS[beh].values():
            for s in settings:
                for m in SHARED_SLOTS:
                    key = (beh, s, m)
                    if key not in fair:
                        raise SystemExit(f"missing fair point {key}")
                    pairs.append(
                        {
                            "behavior": beh,
                            "setting": s,
                            "method": m,
                            "rho_old": float(points[key]["rho"]),
                            "rho_fair": float(fair[key]["rho"]),
                            "spread_failed": verdicts[(beh, s)]["criterion_verdict"] == "FAIL",
                        }
                    )
    if len(pairs) != len(SHARED_SLOTS) * 14:
        raise SystemExit(f"expected {len(SHARED_SLOTS) * 14} old-vs-fair pairs, got {len(pairs)}")
    return pairs


def render_old_vs_fair(pairs: list[dict]) -> dict:
    old = np.array([p["rho_old"] for p in pairs])
    fair = np.array([p["rho_fair"] for p in pairs])
    pearson_r = float(np.corrcoef(old, fair)[0, 1])
    spearman_r = float(stats.spearmanr(old, fair).statistic)

    set_paper_style("blog", font_scale=0.9)
    fig = plt.figure(figsize=(8.6, 9.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.14])
    ax = fig.add_subplot(gs[0, 0])
    caption_ax = fig.add_subplot(gs[1, 0])
    caption_ax.axis("off")
    lo = min(old.min(), fair.min()) - 0.06
    hi = max(old.max(), fair.max()) + 0.06
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#999999", linewidth=0.9, zorder=1)
    # Label declutter: points pile up along the diagonal (esp. the upper-right
    # synthetic/in-dist cluster), so cycle four label quadrants by rank along
    # the diagonal — near neighbours get different offsets. Deterministic; no
    # extra dependency (adjustText is not in the env).
    label_slots = [
        ((3.5, 2.5), "left"),
        ((3.5, -7.0), "left"),
        ((-3.5, 2.5), "right"),
        ((-3.5, -7.0), "right"),
    ]
    diag_order = sorted(range(len(pairs)), key=lambda i: pairs[i]["rho_old"] + pairs[i]["rho_fair"])
    slot_of = {idx: k % len(label_slots) for k, idx in enumerate(diag_order)}
    for i, p in enumerate(pairs):
        c = COLOR[p["method"]]
        failed = p["spread_failed"]
        ax.scatter(
            [p["rho_old"]],
            [p["rho_fair"]],
            marker=SCATTER_MARKER[p["method"]],
            s=34,
            facecolors="none" if failed else c,
            edgecolors=c,
            linewidths=1.1,
            zorder=3,
        )
        offset, halign = label_slots[slot_of[i]]
        ax.annotate(
            f"{p['behavior']} / {SHORT_SETTING[p['setting']]}",
            (p["rho_old"], p["rho_fair"]),
            textcoords="offset points",
            xytext=offset,
            ha=halign,
            fontsize=4.6,
            color="#555555",
            zorder=2,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Spearman rho — older protocol (result2_methods)")
    ax.set_ylabel("Spearman rho — fair protocol (R2FAIR)")
    ax.set_title(
        "Old-protocol vs fair-protocol Spearman rho, per (behavior, setting), four shared methods",
        loc="left",
    )
    handles = [
        Line2D(
            [],
            [],
            marker=SCATTER_MARKER[m],
            linestyle="none",
            markerfacecolor=COLOR[m],
            markeredgecolor=COLOR[m],
            markersize=6,
            label=LABEL[m],
        )
        for m in SHARED_SLOTS
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#666666",
            markersize=6,
            label="spread gate failed — not interpretable",
        )
    )
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.2)
    fig.text(
        0.006,
        0.006,
        CAPTION_SCATTER,
        ha="left",
        va="bottom",
        fontsize=6.3,
        color="#4A4A4A",
        wrap=True,
    )
    savefig_paper(fig, "result2_fivemethod_old_vs_fair", dir=OUT_FIG)
    plt.close(fig)

    diffs = sorted(pairs, key=lambda p: abs(p["rho_fair"] - p["rho_old"]), reverse=True)
    return {
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "mean_abs_diff": float(np.mean(np.abs(fair - old))),
        "largest_abs_disagreements": [
            {**p, "abs_diff": abs(p["rho_fair"] - p["rho_old"])} for p in diffs[:5]
        ],
    }


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    verdicts = load_verdicts()
    points, spread_disagreements = load_points(verdicts)

    n_main, rec_main = render_main(points, verdicts)
    if n_main != 4 * len(GROUPS) * len(METHOD_SLOTS):
        raise SystemExit(f"main figure plotted {n_main} bars, expected 80")
    n_var, rec_var = render_avg_variants(points, verdicts)
    if n_var != 2 * len(GROUPS) * len(METHOD_SLOTS):
        raise SystemExit(f"variants figure plotted {n_var} bars, expected 40")

    pairs = collect_fair_pairs(points, verdicts)
    scatter_stats = render_old_vs_fair(pairs)

    sidecar = {
        "source_points": str(POINTS.relative_to(ROOT)),
        "source_fair_points": str(FAIR_POINTS.relative_to(ROOT)),
        "source_spread": "eval_results/issue_1739/result1_spread/spread_stats.json",
        "spread_crosscheck_disagreements": spread_disagreements,
        "caption_main": CAPTION_MAIN,
        "caption_avg_variants": CAPTION_VARIANTS,
        "caption_old_vs_fair": CAPTION_SCATTER,
        "main_bars": rec_main,
        "avg_variant_bars": rec_var,
        "old_vs_fair_pairs": pairs,
        "old_vs_fair_stats": scatter_stats,
    }
    (OUT_FIG / "fivemethod_values.json").write_text(
        json.dumps(sidecar, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT_FIG / 'result2_fivemethod.png'} ({n_main} bars)")
    print(f"wrote {OUT_FIG / 'result2_fivemethod_avg_variants.png'} ({n_var} bars)")
    print(f"wrote {OUT_FIG / 'result2_fivemethod_old_vs_fair.png'} ({len(pairs)} points)")
    print(f"wrote {OUT_FIG / 'fivemethod_values.json'}")
    print(
        f"spread cross-check: {'NO disagreements' if not spread_disagreements else spread_disagreements}"
    )
    print(
        f"old-vs-fair: pearson_r={scatter_stats['pearson_r']:.3f} "
        f"spearman_r={scatter_stats['spearman_r']:.3f} "
        f"mean_abs_diff={scatter_stats['mean_abs_diff']:.3f}"
    )


if __name__ == "__main__":
    main()
