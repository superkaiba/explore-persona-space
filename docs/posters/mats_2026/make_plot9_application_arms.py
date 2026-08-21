"""Poster plot 9 — application arms: PV vs regression readouts across regimes.

Extends the c5_pv_methods_regimes comparison (persona-vector readouts across
evaluation regimes) with the two supervised-regression arms, so the figure
carries the claim "the mapped-answer route beats both the direct-from-context
route and the persona-vector route" with the regression readout family shown
explicitly. Three behavior panels (evil / sycophancy / hallucination), four
regime groups on x (synthetic, generic chat, in-distrib., OOD), FIVE bars per
group:

  pv_context      arm1_ctx_e1          PV readout on the context vector v_C
  regression_ctx  arm4_ridge_ctx       ridge regression fit on v_C (no map)
  pv_map_linear   arm6_map_proj_e1     PV readout on the mapped answer v_A_hat
                                       (linear map)
  reg_map_linear  arm7_map_ridge_pred  ridge regression fit on v_A_hat
                                       (linear map)
  oracle          arm11_oracle_proj    PV readout on the REAL answer (oracle)

The MLP-map PV arm (pv_map_mlp) from the c5 figure is DROPPED: six bars per
group is illegible at poster-column width, and for the context-vs-mapped
claim the linear-map arm carries the same comparison (the two track each
other closely everywhere except sycophancy, where the MLP map is stronger —
see the data JSON for its values). reg_map_mlp / arm19 (MLP readouts) are
likewise excluded.

Bars within a group are ordered BY READOUT FAMILY -- the three persona-vector
arms (context, mapped answer, real answer/oracle), then the two regression arms
(context, mapped answer) -- so each family reads as one contiguous block.
Color follows the same split: blues for the persona-vector readout (light =
context, dark = mapped answer) with purple for its oracle, warm for the ridge
regression (light = context, dark = mapped answer). Spread-gate-failed cells
(evil: generic chat + both OOD rungs, per the committed Result 1 gate) are
alpha-muted, never deleted; the poster legend renders that gate in reader terms
("behavior almost never occurs here") because all three failing cells are cells
where 92-99% of contexts sit at the floor of the judged score.

Every number is read from the committed fair-protocol points table
eval_results/issue_1739/result2_fair/result2_fair_points.json (output of
scripts/issue1739_result2fair_fig.py::collect); spread verdicts from
eval_results/issue_1739/result1_spread/spread_stats.json. Multi-rung OOD
groups take the simple mean of rung rhos; the interval is the elementwise
mean of the rung CI bounds (same convention as the committed
scripts/issue1739_result2_fourpanel_fig.py — conservative, NOT a resampled
CI of the mean). Pure re-render of committed artifacts: no fits, no GPU.

Run:
    uv run python docs/posters/mats_2026/make_plot9_application_arms.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from explore_persona_space.analysis.paper_plots import (
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[3]
POINTS = REPO / "eval_results/issue_1739/result2_fair/result2_fair_points.json"
SPREAD_STATS = REPO / "eval_results/issue_1739/result1_spread/spread_stats.json"
OUT_DIR = Path(__file__).resolve().parent / "figures"

BEHAVIORS = ["evil", "sycophancy", "hallucination"]

# (method slot, short legend label, color). Blues = PV readout family
# (light = context, dark = mapped answer); warm = ridge regression family
# (light = context, dark = mapped answer); purple = oracle. All Wong
# colorblind-safe hues from paper_plots.PAPER_COLORS.
# Legend labels are spelled out rather than abbreviated: this is the poster's
# application figure and "PV · mapped ans." is unreadable to anyone who has not
# already read the methods.
# Bars are grouped BY READOUT FAMILY -- all three persona-vector arms, then both
# regression arms -- rather than interleaved by location. Interleaving made the
# reader alternate families to compare within one, which is the comparison the
# figure is for; contiguous families let each be read as a block.
METHODS = [
    ("pv_context", "persona vector on the context", "#56B4E9"),
    ("pv_map_linear", "persona vector on the mapped answer", "#0072B2"),
    ("oracle", "persona vector on the real answer (oracle)", "#CC79A7"),
    ("regression_ctx", "regression on the context", "#E69F00"),
    ("reg_map_linear", "regression on the mapped answer", "#D55E00"),
]
METHOD_SLOTS = [m for m, _l, _c in METHODS]
LABEL = {m: lbl for m, lbl, _c in METHODS}
COLOR = {m: c for m, _l, c in METHODS}
ARM_ID = {
    "pv_context": ("arm1_ctx_e1", "linear"),
    "regression_ctx": ("arm4_ridge_ctx", "linear"),
    "pv_map_linear": ("arm6_map_proj_e1", "linear"),
    "reg_map_linear": ("arm7_map_ridge_pred", "linear"),
    "oracle": ("arm11_oracle_proj", "linear"),
}

GROUPS = ["synthetic", "generic chat", "in-distrib.", "OOD"]
# x tick text: at poster font size the full group names ran into each other in a
# three-panel row, so the tick labels are abbreviated and the full names stay in
# the group keys, the data JSON, and the poster prose
GROUP_TICK = {
    "synthetic": "synth.",
    "generic chat": "chat",
    "in-distrib.": "in-dist.",
    "OOD": "OOD",
}
GROUP_SETTINGS = {
    beh: {
        "synthetic": ["pvsynth"],
        "generic chat": ["wildchat_rung"],
        "in-distrib.": ["train"],
        "OOD": ood,
    }
    for beh, ood in {
        "evil": ["hhrt", "toxicchat"],
        "sycophancy": ["aita"],
        "hallucination": ["nqopen", "simpleqa"],
    }.items()
}

FAIL_ALPHA = 0.35
GROUP_WIDTH = 0.8
BAR_WIDTH = GROUP_WIDTH / len(METHOD_SLOTS)


def load_points() -> dict[tuple[str, str, str], dict]:
    """{(behavior, setting, method): point record} for the five figure arms."""
    doc = json.loads(POINTS.read_text())
    method_of = {v: k for k, v in ARM_ID.items()}
    out: dict[tuple[str, str, str], dict] = {}
    for p in doc["points"]:
        slot = method_of.get((p["arm_id"], p.get("map_kind", "linear")))
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


def load_verdicts() -> dict[tuple[str, str], str]:
    """{(behavior, setting): PASS|FAIL} from the committed Result 1 spread gate."""
    doc = json.loads(SPREAD_STATS.read_text())
    return {(c["behavior"], c["setting"]): c["criterion_verdict"] for c in doc["cells"]}


def group_bar(points: dict, verdicts: dict, beh: str, group: str, method: str) -> dict:
    """One bar: rho + CI (+ spread verdict + provenance) for (behavior, group, method).

    Multi-rung OOD groups take the SIMPLE MEAN of rung rhos; the interval is
    the elementwise mean of the rung CI bounds (conservative, not a resampled
    CI of the mean) — the committed fourpanel figure's convention.
    """
    settings = GROUP_SETTINGS[beh][group]
    recs = [points[(beh, s, method)] for s in settings]
    rho = float(np.mean([r["rho"] for r in recs]))
    ci = [
        float(np.mean([r["ci"][0] for r in recs])),
        float(np.mean([r["ci"][1] for r in recs])),
    ]
    failing = [s for s in settings if verdicts[(beh, s)] == "FAIL"]
    if failing and len(failing) != len(settings):
        print(f"WARNING: partial spread-fail in {beh}/{group}: {failing}")
    return {
        "behavior": beh,
        "group": group,
        "method": method,
        "arm_id": ARM_ID[method][0],
        "map_kind": ARM_ID[method][1],
        "rho": rho,
        "ci": ci,
        "settings": settings,
        "per_setting": {
            s: {
                "rho": points[(beh, s, method)]["rho"],
                "ci": points[(beh, s, method)]["ci"],
                "n_eval": points[(beh, s, method)]["n_eval"],
                "source_file": points[(beh, s, method)]["source_file"],
            }
            for s in settings
        },
        "spread_failed": bool(failing),
        "failing_settings": failing,
    }


def main() -> None:
    points = load_points()
    verdicts = load_verdicts()

    panels = []
    for beh in BEHAVIORS:
        groups = []
        for g in GROUPS:
            bars = [group_bar(points, verdicts, beh, g, m) for m in METHOD_SLOTS]
            groups.append({"group": g, "bars": bars})
        panels.append({"behavior": beh, "groups": groups})

    vals = [v for p in panels for g in p["groups"] for b in g["bars"] for v in (b["rho"], *b["ci"])]
    ylim = (min(-0.05, min(vals) - 0.04), max(vals) + 0.04)

    set_paper_style("iclr", font_scale=1.9)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.95), sharey=True, constrained_layout=True)
    n_bars = 0
    for ax, panel in zip(axes, panels, strict=True):
        xs = list(range(len(panel["groups"])))
        ax.axhline(0.0, color="#B0B0B0", linewidth=0.6, zorder=1)
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
                    zorder=3,
                )
                lo, hi = bar["ci"]
                # Non-negative offsets from the value, never raw bounds.
                err_lo = max(0.0, bar["rho"] - lo)
                err_hi = max(0.0, hi - bar["rho"])
                ax.errorbar(
                    [x + offset],
                    [bar["rho"]],
                    yerr=np.array([[err_lo], [err_hi]]),
                    fmt="none",
                    ecolor="#333333",
                    elinewidth=0.5,
                    capsize=0,
                    zorder=4,
                )
                n_bars += 1
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [GROUP_TICK[g] for g in GROUPS],
            rotation=40,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_xlim(-0.6, len(xs) - 0.4)
        ax.set_ylim(*ylim)
        ax.set_title(panel["behavior"].capitalize())
    axes[0].set_ylabel("Spearman $\\rho$")

    handles = [Patch(facecolor=COLOR[m], label=LABEL[m]) for m in METHOD_SLOTS]
    handles.append(
        # "spread gate" named the internal criterion, not what it means to a
        # reader. All three faded cells fail it the same way: 92-99% of contexts
        # sit at the floor of the judged score, so there is essentially no
        # behavior to predict and rho is uninformative.
        Patch(
            facecolor="#999999",
            alpha=FAIL_ALPHA,
            label="faded = behavior almost never occurs here",
        )
    )
    # two columns, not three: the labels are now spelled out and no longer fit
    # three-across at this width
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        columnspacing=1.4,
        handlelength=1.1,
        handletextpad=0.5,
    )

    if n_bars != len(BEHAVIORS) * len(GROUPS) * len(METHOD_SLOTS):
        raise SystemExit(f"plotted {n_bars} bars, expected 60")
    paths = savefig_paper(fig, "plot9_application_arms", dir=OUT_DIR)
    plt.close(fig)

    data_out = {
        "source_points": str(POINTS.relative_to(REPO)),
        "source_spread": str(SPREAD_STATS.relative_to(REPO)),
        "metric": "Spearman rho_frozen (prediction vs judged behaviour expression)",
        "arms_note": (
            "pv_map_mlp (arm6 + MLP map) and reg_map_mlp / arm19 (MLP readouts) are "
            "committed at every cell but EXCLUDED here: 6+ bars per group is illegible "
            "at poster-column width, and the linear-map arm carries the context-vs-mapped "
            "claim."
        ),
        "ood_note": (
            "OOD group = simple mean over the behavior's OOD rungs (evil: hh-rlhf "
            "red-team + ToxicChat; sycophancy: held-out Reddit AITA; hallucination: "
            "NQ-Open + SimpleQA); interval = elementwise mean of rung CI bounds — "
            "conservative, not a resampled CI of the mean."
        ),
        "spread_note": (
            "Alpha-muted groups FAIL the committed Result 1 spread gate (evil: "
            "wildchat_rung, hhrt, toxicchat) — kept for completeness, not interpretable. "
            "The poster legend states the failure in reader terms ('behavior almost "
            "never occurs here') rather than naming the gate: all three cells fail the "
            "floor_ceiling_mass clause with 91.9%/98.7%/99.4% of contexts pinned at the "
            "floor of the judged score (evil/hhrt additionally fails reliability, "
            "r_yy=0.481), so there is essentially no behavior variation for rho to track."
        ),
        "bars": [b for p in panels for g in p["groups"] for b in g["bars"]],
    }
    out_json = OUT_DIR / "plot9_application_arms_data.json"
    out_json.write_text(json.dumps(data_out, indent=2) + "\n", encoding="utf-8")
    for p in paths.values():
        print(p)
    print(out_json)


if __name__ == "__main__":
    main()
