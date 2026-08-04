"""Result 1 for #1739: spread of every behavior in every evaluation setting.

Renders `figures/issue_1739/result1_spread/spread_grid.{png,pdf,meta.json}` and
writes the per-cell statistics table to
`eval_results/issue_1739/result1_spread/spread_stats.json`.

Pure aggregation over already-committed per-context judged DV rows — no fits, no
GPU, no network, no new judging. Reuses the canonical evaluation-setting roster
and labels from `issue1739_recut_common` (extended here with the persona-vectors
synthetic suite, which that module's arm-figure roster omits).

Artifact map (per (behavior x evaluation setting) cell):
  eval_results/issue_1739/dv_dataset/<b>/labeling.json
      the behavior's own rungs. evil + sycophancy carry `per_rollout_scores`
      (graded trait rubric, 0-100); hallucination carries three-way
      correct/abstained/fabricated `counts` whose DV is the fabricated
      FRACTION -- a DIFFERENT construct, rescaled x100 onto the 0-100 scale.
  eval_results/issue_1739/pvsynth/dv_dataset/<b>/labeling.json
      the persona-vectors synthetic suite (graded trait rubric, 0-100).
  <worktree>/wildchat_rung/dv_dataset/<b>/labeling.json
      the random-WildChat generic held-out rung (graded trait rubric, 0-100).
      Per-context rows are worktree-resident (not committed at repo root); the
      committed repo-root aggregate `wildchat_rung/spread/<b>.json` carries only
      (n, mean, sd, histogram), so the per-rollout scores the noise floor needs
      come from the worktree copy. PARITY_CHECKS below reconciles the
      recomputed (n, mean, sd) against that committed aggregate for all six
      wildchat + pvsynth cells; a mismatch raises.

Per-cell statistics (m_i = rollouts kept for context i):
  ybar_i     per-context mean judged score
  SD         between-context SD of ybar_i (ddof=1)
  floor      sqrt(mean_i(s2_i / m_i)), s2_i = within-context variance across
             context i's rollout scores (ddof=1) -- the sampling SD the
             per-context mean inherits, i.e. the noise floor of the DV
  r_yy       max(0, (SD^2 - floor^2) / SD^2), reliability of the context-level DV
  ceiling    sqrt(r_yy), the attenuation ceiling on any correlation with ybar
  fc_mass    fraction of contexts with ybar_i < 5 or > 95 (floor/ceiling mass)
  tie_mass   fraction of contexts sharing the modal ybar_i value
  mdc        minimum detectable correlation, 1.96 / sqrt(n - 3)

Three-part spread criterion (all three required):
  (1) fc_mass < 0.90   (2) r_yy >= 0.5   (3) mdc <= 0.5 * ceiling

The pre-registered gate (plan section "Pre-registered spread floor + fallback";
gates.gate2_spread_floor -- SD >= 10 AND < 80% of contexts in the bottom [0, 10)
bin) is recomputed alongside it for comparison with the shipped 2026-08-01
verdict; it is NOT re-run here, only decomposed per setting.
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from issue1739_recut_common import (  # noqa: E402
    BEHAVIORS,
    ROOT,
    RUNG_LABEL,
    RUNGS,
    WT,
)

OUT_FIG = ROOT / "figures/issue_1739/result1_spread"
OUT_NUM = ROOT / "eval_results/issue_1739/result1_spread"
ER = ROOT / "eval_results/issue_1739"

# --- roster ------------------------------------------------------------------
# recut_common's RUNGS is the arm-figure roster (own rungs + WildChat). The
# spread read additionally covers the persona-vectors synthetic suite, which has
# its own committed labeling.json. Reporting order: own train rung, then the
# behavior-specific OOD corpora, then generic WildChat, then the synthetic suite.
PVSYNTH = "pvsynth"
SETTINGS = {b: [*RUNGS[b], PVSYNTH] for b in BEHAVIORS}
SETTING_LABEL = dict(RUNG_LABEL)
for _b in BEHAVIORS:
    SETTING_LABEL[(_b, PVSYNTH)] = "persona-vectors\nsynthetic suite"

# One colour = one evaluation-setting CLASS, in both panels and every row.
CLASS_OF = {
    "train": "held_out_train",
    "wildchat_rung": "wildchat",
    PVSYNTH: "pvsynth",
}
CLASS_ORDER = ["held_out_train", "behavior_ood", "wildchat", "pvsynth"]
CLASS_LABEL = {
    "held_out_train": "held-out slice of the behavior's own training distribution",
    "behavior_ood": "behavior-specific out-of-distribution corpus",
    "wildchat": "random WildChat (ordinary user traffic)",
    "pvsynth": "persona-vectors synthetic test suite",
}
_pal = paper_palette(len(CLASS_ORDER))
CLASS_COLOR = dict(zip(CLASS_ORDER, _pal))

# Hallucination's own rungs are the fabricated FRACTION, not the graded trait
# score: a different construct, rescaled onto the gate's 0-100 scale.
RATE_CELLS = {("hallucination", r) for r in ("train", "nqopen", "simpleqa")}

# Three-part spread criterion.
FC_MASS_MAX = 0.90
R_YY_MIN = 0.50
MDC_CEILING_FRAC = 0.50
# Pre-registered gate 2 (recomputed for comparison, not re-run).
GATE2_SD_FLOOR = 10.0
GATE2_BOTTOM_BIN_EDGE = 10.0
GATE2_BOTTOM_FRAC_MAX = 0.80


def setting_class(rung: str) -> str:
    return CLASS_OF.get(rung, "behavior_ood")


def _rollout_vectors_graded(rows: list[dict]) -> dict[str, list[np.ndarray]]:
    """Per-context rollout-score vectors, keyed by rung. Drops null rollouts."""
    out: dict[str, list[np.ndarray]] = {}
    for r in rows:
        if r.get("dv") is None:
            continue
        v = np.array([x for x in r["per_rollout_scores"].values() if x is not None], dtype=float)
        if v.size == 0:
            continue
        out.setdefault(r.get("rung"), []).append(v)
    return out


def _rollout_vectors_rate(rows: list[dict]) -> dict[str, list[np.ndarray]]:
    """Hallucination's own rungs: rebuild the per-rollout fabricated indicators.

    `dv` is counts.fabricated / n_decided, so the n_decided rollout-level labels
    are exactly n_fabricated ones and (n_decided - n_fabricated) zeros. Scaled
    x100 onto the gate's 0-100 scale.
    """
    out: dict[str, list[np.ndarray]] = {}
    for r in rows:
        n_dec = int(r["n_decided"])
        if n_dec <= 0:
            continue
        n_fab = int(r["counts"]["fabricated"])
        out.setdefault(r.get("rung"), []).append(
            np.array([100.0] * n_fab + [0.0] * (n_dec - n_fab), dtype=float)
        )
    return out


def cell_stats(per_ctx: list[np.ndarray]) -> dict:
    """Between-context spread, noise floor, reliability and detectability."""
    ybar = np.array([v.mean() for v in per_ctx], dtype=float)
    m = np.array([v.size for v in per_ctx], dtype=float)
    s2 = np.array([v.var(ddof=1) if v.size > 1 else np.nan for v in per_ctx], dtype=float)
    usable = ~np.isnan(s2)
    n = int(ybar.size)
    sd = float(ybar.std(ddof=1)) if n > 1 else 0.0
    floor = float(np.sqrt(np.mean(s2[usable] / m[usable]))) if usable.any() else float("nan")
    r_yy = max(0.0, (sd**2 - floor**2) / sd**2) if sd > 0 and floor == floor else 0.0
    ceiling = float(np.sqrt(r_yy))
    vals, counts = np.unique(np.round(ybar, 9), return_counts=True)
    mdc = float(1.96 / np.sqrt(n - 3)) if n > 3 else float("nan")
    fc_mass = float(((ybar < 5.0) | (ybar > 95.0)).mean())
    bottom_frac = float((ybar < GATE2_BOTTOM_BIN_EDGE).mean())
    crit = {
        "floor_ceiling_mass_ok": fc_mass < FC_MASS_MAX,
        "reliability_ok": r_yy >= R_YY_MIN,
        "detectability_ok": (mdc <= MDC_CEILING_FRAC * ceiling) if mdc == mdc else False,
    }
    return {
        "n_contexts": n,
        "mean": float(ybar.mean()),
        "sd_between_context": sd,
        "noise_floor": floor,
        "r_yy": r_yy,
        "ceiling_sqrt_r_yy": ceiling,
        "floor_ceiling_mass": fc_mass,
        "tie_mass": float(counts.max() / n),
        "tie_value": float(vals[counts.argmax()]),
        "min_detectable_rho": mdc,
        "rollouts_per_context_median": float(np.median(m)),
        "n_contexts_single_rollout": int((~usable).sum()),
        "bottom_bin_frac": bottom_frac,
        "criterion": crit,
        "criterion_verdict": "PASS" if all(crit.values()) else "FAIL",
        "criterion_failing_clauses": [k for k, ok in crit.items() if not ok],
        "prereg_gate2": (
            "PASS" if (sd >= GATE2_SD_FLOOR and bottom_frac < GATE2_BOTTOM_FRAC_MAX) else "FAIL"
        ),
        "prereg_gate2_failing_clauses": [
            k
            for k, ok in (
                ("sd_floor", sd >= GATE2_SD_FLOOR),
                ("bottom_bin", bottom_frac < GATE2_BOTTOM_FRAC_MAX),
            )
            if not ok
        ],
        "_ybar": ybar,
    }


# --- load --------------------------------------------------------------------
PROVENANCE: dict[str, dict] = {}
PARITY_CHECKS: list[dict] = []
CELLS: dict[tuple[str, str], dict] = {}


def _meta(path: Path, d: dict) -> dict:
    src = d.get("meta") or d.get("judge_meta") or {}
    return {
        "artifact": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "committed_at_repo_root": path.is_relative_to(ER),
        "judge_model": src.get("judge_model"),
        "judge_temperature": src.get("judge_temperature"),
        "n_judge_draws": src.get("n_judge_draws") or src.get("n_draws"),
        "judge_max_tokens": src.get("judge_max_tokens"),
        "rejudge_max_tokens": src.get("rejudge_max_tokens"),
        "rubric": src.get("rubric"),
        "git_commit": d.get("git_commit") or src.get("git_commit"),
        "ts": d.get("ts") or src.get("ts"),
        "dv_construct": d.get("dv_construct"),
    }


for b in BEHAVIORS:
    own_path = ER / "dv_dataset" / b / "labeling.json"
    own = json.loads(own_path.read_text())
    vecs = (
        _rollout_vectors_rate(own["rows"])
        if b == "hallucination"
        else _rollout_vectors_graded(own["rows"])
    )
    for rung, per_ctx in vecs.items():
        CELLS[(b, rung)] = cell_stats(per_ctx)
    PROVENANCE[f"{b}|own_rungs"] = _meta(own_path, own) | {
        "rollouts_per_context_planned": 5,
        "note": (
            "fabricated fraction over three-way judged rollouts, rescaled x100"
            if b == "hallucination"
            else "graded trait rubric, mean over judge draws per rollout"
        ),
    }

    for rung, path in (
        ("wildchat_rung", WT / "wildchat_rung/dv_dataset" / b / "labeling.json"),
        (PVSYNTH, ER / "pvsynth/dv_dataset" / b / "labeling.json"),
    ):
        d = json.loads(path.read_text())
        per_ctx = next(iter(_rollout_vectors_graded(d["rows"]).values()))
        CELLS[(b, rung)] = cell_stats(per_ctx)
        PROVENANCE[f"{b}|{rung}"] = _meta(path, d)

        # Reconcile against the committed repo-root aggregate for this cell.
        agg_path = ER / ("wildchat_rung" if rung == "wildchat_rung" else PVSYNTH)
        agg = json.loads((agg_path / "spread" / f"{b}.json").read_text())["spread"]
        got = CELLS[(b, rung)]
        check = {
            "cell": f"{b}/{rung}",
            "committed_aggregate": str((agg_path / "spread" / f"{b}.json").relative_to(ROOT)),
            "n": [agg["n"], got["n_contexts"]],
            "mean": [agg["mean"], got["mean"]],
            "sd": [agg["sd"], got["sd_between_context"]],
        }
        check["ok"] = (
            agg["n"] == got["n_contexts"]
            and abs(agg["mean"] - got["mean"]) < 1e-6
            and abs(agg["sd"] - got["sd_between_context"]) < 1e-6
        )
        PARITY_CHECKS.append(check)
        if not check["ok"]:
            raise SystemExit(f"parity check failed against committed aggregate: {check}")

for b in BEHAVIORS:
    missing = [s for s in SETTINGS[b] if (b, s) not in CELLS]
    if missing:
        raise SystemExit(f"{b}: no per-context DV rows for settings {missing}")

# --- numbers out -------------------------------------------------------------
OUT_NUM.mkdir(parents=True, exist_ok=True)
TABLE = []
for b in BEHAVIORS:
    for s in SETTINGS[b]:
        row = {k: v for k, v in CELLS[(b, s)].items() if not k.startswith("_")}
        TABLE.append(
            {
                "behavior": b,
                "setting": s,
                "setting_label": SETTING_LABEL[(b, s)].replace("\n", " "),
                "setting_class": setting_class(s),
                "dv_construct": (
                    "fabricated_fraction_rescaled_x100"
                    if (b, s) in RATE_CELLS
                    else "trait_rubric_graded_0_100"
                ),
                **row,
            }
        )
CAPTION = (
    "Spread of judged behavior expression across every evaluation setting realized in #1739 "
    "(14 behavior x setting cells). LEFT: binned distribution (25 bins of width 4) of the "
    "per-context mean judged score; the tick marks the setting mean; n is the number of "
    "contexts carrying a DV. RIGHT: between-context SD of that score, with each cell's own "
    "noise floor sqrt(mean_i(s2_i/m_i)) overlaid as a line and the attenuation ceiling "
    "sqrt(r_yy) annotated. Generation recipe as recorded in each artifact: 5 sampled rollouts "
    "per context, each scored by claude-sonnet-4-5-20250929 over 3 judge draws at temperature "
    "1.0 (max_tokens 400, re-judged at 800 on content drops); the graded rubric is a 0-100 "
    "trait score. Hallucination's own rungs (held-out TriviaQA, NQ-Open, SimpleQA) instead "
    "carry the three-way correct/abstained/fabricated rubric and their DV is the fabricated "
    "FRACTION rescaled x100 - a DIFFERENT construct from the graded trait score in every "
    "other cell, not comparable to it. Per-cell artifact provenance: evil + sycophancy + "
    "hallucination own rungs from eval_results/issue_1739/dv_dataset/<behavior>/labeling.json; "
    "persona-vectors synthetic suite from eval_results/issue_1739/pvsynth/dv_dataset/"
    "<behavior>/labeling.json; random WildChat from the issue-1739 worktree's "
    "wildchat_rung/dv_dataset/<behavior>/labeling.json (per-context rows not committed at "
    "repo root; recomputed n, mean and SD reconcile exactly against the committed "
    "wildchat_rung/spread/<behavior>.json aggregate, 6/6 cells including pvsynth)."
)
(OUT_NUM / "spread_stats.json").write_text(
    json.dumps(
        {
            "caption": CAPTION,
            "criterion": {
                "floor_ceiling_mass_max": FC_MASS_MAX,
                "r_yy_min": R_YY_MIN,
                "min_detectable_rho_max_frac_of_ceiling": MDC_CEILING_FRAC,
            },
            "prereg_gate2": {
                "sd_floor": GATE2_SD_FLOOR,
                "bottom_bin_edge": GATE2_BOTTOM_BIN_EDGE,
                "bottom_frac_max": GATE2_BOTTOM_FRAC_MAX,
                "note": "recomputed per setting for comparison; not a re-run of the shipped verdict",
            },
            "cells": TABLE,
            "provenance": PROVENANCE,
            "parity_checks": PARITY_CHECKS,
        },
        indent=1,
    )
)

# --- figure ------------------------------------------------------------------
set_paper_style("blog")
# The blog style enables constrained_layout, which ignores the explicit
# subplots_adjust this figure needs to reserve room for the title block and the
# bottom legend (neither is a constrained-layout-managed artist). Cleared BEFORE
# the figure is created so no layout engine is ever attached.
plt.rcParams["figure.constrained_layout.use"] = False
n_rows = len(BEHAVIORS)
fig, axes = plt.subplots(
    n_rows,
    2,
    figsize=(15.0, 11.4),
    gridspec_kw={"width_ratios": [1.55, 1.0], "hspace": 0.42, "wspace": 0.09},
)

FLOOR_C = "#B00020"
# 25 bins of width 4 keep hallucination's discrete fabricated-fraction values
# (0, 20, ..., 100) in separate bins and put evil's 0-pile in one narrow bin.
BIN_EDGES = np.linspace(0.0, 100.0, 26)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0

for row, b in enumerate(BEHAVIORS):
    settings = SETTINGS[b]
    pos = np.arange(len(settings))[::-1]  # first setting at the top
    colors = [CLASS_COLOR[setting_class(s)] for s in settings]

    # -- LEFT: distribution of the per-context mean judged score.
    # An UNSMOOTHED binned profile, not a KDE violin: hallucination's own-rung
    # DV is the fabricated fraction over 5 rollouts, so it takes ~6 discrete
    # values (58% of SimpleQA contexts sit exactly at 100) and a KDE smears
    # those spikes into a spurious continuum; binning also cannot leak mass
    # outside the DV's bounded [0, 100] support on the floor-piled cells.
    axl = axes[row][0]
    for p, s, c in zip(pos, settings, colors):
        dens, _ = np.histogram(CELLS[(b, s)]["_ybar"], bins=BIN_EDGES, range=(0.0, 100.0))
        h = dens / dens.max() * 0.42 if dens.max() else np.zeros_like(dens, dtype=float)
        axl.fill_between(
            BIN_CENTERS,
            p - h,
            p + h,
            step="mid",
            facecolor=c,
            linewidth=0.0,
            alpha=0.85,
        )
    axl.scatter(
        [CELLS[(b, s)]["mean"] for s in settings],
        pos,
        marker="|",
        s=170,
        linewidths=1.6,
        color="#1A1A1A",
        zorder=5,
        label="setting mean",
    )
    axl.set_yticks(pos)
    axl.set_yticklabels(
        [
            f"{SETTING_LABEL[(b, s)]}\nn={CELLS[(b, s)]['n_contexts']:,}"
            + ("  (fab. rate x100)" if (b, s) in RATE_CELLS else "")
            for s in settings
        ],
        fontsize=8.5,
    )
    axl.set_xlim(-2, 102)
    axl.set_ylim(pos.min() - 0.65, pos.max() + 0.65)
    axl.set_xlabel("per-context mean judged behavior score (0-100)")
    axl.set_title(b, fontsize=12, loc="left")

    # -- RIGHT: between-context SD with the cell's own noise floor overlaid
    axr = axes[row][1]
    sds = [CELLS[(b, s)]["sd_between_context"] for s in settings]
    axr.barh(pos, sds, color=colors, height=0.62, alpha=0.90)
    for p, s in zip(pos, settings):
        c = CELLS[(b, s)]
        axr.plot(
            [c["noise_floor"], c["noise_floor"]],
            [p - 0.31, p + 0.31],
            color=FLOOR_C,
            lw=1.9,
            solid_capstyle="butt",
            zorder=4,
        )
        axr.text(
            c["sd_between_context"] + 1.0,
            p,
            f"ceiling {c['ceiling_sqrt_r_yy']:.2f}",
            va="center",
            ha="left",
            fontsize=8,
            color="#3A3A3A",
        )
    axr.set_yticks(pos)
    axr.set_yticklabels([])
    axr.set_ylim(pos.min() - 0.65, pos.max() + 0.65)
    axr.set_xlim(0, 56)
    axr.set_xlabel("between-context SD of the score (0-100 units)")
    axr.set_title(
        "spread vs its own noise floor",
        fontsize=10.5,
        loc="left",
        color="#5A5A5A",
    )

handles = [Patch(facecolor=CLASS_COLOR[c], label=CLASS_LABEL[c]) for c in CLASS_ORDER]
handles += [
    Line2D([], [], color=FLOOR_C, lw=1.9, label="noise floor of the per-context mean"),
    Line2D(
        [],
        [],
        color="#1A1A1A",
        lw=0,
        marker="|",
        markersize=9,
        markeredgewidth=1.6,
        label="setting mean",
    ),
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=9,
    bbox_to_anchor=(0.5, -0.005),
)
fig.suptitle(
    "Spread of judged behavior expression across every evaluation setting",
    x=0.008,
    y=0.995,
    ha="left",
    fontsize=14,
    fontweight="semibold",
    color="#1A1A1A",
)
fig.text(
    0.008,
    0.965,
    "Left: binned distribution (25 bins) of the per-context mean judged score. Right: between-context SD of that score, "
    "with the noise floor sqrt(mean_i(s2_i/m_i)) it inherits from sampling 5 rollouts per context, "
    "and the\nattenuation ceiling sqrt(r_yy) on any correlation with it. Judge: claude-sonnet-4-5-20250929, "
    "graded 0-100. Hallucination's own rungs are the fabricated fraction rescaled x100 - a DIFFERENT "
    "construct from the graded\ntrait score in the other bars, not comparable to them.",
    ha="left",
    va="top",
    fontsize=9,
    color="#5A5A5A",
)
fig.subplots_adjust(left=0.145, right=0.985, top=0.905, bottom=0.075, hspace=0.42, wspace=0.05)
savefig_paper(fig, "spread_grid", dir=OUT_FIG)
plt.close(fig)

# --- console summary ---------------------------------------------------------
print(f"wrote {OUT_FIG / 'spread_grid.png'}")
print(f"wrote {OUT_NUM / 'spread_stats.json'}")
print(
    f"parity vs committed aggregates: {sum(c['ok'] for c in PARITY_CHECKS)}"
    f"/{len(PARITY_CHECKS)} cells reconciled"
)
hdr = (
    f"{'cell':34s} {'n':>6s} {'SD':>7s} {'floor':>7s} {'r_yy':>6s} {'ceil':>6s} "
    f"{'fc':>6s} {'tie':>6s} {'mdc':>6s} {'crit':>5s} {'gate2':>5s}"
)
print(hdr)
print("-" * len(hdr))
for t in TABLE:
    print(
        f"{t['behavior'] + '/' + t['setting']:34s} {t['n_contexts']:6d} "
        f"{t['sd_between_context']:7.2f} {t['noise_floor']:7.2f} {t['r_yy']:6.3f} "
        f"{t['ceiling_sqrt_r_yy']:6.3f} {t['floor_ceiling_mass']:6.3f} {t['tie_mass']:6.3f} "
        f"{t['min_detectable_rho']:6.3f} {t['criterion_verdict']:>5s} {t['prereg_gate2']:>5s}"
    )
