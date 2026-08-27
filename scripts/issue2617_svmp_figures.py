"""#2617 SVMP figures (unit 3/3): hero two-panel + exploratory dump.

Consumes the reads outputs written by ``scripts/issue2617_svmp_reads.py``
(``eval_results/issue_2617/svmp/{summary.json,perpair.jsonl,percontext.jsonl}``,
or a scratch dir via ``--in-dir``) and renders, per plan section 4.6:

- HERO (two-panel): (left) per-class direction-cos strips + medians by arm,
  with the registered arm's shuffled-null p95 ticks and the #2564
  benign-anchor reference lines; (right) per-pair scatter of |delta refusal
  rate| vs direction cos, colored by pair class (the partial-rho read lives
  in caption PROSE, never on-canvas).
- Exploratory dump: pair-delta retrieval acc@1 bars (class x arm x pool,
  chance lines), calibration slopes, axis-loading flip vs non-flip,
  margin-vs-rate validation scatter, per-context refusal-rate manipulation
  check, |ans_len_delta| vs cos, span-vs-tail pooling twin, and the
  L14/L26 twin-layer table figure.

Conventions: `.claude/skills/paper-plots` — ``set_paper_style()`` +
``savefig_paper`` sidecars (commit + timestamp + per-point data), colorblind
safe, ONE color = ONE meaning across the whole set (arms = blog-role hexes,
pair classes = Wong hexes, disjoint), NO caption/provenance text blocks on
the canvas (axes + ticks + legend + panel titles only). A panel that would
render zero finite points RAISES (fail loud — an empty panel is a data bug,
never a figure to ship).

Run (VM):

    uv run python scripts/issue2617_svmp_figures.py            # committed reads -> figures/issue_2617
    uv run python scripts/issue2617_svmp_figures.py --in-dir /tmp/i2617-reads --out-dir /tmp/i2617-figs
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE numpy/matplotlib — shared-VM thread caps

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
assert (REPO_ROOT / "pyproject.toml").is_file(), REPO_ROOT

ISSUE = 2617
IN_DIR_DEFAULT = REPO_ROOT / "eval_results" / "issue_2617" / "svmp"
OUT_DIR_DEFAULT = REPO_ROOT / "figures" / "issue_2617"
# #2564 benign one-word anchors (committed artifact; plan section 4.6 hero-left
# reference lines). Registered-arm per-slot cos medians live at
# cos_median_by_axis_arm.arm_779ce.{query_oneword_subject,object,verb}.
ANCHOR_JSON_DEFAULT = REPO_ROOT / "eval_results" / "issue_2564" / "gramslot_pilot" / "summary.json"

REGISTERED_ARM = "arm_779ce"
ALL_ARMS = ("arm_779ce", "arm_1738ce", "arm_iddelta")
ARM_LABELS = {
    "arm_779ce": "Single-turn map",
    "arm_1738ce": "Multi-turn map",
    "arm_iddelta": "Raw context shift (identity+bias)",
}
PAIR_CLASSES = (
    "obj_flip",
    "verb_flip",
    "subj_ctl",
    "obj_benign",
    "verb_benign",
    "subj_benign",
    "xstest",
)
CLASS_LABELS = {
    "obj_flip": "Object swap\n(valence flip)",
    "verb_flip": "Verb swap\n(valence flip)",
    "subj_ctl": "Subject swap\n(harmful topic)",
    "obj_benign": "Object swap\n(benign)",
    "verb_benign": "Verb swap\n(benign)",
    "subj_benign": "Subject swap\n(benign)",
    "xstest": "XSTest\nsafe/unsafe",
}
FLIP_GROUPS = ("flip", "mid", "nonflip")
FLIP_GROUP_LABELS = {"flip": "Flip pairs", "mid": "Mid pairs", "nonflip": "Non-flip pairs"}
POOLS = ("full", "constructed", "xstest")
POOL_LABELS = {"full": "All pairs", "constructed": "Constructed pairs", "xstest": "XSTest pairs"}

# One color = one meaning across the whole figure set: arms draw from the
# blog-role palette, pair classes from the Wong palette (disjoint hex sets);
# grey #999999 = shuffled-null band, #5A5A5A = thresholds/diagonals/reference
# slopes, black dotted = #2564 benign anchors, black solid = observed delta.
NULL_GREY = "#999999"
REF_GREY = "#5A5A5A"
ANCHOR_BLACK = "#000000"
OBS_BLACK = "#000000"

JITTER_SEED = 2617
FLIP_HI = 0.5
NONFLIP_LO = 0.1


def _arm_colors() -> dict[str, str]:
    return {
        "arm_779ce": paper_palette_role("primary"),
        "arm_1738ce": paper_palette_role("accent"),
        "arm_iddelta": paper_palette_role("control"),
    }


def _class_colors() -> dict[str, str]:
    pal = paper_palette(len(PAIR_CLASSES))
    return dict(zip(PAIR_CLASSES, pal))


def _finite(vals) -> np.ndarray:
    arr = np.array(
        [float(v) for v in vals if v is not None and np.isfinite(float(v))], dtype=np.float64
    )
    return arr


def _require_points(n: int, panel: str) -> None:
    if n <= 0:
        raise ValueError(
            f"panel {panel!r}: zero finite points — refusing to render a blank panel "
            "(dry-run-judge reads input? see scripts/issue2617_svmp_reads.py)"
        )


def _strip(ax, x0: float, vals, color: str, rng, width: float = 0.07) -> int:
    arr = _finite(vals)
    if len(arr) == 0:
        return 0
    xs = x0 + rng.uniform(-width, width, size=len(arr))
    ax.scatter(xs, arr, s=14, color=color, alpha=0.65, linewidths=0, zorder=3)
    ax.hlines(float(np.median(arr)), x0 - 0.13, x0 + 0.13, color=color, lw=2.4, zorder=4)
    return len(arr)


def load_reads(in_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    summary = json.loads((in_dir / "summary.json").read_text())
    assert summary.get("issue") == ISSUE, summary.get("issue")
    rows = [json.loads(x) for x in (in_dir / "perpair.jsonl").read_text().split("\n") if x]
    ctx_rows = [json.loads(x) for x in (in_dir / "percontext.jsonl").read_text().split("\n") if x]
    assert rows and ctx_rows, (len(rows), len(ctx_rows))
    return summary, rows, ctx_rows


def load_anchors(anchor_json: Path) -> dict[str, float]:
    """#2564 registered-arm benign per-slot cos medians (subject/object/verb)."""
    if not anchor_json.is_file():
        raise FileNotFoundError(
            f"{anchor_json} — the #2564 benign-anchor artifact is required for the hero panel "
            "(sparse checkout? `git sparse-checkout add eval_results/issue_2564`), "
            "or pass --anchor-json"
        )
    s = json.loads(anchor_json.read_text())
    by_slot = s["cos_median_by_axis_arm"][REGISTERED_ARM]
    out = {k.replace("query_oneword_", ""): float(v) for k, v in by_slot.items()}
    assert out and all(np.isfinite(v) for v in out.values()), out
    return out


# ── figures ───────────────────────────────────────────────────────────────


def fig_hero(summary: dict, rows: list[dict], anchors: dict[str, float]) -> plt.Figure:
    arm_c = _arm_colors()
    cls_c = _class_colors()
    rng = np.random.default_rng(JITTER_SEED)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # Left: per-class cos strips by arm + registered-arm null p95 + anchors.
    n_left = 0
    offsets = {a: o for a, o in zip(ALL_ARMS, (-0.27, 0.0, 0.27))}
    by_class_p95 = summary["per_arm"][REGISTERED_ARM].get("by_class_p95", {})
    ticklabels = []
    for ci, cls in enumerate(PAIR_CLASSES):
        cls_rows = [r for r in rows if r["pair_class"] == cls]
        ticklabels.append(f"{CLASS_LABELS[cls]}\nn={len(cls_rows)}")
        for arm in ALL_ARMS:
            n_left += _strip(
                axl, ci + offsets[arm], [r[f"cos_{arm}"] for r in cls_rows], arm_c[arm], rng
            )
        p95 = by_class_p95.get(cls)
        if p95 is not None and np.isfinite(p95):
            axl.hlines(
                float(p95), ci - 0.42, ci + 0.42, color=NULL_GREY, lw=1.4, linestyle="--", zorder=2
            )
    _require_points(n_left, "hero-left per-class cos strips")
    for slot, val in sorted(anchors.items()):
        axl.axhline(val, color=ANCHOR_BLACK, lw=1.0, linestyle=":", zorder=1)
        del slot
    axl.axhline(0.0, color=REF_GREY, lw=0.8, zorder=1)
    axl.set_xticks(range(len(PAIR_CLASSES)), ticklabels, fontsize=7)
    axl.set_ylabel("Direction cos (predicted vs observed pair delta, tail)")
    axl.set_title("Per-class transport by arm")
    handles = [
        Line2D([], [], marker="o", linestyle="", color=arm_c[a], label=ARM_LABELS[a])
        for a in ALL_ARMS
    ]
    handles += [
        Line2D(
            [],
            [],
            color=NULL_GREY,
            linestyle="--",
            label="Shuffled-pair null p95 (single-turn map)",
        ),
        Line2D(
            [],
            [],
            color=ANCHOR_BLACK,
            linestyle=":",
            label="#2564 benign anchors (subject/object/verb)",
        ),
    ]
    axl.legend(handles=handles, loc="lower left", fontsize=7)

    # Right: |delta refusal rate| vs cos (registered arm), colored by class.
    n_right = 0
    for cls in PAIR_CLASSES:
        xs, ys = [], []
        for r in rows:
            if r["pair_class"] != cls:
                continue
            x, y = r["abs_flip"], r[f"cos_{REGISTERED_ARM}"]
            if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
                continue
            xs.append(float(x))
            ys.append(float(y))
        if xs:
            axr.scatter(
                xs,
                ys,
                s=26,
                color=cls_c[cls],
                alpha=0.8,
                linewidths=0,
                label=CLASS_LABELS[cls].replace("\n", " "),
            )
            n_right += len(xs)
    _require_points(n_right, "hero-right |flip| vs cos scatter")
    axr.axvline(NONFLIP_LO, color=REF_GREY, lw=0.9, linestyle="--", zorder=1)
    axr.axvline(FLIP_HI, color=REF_GREY, lw=0.9, linestyle="--", zorder=1)
    axr.set_xlabel("|Δ refusal rate| per pair (member a − member b)")
    axr.set_ylabel("Direction cos (single-turn map, tail)")
    axr.set_title("Flip magnitude vs transport")
    axr.legend(fontsize=7, loc="best")
    fig.suptitle("")
    return fig


def fig_retrieval(summary: dict) -> plt.Figure:
    arm_c = _arm_colors()
    rp = summary["retrieval_pair_rank"]
    fig, axes = plt.subplots(1, len(POOLS), figsize=(13.5, 4.4), sharey=True)
    n_bars = 0
    width = 0.26
    for ax, pool in zip(axes, POOLS):
        cats = ["all"] + [
            c
            for c in PAIR_CLASSES
            if any((rp[a].get(pool) or {}).get("by_class", {}).get(c) for a in ALL_ARMS)
        ]
        labels = []
        for cat in cats:
            if cat == "all":
                n_pool = next((rp[a][pool]["n_pool"] for a in ALL_ARMS if rp[a].get(pool)), 0)
                labels.append(f"All\nn={n_pool}")
            else:
                n_c = next(
                    (
                        (rp[a][pool]["by_class"].get(cat) or {}).get("n", 0)
                        for a in ALL_ARMS
                        if rp[a].get(pool)
                    ),
                    0,
                )
                labels.append(f"{CLASS_LABELS[cat]}\nn={n_c}")
        chance = None
        for ai_, arm in enumerate(ALL_ARMS):
            entry = rp[arm].get(pool)
            if entry is None:
                continue
            chance = entry.get("chance_at_1")
            vals = []
            for cat in cats:
                v = (
                    entry["acc_at_1"]
                    if cat == "all"
                    else (entry["by_class"].get(cat) or {}).get("acc_at_1")
                )
                vals.append(np.nan if v is None else float(v))
            xs = np.arange(len(cats)) + (ai_ - 1) * width
            ax.bar(xs, vals, width=width, color=arm_c[arm], label=ARM_LABELS[arm])
            n_bars += int(np.isfinite(np.asarray(vals)).sum())
        if chance is not None:
            ax.axhline(float(chance), color=NULL_GREY, lw=1.2, linestyle="--")
        ax.set_xticks(range(len(cats)), labels, fontsize=6.5)
        ax.set_title(POOL_LABELS[pool])
        ax.set_ylim(0, 1.05)
    _require_points(n_bars, "retrieval acc@1 bars")
    axes[0].set_ylabel("Pair-delta retrieval acc@1")
    handles = [
        Line2D([], [], marker="s", linestyle="", color=arm_c[a], label=ARM_LABELS[a])
        for a in ALL_ARMS
    ]
    handles.append(Line2D([], [], color=NULL_GREY, linestyle="--", label="Chance (1/n_pool)"))
    axes[0].legend(handles=handles, fontsize=7, loc="upper left")
    return fig


def fig_calibration(summary: dict) -> plt.Figure:
    arm_c = _arm_colors()
    cal = summary["calibration"]
    n_by_class = summary["n_pairs_by_class"]
    fg = summary["flip_groups"]
    cats = list(PAIR_CLASSES) + ["all", "flip", "nonflip"]
    labels = [f"{CLASS_LABELS[c]}\nn={n_by_class.get(c, 0)}" for c in PAIR_CLASSES] + [
        f"All pairs\nn={summary['n_pairs']}",
        f"Flip pairs\nn={fg['n_flip']}",
        f"Non-flip pairs\nn={fg['n_nonflip']}",
    ]
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    width = 0.26
    n = 0
    for ai_, arm in enumerate(ALL_ARMS):
        vals = []
        for cat in cats:
            if cat == "all":
                v = cal[arm].get("slope_all")
            elif cat == "flip":
                v = cal[arm].get("slope_flip")
            elif cat == "nonflip":
                v = cal[arm].get("slope_nonflip")
            else:
                v = cal[arm]["slope_by_class"].get(cat)
            vals.append(np.nan if v is None else float(v))
        xs = np.arange(len(cats)) + (ai_ - 1) * width
        ax.bar(xs, vals, width=width, color=arm_c[arm], label=ARM_LABELS[arm])
        n += int(np.isfinite(np.asarray(vals)).sum())
    _require_points(n, "calibration slope bars")
    ax.axhline(1.0, color=REF_GREY, lw=0.9, linestyle=":")
    ax.set_xticks(range(len(cats)), labels, fontsize=7)
    ax.set_ylabel("Calibration slope (pred ≈ slope × obs, through origin)")
    ax.set_title("Norm calibration by class and flip group")
    ax.legend(fontsize=7)
    return fig


def fig_axis_loading(rows: list[dict]) -> plt.Figure:
    arm_c = _arm_colors()
    rng = np.random.default_rng(JITTER_SEED)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    series = [(f"axis_cos_pred_{a}", ARM_LABELS[a], arm_c[a]) for a in ALL_ARMS]
    series.append(("axis_cos_obs", "Observed delta", OBS_BLACK))
    offs = (-0.3, -0.1, 0.1, 0.3)
    # Rows with flip_group == "undefined" (no live judge rates) get their own
    # labeled group rather than being silently dropped (r2, figure-count-labels).
    groups = list(FLIP_GROUPS)
    if any(r["flip_group"] == "undefined" for r in rows):
        groups.append("undefined")
    n = 0
    labels = []
    for gi, grp in enumerate(groups):
        grp_rows = [r for r in rows if r["flip_group"] == grp]
        labels.append(f"{FLIP_GROUP_LABELS.get(grp, 'Undefined')}\nn={len(grp_rows)}")
        for (key, _lab, col), off in zip(series, offs):
            n += _strip(ax, gi + off, [r[key] for r in grp_rows], col, rng, width=0.05)
    _require_points(n, "axis-loading strips")
    ax.axhline(0.0, color=REF_GREY, lw=0.8)
    ax.set_xticks(range(len(groups)), labels)
    ax.set_ylabel("Refusal-axis loading cos(Δ, r̂ flip axis, LOO)")
    ax.set_title("Flip-axis loading: flip vs non-flip pairs")
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=c, label=lab) for _k, lab, c in series
        ],
        fontsize=7,
    )
    return fig


def fig_margin_validation(ctx_rows: list[dict]) -> plt.Figure:
    cls_c = _class_colors()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    n = 0
    for cls in PAIR_CLASSES:
        xs, ys = [], []
        for c in ctx_rows:
            if c["pair_class"] != cls:
                continue
            m, r = c["margin"], c["refusal_rate"]
            if m is None or r is None or not (np.isfinite(m) and np.isfinite(r)):
                continue
            xs.append(float(m))
            ys.append(float(r))
        if xs:
            ax.scatter(
                xs,
                ys,
                s=22,
                color=cls_c[cls],
                alpha=0.8,
                linewidths=0,
                label=CLASS_LABELS[cls].replace("\n", " "),
            )
            n += len(xs)
    _require_points(n, "margin-vs-rate validation scatter")
    ax.set_xlabel("Teacher-forced opener margin (refusal − helpful, mean LN logP)")
    ax.set_ylabel("Refusal rate per context (judge)")
    ax.set_title("Margin DV validation: margin vs refusal rate")
    ax.legend(fontsize=6.5, loc="best")
    return fig


def fig_manipulation(rows: list[dict], ctx_rows: list[dict]) -> plt.Figure:
    cls_c = _class_colors()
    rng = np.random.default_rng(JITTER_SEED)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # Left: per-context refusal-rate strips by class, sides a/b as fill.
    n_left = 0
    labels = []
    for ci, cls in enumerate(PAIR_CLASSES):
        for side, off in (("a", -0.18), ("b", 0.18)):
            vals = _finite(
                c["refusal_rate"] for c in ctx_rows if c["pair_class"] == cls and c["side"] == side
            )
            if len(vals):
                xs = ci + off + rng.uniform(-0.07, 0.07, size=len(vals))
                face = OBS_BLACK if side == "a" else "none"
                ax_kw = {"facecolors": face, "edgecolors": OBS_BLACK, "linewidths": 0.7}
                axl.scatter(xs, vals, s=16, alpha=0.75, zorder=3, **ax_kw)
                n_left += len(vals)
        n_cls = sum(1 for c in ctx_rows if c["pair_class"] == cls)
        labels.append(f"{CLASS_LABELS[cls]}\nn={n_cls}")
    _require_points(n_left, "manipulation-check per-context strips")
    axl.set_xticks(range(len(PAIR_CLASSES)), labels, fontsize=6.5)
    axl.set_ylabel("Refusal rate per context (judge)")
    axl.set_title("Per-context refusal rate by class and pair member")
    axl.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=OBS_BLACK,
                label="Member a (harmful/variant · unsafe)",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor="none",
                markeredgecolor=OBS_BLACK,
                markeredgewidth=1.0,
                color=OBS_BLACK,
                label="Member b (benign/base · safe)",
            ),
        ],
        fontsize=7,
    )

    # Right: paired rate_b vs rate_a per pair, colored by class.
    n_right = 0
    for cls in PAIR_CLASSES:
        xs, ys = [], []
        for r in rows:
            if r["pair_class"] != cls:
                continue
            ra, rb = r["refusal_rate_a"], r["refusal_rate_b"]
            if ra is None or rb is None:
                continue
            xs.append(float(rb))
            ys.append(float(ra))
        if xs:
            axr.scatter(
                xs,
                ys,
                s=26,
                color=cls_c[cls],
                alpha=0.8,
                linewidths=0,
                label=CLASS_LABELS[cls].replace("\n", " "),
            )
            n_right += len(xs)
    _require_points(n_right, "manipulation-check a-vs-b scatter")
    axr.plot([0, 1], [0, 1], color=REF_GREY, lw=0.9, linestyle="--", zorder=1)
    axr.set_xlabel("Refusal rate, member b (benign / safe)")
    axr.set_ylabel("Refusal rate, member a (harmful / unsafe)")
    axr.set_title("Manipulation check: paired refusal rates")
    axr.legend(fontsize=6.5, loc="best")
    return fig


def fig_len_vs_cos(rows: list[dict]) -> plt.Figure:
    cls_c = _class_colors()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    n = 0
    for cls in PAIR_CLASSES:
        xs, ys = [], []
        for r in rows:
            if r["pair_class"] != cls:
                continue
            ld, cv = r["ans_len_delta"], r[f"cos_{REGISTERED_ARM}"]
            if ld is None or cv is None or not (np.isfinite(ld) and np.isfinite(cv)):
                continue
            xs.append(abs(float(ld)))
            ys.append(float(cv))
        if xs:
            ax.scatter(
                xs,
                ys,
                s=22,
                color=cls_c[cls],
                alpha=0.8,
                linewidths=0,
                label=CLASS_LABELS[cls].replace("\n", " "),
            )
            n += len(xs)
    _require_points(n, "|ans_len_delta| vs cos scatter")
    ax.set_xlabel("|Δ mean answer length| per pair (characters)")
    ax.set_ylabel("Direction cos (single-turn map, tail)")
    ax.set_title("Length confound read: answer-length delta vs transport")
    ax.legend(fontsize=6.5, loc="best")
    return fig


def fig_span_vs_tail(rows: list[dict]) -> plt.Figure:
    arm_c = _arm_colors()
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    n = 0
    for arm in ALL_ARMS:
        xs, ys = [], []
        for r in rows:
            t, s = r[f"cos_{arm}"], r[f"cos_span_{arm}"]
            if t is None or s is None or not (np.isfinite(t) and np.isfinite(s)):
                continue
            xs.append(float(t))
            ys.append(float(s))
        if xs:
            ax.scatter(
                xs, ys, s=20, color=arm_c[arm], alpha=0.7, linewidths=0, label=ARM_LABELS[arm]
            )
            n += len(xs)
    _require_points(n, "span-vs-tail scatter")
    lim = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lim), max(lim)
    ax.plot([lo, hi], [lo, hi], color=REF_GREY, lw=0.9, linestyle="--", zorder=1)
    ax.set_xlabel("Direction cos, tail pooling")
    ax.set_ylabel("Direction cos, span-mean pooling")
    ax.set_title("Pooling twin: span-mean vs tail per pair")
    ax.legend(fontsize=7)
    return fig


def fig_twin_layers(summary: dict) -> plt.Figure:
    arm_c = _arm_colors()
    cls_c = _class_colors()
    n_by_class = summary["n_pairs_by_class"]
    primary = int(summary["layers"]["primary"])
    layers = sorted([primary] + [int(x) for x in summary["layers"]["twins"]])

    def _by_class(arm: str, layer: int) -> dict:
        if layer == primary:
            return summary["per_arm"][arm]["cos_median_by_class"]
        return summary["twin_layers"][str(layer)][arm]["cos_median_by_class"]

    def _acc(arm: str, layer: int):
        if layer == primary:
            entry = summary["retrieval_pair_rank"][arm].get("full")
            return None if entry is None else entry.get("acc_at_1")
        return summary["twin_layers"][str(layer)][arm].get("pair_acc_at_1_full")

    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=(1.25, 1.0))
    n = 0
    top_axes = []
    for ai_, arm in enumerate(ALL_ARMS):
        ax = fig.add_subplot(gs[0, ai_])
        top_axes.append(ax)
        for cls in PAIR_CLASSES:
            ys = []
            for layer in layers:
                v = _by_class(arm, layer).get(cls)
                ys.append(np.nan if v is None else float(v))
            if np.isfinite(np.asarray(ys)).any():
                ax.plot(
                    layers,
                    ys,
                    marker="o",
                    ms=4,
                    lw=1.3,
                    color=cls_c[cls],
                    label=f"{CLASS_LABELS[cls].replace(chr(10), ' ')} (n={n_by_class.get(cls, 0)})",
                )
                n += int(np.isfinite(np.asarray(ys)).sum())
        ax.set_title(ARM_LABELS[arm])
        ax.set_xticks(layers, [f"L{x}" for x in layers])
        if ai_ == 0:
            ax.set_ylabel("Direction cos median per class (tail)")
    ylims = [ax.get_ylim() for ax in top_axes]
    lo, hi = min(y[0] for y in ylims), max(y[1] for y in ylims)
    for ax in top_axes:
        ax.set_ylim(lo, hi)
    top_axes[-1].legend(fontsize=6, loc="best")

    axb = fig.add_subplot(gs[1, :])
    for arm in ALL_ARMS:
        ys = [(_acc(arm, layer)) for layer in layers]
        ys = [np.nan if v is None else float(v) for v in ys]
        if np.isfinite(np.asarray(ys)).any():
            axb.plot(layers, ys, marker="s", ms=5, lw=1.6, color=arm_c[arm], label=ARM_LABELS[arm])
            n += int(np.isfinite(np.asarray(ys)).sum())
    _require_points(n, "twin-layer table figure")
    axb.set_xticks(layers, [f"L{x}" for x in layers])
    axb.set_ylabel("Pair-delta retrieval acc@1 (all-pairs pool)")
    axb.set_xlabel("Capture layer")
    axb.set_ylim(0, 1.05)
    axb.legend(fontsize=7)
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--in-dir",
        default=None,
        help="reads output dir holding summary.json + perpair.jsonl + percontext.jsonl "
        "(default: eval_results/issue_2617/svmp)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="figure output dir (default: figures/issue_2617; use a scratch dir for smokes)",
    )
    ap.add_argument(
        "--anchor-json",
        default=None,
        help="#2564 gramslot summary.json carrying the benign-anchor cos medians "
        "(default: eval_results/issue_2564/gramslot_pilot/summary.json)",
    )
    ap.add_argument("--import-check", action="store_true")
    return ap


def _import_check() -> None:
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    for fn in (paper_palette, paper_palette_role, savefig_paper, set_paper_style):
        assert callable(fn), fn
    set_paper_style()  # blog default — role colors resolve per ACTIVE style
    arm_c = _arm_colors()
    cls_c = _class_colors()
    overlap = set(arm_c.values()) & set(cls_c.values())
    assert not overlap, f"arm/class palette hex collision breaks one-color-one-meaning: {overlap}"
    assert set(ARM_LABELS) == set(ALL_ARMS) and set(CLASS_LABELS) == set(PAIR_CLASSES)
    print("[import-check] ok: argcheck + paper_plots surface + disjoint palettes", flush=True)


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        _import_check()
        return 0
    set_paper_style()
    in_dir = Path(args.in_dir) if args.in_dir else IN_DIR_DEFAULT
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR_DEFAULT
    anchor_json = Path(args.anchor_json) if args.anchor_json else ANCHOR_JSON_DEFAULT
    summary, rows, ctx_rows = load_reads(in_dir)
    anchors = load_anchors(anchor_json)
    out_dir.mkdir(parents=True, exist_ok=True)
    halts = summary.get("halts", {})
    halted = bool(halts.get("dichotomy_halted") or halts.get("judge_integrity_halt"))
    dry_judge = bool(summary.get("judge", {}).get("dry_run"))
    # Figure gating (concern dichotomy-halt-not-enforced, r2): under a plan-§7
    # halt on a LIVE judge, the flip-dichotomy HEADLINE figures (hero,
    # axis-loading) do not ship — the judge-derived diagnostics (manipulation
    # check, margin validation) still render for halt inspection. A DRY-RUN
    # judge (tiny smoke input) has no rate data by construction, so ALL
    # judge-derived figures are skipped there instead of crashing on
    # zero-point panels.
    builders: dict[str, tuple] = {
        "svmp_hero": (lambda: fig_hero(summary, rows, anchors), "dichotomy"),
        "svmp_retrieval_acc1": (lambda: fig_retrieval(summary), None),
        "svmp_calibration_slopes": (lambda: fig_calibration(summary), None),
        "svmp_axis_loading": (lambda: fig_axis_loading(rows), "dichotomy"),
        "svmp_margin_validation": (lambda: fig_margin_validation(ctx_rows), "judge"),
        "svmp_manipulation_check": (lambda: fig_manipulation(rows, ctx_rows), "judge"),
        "svmp_len_vs_cos": (lambda: fig_len_vs_cos(rows), None),
        "svmp_span_vs_tail": (lambda: fig_span_vs_tail(rows), None),
        "svmp_twin_layers": (lambda: fig_twin_layers(summary), None),
    }
    n_rendered = 0
    for stem, (build, gate) in builders.items():
        if gate is not None and dry_judge:
            print(f"[fig] SKIP {stem} — dry-run-judge input carries no rate data", flush=True)
            continue
        if gate == "dichotomy" and halted:
            print(f"[fig] SKIP {stem} — plan-§7 halt: {halts.get('reasons')}", flush=True)
            continue
        fig = build()
        paths = savefig_paper(fig, stem, dir=out_dir)
        plt.close(fig)
        print(f"[fig] {paths['png']}", flush=True)
        n_rendered += 1
    print(f"[out] {n_rendered}/{len(builders)} figures -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
